from __future__ import annotations

import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).parents[3]
ACTOR = ROOT / "verl" / "workers" / "actor" / "megatron_actor.py"
OBSERVER = ROOT / "verl" / "utils" / "debug" / "eu_derpo.py"
CONFIG = ROOT / "verl" / "workers" / "config" / "actor.py"
ACTOR_YAML = ROOT / "verl" / "trainer" / "config" / "actor" / "actor.yaml"
OPTIMIZER = ROOT / "verl" / "utils" / "megatron" / "optimizer.py"


def function_source(path: Path, name: str) -> str:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return ast.get_source_segment(source, node)
    raise AssertionError(f"missing function {name}")


class TestEUDERPOV121ActorContract(unittest.TestCase):
    def test_actor_requires_native_hdo_partial_cpu_offload(self):
        source = ACTOR.read_text(encoding="utf-8")
        self.assertIn('optimizer_override.get("optimizer_cpu_offload") is not True', source)
        self.assertIn('optimizer_override.get("optimizer_offload_fraction") != 0.75', source)
        self.assertIn("optimizer_cpu_offload_enabled=1 optimizer_offload_fraction=0.75", source)

    def test_actor_requires_original_phase_offload_contract(self):
        source = ACTOR.read_text(encoding="utf-8")
        phase_guard = source[source.index('"phase_offload"'):source.index('"native_hdo_cpu_offload"')]
        self.assertIn("param_offload", phase_guard)
        self.assertIn("grad_offload", phase_guard)
        self.assertIn("optimizer_offload", phase_guard)

    def test_post_checkpoint_optimizer_skip_defaults_off(self):
        self.assertIn(
            "skip_post_checkpoint_optimizer_offload: bool = False",
            CONFIG.read_text(encoding="utf-8"),
        )
        self.assertIn(
            "skip_post_checkpoint_optimizer_offload: false",
            ACTOR_YAML.read_text(encoding="utf-8"),
        )
        self.assertIn(
            "preserve_hdo_optimizer_residency_between_steps: bool = False",
            CONFIG.read_text(encoding="utf-8"),
        )
        self.assertIn(
            "preserve_hdo_optimizer_residency_between_steps: false",
            ACTOR_YAML.read_text(encoding="utf-8"),
        )
        self.assertIn(
            "release_actor_cuda_cache_before_rollout_wakeup: bool = False",
            CONFIG.read_text(encoding="utf-8"),
        )
        self.assertIn(
            "release_actor_cuda_cache_before_rollout_wakeup: false",
            ACTOR_YAML.read_text(encoding="utf-8"),
        )
        self.assertIn(
            "offload_optimizer_copy_params_for_rollout: bool = False",
            CONFIG.read_text(encoding="utf-8"),
        )
        self.assertIn(
            "offload_optimizer_copy_params_for_rollout: false",
            ACTOR_YAML.read_text(encoding="utf-8"),
        )

    def test_optimizer_override_is_passed_to_mcore_optimizer_config(self):
        source = function_source(OPTIMIZER, "init_megatron_optim_config")
        self.assertIn("for k, v in override_config.items()", source)
        self.assertIn("optim_args[k] = v", source)
        self.assertLess(source.index("optim_args[k] = v"), source.index("OptimizerConfig(**optim_args)"))

    def test_update_policy_has_no_full_auxiliary_forward(self):
        source = function_source(ACTOR, "update_policy")
        self.assertIn("run_router_only_step_e", source)
        self.assertNotIn("start_aux_batch", source)
        self.assertNotIn("finish_aux_batch", source)
        self.assertNotIn("auxiliary_output = self.forward_backward_batch", source)

    def test_router_only_runner_calls_gating_not_router_or_routing(self):
        source = function_source(OBSERVER, "run_router_only_step_e")
        self.assertIn("_verl_eu_derpo_original_gating", source)
        self.assertNotIn("router.forward(", source)
        self.assertNotIn("router.routing(", source)
        self.assertNotIn("topk(", source)
        self.assertNotIn("dispatcher", source)
        self.assertIn("retain_graph=False", source)
        self.assertIn("create_graph=False", source)

    def test_finalize_and_optimizer_order_is_explicit(self):
        source = function_source(ACTOR, "update_policy")
        wrapped = source.index("wrap_native_finalize")
        main = source.index("metric_micro_batch = self.forward_backward_batch")
        step_e = source.index("run_router_only_step_e")
        validate = source.index("validate_before_optimizer_step")
        optimizer = source.index("self.actor_optimizer.step()")
        self.assertLess(wrapped, main)
        self.assertLess(main, step_e)
        self.assertLess(step_e, validate)
        self.assertLess(validate, optimizer)
        self.assertEqual(source.count("self.actor_optimizer.step()"), 1)

    def test_config_is_frozen_to_v121_actual_f_router_only(self):
        source = CONFIG.read_text(encoding="utf-8")
        self.assertIn('version: str = "1.2.1"', source)
        self.assertIn('step_e_implementation: str = "actual_f_router_only"', source)
        self.assertIn('cache_backend: str = "cpu_pageable"', source)
        self.assertIn("staging_mib: int = 64", source)
        self.assertIn("natural_topk_step_e: bool = False", source)
        self.assertIn("full_aux_forward: bool = False", source)


if __name__ == "__main__":
    unittest.main()
