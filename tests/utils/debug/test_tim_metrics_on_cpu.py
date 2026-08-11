import importlib.util
import math
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch


class _DataProto:
    def __init__(self, batch):
        self.batch = batch


class TopKRouter(torch.nn.Module):
    """MCore 0.16-shaped router: gating is a bound method, not a child module."""

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.arange(12, dtype=torch.float32).reshape(4, 3) / 12)
        self.topk = 2

    def gating(self, inputs):
        output = torch.nn.functional.linear(inputs, self.weight)
        self.last_gating_output = output
        return output

    def forward(self, inputs):
        logits = self.gating(inputs)
        indices = torch.topk(logits, self.topk, dim=-1).indices
        routing_map = torch.zeros_like(logits, dtype=torch.bool).scatter(-1, indices, True)
        return torch.softmax(logits, dim=-1), routing_map


def _router_config(**overrides):
    values = {
        "virtual_pipeline_model_parallel_size": None,
        "tensor_model_parallel_size": 1,
        "sequence_parallel": False,
        "moe_router_score_function": "softmax",
        "moe_router_pre_softmax": True,
    }
    values.update(overrides)
    return types.SimpleNamespace(**values)


module_path = Path(__file__).parents[3] / "verl" / "utils" / "debug" / "metrics.py"


def _load_metrics_module():
    protocol = types.ModuleType("verl.protocol")
    protocol.DataProto = _DataProto
    modules = {"verl": types.ModuleType("verl"), "verl.protocol": protocol}
    spec = importlib.util.spec_from_file_location("tim_metrics_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, modules):
        spec.loader.exec_module(module)
    return module


metrics_module = _load_metrics_module()
router_module_path = module_path.with_name("router_shift.py")
router_spec = importlib.util.spec_from_file_location("router_shift_under_test", router_module_path)
router_module = importlib.util.module_from_spec(router_spec)
router_spec.loader.exec_module(router_module)


class TestTimMetrics(unittest.TestCase):
    def test_module_loading_does_not_pollute_sys_modules(self):
        before = {name: sys.modules.get(name) for name in ("verl", "verl.protocol")}

        _load_metrics_module()

        for name, previous in before.items():
            if previous is None:
                self.assertNotIn(name, sys.modules)
            else:
                self.assertIs(sys.modules[name], previous)

    def test_uses_response_mask_and_detaches(self):
        rollout_log_probs = torch.full((1, 3), -3.0, requires_grad=True)
        actor_log_probs = torch.tensor(
            [[-3.0 + math.log(4.0), -3.0 - math.log(4.0), -1.0]], requires_grad=True
        )
        data = _DataProto(
            {
                "rollout_log_probs": rollout_log_probs,
                "old_log_probs": actor_log_probs,
                "response_mask": torch.tensor([[1, 1, 0]]),
                "responses": torch.zeros((1, 3)),
            }
        )

        metrics = metrics_module.calculate_debug_metrics(data)

        self.assertAlmostEqual(metrics["diag/tim/kl_k3"], 1.125, places=6)
        self.assertAlmostEqual(metrics["diag/tim/logprob_abs_mean"], math.log(4.0), places=6)
        self.assertEqual(metrics["diag/tim/extreme_frac_tau2"], 1.0)
        self.assertIsNone(rollout_log_probs.grad)
        self.assertIsNone(actor_log_probs.grad)

    def test_bfloat16_inputs_use_finite_fp32_diagnostics(self):
        rollout = torch.tensor([[-3.0, -3.0, -3.0]], dtype=torch.bfloat16, requires_grad=True)
        actor = torch.tensor(
            [[-3.0 + math.log(4.0), -3.0 - math.log(4.0), -1.0]],
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        data = _DataProto(
            {
                "rollout_log_probs": rollout,
                "old_log_probs": actor,
                "response_mask": torch.tensor([[1, 1, 0]]),
                "responses": torch.zeros((1, 3)),
            }
        )

        metrics = metrics_module.calculate_debug_metrics(data)

        for name in ("diag/tim/kl_k3", "diag/tim/logprob_abs_mean", "diag/tim/extreme_frac_tau2"):
            self.assertTrue(math.isfinite(metrics[name]))
        self.assertAlmostEqual(metrics["diag/tim/kl_k3"], 1.125, delta=0.02)
        self.assertAlmostEqual(metrics["diag/tim/logprob_abs_mean"], math.log(4.0), delta=0.01)
        self.assertIsNone(rollout.grad)
        self.assertIsNone(actor.grad)

    def test_loss_mask_fallback(self):
        data = _DataProto(
            {
                "rollout_log_probs": torch.tensor([[-3.0, -3.0]]),
                "old_log_probs": torch.tensor([[-3.0 + math.log(4.0), -1.0]]),
                "loss_mask": torch.tensor([[1, 0]]),
                "responses": torch.zeros((1, 2)),
            }
        )

        metrics = metrics_module.calculate_debug_metrics(data)

        self.assertAlmostEqual(metrics["diag/tim/logprob_abs_mean"], math.log(4.0), places=6)

    def test_attention_mask_is_sliced_to_response(self):
        data = _DataProto(
            {
                "rollout_log_probs": torch.tensor([[-3.0, -3.0]]),
                "old_log_probs": torch.tensor([[-3.0 + math.log(4.0), -1.0]]),
                "attention_mask": torch.tensor([[1, 1, 1, 0]]),
                "responses": torch.zeros((1, 2)),
            }
        )

        metrics = metrics_module.calculate_debug_metrics(data)

        self.assertAlmostEqual(metrics["diag/tim/logprob_abs_mean"], math.log(4.0), places=6)

    def test_no_mask_uses_legacy_all_token_fallback(self):
        data = _DataProto(
            {
                "rollout_log_probs": torch.tensor([[-3.0, -3.0]]),
                "old_log_probs": torch.tensor([[-3.0 + math.log(4.0), -3.0]]),
                "responses": torch.zeros((1, 2)),
            }
        )

        metrics = metrics_module.calculate_debug_metrics(data)

        self.assertAlmostEqual(metrics["diag/tim/logprob_abs_mean"], math.log(4.0) / 2, places=6)


class TestRouterShiftContract(unittest.TestCase):
    def test_zero_delta_and_masked_padding(self):
        values = torch.zeros((1, 2, 3, 4), dtype=torch.bfloat16)
        metrics = metrics_module.calculate_router_shift_metrics(values, values, torch.tensor([[1, 0]]))

        self.assertEqual(metrics["diag/router_shift/ratio_mean"], 1.0)
        self.assertEqual(metrics["diag/router_shift/clipfrac_gamma_0_8"], 0.0)

    def test_pure_math_uses_old_experts_and_response_mask(self):
        old = torch.zeros((1, 2, 2, 2), requires_grad=True)
        current = torch.tensor(
            [[[[0.0, 0.0], [math.log(4.0), math.log(4.0)]], [[10.0, 10.0], [10.0, 10.0]]]],
            requires_grad=True,
        )

        metrics = metrics_module.calculate_router_shift_metrics(old, current, torch.tensor([[1, 0]]))

        self.assertAlmostEqual(metrics["diag/router_shift/ratio_mean"], 0.5, places=6)
        self.assertEqual(metrics["diag/router_shift/clipfrac_gamma_0_8"], 1.0)
        self.assertIsNone(old.grad)
        self.assertIsNone(current.grad)

    def test_current_full_logits_cover_old_expert_outside_current_topk(self):
        old_logits = torch.tensor([[5.0, 4.0, 0.0, -1.0]])
        old_map = torch.tensor([[1, 1, 0, 0]], dtype=torch.bool)
        old_indices, old_selected = router_module.RouterShiftObserver.selected_old_router_log_probs(
            old_logits, old_map, topk=2
        )
        current_logits = torch.tensor([[0.0, -1.0, 5.0, 4.0]], dtype=torch.bfloat16)

        diff = router_module.RouterShiftObserver.current_abs_diff_sum(
            current_logits, old_indices, old_selected
        )
        reference = (
            torch.log_softmax(current_logits.float(), dim=-1).gather(-1, old_indices) - old_selected
        ).abs().sum(-1)

        self.assertTrue(torch.isfinite(diff).all())
        torch.testing.assert_close(diff, reference)
        self.assertEqual(set(old_indices[0].tolist()), {0, 1})

    def test_pp3_aggregates_delta_before_exponentiation(self):
        local = [torch.tensor([0.0]), torch.tensor([math.log(2.0)]), torch.tensor([math.log(4.0)])]

        gamma = router_module.RouterShiftObserver.aggregate_partials(local, [1, 1, 1], topk=1)

        expected = torch.exp(-sum(local) / 3)
        wrong = sum(torch.exp(-value) for value in local) / 3
        torch.testing.assert_close(gamma, expected)
        self.assertFalse(torch.allclose(gamma, wrong))

    def test_sample_ids_restore_records_after_minibatch_reordering(self):
        observer = object.__new__(router_module.RouterShiftObserver)
        observer.old_cache = {
            3: (torch.tensor([[[3]]], dtype=torch.uint8), torch.tensor([[[0.3]]])),
            7: (torch.tensor([[[7]]], dtype=torch.uint8), torch.tensor([[[0.7]]])),
        }
        observer.routers = [object()]
        observer._pack_sequence_parallel = lambda records, *_args: records.squeeze(1)

        observer.begin_current_microbatch(
            torch.zeros((2, 1)),
            torch.ones((2, 1), dtype=torch.bool),
            torch.tensor([7, 3]),
            response_length=1,
        )

        self.assertTrue(torch.equal(observer._current_indices[0, :, 0], torch.tensor([7, 3])))
        torch.testing.assert_close(observer._current_old_log_probs[0, :, 0], torch.tensor([0.7, 0.3]))

    def test_observer_hooks_do_not_change_router_output_or_gradients(self):
        baseline = TopKRouter()
        observed = TopKRouter()
        observed.load_state_dict(baseline.state_dict())
        parameter_count = sum(parameter.numel() for parameter in observed.parameters())
        observer = router_module.RouterShiftObserver([observed], _router_config())
        observer.start_old_batch()
        observer.begin_old_microbatch()
        baseline_input = torch.tensor([[1.0, 2.0, 3.0]], requires_grad=True)
        observed_input = baseline_input.detach().clone().requires_grad_(True)

        baseline_output = baseline(baseline_input)[0]
        observed_output = observed(observed_input)[0]
        baseline_output[:, 0].sum().backward()
        observed_output[:, 0].sum().backward()

        torch.testing.assert_close(observed_output, baseline_output)
        torch.testing.assert_close(observed_input.grad, baseline_input.grad)
        torch.testing.assert_close(observed.weight.grad, baseline.weight.grad)
        self.assertEqual(sum(parameter.numel() for parameter in observed.parameters()), parameter_count)
        self.assertFalse(observer._old_log_probs[0].requires_grad)

    def test_old_cache_preserves_fp32_zero_and_tiny_router_shift(self):
        router = TopKRouter()
        observer = router_module.RouterShiftObserver([router], _router_config())
        observer.start_old_batch()
        observer.begin_old_microbatch()

        router(torch.tensor([[1.0, 2.0, 3.0]]))
        old_logits = router.last_gating_output.detach().clone()
        old_indices = observer._old_indices[0]
        cached_old_log_probs = observer._old_log_probs[0]

        self.assertEqual(old_indices.dtype, torch.uint8)
        self.assertEqual(cached_old_log_probs.dtype, torch.float32)
        zero_diff = observer.current_abs_diff_sum(old_logits, old_indices, cached_old_log_probs)
        torch.testing.assert_close(zero_diff, torch.zeros_like(zero_diff), atol=0, rtol=0)
        torch.testing.assert_close(torch.exp(-zero_diff / router.topk), torch.ones_like(zero_diff))

        current_logits = old_logits.clone()
        current_logits[0, old_indices[0, 0].item()] += 1e-3
        measured = observer.current_abs_diff_sum(current_logits, old_indices, cached_old_log_probs)
        fp32_reference = observer.current_abs_diff_sum(
            current_logits,
            old_indices,
            torch.log_softmax(old_logits.float(), dim=-1).gather(-1, old_indices.long()),
        )
        self.assertGreater(measured.item(), 0)
        torch.testing.assert_close(measured, fp32_reference, atol=1e-7, rtol=1e-5)

    def test_method_gating_wrap_is_instance_scoped_and_not_nested(self):
        observed = TopKRouter()
        untouched = TopKRouter()
        original = observed.gating
        untouched_original = untouched.gating

        observer = router_module.RouterShiftObserver([observed], _router_config())
        wrapped = observed.__dict__["gating"]

        self.assertEqual(observed._verl_router_shift_original_gating, original)
        self.assertNotIn("gating", untouched.__dict__)
        self.assertEqual(untouched.gating, untouched_original)
        returned_logits = observed.gating(torch.ones((1, 3)))
        self.assertIs(returned_logits, observed.last_gating_output)
        with self.assertRaisesRegex(RuntimeError, "already attached"):
            router_module.RouterShiftObserver([observed], _router_config())
        self.assertIs(observed.__dict__["gating"], wrapped)

        observer.close()
        self.assertEqual(observed.gating, original)
        self.assertFalse(hasattr(observed, "_verl_router_shift_original_gating"))

    def test_router_semantics_compatibility_gate(self):
        with self.assertRaisesRegex(ValueError, "score_function='softmax'"):
            router_module.RouterShiftObserver(
                [TopKRouter()], _router_config(moe_router_score_function="sigmoid")
            )
        with self.assertRaisesRegex(ValueError, "pre_softmax=True"):
            router_module.RouterShiftObserver(
                [TopKRouter()], _router_config(moe_router_pre_softmax=False)
            )

    def test_real_mcore_016_gating_api_when_available(self):
        try:
            from megatron.core.transformer.moe.router import TopKRouter as MCoreTopKRouter
        except ImportError:
            self.skipTest("Megatron-Core is not installed")

        self.assertTrue(callable(MCoreTopKRouter.gating))
        self.assertFalse(isinstance(MCoreTopKRouter.gating, torch.nn.Module))

    def test_diagnostics_default_disabled_and_collective_is_post_schedule(self):
        root = module_path.parents[3]
        config_source = (root / "verl" / "workers" / "config" / "actor.py").read_text()
        actor_source = (root / "verl" / "workers" / "actor" / "megatron_actor.py").read_text()
        trainer_source = (root / "verl" / "trainer" / "ppo" / "ray_trainer.py").read_text()

        self.assertIn("enabled: bool = False", config_source)
        self.assertIn("if self.config.router_shift_diagnostics.enabled:", actor_source)
        self.assertIn('torch.arange(\n                                    len(batch)', trainer_source)
        schedule_end = actor_source.index("# loss_reduces contains the stats returned from loss_func")
        aggregate = actor_source.index("self.router_shift_observer.finish_current_batch()")
        self.assertLess(schedule_end, aggregate)
        observer_source = router_module_path.read_text()
        self.assertIn("-response_length - 1 : -1", observer_source)
        self.assertIn("get_pipeline_model_parallel_group()", observer_source)
        self.assertIn("get_data_parallel_group()", observer_source)

    def test_current_dataflow_records_indices_but_not_router_scores(self):
        root = module_path.parents[3]
        replay_source = (root / "verl" / "utils" / "megatron" / "router_replay_patch.py").read_text()
        worker_source = (root / "verl" / "workers" / "megatron_workers.py").read_text()

        self.assertIn("def record_indices", replay_source)
        self.assertNotIn("def record_log_probs", replay_source)
        self.assertIn('self.enable_routing_replay = self.router_replay.mode != "disabled"', worker_source)
        self.assertIn('tensors = {"ref_log_prob": output} if is_lora else {"old_log_probs": output}', worker_source)
        self.assertNotIn('output.batch["old_router_log_probs"]', worker_source)


if __name__ == "__main__":
    unittest.main()
