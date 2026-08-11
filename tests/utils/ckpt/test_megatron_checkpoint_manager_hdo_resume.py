# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import ast
import importlib.util
import unittest
from pathlib import Path
from types import MethodType, SimpleNamespace


MANAGER = Path(__file__).resolve().parents[3] / "verl" / "utils" / "checkpoint" / "megatron_checkpoint_manager.py"


class HybridDeviceOptimizer:
    def __init__(self, initialized=True, parameter=None, rng=None):
        self.state = {object(): {"master_param": object()}} if initialized else {}
        self.parameter = parameter
        self.rng = rng
        self.dummy_steps = 0

    def dummy_step(self):
        self.dummy_steps += 1
        self.state = {object(): {"master_param": object()}}
        if self.parameter is not None:
            self.parameter.value = "dummy-update"
        if self.rng is not None:
            self.rng.value = "dummy-consumed"


class DistributedOptimizer:
    def __init__(self, optimizer):
        self.optimizer = optimizer


class ChainedOptimizer:
    def __init__(self, *optimizers):
        self.chained_optimizers = list(optimizers)
        self.loading_flags = []
        self.loaded_states = []

    def sharded_state_dict(self, state_dict, *, is_loading, metadata):
        self.loading_flags.append(is_loading)
        hdos = [
            optimizer.optimizer
            for optimizer in self.chained_optimizers
            if isinstance(optimizer, DistributedOptimizer)
            and isinstance(optimizer.optimizer, HybridDeviceOptimizer)
        ]
        if is_loading and metadata.get("distrib_optim_sharding_type") == "dp_reshardable" and hdos:
            for hdo in hdos:
                if not hdo.state:
                    hdo.dummy_step()
            raise KeyError("param_to_fp32_param identity mismatch")
        return {"template": True}

    def load_state_dict(self, state_dict):
        self.loaded_states.append(state_dict)
        for optimizer in self.chained_optimizers:
            if isinstance(optimizer, DistributedOptimizer) and isinstance(
                optimizer.optimizer, HybridDeviceOptimizer
            ):
                optimizer.optimizer.state = state_dict


class OrdinaryOptimizer:
    def __init__(self):
        self.loading_flags = []

    def sharded_state_dict(self, state_dict, *, is_loading, metadata):
        self.loading_flags.append(is_loading)
        return {"template": True}


class Model:
    def __init__(self, parameter=None):
        self.parameter = parameter

    def sharded_state_dict(self, **kwargs):
        return {"weight": "model-template", "metadata": kwargs["metadata"]}

    def load_state_dict(self, state_dict):
        if self.parameter is not None:
            self.parameter.value = state_dict["weight"]


def _manager(optimizer, model=None, *, mcore_016=True):
    manager = SimpleNamespace(
        model=[model or Model()],
        optimizer=optimizer,
        lr_scheduler=None,
        get_rng_state=lambda: {"rng": "extra-template"},
        _build_sharded_state_dict_metadata=lambda: {
            "distrib_optim_sharding_type": "dp_reshardable",
            "chained_optim_avoid_prefix": True,
        },
    )
    manager.generate_state_dict = MethodType(_generate_state_dict(mcore_016=mcore_016), manager)
    return manager


def _production_methods(*, mcore_016=True):
    tree = ast.parse(MANAGER.read_text(encoding="utf-8"), filename=str(MANAGER))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "MegatronCheckpointManager")
    methods = [
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name in {"generate_state_dict", "load_checkpoint"}
    ]
    helper = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name
            in {
                "_is_mcore_016_hdo_dp_reshardable_resume",
                "_prepare_mcore_016_hdo_dp_reshardable_resume",
            }
        ),
        None,
    )
    if helper is not None:
        helper.body = [node for node in helper.body if not isinstance(node, ast.ImportFrom)]
    namespace = {
        "torch": SimpleNamespace(distributed=SimpleNamespace(barrier=lambda: None)),
        "mpu": SimpleNamespace(
            set_virtual_pipeline_model_parallel_rank=lambda rank: None,
            get_data_parallel_group=lambda **kwargs: "dp-cp-group",
        ),
        "mcore_ge_014": True,
        "mcore_016": mcore_016,
        "ChainedOptimizer": ChainedOptimizer,
        "DistributedOptimizer": DistributedOptimizer,
        "HybridDeviceOptimizer": HybridDeviceOptimizer,
        "os": SimpleNamespace(
            path=SimpleNamespace(exists=lambda path: True, join=lambda *parts: "/".join(parts)),
            remove=lambda path: None,
        ),
        "dist_checkpointing": SimpleNamespace(
            load_content_metadata=lambda checkpoint_dir: {"distrib_optim_sharding_type": "dp_reshardable"}
        ),
        "get_dist_checkpoint_path": lambda path: f"{path}/dist_ckpt",
        "log_with_rank": lambda *args, **kwargs: None,
        "logger": None,
    }
    nodes = [
        ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
        *(node for node in (helper, *methods) if node is not None),
    ]
    module = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    exec(compile(module, str(MANAGER), "exec"), namespace)
    return namespace


def _generate_state_dict(*, mcore_016=True):
    return _production_methods(mcore_016=mcore_016)["generate_state_dict"]


class TestMegatronCheckpointManagerHDOResume(unittest.TestCase):
    def test_hdo_resume_template_avoids_parameter_identity_reload(self):
        optimizer = ChainedOptimizer(DistributedOptimizer(HybridDeviceOptimizer(initialized=True)))

        state_dict = _manager(optimizer).generate_state_dict(is_loading=True)

        self.assertEqual(optimizer.loading_flags, [False])
        self.assertEqual(state_dict["optimizer"], {"template": True})

    def test_empty_hdo_is_prepared_before_template_generation(self):
        hdo = HybridDeviceOptimizer(initialized=False)
        optimizer = ChainedOptimizer(DistributedOptimizer(hdo))

        state_dict = _manager(optimizer).generate_state_dict(is_loading=True)

        self.assertEqual(hdo.dummy_steps, 1)
        self.assertTrue(hdo.state)
        self.assertEqual(optimizer.loading_flags, [False])
        self.assertEqual(state_dict["optimizer"], {"template": True})

    def test_scope_excludes_save_ordinary_and_other_format(self):
        save_optimizer = ChainedOptimizer(DistributedOptimizer(HybridDeviceOptimizer(initialized=True)))
        _manager(save_optimizer).generate_state_dict(is_loading=False)
        self.assertEqual(save_optimizer.loading_flags, [False])

        ordinary_optimizer = OrdinaryOptimizer()
        _manager(ordinary_optimizer).generate_state_dict(is_loading=True)
        self.assertEqual(ordinary_optimizer.loading_flags, [True])

        non_hdo = ChainedOptimizer(DistributedOptimizer(object()))
        _manager(non_hdo).generate_state_dict(is_loading=True)
        self.assertEqual(non_hdo.loading_flags, [True])

        other_mcore = ChainedOptimizer(DistributedOptimizer(HybridDeviceOptimizer(initialized=True)))
        with self.assertRaisesRegex(KeyError, "param_to_fp32_param identity mismatch"):
            _manager(other_mcore, mcore_016=False).generate_state_dict(is_loading=True)
        self.assertEqual(other_mcore.loading_flags, [True])

        other_format = ChainedOptimizer(DistributedOptimizer(HybridDeviceOptimizer(initialized=True)))
        manager = _manager(other_format)
        manager.generate_state_dict(
            is_loading=True,
            metadata={"distrib_optim_sharding_type": "fully_reshardable"},
        )
        self.assertEqual(other_format.loading_flags, [True])

    def test_partial_loads_do_not_trigger_full_resume_preparation(self):
        model_only = ChainedOptimizer(DistributedOptimizer(HybridDeviceOptimizer(initialized=False)))
        _manager(model_only).generate_state_dict(generate_optimizer=False, is_loading=True)
        self.assertEqual(model_only.loading_flags, [])

        optimizer_only = ChainedOptimizer(DistributedOptimizer(HybridDeviceOptimizer(initialized=True)))
        with self.assertRaisesRegex(KeyError, "param_to_fp32_param identity mismatch"):
            _manager(optimizer_only).generate_state_dict(
                generate_model=False,
                generate_optimizer=True,
                generate_extra=False,
                is_loading=True,
            )
        self.assertEqual(optimizer_only.loading_flags, [True])

    def test_model_optimizer_and_extra_templates_remain_present(self):
        optimizer = ChainedOptimizer(DistributedOptimizer(HybridDeviceOptimizer(initialized=True)))

        state_dict = _manager(optimizer).generate_state_dict(is_loading=True)

        self.assertEqual(state_dict["model"]["weight"], "model-template")
        self.assertIn("optimizer", state_dict)
        self.assertEqual(state_dict["rng_state"], {"rng": "extra-template"})

    def test_final_optimizer_restore_is_still_in_load_checkpoint(self):
        parameter = SimpleNamespace(value="initial-model")
        rng = SimpleNamespace(value="initial-rng")
        scheduler = SimpleNamespace(step_calls=0, state_dict=lambda: {"scheduler": "initial"})
        hdo = HybridDeviceOptimizer(initialized=False, parameter=parameter, rng=rng)
        optimizer = ChainedOptimizer(DistributedOptimizer(hdo))
        manager = _manager(optimizer, Model(parameter))
        methods = _production_methods()
        checkpoint_optimizer_state = {"checkpoint": "optimizer-state"}
        methods["load_checkpoint"].__globals__["load_dist_checkpointing"] = lambda **kwargs: {
            "model": {"weight": "checkpoint-model"},
            "optimizer": checkpoint_optimizer_state,
            "rng_state": {"value": "checkpoint-rng"},
        }
        manager.generate_state_dict = MethodType(methods["generate_state_dict"], manager)
        manager.load_checkpoint = MethodType(methods["load_checkpoint"], manager)
        manager.should_load_model = True
        manager.should_load_optimizer = True
        manager.should_load_extra = True
        manager.use_dist_checkpointing = True
        manager.use_distributed_optimizer = True
        manager.use_hf_checkpoint = False
        manager.use_checkpoint_opt_param_scheduler = False
        manager.lr_scheduler = scheduler
        manager.rank = 0
        manager.peft_cls = None
        manager.global_step = 1
        manager.load_rng_states = lambda state: setattr(rng, "value", state["value"])

        manager.load_checkpoint("checkpoint", del_local_after_load=False)

        self.assertEqual(hdo.dummy_steps, 1)
        self.assertEqual(parameter.value, "checkpoint-model")
        self.assertEqual(optimizer.loaded_states, [checkpoint_optimizer_state])
        self.assertIs(hdo.state, checkpoint_optimizer_state)
        self.assertEqual(rng.value, "checkpoint-rng")
        self.assertEqual(manager.global_step, 1)
        self.assertEqual(scheduler.step_calls, 0)

    @unittest.skipUnless(importlib.util.find_spec("megatron") is not None, "Megatron-Core is not installed")
    def test_real_mcore_class_identity(self):
        from megatron.core.optimizer import ChainedOptimizer as RealChainedOptimizer
        from megatron.core.optimizer.cpu_offloading import HybridDeviceOptimizer as RealHybridDeviceOptimizer
        from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer as RealDistributedOptimizer

        hdo = object.__new__(RealHybridDeviceOptimizer)
        hdo.state = {object(): {}}
        distributed = object.__new__(RealDistributedOptimizer)
        distributed.optimizer = hdo
        chained = object.__new__(RealChainedOptimizer)
        chained.chained_optimizers = [distributed]
        from verl.utils.checkpoint.megatron_checkpoint_manager import (
            _prepare_mcore_016_hdo_dp_reshardable_resume as helper,
        )

        self.assertTrue(
            helper(
                chained,
                is_loading=True,
                metadata={"distrib_optim_sharding_type": "dp_reshardable"},
                full_resume=True,
            )
        )


if __name__ == "__main__":
    unittest.main()
