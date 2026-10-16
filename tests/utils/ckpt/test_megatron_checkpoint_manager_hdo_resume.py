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
import copy
import gc
import importlib.util
import unittest
from pathlib import Path
from types import MethodType, SimpleNamespace

import torch


MANAGER = Path(__file__).resolve().parents[3] / "verl" / "utils" / "checkpoint" / "megatron_checkpoint_manager.py"


class TensorValue:
    def __init__(self, value):
        self.value = value
        self.data = self

    def copy_(self, other):
        self.value = other.value


class Parameter(TensorValue):
    def __init__(self):
        super().__init__("initial-param")


class HybridDeviceOptimizer:
    def __init__(self, initialized=True, parameter=None, rng=None, realistic_restore=False):
        self.state = {object(): {"master_param": object()}} if initialized else {}
        self.parameter = parameter
        self.rng = rng
        self.dummy_steps = 0
        self.param_update_in_fp32 = realistic_restore
        if realistic_restore:
            self.native_fp32_param = Parameter()
            self.low_precision_param = Parameter()
            self.old_fp32_param = TensorValue("old-master")
            self.param_groups = [
                {"params": [self.native_fp32_param, self.low_precision_param], "lr": "initial-lr"}
            ]
            self.param_to_fp32_param = {self.low_precision_param: self.old_fp32_param}
            self.fp32_param_to_orig_param = {self.old_fp32_param: self.low_precision_param}

    def dummy_step(self):
        self.dummy_steps += 1
        self.state = {object(): {"master_param": object()}}
        if self.parameter is not None:
            self.parameter.value = "dummy-update"
        if self.rng is not None:
            self.rng.value = "dummy-consumed"

    def load_state_dict(self, state_dict):
        # MCore 0.16 HDO pre-hook: temporarily expose master params to torch.
        old_reverse_mapping = self.fp32_param_to_orig_param
        current_params = [
            self.param_to_fp32_param.get(param, param) for param in self.param_groups[0]["params"]
        ]

        # torch.optim.Optimizer.load_state_dict: saved integer ids map by group order.
        saved_params = state_dict["param_groups"][0]["params"]
        id_map = dict(zip(saved_params, current_params))
        self.state = {id_map[param_id]: value for param_id, value in state_dict["state"].items()}
        self.param_groups = [{**state_dict["param_groups"][0], "params": current_params}]

        # MCore 0.16 HDO post-hook: return old master keys to original params.
        self.state = {old_reverse_mapping.get(param, param): value for param, value in self.state.items()}
        self.param_groups[0]["params"] = [self.native_fp32_param, self.low_precision_param]

        # _init_sub_optimizers() creates a fresh master object for low-precision params;
        # native FP32 params intentionally have no param_to_fp32_param entry.
        new_fp32_param = TensorValue("new-master")
        self.param_to_fp32_param = {self.low_precision_param: new_fp32_param}
        self.fp32_param_to_orig_param = {new_fp32_param: self.low_precision_param}
        self._update_fp32_params_by_new_state()

    def _update_fp32_params_by_new_state(self):
        if not self.param_update_in_fp32:
            return
        for param, value in self.state.items():
            fp32_param = self.param_to_fp32_param[param]
            fp32_param.data.copy_(value["master_param"])


class DistributedOptimizer:
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.config = SimpleNamespace(use_precision_aware_optimizer_no_fp8_or_ds_fp8=False)

    def _set_main_param_and_optimizer_states(self, model_param, tensors):
        group_index, group_order = self.model_param_group_index_map[model_param]
        main_param = self.optimizer.param_groups[group_index]["params"][group_order]
        optim_state = self.optimizer.state[main_param]
        if self.config.use_precision_aware_optimizer_no_fp8_or_ds_fp8:
            for key, value in tensors.items():
                optim_state["master_param" if key == "param" else key] = value
            return
        for key, destination in {"param": main_param, **optim_state}.items():
            destination.copy_(tensors[key])

    def load_state_dict(self, state_dict):
        groups = state_dict["optimizer"]["param_groups"]
        # Real MCore restores the non-sharded step before copying the sharded tensor payload.
        self.authoritative_step = TensorValue(groups[0]["step"])
        initial_state = {
            param_id: {**{key: TensorValue("initial") for key in ("exp_avg", "exp_avg_sq", "master_param")},
                       "step": self.authoritative_step}
            for param_id in state_dict["param_state"]
        }
        self.optimizer.load_state_dict({"state": initial_state, "param_groups": groups})
        params = self.optimizer.param_groups[0]["params"]
        self.model_param_group_index_map = {param: (0, order) for order, param in enumerate(params)}
        for param, tensors in zip(params, state_dict["param_state"].values()):
            self._set_main_param_and_optimizer_states(param, {"param": tensors["master_param"], **tensors})


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
                if optimizer.optimizer.param_update_in_fp32:
                    optimizer.load_state_dict(state_dict)
                else:
                    optimizer.optimizer.state = state_dict


class OrdinaryOptimizer:
    def __init__(self):
        self.loading_flags = []
        self.loaded_states = []

    def sharded_state_dict(self, state_dict, *, is_loading, metadata):
        self.loading_flags.append(is_loading)
        return {"template": True}

    def load_state_dict(self, state_dict):
        self.loaded_states.append(state_dict)


class ExactOptimizer:
    def __init__(self):
        self.loading_flags = []
        self.loaded_state = None

    def sharded_state_dict(self, state_dict, *, is_loading, metadata):
        self.loading_flags.append(is_loading)
        return {"optimizer-template": True}

    def load_state_dict(self, state_dict):
        self.loaded_state = {
            "optimizer": {"param_groups": [dict(state_dict["optimizer"]["param_groups"][0])]},
            "param_state": {
                key: {name: value.clone() for name, value in values.items()}
                for key, values in state_dict["param_state"].items()
            },
        }


class Model:
    def __init__(self, parameter=None):
        self.parameter = parameter

    def sharded_state_dict(self, **kwargs):
        return {"weight": "model-template", "metadata": kwargs["metadata"]}

    def load_state_dict(self, state_dict):
        if self.parameter is not None:
            self.parameter.value = state_dict["weight"]


def _manager(optimizer, model=None, *, mcore_016=True, extra_state_provider=None, extra_state_loader=None):
    manager = SimpleNamespace(
        model=[model or Model()],
        optimizer=optimizer,
        lr_scheduler=None,
        get_rng_state=lambda: {"rng": "extra-template"},
        _build_sharded_state_dict_metadata=lambda: {
            "distrib_optim_sharding_type": "dp_reshardable",
            "chained_optim_avoid_prefix": True,
        },
        extra_state_provider=extra_state_provider,
        extra_state_loader=extra_state_loader,
    )
    manager.generate_state_dict = MethodType(_generate_state_dict(mcore_016=mcore_016), manager)
    return manager


def _production_methods(*, mcore_016=True):
    tree = ast.parse(MANAGER.read_text(encoding="utf-8"), filename=str(MANAGER))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "MegatronCheckpointManager")
    methods = [
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name in {"generate_state_dict", "load_checkpoint", "load_rng_states"}
    ]
    helpers = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name
        in {
            "_is_mcore_016_hdo_dp_reshardable_resume",
            "_prepare_mcore_016_hdo_dp_reshardable_resume",
            "_load_mcore_016_hdo_checkpoint_state",
        }
    ]
    for helper in helpers:
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
        "gc": gc,
    }
    nodes = [
        ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
        *helpers,
        *methods,
    ]
    module = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    exec(compile(module, str(MANAGER), "exec"), namespace)
    return namespace


def _generate_state_dict(*, mcore_016=True):
    return _production_methods(mcore_016=mcore_016)["generate_state_dict"]


def _step_restore_case(step=250):
    hdo = HybridDeviceOptimizer(initialized=False)
    main_param = torch.zeros(2)
    hdo.param_groups = [{"params": [main_param]}]
    hdo.state = {main_param: {key: torch.zeros(2) for key in ("exp_avg", "exp_avg_sq", "master_param")}}
    distributed = DistributedOptimizer(hdo)
    distributed.model_param_group_index_map = {main_param: (0, 0)}
    chained = ChainedOptimizer(distributed)
    source = {key: torch.full((2,), value) for key, value in
              (("param", 1.), ("exp_avg", 2.), ("exp_avg_sq", 3.), ("master_param", 4.))}
    checkpoint = {"optimizer": {"param_groups": [{"step": step}]}, "param_state": source}
    authority = {}

    def load(state_dict):
        authority["step"] = torch.tensor(state_dict["optimizer"]["param_groups"][0]["step"])
        hdo.state[main_param]["step"] = authority["step"]
        distributed._set_main_param_and_optimizer_states(main_param, state_dict["param_state"])
        hdo._update_fp32_params_by_new_state()

    chained.load_state_dict = load
    return chained, distributed, hdo, main_param, checkpoint, authority


class TestMegatronCheckpointManagerHDOResume(unittest.TestCase):
    def test_missing_sharded_step_preserves_param_group_authority_and_object(self):
        helper = _production_methods()["_load_mcore_016_hdo_checkpoint_state"]
        for saved_step in (11, 250):
            with self.subTest(step=saved_step):
                chained, distributed, hdo, param, checkpoint, authority = _step_restore_case(saved_step)
                self.assertNotIn("step", checkpoint["param_state"])
                self.assertTrue(helper(chained, checkpoint, {"distrib_optim_sharding_type": "dp_reshardable"}, True))
                torch.testing.assert_close(param, checkpoint["param_state"]["param"])
                for key in ("exp_avg", "exp_avg_sq", "master_param"):
                    torch.testing.assert_close(hdo.state[param][key], checkpoint["param_state"][key])
                self.assertIs(hdo.state[param]["step"], authority["step"])
                self.assertEqual(hdo.state[param]["step"].item(), saved_step)
                self.assertNotIn("_set_main_param_and_optimizer_states", vars(distributed))
                self.assertNotIn("_update_fp32_params_by_new_state", vars(hdo))

    def test_missing_tensor_state_still_fails_and_restores_both_methods(self):
        helper = _production_methods()["_load_mcore_016_hdo_checkpoint_state"]
        for missing_key in ("param", "exp_avg", "exp_avg_sq", "master_param"):
            with self.subTest(missing=missing_key):
                chained, distributed, hdo, param, checkpoint, authority = _step_restore_case()
                del checkpoint["param_state"][missing_key]
                with self.assertRaises(KeyError) as error:
                    helper(chained, checkpoint, {"distrib_optim_sharding_type": "dp_reshardable"}, True)
                self.assertEqual(error.exception.args, (missing_key,))
                self.assertIs(hdo.state[param]["step"], authority["step"])
                self.assertNotIn("_set_main_param_and_optimizer_states", vars(distributed))
                self.assertNotIn("_update_fp32_params_by_new_state", vars(hdo))

    def test_source_step_and_precision_aware_path_keep_original_behavior(self):
        helper = _production_methods()["_load_mcore_016_hdo_checkpoint_state"]
        for precision_aware, source_step in ((False, True), (True, True), (True, False)):
            with self.subTest(precision_aware=precision_aware, source_step=source_step):
                chained, distributed, hdo, param, checkpoint, authority = _step_restore_case()
                distributed.config.use_precision_aware_optimizer_no_fp8_or_ds_fp8 = precision_aware
                if source_step:
                    checkpoint["param_state"]["step"] = torch.tensor(17)
                helper(chained, checkpoint, {"distrib_optim_sharding_type": "dp_reshardable"}, True)
                self.assertEqual(hdo.state[param]["step"].item(), 17 if source_step else 250)
                if precision_aware and source_step:
                    self.assertIs(hdo.state[param]["step"], checkpoint["param_state"]["step"])
                else:
                    self.assertIs(hdo.state[param]["step"], authority["step"])

    def test_step_and_existing_instance_methods_restore_after_copy_failure(self):
        helper = _production_methods()["_load_mcore_016_hdo_checkpoint_state"]
        chained, distributed, hdo, param, checkpoint, authority = _step_restore_case()

        def fail_copy(model_param, tensors):
            self.assertNotIn("step", hdo.state[param])
            raise RuntimeError("copy failed")

        original_hook = hdo._update_fp32_params_by_new_state
        distributed._set_main_param_and_optimizer_states = fail_copy
        hdo._update_fp32_params_by_new_state = original_hook
        with self.assertRaisesRegex(RuntimeError, "copy failed"):
            helper(chained, checkpoint, {"distrib_optim_sharding_type": "dp_reshardable"}, True)
        self.assertIs(hdo.state[param]["step"], authority["step"])
        self.assertIs(distributed._set_main_param_and_optimizer_states, fail_copy)
        self.assertIs(hdo._update_fp32_params_by_new_state, original_hook)

    def test_rng_loader_restores_all_actor_rng_domains(self):
        restored = {}
        methods = _production_methods()
        load = methods["load_rng_states"]
        load.__globals__.update(
            random=SimpleNamespace(setstate=lambda state: restored.update(python=state)),
            np=SimpleNamespace(random=SimpleNamespace(set_state=lambda state: restored.update(numpy=state))),
            torch=SimpleNamespace(set_rng_state=lambda state: restored.update(torch=state)),
            get_device_name=lambda: "cuda",
            get_torch_device=lambda: SimpleNamespace(set_rng_state=lambda state: restored.update(cuda=state)),
            tensor_parallel=SimpleNamespace(get_cuda_rng_tracker=lambda: SimpleNamespace(
                set_states=lambda state: restored.update(tracker=state))),
        )
        load(None, [{"random_rng_state": "python", "np_rng_state": "numpy", "torch_rng_state": "torch",
                     "cuda_rng_state": "cuda", "rng_tracker_states": {"tracker": 250}}])
        self.assertEqual(restored, {"python": "python", "numpy": "numpy", "torch": "torch",
                                    "cuda": "cuda", "tracker": {"tracker": 250}})

    def test_eu_derpo_controller_extra_state_survives_mocked_dcp_round_trip(self):
        eu_path = MANAGER.parents[2] / "trainer" / "ppo" / "eu_derpo.py"
        spec = importlib.util.spec_from_file_location("eu_derpo_checkpoint_test", eu_path)
        eu = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(eu)
        saved_controller = eu.P95MCTRController(0.012, 0.0027, 0.012, 0.018)
        saved_controller.update(0.018, 7)
        restored_controller = eu.P95MCTRController(0.012, 0.0027, 0.012, 0.018)
        manager = _manager(
            OrdinaryOptimizer(),
            extra_state_provider=saved_controller.state_dict,
            extra_state_loader=restored_controller.load_state_dict,
        )
        methods = _production_methods()
        manager.generate_state_dict = MethodType(methods["generate_state_dict"], manager)
        manager.load_checkpoint = MethodType(methods["load_checkpoint"], manager)
        manager.checkpoint_load_contents = {"extra"}
        manager.use_dist_checkpointing = True
        manager.use_distributed_optimizer = False
        manager.rank = 0
        manager.peft_cls = None
        manager.load_rng_states = lambda state: None

        storage = {}

        def save_dist_checkpointing(*, sharded_state_dict, ckpt_dir):
            storage[ckpt_dir] = copy.deepcopy(sharded_state_dict)

        save_template = manager.generate_state_dict(
            generate_model=False, generate_optimizer=False, generate_extra=True
        )
        save_dist_checkpointing(sharded_state_dict=save_template, ckpt_dir="checkpoint/dist_ckpt")
        methods["load_checkpoint"].__globals__["load_dist_checkpointing"] = (
            lambda *, sharded_state_dict, ckpt_dir: copy.deepcopy(storage[ckpt_dir])
        )
        manager.load_checkpoint("checkpoint", load_contents=("extra",), del_local_after_load=False)

        self.assertEqual(restored_controller.delta_current, saved_controller.delta_current)
        self.assertEqual(restored_controller.controller_step, saved_controller.controller_step)
        wrong_beta = eu.P95MCTRController(0.012, 0.01, 0.012, 0.018)
        with self.assertRaisesRegex(ValueError, "differs from resolved config"):
            wrong_beta.load_state_dict(storage["checkpoint/dist_ckpt"]["eu_derpo_controller"])

    def test_staged_load_clears_template_and_payload_on_failure(self):
        class TrackedDict(dict):
            cleared = False

            def clear(self):
                self.cleared = True
                super().clear()

        class FailingOptimizer:
            def load_state_dict(self, state_dict):
                raise RuntimeError("optimizer restore failed")

        template = TrackedDict({"optimizer": {"template": True}})
        payload = TrackedDict({"optimizer": {"checkpoint": True}})
        manager = _manager(FailingOptimizer())
        methods = _production_methods()
        manager.generate_state_dict = lambda *args, **kwargs: template
        manager.load_checkpoint = MethodType(methods["load_checkpoint"], manager)
        manager.checkpoint_load_contents = {"optimizer"}
        manager.use_dist_checkpointing = True
        manager.use_distributed_optimizer = True
        manager.use_checkpoint_opt_param_scheduler = False
        manager.rank = 0
        manager.peft_cls = None
        methods["load_checkpoint"].__globals__["load_dist_checkpointing"] = lambda **kwargs: payload

        with self.assertRaisesRegex(RuntimeError, "optimizer restore failed"):
            manager.load_checkpoint("checkpoint", load_contents=("optimizer",))

        self.assertTrue(template.cleared)
        self.assertTrue(payload.cleared)

    def test_staged_load_reuses_metadata_and_restores_exact_state_once(self):
        optimizer = ExactOptimizer()
        model_parameter = SimpleNamespace(value=torch.tensor([-1.0]))
        scheduler = SimpleNamespace(loaded=[], state_dict=lambda: {"step": -1})
        scheduler.load_state_dict = lambda state: scheduler.loaded.append(dict(state))
        manager = _manager(optimizer, Model(model_parameter))
        methods = _production_methods()
        manager.generate_state_dict = MethodType(methods["generate_state_dict"], manager)
        manager.load_checkpoint = MethodType(methods["load_checkpoint"], manager)
        manager.checkpoint_load_contents = {"model", "optimizer", "extra"}
        manager.use_dist_checkpointing = True
        manager.use_distributed_optimizer = True
        manager.use_hf_checkpoint = False
        # Formal Qwen3 keeps configured LR/decay values but must still restore scheduler num_steps.
        manager.use_checkpoint_opt_param_scheduler = False
        manager.lr_scheduler = scheduler
        manager.rank = 0
        manager.peft_cls = None
        manager.vanilla_bridge = True

        metadata = {"distrib_optim_sharding_type": "dp_reshardable"}
        metadata_loads = []
        methods["load_checkpoint"].__globals__["dist_checkpointing"].load_content_metadata = (
            lambda **kwargs: metadata_loads.append(kwargs) or metadata
        )
        checkpoint_optimizer = {
            "optimizer": {
                "param_groups": [
                    {
                        "lr": 1.0e-6,
                        "betas": (0.9, 0.95),
                        "eps": 1.0e-15,
                        "weight_decay": 0.0,
                        "step": 100,
                    }
                ]
            },
            "param_state": {
                0: {
                    "exp_avg": torch.tensor([1.25, -2.5]),
                    "exp_avg_sq": torch.tensor([3.5, 4.75]),
                    "master_param": torch.tensor([5.0, 6.0]),
                }
            },
        }
        rng_state = {"python": "python-rng", "numpy": "numpy-rng", "torch": "torch-rng"}
        stage_keys = []

        def load_dist_checkpointing(*, sharded_state_dict, ckpt_dir):
            stage_keys.append(tuple(sharded_state_dict))
            if "model" in sharded_state_dict:
                return {"model": {"weight": torch.tensor([42.0])}, "rng_state": rng_state}
            return {
                "optimizer": checkpoint_optimizer,
                "lr_scheduler": {"step": 100, "last_lr": 1.0e-6},
            }

        methods["load_checkpoint"].__globals__["load_dist_checkpointing"] = load_dist_checkpointing
        restored_rng = []
        manager.load_rng_states = lambda state: restored_rng.append(dict(state))

        loaded_metadata = manager.load_checkpoint(
            "checkpoint", load_contents=("model", "extra"), del_local_after_load=False
        )
        template_events = []
        manager.load_checkpoint(
            "checkpoint",
            load_contents=("optimizer",),
            sharded_sd_metadata=loaded_metadata,
            stage_callback=template_events.append,
            del_local_after_load=False,
        )

        self.assertIs(loaded_metadata, metadata)
        self.assertEqual(len(metadata_loads), 1)
        self.assertEqual(stage_keys, [("model", "rng_state"), ("optimizer", "lr_scheduler")])
        self.assertEqual(template_events, ["after_optimizer_template"])
        self.assertTrue(torch.equal(model_parameter.value, torch.tensor([42.0])))
        self.assertEqual(restored_rng, [rng_state])
        self.assertEqual(scheduler.loaded, [{"step": 100, "last_lr": 1.0e-6}])
        self.assertEqual(
            optimizer.loaded_state["optimizer"]["param_groups"],
            checkpoint_optimizer["optimizer"]["param_groups"],
        )
        for name, expected in checkpoint_optimizer["param_state"][0].items():
            self.assertTrue(torch.equal(optimizer.loaded_state["param_state"][0][name], expected))

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

    def test_final_optimizer_restore_preserves_checkpoint_state(self):
        self._check_full_hdo_restore(explicit=False)

    def test_explicit_full_hdo_restore_includes_scheduler_and_controller(self):
        self._check_full_hdo_restore(explicit=True)

    def _check_full_hdo_restore(self, *, explicit):
        parameter = SimpleNamespace(value="initial-model")
        rng = SimpleNamespace(value="initial-rng")
        scheduler = SimpleNamespace(step_calls=0, state_dict=lambda: {"scheduler": "initial"})
        scheduler.loaded = []
        scheduler.load_state_dict = lambda state: scheduler.loaded.append(dict(state))
        controller_states = []
        hdo = HybridDeviceOptimizer(initialized=True, parameter=parameter, rng=rng, realistic_restore=True)
        optimizer = ChainedOptimizer(DistributedOptimizer(hdo))
        manager = _manager(optimizer, Model(parameter))
        methods = _production_methods()
        checkpoint_optimizer_state = {
            "optimizer": {
                "param_groups": [
                    {
                        "params": [0, 1],
                        "lr": 0.125,
                        "betas": (0.8, 0.95),
                        "weight_decay": 0.01,
                        "step": 250,
                    }
                ]
            },
            "param_state": {
                0: {
                    "exp_avg": TensorValue("native-exp-avg"),
                    "exp_avg_sq": TensorValue("native-exp-avg-sq"),
                    "master_param": TensorValue("native-master"),
                },
                1: {
                    "exp_avg": TensorValue("low-exp-avg"),
                    "exp_avg_sq": TensorValue("low-exp-avg-sq"),
                    "master_param": TensorValue("low-master"),
                },
            },
        }
        methods["load_checkpoint"].__globals__["load_dist_checkpointing"] = lambda **kwargs: {
            "model": {"weight": "checkpoint-model"},
            "optimizer": checkpoint_optimizer_state,
            "rng_state": {"value": "checkpoint-rng"},
            "lr_scheduler": {"num_steps": 250},
            "eu_derpo_controller": {"mctr_delta_current": 0.013, "mctr_controller_step": 250},
        }
        manager.generate_state_dict = MethodType(methods["generate_state_dict"], manager)
        manager.load_checkpoint = MethodType(methods["load_checkpoint"], manager)
        manager.should_load_model = True
        manager.should_load_optimizer = True
        manager.should_load_extra = True
        manager.checkpoint_load_contents = {"model", "optimizer", "extra"}
        manager.use_dist_checkpointing = True
        manager.use_distributed_optimizer = True
        manager.use_hf_checkpoint = False
        manager.use_checkpoint_opt_param_scheduler = False
        manager.lr_scheduler = scheduler
        manager.rank = 0
        manager.peft_cls = None
        manager.global_step = 1
        manager.load_rng_states = lambda state: setattr(rng, "value", state["value"])
        manager.extra_state_loader = lambda state: controller_states.append(dict(state))

        manager.load_checkpoint(
            "checkpoint", del_local_after_load=False,
            load_contents=("model", "optimizer", "extra") if explicit else None,
        )

        self.assertEqual(hdo.dummy_steps, 0)
        self.assertEqual(parameter.value, "checkpoint-model")
        self.assertEqual(optimizer.loaded_states, [checkpoint_optimizer_state])
        self.assertEqual(set(hdo.state), {hdo.native_fp32_param, hdo.low_precision_param})
        self.assertIn(hdo.low_precision_param, hdo.param_to_fp32_param)
        self.assertNotIn(hdo.native_fp32_param, hdo.param_to_fp32_param)
        for param, prefix in (
            (hdo.native_fp32_param, "native"),
            (hdo.low_precision_param, "low"),
        ):
            self.assertEqual(hdo.state[param]["exp_avg"].value, f"{prefix}-exp-avg")
            self.assertEqual(hdo.state[param]["exp_avg_sq"].value, f"{prefix}-exp-avg-sq")
            self.assertEqual(hdo.state[param]["master_param"].value, f"{prefix}-master")
            self.assertEqual(hdo.state[param]["step"].value, 250)
            self.assertIs(hdo.state[param]["step"], optimizer.chained_optimizers[0].authoritative_step)
        self.assertEqual(hdo.param_groups[0]["lr"], 0.125)
        self.assertEqual(hdo.param_groups[0]["betas"], (0.8, 0.95))
        self.assertEqual(hdo.param_groups[0]["weight_decay"], 0.01)
        self.assertNotIn("_set_main_param_and_optimizer_states", vars(optimizer.chained_optimizers[0]))
        self.assertNotIn("_update_fp32_params_by_new_state", vars(hdo))
        self.assertEqual(rng.value, "checkpoint-rng")
        self.assertEqual(manager.global_step, 1)
        self.assertEqual(scheduler.step_calls, 0)
        self.assertEqual(scheduler.loaded, [{"num_steps": 250}] if explicit else [])
        self.assertEqual(controller_states, [{"mctr_delta_current": 0.013, "mctr_controller_step": 250}])

    def test_final_restore_compatibility_scope_is_exact(self):
        checkpoint_state = {"checkpoint": "state"}
        metadata = {"distrib_optim_sharding_type": "dp_reshardable"}

        ordinary = OrdinaryOptimizer()
        helper = _production_methods()["_load_mcore_016_hdo_checkpoint_state"]
        self.assertFalse(helper(ordinary, checkpoint_state, metadata, full_resume=True))
        self.assertEqual(ordinary.loaded_states, [checkpoint_state])

        non_hdo = DistributedOptimizer(OrdinaryOptimizer())
        non_hdo_chain = ChainedOptimizer(non_hdo)
        self.assertFalse(helper(non_hdo_chain, checkpoint_state, metadata, full_resume=True))
        self.assertEqual(non_hdo_chain.loaded_states, [checkpoint_state])
        self.assertNotIn("_set_main_param_and_optimizer_states", vars(non_hdo))

        for helper_metadata, full_resume in (
            ({"distrib_optim_sharding_type": "fully_reshardable"}, True),
            (metadata, False),
        ):
            hdo = HybridDeviceOptimizer(initialized=True)
            optimizer = ChainedOptimizer(DistributedOptimizer(hdo))
            self.assertFalse(helper(optimizer, checkpoint_state, helper_metadata, full_resume=full_resume))
            self.assertEqual(optimizer.loaded_states, [checkpoint_state])

        hdo = HybridDeviceOptimizer(initialized=True)
        optimizer = ChainedOptimizer(DistributedOptimizer(hdo))
        other_version_helper = _production_methods(mcore_016=False)[
            "_load_mcore_016_hdo_checkpoint_state"
        ]
        self.assertFalse(other_version_helper(optimizer, checkpoint_state, metadata, full_resume=True))
        self.assertEqual(optimizer.loaded_states, [checkpoint_state])

    @unittest.skipUnless(importlib.util.find_spec("megatron") is not None, "Megatron-Core is not installed")
    def test_real_mcore_final_restore_compatibility(self):
        import megatron.core
        from megatron.core.optimizer import ChainedOptimizer as RealChainedOptimizer
        from megatron.core.optimizer.cpu_offloading import HybridDeviceOptimizer as RealHybridDeviceOptimizer
        from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer as RealDistributedOptimizer
        from packaging import version

        if version.parse(megatron.core.__version__).release[:2] != (0, 16):
            self.skipTest("requires Megatron-Core 0.16.x")

        hdo = object.__new__(RealHybridDeviceOptimizer)
        model_param = torch.zeros(2)
        main_param = torch.zeros(2)
        hdo.param_update_in_fp32 = True
        authoritative_step = torch.tensor(250)
        hdo.state = {main_param: {
            "exp_avg": torch.zeros(2), "exp_avg_sq": torch.zeros(2),
            "master_param": torch.zeros(2), "step": authoritative_step,
        }}
        hdo.param_groups = [{"params": [main_param], "step": 250}]
        hdo.param_to_fp32_param = {}
        with self.assertRaises(KeyError):
            hdo._update_fp32_params_by_new_state()

        distributed = object.__new__(RealDistributedOptimizer)
        distributed.optimizer = hdo
        distributed.config = SimpleNamespace(use_precision_aware_optimizer_no_fp8_or_ds_fp8=False)
        distributed.model_param_group_index_map = {model_param: (0, 0)}
        source = {key: torch.full((2,), value) for key, value in
                  (("param", 1.), ("exp_avg", 2.), ("exp_avg_sq", 3.), ("master_param", 4.))}
        self.assertNotIn("step", source)
        with self.assertRaises(KeyError) as error:
            RealDistributedOptimizer._set_main_param_and_optimizer_states(distributed, model_param, source)
        self.assertEqual(error.exception.args, ("step",))
        main_param.zero_()
        for key in ("exp_avg", "exp_avg_sq", "master_param"):
            hdo.state[main_param][key].zero_()

        chained = object.__new__(RealChainedOptimizer)
        chained.chained_optimizers = [distributed]
        load_calls = []

        def load_state_dict(optimizer, state_dict):
            load_calls.append(state_dict)
            # Dispatch through the instance wrapper into the real, failing MCore setter.
            optimizer.chained_optimizers[0]._set_main_param_and_optimizer_states(model_param, source)
            optimizer.chained_optimizers[0].optimizer._update_fp32_params_by_new_state()

        chained.load_state_dict = MethodType(load_state_dict, chained)
        from verl.utils.checkpoint.megatron_checkpoint_manager import (
            _load_mcore_016_hdo_checkpoint_state as helper,
        )

        self.assertTrue(
            helper(
                chained,
                {"checkpoint": "optimizer-state"},
                metadata={"distrib_optim_sharding_type": "dp_reshardable"},
                full_resume=True,
            )
        )
        self.assertEqual(load_calls, [{"checkpoint": "optimizer-state"}])
        torch.testing.assert_close(main_param, source["param"])
        for key in ("exp_avg", "exp_avg_sq", "master_param"):
            torch.testing.assert_close(hdo.state[main_param][key], source[key])
        self.assertIs(hdo.state[main_param]["step"], authoritative_step)
        self.assertEqual(authoritative_step.item(), 250)
        self.assertNotIn("_set_main_param_and_optimizer_states", vars(distributed))
        self.assertNotIn("_update_fp32_params_by_new_state", vars(hdo))
        self.assertIs(distributed._set_main_param_and_optimizer_states.__func__,
                      RealDistributedOptimizer._set_main_param_and_optimizer_states)
        with self.assertRaises(KeyError):
            hdo._update_fp32_params_by_new_state()


if __name__ == "__main__":
    unittest.main()
