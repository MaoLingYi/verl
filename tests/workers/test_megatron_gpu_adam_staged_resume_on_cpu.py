# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import ast
import importlib.util
import unittest
from pathlib import Path
from types import MethodType, SimpleNamespace

import torch


WORKER = Path(__file__).resolve().parents[2] / "verl" / "workers" / "megatron_workers.py"


def _worker_method(namespace):
    tree = ast.parse(WORKER.read_text(encoding="utf-8"), filename=str(WORKER))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "ActorRolloutRefWorker")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "load_checkpoint")
    offload_actor = next(
        node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "_offload_actor_optimizer"
    )
    method.decorator_list = []
    exec(compile(ast.Module(body=[offload_actor, method], type_ignores=[]), str(WORKER), "exec"), namespace)
    return namespace["load_checkpoint"], namespace["_offload_actor_optimizer"]


def _module_function(name, namespace, *, strip_imports=False):
    tree = ast.parse(WORKER.read_text(encoding="utf-8"), filename=str(WORKER))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
    if strip_imports:
        function.body = [node for node in function.body if not isinstance(node, ast.ImportFrom)]
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(WORKER), "exec"), namespace)
    return namespace[name]


class CheckpointManager:
    use_dist_checkpointing = True
    should_load_model = True
    should_load_optimizer = True
    should_load_extra = True

    def _build_sharded_state_dict_metadata(self):
        return {"distrib_optim_sharding_type": "dp_reshardable"}

    def __init__(self, events, residency, fail_stage=None):
        self.events = events
        self.residency = residency
        self.fail_stage = fail_stage
        self.metadata = {"distrib_optim_sharding_type": "dp_reshardable"}

    def load_checkpoint(self, **kwargs):
        contents = tuple(kwargs["load_contents"])
        self.events.append(("load", contents))
        if contents == ("model", "extra"):
            assert self.residency == {"model": True, "optimizer": False}
            if self.fail_stage == "model":
                raise RuntimeError("model restore failed")
            self.events.append("model_restore")
            return self.metadata
        assert contents == ("optimizer",)
        assert kwargs["sharded_sd_metadata"] is self.metadata
        assert self.residency == {"model": False, "optimizer": True}
        kwargs["stage_callback"]("after_optimizer_template")
        if self.fail_stage == "optimizer":
            raise RuntimeError("optimizer restore failed")
        self.events.append("optimizer_restore")


def _staged_worker(
    events, *, fail_stage=None, model_offload_reduces_memory=True, optimizer_offload_reduces_memory=True,
    gpu_adam=True, hdo=False, empty_hdo=False, normalize_rollout=False,
):
    residency = {"model": False, "optimizer": False}

    def log_memory(stage, **kwargs):
        events.append(("mem", stage))
        allocated = {
            "after_model_load": 100,
            "after_model_offload": 10 if model_offload_reduces_memory else 100,
            "after_optimizer_load": 120,
            "after_optimizer_offload": 10 if optimizer_offload_reduces_memory else 120,
        }.get(stage, 10)
        return {"gpu_available": True, "gpu_allocated": allocated}

    def model_onload(model):
        assert not residency["optimizer"]
        residency["model"] = True
        events.append("model_onload")

    def model_offload(model):
        residency["model"] = False
        events.append("model_offload")

    def optimizer_onload(optimizer):
        assert hdo or not residency["model"]
        residency["optimizer"] = True
        events.append("optimizer_onload")

    def optimizer_offload(optimizer):
        residency["optimizer"] = False
        events.append("optimizer_offload")

    def dummy_bootstrap():
        assert residency == {"model": False, "optimizer": True}
        events.append("dummy_bootstrap")
        if fail_stage == "bootstrap":
            raise RuntimeError("bootstrap failed")
        worker.actor_optimizer.state = {"dummy": {"step": 1}}

    namespace = {
        "_mcore_016_hdo_resume_needs_bootstrap": lambda optimizer, *args: not optimizer.state,
        "_prepare_mcore_016_hdo_dp_reshardable_resume": lambda *args: dummy_bootstrap(),
        "load_megatron_model_to_gpu": model_onload,
        "offload_megatron_model_to_cpu": model_offload,
        "load_megatron_optimizer": optimizer_onload,
        "offload_megatron_optimizer": optimizer_offload,
        "load_megatron_optimizer_copy_params_to_gpu": lambda *_: (_ for _ in ()).throw(
            AssertionError("selective helper must not be called during staged resume")
        ),
        "offload_megatron_optimizer_copy_params_to_cpu": lambda *_: (_ for _ in ()).throw(
            AssertionError("selective helper must not be called during staged resume")
        ),
        "aggressive_empty_cache": lambda **kwargs: events.append("empty_cache"),
        "_is_gpu_adam_distributed_optimizer": lambda optimizer: gpu_adam,
        "_is_hdo_distributed_optimizer": lambda optimizer: hdo,
        "checkpoint_mcore_016": True,
        "log_eu_derpo_memory": lambda stage, *args, **kwargs: events.append(stage),
        "megatron_model_cpu_data_bytes": lambda models: 8 * 1024**3,
        "_best_effort_release_host_allocator_after_copy_param_restore": lambda: events.append("host_reclaim"),
        "_log_resume_memory": log_memory,
        "_reset_resume_peak_memory": lambda: events.append("reset_peak"),
        "log_gpu_memory_usage": lambda *args, **kwargs: None,
        "logger": SimpleNamespace(
            warning=lambda message, *args, **kwargs: events.append(("warning", message % args))
        ),
        "_GIB": 1024**3,
    }
    worker = SimpleNamespace(
        _is_offload_param=True,
        _is_offload_optimizer=True,
        actor_module=object(),
        actor_optimizer=SimpleNamespace(state={} if empty_hdo else {"existing": {}}),
        checkpoint_mananager=CheckpointManager(events, residency, fail_stage),
        _hdo_optimizer_residency_preserved=True,
        _preserve_hdo_optimizer_residency=lambda: normalize_rollout,
        _selective_optimizer_copy_param_offload_enabled=lambda: normalize_rollout,
    )

    def selective_offload():
        assert residency == {"model": False, "optimizer": True}
        events.append("selective_optimizer_offload")
        if fail_stage == "normalization":
            raise RuntimeError("normalization failed")
        residency["optimizer"] = False
        worker._optimizer_copy_params_offloaded_for_rollout = True
        worker._optimizer_copy_params_cuda_bytes_before_rollout_offload = 64
        worker._hdo_optimizer_residency_preserved = True

    worker._offload_actor_optimizer_copy_params_for_rollout = selective_offload
    load_checkpoint, offload_actor = _worker_method(namespace)
    worker._offload_actor_optimizer = MethodType(offload_actor, worker)
    worker.load_checkpoint = MethodType(load_checkpoint, worker)
    tree = ast.parse(WORKER.read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "ActorRolloutRefWorker")
    update_load = next(node for node in cls.body if isinstance(node, ast.FunctionDef)
                       and node.name == "_load_actor_optimizer_for_update")
    exec(compile(ast.Module(body=[update_load], type_ignores=[]), str(WORKER), "exec"), namespace)
    worker._load_actor_optimizer_for_update = MethodType(namespace[update_load.name], worker)
    return worker, residency


class TestMegatronGPUAdamStagedResume(unittest.TestCase):
    def test_successful_cold_resume_normalizes_once_after_payloads_before_rollout(self):
        events = []
        worker, residency = _staged_worker(events, gpu_adam=False, hdo=True, normalize_rollout=True)
        authority = {"exp_avg": torch.tensor([2.]), "exp_avg_sq": torch.tensor([3.]),
                     "master_param": torch.tensor([4.]), "step": torch.tensor(250)}
        groups = [{"params": [object()], "lr": 1e-6, "step": 250}]
        extras = {
            "scheduler": {"num_steps": 250, "lr": 1e-6},
            "rng": {"python": 1, "numpy": 2, "torch": 3, "cuda": 4, "tracker": 5},
            "mctr": {"controller_step": 250, "delta_current": 0.013},
        }
        expected_state = {key: tensor.clone() for key, tensor in authority.items()}
        expected_groups = [dict(group, params=list(group["params"])) for group in groups]
        expected_extras = {key: dict(value) if isinstance(value, dict) else value for key, value in extras.items()}

        def load(**kwargs):
            worker.actor_optimizer.state = authority
            worker.actor_optimizer.param_groups = groups
            worker.restored_extra = extras
            kwargs["stage_callback"]("after_model_restore")
            events.append("checkpoint_payloads_released")

        worker.checkpoint_mananager.load_checkpoint = load
        worker.load_checkpoint("global_step_250/actor", staged_restore=True)
        order = ["checkpoint_payloads_released", "before_resume_rollout_residency", "optimizer_onload",
                 "selective_optimizer_offload", "after_resume_rollout_residency", "after_full_restore_cleanup"]
        self.assertEqual([events.index(stage) for stage in order], sorted(events.index(stage) for stage in order))
        self.assertEqual(events.count("selective_optimizer_offload"), 1)
        self.assertTrue(worker._optimizer_copy_params_offloaded_for_rollout)
        self.assertTrue(worker._hdo_optimizer_residency_preserved)
        self.assertEqual(events.count("optimizer_offload"), 1)  # Only startup; final cleanup preserves partial HDO.
        self.assertIs(worker.actor_optimizer.state, authority)
        self.assertIs(worker.actor_optimizer.param_groups, groups)
        self.assertIs(worker.restored_extra, extras)
        self.assertEqual(worker.actor_optimizer.state["step"].item(), 250)
        self.assertEqual(residency, {"model": False, "optimizer": False})
        namespace = worker.load_checkpoint.__func__.__globals__
        tree = ast.parse(WORKER.read_text(encoding="utf-8"))
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                   and node.name == "ActorRolloutRefWorker")
        restore = next(node for node in cls.body if isinstance(node, ast.FunctionDef)
                       and node.name == "_restore_actor_optimizer_copy_params_for_update")
        copy_state = {"cuda": False}

        def restore_copy_params(optimizer):
            self.assertIs(optimizer, worker.actor_optimizer)
            self.assertTrue(worker._optimizer_copy_params_offloaded_for_rollout)
            self.assertFalse(copy_state["cuda"])
            events.append("selective_copy_param_restore")
            copy_state["cuda"] = True
            return {"moved_tensors": 2, "moved_bytes": 64}

        def reclaim():
            self.assertTrue(copy_state["cuda"])
            self.assertFalse(worker._optimizer_copy_params_offloaded_for_rollout)
            events.append("selective_host_reclaim")
            return {"gc_collected": 1, "malloc_trim_attempted": 1, "malloc_trim_rc": 1}

        namespace.update(
            load_megatron_optimizer=lambda *_: self.fail("generic full optimizer load must remain skipped"),
            load_megatron_optimizer_copy_params_to_gpu=restore_copy_params,
            estimate_optimizer_phase_offload_memory=lambda _: {
                "optimizer_phase_cuda_copy_param_bytes": 64 if copy_state["cuda"] else 0,
                "optimizer_phase_cuda_state_bytes": 80,
            },
            get_torch_device=lambda: SimpleNamespace(synchronize=lambda: events.append("restore_cuda_sync")),
            torch=SimpleNamespace(distributed=SimpleNamespace(is_initialized=lambda: False)),
            psutil=SimpleNamespace(
                Process=lambda: SimpleNamespace(memory_info=lambda: SimpleNamespace(rss=8 * 1024**3)),
                virtual_memory=lambda: SimpleNamespace(available=128 * 1024**3),
            ),
            time=SimpleNamespace(perf_counter=lambda: 0.0),
            _best_effort_release_host_allocator_after_copy_param_restore=reclaim,
        )

        def complete_transition(stage, failure, error, moved):
            if error is not None:
                raise error
            events.append(stage)

        worker._complete_optimizer_copy_param_transition = complete_transition
        worker._actor_optimizer_residency = lambda: (
            "ROLLOUT_PARTIAL" if worker._optimizer_copy_params_offloaded_for_rollout else "TRAINING_RESIDENT"
        )
        exec(compile(ast.Module(body=[restore], type_ignores=[]), str(WORKER), "exec"), namespace)
        worker._restore_actor_optimizer_copy_params_for_update = MethodType(namespace[restore.name], worker)
        self.assertTrue(worker._restore_actor_optimizer_copy_params_for_update())
        self.assertFalse(worker._optimizer_copy_params_offloaded_for_rollout)
        self.assertTrue(worker._hdo_optimizer_residency_preserved)
        self.assertEqual(events.count("selective_copy_param_restore"), 1)
        self.assertEqual(events.count("selective_host_reclaim"), 1)
        transition_order = ["selective_copy_param_restore", "restore_cuda_sync", "restore_sanity",
                            "selective_host_reclaim", "after_optimizer_copy_param_restore"]
        positions = [events.index(stage) for stage in transition_order]
        self.assertEqual(positions, sorted(positions))
        event_count = len(events)
        self.assertFalse(worker._restore_actor_optimizer_copy_params_for_update())
        self.assertFalse(worker._load_actor_optimizer_for_update())
        self.assertEqual(len(events), event_count)
        self.assertIs(worker.actor_optimizer.state, authority)
        for key, expected in expected_state.items():
            torch.testing.assert_close(worker.actor_optimizer.state[key], expected, rtol=0, atol=0)
        self.assertIs(worker.actor_optimizer.param_groups, groups)
        self.assertEqual(groups, expected_groups)
        self.assertIs(worker.restored_extra, extras)
        self.assertEqual(extras, expected_extras)

    def test_failed_normalization_full_offloads_and_does_not_retry_or_enter_rollout(self):
        events = []
        worker, residency = _staged_worker(events, gpu_adam=False, hdo=True,
                                          normalize_rollout=True, fail_stage="normalization")
        worker.checkpoint_mananager.load_checkpoint = lambda **kwargs: kwargs["stage_callback"]("after_model_restore")
        with self.assertRaisesRegex(RuntimeError, "normalization failed"):
            worker.load_checkpoint("global_step_250/actor", staged_restore=True)
        self.assertEqual(events.count("selective_optimizer_offload"), 1)
        self.assertEqual(residency, {"model": False, "optimizer": False})
        self.assertFalse(worker._optimizer_copy_params_offloaded_for_rollout)
        self.assertFalse(worker._hdo_optimizer_residency_preserved)
        self.assertNotIn("after_resume_rollout_residency", events)

    def test_fresh_run_does_not_normalize_even_when_selective_policy_enabled(self):
        events = []
        worker, _ = _staged_worker(events, gpu_adam=False, hdo=True, normalize_rollout=True)
        worker.load_checkpoint(None, staged_restore=True)
        self.assertEqual(events, ["model_offload", "optimizer_offload"])

    def test_empty_hdo_bootstraps_once_before_model_onload_and_returns_to_native_update(self):
        events = []
        worker, residency = _staged_worker(events, gpu_adam=False, hdo=True, empty_hdo=True)

        def load(**kwargs):
            self.assertTrue(worker.actor_optimizer.state)
            self.assertEqual(residency, {"model": True, "optimizer": False})
            events.append("model_restore")
            kwargs["stage_callback"]("after_model_restore")
            self.assertEqual(residency, {"model": False, "optimizer": False})
            events.append("optimizer_restore")

        worker.checkpoint_mananager.load_checkpoint = load
        worker.load_checkpoint("global_step_250/actor", staged_restore=True)
        order = ["before_hdo_bootstrap", "optimizer_onload", "after_hdo_onload_for_bootstrap",
                 "dummy_bootstrap", "after_hdo_dummy_bootstrap", "after_hdo_bootstrap_offload",
                 "after_hdo_bootstrap_reclaim", "model_onload", "model_restore", "after_model_restore",
                 "optimizer_restore", "after_full_restore_cleanup"]
        positions = [events.index(event) for event in order]
        self.assertEqual(positions, sorted(positions))
        bootstrap_end = events.index("after_hdo_bootstrap_reclaim")
        self.assertIn("optimizer_offload", events[events.index("dummy_bootstrap"):bootstrap_end])
        self.assertIn("empty_cache", events[events.index("after_hdo_bootstrap_offload"):bootstrap_end])
        self.assertIn("host_reclaim", events[:bootstrap_end])
        self.assertEqual(events.count("dummy_bootstrap"), 1)
        self.assertEqual(residency, {"model": False, "optimizer": False})
        self.assertTrue(worker._load_actor_optimizer_for_update())
        self.assertEqual(events[-1], "optimizer_onload")
        self.assertEqual(events.count("dummy_bootstrap"), 1)

    def test_bootstrap_and_restore_failures_preserve_primary_even_when_cleanup_fails(self):
        for phase in ("bootstrap", "model", "optimizer"):
            with self.subTest(phase=phase):
                events = []
                worker, residency = _staged_worker(events, gpu_adam=False, hdo=True, empty_hdo=True)
                namespace = worker.load_checkpoint.__func__.__globals__
                primary = RuntimeError(phase + " primary failure")

                def fail():
                    raise primary

                if phase == "bootstrap":
                    namespace["_prepare_mcore_016_hdo_dp_reshardable_resume"] = lambda *args: fail()

                def load(**kwargs):
                    if phase == "optimizer":
                        kwargs["stage_callback"]("after_model_restore")
                    fail()

                worker.checkpoint_mananager.load_checkpoint = load
                namespace["aggressive_empty_cache"] = lambda **kwargs: (
                    (_ for _ in ()).throw(RuntimeError("cleanup CUDA failure"))
                    if (phase == "bootstrap" or "model_onload" in events) else None
                )
                with self.assertRaises(RuntimeError) as error:
                    worker.load_checkpoint("global_step_250/actor", staged_restore=True)
                self.assertIs(error.exception, primary)
                self.assertEqual(residency, {"model": False, "optimizer": False})
                self.assertTrue(any(isinstance(event, tuple) and event[0] == "warning" for event in events))
                self.assertEqual(events[-1], "after_full_restore_cleanup")
                if phase == "bootstrap":
                    self.assertNotIn("model_onload", events)

    @unittest.skipUnless(importlib.util.find_spec("megatron") is not None, "Megatron-Core is not installed")
    def test_real_mcore_cold_restore_empty_hdo_bootstrap_lifecycle(self):
        self._check_real_dummy_bootstrap("cpu")

    @unittest.skipUnless(torch.cuda.is_available() and importlib.util.find_spec("megatron") is not None,
                         "requires CUDA and Megatron-Core")
    def test_real_mcore_gpu_cold_restore_empty_hdo_bootstrap_lifecycle(self):
        self._check_real_dummy_bootstrap("cuda")

    def _check_real_dummy_bootstrap(self, device):
        import megatron.core
        from packaging import version
        from megatron.core.optimizer.cpu_offloading import HybridDeviceOptimizer as RealHDO
        from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer as RealDistributed
        from verl.utils.checkpoint.megatron_checkpoint_manager import (
            _mcore_016_hdo_resume_needs_bootstrap, _prepare_mcore_016_hdo_dp_reshardable_resume,
        )

        if version.parse(megatron.core.__version__).release[:2] != (0, 16):
            self.skipTest("requires Megatron-Core 0.16.x")
        events = []
        worker, residency = _staged_worker(events, gpu_adam=False, hdo=True, empty_hdo=True)
        param = torch.zeros(2, device=device)
        hdo = object.__new__(RealHDO)
        torch.optim.Optimizer.__init__(hdo, [param], {"lr": 1e-6})
        self.assertFalse(hdo.state)

        def instrumented_step(self):
            # Real dummy_step creates gradients and calls this step at the residency boundary.
            assert residency == {"model": False, "optimizer": True}
            assert param.grad is not None
            events.append("real_dummy_step_entry")
            self.state[param] = {"step": torch.tensor(1), "exp_avg": torch.full_like(param, 111)}

        hdo.step = MethodType(instrumented_step, hdo)
        distributed = object.__new__(RealDistributed)
        distributed.optimizer = hdo
        worker.actor_optimizer = distributed
        namespace = worker.load_checkpoint.__func__.__globals__
        namespace["_mcore_016_hdo_resume_needs_bootstrap"] = _mcore_016_hdo_resume_needs_bootstrap
        namespace["_prepare_mcore_016_hdo_dp_reshardable_resume"] = _prepare_mcore_016_hdo_dp_reshardable_resume

        def load(**kwargs):
            self.assertEqual(hdo.state[param]["step"].item(), 1)
            self.assertIsNone(param.grad)
            self.assertFalse(_mcore_016_hdo_resume_needs_bootstrap(distributed,
                             {"distrib_optim_sharding_type": "dp_reshardable"}, True))
            self.assertTrue(_prepare_mcore_016_hdo_dp_reshardable_resume(distributed, True,
                            {"distrib_optim_sharding_type": "dp_reshardable"}, True, cold_restore=True))
            kwargs["stage_callback"]("after_model_restore")

        worker.checkpoint_mananager.load_checkpoint = load
        worker.load_checkpoint("global_step_250/actor", staged_restore=True)
        self.assertEqual(events.count("real_dummy_step_entry"), 1)
        self.assertEqual(residency, {"model": False, "optimizer": False})

    def test_explicit_hdo_resume_uses_full_state_contents_and_cleans_residency(self):
        events = []
        worker, residency = _staged_worker(events, gpu_adam=False, hdo=True)
        restored = {}
        saved = {
            "model": "checkpoint-model", "optimizer": {"step": 250},
            "lr_scheduler": {"num_steps": 250},
            "eu_derpo_controller": {"mctr_delta_current": 0.013, "mctr_controller_step": 250},
            "rng_state": {"python": 1, "numpy": 2, "torch": 3, "cuda": 4, "tracker": 5},
        }

        def load(**kwargs):
            self.assertEqual(kwargs["local_path"], "global_step_250/actor")
            self.assertEqual(kwargs["load_contents"], ("model", "optimizer", "extra"))
            self.assertEqual(residency, {"model": True, "optimizer": False})
            self.assertTrue(kwargs["cold_restore"])
            for stage in ("before_model_restore", "after_model_restore", "after_model_restore_reclaim",
                          "before_optimizer_common_restore", "after_optimizer_common_restore",
                          "before_optimizer_param_state_restore", "after_optimizer_param_state_restore"):
                kwargs["stage_callback"](stage)
            # The real manager's explicit-contents scheduler/MCTR/RNG round trip is tested separately.
            restored.update(saved)
            events.append("full_state_restore")

        worker.checkpoint_mananager.load_checkpoint = load
        worker.load_checkpoint("global_step_250/actor", staged_restore=True)

        self.assertEqual(restored, saved)
        self.assertEqual(events.count("full_state_restore"), 1)
        self.assertNotIn("optimizer_onload", events)
        self.assertNotIn("dummy_bootstrap", events)
        self.assertLess(events.index("optimizer_offload"), events.index("model_onload"))
        self.assertLess(events.index("after_model_restore_reclaim"), events.index("before_optimizer_common_restore"))
        self.assertLess(events.index("full_state_restore"), events.index("host_reclaim"))
        self.assertEqual(events[-1], "after_full_restore_cleanup")
        self.assertEqual(residency, {"model": False, "optimizer": False})
        self.assertFalse(worker._optimizer_copy_params_offloaded_for_rollout)
        self.assertNotIn("reset_peak", events)
        # The next training update uses the unchanged native residency loader, not cold restore.
        self.assertTrue(worker._load_actor_optimizer_for_update())
        self.assertEqual(events[-1], "optimizer_onload")
        self.assertEqual(events.count("full_state_restore"), 1)
        self.assertEqual(events.count("after_full_restore_cleanup"), 1)

    def test_explicit_resume_rejects_unsupported_optimizer_before_loading(self):
        events = []
        worker, _ = _staged_worker(events, gpu_adam=False)
        with self.assertRaisesRegex(RuntimeError, "Explicit resume requires"):
            worker.load_checkpoint("global_step_250/actor", staged_restore=True)
        self.assertEqual(events, [])

    def test_hdo_explicit_resume_failure_cleans_residency_without_retry(self):
        events = []
        worker, residency = _staged_worker(events, gpu_adam=False, hdo=True)

        def fail(**kwargs):
            events.append("failed_load")
            raise RuntimeError("corrupt payload")

        worker.checkpoint_mananager.load_checkpoint = fail
        with self.assertRaisesRegex(RuntimeError, "corrupt payload"):
            worker.load_checkpoint("global_step_250/actor", staged_restore=True)
        self.assertEqual(events.count("failed_load"), 1)
        self.assertEqual(residency, {"model": False, "optimizer": False})

    def test_none_checkpoint_keeps_existing_offload_only_path(self):
        events = []
        worker, residency = _staged_worker(events)

        worker.load_checkpoint(None, staged_restore=True)

        self.assertEqual(residency, {"model": False, "optimizer": False})
        self.assertEqual(events, ["model_offload", "optimizer_offload"])
        self.assertFalse(worker._hdo_optimizer_residency_preserved)

    def test_model_and_optimizer_restore_in_separate_residency_stages(self):
        events = []
        worker, residency = _staged_worker(events)
        runtime_optimizer = worker.actor_optimizer

        worker.load_checkpoint("global_step_100", staged_restore=True)

        self.assertEqual(residency, {"model": False, "optimizer": False})
        self.assertIs(worker.actor_optimizer, runtime_optimizer)
        self.assertLess(events.index("model_offload"), events.index("optimizer_onload"))
        self.assertEqual(events.count(("load", ("model", "extra"))), 1)
        self.assertEqual(events.count(("load", ("optimizer",))), 1)
        self.assertIn(("mem", "after_optimizer_template"), events)
        self.assertEqual(events.count("reset_peak"), 2)
        self.assertEqual(events[-1], ("mem", "restore_complete"))

    def test_model_failure_offloads_and_never_onloads_optimizer(self):
        events = []
        worker, residency = _staged_worker(events, fail_stage="model")

        with self.assertRaisesRegex(RuntimeError, "model restore failed"):
            worker.load_checkpoint("global_step_100", staged_restore=True)

        self.assertEqual(residency, {"model": False, "optimizer": False})
        self.assertIn("model_offload", events)
        self.assertNotIn("optimizer_onload", events)

    def test_model_offload_delta_warns_and_optimizer_stage_continues(self):
        events = []
        worker, residency = _staged_worker(events, model_offload_reduces_memory=False)

        worker.load_checkpoint("global_step_100", staged_restore=True)

        self.assertEqual(residency, {"model": False, "optimizer": False})
        self.assertIn("optimizer_onload", events)
        self.assertTrue(any("model offload did not reduce" in event[1] for event in events if event[0] == "warning"))
        self.assertEqual(events[-1], ("mem", "restore_complete"))

    def test_optimizer_offload_delta_warns_without_failing_restore(self):
        events = []
        worker, residency = _staged_worker(events, optimizer_offload_reduces_memory=False)

        worker.load_checkpoint("global_step_100", staged_restore=True)

        self.assertEqual(residency, {"model": False, "optimizer": False})
        self.assertTrue(
            any("optimizer offload did not reduce" in event[1] for event in events if event[0] == "warning")
        )
        self.assertEqual(events[-1], ("mem", "restore_complete"))

    def test_optimizer_failure_keeps_model_offloaded_and_cleans_optimizer(self):
        events = []
        worker, residency = _staged_worker(events, fail_stage="optimizer")

        with self.assertRaisesRegex(RuntimeError, "optimizer restore failed"):
            worker.load_checkpoint("global_step_100", staged_restore=True)

        self.assertEqual(residency, {"model": False, "optimizer": False})
        self.assertLess(events.index("model_offload"), events.index("optimizer_onload"))
        self.assertIn("optimizer_offload", events)
        self.assertNotIn(("mem", "restore_complete"), events)

    def test_runtime_backend_accepts_only_distributed_fused_adam(self):
        class FusedAdam:
            pass

        class HybridDeviceOptimizer:
            pass

        class DistributedOptimizer:
            def __init__(self, optimizer):
                self.optimizer = optimizer

        class ChainedOptimizer:
            def __init__(self, *optimizers):
                self.chained_optimizers = optimizers

        predicate = _module_function(
            "_is_gpu_adam_distributed_optimizer",
            {
                "FusedAdam": FusedAdam,
                "DistributedOptimizer": DistributedOptimizer,
                "ChainedOptimizer": ChainedOptimizer,
            },
            strip_imports=True,
        )

        self.assertTrue(predicate(DistributedOptimizer(FusedAdam())))
        self.assertTrue(predicate(ChainedOptimizer(DistributedOptimizer(FusedAdam()))))
        self.assertFalse(predicate(DistributedOptimizer(HybridDeviceOptimizer())))
        self.assertFalse(predicate(DistributedOptimizer(object())))
        hdo_predicate = _module_function(
            "_is_hdo_distributed_optimizer",
            {
                "HybridDeviceOptimizer": HybridDeviceOptimizer,
                "DistributedOptimizer": DistributedOptimizer,
                "ChainedOptimizer": ChainedOptimizer,
            },
            strip_imports=True,
        )
        self.assertTrue(hdo_predicate(DistributedOptimizer(HybridDeviceOptimizer())))
        self.assertTrue(hdo_predicate(ChainedOptimizer(DistributedOptimizer(HybridDeviceOptimizer()))))
        self.assertFalse(hdo_predicate(DistributedOptimizer(FusedAdam())))
        self.assertFalse(hdo_predicate(DistributedOptimizer(object())))
        self.assertFalse(hdo_predicate(ChainedOptimizer()))
        self.assertFalse(hdo_predicate(ChainedOptimizer(
            DistributedOptimizer(HybridDeviceOptimizer()), DistributedOptimizer(FusedAdam())
        )))

    def test_host_memory_threshold_warns_and_returns_telemetry(self):
        memory = SimpleNamespace(used=850 * 1024**3, available=150 * 1024**3, percent=85.0)
        warnings = []
        guard = _module_function(
            "_log_resume_memory",
            {
                "psutil": SimpleNamespace(virtual_memory=lambda: memory),
                "get_torch_device": lambda: SimpleNamespace(is_available=lambda: False),
                "torch": SimpleNamespace(distributed=SimpleNamespace(is_initialized=lambda: False)),
                "os": SimpleNamespace(getenv=lambda *args: None),
                "logger": SimpleNamespace(warning=lambda message, *args: warnings.append(message % args)),
                "_GIB": 1024**3,
                "_RESUME_MAX_HOST_MEMORY_PERCENT": 80.0,
                "_RESUME_MIN_HOST_AVAILABLE_BYTES": 200 * 1024**3,
            },
        )

        telemetry = guard("before_optimizer_load")

        self.assertEqual(telemetry["host_used"], memory.used)
        self.assertEqual(telemetry["host_available"], memory.available)
        self.assertEqual(telemetry["host_percent"], memory.percent)
        self.assertTrue(any("RESUME_MEM_WARNING host budget" in warning for warning in warnings))

    def test_peak_reset_is_cpu_safe(self):
        reset_calls = []
        reset = _module_function(
            "_reset_resume_peak_memory",
            {
                "get_torch_device": lambda: SimpleNamespace(
                    is_available=lambda: False,
                    reset_peak_memory_stats=lambda: reset_calls.append(True),
                )
            },
        )

        reset()

        self.assertEqual(reset_calls, [])


if __name__ == "__main__":
    unittest.main()
