# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import ast
import unittest
from pathlib import Path
from types import MethodType, SimpleNamespace


WORKER = Path(__file__).resolve().parents[2] / "verl" / "workers" / "megatron_workers.py"


def _worker_method(namespace):
    tree = ast.parse(WORKER.read_text(encoding="utf-8"), filename=str(WORKER))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "ActorRolloutRefWorker")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "load_checkpoint")
    method.decorator_list = []
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(WORKER), "exec"), namespace)
    return namespace["load_checkpoint"]


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


def _staged_worker(events, *, fail_stage=None, model_offload_reduces_memory=True):
    residency = {"model": False, "optimizer": False}

    def log_memory(stage, **kwargs):
        events.append(("mem", stage))
        allocated = {
            "after_model_load": 100,
            "after_model_offload": 10 if model_offload_reduces_memory else 100,
            "after_optimizer_load": 120,
            "after_optimizer_offload": 10,
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
        assert not residency["model"]
        residency["optimizer"] = True
        events.append("optimizer_onload")

    def optimizer_offload(optimizer):
        residency["optimizer"] = False
        events.append("optimizer_offload")

    namespace = {
        "load_megatron_model_to_gpu": model_onload,
        "offload_megatron_model_to_cpu": model_offload,
        "load_megatron_optimizer": optimizer_onload,
        "offload_megatron_optimizer": optimizer_offload,
        "aggressive_empty_cache": lambda **kwargs: events.append("empty_cache"),
        "_is_gpu_adam_distributed_optimizer": lambda optimizer: True,
        "_log_resume_memory": log_memory,
        "log_gpu_memory_usage": lambda *args, **kwargs: None,
        "logger": None,
    }
    worker = SimpleNamespace(
        _is_offload_param=True,
        _is_offload_optimizer=True,
        actor_module=object(),
        actor_optimizer=object(),
        checkpoint_mananager=CheckpointManager(events, residency, fail_stage),
    )
    worker.load_checkpoint = MethodType(_worker_method(namespace), worker)
    return worker, residency


class TestMegatronGPUAdamStagedResume(unittest.TestCase):
    def test_none_checkpoint_keeps_existing_offload_only_path(self):
        events = []
        worker, residency = _staged_worker(events)

        worker.load_checkpoint(None, staged_restore=True)

        self.assertEqual(residency, {"model": False, "optimizer": False})
        self.assertEqual(events, ["model_offload", "optimizer_offload"])

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
        self.assertEqual(events[-1], ("mem", "restore_complete"))

    def test_model_failure_offloads_and_never_onloads_optimizer(self):
        events = []
        worker, residency = _staged_worker(events, fail_stage="model")

        with self.assertRaisesRegex(RuntimeError, "model restore failed"):
            worker.load_checkpoint("global_step_100", staged_restore=True)

        self.assertEqual(residency, {"model": False, "optimizer": False})
        self.assertIn("model_offload", events)
        self.assertNotIn("optimizer_onload", events)

    def test_model_stage_stops_if_offload_does_not_reduce_gpu_allocation(self):
        events = []
        worker, residency = _staged_worker(events, model_offload_reduces_memory=False)

        with self.assertRaisesRegex(RuntimeError, "model offload did not reduce"):
            worker.load_checkpoint("global_step_100", staged_restore=True)

        self.assertEqual(residency, {"model": False, "optimizer": False})
        self.assertNotIn("optimizer_onload", events)

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

    def test_host_memory_guard_fails_before_large_allocation(self):
        class ResumeHostMemoryBudgetExceeded(RuntimeError):
            pass

        memory = SimpleNamespace(used=850 * 1024**3, available=150 * 1024**3, percent=85.0)
        guard = _module_function(
            "_log_resume_memory",
            {
                "psutil": SimpleNamespace(virtual_memory=lambda: memory),
                "get_torch_device": lambda: SimpleNamespace(is_available=lambda: False),
                "torch": SimpleNamespace(distributed=SimpleNamespace(is_initialized=lambda: False)),
                "os": SimpleNamespace(getenv=lambda *args: None),
                "logger": SimpleNamespace(warning=lambda *args, **kwargs: None),
                "_GIB": 1024**3,
                "_RESUME_MAX_HOST_MEMORY_PERCENT": 80.0,
                "_RESUME_MIN_HOST_AVAILABLE_BYTES": 200 * 1024**3,
                "ResumeHostMemoryBudgetExceeded": ResumeHostMemoryBudgetExceeded,
            },
        )

        with self.assertRaisesRegex(ResumeHostMemoryBudgetExceeded, "used=85.0%"):
            guard("before_optimizer_load", enforce_host_budget=True)


if __name__ == "__main__":
    unittest.main()
