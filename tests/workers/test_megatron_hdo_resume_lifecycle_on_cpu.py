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


class Parameter:
    def __init__(self, is_cuda):
        self.is_cuda = is_cuda
        self.grad = "checkpoint-gradient"


class SubOptimizer:
    def __init__(self, param):
        self.param_groups = [{"params": [param]}]


class HybridDeviceOptimizer:
    def __init__(self, original_param):
        self.original_param = original_param
        self.state = {}
        self.rebuild_topology()

    def rebuild_topology(self):
        original_param = self.original_param
        if original_param.is_cuda:
            inner_param = Parameter(is_cuda=False)
            self.gpu_params_map_cpu_copy = {original_param: inner_param}
            self.cpu_copys_map_gpu_param = {inner_param: original_param}
        else:
            inner_param = original_param
            self.gpu_params_map_cpu_copy = {}
            self.cpu_copys_map_gpu_param = {}
        self.cpu_optimizers = [SubOptimizer(inner_param)]
        self.param_to_inner_param = {original_param: inner_param}
        self.inner_param_to_orig_param = {inner_param: original_param}

    def set_sub_optimizer_grads(self):
        for optimizer in self.cpu_optimizers:
            for group in optimizer.param_groups:
                for param in group["params"]:
                    self.cpu_copys_map_gpu_param[param]


class DistributedOptimizer:
    def __init__(self, original_param):
        self.shard_fp32_from_float16_groups = [[original_param]]
        self.optimizer = HybridDeviceOptimizer(original_param)


class CheckpointManager:
    def __init__(self, distributed_optimizer, events, checkpoint_state):
        self.distributed_optimizer = distributed_optimizer
        self.events = events
        self.checkpoint_state = checkpoint_state
        self.calls = 0

    def load_checkpoint(self, **kwargs):
        self.calls += 1
        hdo = self.distributed_optimizer.optimizer
        self.events.append(("restore", hdo.original_param.is_cuda))
        hdo.rebuild_topology()
        hdo.state = self.checkpoint_state


def _load_checkpoint_method(load_optimizer, offload_optimizer):
    tree = ast.parse(WORKER.read_text(encoding="utf-8"), filename=str(WORKER))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "ActorRolloutRefWorker")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "load_checkpoint")
    method.decorator_list = []
    namespace = {
        "load_megatron_optimizer": load_optimizer,
        "offload_megatron_optimizer": offload_optimizer,
        "load_megatron_model_to_gpu": lambda model: None,
        "offload_megatron_model_to_cpu": lambda model: None,
        "log_gpu_memory_usage": lambda *args, **kwargs: None,
        "logger": None,
    }
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(WORKER), "exec"), namespace)
    return namespace["load_checkpoint"]


class TestMegatronHDOResumeLifecycle(unittest.TestCase):
    def test_offloaded_optimizer_is_reloaded_before_restore_and_first_step(self):
        events = []
        original_param = Parameter(is_cuda=True)
        distributed_optimizer = DistributedOptimizer(original_param)
        checkpoint_state = {
            "exp_avg": "checkpoint-exp-avg",
            "exp_avg_sq": "checkpoint-exp-avg-sq",
            "master_param": "checkpoint-master-param",
            "step": 1,
            "param_groups": {
                "lr": 1e-6,
                "betas": (0.9, 0.95),
                "weight_decay": 0.1,
            },
        }

        def load_optimizer(optimizer):
            events.append("load")
            original_param.is_cuda = True

        def offload_optimizer(optimizer):
            events.append("offload")
            original_param.is_cuda = False

        # Actor initialization externally offloads the DistributedOptimizer main param.
        offload_optimizer(distributed_optimizer)
        events.clear()
        checkpoint_manager = CheckpointManager(distributed_optimizer, events, checkpoint_state)
        worker = SimpleNamespace(
            _is_offload_param=False,
            _is_offload_optimizer=True,
            actor_module=object(),
            actor_optimizer=distributed_optimizer,
            checkpoint_mananager=checkpoint_manager,
        )
        worker.load_checkpoint = MethodType(
            _load_checkpoint_method(load_optimizer, offload_optimizer), worker
        )

        worker.load_checkpoint("global_step_1")
        load_optimizer(distributed_optimizer)  # update_actor training residency

        hdo = distributed_optimizer.optimizer
        cpu_param = hdo.cpu_optimizers[0].param_groups[0]["params"][0]
        hdo.set_sub_optimizer_grads()
        self.assertEqual(events[:3], ["load", ("restore", True), "offload"])
        self.assertIsNot(cpu_param, original_param)
        self.assertIs(hdo.cpu_copys_map_gpu_param[cpu_param], original_param)
        self.assertTrue(original_param.is_cuda)
        self.assertIs(hdo.param_to_inner_param[original_param], cpu_param)
        self.assertIs(hdo.inner_param_to_orig_param[cpu_param], original_param)
        self.assertIs(hdo.state, checkpoint_state)
        self.assertEqual(hdo.state["exp_avg"], "checkpoint-exp-avg")
        self.assertEqual(hdo.state["exp_avg_sq"], "checkpoint-exp-avg-sq")
        self.assertEqual(hdo.state["master_param"], "checkpoint-master-param")
        self.assertEqual(hdo.state["step"], 1)
        self.assertEqual(
            hdo.state["param_groups"],
            {"lr": 1e-6, "betas": (0.9, 0.95), "weight_decay": 0.1},
        )

    def test_none_checkpoint_keeps_existing_offload_only_behavior(self):
        events = []
        manager = SimpleNamespace(load_checkpoint=lambda **kwargs: events.append("restore"))
        worker = SimpleNamespace(
            _is_offload_param=False,
            _is_offload_optimizer=True,
            actor_module=object(),
            actor_optimizer=object(),
            checkpoint_mananager=manager,
        )
        worker.load_checkpoint = MethodType(
            _load_checkpoint_method(
                lambda optimizer: events.append("load"),
                lambda optimizer: events.append("offload"),
            ),
            worker,
        )

        worker.load_checkpoint(None)

        self.assertEqual(events, ["offload"])

    def test_non_offloaded_optimizer_is_not_moved(self):
        events = []
        manager = SimpleNamespace(load_checkpoint=lambda **kwargs: events.append("restore"))
        worker = SimpleNamespace(
            _is_offload_param=False,
            _is_offload_optimizer=False,
            actor_module=object(),
            actor_optimizer=object(),
            checkpoint_mananager=manager,
        )
        worker.load_checkpoint = MethodType(
            _load_checkpoint_method(
                lambda optimizer: events.append("load"),
                lambda optimizer: events.append("offload"),
            ),
            worker,
        )

        worker.load_checkpoint("global_step_1")

        self.assertEqual(events, ["restore"])

    def test_restore_failure_reoffloads_optimizer(self):
        events = []

        def fail_restore(**kwargs):
            events.append("restore")
            raise RuntimeError("checkpoint restore failed")

        manager = SimpleNamespace(load_checkpoint=fail_restore)
        worker = SimpleNamespace(
            _is_offload_param=False,
            _is_offload_optimizer=True,
            actor_module=object(),
            actor_optimizer=object(),
            checkpoint_mananager=manager,
        )
        worker.load_checkpoint = MethodType(
            _load_checkpoint_method(
                lambda optimizer: events.append("load"),
                lambda optimizer: events.append("offload"),
            ),
            worker,
        )

        with self.assertRaisesRegex(RuntimeError, "checkpoint restore failed"):
            worker.load_checkpoint("global_step_1")

        self.assertEqual(events, ["load", "restore", "offload"])


if __name__ == "__main__":
    unittest.main()
