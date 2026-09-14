from __future__ import annotations

import ast
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest


WORKER = Path(__file__).resolve().parents[2] / "verl" / "workers" / "megatron_workers.py"
MANAGER = Path(__file__).resolve().parents[2] / "verl" / "utils" / "checkpoint" / "megatron_checkpoint_manager.py"
UTILS = Path(__file__).resolve().parents[2] / "verl" / "utils" / "megatron_utils.py"


def _residency_helpers():
    tree = ast.parse(UTILS.read_text(encoding="utf-8"), filename=str(UTILS))
    names = {"is_megatron_model_offloaded", "is_megatron_optimizer_offloaded"}
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]

    class DDP:
        pass

    class ChainedOptimizer:
        pass

    namespace = {"DDP": DDP, "ChainedOptimizer": ChainedOptimizer}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(UTILS), "exec"), namespace)
    return namespace, DDP


def _save_checkpoint_method(events, residency, fail=False):
    tree = ast.parse(WORKER.read_text(encoding="utf-8"), filename=str(WORKER))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "ActorRolloutRefWorker")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "save_checkpoint")
    method.decorator_list = []

    def load_model(_):
        assert not residency["model"]
        residency["model"] = True
        events.append("load_model")

    def load_optimizer(_):
        assert not residency["optimizer"]
        residency["optimizer"] = True
        events.append("load_optimizer")

    def offload_model(_):
        assert residency["model"]
        residency["model"] = False
        events.append("offload_model")

    def offload_optimizer(_):
        assert residency["optimizer"]
        residency["optimizer"] = False
        events.append("offload_optimizer")

    namespace = {
        "is_megatron_model_offloaded": lambda _: not residency["model"],
        "is_megatron_optimizer_offloaded": lambda _: not residency["optimizer"],
        "load_megatron_model_to_gpu": load_model,
        "load_megatron_optimizer": load_optimizer,
        "offload_megatron_model_to_cpu": offload_model,
        "offload_megatron_optimizer": offload_optimizer,
        "megatron_model_cpu_data_bytes": lambda _: 0,
        "log_eu_derpo_memory": lambda stage, *_: events.append(stage),
        "torch": SimpleNamespace(distributed=SimpleNamespace(barrier=lambda: events.append("worker_barrier"))),
    }
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(WORKER), "exec"), namespace)

    class CheckpointManager:
        def save_checkpoint(self, stage_callback=None, **_):
            assert residency == {"model": True, "optimizer": True}
            events.append("save")
            stage_callback("after_checkpoint_state_dict")
            if fail:
                raise RuntimeError("save failed")
            stage_callback("after_checkpoint_write")

    worker = SimpleNamespace(
        _is_offload_param=True,
        _is_offload_optimizer=True,
        actor_module=object(),
        actor_optimizer=object(),
        config=SimpleNamespace(actor=SimpleNamespace(eu_derpo=SimpleNamespace(enabled=True))),
        checkpoint_mananager=CheckpointManager(),
    )
    return MethodType(namespace["save_checkpoint"], worker)


def test_phase_offloaded_save_loads_training_residency_then_reoffloads():
    events = []
    residency = {"model": False, "optimizer": False}
    save = _save_checkpoint_method(events, residency)
    save("/checkpoint", global_step=1)
    assert residency == {"model": False, "optimizer": False}
    assert events == [
        "before_checkpoint",
        "load_model",
        "load_optimizer",
        "after_checkpoint_training_residency_load",
        "save",
        "after_checkpoint_state_dict",
        "after_checkpoint_write",
        "worker_barrier",
        "offload_model",
        "offload_optimizer",
        "after_checkpoint_reoffload",
    ]


def test_save_failure_still_reoffloads():
    events = []
    residency = {"model": False, "optimizer": False}
    save = _save_checkpoint_method(events, residency, fail=True)
    with pytest.raises(RuntimeError, match="save failed"):
        save("/checkpoint", global_step=1)
    assert residency == {"model": False, "optimizer": False}
    assert events[-3:] == ["offload_model", "offload_optimizer", "after_checkpoint_reoffload"]


def test_already_training_residency_is_not_loaded_or_offloaded_again():
    events = []
    residency = {"model": True, "optimizer": True}
    save = _save_checkpoint_method(events, residency)
    save("/checkpoint", global_step=1)
    assert residency == {"model": True, "optimizer": True}
    assert not any(event.startswith("load_") or event.startswith("offload_") for event in events)


def test_residency_guards_read_storage_and_device_not_python_identity():
    helpers, ddp_type = _residency_helpers()
    model = ddp_type()
    size = {"value": 0}
    param_data = SimpleNamespace(storage=lambda: SimpleNamespace(size=lambda: size["value"]))
    model.buffers = [SimpleNamespace(param_data=param_data)]
    model.expert_parallel_buffers = []
    assert helpers["is_megatron_model_offloaded"]([model])
    size["value"] = 4
    assert not helpers["is_megatron_model_offloaded"]([model])

    param = SimpleNamespace(device=SimpleNamespace(type="cpu"))
    optimizer = SimpleNamespace(shard_fp32_from_float16_groups=[[param]])
    assert helpers["is_megatron_optimizer_offloaded"](optimizer)
    param.device.type = "cuda"
    assert not helpers["is_megatron_optimizer_offloaded"](optimizer)


def test_manager_telemetry_brackets_state_dict_and_sync_write():
    source = MANAGER.read_text(encoding="utf-8")
    start = source.index("    def save_checkpoint(")
    save = source[start:]
    generated = save.index("state_dict = self.generate_state_dict")
    state_dict_stage = save.index('stage_callback("after_checkpoint_state_dict")')
    write = save.index("async_save_request = save_dist_checkpointing")
    write_stage = save.index('stage_callback("after_checkpoint_write")')
    assert generated < state_dict_stage < write < write_stage
