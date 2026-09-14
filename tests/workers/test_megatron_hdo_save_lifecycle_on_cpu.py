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


def _save_checkpoint_method(events, residency, *, held=False, fail=False, cuda_free_gib=32):
    tree = ast.parse(WORKER.read_text(encoding="utf-8"), filename=str(WORKER))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "ActorRolloutRefWorker")
    release = next(
        node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "_release_checkpoint_residency_hold"
    )
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

    class Flag:
        def __init__(self, value):
            self.value = value

        def item(self):
            return self.value

    class Device:
        def mem_get_info(self):
            return cuda_free_gib * 1024**3, 80 * 1024**3

        def memory_allocated(self):
            return 46 * 1024**3

        def memory_reserved(self):
            return 46 * 1024**3

    distributed = SimpleNamespace(
        barrier=lambda: events.append("worker_barrier"),
        all_reduce=lambda _, **__: events.append("headroom_all_reduce"),
        ReduceOp=SimpleNamespace(MAX="max"),
        get_rank=lambda: 0,
    )
    fake_torch = SimpleNamespace(
        distributed=distributed,
        int32="int32",
        tensor=lambda value, **_: Flag(value),
    )
    namespace = {
        "_GIB": 1024**3,
        "_CHECKPOINT_MIN_CUDA_FREE_BYTES": 16 * 1024**3,
        "_CHECKPOINT_MAX_CUDA_RESERVED_BYTES": 64 * 1024**3,
        "is_megatron_model_offloaded": lambda _: not residency["model"],
        "is_megatron_optimizer_offloaded": lambda _: not residency["optimizer"],
        "load_megatron_model_to_gpu": load_model,
        "load_megatron_optimizer": load_optimizer,
        "offload_megatron_model_to_cpu": offload_model,
        "offload_megatron_optimizer": offload_optimizer,
        "megatron_model_cpu_data_bytes": lambda _: 0,
        "log_eu_derpo_memory": lambda stage, *_, **__: events.append(stage),
        "aggressive_empty_cache": lambda **_: events.append("empty_cache"),
        "get_torch_device": Device,
        "get_device_name": lambda: "cpu",
        "torch": fake_torch,
    }
    exec(compile(ast.Module(body=[release, method], type_ignores=[]), str(WORKER), "exec"), namespace)

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
        _checkpoint_training_residency_held=held,
    )
    worker._release_checkpoint_residency_hold = MethodType(
        namespace["_release_checkpoint_residency_hold"], worker
    )
    return MethodType(namespace["save_checkpoint"], worker)


def _finish_update_method(events, residency):
    tree = ast.parse(WORKER.read_text(encoding="utf-8"), filename=str(WORKER))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "ActorRolloutRefWorker")
    method = next(
        node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "_finish_actor_update_residency"
    )

    namespace = {
        "offload_megatron_model_to_cpu": lambda _: (events.append("offload_model"), residency.update(model=False)),
        "offload_megatron_optimizer": lambda _: (
            events.append("offload_optimizer"), residency.update(optimizer=False)
        ),
        "log_gpu_memory_usage": lambda message, **_: events.append(message),
        "log_eu_derpo_memory": lambda stage, *_, **__: events.append(stage),
        "megatron_model_cpu_data_bytes": lambda _: 0,
        "logger": object(),
    }
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(WORKER), "exec"), namespace)
    worker = SimpleNamespace(
        _is_offload_param=True,
        _is_offload_optimizer=True,
        _checkpoint_training_residency_held=False,
        actor_module=object(),
        actor_optimizer=object(),
    )
    return MethodType(namespace["_finish_actor_update_residency"], worker), worker


def test_non_save_update_phase_offloads_exactly_once():
    events = []
    residency = {"model": True, "optimizer": True}
    finish, worker = _finish_update_method(events, residency)
    finish(False, True)
    assert residency == {"model": False, "optimizer": False}
    assert events.count("offload_model") == 1
    assert events.count("offload_optimizer") == 1
    assert events[-1] == "after_actor_phase_offload"
    assert worker._checkpoint_training_residency_held is False


def test_save_update_holds_training_residency_without_offload():
    events = []
    residency = {"model": True, "optimizer": True}
    finish, worker = _finish_update_method(events, residency)
    finish(True, True)
    assert residency == {"model": True, "optimizer": True}
    assert events == ["checkpoint_hold_enter"]
    assert worker._checkpoint_training_residency_held is True


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


def test_checkpoint_hold_skips_reload_and_reoffloads_after_sync_write():
    events = []
    residency = {"model": True, "optimizer": True}
    save = _save_checkpoint_method(events, residency, held=True)
    save("/checkpoint", global_step=1)
    assert residency == {"model": False, "optimizer": False}
    assert events == [
        "before_checkpoint",
        "after_checkpoint_training_residency_load",
        "headroom_all_reduce",
        "checkpoint_gpu_headroom",
        "save",
        "after_checkpoint_state_dict",
        "after_checkpoint_write",
        "worker_barrier",
        "offload_model",
        "offload_optimizer",
        "after_checkpoint_reoffload",
        "checkpoint_hold_exit",
        "empty_cache",
    ]


def test_checkpoint_hold_save_failure_still_reoffloads():
    events = []
    residency = {"model": True, "optimizer": True}
    save = _save_checkpoint_method(events, residency, held=True, fail=True)
    with pytest.raises(RuntimeError, match="save failed"):
        save("/checkpoint", global_step=1)
    assert residency == {"model": False, "optimizer": False}
    assert events[-5:] == [
        "offload_model",
        "offload_optimizer",
        "after_checkpoint_reoffload",
        "checkpoint_hold_exit",
        "empty_cache",
    ]


def test_checkpoint_gpu_headroom_failure_skips_write_and_releases_hold():
    events = []
    residency = {"model": True, "optimizer": True}
    save = _save_checkpoint_method(events, residency, held=True, cuda_free_gib=15)
    with pytest.raises(RuntimeError, match="CHECKPOINT_GPU_HEADROOM_INSUFFICIENT"):
        save("/checkpoint", global_step=1)
    assert "save" not in events
    assert residency == {"model": False, "optimizer": False}
    assert events[-5:] == [
        "offload_model",
        "offload_optimizer",
        "after_checkpoint_reoffload",
        "checkpoint_hold_exit",
        "empty_cache",
    ]


def test_checkpoint_hold_release_is_idempotent():
    events = []
    residency = {"model": True, "optimizer": True}
    save = _save_checkpoint_method(events, residency, held=True)
    worker = save.__self__
    assert worker._release_checkpoint_residency_hold() is True
    after_first = list(events)
    assert worker._release_checkpoint_residency_hold() is False
    assert events == after_first


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
