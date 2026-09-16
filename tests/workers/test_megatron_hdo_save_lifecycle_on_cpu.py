from __future__ import annotations

import ast
import contextlib
from pathlib import Path
import time
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


def _save_checkpoint_method(
    events,
    residency,
    *,
    held=False,
    fail=False,
    cuda_free_gib=32,
    skip_post_checkpoint_optimizer_offload=False,
    preserve_hdo_optimizer_residency_between_steps=False,
):
    tree = ast.parse(WORKER.read_text(encoding="utf-8"), filename=str(WORKER))
    cls = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "ActorRolloutRefWorker"
    )
    release = next(
        node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "_release_checkpoint_residency_hold"
    )
    offload_actor = next(
        node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "_offload_actor_optimizer"
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
        "log_eu_derpo_memory": lambda stage, *_, **__: (
            events.append(stage) or {"mem_available_gib": 400.0, "rss_gib": 100.0}
        ),
        "aggressive_empty_cache": lambda **_: events.append("empty_cache"),
        "get_torch_device": Device,
        "get_device_name": lambda: "cpu",
        "logger": SimpleNamespace(warning=lambda message: events.append(message)),
        "torch": fake_torch,
    }
    exec(compile(ast.Module(body=[offload_actor, release, method], type_ignores=[]), str(WORKER), "exec"), namespace)

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
        config=SimpleNamespace(
            actor=SimpleNamespace(
                eu_derpo=SimpleNamespace(
                    enabled=True,
                    skip_post_checkpoint_optimizer_offload=skip_post_checkpoint_optimizer_offload,
                    preserve_hdo_optimizer_residency_between_steps=(
                        preserve_hdo_optimizer_residency_between_steps
                    ),
                )
            )
        ),
        checkpoint_mananager=CheckpointManager(),
        _checkpoint_training_residency_held=held,
        _hdo_optimizer_residency_preserved=False,
    )
    worker._preserve_hdo_optimizer_residency = MethodType(
        lambda self: bool(
            self.config.actor.eu_derpo.enabled
            and self.config.actor.eu_derpo.preserve_hdo_optimizer_residency_between_steps
        ),
        worker,
    )
    worker._offload_actor_optimizer = MethodType(namespace["_offload_actor_optimizer"], worker)
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
    offload_actor = next(
        node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "_offload_actor_optimizer"
    )

    namespace = {
        "offload_megatron_model_to_cpu": lambda _: (events.append("offload_model"), residency.update(model=False)),
        "offload_megatron_optimizer": lambda _: (
            events.append("offload_optimizer"), residency.update(optimizer=False)
        ),
        "log_gpu_memory_usage": lambda message, **_: events.append(message),
        "log_eu_derpo_memory": lambda stage, *_, **__: (
            events.append(stage) or {"mem_available_gib": 400.0, "rss_gib": 100.0}
        ),
        "megatron_model_cpu_data_bytes": lambda _: 0,
        "estimate_optimizer_phase_offload_memory": lambda _: {
            "optimizer_phase_cuda_total_bytes": 8 * 1024**3
        },
        "psutil": SimpleNamespace(
            virtual_memory=lambda: SimpleNamespace(available=400 * 1024**3),
            Process=lambda: SimpleNamespace(memory_info=lambda: SimpleNamespace(rss=100 * 1024**3)),
        ),
        "time": time,
        "_GIB": 1024**3,
        "logger": SimpleNamespace(warning=lambda message: events.append(message)),
    }
    exec(compile(ast.Module(body=[offload_actor, method], type_ignores=[]), str(WORKER), "exec"), namespace)
    worker = SimpleNamespace(
        _is_offload_param=True,
        _is_offload_optimizer=True,
        _checkpoint_training_residency_held=False,
        actor_module=object(),
        actor_optimizer=object(),
        _hdo_optimizer_residency_preserved=False,
    )
    worker._offload_actor_optimizer = MethodType(namespace["_offload_actor_optimizer"], worker)
    return MethodType(namespace["_finish_actor_update_residency"], worker), worker


def test_non_save_update_phase_offloads_exactly_once():
    events = []
    residency = {"model": True, "optimizer": True}
    finish, worker = _finish_update_method(events, residency)
    finish(False, True)
    assert residency == {"model": False, "optimizer": False}
    assert events.count("offload_model") == 1
    assert events.count("offload_optimizer") == 1
    assert events[-1] == "after_optimizer_residency_decision"
    assert worker._checkpoint_training_residency_held is False


def test_non_save_update_preserves_partial_hdo_but_offloads_model():
    events = []
    residency = {"model": True, "optimizer": True}
    finish, worker = _finish_update_method(events, residency)
    finish(False, True, True)
    assert residency == {"model": False, "optimizer": True}
    assert events.count("offload_model") == 1
    assert "offload_optimizer" not in events
    assert worker._hdo_optimizer_residency_preserved is True
    assert events == [
        "before_actor_residency_transition",
        "offload_model",
        "After offload actor params and grad during update_actor",
        "after_model_offload",
        "optimizer phase offload skipped to preserve native partial HDO residency",
        "after_optimizer_residency_decision",
    ]


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
        "post_ckpt_reoffload_before",
        "offload_model",
        "post_ckpt_after_param_offload",
        "post_ckpt_before_optimizer_offload",
        "offload_optimizer",
        "post_ckpt_after_optimizer_offload",
        "post_ckpt_reoffload_done",
        "checkpoint_hold_exit",
        "empty_cache",
    ]


def test_checkpoint_hold_experiment_offloads_param_but_keeps_hdo_gpu_residency():
    events = []
    residency = {"model": True, "optimizer": True}
    save = _save_checkpoint_method(events, residency, held=True, skip_post_checkpoint_optimizer_offload=True)
    assert save.__self__._is_offload_optimizer is True
    save("/checkpoint", global_step=1)
    assert residency == {"model": False, "optimizer": True}
    assert "offload_model" in events
    assert "offload_optimizer" not in events
    assert "optimizer phase offload skipped by experiment" in events
    stages = [
        "post_ckpt_reoffload_before",
        "post_ckpt_after_param_offload",
        "post_ckpt_before_optimizer_offload",
        "post_ckpt_after_optimizer_offload",
        "post_ckpt_reoffload_done",
        "checkpoint_hold_exit",
    ]
    assert [event for event in events if event in stages] == stages


def test_checkpoint_hold_preserve_flag_keeps_partial_hdo_residency():
    events = []
    residency = {"model": True, "optimizer": True}
    save = _save_checkpoint_method(
        events,
        residency,
        held=True,
        preserve_hdo_optimizer_residency_between_steps=True,
    )
    save("/checkpoint", global_step=1)
    assert residency == {"model": False, "optimizer": True}
    assert "offload_optimizer" not in events
    assert save.__self__._hdo_optimizer_residency_preserved is True


def test_checkpoint_hold_experiment_save_failure_still_releases_hold():
    events = []
    residency = {"model": True, "optimizer": True}
    save = _save_checkpoint_method(
        events, residency, held=True, fail=True, skip_post_checkpoint_optimizer_offload=True
    )
    with pytest.raises(RuntimeError, match="save failed"):
        save("/checkpoint", global_step=1)
    assert residency == {"model": False, "optimizer": True}
    assert save.__self__._checkpoint_training_residency_held is False
    assert events[-2:] == ["checkpoint_hold_exit", "empty_cache"]


def test_checkpoint_hold_save_failure_still_reoffloads():
    events = []
    residency = {"model": True, "optimizer": True}
    save = _save_checkpoint_method(events, residency, held=True, fail=True)
    with pytest.raises(RuntimeError, match="save failed"):
        save("/checkpoint", global_step=1)
    assert residency == {"model": False, "optimizer": False}
    assert events[-9:] == [
        "post_ckpt_reoffload_before",
        "offload_model",
        "post_ckpt_after_param_offload",
        "post_ckpt_before_optimizer_offload",
        "offload_optimizer",
        "post_ckpt_after_optimizer_offload",
        "post_ckpt_reoffload_done",
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
    assert events[-9:] == [
        "post_ckpt_reoffload_before",
        "offload_model",
        "post_ckpt_after_param_offload",
        "post_ckpt_before_optimizer_offload",
        "offload_optimizer",
        "post_ckpt_after_optimizer_offload",
        "post_ckpt_reoffload_done",
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


def test_experiment_flags_only_change_intended_optimizer_lifecycle_calls():
    source = WORKER.read_text(encoding="utf-8")
    actor_worker = source[
        source.index("class ActorRolloutRefWorker") : source.index("class AsyncActorRolloutRefWorker")
    ]
    assert actor_worker.count("offload_megatron_optimizer(self.actor_optimizer)") == 1
    assert "def _offload_actor_optimizer" in actor_worker
    assert source.count("skip_post_checkpoint_optimizer_offload") == 1
    release = source[source.index("    def _release_checkpoint_residency_hold") :]
    assert "skip_post_checkpoint_optimizer_offload" in release
    init = source[source.index("    def init_model") : source.index("    async def rollout_mode")]
    normal_update = source[
        source.index("    def _finish_actor_update_residency") :
        source.index("    def _release_checkpoint_residency_hold")
    ]
    generate = source[source.index("    def generate_sequences") : source.index("    def compute_log_prob")]
    assert "if self._is_offload_optimizer:" in init
    assert "if self._is_offload_optimizer and not preserve_hdo:" in normal_update
    assert "if self._is_offload_optimizer and not preserve_hdo:" in generate
    load = source[source.index("    def load_checkpoint") : source.index("    def load_pretrained_model")]
    assert "self._is_offload_optimizer" in load
    assert "optimizer_offload=true" in load


def _worker_method(name):
    tree = ast.parse(WORKER.read_text(encoding="utf-8"), filename=str(WORKER))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "ActorRolloutRefWorker")
    method = next(node for node in cls.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name)
    method.decorator_list = []
    return method


def test_next_actor_update_does_not_double_load_preserved_optimizer():
    events = []
    namespace = {"load_megatron_optimizer": lambda _: events.append("load_optimizer")}
    exec(compile(ast.Module(body=[_worker_method("_load_actor_optimizer_for_update")], type_ignores=[]), str(WORKER), "exec"), namespace)
    worker = SimpleNamespace(
        _is_offload_optimizer=True,
        _hdo_optimizer_residency_preserved=True,
        actor_optimizer=object(),
    )
    load = MethodType(namespace["_load_actor_optimizer_for_update"], worker)
    assert load() is False
    assert events == []
    worker._hdo_optimizer_residency_preserved = False
    assert load() is True
    assert events == ["load_optimizer"]


def test_generate_sequences_preserve_flag_never_full_offloads_optimizer():
    events = []

    @contextlib.contextmanager
    def simple_timer(_, timing):
        yield
        timing["generate_sequences"] = 1.25

    class Output:
        def __init__(self):
            self.meta_info = {}

        def to(self, _):
            return self

    namespace = {
        "get_device_name": lambda: "cpu",
        "log_eu_derpo_memory": lambda stage, *_, **__: events.append(stage),
        "megatron_model_cpu_data_bytes": lambda _: 0,
        "offload_megatron_optimizer": lambda _: events.append("offload_optimizer"),
        "simple_timer": simple_timer,
        "topk_reduce_ratio_min_max": lambda _: (1.0, 1.0, 1.0),
        "reduce_timing": lambda timing: timing,
        "aggressive_empty_cache": lambda **_: None,
        "log_gpu_memory_usage": lambda *_, **__: None,
        "get_event_loop": lambda: None,
        "logger": object(),
    }
    exec(
        compile(
            ast.Module(
                body=[_worker_method("_offload_actor_optimizer"), _worker_method("generate_sequences")],
                type_ignores=[],
            ),
            str(WORKER),
            "exec",
        ),
        namespace,
    )
    prompts = SimpleNamespace(to=lambda _: prompts, meta_info={})
    worker = SimpleNamespace(
        _is_rollout=True,
        _is_actor=False,
        _is_offload_optimizer=True,
        actor_optimizer=object(),
        actor_module=object(),
        generation_config=None,
        tokenizer=SimpleNamespace(eos_token_id=1, pad_token_id=0),
        config=SimpleNamespace(actor=SimpleNamespace(eu_derpo=SimpleNamespace(enabled=True))),
        rollout=SimpleNamespace(generate_sequences=lambda prompts: Output()),
        _preserve_hdo_optimizer_residency=lambda: True,
    )
    worker._hdo_optimizer_residency_preserved = True
    worker._offload_actor_optimizer = MethodType(namespace["_offload_actor_optimizer"], worker)
    output = MethodType(namespace["generate_sequences"], worker)(prompts)
    assert "offload_optimizer" not in events
    assert events == ["before_generate_sequences", "after_generate_sequences"]
    assert output.meta_info["timing"]["generate_sequences"] == 1.25
    events.clear()
    worker._preserve_hdo_optimizer_residency = lambda: False
    MethodType(namespace["generate_sequences"], worker)(prompts)
    assert events == ["before_generate_sequences", "offload_optimizer", "after_generate_sequences"]
    assert worker._hdo_optimizer_residency_preserved is False


def test_preserve_validation_rejects_missing_partial_hdo_groups():
    namespace = {"estimate_hdo_memory": lambda _: {"hdo_cpu_param_numel": 1, "hdo_gpu_param_numel": 0}}
    exec(compile(ast.Module(body=[_worker_method("_validate_hdo_optimizer_residency_config")], type_ignores=[]), str(WORKER), "exec"), namespace)
    worker = SimpleNamespace(
        _is_actor=True,
        _is_offload_optimizer=True,
        actor_optimizer=object(),
        config=SimpleNamespace(
            actor=SimpleNamespace(
                optim=SimpleNamespace(
                    override_optimizer_config={"optimizer_cpu_offload": True, "optimizer_offload_fraction": 0.75}
                )
            )
        ),
        _preserve_hdo_optimizer_residency=lambda: True,
    )
    with pytest.raises(RuntimeError, match="EU_DERPO_PRESERVE_HDO_REQUIRES_PARTIAL_HDO"):
        MethodType(namespace["_validate_hdo_optimizer_residency_config"], worker)()


def test_preserved_hdo_low_gpu_headroom_warns_without_aborting_rollout_sync():
    class Device:
        def mem_get_info(self):
            return 8 * 1024**3, 80 * 1024**3

        def memory_reserved(self):
            return 48 * 1024**3

        def memory_allocated(self):
            return 47 * 1024**3

    warnings = []
    namespace = {
        "get_torch_device": Device,
        "_GIB": 1024**3,
        "_CHECKPOINT_MIN_CUDA_FREE_BYTES": 16 * 1024**3,
        "_CHECKPOINT_MAX_CUDA_RESERVED_BYTES": 64 * 1024**3,
        "logger": SimpleNamespace(warning=warnings.append),
        "torch": SimpleNamespace(distributed=SimpleNamespace(get_rank=lambda: 0)),
    }
    exec(
        compile(
            ast.Module(body=[_worker_method("_warn_eu_derpo_hdo_gpu_headroom")], type_ignores=[]),
            str(WORKER),
            "exec",
        ),
        namespace,
    )
    worker = SimpleNamespace(_hdo_optimizer_residency_preserved=True)
    warn = MethodType(namespace["_warn_eu_derpo_hdo_gpu_headroom"], worker)
    warn("before_rollout_update_weights")
    assert warnings == [
        "EU-DERPO GPU headroom low: stage=before_rollout_update_weights rank=0 "
        "allocated_gib=47.00 reserved_gib=48.00 free_gib=8.00 total_gib=80.00"
    ]


def test_pre_wakeup_cuda_cache_release_is_opt_in_and_preserves_hdo_state():
    events = []
    extras = {}
    state = {"released": False, "fail": False}

    class Device:
        def synchronize(self):
            events.append("synchronize")

        def empty_cache(self):
            events.append("empty_cache")
            if state["fail"]:
                raise RuntimeError("empty cache failed")
            state["released"] = True

        def mem_get_info(self):
            free = 11 if state["released"] else 8
            return free * 1024**3, 80 * 1024**3

        def memory_allocated(self):
            return 22 * 1024**3

        def memory_reserved(self):
            reserved = 22.5 if state["released"] else 25
            return reserved * 1024**3

    device = Device()

    def log_memory(stage, *_, extra=None):
        events.append(stage)
        if extra is not None:
            assert all(not isinstance(value, (dict, list)) for value in extra.values())
            extras[stage] = extra
        free, total = device.mem_get_info()
        return {
            "cuda_allocated_gib": device.memory_allocated() / 1024**3,
            "cuda_reserved_gib": device.memory_reserved() / 1024**3,
            "cuda_free_gib": free / 1024**3,
            "cuda_total_gib": total / 1024**3,
        }

    namespace = {
        "_GIB": 1024**3,
        "get_torch_device": lambda: device,
        "log_eu_derpo_memory": log_memory,
        "megatron_model_cpu_data_bytes": lambda _: 0,
        "time": time,
    }
    exec(
        compile(
            ast.Module(
                body=[_worker_method("_release_actor_cuda_cache_before_rollout_wakeup")],
                type_ignores=[],
            ),
            str(WORKER),
            "exec",
        ),
        namespace,
    )
    worker = SimpleNamespace(
        actor_optimizer=object(),
        actor_module=object(),
        _hdo_optimizer_residency_preserved=True,
        config=SimpleNamespace(
            rollout=SimpleNamespace(free_cache_engine=True),
            actor=SimpleNamespace(
                eu_derpo=SimpleNamespace(
                    enabled=True,
                    release_actor_cuda_cache_before_rollout_wakeup=False,
                )
            ),
        ),
    )
    release = MethodType(namespace["_release_actor_cuda_cache_before_rollout_wakeup"], worker)
    assert release() is False
    assert events == []

    worker.config.actor.eu_derpo.release_actor_cuda_cache_before_rollout_wakeup = True
    assert release() is True
    assert events == [
        "before_rollout_cuda_cache_release",
        "synchronize",
        "empty_cache",
        "after_rollout_cuda_cache_release",
    ]
    after = extras["after_rollout_cuda_cache_release"]
    assert after["reserved_minus_allocated_before_gib"] == 3
    assert after["reserved_minus_allocated_after_gib"] == 0.5
    assert after["cuda_free_gain_gib"] == 3
    assert after["cache_release_elapsed_s"] >= 0
    assert worker._hdo_optimizer_residency_preserved is True
    method_source = ast.get_source_segment(
        WORKER.read_text(encoding="utf-8"),
        _worker_method("_release_actor_cuda_cache_before_rollout_wakeup"),
    )
    assert "offload_megatron_optimizer" not in method_source

    events.clear()
    state.update(released=False, fail=True)
    with pytest.raises(RuntimeError, match="empty cache failed"):
        release()


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
    assert "stage_callback=stage_callback" in save
