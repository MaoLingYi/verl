from __future__ import annotations

import ast
import gc
import time
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]
UTILS = ROOT / "verl" / "utils" / "megatron_utils.py"
WORKER = ROOT / "verl" / "workers" / "megatron_workers.py"


class ChainedOptimizer:
    pass


class FakeStorage:
    next_ptr = 1

    def __init__(self, nbytes):
        self._nbytes = nbytes
        self._ptr = FakeStorage.next_ptr
        FakeStorage.next_ptr += 1

    def data_ptr(self):
        return self._ptr

    def nbytes(self):
        return self._nbytes


class FakeTensor:
    def __init__(self, device, nbytes=16, *, pinned=False, fail_target=None):
        self.device = SimpleNamespace(type=device, index=0 if device == "cuda" else None)
        self._storage = FakeStorage(nbytes)
        self._pinned = pinned
        self._fail_target = fail_target

    @property
    def data(self):
        return self

    @data.setter
    def data(self, other):
        self.device = other.device
        self._storage = other._storage
        self._pinned = other._pinned

    def to(self, device, non_blocking=True):
        target = "cuda" if str(device).startswith("cuda") else str(device)
        if target == self._fail_target:
            raise RuntimeError(f"failed moving to {target}")
        return FakeTensor(
            target,
            self._storage.nbytes(),
            pinned=target == "cpu" and non_blocking,
            fail_target=self._fail_target,
        )

    def copy_(self, other, non_blocking=False):
        assert not non_blocking
        if other._fail_target == "cpu":
            raise RuntimeError("failed copying to cpu")
        assert self._storage.nbytes() == other._storage.nbytes()
        return self

    def untyped_storage(self):
        return self._storage

    def is_pinned(self):
        return self._pinned


def _selective_helpers():
    tree = ast.parse(UTILS.read_text(encoding="utf-8"), filename=str(UTILS))
    names = {
        "_iter_megatron_copy_params",
        "_copy_param_transition_error",
        "offload_megatron_optimizer_copy_params_to_cpu",
        "load_megatron_optimizer_copy_params_to_gpu",
    }
    definitions = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "MegatronOptimizerCopyParamTransitionError":
            definitions.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in names:
            node.decorator_list = []
            definitions.append(node)
    fake_torch = SimpleNamespace(
        empty_like=lambda tensor, *, device, pin_memory: FakeTensor(
            device, tensor.untyped_storage().nbytes(), pinned=pin_memory
        )
    )
    namespace = {
        "ChainedOptimizer": ChainedOptimizer,
        "get_device_id": lambda: "cuda:0",
        "torch": fake_torch,
    }
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(UTILS), "exec"), namespace)
    return namespace


def _fake_hdo():
    cpu_selected_outer = FakeTensor("cuda", 24)
    gpu_selected_outer = FakeTensor("cuda", 40)
    cpu_inner = FakeTensor("cpu", 24, pinned=True)
    exp_avg = FakeTensor("cuda", 40)
    exp_avg_sq = FakeTensor("cuda", 40)
    hdo = SimpleNamespace(
        param_groups=[{"params": [cpu_selected_outer, gpu_selected_outer]}],
        param_to_inner_param={cpu_selected_outer: cpu_inner, gpu_selected_outer: gpu_selected_outer},
        inner_param_to_orig_param={cpu_inner: cpu_selected_outer, gpu_selected_outer: gpu_selected_outer},
        gpu_params_map_cpu_copy={cpu_selected_outer: cpu_inner},
        cpu_copys_map_gpu_param={cpu_inner: cpu_selected_outer},
        state={
            cpu_selected_outer: {"master_param": cpu_inner},
            gpu_selected_outer: {
                "master_param": gpu_selected_outer,
                "exp_avg": exp_avg,
                "exp_avg_sq": exp_avg_sq,
            },
        },
    )
    outer = SimpleNamespace(
        shard_fp32_from_float16_groups=[[cpu_selected_outer, gpu_selected_outer]],
        optimizer=hdo,
    )
    return outer, hdo, (cpu_selected_outer, gpu_selected_outer), (exp_avg, exp_avg_sq), cpu_inner


def test_selective_copy_param_round_trips_preserve_hdo_identity_and_moments():
    helpers = _selective_helpers()
    outer, hdo, copy_params, moments, cpu_inner = _fake_hdo()
    identities = {
        "copy": tuple(map(id, copy_params)),
        "param_groups": tuple(map(id, hdo.param_groups[0]["params"])),
        "param_to_inner": tuple((id(key), id(value)) for key, value in hdo.param_to_inner_param.items()),
        "inner_to_orig": tuple((id(key), id(value)) for key, value in hdo.inner_param_to_orig_param.items()),
        "gpu_to_cpu": tuple((id(key), id(value)) for key, value in hdo.gpu_params_map_cpu_copy.items()),
        "cpu_to_gpu": tuple((id(key), id(value)) for key, value in hdo.cpu_copys_map_gpu_param.items()),
    }
    optimizer_steps = 0

    for _ in range(3):
        moved = helpers["offload_megatron_optimizer_copy_params_to_cpu"](outer)
        assert moved == {"moved_tensors": 2, "moved_bytes": 64}
        assert [param.device.type for param in copy_params] == ["cpu", "cpu"]
        assert not any(param.is_pinned() for param in copy_params)
        assert cpu_inner.device.type == "cpu" and cpu_inner.is_pinned()
        assert [state.device.type for state in moments] == ["cuda", "cuda"]
        assert hdo.state[copy_params[1]]["master_param"] is copy_params[1]
        assert helpers["offload_megatron_optimizer_copy_params_to_cpu"](outer) == {
            "moved_tensors": 0,
            "moved_bytes": 0,
        }

        restored = helpers["load_megatron_optimizer_copy_params_to_gpu"](outer)
        assert restored == {"moved_tensors": 2, "moved_bytes": 64}
        assert [param.device.type for param in copy_params] == ["cuda", "cuda"]
        assert [state.device.type for state in moments] == ["cuda", "cuda"]
        assert helpers["load_megatron_optimizer_copy_params_to_gpu"](outer) == {
            "moved_tensors": 0,
            "moved_bytes": 0,
        }
        assert all(param.device.type == "cuda" for param in hdo.param_groups[0]["params"])
        assert all(state.device.type == "cuda" for state in moments)
        optimizer_steps += 1  # fake optimizer.step precondition is now satisfied

    assert tuple(map(id, copy_params)) == identities["copy"]
    assert tuple(map(id, hdo.param_groups[0]["params"])) == identities["param_groups"]
    assert tuple((id(k), id(v)) for k, v in hdo.param_to_inner_param.items()) == identities["param_to_inner"]
    assert tuple((id(k), id(v)) for k, v in hdo.inner_param_to_orig_param.items()) == identities["inner_to_orig"]
    assert tuple((id(k), id(v)) for k, v in hdo.gpu_params_map_cpu_copy.items()) == identities["gpu_to_cpu"]
    assert tuple((id(k), id(v)) for k, v in hdo.cpu_copys_map_gpu_param.items()) == identities["cpu_to_gpu"]
    assert optimizer_steps == 3


def test_selective_transition_failure_reports_partial_progress_and_stops():
    helpers = _selective_helpers()
    first = FakeTensor("cuda", 8)
    second = FakeTensor("cuda", 16, fail_target="cpu")
    outer = SimpleNamespace(shard_fp32_from_float16_groups=[[first, second]])
    with pytest.raises(RuntimeError, match=r"offload failed after moved_tensors=1 moved_bytes=8"):
        helpers["offload_megatron_optimizer_copy_params_to_cpu"](outer)
    assert first.device.type == "cpu"
    assert second.device.type == "cuda"


def test_copy_param_host_preflight_uses_node_sum_plus_existing_margin():
    tree = ast.parse(WORKER.read_text(encoding="utf-8"), filename=str(WORKER))
    function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_optimizer_copy_param_host_decision"
    )
    namespace = {"HOST_RAM_SAFETY_MARGIN_BYTES": 32}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(WORKER), "exec"), namespace)
    decide = namespace["_optimizer_copy_param_host_decision"]
    assert decide([10, 20], 62)["passed"]
    assert not decide([10, 20], 61)["passed"]


def test_host_preflight_and_collective_failure_precede_rollout_transition():
    source = WORKER.read_text(encoding="utf-8")
    start = source.index("    def _offload_actor_optimizer_copy_params_for_rollout")
    end = source.index("    def _restore_actor_optimizer_copy_params_for_update", start)
    offload = source[start:end]
    assert offload.index("self._optimizer_copy_param_host_preflight") < offload.index(
        "offload_megatron_optimizer_copy_params_to_cpu"
    )
    assert offload.index('self._complete_optimizer_copy_param_transition("offload"') < offload.index(
        "self._optimizer_copy_params_offloaded_for_rollout = True"
    )

    start = source.index("    def _complete_optimizer_copy_param_transition")
    end = source.index("    def _offload_actor_optimizer_copy_params_for_rollout", start)
    completion = source[start:end]
    assert "torch.distributed.all_reduce" in completion
    assert "EU_DERPO_COPY_PARAM_RESIDENCY_TRANSITION_FAILED" in completion


def test_selective_transition_telemetry_stays_out_of_actor_scalar_metrics():
    source = WORKER.read_text(encoding="utf-8")
    for stage in (
        "before_optimizer_copy_param_offload",
        "after_optimizer_copy_param_offload",
        "before_optimizer_copy_param_restore",
        "after_optimizer_copy_param_restore",
    ):
        assert f'"{stage}"' in source
    transition = source[
        source.index("    def _offload_actor_optimizer_copy_params_for_rollout") :
        source.index("    def _load_actor_optimizer_for_update")
    ]
    assert "copy_param_offload_active" in transition
    assert "copy_param_transition_elapsed_s" in transition
    assert "metrics[" not in transition


def test_generic_helpers_keep_276d331_behavior_and_never_call_selective_helpers():
    tree = ast.parse(UTILS.read_text(encoding="utf-8"), filename=str(UTILS))
    names = {
        "offload_megatron_copy_params",
        "load_megatron_copy_params",
        "offload_megatron_optimizer",
        "load_megatron_optimizer",
    }
    definitions = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            node.decorator_list = []
            definitions.append(node)

    events = []
    device = SimpleNamespace(empty_cache=lambda: events.append("empty_cache"))
    namespace = {
        "ChainedOptimizer": ChainedOptimizer,
        "gc": gc,
        "get_device_id": lambda: "cuda:0",
        "get_global_memory_buffer": lambda: SimpleNamespace(buffer={}),
        "get_torch_device": lambda: device,
        "torch": SimpleNamespace(Tensor=FakeTensor),
        "offload_megatron_optimizer_copy_params_to_cpu": lambda *_: (_ for _ in ()).throw(
            AssertionError("selective helper must not be called")
        ),
        "load_megatron_optimizer_copy_params_to_gpu": lambda *_: (_ for _ in ()).throw(
            AssertionError("selective helper must not be called")
        ),
    }
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(UTILS), "exec"), namespace)

    param = FakeTensor("cuda", 32)
    outer = SimpleNamespace(shard_fp32_from_float16_groups=[[param]], optimizer=None)
    namespace["offload_megatron_optimizer"](outer)
    assert param.device.type == "cpu"
    assert param.is_pinned()  # exact generic non_blocking behavior is intentionally unchanged
    namespace["load_megatron_optimizer"](outer)
    assert param.device.type == "cuda"
    assert events == ["empty_cache", "empty_cache"]


def test_selective_source_explicitly_allocates_pageable_cpu_storage():
    source = UTILS.read_text(encoding="utf-8")
    start = source.index("def offload_megatron_optimizer_copy_params_to_cpu")
    end = source.index("def load_megatron_optimizer_copy_params_to_gpu", start)
    selective = source[start:end]
    assert 'torch.empty_like(tensor.data, device="cpu", pin_memory=False)' in selective
    assert "cpu_data.copy_(tensor.data, non_blocking=False)" in selective
    assert '.to("cpu", non_blocking=True)' not in selective

    start = source.index("def offload_megatron_copy_params")
    end = source.index("def load_megatron_copy_params", start)
    generic = source[start:end]
    assert 'tensor.data = tensor.data.to("cpu", non_blocking=True)' in generic
    assert "offload_megatron_optimizer_copy_params_to_cpu" not in generic


def _host_reclaim_helper(*, trim_result=1, load_error=None):
    tree = ast.parse(WORKER.read_text(encoding="utf-8"), filename=str(WORKER))
    function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_best_effort_release_host_allocator_after_copy_param_restore"
    )

    class Trim:
        argtypes = None
        restype = None

        def __call__(self, value):
            assert value == 0
            return trim_result

    warnings = []

    def load_libc(_):
        if load_error is not None:
            raise load_error
        return SimpleNamespace(malloc_trim=Trim())

    namespace = {
        "ctypes": SimpleNamespace(CDLL=load_libc, c_size_t=object(), c_int=object()),
        "gc": SimpleNamespace(collect=lambda: 7),
        "logger": SimpleNamespace(warning=lambda *args: warnings.append(args)),
        "sys": SimpleNamespace(platform="linux"),
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(WORKER), "exec"), namespace)
    return namespace[function.name](), warnings


def test_host_reclaim_is_best_effort_when_malloc_trim_returns_zero_or_is_unavailable():
    result, warnings = _host_reclaim_helper(trim_result=0)
    assert result == {"gc_collected": 7, "malloc_trim_attempted": 1, "malloc_trim_rc": 0}
    assert not warnings

    result, warnings = _host_reclaim_helper(load_error=OSError("no libc"))
    assert result == {"gc_collected": 7, "malloc_trim_attempted": 1, "malloc_trim_rc": None}
    assert len(warnings) == 1


def test_selective_restore_reclaims_host_only_after_cuda_restore_and_sanity():
    tree = ast.parse(WORKER.read_text(encoding="utf-8"), filename=str(WORKER))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "ActorRolloutRefWorker")
    method = next(
        node for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "_restore_actor_optimizer_copy_params_for_update"
    )
    events = []
    state = {"copy_cuda": False}

    def estimate(_):
        return {
            "optimizer_phase_cuda_copy_param_bytes": 64 if state["copy_cuda"] else 0,
            "optimizer_phase_cuda_state_bytes": 80,
        }

    def restore(_):
        events.append("restore_cuda")
        state["copy_cuda"] = True
        return {"moved_tensors": 2, "moved_bytes": 64}

    def reclaim():
        assert state["copy_cuda"]
        assert not worker._optimizer_copy_params_offloaded_for_rollout
        events.append("host_reclaim")
        return {"gc_collected": 1, "malloc_trim_attempted": 1, "malloc_trim_rc": 1}

    namespace = {
        "_GIB": 1024**3,
        "_best_effort_release_host_allocator_after_copy_param_restore": reclaim,
        "estimate_optimizer_phase_offload_memory": estimate,
        "get_device_name": lambda: "cuda:0",
        "get_torch_device": lambda: SimpleNamespace(synchronize=lambda: events.append("cuda_sync")),
        "load_megatron_optimizer_copy_params_to_gpu": restore,
        "log_eu_derpo_memory": lambda stage, *args, **kwargs: events.append(stage),
        "megatron_model_cpu_data_bytes": lambda _: 0,
        "psutil": SimpleNamespace(
            Process=lambda: SimpleNamespace(memory_info=lambda: SimpleNamespace(rss=8 * 1024**3)),
            virtual_memory=lambda: SimpleNamespace(available=128 * 1024**3),
        ),
        "time": time,
        "torch": SimpleNamespace(
            distributed=SimpleNamespace(is_initialized=lambda: False),
        ),
    }
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(WORKER), "exec"), namespace)
    worker = SimpleNamespace(
        actor_module=object(),
        actor_optimizer=object(),
        _optimizer_copy_params_offloaded_for_rollout=True,
        _optimizer_copy_params_cuda_bytes_before_rollout_offload=64,
        _complete_optimizer_copy_param_transition=lambda stage, failure, error, moved: (
            (_ for _ in ()).throw(error) if error else events.append(stage)
        ),
    )
    worker._actor_optimizer_residency = lambda: (
        "ROLLOUT_PARTIAL" if worker._optimizer_copy_params_offloaded_for_rollout else "TRAINING_RESIDENT"
    )
    restore_method = MethodType(namespace[method.name], worker)

    assert restore_method()
    assert events.count("host_reclaim") == 1
    assert events.index("restore_cuda") < events.index("cuda_sync") < events.index("restore_sanity")
    assert events.index("restore_sanity") < events.index("before_optimizer_copy_param_host_reclaim")
    assert events.index("before_optimizer_copy_param_host_reclaim") < events.index("host_reclaim")
    assert events.index("host_reclaim") < events.index("after_optimizer_copy_param_host_reclaim")


def test_host_reclaim_is_selective_only_and_does_not_allocate_cpu_tensors():
    worker_source = WORKER.read_text(encoding="utf-8")
    start = worker_source.index("def _best_effort_release_host_allocator_after_copy_param_restore")
    end = worker_source.index("\n\ndef ", start + 4)
    helper = worker_source[start:end]
    restore_start = worker_source.index("    def _restore_actor_optimizer_copy_params_for_update")
    restore_end = worker_source.index("    def _load_actor_optimizer_for_update", restore_start)
    restore = worker_source[restore_start:restore_end]
    combined = helper + restore
    assert 'empty_like' not in combined
    assert '.cpu(' not in combined
    assert '.to("cpu"' not in combined

    generic_source = UTILS.read_text(encoding="utf-8")
    assert "_best_effort_release_host_allocator_after_copy_param_restore" not in generic_source
    assert "_best_effort_release_host_allocator_after_copy_param_restore" not in worker_source[
        worker_source.index("    def load_checkpoint") : restore_start
    ]
