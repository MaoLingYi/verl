from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

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
        return FakeTensor(target, self._storage.nbytes(), pinned=False, fail_target=self._fail_target)

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
    namespace = {"ChainedOptimizer": ChainedOptimizer, "get_device_id": lambda: "cuda:0"}
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
