from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import torch


ROOT = Path(__file__).parents[2]


def _load_memory_utils():
    device = types.ModuleType("verl.utils.device")
    device.get_torch_device = lambda: torch.cuda
    device.is_cuda_available = False
    modules = {
        "verl": types.ModuleType("verl"),
        "verl.utils": types.ModuleType("verl.utils"),
        "verl.utils.device": device,
    }
    saved = {name: sys.modules.get(name) for name in modules}
    sys.modules.update(modules)
    try:
        spec = importlib.util.spec_from_file_location("eu_derpo_memory_utils", ROOT / "verl/utils/memory_utils.py")
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        for name, previous in saved.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


memory_utils = _load_memory_utils()
_parse_proc_kib = memory_utils._parse_proc_kib
estimate_hdo_memory = memory_utils.estimate_hdo_memory
estimate_optimizer_phase_offload_memory = memory_utils.estimate_optimizer_phase_offload_memory


def test_parse_proc_kib_and_missing_fields():
    parsed = _parse_proc_kib("VmRSS:\t1024 kB\nRssAnon: 512 kB\nnot-a-value: x kB\n")
    assert parsed == {"VmRSS": 1024**2, "RssAnon": 512 * 1024}


def test_post_step_e_measures_page_release_independently_of_trim_return(monkeypatch):
    events = []
    snapshots = iter((
        {"rss_bytes": 500, "mem_available_bytes": 200},
        {"rss_bytes": 500, "mem_available_bytes": 180},
    ))

    def snapshot(_):
        events.append("snapshot")
        return next(snapshots)

    def reclaim():
        events.append("reclaim")
        return {"gc_collected": 3, "malloc_trim_rc": 1}

    monkeypatch.setattr(memory_utils, "_host_reclaim_snapshot", snapshot)
    monkeypatch.setattr(memory_utils, "best_effort_release_host_allocator", reclaim)
    result = memory_utils.post_step_e_host_reclaim(generation=2, hidden_storage={"storage_bytes": 0})
    assert events == ["snapshot", "reclaim", "snapshot"]
    assert result["trim_supported"] and result["trim_rc"] == 1 and result["gc_collected"] == 3
    assert result["rss_drop_bytes"] == 0 and result["available_gain_bytes"] == -20
    assert result["generation"] == 2 and result["hidden_storage"]["storage_bytes"] == 0


def test_reclaim_snapshot_keeps_ray_scopes_and_unknown_counters_distinct(monkeypatch):
    monkeypatch.setattr(memory_utils.sys, "platform", "linux")
    monkeypatch.setattr(
        memory_utils, "_read_proc_kib", lambda path: {"MemAvailable": 400} if path.endswith("meminfo") else {}
    )
    monkeypatch.setitem(sys.modules, "ray", SimpleNamespace(is_initialized=lambda: True))
    monkeypatch.setitem(
        sys.modules, "ray._private.utils", SimpleNamespace(get_system_memory=lambda: 700, get_used_memory=lambda: 310)
    )
    counters = memory_utils._host_reclaim_snapshot((950, 1000))
    assert counters["rss_bytes"] is counters["pss_bytes"] is counters["rss_anon_bytes"] is None
    assert (counters["ray_python_used_bytes"], counters["ray_python_total_bytes"]) == (310, 700)
    assert (counters["raylet_linux_used_bytes"], counters["raylet_threshold_bytes"]) == (600, 950)
    monkeypatch.setitem(sys.modules, "ray", None)
    counters = memory_utils._host_reclaim_snapshot(None)
    assert counters["ray_python_used_bytes"] is counters["raylet_linux_used_bytes"] is None


def test_reclaim_snapshot_reads_pinned_counters_only_after_cuda_initialization(monkeypatch):
    current = {"allocated_bytes.current": 32, "reserved_bytes.current": 128}
    calls = []

    def pinned_stats():
        calls.append(True)
        return current

    monkeypatch.setattr(torch.cuda.memory, "host_memory_stats", pinned_stats, raising=False)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    cold = memory_utils._host_reclaim_snapshot(None)
    assert not calls and cold["pinned_allocated_bytes"] is cold["pinned_reserved_bytes"] is None
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    warm = memory_utils._host_reclaim_snapshot(None)
    assert len(calls) == 1 and (warm["pinned_allocated_bytes"], warm["pinned_reserved_bytes"]) == (32, 128)
    current.clear()
    unknown = memory_utils._host_reclaim_snapshot(None)
    assert unknown["pinned_allocated_bytes"] is unknown["pinned_reserved_bytes"] is None


def test_missing_smaps_uses_rss_for_pss(monkeypatch):
    monkeypatch.setattr(
        memory_utils,
        "_read_proc_kib",
        lambda path: {"VmRSS": 2 * 1024**3} if path.endswith("status") else {},
    )
    monkeypatch.setattr(memory_utils, "_read_cgroup_memory", lambda: (None, None))
    snapshot = memory_utils.log_eu_derpo_memory("test")
    assert snapshot["rss_gib"] == 2.0
    assert snapshot["pss_gib"] == 2.0
    assert snapshot["cuda_free_gib"] >= 0.0
    assert snapshot["cuda_total_gib"] >= snapshot["cuda_free_gib"]


def test_phase_log_distinguishes_pinned_lifetime_peak_and_unknown(monkeypatch):
    gib = 1024**3
    monkeypatch.setattr(memory_utils, "_read_proc_kib", lambda _: {"VmHWM": 7 * gib})
    monkeypatch.setattr(memory_utils, "_read_cgroup_memory", lambda: (5 * gib, 8 * gib))
    monkeypatch.setattr(memory_utils, "get_torch_device", lambda: SimpleNamespace(is_available=lambda: False))
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    counters = {"allocated_bytes.current": gib, "allocated_bytes.peak": 3 * gib,
                "reserved_bytes.current": 2 * gib, "reserved_bytes.peak": 4 * gib}
    monkeypatch.setattr(torch.cuda.memory, "host_memory_stats", lambda: counters, raising=False)
    snapshot = memory_utils.log_eu_derpo_memory("after_full_restore_cleanup")
    assert snapshot["pinned_allocated_current_gib"] == 1
    assert snapshot["pinned_allocated_peak_gib"] == 3
    assert snapshot["pinned_reserved_current_gib"] == 2
    assert snapshot["pinned_reserved_peak_gib"] == 4
    assert snapshot["rss_process_peak_gib"] == 7 and snapshot["pid"] > 0
    assert (snapshot["cgroup_current_gib"], snapshot["cgroup_max_gib"]) == (5, 8)
    counters.clear()
    unknown = memory_utils.log_eu_derpo_memory("unavailable")
    assert all(unknown[f"pinned_{name}_{period}_gib"] is None
               for name in ("allocated", "reserved") for period in ("current", "peak"))


def test_memory_log_includes_caller_lifecycle_fields(monkeypatch):
    monkeypatch.setattr(memory_utils, "_read_proc_kib", lambda _: {})
    monkeypatch.setattr(memory_utils, "_read_cgroup_memory", lambda: (None, None))
    snapshot = memory_utils.log_eu_derpo_memory(
        "checkpoint_hold_enter",
        extra={"checkpoint_hold_active": 1, "defer_phase_offload_for_checkpoint": 1},
    )
    assert snapshot["checkpoint_hold_active"] == 1
    assert snapshot["defer_phase_offload_for_checkpoint"] == 1


def test_memory_instrumentation_has_no_tensor_copy_or_cuda_sync():
    source = Path(memory_utils.__file__).read_text(encoding="utf-8")
    start = source.index("def log_eu_derpo_memory(")
    end = source.index("\ndef aggressive_empty_cache", start)
    instrumentation = source[start:end]
    assert ".clone(" not in instrumentation
    assert ".to(" not in instrumentation
    assert ".cpu(" not in instrumentation
    assert ".synchronize(" not in instrumentation


def test_hdo_estimate_uses_real_selected_params_and_deduplicates_state_aliases():
    cpu_param = torch.zeros(7, dtype=torch.float32)
    exp_avg = torch.zeros_like(cpu_param)
    exp_avg_sq = torch.zeros_like(cpu_param)
    hdo = SimpleNamespace(
        param_to_inner_param={object(): cpu_param},
        gpu_params_map_cpu_copy={object(): cpu_param},
        cpu_copy_map_grad={},
        sub_optimizers=[SimpleNamespace(state={cpu_param: {"exp_avg": exp_avg, "exp_avg_sq": exp_avg_sq}})],
    )
    outer = SimpleNamespace(optimizer=hdo)

    estimate = estimate_hdo_memory(outer)

    assert estimate["hdo_cpu_master_bytes"] == 7 * 4
    assert estimate["hdo_cpu_exp_avg_bytes"] == 7 * 4
    assert estimate["hdo_cpu_exp_avg_sq_bytes"] == 7 * 4
    assert estimate["hdo_cpu_estimated_bytes"] == 7 * 16
    assert estimate["hdo_full_cpu_estimated_bytes"] == 7 * 16
    assert estimate["hdo_partial_cpu_savings_estimated_bytes"] == 0


def test_hdo_estimate_counts_shared_storage_once():
    cpu_param = torch.zeros(8, dtype=torch.float32)
    shared_state = cpu_param.view(2, 4)
    hdo = SimpleNamespace(
        param_to_inner_param={object(): cpu_param},
        gpu_params_map_cpu_copy={object(): cpu_param},
        cpu_copy_map_grad={},
        sub_optimizers=[SimpleNamespace(state={cpu_param: {"other_a": shared_state, "other_b": cpu_param}})],
    )

    estimate = estimate_hdo_memory(SimpleNamespace(optimizer=hdo))

    assert estimate["hdo_cpu_master_bytes"] == 8 * 4
    assert estimate["hdo_cpu_other_bytes"] == 0


def test_phase_offload_estimate_counts_exact_cuda_sources_without_copying():
    cpu = torch.zeros(3)
    outer = SimpleNamespace(
        shard_fp32_from_float16_groups=[[cpu]],
        optimizer=SimpleNamespace(sub_optimizers=[SimpleNamespace(state={cpu: {"exp_avg": cpu}})]),
    )
    estimate = estimate_optimizer_phase_offload_memory(outer)
    assert estimate == {
        "optimizer_phase_cuda_copy_param_bytes": 0,
        "optimizer_phase_cuda_state_bytes": 0,
        "optimizer_phase_cuda_total_bytes": 0,
    }
    source = Path(memory_utils.__file__).read_text(encoding="utf-8")
    start = source.index("def estimate_optimizer_phase_offload_memory(")
    end = source.index("\ndef log_eu_derpo_memory", start)
    instrumentation = source[start:end]
    assert ".to(" not in instrumentation
    assert ".cpu(" not in instrumentation
    assert ".clone(" not in instrumentation


def test_hdo_logical_fraction_survives_phase_move_to_cpu():
    cpu_selected = torch.zeros(3, dtype=torch.float32)
    gpu_selected_but_phase_offloaded = torch.zeros(1, dtype=torch.float32)
    hdo = SimpleNamespace(
        param_to_inner_param={object(): cpu_selected, object(): gpu_selected_but_phase_offloaded},
        gpu_params_map_cpu_copy={object(): cpu_selected},
        cpu_copy_map_grad={},
        sub_optimizers=[],
    )
    estimate = estimate_hdo_memory(SimpleNamespace(optimizer=hdo))
    assert estimate["hdo_realized_offload_fraction"] == 0.75
    assert estimate["hdo_partial_cpu_savings_estimated_bytes"] == 16
    assert estimate["hdo_gpu_incremental_state_estimated_bytes"] == 8
