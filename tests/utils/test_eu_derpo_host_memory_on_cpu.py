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
