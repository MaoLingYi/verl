import importlib.util
import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace as N

import pytest
import torch


ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location("r3_resume", ROOT / "verl/utils/checkpoint/r3_resume.py")
resume = importlib.util.module_from_spec(spec)
spec.loader.exec_module(resume)


def test_real_dcp_byte_io_authority_and_checkpoint_unchanged(tmp_path):
    from torch.distributed.checkpoint import FileSystemWriter, save
    root = tmp_path / "global_step_250"
    dcp = root / "actor/dist_ckpt"
    dcp.mkdir(parents=True)
    payload = {"optimizer": {"0": {"optimizer": {"param_groups": [{"step": 250, "params": [0]}]}}},
               "model": torch.ones(2), "rng_state": {"cpu": torch.get_rng_state()}}
    # Disable the writer's automatic CUDA copy-ahead stream even for CPU-only fixtures.
    save(payload, storage_writer=FileSystemWriter(str(dcp), per_thread_copy_ahead=0))
    torch.save({"sampler": "fixture only"}, root / "data.pt")
    before = {p: (p.stat().st_mtime_ns, p.read_bytes()) for p in root.rglob("*") if p.is_file()}
    result = resume.inspect_checkpoint(root)
    assert result["HDO_AUTHORITATIVE_OPTIMIZER_STEP"] == 250
    assert result["optimizer_entries"]
    assert result["MODEL_OPTIMIZER_RESTORE"] == "NOT_RUN"
    assert before == {p: (p.stat().st_mtime_ns, p.read_bytes()) for p in root.rglob("*") if p.is_file()}
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("payload", [
    {"lr_scheduler": {"num_steps": 250}, "global_step": 250},
    {"optimizer": {"param_groups": [{"params": [0]}]}, "global_step": 250},
    {"optimizer": {"param_groups": [{"step": 249}]}},
    {"optimizer": {"param_groups": [{"step": 250}, {"step": 251}]}},
])
def test_authority_cannot_be_synthesized_from_scheduler_or_global_step(payload):
    with pytest.raises(ValueError):
        resume.require_steps(resume.optimizer_group_steps(payload), 250)


def test_restored_optimizer_must_match_checkpoint_and_two_updates():
    inner = N(param_groups=[{"step": 250}])
    outer = N(optimizer=inner)
    payload = {"optimizer": {"param_groups": [{"step": 250}]}}
    resume.check_restored_optimizer(outer, payload, 250)
    inner.param_groups[0]["step"] = 1
    with pytest.raises(ValueError):
        resume.check_restored_optimizer(outer, payload, 250)
    inner.param_groups[0]["step"] = 250
    completed = []
    for _ in range(2):
        inner.param_groups[0]["step"] += 1
        completed.append(inner.param_groups[0]["step"])
    assert completed == [251, 252]


def test_diagnostic_disabled_has_no_io_and_enabled_has_no_cuda(monkeypatch, tmp_path):
    path = ROOT / "verl/utils/r3_host_memory.py"
    module_spec = importlib.util.spec_from_file_location("r3_host", path)
    host = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(host)
    monkeypatch.delenv("R3_HOST_RAM_DIAG", raising=False)
    monkeypatch.setattr(host, "proc_values", lambda path: pytest.fail("disabled monitor read /proc"))
    assert host.log_host_phase("disabled") is None
    monkeypatch.setenv("R3_HOST_RAM_DIAG", "1")
    monkeypatch.setattr(host, "proc_values", lambda path: {"MemTotal": 10000, "MemAvailable": 4000,
                                                          "MemFree": 1000, "Cached": 2000})
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: pytest.fail("CUDA sync"))
    result = host.log_host_phase("rollout_start", 251)
    assert result["node_memory_used_MemAvailable_basis"] == 6000
    assert result["ray_95pct_headroom_estimate"] == 2500
    assert result["step"] == 251 and result["ray_object_store"] == "UNKNOWN"
    discovery = tmp_path / "prom_metrics_service_discovery.json"
    discovery.write_text(json.dumps([{"targets": ["127.0.0.1:1234"]}]))
    node = N(address_info={}, get_session_dir_path=lambda: str(tmp_path / "session"))
    monkeypatch.setitem(sys.modules, "ray", N(_private=N(worker=N(global_worker=N(node=node)))))
    def metrics(url, timeout):
        assert url == "http://127.0.0.1:1234/metrics" and timeout == .2
        return io.BytesIO(b'ray_object_store_memory{Location="MMAP_SHM"} 123\nother 456\n')
    monkeypatch.setattr(host, "urlopen", metrics)
    assert host.log_host_phase("rollout_end", 251)["ray_object_store"] == [
        'ray_object_store_memory{Location="MMAP_SHM"} 123']
    records = []
    monkeypatch.setattr(host, "log_host_phase", lambda phase, step, *_: records.append((phase, step)))

    @host.host_phase("checkpoint")
    def save(owner, path, hdfs, step):
        raise ValueError("original checkpoint failure")

    with pytest.raises(ValueError, match="original checkpoint failure"):
        save(N(), "/new/checkpoint", None, 300)
    assert records == [("before_checkpoint", 300), ("after_checkpoint", 300)]
