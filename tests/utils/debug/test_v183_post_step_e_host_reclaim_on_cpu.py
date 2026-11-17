"""Isolated CPU probe for allocator pages retained after consuming real hybrid hidden blocks."""
import gc
import importlib.util
import json
import platform
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from test_v183_hidden_store_on_cpu import ROOT, local_transport, store


MIB = 1 << 20
ROWS, WIDTH = 1024, 2048  # 4 MiB per BF16 block


def load_reclaimer():
    spec = importlib.util.spec_from_file_location("probe_memory_utils", ROOT / "verl/utils/memory_utils.py")
    memory_utils = importlib.util.module_from_spec(spec)
    # Only the optional training device import is stubbed. Procfs, libc, GC and tensors are real.
    device = SimpleNamespace(get_torch_device=lambda: torch.device("cpu"), is_cuda_available=False)
    with mock.patch.dict(sys.modules, {"verl.utils.device": device}):
        spec.loader.exec_module(memory_utils)
    return memory_utils.post_step_e_host_reclaim


def rss():
    if sys.platform.startswith("linux"):
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
        raise AssertionError("VmRSS missing from /proc/self/status")
    return None


def exercise(root, allocator, blocks, *, physical):
    torch.set_num_threads(1)
    reclaim = load_reclaimer()
    source = torch.ones(ROWS, WIDTH, dtype=torch.bfloat16)
    replay = torch.zeros_like(source)
    mask = torch.ones(ROWS, dtype=torch.bool)
    block_bytes = source.untyped_storage().nbytes()
    assert block_bytes == store.WRITE_BYTES == 4 * MIB
    payload = blocks * block_bytes
    weight = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.Adam([weight], lr=0.01)
    # Initialise autograd runtime without materialising Adam state before the first real gradient.
    (weight * 0).backward()
    optimizer.zero_grad(set_to_none=True)
    if allocator == "preheated":
        # A real allocation/free above the hidden block size exercises glibc's adaptive mmap threshold.
        warm = torch.empty(16 * MIB, dtype=torch.uint8).fill_(1)
        del warm
    observations = []
    for generation in range(3):
        abort_partial = generation == 2
        baseline = rss()
        guards = []
        expected_grad = 0
        with local_transport():
            hidden = store.HybridHiddenStore(root, "logical-to-physical", 0, generation, blocks * ROWS, WIDTH,
                                             1, payload, ram_limit_bytes=payload, pressure=lambda: 2 << 30)
            try:
                for block in range(blocks):
                    source.fill_(block + 1)
                    hidden.write_rows(0, block * ROWS, (block + 1) * ROWS, source, mask)
                    guards.append(torch.full((32 * 1024,), block + 1, dtype=torch.uint8))
                hidden.seal_capture()
                assert hidden.ram_storage_snapshot() == dict(logical_ram_bytes=payload, live_storage_bytes=payload,
                                                             live_storage_blocks=blocks, resident_entries=blocks,
                                                             span_entries=blocks)
                sealed_rss = rss()
                for block in range(blocks // 2 if abort_partial else blocks):
                    hidden.read_rows_into(0, block * ROWS, (block + 1) * ROWS, replay)
                    assert torch.equal(replay, source.fill_(block + 1))
                    (weight * replay[0, 0].float()).backward()
                    expected_grad += block + 1
                if abort_partial:
                    assert hidden.ram_storage_snapshot()["live_storage_bytes"] == (blocks - blocks // 2) * block_bytes
                    raise InterruptedError("injected after partial replay")
                # Verify consumption itself relinquished ownership, before close/abort can hide a leak.
                assert not any(hidden.ram_storage_snapshot().values())
                hidden.release_layer(0)
                hidden.close()
            except InterruptedError as exc:
                if not abort_partial or str(exc) != "injected after partial replay":
                    raise
            finally:
                hidden.abort()
            assert not hidden.fds and not hidden.owned and hidden.stage is None
            assert hidden.spill_written_bytes == hidden.spill_read_bytes == 0

        ledger = hidden.ram_storage_snapshot()
        assert not any(ledger.values())
        assert weight.grad.item() == expected_grad
        grad_ptr = weight.grad.data_ptr()
        state_before = {k: (v.data_ptr(), v.clone()) for k, v in optimizer.state.get(weight, {}).items()}
        assert bool(state_before) == (generation > 0)
        gc.collect()  # Control: old boundary, no explicit libc trim.
        report = reclaim(generation=generation, hidden_storage=ledger)
        assert weight.grad.data_ptr() == grad_ptr and weight.grad.item() == expected_grad
        for key, (ptr, value) in state_before.items():
            assert optimizer.state[weight][key].data_ptr() == ptr and torch.equal(optimizer.state[weight][key], value)
        for block, guard in enumerate(guards):
            assert guard.min().item() == guard.max().item() == block + 1
        if not abort_partial:
            optimizer.step()
        assert int(optimizer.state[weight]["step"]) == min(generation + 1, 2)
        assert torch.isfinite(optimizer.state[weight]["exp_avg"]).all()
        optimizer.zero_grad(set_to_none=True)
        release_path = "not_measured"
        if physical:
            # This is a fresh single-threaded process. Allow 64 MiB for libraries, live guards,
            # allocator metadata and kernel accounting. Logical counters alone cannot satisfy this.
            assert report["trim_supported"], report
            before, after = report["before"], report["after"]
            for key in ("rss_bytes", "pss_bytes", "rss_anon_bytes"):
                assert before[key] > 0 and after[key] > 0, report
            assert sealed_rss >= baseline + payload * 3 // 4, (baseline, sealed_rss, report)
            assert after["rss_bytes"] <= baseline + 64 * MIB, (baseline, report)
            assert sealed_rss - after["rss_bytes"] >= payload // 2, (sealed_rss, report)
            retained = before["rss_bytes"] - baseline
            release_path = "retained_after_gc" if retained > 64 * MIB else "already_within_headroom_after_gc"
            if retained > 64 * MIB:
                assert report["rss_drop_bytes"] >= retained - 64 * MIB, report
        observations.append(dict(generation=generation, partial_abort=abort_partial, baseline_rss=baseline,
                                 sealed_rss=sealed_rss, release_path=release_path, hidden_storage=ledger, reclaim=report))
    if physical:
        final_rss = [row["reclaim"]["after"]["rss_bytes"] for row in observations]
        assert max(final_rss) - min(final_rss) <= 64 * MIB, final_rss
    return dict(allocator=allocator, block_bytes=block_bytes, dtype="bfloat16", payload_bytes=payload,
                shape=[1, blocks * ROWS, WIDTH], process=platform.platform(), observations=observations)


def test_post_step_e_keeps_live_data_and_adam_state(tmp_path):
    exercise(tmp_path / "hidden", "default", 2, physical=False)


@pytest.mark.parametrize("allocator", ["default", "preheated"])
@pytest.mark.skipif(sys.platform != "linux" or platform.libc_ver()[0] != "glibc",
                    reason="physical release needs Linux procfs and glibc malloc_trim")
def test_linux_physical_reclaim(tmp_path, allocator):
    result = subprocess.run([sys.executable, __file__, allocator, str(tmp_path)], check=True,
                            text=True, stdout=subprocess.PIPE)
    output = json.loads(result.stdout.splitlines()[-1])
    assert output["payload_bytes"] == 256 * MIB and len(output["observations"]) == 3
    print(json.dumps(output))


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] not in ("default", "preheated") or sys.platform != "linux":
        raise SystemExit("Usage on Linux: python test_v183_post_step_e_host_reclaim_on_cpu.py "
                         "default|preheated EXISTING_TEMP_DIR")
    print(json.dumps(exercise(Path(sys.argv[2]) / "hidden", sys.argv[1], 64, physical=True)))
