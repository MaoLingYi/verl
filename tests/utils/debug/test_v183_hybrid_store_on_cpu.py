"""Use production storage and replay, with the existing local syscall fixture in place of NFS."""
import errno
import json
import os
import sys
import tempfile
from pathlib import Path
from unittest import mock

import pytest
import torch

from test_v183_hidden_store_on_cpu import local_transport, memory_sample, store


@pytest.mark.parametrize("quota", [0, 13, 4096])
def test_capture_and_replay_across_both_backends(tmp_path, monkeypatch, quota):
    monkeypatch.setattr(store, "WRITE_BYTES", 16)
    rows, width, layers = 9, 3, 2
    payload = rows * width * layers * 2
    source = torch.arange(72, dtype=torch.int16).reshape(12, 6)[:, ::2]
    source[0] = torch.tensor([-1, -32768, 32704], dtype=torch.int16)
    valid = torch.ones(12, dtype=torch.bool)
    valid[1::5] = False
    for generation in range(2):
        with local_transport(limit=7):
            s = store.HybridHiddenStore(tmp_path / "cache", "case", 2, generation, rows, width,
                                        layers, payload, ram_limit_bytes=quota, pressure=lambda: 5 << 30)
            try:
                for layer in range(layers):
                    s.write_rows(layer, 0, 9, source.view(torch.bfloat16), valid)
                s.seal_capture()
                in_ram = min(quota // 6 * 6, payload)
                assert s.ram_high_water_bytes == s.ram_hidden_bytes == in_ram
                assert sum(os.fstat(fd).st_size for fd in s.fds.values()) == payload - in_ram
                assert s.spill_written_bytes == payload - in_ram
                for layer in range(layers):
                    result = torch.empty((9, width), dtype=torch.bfloat16)
                    for first, end in ((0, 1), (1, 6), (6, 9)):
                        s.read_rows_into(layer, first, end, result[first:end])
                    assert torch.equal(result.view(torch.int16), source[valid])
                    s.release_layer(layer)
                assert s.ram_hidden_bytes == 0
                assert s.spill_read_bytes == s.spill_written_bytes and s.logical_hidden_bytes == payload
                assert s.d2h_copy_seconds == 0
                assert (s.nas_read_seconds > 0) == (s.spill_read_bytes > 0)
                assert (s.nas_write_seconds > 0) == (s.spill_written_bytes > 0)
                s.close()
            finally:
                s.abort()
            assert not list(tmp_path.rglob("*.bin"))


def test_hysteresis_and_bounded_eviction(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "WRITE_BYTES", 16)
    slack = [2 << 30]
    mask = torch.ones(2, dtype=torch.bool)
    source = torch.arange(32).reshape(8, 4).to(torch.bfloat16)
    with local_transport():
        s = store.HybridHiddenStore(tmp_path/"cache", "evict", 0, 0, 8, 4, 1, 64,
                                    ram_limit_bytes=64, pressure=lambda: slack[0])
        try:
            s.write_rows(0, 0, 2, source[:2], mask)
            s.write_rows(0, 2, 4, source[2:4], mask)
            slack[0] = -1
            s.poll_pressure()
            assert s.ram_hidden_bytes == 16 and s.spill_written_bytes == 16
            slack[0] = 1
            s.write_rows(0, 4, 6, source[4:6], mask)
            assert s.ram_hidden_bytes == 0 and s.spill_written_bytes == 48
            slack[0] = 1 << 30
            s.write_rows(0, 6, 8, source[6:], mask)
            s.seal_capture()
            result = torch.empty_like(source)
            s.read_rows_into(0, 0, 7, result[:7])
            assert s.ram_hidden_bytes == 16
            slack[0] = 0
            s.read_rows_into(0, 7, 8, result[7:])
            assert s.ram_hidden_bytes == 0 and s.spill_written_bytes == s.spill_read_bytes == 56
            assert s.header["written_bytes"] == 56 and torch.equal(result, source)
            s.release_layer(0)
            s.close()
        finally:
            s.abort()


@pytest.mark.parametrize("layers", [1, 10])
def test_payload_growth_with_constant_residency(tmp_path, monkeypatch, layers):
    monkeypatch.setattr(store, "WRITE_BYTES", 256)
    rows, width, quota = 40, 16, 1024
    mask = torch.ones(rows, dtype=torch.bool)
    source = torch.ones(rows, width, dtype=torch.bfloat16)
    with local_transport():
        s = store.HybridHiddenStore(tmp_path/"cache", "growth", 0, 0, rows, width, layers,
                                    rows*width*2*layers, ram_limit_bytes=quota, pressure=lambda: 2 << 30)
        try:
            for layer in range(layers):
                s.write_rows(layer, 0, rows, source, mask)
            s.seal_capture()
            allocations = [entry.payload.untyped_storage() for entry in s._resident.values()]
            assert len({a.data_ptr() for a in allocations}) == len(allocations)
            assert max(a.nbytes() for a in allocations) <= 256
            assert sum(a.nbytes() for a in allocations) == s.ram_high_water_bytes == quota
            assert s.spill_written_bytes + quota == s.logical_hidden_bytes
        finally:
            s.abort()
            assert not s.ram_hidden_bytes


def test_eviction_write_and_replay_errors_abort(tmp_path):
    slack = [2 << 30]
    with local_transport():
        s = store.HybridHiddenStore(tmp_path/"cache", "error", 0, 1, 2, 4, 1, 16,
                                    ram_limit_bytes=8, pressure=lambda: slack[0])
        source = torch.ones(2, 4, dtype=torch.bfloat16)
        try:
            s.write_rows(0, 0, 2, source, torch.ones(2, dtype=torch.bool))
            s.seal_capture()
            slack[0] = 0
            with mock.patch.object(os, "pwritev", side_effect=OSError(errno.ENOSPC, "injected")):
                with pytest.raises(OSError, match="injected"):
                    s.read_rows_into(0, 0, 2, source)
            assert s.ram_hidden_bytes == 8 and s.consumed == [0]
            with mock.patch.object(os, "preadv", side_effect=OSError(errno.EIO, "injected")):
                with pytest.raises(OSError, match="injected"):
                    s.read_rows_into(0, 0, 2, source)
            with pytest.raises(RuntimeError, match="unread"):
                s.release_layer(0)
        finally:
            s.abort()
            s.abort()
    assert not list(tmp_path.rglob("*.bin"))


def profile(root, layers):
    """Fresh-process RSS includes PyTorch, the source/staging tensors and the bounded RAM cache."""
    import psutil

    process = psutil.Process()
    rows, width, quota = 4096, 2048, 8 << 20
    torch.set_num_threads(1)
    source = torch.ones(1024, width, dtype=torch.bfloat16)
    mask = torch.ones(len(source), dtype=torch.bool)
    before = memory_sample()
    peak = before["rss"]
    with local_transport():
        s = store.HybridHiddenStore(root, "profile", 0, 0, rows, width, layers, rows*width*2*layers,
                                    ram_limit_bytes=quota, pressure=lambda: 2 << 30)
        try:
            for layer in range(layers):
                for row in range(0, rows, len(mask)):
                    s.write_rows(layer, row, row + len(mask), source, mask)
                    peak = max(peak, process.memory_info().rss)
            s.seal_capture()
            for layer in range(layers):
                for row in range(0, rows, len(mask)):
                    s.read_rows_into(layer, row, row + len(mask), source)
                    peak = max(peak, process.memory_info().rss)
                s.release_layer(layer)
            s.close()
        finally:
            s.abort()
    print(json.dumps(dict(transport="local file syscall fixture", shape=[layers, rows, width],
                          dtype="bfloat16", quota=quota, logical=s.logical_hidden_bytes,
                          ram_peak=s.ram_high_water_bytes, spill_write=s.spill_written_bytes,
                          spill_read=s.spill_read_bytes, before=before, sampled_peak_rss=peak,
                          process_peak_rss=getattr(process.memory_info(), "peak_wset", None),
                          after=memory_sample())))


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as folder:
        profile(Path(folder)/"cache", int(sys.argv[1]))
