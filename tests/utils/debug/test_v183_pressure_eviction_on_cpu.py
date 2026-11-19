"""Production bounded pressure spilling; only mount/I/O and OS accounting are CPU fixtures."""
import errno
import os
from pathlib import Path
import weakref

import pytest
import torch

from test_v183_hidden_store_on_cpu import local_transport, store


GIB = 1024**3


class Pressure:
    def __init__(self, share=(1, 1)):
        self.slack = 2 * GIB
        self.valid = True
        self.hard = 8 * GIB
        self.owner_share = share
        self.refreshes = 0
        self.reclaims = 0
        self.after_reclaim = None

    def __call__(self):
        return self.slack

    def snapshot(self):
        return dict(valid=self.valid, hard_headroom_bytes=self.hard,
                    unavailable_reason=None if self.valid else "cgroup membership changed")

    def refresh(self):
        self.refreshes += 1
        return self.slack

    def reclaim(self):
        self.reclaims += 1
        if self.after_reclaim:
            self.after_reclaim()


def capture(root, pressure, *, rows=16, width=4, layers=1, batch=48, rank=0, generation=0):
    payload = rows * width * 2 * layers
    s = store.HybridHiddenStore(root, "pressure", rank, generation, rows, width, layers, payload,
                               ram_limit_bytes=payload, pressure=pressure, pressure_evict_bytes=batch)
    # Include arbitrary BF16 bit patterns, not just finite numeric values.
    bits = torch.arange(rows * width * 2, dtype=torch.int16).reshape(rows, width * 2)[:, ::2]
    bits[0, :3] = torch.tensor([-1, -32768, 32704], dtype=torch.int16)
    assert not bits.is_contiguous()
    for layer in range(layers):
        s.write_rows(layer, 0, rows, bits.view(torch.bfloat16), torch.ones(rows, dtype=torch.bool))
    s.seal_capture()
    return s, bits


def consume(s, expected):
    s._pressure.slack = 2 * GIB
    for layer in range(s.layers):
        output = torch.empty((s.rows, s.width), dtype=torch.bfloat16)
        for first, end in ((0, 1), (1, s.rows - 1), (s.rows - 1, s.rows)):
            s.read_rows_into(layer, first, end, output[first:end])
        assert torch.equal(output.view(torch.int16), expected)
        s.release_layer(layer)
    s.close()
    assert not any(s.ram_storage_snapshot().values())


def test_pressure_batch_releases_multiple_blocks_without_new_staging(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "WRITE_BYTES", 16)
    pressure = Pressure((1, 8))
    with local_transport(limit=7):
        s, bits = capture(tmp_path / "cache", pressure, layers=2)
        try:
            references = [weakref.ref(span.payload) for span in s._resident.values()]
            pressure.slack = -32
            def check_freed():
                assert sum(ref() is None for ref in references) == 3
            pressure.after_reclaim = check_freed
            # Eviction uses views into its original blocks, not another hidden/staging copy.
            with monkeypatch.context() as patch:
                patch.setattr(torch, "empty", lambda *a, **k: pytest.fail("unexpected pressure allocation"))
                s.poll_pressure()
            assert s.pressure_batches == pressure.refreshes == pressure.reclaims == 1
            assert s.pressure_evicted_blocks == 3
            assert s.pressure_evicted_bytes == s.pressure_released_bytes == 48
            assert s.ram_hidden_bytes == 208 and s._spilling
            assert s.pressure_reason_counts == dict(soft=1, unknown=0, hard=0)
            assert s.stage is None
            # Unchanged physical observation must not re-open RAM admission.
            assert pressure.slack == -32
            pressure.after_reclaim = None
            consume(s, bits)
            assert s.spill_written_bytes == s.spill_read_bytes == 48
        finally:
            s.abort()


def test_owner_shares_do_not_each_reclaim_whole_node_deficit(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "WRITE_BYTES", 16)
    released = []
    with local_transport():
        for rank, share in enumerate(((1, 4), (3, 4))):
            pressure = Pressure(share)
            s, bits = capture(tmp_path / "cache", pressure, batch=64, rank=rank)
            try:
                # Owners remain in hysteresis after a preceding low-water sample.
                s._spilling = True
                pressure.slack = GIB - 64
                s.poll_pressure()
                released.append(s.pressure_released_bytes)
                consume(s, bits)
            finally:
                s.abort()
    assert released == [16, 48] and sum(released) == 64


def test_partial_span_releases_whole_storage_but_writes_only_unread_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "WRITE_BYTES", 32)
    pressure = Pressure()
    with local_transport():
        s, bits = capture(tmp_path / "cache", pressure, rows=4, batch=32)
        try:
            output = torch.empty(4, 4, dtype=torch.bfloat16)
            s.read_rows_into(0, 0, 1, output[:1])
            pressure.slack = 0
            s.poll_pressure()
            assert s.pressure_released_bytes == 32 and s.pressure_evicted_bytes == 24
            assert s.ram_hidden_bytes == 0
            pressure.slack = 2 * GIB
            s.read_rows_into(0, 1, 4, output[1:])
            assert torch.equal(output.view(torch.int16), bits)
            assert s.spill_written_bytes == s.spill_read_bytes == 24
            s.release_layer(0)
            s.close()
        finally:
            s.abort()


def test_unknown_accounting_stops_ram_admission_and_spills_bounded(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "WRITE_BYTES", 16)
    pressure = Pressure()
    with local_transport():
        s, bits = capture(tmp_path / "cache", pressure, batch=32)
        try:
            pressure.valid = False
            # Unknown accounting cannot be overridden by a previously positive slack.
            s.poll_pressure()
            assert s.ram_hidden_bytes == 96 and s._spilling
            assert s.pressure_reason_counts["unknown"] == 1
            assert s.pressure_evicted_bytes == 32
            pressure.valid = True
            consume(s, bits)
        finally:
            s.abort()


@pytest.mark.parametrize("after_write", [False, True])
def test_known_hard_headroom_exhaustion_propagates_without_consuming(tmp_path, monkeypatch, after_write):
    monkeypatch.setattr(store, "WRITE_BYTES", 16)
    pressure = Pressure()
    with local_transport():
        s, _ = capture(tmp_path / "cache", pressure, batch=32)
        try:
            pressure.slack = -1
            if after_write:
                pressure.after_reclaim = lambda: setattr(pressure, "hard", 0)
            else:
                pressure.hard = 0
            with pytest.raises(MemoryError, match="hard headroom"):
                s.poll_pressure()
            assert s.consumed == [0]
            assert s.ram_hidden_bytes == (96 if after_write else 128)
            assert s.pressure_reason_counts["hard"] == 1
        finally:
            s.abort()
            s.abort()
        assert not s.fds and not any(s.ram_storage_snapshot().values())


@pytest.mark.parametrize("error", [errno.ENOSPC, errno.EIO, getattr(errno, "ESTALE", 116),
                                  getattr(errno, "EDQUOT", 122)])
def test_second_write_failure_keeps_unwritten_tensor_authoritative(tmp_path, monkeypatch, error):
    monkeypatch.setattr(store, "WRITE_BYTES", 16)
    pressure = Pressure()
    with local_transport():
        s, _ = capture(tmp_path / "cache", pressure, batch=48)
        neighbor = tmp_path / "user-owned"
        neighbor.write_bytes(b"preserve")
        try:
            pressure.slack = -1
            previous = list(s._resident.items())
            original_transfer = store._transfer
            calls = []
            def write(fd, tensor, offset, **kwargs):
                calls.append(offset)
                if len(calls) == 2:
                    raise OSError(error, "injected second pressure write")
                return original_transfer(fd, tensor, offset, **kwargs)
            with monkeypatch.context() as patch:
                patch.setattr(store, "_transfer", write)
                with pytest.raises(OSError) as caught:
                    s.poll_pressure()
            assert caught.value.errno == error
            assert s.pressure_evicted_bytes == 16 and s.ram_hidden_bytes == 112
            assert s.consumed == [0] and pressure.reclaims == pressure.refreshes == 0
            key, span = previous[-2]
            assert s._resident[key] is span and isinstance(span.payload, torch.Tensor)
            with pytest.raises(RuntimeError, match="unread"):
                s.release_layer(0)
        finally:
            s.abort()
        assert neighbor.read_bytes() == b"preserve"
        assert not s.fds and not s.owned and s.ram_hidden_bytes == 0


@pytest.mark.parametrize("quota", [0, 48, 1024])
def test_multi_generation_spill_or_no_payload_io(tmp_path, monkeypatch, quota):
    monkeypatch.setattr(store, "WRITE_BYTES", 16)
    with local_transport():
        for generation in range(3):
            pressure = Pressure()
            rows, width, layers = 5, 4, 2
            payload = rows * width * 2 * layers
            s = store.HybridHiddenStore(tmp_path / "cache", "generations", 0, generation, rows, width,
                                       layers, payload, ram_limit_bytes=quota, pressure=pressure)
            bits = torch.arange(rows * width, dtype=torch.int16).reshape(rows, width)
            try:
                for layer in range(layers):
                    s.write_rows(layer, 0, rows, bits.view(torch.bfloat16), torch.ones(rows, dtype=torch.bool))
                s.seal_capture()
                assert s.ram_hidden_bytes == min(quota, payload)
                consume(s, bits)
                assert s.spill_read_bytes == s.spill_written_bytes == max(0, payload - quota)
                if quota >= payload:
                    assert s.transfer_stats["read"]["calls"] == s.transfer_stats["write"]["calls"] == 0
                assert s.pressure_evicted_bytes == 0
            finally:
                s.abort()
            assert not list(tmp_path.rglob("*.bin"))


@pytest.mark.parametrize("limit", [True, 0, 2, 65 * 1024**2, 4.0 * 1024**2])
def test_pressure_budget_validation_precedes_directory_creation(tmp_path, limit):
    with pytest.raises(ValueError, match="pressure batch"):
        store.HybridHiddenStore(tmp_path / "cache", "bad", 0, 0, 1, 4, 1, 8,
                               ram_limit_bytes=8, pressure=Pressure(), pressure_evict_bytes=limit)
    assert not (tmp_path / "cache").exists()


def profile_pressure(root, variant, size_mib):
    """Production poll comparison with static OS readings and local file transport, not NFS."""
    import ast
    import json
    import subprocess
    import time
    from types import MethodType

    import psutil

    from test_v183_hidden_store_on_cpu import ROOT, memory_sample

    torch.set_num_threads(1)
    width = 2048
    block_rows = store.WRITE_BYTES // (width * 2)
    rows = size_mib * 1024**2 // (width * 2)
    bits = torch.arange(block_rows * width, dtype=torch.int32).short().reshape(block_rows, width)
    hidden = bits.view(torch.bfloat16)
    mask = torch.ones(block_rows, dtype=torch.bool)
    pressure = Pressure()
    root.mkdir(parents=True, exist_ok=True)
    with local_transport():
        s = store.HybridHiddenStore(root / "hidden", variant, 0, 0, rows, width, 1, rows * width * 2,
                                   ram_limit_bytes=rows * width * 2, pressure=pressure)
        try:
            if variant == "old":
                source = subprocess.check_output([
                    "git", "-c", "safe.directory=" + str(ROOT), "-C", str(ROOT), "show",
                    "ed6a60883504f4a87aee5abfdd7a7ce8f763ac53:verl/utils/debug/eu_hidden_store.py",
                ], text=True, encoding="utf-8")
                cls = next(node for node in ast.parse(source).body
                           if isinstance(node, ast.ClassDef) and node.name == "HybridHiddenStore")
                function = next(node for node in cls.body
                                if isinstance(node, ast.FunctionDef) and node.name == "poll_pressure")
                namespace = {}
                exec(compile(ast.Module(body=[function], type_ignores=[]), "old_store", "exec"), namespace)
                s.poll_pressure = MethodType(namespace["poll_pressure"], s)
            for first in range(0, rows, block_rows):
                s.write_rows(0, first, first + block_rows, hidden, mask)
            s.seal_capture()
            before = memory_sample()
            process = psutil.Process()
            peak = before["rss"]
            pressure.slack = -1
            polls = 0
            started = time.perf_counter()
            while s.ram_hidden_bytes:
                s.poll_pressure()
                polls += 1
                peak = max(peak, process.memory_info().rss)
            spill_seconds = time.perf_counter() - started
            pressure.slack = 2 * GIB
            output = torch.empty_like(hidden)
            started = time.perf_counter()
            for first in range(0, rows, block_rows):
                s.read_rows_into(0, first, first + block_rows, output)
                assert torch.equal(output.view(torch.int16), bits)
            replay_seconds = time.perf_counter() - started
            s.release_layer(0)
            s.close()
            print(json.dumps(dict(
                variant=variant, scope="production poll, static OS pressure fixture, local buffered syscalls",
                shape=[rows, width], dtype="bfloat16", payload_bytes=rows * width * 2,
                io_batch_bytes=s.io_batch_bytes, polls=polls, fresh_samples=pressure.refreshes,
                reclaim_callback_calls=pressure.reclaims, spill_written_bytes=s.spill_written_bytes,
                spill_read_bytes=s.spill_read_bytes, transfer_stats=s.transfer_stats,
                spill_seconds=spill_seconds, replay_and_bitcheck_seconds=replay_seconds,
                before=before, after=memory_sample(), sampled_peak_rss=peak, bit_exact=True,
                gpu_validation="NOT_RUN", nas_direct_io_validation="NOT_RUN",
            )))
        finally:
            s.abort()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--variant", choices=["old", "new"], required=True)
    parser.add_argument("--size-mib", type=int, choices=[64, 128], default=64)
    args = parser.parse_args()
    profile_pressure(args.benchmark_root, args.variant, args.size_mib)
