"""Production synchronous I/O batching with a local OS-transport fixture."""
from collections import deque
import errno
import os
from unittest import mock

import pytest
import torch

from test_v183_hidden_store_on_cpu import local_transport, store


MIB = 1024 ** 2


@pytest.mark.parametrize("batch_mib", [4, 16, 64])
@pytest.mark.parametrize("ram_bytes", [None, 5 * MIB])
def test_batched_capture_keeps_noncontiguous_bf16_rows_and_math_chunks(tmp_path, batch_mib, ram_bytes):
    width, input_rows, layers = 2048, 3001, 2
    bits = torch.arange(input_rows * width * 2, dtype=torch.int32).short().reshape(input_rows, width * 2)
    bits[0, :6] = torch.tensor([-1, 1, -32768, 2, 32704, 3], dtype=torch.int16)
    hidden = bits[:, ::2].view(torch.bfloat16)
    assert not hidden.is_contiguous()
    valid = torch.ones(input_rows, dtype=torch.bool)
    valid[6::7] = False
    rows = int(valid.sum())
    expected = bits[:, ::2][valid]
    split = 1357
    first_rows = int(valid[:split].sum())
    payload_bytes = rows * width * 2 * layers
    batch_bytes = batch_mib * MIB
    writes = []
    with local_transport():
        args = (tmp_path / "hidden", "batch-bits", 0, 0, rows, width, layers, payload_bytes)
        if ram_bytes is None:
            s = store.DirectHiddenStore(*args, io_batch_bytes=batch_bytes)
        else:
            s = store.HybridHiddenStore(*args, io_batch_bytes=batch_bytes,
                                       ram_limit_bytes=ram_bytes, pressure=lambda: 2 << 30)
        try:
            stage_bytes = s.stage.untyped_storage().nbytes()
            assert stage_bytes == min(rows, batch_bytes // (2 * width)) * width * 2
            original = os.pwritev

            def counted(fd, buffers, offset):
                writes.append((fd, offset, len(buffers[0])))
                return original(fd, buffers, offset)

            with mock.patch.object(os, "pwritev", counted):
                for start, end, source_slice in ((0, first_rows, slice(0, split)),
                                                 (first_rows, rows, slice(split, None))):
                    for layer in range(layers):
                        s.write_rows(layer, start, end, hidden[source_slice], valid[source_slice])
            assert writes and max(length for _, _, length in writes) <= batch_bytes
            if ram_bytes is not None:
                actual = s.ram_storage_snapshot()
                assert actual["live_storage_bytes"] == actual["logical_ram_bytes"] == ram_bytes
                assert max(span.payload.untyped_storage().nbytes()
                           for span in s._resident.values()) <= store.WRITE_BYTES
                assert actual["live_storage_bytes"] + stage_bytes <= ram_bytes + batch_bytes
            s.seal_capture()
            assert s.stage is None
            assert s.spill_written_bytes == payload_bytes - (ram_bytes or 0)
            for layer in range(layers):
                result = torch.empty(rows, width, dtype=torch.bfloat16)
                # The consumer's boundaries stay fixed for all physical I/O batch sizes.
                for start, end in ((0, 17), (17, 1027), (1027, rows)):
                    s.read_rows_into(layer, start, end, result[start:end])
                assert torch.equal(result.view(torch.int16), expected)
                s.release_layer(layer)
            assert s.spill_read_bytes == s.spill_written_bytes
            s.close()
        finally:
            s.abort()
        assert not s.fds and not s.owned and s.stage is None


@pytest.mark.parametrize("batch_mib", [4, 16, 64])
def test_batched_short_io_eintr_seal_and_order(tmp_path, batch_mib):
    source = torch.arange(35, dtype=torch.int16).reshape(7, 5).view(torch.bfloat16)
    mask = torch.ones(7, dtype=torch.bool)
    with local_transport(limit=7):
        s = store.DirectHiddenStore(tmp_path / "hidden", "short", 0, 0, 7, 5, 2, 140,
                                    io_batch_bytes=batch_mib * MIB)
        try:
            write, read = os.pwritev, os.preadv
            attempts = {"write": 0, "read": 0}

            def interrupted(operation, kind, *args):
                attempts[kind] += 1
                if attempts[kind] == 1:
                    raise InterruptedError()
                return operation(*args)

            with mock.patch.object(os, "pwritev", lambda *args: interrupted(write, "write", *args)):
                s.write_rows(0, 0, 7, source, mask)
                with pytest.raises(RuntimeError, match="incomplete"):
                    s.seal_capture()
                s.write_rows(1, 0, 7, source, mask)
            with pytest.raises(RuntimeError, match="capture"):
                s.write_rows(1, 0, 7, source, mask)
            s.seal_capture()
            out = torch.empty_like(source)
            with mock.patch.object(os, "preadv", lambda *args: interrupted(read, "read", *args)):
                for layer in range(2):
                    s.read_rows_into(layer, 0, 7, out)
                    assert torch.equal(source.view(torch.int16), out.view(torch.int16))
                    s.release_layer(layer)
            assert attempts["write"] == attempts["read"] == 21
            s.close()
        finally:
            s.abort()


@pytest.mark.parametrize("batch_mib,expected_calls", [(4, 3), (16, 1), (64, 1)])
def test_only_physically_contiguous_spans_merge(tmp_path, batch_mib, expected_calls):
    source = torch.arange(24).reshape(6, 4).to(torch.bfloat16)
    with local_transport():
        s = store.HybridHiddenStore(tmp_path / "hidden", "adjacent", 0, 0, 6, 4, 1, 48,
                                   ram_limit_bytes=0, pressure=lambda: 2 << 30,
                                   io_batch_bytes=batch_mib * MIB)
        try:
            s.write_rows(0, 0, 6, source, torch.ones(6, dtype=torch.bool))
            # Split one valid captured extent without changing any physical or logical row.
            s._spans[0] = deque(store._Span(i, i + 2, i) for i in (0, 2, 4))
            s.seal_capture()
            out = torch.empty_like(source)
            with mock.patch.object(os, "preadv", wraps=os.preadv) as reads:
                s.read_rows_into(0, 0, 6, out)
            assert reads.call_count == expected_calls
            assert torch.equal(out, source)
            assert s.consumed == [6] and not any(s.ram_storage_snapshot().values())
            s.release_layer(0)
            s.close()
        finally:
            s.abort()


def test_reversed_pressure_evictions_keep_offsets_and_one_block_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "WRITE_BYTES", 16)
    source = torch.arange(32).reshape(8, 4).to(torch.bfloat16)
    slack = [2 << 30]
    with local_transport():
        s = store.HybridHiddenStore(tmp_path / "hidden", "reverse", 0, 0, 8, 4, 1, 64,
                                   ram_limit_bytes=64, pressure=lambda: slack[0], io_batch_bytes=16 * MIB)
        try:
            s.write_rows(0, 0, 8, source, torch.ones(8, dtype=torch.bool))
            assert len(s._resident) == 4
            slack[0] = -1
            for expected_remaining in (48, 32, 16, 0):
                s.poll_pressure()
                assert s.ram_hidden_bytes == expected_remaining
            s.seal_capture()
            out = torch.empty_like(source)
            with mock.patch.object(os, "preadv", wraps=os.preadv) as reads:
                s.read_rows_into(0, 0, 8, out)
            assert reads.call_count == 4
            assert torch.equal(out, source)
            assert s.spill_written_bytes == s.spill_read_bytes == 64
            s.release_layer(0)
            s.close()
        finally:
            s.abort()


@pytest.mark.parametrize("phase", ["write", "read"])
@pytest.mark.parametrize("error", [errno.EIO, getattr(errno, "ESTALE", 116), errno.ENOSPC,
                                  getattr(errno, "EDQUOT", 122), errno.EACCES])
def test_batched_failure_stops_consumption_and_preserves_peer(tmp_path, phase, error):
    sibling = tmp_path / "hf"
    sibling.write_bytes(b"keep")
    source = torch.ones(3, 4, dtype=torch.bfloat16)
    valid = torch.ones(3, dtype=torch.bool)
    with local_transport():
        peer = store.DirectHiddenStore(tmp_path / "hidden", "peer", 1, 0, 3, 4, 1, 24)
        s = store.HybridHiddenStore(tmp_path / "hidden", "failed", 0, 0, 3, 4, 1, 24,
                                   ram_limit_bytes=0, pressure=lambda: 2 << 30, io_batch_bytes=16 * MIB)
        peer_paths = list(peer.owned.values())
        try:
            if phase == "read":
                s.write_rows(0, 0, 3, source, valid)
                s.seal_capture()
            name = "pwritev" if phase == "write" else "preadv"
            with mock.patch.object(os, name, side_effect=OSError(error, "injected")):
                with pytest.raises(OSError) as caught:
                    if phase == "write":
                        s.write_rows(0, 0, 3, source, valid)
                    else:
                        s.read_rows_into(0, 0, 3, source)
            assert caught.value.errno == error
            assert s.consumed == [0]
            if phase == "write":
                assert s.written == [0]
                with pytest.raises(RuntimeError, match="incomplete"):
                    s.seal_capture()
            else:
                with pytest.raises(RuntimeError, match="unread"):
                    s.release_layer(0)
            s.abort()
            s.abort()
            assert not s.fds and not s.owned and not any(s.ram_storage_snapshot().values())
            assert sibling.read_bytes() == b"keep" and all(path.exists() for path in peer_paths)
        finally:
            s.abort()
            peer.abort()


@pytest.mark.parametrize("batch_mib", [4, 16, 64])
def test_empty_owner_needs_no_payload_or_system_calls(tmp_path, batch_mib):
    with local_transport():
        s = store.HybridHiddenStore(tmp_path / "hidden", "empty", 0, 0, 0, 2048, 2, 0,
                                   ram_limit_bytes=64 * MIB, pressure=lambda: 2 << 30,
                                   io_batch_bytes=batch_mib * MIB)
        try:
            assert s.stage.untyped_storage().nbytes() == 0
            for layer in range(2):
                s.write_rows(layer, 0, 0, torch.empty(1, 2048), torch.zeros(1, dtype=torch.bool))
            s.seal_capture()
            for layer in range(2):
                s.read_rows_into(layer, 0, 0, torch.empty(0, 2048, dtype=torch.bfloat16))
                s.release_layer(layer)
            assert all(stats["calls"] == 0 for stats in s.transfer_stats.values())
            s.close()
        finally:
            s.abort()


@pytest.mark.parametrize("batch_bytes", [True, 0, 2 * MIB, 65 * MIB, 16.0 * MIB])
def test_invalid_io_budget_fails_before_directory_creation(tmp_path, batch_bytes):
    with local_transport():
        root = tmp_path / "hidden"
        with pytest.raises(ValueError, match="I/O batch"):
            store.DirectHiddenStore(root, "invalid", 0, 0, 1, 2, 1, 4, io_batch_bytes=batch_bytes)
        assert not root.exists()


def test_merged_read_truncation_keeps_cursor_uncommitted(tmp_path):
    source = torch.arange(24).reshape(6, 4).to(torch.bfloat16)
    with local_transport():
        s = store.HybridHiddenStore(tmp_path / "hidden", "truncated", 0, 0, 6, 4, 1, 48,
                                   ram_limit_bytes=0, pressure=lambda: 2 << 30,
                                   io_batch_bytes=16 * MIB)
        try:
            s.write_rows(0, 0, 6, source, torch.ones(6, dtype=torch.bool))
            s._spans[0] = deque([store._Span(0, 3, 0), store._Span(3, 6, 3)])
            s.seal_capture()
            os.ftruncate(s.fds[0], 25)
            with pytest.raises(OSError, match="Incomplete"):
                s.read_rows_into(0, 0, 6, torch.empty_like(source))
            assert s.consumed == [0] and s.spill_read_bytes == 0
            with pytest.raises(RuntimeError, match="unread"):
                s.release_layer(0)
        finally:
            s.abort()
        assert not s.fds and not s.owned


def test_adjacent_logical_spans_with_physical_gap_do_not_merge(tmp_path):
    source = torch.arange(24).reshape(6, 4).to(torch.bfloat16)
    with local_transport():
        s = store.HybridHiddenStore(tmp_path / "hidden", "gap", 0, 0, 6, 4, 1, 48,
                                   ram_limit_bytes=0, pressure=lambda: 2 << 30,
                                   io_batch_bytes=16 * MIB)
        try:
            s.write_rows(0, 0, 6, source, torch.ones(6, dtype=torch.bool))
            # A compact-spill offset can follow existing extents after pressure eviction.
            # Keep a real, unreferenced extent between the two logical neighbours.
            next_offset = s._append_spill(0, source[3:])
            assert next_offset == 6
            s._spans[0] = deque([store._Span(0, 3, 0), store._Span(3, 6, next_offset)])
            s.seal_capture()
            out = torch.empty_like(source)
            with mock.patch.object(os, "preadv", wraps=os.preadv) as reads:
                s.read_rows_into(0, 0, 6, out)
            assert [call.args[2] for call in reads.call_args_list] == [0, 48]
            assert torch.equal(out, source)
            s.release_layer(0)
            s.close()
        finally:
            s.abort()
