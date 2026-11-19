"""Temporary BF16 actual-F rows on a shared Linux NFSv4.1 mount."""

from __future__ import annotations

import errno
import json
import logging
import os
from pathlib import Path
import re
import socket
import stat
import time
from collections import deque
from contextlib import ExitStack, contextmanager

WRITE_BYTES = 4 * 1024**2
LOG = logging.getLogger(__name__)


def _component(value):
    if not re.fullmatch(r"[A-Za-z0-9_-][A-Za-z0-9._-]*", value):
        raise ValueError(f"Invalid HiddenStore path component: {value!r}")
    return value


def _verify_mount(fd):
    # Inspect the open directory, including bind/overmounts, before any mkdir.
    info = Path(f"/proc/self/fdinfo/{fd}").read_text()
    mount_id = next(line.split()[1] for line in info.splitlines() if line.startswith("mnt_id:"))
    for line in Path("/proc/self/mountinfo").read_text().splitlines():
        fields, filesystem = line.split(" - ", 1)
        if fields.split()[0] != mount_id:
            continue
        fs, _, options = filesystem.split()
        options = set(options.split(","))
        if (fs in {"nfs4", "nfs"} and "vers=4.1" in options and "hard" in options
                and "rw" in fields.split()[5].split(",")):
            return
        break
    raise OSError(f"HiddenStore parent requires writable hard NFSv4.1 (mount id {mount_id})")


@contextmanager
def mounted_root(root):
    """Pin the existing parent so an unmount cannot redirect mkdir onto /."""
    root = Path(root)
    if not root.is_absolute() or root == root.parent or root.resolve() != root:
        raise ValueError(f"HiddenStore needs a canonical absolute path without symlinks: {root}")
    if not hasattr(os, "O_DIRECT"):
        raise OSError(errno.ENOTSUP, "HiddenStore requires Linux NFS O_DIRECT")
    flags = os.O_DIRECTORY | os.O_NOFOLLOW
    fd = os.open(root.parent, flags)
    try:
        _verify_mount(fd)
        pinned = Path(f"/proc/self/fd/{fd}")
        if pinned.resolve(strict=True) != root.parent:
            raise OSError("HiddenStore parent changed while resolving mount")
        yield pinned / root.name
    finally:
        os.close(fd)


def _mkdir(path, device, *, exclusive=False):
    path.mkdir(mode=0o700, exist_ok=not exclusive)
    info = path.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_dev != device:
        raise OSError(f"HiddenStore path crosses a mount or symlink: {path}")


@contextmanager
def _directory_handle(path, device):
    fd = os.open(path, os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        if os.fstat(fd).st_dev != device:
            raise OSError(f"HiddenStore directory device changed: {path}")
        yield Path(f"/proc/self/fd/{fd}")
    finally:
        os.close(fd)


@contextmanager
def generation_dir(root, components, shared_bytes):
    """Anchor every child to its verified directory fd (including during unlink)."""
    with mounted_root(root) as base:
        device = base.parent.stat().st_dev
        fs = os.statvfs(base.parent)
        available = fs.f_bavail * fs.f_frsize
        if shared_bytes > available:
            raise OSError(errno.ENOSPC, f"Shared HiddenStore needs {shared_bytes} bytes, available {available}")
        with ExitStack() as stack:
            path = base
            for index, name in enumerate((base.name, *components)):
                if index:
                    path = parent / name
                last = index == len(components)
                _mkdir(path, device, exclusive=last)
                parent = stack.enter_context(_directory_handle(path, device))
            try:
                yield parent, available
            finally:
                # Only the exclusively created generation can be removed.
                try:
                    if path.lstat().st_ino == parent.stat().st_ino:
                        path.rmdir()
                except OSError as exc:
                    if exc.errno != errno.ENOTEMPTY:
                        raise
                    LOG.warning("HiddenStore left nonempty generation %s", components)


def _open_payload(path):
    return os.open(path, os.O_CREAT | os.O_EXCL | os.O_RDWR | os.O_NOFOLLOW | os.O_DIRECT, 0o600)


def _transfer(fd, tensor, offset, *, write, counters=None):
    """NFS direct I/O accepts arbitrary byte lengths. Retry only untransferred bytes."""
    import torch

    data = memoryview(tensor.view(-1).view(torch.uint8).numpy()).cast("B")
    operation = os.pwritev if write else os.preadv
    while data:
        if counters is not None:
            counters["calls"] += 1
            counters["requested_bytes"] += len(data)
            # Power-of-two bins keep telemetry bounded even with arbitrary short I/O.
            bucket = (len(data) - 1).bit_length()
            sizes = counters["request_size_log2"]
            sizes[bucket] = sizes.get(bucket, 0) + 1
        try:
            count = operation(fd, [data], offset)
        except InterruptedError:
            if counters is not None:
                counters["eintr"] += 1
            continue
        if not 0 < count <= len(data):
            raise OSError(errno.EIO, f"Incomplete HiddenStore transfer at offset {offset}")
        if counters is not None:
            counters["completed_bytes"] += count
        offset += count
        data = data[count:]


class DirectHiddenStore:
    """One generation per SP-local rank. Capture and replay use disjoint buffer lifetimes."""

    def __init__(self, root, attempt, rank, generation, rows, width, layers, shared_bytes, *,
                 io_batch_bytes=None):
        import torch

        io_batch_bytes = WRITE_BYTES if io_batch_bytes is None else io_batch_bytes
        if type(io_batch_bytes) is not int or not WRITE_BYTES <= io_batch_bytes <= 64 * 1024**2:
            raise ValueError("HiddenStore I/O batch must be between the capture block and 64 MiB")
        if (min(rank, generation, rows) < 0 or min(width, layers) <= 0 or width * 2 > WRITE_BYTES
                or shared_bytes < rows * width * 2 * layers):
            raise ValueError("Invalid HiddenStore dimensions")
        self.rows, self.width, self.layers = rows, width, layers
        self.io_batch_bytes = io_batch_bytes
        self.transfer_stats = {
            direction: dict(calls=0, requested_bytes=0, completed_bytes=0, eintr=0, request_size_log2={})
            for direction in ("read", "write")
        }
        self.logical_hidden_bytes = rows * width * 2 * layers
        self.ram_hidden_bytes = self.ram_high_water_bytes = 0
        self.spill_written_bytes = self.spill_read_bytes = 0
        self.d2h_copy_seconds = self.nas_write_seconds = self.nas_read_seconds = 0.0
        self._disk_rows = [0] * layers
        self.written = [0] * layers
        self.consumed = [0] * layers
        self.inodes = {}
        self.fds = {}
        self.owned = {}
        self.sealed = False
        self.closed = False
        self.stage = None
        self.directory = None
        self.manifest = None
        host = _component(socket.gethostname())
        attempt = _component(attempt)
        self.location = Path(root) / attempt / host / f"rank_{rank}" / f"generation_{generation}"
        self.header = dict(format=1, backend="nfs_direct", dtype="bfloat16", shape=[rows, width],
                           layers=layers, rank=rank, generation=generation, attempt=attempt, hostname=host,
                           logical_bytes=rows * width * 2 * layers, reserved_bytes=0,
                           io_batch_bytes=io_batch_bytes)
        components = (attempt, host, f"rank_{rank}", f"generation_{generation}")
        self.parent = generation_dir(root, components, shared_bytes)
        self.directory, available = self.parent.__enter__()
        try:
            self.manifest = self.directory / "manifest.json"
            self._snapshot()
            for layer in range(layers):
                path = self.directory / f"layer_{layer}.bin"
                fd = _open_payload(path)
                self.fds[layer] = fd
                self.owned[layer] = path
                self.inodes[layer] = os.fstat(fd).st_ino
            self._snapshot()
            self.stage = torch.empty((min(rows, io_batch_bytes // (2 * width)), width),
                                     dtype=torch.bfloat16, device="cpu")
            self.write_stage_bytes = self.stage.untyped_storage().nbytes()
            print(f"EU-DERPO HiddenStore {self.location} shared_required={shared_bytes} "
                  f"available={available} hard_reserved=0 write_buffer={self.stage.numel() * 2}", flush=True)
        except BaseException:
            self.abort()
            raise

    def _snapshot(self):
        record = dict(self.header, sealed=self.sealed, written_rows=self.written, consumed_rows=self.consumed)
        record["inodes"] = self.inodes
        pending = self.manifest.with_suffix(".next")
        pending.write_text(json.dumps(record) + "\n", encoding="utf-8")
        pending.replace(self.manifest)

    def write_rows(self, layer, start, end, hidden, valid):
        import torch

        if (self.closed or self.sealed or layer not in self.fds or start != self.written[layer]
                or not start <= end <= self.rows or hidden.shape != (valid.numel(), self.width)):
            raise RuntimeError("HiddenStore capture order/shape mismatch")
        if valid.ndim != 1 or valid.dtype != torch.bool:
            raise ValueError("HiddenStore requires a one-dimensional valid-row mask")
        positions = valid.nonzero(as_tuple=True)[0]
        if positions.numel() != end - start:
            raise RuntimeError("HiddenStore planned/captured rows differ")
        if start == end:
            return
        while self.written[layer] < end:
            cursor = self.written[layer]
            first = cursor - start
            indices = positions[first:first + len(self.stage)]
            self.written[layer] += self._capture_block(layer, cursor, hidden, indices)

    def _copy_rows(self, block, hidden, indices):
        import torch

        if hidden.device.type == "cpu" and hidden.dtype == block.dtype:
            torch.index_select(hidden.detach(), 0, indices, out=block)
        else:
            selected = hidden.detach().index_select(0, indices)
            started = time.perf_counter()
            block.copy_(selected, non_blocking=False)
            if hidden.device.type != "cpu":
                self.d2h_copy_seconds += time.perf_counter() - started

    def _append_spill(self, layer, block):
        offset = self._disk_rows[layer]
        fd = self.fds[layer]
        if self.sealed and os.fstat(fd).st_size != offset * self.width * 2:
            raise OSError(errno.EIO, "Sealed HiddenStore file changed before eviction")
        started = time.perf_counter()
        _transfer(fd, block, offset * self.width * 2, write=True, counters=self.transfer_stats["write"])
        self.nas_write_seconds += time.perf_counter() - started
        self.spill_written_bytes += block.numel() * 2
        self._disk_rows[layer] += len(block)
        if self.sealed:
            self._allocated[layer] = os.fstat(fd).st_blocks * 512
            self.header.update(written_bytes=self.spill_written_bytes, allocated_bytes=sum(self._allocated))
        return offset

    def _capture_block(self, layer, start, hidden, indices):
        block = self.stage[:len(indices)]
        # Batch NAS writes without increasing the existing D2H selection temporary.
        rows_per_copy = WRITE_BYTES // (self.width * 2)
        for first in range(0, len(indices), rows_per_copy):
            self._copy_rows(block[first:first + rows_per_copy], hidden, indices[first:first + rows_per_copy])
        self._append_spill(layer, block)
        return len(block)

    def seal_capture(self):
        if self.closed or self.sealed or any(n != self.rows for n in self.written):
            raise RuntimeError("HiddenStore capture is incomplete or already sealed")
        sizes = [os.fstat(fd) for fd in self.fds.values()]
        if any(s.st_size != self._disk_rows[layer] * self.width * 2 for layer, s in enumerate(sizes)):
            raise OSError(errno.EIO, "HiddenStore payload length differs from complete captured rows")
        self.stage = None
        self.sealed = True
        self._allocated = [s.st_blocks * 512 for s in sizes]
        self.header.update(written_bytes=self.spill_written_bytes, allocated_bytes=sum(self._allocated))
        self._snapshot()

    def read_rows_into(self, layer, start, end, destination):
        import torch

        if (self.closed or not self.sealed or layer not in self.fds or start != self.consumed[layer]
                or not start <= end <= self.rows or destination.shape != (end - start, self.width)
                or destination.dtype != torch.bfloat16 or destination.device.type != "cpu"
                or not destination.is_contiguous()):
            raise RuntimeError("HiddenStore replay order/shape mismatch")
        self._read_range(layer, start, end, destination)
        self.consumed[layer] = end

    def _read_spill(self, layer, offset, destination):
        started = time.perf_counter()
        _transfer(self.fds[layer], destination, offset * self.width * 2, write=False,
                  counters=self.transfer_stats["read"])
        self.nas_read_seconds += time.perf_counter() - started
        self.spill_read_bytes += destination.numel() * 2

    def _read_range(self, layer, start, end, destination):
        self._read_spill(layer, start, destination)

    def _remove_layer(self, layer):
        if layer in self.fds:
            os.close(self.fds.pop(layer))
        path = self.owned[layer]
        if path.lstat().st_ino != self.inodes.get(layer):
            raise OSError(f"HiddenStore payload identity changed: {path}")
        path.unlink()
        del self.owned[layer]

    def release_layer(self, layer):
        if not self.sealed or self.consumed[layer] != self.rows:
            raise RuntimeError("HiddenStore layer has unread rows")
        self._remove_layer(layer)
        self._snapshot()

    def close(self):
        if self.closed:
            return
        if self.owned or not self.sealed:
            raise RuntimeError("HiddenStore still has live payload")
        self.manifest.unlink()
        self.parent.__exit__(None, None, None)
        self.closed = True

    def abort(self):
        if self.closed:
            return
        self.stage = None
        for layer in list(self.owned):
            try:
                self._remove_layer(layer)
            except OSError:
                LOG.exception("Could not clean HiddenStore layer %s at %s", layer, self.location)
        try:
            if self.manifest and not self.owned:
                self.manifest.with_suffix(".next").unlink(missing_ok=True)
                self.manifest.unlink(missing_ok=True)
        except OSError:
            LOG.exception("Could not clean HiddenStore metadata at %s", self.location)
        finally:
            try:
                self.parent.__exit__(None, None, None)
            except OSError:
                LOG.exception("Could not close HiddenStore parent at %s", self.location)
            self.closed = True


class _Span:
    __slots__ = ("start", "end", "payload")

    def __init__(self, start, end, payload):
        self.start, self.end = start, end
        self.payload = payload  # Independent CPU tensor, or first row in the compact spill file.


class HybridHiddenStore(DirectHiddenStore):
    """One owner of a node-granted quota. Synchronous capture, pressure eviction and replay."""

    def __init__(self, root, attempt, rank, generation, rows, width, layers, shared_bytes, *,
                 ram_limit_bytes, pressure, io_batch_bytes=None):
        if ram_limit_bytes < 0:
            raise ValueError("HiddenStore RAM quota must be nonnegative")
        self.ram_limit_bytes = ram_limit_bytes
        self._pressure = pressure
        self._spilling = False
        self._spans = [deque() for _ in range(layers)]
        self._resident = {}
        super().__init__(root, attempt, rank, generation, rows, width, layers, shared_bytes,
                         io_batch_bytes=io_batch_bytes)
        try:
            self.header.update(backend="ram_nfs_hybrid", ram_limit_bytes=ram_limit_bytes)
            self._snapshot()
        except BaseException:
            self.abort()
            raise

    def poll_pressure(self):
        """Below the reserved headroom, stop admission and spill at most one bounded block per poll."""
        if not self.ram_limit_bytes:
            return
        slack = self._pressure()
        if slack <= 0:
            self._spilling = True
        elif slack >= 1024**3:
            self._spilling = False
        if self._spilling and self._resident:
            layer, original_start = key = next(reversed(self._resident))
            span = self._resident[key]
            unread = max(self.consumed[layer], original_start)
            tensor = span.payload
            offset = self._append_spill(layer, tensor[unread - original_start:])
            span.start, span.payload = unread, offset
            del self._resident[key]
            self.ram_hidden_bytes -= tensor.numel() * 2

    def _capture_block(self, layer, start, hidden, indices):
        import torch

        self.poll_pressure()
        free_rows = (self.ram_limit_bytes - self.ram_hidden_bytes) // (self.width * 2)
        runs = self._spans[layer]
        if free_rows and not self._spilling:
            count = min(free_rows, len(indices), WRITE_BYTES // (self.width * 2))
            block = torch.empty((count, self.width), device="cpu", dtype=torch.bfloat16)
            self._copy_rows(block, hidden, indices[:count])
            span = _Span(start, start + count, block)
            runs.append(span)
            self._resident[layer, start] = span
            self.ram_hidden_bytes += block.numel() * 2
            self.ram_high_water_bytes = max(self.ram_hidden_bytes, self.ram_high_water_bytes)
            return count

        offset = self._disk_rows[layer]
        count = super()._capture_block(layer, start, hidden, indices)
        # With a full/zero quota, adjacent writes replay in one transfer per original Step E chunk.
        if (runs and isinstance(runs[-1].payload, int)
                and runs[-1].payload + start - runs[-1].start == offset):
            runs[-1].end = start + count
        else:
            runs.append(_Span(start, start + count, offset))
        return count

    def _read_range(self, layer, start, end, destination):
        self.poll_pressure()
        runs = self._spans[layer]
        cursor = start
        while cursor < end:
            span = runs[0]
            if self.io_batch_bytes > WRITE_BYTES and isinstance(span.payload, int):
                # Eviction can leave adjacent spans; only contiguous disk offsets can merge.
                while len(runs) > 1 and span.end < end:
                    following = runs[1]
                    if (not isinstance(following.payload, int) or following.start != span.end
                            or following.payload != span.payload + span.end - span.start):
                        break
                    runs.popleft()
                    following.start, following.payload = span.start, span.payload
                    span = following
            stop = min(end, span.end)
            if not span.start <= cursor < stop:
                raise RuntimeError("HiddenStore replay coverage mismatch")
            dest = destination[cursor - start:stop - start]
            first = cursor - span.start
            if isinstance(span.payload, int):
                self._read_spill(layer, span.payload + first, dest)
            else:
                dest.copy_(span.payload[first:first + len(dest)])
                if stop == span.end:
                    self.ram_hidden_bytes -= span.payload.numel() * 2
                    del self._resident[layer, span.start]
            if stop == span.end:
                runs.popleft()
            cursor = stop

    def ram_storage_snapshot(self):
        """Count BF16 storages once across the resident index and replay spans."""
        storages = {}
        for spans in (self._resident.values(), *self._spans):
            for span in spans:
                if not isinstance(span.payload, int):
                    storage = span.payload.untyped_storage()
                    storages[storage.data_ptr()] = storage.nbytes()
        return {
            "logical_ram_bytes": self.ram_hidden_bytes,
            "live_storage_bytes": sum(storages.values()),
            "live_storage_blocks": len(storages),
            "resident_entries": len(self._resident),
            "span_entries": sum(map(len, self._spans)),
        }

    def close(self):
        # Called in the existing final-layer I/O error vote, before optimizer admission.
        if self.ram_hidden_bytes or self._resident or any(self._spans):
            raise RuntimeError("HiddenStore replay left resident blocks or spans")
        super().close()

    def abort(self):
        self._resident.clear()
        self._spans.clear()
        self.ram_hidden_bytes = 0
        super().abort()


def manage_orphans(root, attempt, unlink_one=None):
    """List one explicit attempt; removal requires one registered relative file at a time."""
    with mounted_root(root) as base:
        folder = base / _component(attempt)
        if not folder.is_dir() or folder.is_symlink() or base.is_symlink():
            raise ValueError("Missing or redirected HiddenStore attempt")
        device = base.parent.stat().st_dev
        # Re-open the selected lineage without following links before touching any file.
        for candidate in folder.glob("*/rank_*/generation_*"):
            host, rank, generation = candidate.relative_to(folder).parts
            with ExitStack() as stack:
                owner = base.parent
                for component in (base.name, attempt, host, rank, generation):
                    owner = stack.enter_context(_directory_handle(owner / component, device))
                manifest = owner / "manifest.json"
                record = {}
                if manifest.exists():
                    if not stat.S_ISREG(manifest.lstat().st_mode):
                        raise ValueError("Redirected HiddenStore manifest")
                    record = json.loads(manifest.read_text(encoding="utf8"))
                    if (record["format"] != 1 or record["attempt"] != attempt or record["hostname"] != host
                            or rank != f"rank_{record['rank']}" or generation != f"generation_{record['generation']}"):
                        raise ValueError("HiddenStore manifest owner mismatch")
                for path in owner.glob("layer_*.bin"):
                    match = re.fullmatch(r"layer_([0-9]+)\.bin", path.name)
                    if not match:
                        continue
                    info = path.lstat()
                    registered = (stat.S_ISREG(info.st_mode) and
                                  info.st_ino == record.get("inodes", {}).get(match[1]))
                    name = "/".join((host, rank, generation, path.name))
                    if unlink_one == name and registered:
                        path.unlink()
                        return
                    if unlink_one is None:
                        print(name, info.st_size, "registered" if registered else "unregistered (inspect manually)")
        if unlink_one is not None:
            raise ValueError("File is not registered to this attempt")


def probe(root):
    import time
    import uuid
    import torch

    rows, width = 128, 2048
    store = DirectHiddenStore(root, "probe-" + uuid.uuid4().hex, 0, 0, rows, width, 1, rows * width * 2)
    try:
        source = torch.arange(rows * width, dtype=torch.int32).short().view(torch.bfloat16).reshape(rows, width)
        mask = torch.ones(rows, dtype=torch.bool)
        start = time.monotonic()
        store.write_rows(0, 0, rows, source, mask)
        store.seal_capture()
        written = time.monotonic()
        recovered = torch.empty_like(source)
        store.read_rows_into(0, 0, rows, recovered)
        read = time.monotonic()
        if not torch.equal(recovered.view(torch.int16), source.view(torch.int16)):
            raise IOError("BF16 bit mismatch")
        store.release_layer(0)
        store.close()
        print(json.dumps(dict(backend="nfs_direct", bytes_per_direction=rows * width * 2,
                              write_seconds=written - start, read_seconds=read - written)))
    finally:
        store.abort()
    with mounted_root(root) as base:
        rank_dir = base / store.header["attempt"] / store.header["hostname"] / "rank_0"
        for directory in (rank_dir, rank_dir.parent, rank_dir.parent.parent):
            directory.rmdir()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--probe", action="store_true")
    group.add_argument("--check-parent", action="store_true", help="verify mount before creating cache directories")
    group.add_argument("--attempt", help="list registered payloads of this exact attempt")
    parser.add_argument("--unlink-one", metavar="RELATIVE_FILE", help="delete just this registered orphan after confirming the attempt has stopped")
    args = parser.parse_args()
    if not args.attempt and args.unlink_one:
        parser.error("--unlink-one requires --attempt")
    if args.check_parent:
        with mounted_root(args.root):
            pass
    elif args.probe:
        probe(args.root)
    else:
        manage_orphans(args.root, args.attempt, args.unlink_one)
