"""Exercise the production store. Only mount discovery and OS syscalls are emulated off NFS."""
import contextlib
import errno
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location("store_under_test", ROOT / "verl/utils/debug/eu_hidden_store.py")
store = importlib.util.module_from_spec(spec)
spec.loader.exec_module(store)


@contextlib.contextmanager
def local_transport(limit=None):
    """Explicit local-file stand-in, no production backend switch or silent fallback."""
    def pwritev(fd, buffers, offset):
        os.lseek(fd, offset, os.SEEK_SET)
        return os.write(fd, buffers[0][:limit])

    def preadv(fd, buffers, offset):
        os.lseek(fd, offset, os.SEEK_SET)
        data = os.read(fd, len(buffers[0][:limit]))
        buffers[0][:len(data)] = data
        return len(data)

    original_fstat = os.fstat

    def fstat(fd):
        st = original_fstat(fd)
        return SimpleNamespace(st_ino=st.st_ino, st_dev=st.st_dev, st_size=st.st_size, st_blocks=getattr(st, "st_blocks", 0))

    with contextlib.ExitStack() as stack:
        for name, value in (("pwritev", pwritev), ("preadv", preadv),
                            ("statvfs", lambda path: SimpleNamespace(f_bavail=2**44, f_frsize=1)),
                            ("fstat", fstat)):
            stack.enter_context(mock.patch.object(os, name, value, create=True))
        stack.enter_context(mock.patch.object(store, "mounted_root", lambda root: contextlib.nullcontext(Path(root))))
        if os.name == "nt":
            stack.enter_context(mock.patch.object(store, "_directory_handle", lambda path, device: contextlib.nullcontext(path)))
        stack.enter_context(mock.patch.object(store, "_open_payload", lambda p:
                            os.open(p, os.O_CREAT | os.O_EXCL | os.O_RDWR | getattr(os, "O_BINARY", 0), 0o600)))
        yield


@pytest.fixture
def local(tmp_path):
    with local_transport(limit=11):
        yield tmp_path / "hidden"


def opened(root, rows=5, width=3, layers=2, attempt="v183-test", rank=0, generation=0):
    return store.DirectHiddenStore(root, attempt, rank, generation, rows, width, layers, rows * width * 2 * layers)


def test_bit_exact_short_io_and_chunk_boundaries(local, monkeypatch):
    monkeypatch.setattr(store, "WRITE_BYTES", 18)
    s = opened(local)
    # Include all-ones NaN payload, signed zero, and non-contiguous source columns.
    bits = torch.tensor([[-1, 0, -32768, 3, 32704, 5]], dtype=torch.int16).repeat(7, 1)
    bits[:, 2] += torch.arange(7, dtype=torch.int16)
    hidden = bits[:, ::2].view(torch.bfloat16)
    mask = torch.tensor([1, 0, 1, 1, 0, 1, 1], dtype=torch.bool)
    dst = torch.empty(5, 3, dtype=torch.bfloat16)
    try:
        with pytest.raises(RuntimeError, match="replay"):
            s.read_rows_into(0, 0, 5, dst)
        for l in range(2):
            s.write_rows(l, 0, 2, hidden[:3], mask[:3])
            with pytest.raises(RuntimeError, match="capture order"):
                s.write_rows(l, 0, 2, hidden[:3], mask[:3])
            with pytest.raises(RuntimeError, match="incomplete"):
                s.seal_capture()
            s.write_rows(l, 2, 5, hidden[3:], mask[3:])
        assert s.stage.untyped_storage().nbytes() == 18
        s.seal_capture()
        assert s.stage is None
        assert s.header["written_bytes"] == 60 and s.header["reserved_bytes"] == 0
        for l in range(2):
            s.read_rows_into(l, 0, 1, dst[:1])
            with pytest.raises(RuntimeError, match="unread"):
                s.release_layer(l)
            s.read_rows_into(l, 1, 5, dst[1:])
            assert torch.equal(dst.view(torch.int16), bits[mask][:, ::2])
            s.release_layer(l)
        assert json.loads(s.manifest.read_text())["consumed_rows"] == [5, 5]
        s.close()
        s.close()
    finally:
        s.abort()
    assert not list(local.rglob("*.bin"))


def test_empty_multiple_generations_and_attempt_ownership(local):
    peer = opened(local, attempt="other", rank=3)
    try:
        for gen in range(3):
            for rank in (0, 1):
                s = opened(local, 0, rank=rank, generation=gen)
                with pytest.raises(FileExistsError):
                    opened(local, 0, rank=rank, generation=gen)
                for l in range(2):
                    s.write_rows(l, 0, 0, torch.zeros(2, 3), torch.zeros(2, dtype=torch.bool))
                s.seal_capture()
                for l in range(2):
                    s.release_layer(l)
                s.close()
            assert set(local.rglob("*.bin")) == {p.resolve() for p in peer.owned.values()}
    finally:
        peer.abort()
        peer.abort()


@pytest.mark.parametrize("failure", [errno.EIO, getattr(errno, "ESTALE", 116), errno.ENOSPC,
                                     getattr(errno, "EDQUOT", 122), errno.EACCES, errno.EINVAL])
def test_capture_failure_preserves_other_files(local, failure):
    sibling = local.parent / "hf"
    sibling.write_bytes(b"user")
    s = opened(local, 1, 3, 1)
    with mock.patch.object(os, "pwritev", side_effect=OSError(failure, "injected")):
        with pytest.raises(OSError) as error:
            s.write_rows(0, 0, 1, torch.ones(1, 3), torch.ones(1, dtype=torch.bool))
        assert error.value.errno == failure
    with pytest.raises(RuntimeError, match="incomplete"):
        s.seal_capture()
    s.abort()
    assert sibling.read_bytes() == b"user" and not list(local.rglob("*.bin"))


def test_retry_eintr_and_detect_truncation(local):
    s = opened(local, 1, 3, 1)
    try:
        source = torch.ones(1, 3, dtype=torch.bfloat16)
        write = os.pwritev
        calls = [InterruptedError()]
        def interrupt(*args):
            if calls:
                raise calls.pop()
            return write(*args)
        with mock.patch.object(os, "pwritev", interrupt):
            s.write_rows(0, 0, 1, source, torch.ones(1, dtype=torch.bool))
        s.seal_capture()
        os.ftruncate(s.fds[0], 2)
        with pytest.raises(OSError, match="Incomplete"):
            s.read_rows_into(0, 0, 1, source)
        with pytest.raises(RuntimeError, match="unread"):
            s.release_layer(0)
    finally:
        s.abort()


def test_shared_capacity_and_failed_open_cleanup(local):
    with mock.patch.object(os, "statvfs", return_value=SimpleNamespace(f_bavail=100, f_frsize=1)):
        with pytest.raises(OSError) as err:
            store.DirectHiddenStore(local, "v183-test", 0, 1, 8, 2, 1, shared_bytes=256)
        assert err.value.errno == errno.ENOSPC and not local.exists()
    first = store._open_payload
    def fail_second(path):
        if path.name == "layer_1.bin":
            raise OSError(errno.EINVAL, "O_DIRECT unsupported")
        return first(path)
    with mock.patch.object(store, "_open_payload", fail_second):
        with pytest.raises(OSError, match="unsupported"):
            opened(local)
    assert not list(local.rglob("*.bin"))


def test_missing_mount_cannot_create_directory(tmp_path):
    path = tmp_path / "hidden"
    with pytest.raises(OSError):
        with store.mounted_root(path):
            pytest.fail("local filesystem accepted as NFS")
    assert not path.exists()


@pytest.mark.parametrize("fs,ok", [("nfs4 server:/x rw,vers=4.1,hard", True),
                                   ("nfs server:/x rw,vers=3,hard", False),
                                   ("ext4 /dev/sda rw", False)])
def test_check_open_parent_mount_id(fs, ok):
    info = "1 0 1:2 / / rw - nfs4 server:/x rw,vers=4.1,hard\n2 1 1:3 / /actual rw - " + fs
    def read(p):
        return "mnt_id:\t2\n" if "fdinfo" in p.parts else info
    with mock.patch.object(Path, "read_text", read):
        if ok:
            store._verify_mount(3)
        else:
            with pytest.raises(OSError):
                store._verify_mount(3)


def test_attempt_listing_and_single_registered_orphan(local, capsys):
    s = opened(local)
    try:
        other = s.directory / "unregistered"
        other.write_bytes(b"keep")
        with pytest.raises(ValueError, match="registered"):
            store.manage_orphans(local, "v183-test", str(other.resolve().relative_to(local / "v183-test")))
        store.manage_orphans(local, "v183-test")
        assert "layer_1.bin" in capsys.readouterr().out
        layer = s.owned[1]
        os.close(s.fds.pop(1))
        store.manage_orphans(local, "v183-test", layer.resolve().relative_to(local / "v183-test").as_posix())
        assert not layer.exists() and other.read_bytes() == b"keep"
        del s.owned[1]
        other.unlink()
    finally:
        s.abort()


def test_frozen_observer_file_vs_pageable(tmp_path, monkeypatch):
    """Run real F/R hooks and Step E on both stores with the same differentiable toy model."""
    import test_eu_derpo_observer_on_cpu as harness

    def load_math(name):
        full = "verl.trainer.ppo." + name
        spec = importlib.util.spec_from_file_location(full, ROOT/"verl/trainer/ppo"/(name+".py"))
        mod = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, full, mod)
        spec.loader.exec_module(mod)
        return mod

    def run(backend, weight):
        case = harness.TestEUDERPOObserver()
        case.setUp()
        try:
            obs = case.observer
            obs.cache_backend, obs.cache_root, obs.cache_attempt_id = backend, str(tmp_path/"hidden"), "v183-case"
            monkeypatch.setitem(sys.modules, "verl.utils.debug.eu_hidden_store", store)
            eu = sys.modules["verl.trainer.ppo.eu_derpo"]
            v18 = load_math("eu_derpo_v18")
            load_math("eu_derpo_v182")
            v183 = load_math("eu_derpo_v183")
            groups = torch.tensor([4, 4])
            obs.start_main_batch(case.sample_ids, groups, case.attention_mask, case.response_mask, 2, 0)
            if backend == "nfs_direct":
                assert obs._hidden_cache is None
            gamma, objectives, masks, population = [], [], [], []
            for i, sample in enumerate(case.sample_ids):
                inputs, attn = case.input_ids[i:i+1], case.attention_mask[i:i+1]
                valid, ids = case.response_mask[i:i+1], sample.view(1)
                obs.begin_main_microbatch(inputs, attn, ids, valid, 2)
                with torch.no_grad():
                    case.model(case.hidden[i])
                routes = obs.finish_main_microbatch(inputs, attn, ids, 2)
                outputs = case.model(case.hidden[i])
                z = torch.stack([(p * torch.linspace(.1, .9, 128)).sum(-1)
                                 for p, _ in outputs]).mean(0)[:2].log().view(1, 2) - 2
                behavior = z.detach() - torch.tensor([[.001, 1.]]) * (1 if i == 0 else -1)
                adv = torch.full_like(z, 1 if i == 0 else -1)
                stats = v18.cluster_statistics(z, behavior, adv, valid, routes, 128, .012)
                g = v183.schedule_coefficients(stats, routes, valid, 3, 2, 1)
                gamma.append(g)
                objectives.append(v183.objective_numerator(stats, routes, valid)/3)
                masks.append(stats.mask)
                population.append(stats.divergence[stats.directional_outward])
                obs.record_main_logprobs(ids, routes, z)
                (-(g * z).sum()/2).backward()
            native = torch.stack([r.weight.grad.clone() for r in case.model.routers])
            for r in case.model.routers:
                r.weight.main_grad = r.weight.grad.clone()
            obs.wrap_native_finalize(lambda: None)()
            teacher, squares, counts, _ = obs.finish_main_batch()
            normalized, _ = eu.normalize_group_utility(teacher, counts, groups, 1e-6, 2, 5e-7)
            stats = SimpleNamespace(count=counts, mask=torch.cat(masks))
            with mock.patch.object(case.observer_module, "_staging_chunk_rows", return_value=2):
                metrics = obs.run_router_only_step_e(case.sample_ids, stats, normalized,
                    (counts > 0).sum((1,2)), weight, 0)
            assert metrics["aux_reduce_count_by_layer"] == metrics["main_grad_add_count_by_layer"] == [1]*48
            assert metrics["step_e_max_chunk_rows"] == 2
            obs.validate_before_optimizer_step(0)
            added = torch.stack([r.weight.main_grad for r in case.model.routers]) - native
            assert native.abs().sum() > 0 and (added.abs().sum().item() > 0) == bool(weight)
            obs.finish_optimizer_step(1)
            return (teacher, squares, counts, torch.cat(masks), torch.cat(population), torch.cat(gamma),
                    torch.stack(objectives).sum(), native, added)
        finally:
            case.tearDown()

    with local_transport(), mock.patch.object(store, "WRITE_BYTES", 8):
        for weight in (0., .2):
            reference, actual = run("cpu_pageable", weight), run("nfs_direct", weight)
            assert all(torch.equal(left, right) for left, right in zip(reference, actual))
            assert not list(tmp_path.rglob("*.bin"))


@pytest.mark.parametrize("phase", ["capture", "replay"])
def test_io_error_reaches_observer_before_optimizer(local, monkeypatch, phase):
    import test_eu_derpo_observer_on_cpu as harness
    case = harness.TestEUDERPOObserver()
    case.setUp()
    try:
        obs = case.observer
        obs.cache_backend, obs.cache_root, obs.cache_attempt_id = "nfs_direct", str(local), "failed"
        monkeypatch.setitem(sys.modules, "verl.utils.debug.eu_hidden_store", store)
        syscall = "pwritev" if phase == "capture" else "preadv"
        with mock.patch.object(os, syscall, side_effect=OSError(errno.EIO, "injected")):
            case.run_main_backward()
            if phase == "capture":
                with pytest.raises(RuntimeError, match="HiddenStore seal failed") as err:
                    obs.finish_main_batch()
            else:
                _, counts, _ = case.finish_main()
                stats = SimpleNamespace(count=counts, mask=counts > 0)
                with pytest.raises(RuntimeError, match="HiddenStore replay layer 0 failed") as err:
                    obs.run_router_only_step_e(case.sample_ids, stats, counts, (counts>0).sum((1,2)), .2, 0)
        assert err.value.__cause__.errno == errno.EIO
        with pytest.raises(RuntimeError, match="optimizer step attempted before"):
            obs.validate_before_optimizer_step(0)
        assert obs._aux_reduce_count == obs._main_grad_add_count == [0]*48
    finally:
        case.tearDown()
    assert not list(local.rglob("*.bin"))


def test_peer_error_vote_rejects_successful_local_io(monkeypatch):
    import test_eu_derpo_observer_on_cpu as harness
    case = harness.TestEUDERPOObserver()
    case.setUp()
    try:
        monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
        def failed_peer(flag, op):
            assert flag.item() == 0 and op == torch.distributed.ReduceOp.MAX
            flag.fill_(1)
        monkeypatch.setattr(torch.distributed, "all_reduce", failed_peer)
        with pytest.raises(RuntimeError, match="HiddenStore replay layer 3 failed"):
            case.observer._check_store_error("replay layer 3", None)
    finally:
        case.tearDown()


def test_zero_write_and_incomplete_seal(local):
    s = opened(local, 1, 3, 1)
    try:
        source = torch.ones(1, 3, dtype=torch.bfloat16)
        valid = torch.ones(1, dtype=torch.bool)
        with mock.patch.object(os, "pwritev", return_value=0), pytest.raises(OSError, match="Incomplete"):
            s.write_rows(0, 0, 1, source, valid)
        s.write_rows(0, 0, 1, source, valid)
        os.ftruncate(s.fds[0], 2)
        with pytest.raises(OSError, match="length"):
            s.seal_capture()
    finally:
        s.abort()


def test_physical_open_flags(monkeypatch):
    for name, default in (("O_DIRECT", 0x4000), ("O_NOFOLLOW", 0x20000)):
        monkeypatch.setattr(os, name, getattr(os, name, default), raising=False)
    with mock.patch.object(os, "open", return_value=3) as op:
        assert store._open_payload("layer_0.bin") == 3
    flags = op.call_args.args[1]
    assert flags & os.O_DIRECT and flags & os.O_EXCL and flags & os.O_NOFOLLOW
    assert not flags & os.O_APPEND


def memory_sample():
    import psutil
    proc = psutil.Process()
    info = proc.memory_full_info()
    result = dict(rss=info.rss, pss=getattr(info, "pss", None))
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        for line in meminfo.read_text().splitlines():
            key, value = line.split(":", 1)
            if key in {"Dirty", "Writeback"}:
                result[key + "_bytes"] = int(value.split()[0]) * 1024
        # v2 uses the process-relative path; a cgroup namespace may expose its root here.
        result["cgroup_membership"] = Path("/proc/self/cgroup").read_text().splitlines()
        for line in result["cgroup_membership"]:
            hierarchy, controllers, relative = line.split(":", 2)
            if ".." in Path(relative).parts:
                continue
            if hierarchy == "0" and not controllers:
                usage = Path("/sys/fs/cgroup") / relative.lstrip("/") / "memory.current"
            elif "memory" in controllers.split(","):
                usage = Path("/sys/fs/cgroup/memory") / relative.lstrip("/") / "memory.usage_in_bytes"
            else:
                continue
            if usage.is_file():
                result["cgroup_usage_bytes"] = int(usage.read_text())
    return result


def profile(root, layers):
    """Run in a fresh process. OS counters identify the reference host and cache boundary."""
    import psutil
    import time
    process = psutil.Process()
    torch.set_num_threads(1)
    rows, width = 4096, 2048
    source = torch.ones(1024, width, dtype=torch.bfloat16)
    valid = torch.ones(len(source), dtype=torch.bool)
    before = memory_sample()
    rss = before["rss"]
    peak = rss
    started = time.monotonic()
    with local_transport():
        s = opened(root, rows=rows, width=width, layers=layers)
        stage = s.stage.untyped_storage().nbytes()
        try:
            for l in range(layers):
                for i in range(0, rows, len(source)):
                    s.write_rows(l, i, i + len(source), source, valid)
                    peak = max(peak, process.memory_info().rss)
            s.seal_capture()
            for l in range(layers):
                for i in range(0, rows, len(source)):
                    s.read_rows_into(l, i, i + len(source), source)
                s.release_layer(l)
            s.close()
        finally:
            s.abort()
    print(json.dumps(dict(transport="local file syscall fixture", shape=[layers, rows, width],
                          payload=rows * width * 2 * layers, write_stage=stage,
                          before_rss=rss, sampled_peak_rss=peak, peak_wset=getattr(process.memory_info(),"peak_wset",None),
                          before=before, after=memory_sample(), seconds=time.monotonic()-started)))


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as folder:
        profile(Path(folder) / "owned", int(sys.argv[1]))
