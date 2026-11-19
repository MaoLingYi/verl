"""Reproducible CPU-local syscall benchmark; the transport fixture does not measure NFS."""
import argparse
from collections import Counter
from contextlib import ExitStack
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from types import ModuleType
from unittest import mock

import psutil
import torch

from test_v183_hidden_store_on_cpu import ROOT, local_transport, memory_sample, store


BASELINE = "610e4d7bb39f5b7c9c4d6bc4bb28d6b4c9e5b6f6"
MIB = 1024**2
PRODUCTION_LOGICAL = 203601936384
PRODUCTION_RAM = 68719476736


def load_backend(baseline):
    if not baseline:
        return store
    path = "verl/utils/debug/eu_hidden_store.py"
    result = subprocess.run(
        ["git", "-c", "safe.directory=" + str(ROOT), "-C", str(ROOT), "show", BASELINE + ":" + path],
        check=True, stdout=subprocess.PIPE, encoding="utf-8",
    )
    module = ModuleType("baseline_hidden_store")
    exec(compile(result.stdout, BASELINE + ":" + path, "exec"), module.__dict__)
    return module


def profile(root, implementation, batch_mib, *, small=False):
    """Only this owner's registered payloads are removed by normal close/abort."""
    torch.set_num_threads(1)
    backend = load_backend(implementation == "baseline")
    width = 2048
    response_rows = [4096] if small else [1024, 4096, 8192, 5632, 1536, 4096]
    layers = 2 if small else 4
    rows = sum(response_rows)
    logical = rows * width * 2 * layers
    quota = (logical * PRODUCTION_RAM // PRODUCTION_LOGICAL // (width * 2)) * (width * 2)
    # Keep the production Step E mathematical chunk budget, including support and coefficients.
    replay_rows = (64 * MIB) // (width * 2 + 8 * 5)
    source_bits = torch.arange(max(response_rows) * width, dtype=torch.int32).short()
    source_bits = source_bits.view(max(response_rows), width)
    source_bits[:, 0] = torch.arange(max(response_rows), dtype=torch.int16)
    source = source_bits.view(torch.bfloat16)
    valid = torch.ones(max(response_rows), dtype=torch.bool)
    replay = torch.empty((min(rows, replay_rows), width), dtype=torch.bfloat16)
    process = psutil.Process()
    before = memory_sample()
    peak_rss = before["rss"]
    requested = {"read": Counter(), "write": Counter()}
    completed = {"read": 0, "write": 0}
    elapsed = {"read": 0., "write": 0.}
    phases = {"capture": 0., "replay": 0., "seal": 0., "release_close": 0.}
    ram_peak = stage_peak = 0
    options = {} if implementation == "baseline" else {"io_batch_bytes": batch_mib * MIB}

    def measured(operation, direction):
        def transfer(fd, buffers, offset):
            count = sum(len(buffer) for buffer in buffers)
            requested[direction][count] += 1
            start = time.perf_counter()
            result = operation(fd, buffers, offset)
            elapsed[direction] += time.perf_counter() - start
            completed[direction] += result
            return result
        return transfer

    root.mkdir(parents=True, exist_ok=True)
    with local_transport(), ExitStack() as patches:
        # Both versions use identical CPU transport; no production NFS fallback is added.
        for name in ("mounted_root", "_directory_handle", "_open_payload"):
            patches.enter_context(mock.patch.object(backend, name, getattr(store, name)))
        for name, direction in (("preadv", "read"), ("pwritev", "write")):
            patches.enter_context(mock.patch.object(os, name, measured(getattr(os, name), direction)))
        instance = backend.HybridHiddenStore(
            root / "cache", "bench", 0, 0, rows, width, layers, logical,
            ram_limit_bytes=quota, pressure=lambda: 2 << 30, **options,
        )
        total_started = time.perf_counter()
        try:
            stage_peak = instance.stage.untyped_storage().nbytes()
            cursor = 0
            for response, length in enumerate(response_rows):
                # Actual F visits all layers of one response before moving to the next response.
                for layer in range(layers):
                    source_bits[:, 1] = response * layers + layer
                    started = time.perf_counter()
                    instance.write_rows(layer, cursor, cursor + length, source[:length], valid[:length])
                    phases["capture"] += time.perf_counter() - started
                    peak_rss = max(peak_rss, process.memory_info().rss)
                cursor += length
            ram_peak = instance.ram_high_water_bytes
            started = time.perf_counter()
            instance.seal_capture()
            phases["seal"] += time.perf_counter() - started
            for layer in range(layers):
                for cursor in range(0, rows, replay_rows):
                    end = min(rows, cursor + replay_rows)
                    destination = replay[:end - cursor]
                    started = time.perf_counter()
                    instance.read_rows_into(layer, cursor, end, destination)
                    phases["replay"] += time.perf_counter() - started
                    peak_rss = max(peak_rss, process.memory_info().rss)
                    # Check raw BF16 bits, including NaN payloads, without allocating a full expected layer.
                    origin = 0
                    for response, length in enumerate(response_rows):
                        first, stop = max(cursor, origin), min(end, origin + length)
                        if first < stop:
                            source_bits[:, 1] = response * layers + layer
                            assert torch.equal(destination[first - cursor:stop - cursor].view(torch.int16),
                                               source_bits[first - origin:stop - origin])
                        origin += length
                started = time.perf_counter()
                instance.release_layer(layer)
                phases["release_close"] += time.perf_counter() - started
            started = time.perf_counter()
            instance.close()
            phases["release_close"] += time.perf_counter() - started
            assert completed["write"] == completed["read"] == logical - quota
            assert instance.spill_read_bytes == instance.spill_written_bytes == logical - quota
            assert instance.ram_hidden_bytes == 0 and not instance._resident
            assert not instance._spans or not any(instance._spans)
            metrics = dict(getattr(instance, "transfer_stats", {}))
        finally:
            instance.abort()
        wall = time.perf_counter() - total_started
    assert not list(root.rglob("*.bin"))
    info = process.memory_info()
    return dict(
        transport="CPU local buffered file syscall fixture; NFS O_DIRECT NOT_RUN",
        implementation=implementation, batch_mib=batch_mib, baseline_commit=BASELINE,
        concurrency=1, owner_rank=0, dtype="bfloat16", shape=[layers, rows, width],
        response_rows=response_rows, mathematical_chunk_rows=replay_rows,
        logical_hidden_bytes=logical, ram_quota_bytes=quota, ram_peak_bytes=ram_peak,
        production_node_cap_bytes=PRODUCTION_RAM, production_logical_bytes=PRODUCTION_LOGICAL,
        scaling_note="payload and RAM quota scaled proportionally; no 96 GiB change",
        capture_staging_bytes=stage_peak, additional_capture_staging_bytes=max(0, stage_peak - 4 * MIB),
        replay_hidden_staging_bytes=replay.untyped_storage().nbytes(),
        calls={key: sum(value.values()) for key, value in requested.items()},
        requested_sizes={key: dict(sorted(value.items())) for key, value in requested.items()},
        completed_bytes=completed, syscall_seconds=elapsed, phase_seconds=phases,
        measured_io_phase_seconds=phases["capture"] + phases["replay"],
        wall_including_integrity_seconds=wall,
        syscall_mib_s={key: completed[key] / MIB / elapsed[key] if elapsed[key] else None for key in elapsed},
        phase_mib_s={key: completed[key] / MIB / phases[phase] for key, phase in
                     (("write", "capture"), ("read", "replay"))},
        memory_before=before, memory_after=memory_sample(), sampled_peak_rss=peak_rss,
        process_peak_rss=getattr(info, "peak_wset", None), storage_metrics=metrics,
        cleanup_complete=True, bit_exact=True,
    )


def test_small_real_old_and_new_io_profile(tmp_path):
    results = [profile(tmp_path / name, name, batch, small=True) for name, batch in
               (("baseline", 4), ("new4", 4), ("new16", 16), ("new64", 64))]
    baseline = results[0]
    for row in results[1:]:
        assert row["completed_bytes"] == baseline["completed_bytes"]
        assert row["ram_peak_bytes"] == baseline["ram_peak_bytes"]
        assert row["calls"]["read"] == baseline["calls"]["read"]
    assert results[1]["calls"]["write"] == baseline["calls"]["write"]
    assert results[2]["calls"]["write"] < baseline["calls"]["write"]
    assert results[3]["calls"]["write"] <= results[2]["calls"]["write"]


def benchmark(destination, repetitions):
    destination.mkdir(parents=True, exist_ok=True)
    records = []
    variants = [("baseline", 4), ("new4", 4), ("new16", 16), ("new64", 64)]
    for repeat in range(repetitions):
        for name, batch in variants:
            root = destination / (name + "-" + str(repeat))
            result = subprocess.run(
                [sys.executable, __file__, "--one", name, "--batch-mib", str(batch), "--root", str(root)],
                check=True, stdout=subprocess.PIPE, encoding="utf-8",
            )
            row = json.loads(result.stdout.splitlines()[-1])
            records.append(row)
    median = {}
    for name, _ in variants:
        group = [row for row in records if row["implementation"] == name]
        median[name] = {key: statistics.median(row[key] for row in group)
                        for key in ("measured_io_phase_seconds", "wall_including_integrity_seconds")}
    for name, value in median.items():
        value["io_phase_speedup_vs_baseline"] = (
            median["baseline"]["measured_io_phase_seconds"] / value["measured_io_phase_seconds"]
        )
    report = dict(repetitions=repetitions, variants=median, measurements=records,
                  scope="single-owner CPU local filesystem, buffered OS cache; no NFS/GPU inference")
    output = destination / "results.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(dict(output=str(output), variants=median)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--one", choices=("baseline", "new4", "new16", "new64"))
    parser.add_argument("--batch-mib", type=int, default=4)
    parser.add_argument("--repetitions", type=int, default=3)
    args = parser.parse_args()
    if args.one:
        print(json.dumps(profile(args.root, args.one, args.batch_mib)))
    else:
        benchmark(args.root, args.repetitions)
