# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import ctypes
import gc
import inspect
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

from verl.utils.device import get_torch_device, is_cuda_available

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

_GIB = 1024**3


def _parse_proc_kib(text: str) -> dict[str, int]:
    """Parse kB-valued /proc records without adding a process dependency."""
    values = {}
    for line in text.splitlines():
        fields = line.split()
        if len(fields) >= 2 and fields[0].endswith(":"):
            try:
                values[fields[0][:-1]] = int(fields[1]) * 1024
            except ValueError:
                pass
    return values


def _read_proc_kib(path: str) -> dict[str, int]:
    try:
        with open(path, encoding="ascii") as stream:
            return _parse_proc_kib(stream.read())
    except OSError:
        return {}


def best_effort_release_host_allocator():
    """Release unreachable Python objects and free glibc heap pages in this process."""
    collected = gc.collect()
    attempted = sys.platform.startswith("linux")
    rc = None
    if attempted:
        try:
            malloc_trim = ctypes.CDLL("libc.so.6").malloc_trim
            malloc_trim.argtypes = [ctypes.c_size_t]
            malloc_trim.restype = ctypes.c_int
            rc = int(malloc_trim(0))
        except (AttributeError, OSError) as error:
            logger.warning("EU-DERPO malloc_trim unavailable: %r", error)
    return {"gc_collected": collected, "malloc_trim_attempted": int(attempted), "malloc_trim_rc": rc}


def _host_reclaim_snapshot(ray_limit):
    status = _read_proc_kib("/proc/self/status")
    rollup = _read_proc_kib("/proc/self/smaps_rollup")
    meminfo = _read_proc_kib("/proc/meminfo")
    rss = status.get("VmRSS", rollup.get("Rss"))
    available = meminfo.get("MemAvailable")
    if sys.platform == "win32":
        import psutil

        rss = psutil.Process().memory_info().rss
        available = psutil.virtual_memory().available

    pinned = {}
    if torch.cuda.is_initialized():
        host_memory_stats = getattr(torch.cuda.memory, "host_memory_stats", None)
        if host_memory_stats is not None:
            try:
                pinned = host_memory_stats()
            except (AttributeError, RuntimeError) as exc:
                logger.warning("CUDA host allocator counters unavailable: %r", exc)

    ray_used = ray_total = None
    ray = sys.modules.get("ray")
    if ray is not None and ray.is_initialized():
        try:
            from ray._private.utils import get_system_memory, get_used_memory

            ray_used = get_used_memory()
            ray_total = get_system_memory()
        except (ImportError, AttributeError, OSError, ValueError, RuntimeError) as exc:
            logger.warning("Ray Python memory counters unavailable: %r", exc)
    # Only the existing raylet probe supplies a limit verified against the physical
    # Linux accounting scope. The Python working set can follow a different cgroup.
    threshold, total = ray_limit if ray_limit is not None else (None, None)
    return {
        "rss_bytes": rss,
        "pss_bytes": rollup.get("Pss"),
        "rss_anon_bytes": status.get("RssAnon"),
        "mem_available_bytes": available,
        "pinned_allocated_bytes": pinned.get("allocated_bytes.current"),
        "pinned_reserved_bytes": pinned.get("reserved_bytes.current"),
        "ray_python_used_bytes": ray_used,
        "ray_python_total_bytes": ray_total,
        "raylet_linux_used_bytes": total - available if total is not None and available is not None else None,
        "raylet_threshold_bytes": threshold,
        "raylet_total_bytes": total,
    }


def post_step_e_host_reclaim(*, generation, hidden_storage, ray_limit=None):
    """Per-actor measurements and reclaim after Step E releases its final replay buffers."""
    before = _host_reclaim_snapshot(ray_limit)
    start = time.perf_counter()
    reclaimed = best_effort_release_host_allocator()
    elapsed = time.perf_counter() - start
    after = _host_reclaim_snapshot(ray_limit)
    rss_drop = available_gain = None
    if before["rss_bytes"] is not None and after["rss_bytes"] is not None:
        rss_drop = before["rss_bytes"] - after["rss_bytes"]
    if before["mem_available_bytes"] is not None and after["mem_available_bytes"] is not None:
        available_gain = after["mem_available_bytes"] - before["mem_available_bytes"]
    result = {
        "rank": torch.distributed.get_rank() if torch.distributed.is_initialized() else None,
        "pid": os.getpid(),
        "generation": generation,
        "hidden_storage": hidden_storage,
        "before": before,
        "after": after,
        "rss_drop_bytes": rss_drop,
        "available_gain_bytes": available_gain,
        "gc_collected": reclaimed["gc_collected"],
        "trim_supported": reclaimed["malloc_trim_rc"] is not None,
        "trim_rc": reclaimed["malloc_trim_rc"],
        "elapsed_seconds": elapsed,
    }
    logger.warning("EU-DERPO post_step_e_host_reclaim %s", result)
    return result


def _read_cgroup_memory():
    for current_path, max_path in (
        ("/sys/fs/cgroup/memory.current", "/sys/fs/cgroup/memory.max"),
        ("/sys/fs/cgroup/memory/memory.usage_in_bytes", "/sys/fs/cgroup/memory/memory.limit_in_bytes"),
    ):
        try:
            current = int(Path(current_path).read_text(encoding="ascii").strip())
            maximum_text = Path(max_path).read_text(encoding="ascii").strip()
            maximum = None if maximum_text == "max" else int(maximum_text)
            return current, maximum
        except (OSError, ValueError):
            continue
    return None, None


def _storage_key_and_bytes(tensor: torch.Tensor):
    storage = tensor.untyped_storage()
    return (tensor.device.type, tensor.device.index, storage.data_ptr()), storage.nbytes()


def estimate_hdo_memory(optimizer) -> dict[str, int]:
    """Estimate materialized and first-step HDO tensor bytes from its real groups."""
    result = {
        "hdo_cpu_master_bytes": 0,
        "hdo_cpu_exp_avg_bytes": 0,
        "hdo_cpu_exp_avg_sq_bytes": 0,
        "hdo_cpu_grad_bytes": 0,
        "hdo_cpu_other_bytes": 0,
        "hdo_gpu_master_bytes": 0,
        "hdo_gpu_exp_avg_bytes": 0,
        "hdo_gpu_exp_avg_sq_bytes": 0,
        "hdo_gpu_other_bytes": 0,
        "hdo_gpu_materialized_bytes": 0,
        "hdo_cpu_estimated_bytes": 0,
        "hdo_gpu_estimated_bytes": 0,
        "hdo_full_cpu_estimated_bytes": 0,
        "hdo_partial_cpu_savings_estimated_bytes": 0,
        "hdo_gpu_incremental_state_estimated_bytes": 0,
        "hdo_cpu_param_numel": 0,
        "hdo_gpu_param_numel": 0,
        "hdo_realized_offload_fraction": 0.0,
    }
    outers = getattr(optimizer, "chained_optimizers", (optimizer,))
    seen: set[tuple] = set()

    def add(name, tensor):
        if not isinstance(tensor, torch.Tensor):
            return
        key, size = _storage_key_and_bytes(tensor)
        if key not in seen:
            seen.add(key)
            result[name] += size

    for outer in outers:
        hdo = getattr(outer, "optimizer", None)
        if not all(hasattr(hdo, attr) for attr in ("sub_optimizers", "param_to_inner_param")):
            continue
        inner_params = set(hdo.param_to_inner_param.values())
        for param in inner_params:
            device = "gpu" if param.device.type == "cuda" else param.device.type
            add(f"hdo_{device}_master_bytes", param)
        for param in getattr(hdo, "cpu_copy_map_grad", {}).values():
            add("hdo_cpu_grad_bytes", param)
        for sub_optimizer in hdo.sub_optimizers:
            for state in sub_optimizer.state.values():
                for name, tensor in state.items():
                    if name == "master_param":
                        continue
                    device = "gpu" if isinstance(tensor, torch.Tensor) and tensor.device.type == "cuda" else "cpu"
                    field = name if name in ("exp_avg", "exp_avg_sq") else "other"
                    add(f"hdo_{device}_{field}_bytes", tensor)

        # CPUAdam/FusedAdam lazily create FP32 grad and two FP32 moments. Use the
        # actual selected inner params, so tensor-boundary fraction overshoot is included.
        cpu_selected = set(getattr(hdo, "gpu_params_map_cpu_copy", {}).values())
        cpu_numel = sum(param.numel() for param in cpu_selected)
        gpu_numel = sum(param.numel() for param in inner_params - cpu_selected)
        result["hdo_cpu_param_numel"] += cpu_numel
        result["hdo_gpu_param_numel"] += gpu_numel
        result["hdo_cpu_estimated_bytes"] += cpu_numel * 16
        result["hdo_gpu_estimated_bytes"] += gpu_numel * 12
    total_numel = result["hdo_cpu_param_numel"] + result["hdo_gpu_param_numel"]
    if total_numel:
        result["hdo_realized_offload_fraction"] = result["hdo_cpu_param_numel"] / total_numel
        result["hdo_full_cpu_estimated_bytes"] = total_numel * 16
        result["hdo_partial_cpu_savings_estimated_bytes"] = result["hdo_gpu_param_numel"] * 16
        result["hdo_gpu_incremental_state_estimated_bytes"] = result["hdo_gpu_param_numel"] * 8
    result["hdo_gpu_materialized_bytes"] = sum(
        result[name]
        for name in (
            "hdo_gpu_master_bytes",
            "hdo_gpu_exp_avg_bytes",
            "hdo_gpu_exp_avg_sq_bytes",
            "hdo_gpu_other_bytes",
        )
    )
    return result


def estimate_optimizer_phase_offload_memory(optimizer) -> dict[str, int]:
    """Count materialized CUDA storages moved by offload_megatron_optimizer()."""
    result = {
        "optimizer_phase_cuda_copy_param_bytes": 0,
        "optimizer_phase_cuda_state_bytes": 0,
        "optimizer_phase_cuda_total_bytes": 0,
    }
    outers = getattr(optimizer, "chained_optimizers", (optimizer,))
    seen: set[tuple] = set()

    def add(name, tensor):
        if not isinstance(tensor, torch.Tensor) or tensor.device.type != "cuda":
            return
        key, size = _storage_key_and_bytes(tensor)
        if key not in seen:
            seen.add(key)
            result[name] += size

    for outer in outers:
        for group in getattr(outer, "shard_fp32_from_float16_groups", ()):
            for tensor in group:
                add("optimizer_phase_cuda_copy_param_bytes", tensor)
        inner = getattr(outer, "optimizer", None)
        optimizers = getattr(inner, "sub_optimizers", (inner,))
        for sub_optimizer in optimizers:
            if sub_optimizer is None:
                continue
            for state in getattr(sub_optimizer, "state", {}).values():
                for tensor in state.values():
                    add("optimizer_phase_cuda_state_bytes", tensor)
    result["optimizer_phase_cuda_total_bytes"] = (
        result["optimizer_phase_cuda_copy_param_bytes"]
        + result["optimizer_phase_cuda_state_bytes"]
    )
    return result


def log_eu_derpo_memory(
    stage: str,
    optimizer=None,
    phase_model_cpu_backing_bytes: int = 0,
    extra=None,
) -> dict:
    """Emit one non-collective process/node/GPU memory record per caller rank."""
    status = _read_proc_kib("/proc/self/status")
    rollup = _read_proc_kib("/proc/self/smaps_rollup")
    meminfo = _read_proc_kib("/proc/meminfo")
    cgroup_current, cgroup_max = _read_cgroup_memory()
    device = get_torch_device()
    gpu = device.is_available()
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    cuda_free, cuda_total = device.mem_get_info() if gpu else (0, 0)
    values = {
        "stage": stage,
        "rank": rank,
        "rss_gib": status.get("VmRSS", rollup.get("Rss", 0)) / _GIB,
        "pss_gib": rollup.get("Pss", status.get("VmRSS", 0)) / _GIB,
        "rss_anon_gib": status.get("RssAnon", 0) / _GIB,
        "rss_file_gib": status.get("RssFile", 0) / _GIB,
        "private_clean_gib": rollup.get("Private_Clean", 0) / _GIB,
        "private_dirty_gib": rollup.get("Private_Dirty", 0) / _GIB,
        "mem_available_gib": meminfo.get("MemAvailable", 0) / _GIB,
        "cgroup_current_gib": None if cgroup_current is None else cgroup_current / _GIB,
        "cgroup_max_gib": None if cgroup_max is None else cgroup_max / _GIB,
        "cuda_allocated_gib": device.memory_allocated() / _GIB if gpu else 0.0,
        "cuda_reserved_gib": device.memory_reserved() / _GIB if gpu else 0.0,
        "cuda_max_allocated_gib": device.max_memory_allocated() / _GIB if gpu else 0.0,
        "cuda_max_reserved_gib": device.max_memory_reserved() / _GIB if gpu else 0.0,
        "cuda_free_gib": cuda_free / _GIB,
        "cuda_total_gib": cuda_total / _GIB,
        "phase_model_cpu_backing_gib": phase_model_cpu_backing_bytes / _GIB,
    }
    values.update(
        {key.replace("_bytes", "_gib"): value / _GIB if key.endswith("_bytes") else value
         for key, value in estimate_hdo_memory(optimizer).items()}
    )
    values.update(
        {key.replace("_bytes", "_gib"): value / _GIB
         for key, value in estimate_optimizer_phase_offload_memory(optimizer).items()}
    )
    if extra:
        values.update(extra)
    logger.warning("EU-DERPO memory: %s", " ".join(f"{key}={value}" for key, value in values.items()))
    return values


def aggressive_empty_cache(force_sync: bool = True, max_retries: int = 3) -> None:
    """
    More aggressive GPU memory cleanup function, tries to release PyTorch reserved
    but unallocated memory.

    Args:
        force_sync: Whether to force device synchronization
        max_retries: Maximum number of retries
    """
    device = get_torch_device()
    if not device.is_available():
        return

    for attempt in range(max_retries):
        # Record memory status before cleanup
        before_reserved = device.memory_reserved()
        before_allocated = device.memory_allocated()

        # Run garbage collection
        gc.collect()

        # Clear PyTorch cache
        device.empty_cache()

        # Force synchronization (optional)
        if force_sync:
            device.synchronize()

        # Record memory status after cleanup
        after_reserved = device.memory_reserved()
        after_allocated = device.memory_allocated()

        # Calculate freed memory
        reserved_freed = before_reserved - after_reserved
        allocated_freed = before_allocated - after_allocated

        logger.info(
            f"Memory cleanup attempt {attempt + 1}: Freed {reserved_freed / 1024**3:.2f} GB reserved, "
            f"{allocated_freed / 1024**3:.2f} GB allocated"
        )

        # Stop retrying if little memory was freed
        if reserved_freed < 1024**3:  # less than 1GB
            break


def reset_memory_stats() -> None:
    """Reset GPU memory statistics"""
    if get_torch_device().is_available():
        device = get_torch_device()
        device.reset_peak_memory_stats()
        device.reset_accumulated_memory_stats()


def get_memory_info() -> dict:
    """Get detailed GPU memory information"""
    if not get_torch_device().is_available():
        return {}

    device = get_torch_device()
    device_id = device.current_device()

    return {
        "total_memory_gb": device.get_device_properties(device_id).total_memory / 1024**3,
        "reserved_memory_gb": device.memory_reserved() / 1024**3,
        "allocated_memory_gb": device.memory_allocated() / 1024**3,
        "cached_memory_gb": (device.memory_reserved() - device.memory_allocated()) / 1024**3,
        "max_memory_allocated_gb": device.max_memory_allocated() / 1024**3,
        "max_memory_reserved_gb": device.max_memory_reserved() / 1024**3,
    }


def log_memory_usage(stage: str = "current") -> None:
    """Log GPU memory usage"""
    if not get_torch_device().is_available():
        return

    info = get_memory_info()
    logger.info(
        f"Memory usage [{stage}]: "
        f"Total: {info['total_memory_gb']:.2f} GB, "
        f"Allocated: {info['allocated_memory_gb']:.2f} GB, "
        f"Reserved: {info['reserved_memory_gb']:.2f} GB, "
        f"Cached: {info['cached_memory_gb']:.2f} GB"
    )


def optimize_memory_for_inference() -> None:
    """Optimize GPU memory usage for inference"""
    if not get_torch_device().is_available():
        return

    # Set a more aggressive memory allocation policy
    get_torch_device().set_per_process_memory_fraction(0.95)  # Use 95% of GPU memory

    # Clear cache
    aggressive_empty_cache(force_sync=True)

    logger.info("Optimized GPU memory usage for inference")


def optimize_memory_for_training() -> None:
    """Optimize GPU memory usage for training"""
    if not get_torch_device().is_available():
        return

    # Set a moderate memory allocation policy
    get_torch_device().set_per_process_memory_fraction(0.9)  # Use 90% of GPU memory

    # Clear cache
    aggressive_empty_cache(force_sync=False)

    logger.info("Optimized GPU memory usage for training")


def enable_memory_visualize(
    trace_alloc_max_entries: int = 200_000,
    stack_depth: int = 32,
    context: str = "all",
    stacks: str = "all",
    devices=None,
    record_context: bool = True,
):
    """
    Enables memory history recording for CUDA allocations. This function
    should be called before any large-scale CUDA allocations. For DDP or
    multi-process setups, it must be called on each rank.

    Args:
        trace_alloc_max_entries (int): Maximum number of allocation entries
            to record.
        stack_depth (int): The depth of the call stack to capture for each
            allocation. (Supported by some PyTorch versions).
        context (str): The type of memory events to record.
            'alloc': records only allocation events.
            'state': records memory state changes.
            'all': records both.
        stacks (str): The type of call stacks to record.
            'python': records Python stacks.
            'cpp': records C++ stacks (available in some versions).
            'all': records both.
        devices (Union[int, list[int], None]): The device for which to enable
            memory history. `None` enables it for the current default device.
        record_context (bool): Whether to record context information for
            allocations. Required by older PyTorch versions.
    """
    # Memory history recording is CUDA-specific functionality
    if not is_cuda_available:
        logger.warning("[memory_visualize] Memory history recording is only available on CUDA devices")
        return

    f = get_torch_device().memory._record_memory_history
    params = set(inspect.signature(f).parameters.keys())

    def _one_call(dev_kw=None):
        kwargs = {}
        if "context" in params:
            kwargs["context"] = context
        if "stacks" in params:
            kwargs["stacks"] = stacks
        if "max_entries" in params:
            kwargs["max_entries"] = trace_alloc_max_entries
        elif "trace_alloc_max_entries" in params:
            kwargs["trace_alloc_max_entries"] = trace_alloc_max_entries
        if "stack_depth" in params:
            kwargs["stack_depth"] = stack_depth
        if dev_kw is not None:
            if "device" in params:
                kwargs["device"] = dev_kw
            elif "devices" in params:
                kwargs["devices"] = dev_kw if isinstance(dev_kw, list) else [dev_kw]
        if "record_context" in params:
            kwargs["record_context"] = record_context

        try:
            f(**kwargs)
            return "native", kwargs
        except TypeError:
            try:
                if "trace_alloc_max_entries" in params and "record_context" in params:
                    f(enabled=True, trace_alloc_max_entries=trace_alloc_max_entries, record_context=True)
                    return "legacy", {
                        "enabled": True,
                        "trace_alloc_max_entries": trace_alloc_max_entries,
                        "record_context": True,
                    }
                else:
                    f(enabled=True)
                    return "legacy-min", {"enabled": True}
            except Exception:
                raise

    if devices is None or isinstance(devices, str | int | torch.device):
        mode, used = _one_call(devices if devices is not None else None)
    else:
        mode, used = "multi-device", {}
        for d in list(devices):
            _mode, _used = _one_call(d)
            used[f"dev{d}"] = _used

    device = get_torch_device()
    if device.is_available():
        device.reset_peak_memory_stats()
        device.synchronize()

    rank = int(os.environ.get("RANK", "0") or 0)
    logger.info(f"[memory_visualize][rank {rank}] recording enabled ({mode}); args={used}")


class MemorySnapshotSampler:
    """
    A utility class that dumps GPU memory snapshots.
    This is useful for monitoring memory usage over a long-running process.

    The dumped files can be visualized with https://docs.pytorch.org/memory_viz

    Args:
        out_dir (str): The directory where the snapshots will be saved.
        tag (str): A tag for the snapshot filenames.
    """

    def __init__(self, out_dir: str = "./mem_snapshots", tag: str = "periodic"):
        self.out_dir = out_dir
        self.tag = tag

    def dump_memory_snapshot(self, out_dir: str = "./mem_snapshots", tag: str = "snapshot", sub_dir: str = None):
        """
        Generates a memory snapshot and saves it as a pickle file in a specified directory.
        The files are organized by timestamp in subdirectories, with all ranks' files
        placed in the same timestamp subdirectory.

        Args:
            out_dir (str): The directory where the snapshot file will be saved.
                The directory is created if it does not exist.
            tag (str): A string tag to prepend to the filename for easier identification.
            sub_dir (str): A subdirectory to place the snapshot file in.
        """
        if sub_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M")
            out_path = Path(out_dir) / timestamp
        else:
            out_path = Path(out_dir) / sub_dir
        out_path.mkdir(parents=True, exist_ok=True)

        # get the GPU rank on the current process
        rank = os.environ.get("RANK", "0")
        pid = os.getpid()
        # todo(chenyang): check wether we need to sync all ranks before dump
        fname = f"{tag}_rank{rank}_pid{pid}.pickle"
        path = out_path / fname

        device = get_torch_device()
        if not device.is_available():
            logger.warning("[memory_visualize] is only available on CUDA devices.")
            return
        try:
            device.synchronize()
            # Memory snapshot is CUDA-specific functionality
            device.memory._dump_snapshot(str(path))
            logger.info(f"[memory_visualize] dumped: {path}")
        except Exception as e:
            logger.info(f"[memory_visualize][warn] dump failed: {e}")
