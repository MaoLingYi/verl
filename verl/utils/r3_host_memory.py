"""Opt-in phase telemetry: no collectives, CUDA API, tensor copy or retained state."""
import json
import logging
import os
import socket
import sys
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from urllib.request import urlopen


def host_phase(phase):
    def decorate(function):
        @wraps(function)
        def wrapped(self, *args, **kwargs):
            if os.getenv("R3_HOST_RAM_DIAG") != "1":
                return function(self, *args, **kwargs)
            step = kwargs.get("global_step", getattr(self, "global_steps", None))
            if phase == "checkpoint" and len(args) >= 3:
                step = args[2]
            if args and hasattr(args[0], "meta_info"):
                step = args[0].meta_info.get("global_steps", step)
            if phase == "restore":
                step = os.getenv("R3_EXPECTED_RESUME_STEP")
            optimizer, model = getattr(self, "actor_optimizer", None), getattr(self, "actor_module", None)
            log_host_phase("before_" + phase, step, optimizer, model)
            try:
                return function(self, *args, **kwargs)
            finally:
                log_host_phase("after_" + phase, step, optimizer, model)
        return wrapped
    return decorate


def proc_values(path):
    try:
        return {parts[0].rstrip(":"): int(parts[1]) * 1024
                for line in Path(path).read_text().splitlines()
                if len(parts := line.split()) >= 3 and parts[2] == "kB"}
    except (OSError, ValueError):
        return {}


def log_host_phase(phase, step=None, optimizer=None, model=None):
    if os.getenv("R3_HOST_RAM_DIAG") != "1":
        return None
    values = {"timestamp": datetime.now(timezone.utc).isoformat(), "hostname": socket.gethostname(),
              "pid": os.getpid(), "rank": os.getenv("RANK"), "step": step, "phase": phase,
              "errors": {}, "ray_object_store": "UNKNOWN", "ray_headroom_bytes": None}

    def read(name, fn):
        try:
            values[name] = fn()
        except Exception as exc:
            values["errors"][name] = repr(exc)

    mem = proc_values("/proc/meminfo")
    status, rollup = proc_values("/proc/self/status"), proc_values("/proc/self/smaps_rollup")
    values.update(MemTotal=mem.get("MemTotal"), MemAvailable=mem.get("MemAvailable"),
                  RSS=status.get("VmRSS"), PSS=rollup.get("Pss"))
    torch = sys.modules.get("torch")
    if torch is not None and torch.distributed.is_initialized():
        read("rank", torch.distributed.get_rank)
    total = mem.get("MemTotal")
    if total:
        values["node_memory_used_MemAvailable_basis"] = total - mem.get("MemAvailable", 0)
        # Ray Linux accounting subtracts free, buffers, cached and SReclaimable.
        used = total - sum(mem.get(k, 0) for k in ("MemFree", "Buffers", "Cached", "SReclaimable"))
        values["ray_linux_used_estimate"] = used
        values["ray_95pct_headroom_estimate"] = int(total * .95) - used
    # A finite cgroup limit can change Ray accounting; estimates are diagnostic, never a guard.
    for path in ("/sys/fs/cgroup/memory.current", "/sys/fs/cgroup/memory.max",
                 "/sys/fs/cgroup/memory/memory.usage_in_bytes", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        if Path(path).is_file():
            read(path, lambda path=path: Path(path).read_text().strip())
    ray = sys.modules.get("ray")
    if ray is not None:
        def ray_metrics():
            node = ray._private.worker.global_worker.node
            port = getattr(node, "metrics_export_port", None) or node.address_info.get("metrics_export_port")
            if not port:
                discovery = Path(node.get_session_dir_path()).parent / "prom_metrics_service_discovery.json"
                groups = json.loads(discovery.read_text())
                local_ip = node.address_info.get("node_ip_address")
                targets = [target for group in groups for target in group.get("targets", [])
                           if target.rsplit(":", 1)[0] in (local_ip, "127.0.0.1")]
                if len(targets) == 1:
                    port = int(targets[0].rsplit(":", 1)[1])
            if not port:
                return "UNKNOWN: local Ray metrics port unavailable"
            with urlopen(f"http://127.0.0.1:{int(port)}/metrics", timeout=.2) as response:
                text = response.read(2 * 1024**2).decode()
            return [line for line in text.splitlines()
                    if line.startswith("ray_object_store_memory") and not line.startswith("#")]
        read("ray_object_store", ray_metrics)
    if optimizer is not None:
        def optimizer_summary():
            from verl.utils.memory_utils import estimate_hdo_memory, estimate_optimizer_phase_offload_memory
            return {**estimate_hdo_memory(optimizer), **estimate_optimizer_phase_offload_memory(optimizer)}
        read("optimizer", optimizer_summary)
    if model is not None:
        def backing():
            from verl.utils.megatron_utils import megatron_model_cpu_data_bytes
            return megatron_model_cpu_data_bytes(model)
        read("model_cpu_backing_bytes", backing)
    try:
        logging.getLogger(__name__).warning("R3_HOST_RAM %s", json.dumps(values, sort_keys=True))
    except Exception:
        pass
    return values
