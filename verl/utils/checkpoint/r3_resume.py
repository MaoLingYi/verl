"""Read-only R3 checkpoint authority checks; no MCore, Ray or CUDA imports."""
import io
import re
from pathlib import Path


def optimizer_group_steps(value):
    """Read only optimizer param_groups, never scheduler/global_step or tensor state."""
    steps = []
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "param_groups":
                if not isinstance(child, (list, tuple)) or not child:
                    raise ValueError("empty optimizer param_groups")
                for group in child:
                    if "step" not in group:
                        raise ValueError("checkpoint optimizer param_group missing authoritative step")
                    step = group["step"]
                    if hasattr(step, "item"):
                        step = step.item()
                    if isinstance(step, bool) or not isinstance(step, (int, float)) or int(step) != step:
                        raise ValueError("invalid authoritative optimizer step")
                    steps.append(int(step))
            elif key not in ("state", "param_state", "lr_scheduler", "rng_state"):
                steps.extend(optimizer_group_steps(child))
    elif isinstance(value, (list, tuple)):
        for child in value:
            steps.extend(optimizer_group_steps(child))
    return steps


def require_steps(steps, expected):
    if not steps or any(step != expected for step in steps):
        raise ValueError(f"authoritative optimizer steps {steps} != expected {expected}; no synthesis allowed")
    return steps


def check_restored_optimizer(optimizer, payload, expected):
    require_steps(optimizer_group_steps(payload), expected)
    restored = []
    for outer in getattr(optimizer, "chained_optimizers", (optimizer,)):
        inner = getattr(outer, "optimizer", outer)
        restored.extend(optimizer_group_steps({"param_groups": inner.param_groups}))
    require_steps(restored, expected)
    print(f"R3_RESTORE RESTORED_OPTIMIZER_STEP={expected} groups={len(restored)}", flush=True)


def inspect_checkpoint(checkpoint, expected=250):
    """Read only small trusted DCP BYTE_IO optimizer metadata, never model tensors."""
    import torch
    from torch.distributed.checkpoint import FileSystemReader
    from torch.distributed.checkpoint.metadata import BytesStorageMetadata

    root = Path(checkpoint).resolve(strict=True)
    if root.name != f"global_step_{expected}":
        raise ValueError("checkpoint directory step mismatch")
    dcp = root / "actor/dist_ckpt"
    if not dcp.is_dir():
        raise ValueError("missing actor/dist_ckpt")
    files = [p for p in root.rglob("*") if p.is_file()]
    sizes = [p.stat().st_size for p in files]
    if not sizes or any(size == 0 for size in sizes):
        raise ValueError("empty checkpoint or zero-byte files")
    reader = FileSystemReader(str(dcp))
    metadata = reader.read_metadata()
    steps, entries = [], []
    split_groups, step_groups = set(), set()
    for index, storage in metadata.storage_data.items():
        key = index.fqn
        group = re.match(r"(.*\.param_groups\.\d+)\.(.+)$", key)
        if group and "optimizer" in key:
            split_groups.add(group[1])
        if not isinstance(metadata.state_dict_metadata[key], BytesStorageMetadata):
            continue
        if "optimizer" not in key or "param_state" in key:
            continue
        if group and group[2] != "step":
            continue
        path = (dcp / storage.relative_path).resolve(strict=True)
        if dcp.resolve() not in path.parents or not 0 < storage.length <= 64 * 1024**2:
            raise ValueError("invalid optimizer BYTE_IO storage extent")
        with path.open("rb") as stream:
            stream.seek(storage.offset)
            raw = stream.read(storage.length)
        if len(raw) != storage.length:
            raise ValueError("truncated optimizer BYTE_IO")
        value = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False)
        if key.endswith("param_groups"):
            value = {"param_groups": value}
        elif group and group[2] == "step":
            value = {"param_groups": [{"step": value}]}
            step_groups.add(group[1])
        elif re.search(r"\.param_groups\.\d+$", key):
            value = {"param_groups": [value]}
        elif "param_groups" in key:
            raise ValueError(f"unsupported split param_groups BYTE_IO layout: {key}")
        found = optimizer_group_steps(value)
        steps.extend(found)
        entries.append({"key": key, "steps": found})
    if split_groups != step_groups:
        raise ValueError("optimizer param_groups missing checkpoint step: " + repr(split_groups - step_groups))
    require_steps(steps, expected)
    return {"checkpoint": str(root), "file_count": len(files), "bytes": sum(sizes),
            "zero_byte_files": 0, "optimizer_entries": entries,
            "HDO_AUTHORITATIVE_OPTIMIZER_STEP": expected,
            "TRAIN_DATA_ORDER_RESTORE": "UNKNOWN" if (root / "data.pt").is_file() else "FAIL",
            "RNG_RESTORE": "UNKNOWN", "MODEL_OPTIMIZER_RESTORE": "NOT_RUN",
            "EXACT_TRAJECTORY_CONTINUITY": "NOT_PROVEN"}
