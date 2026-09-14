from __future__ import annotations

import ast
import sys
import types
import warnings
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.distributed.checkpoint import FileSystemWriter, load as torch_dist_load, save as torch_dist_save


SOURCE = Path(__file__).resolve().parents[3] / "verl" / "utils" / "megatron" / "dist_checkpointing.py"


def test_planned_tensor_bytes_uses_real_plan_shapes_and_dtypes():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"), filename=str(SOURCE))
    fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_planned_tensor_bytes")
    namespace = {"torch": SimpleNamespace(_utils=SimpleNamespace(_element_size=lambda dtype: dtype))}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(SOURCE), "exec"), namespace)
    plan = SimpleNamespace(
        items=[
            SimpleNamespace(tensor_data=SimpleNamespace(size=(2, 3), properties=SimpleNamespace(dtype=4))),
            SimpleNamespace(tensor_data=SimpleNamespace(size=(5,), properties=SimpleNamespace(dtype=2))),
            SimpleNamespace(tensor_data=None),
        ]
    )
    assert namespace["_planned_tensor_bytes"](plan) == (34, 24)


def test_bounded_strategy_uses_single_thread_streaming_writer_and_mcore_planner(monkeypatch):
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"), filename=str(SOURCE))
    wanted = {"_planned_tensor_bytes", "_TelemetryFileSystemWriter", "_bounded_sync_torch_dist_strategy"}
    nodes = [
        node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in wanted
    ]
    events = []

    class FileSystemWriter:
        def __init__(self, path, **kwargs):
            events.append(("writer", path, kwargs))

    class TorchDistSaveShardedStrategy:
        def __init__(self, backend, version, keep_only_main_replica=True, thread_count=2):
            self.backend = backend
            self.version = version
            self.keep_only_main_replica = keep_only_main_replica
            self.thread_count = thread_count

    class MCoreSavePlanner:
        def __init__(self, **kwargs):
            events.append(("planner", kwargs))

    fake = types.ModuleType("megatron.core.dist_checkpointing.strategies.torch")
    fake.MCoreSavePlanner = MCoreSavePlanner
    fake.TorchDistSaveShardedStrategy = TorchDistSaveShardedStrategy
    fake._replace_state_dict_keys_with_sharded_keys = lambda state, keep: (state, {}, {})
    fake.mcore_to_pyt_state_dict = lambda state, loading: state
    monkeypatch.setitem(sys.modules, fake.__name__, fake)

    namespace = {
        "FileSystemWriter": FileSystemWriter,
        "_SYNC_DCP_COPY_AHEAD_BYTES": 64 * 1024**2,
        "torch": SimpleNamespace(_utils=SimpleNamespace(_element_size=lambda _: 4)),
        "torch_dist_save": lambda state, **kwargs: events.append(("save", state, kwargs)),
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), "exec"), namespace)
    callback = lambda stage, extra=None: events.append((stage, extra))
    strategy = namespace["_bounded_sync_torch_dist_strategy"](callback)
    strategy.save({"model": "shard"}, "/checkpoint")

    assert strategy.thread_count == 1
    assert ("writer", "/checkpoint", {"thread_count": 1, "per_thread_copy_ahead": 64 * 1024**2}) in events
    assert any(event[0] == "planner" for event in events)
    assert any(event[0] == "save" for event in events)
    assert ("checkpoint_dcp_after_mcore_translation", None) in events


def test_bounded_writer_produces_a_loadable_dcp(tmp_path):
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"), filename=str(SOURCE))
    nodes = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef))
        and node.name in {"_planned_tensor_bytes", "_TelemetryFileSystemWriter"}
    ]
    namespace = {
        "FileSystemWriter": FileSystemWriter,
        "_SYNC_DCP_COPY_AHEAD_BYTES": 64 * 1024**2,
        "torch": torch,
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), "exec"), namespace)

    expected = torch.arange(32, dtype=torch.float32)
    writer = namespace["_TelemetryFileSystemWriter"](tmp_path, None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        torch_dist_save({"model": expected}, storage_writer=writer)
        restored = {"model": torch.empty_like(expected)}
        torch_dist_load(restored, checkpoint_id=tmp_path)

    assert torch.equal(restored["model"], expected)
    assert (tmp_path / ".metadata").is_file()


def test_sync_selects_bounded_strategy_while_async_keeps_mcore_default():
    source = SOURCE.read_text(encoding="utf-8")
    save = source[source.index("def save_dist_checkpointing(") : source.index("def load_dist_checkpointing(")]
    assert "if async_save" in save
    assert "else _bounded_sync_torch_dist_strategy(stage_callback)" in save
    assert "FullyParallelSaveStrategyWrapper" in save
    assert "async_sharded_save=async_save" in save
    assert '"checkpoint_dcp_before_strategy_save"' in save
    assert '"checkpoint_dcp_after_strategy_save"' in save
