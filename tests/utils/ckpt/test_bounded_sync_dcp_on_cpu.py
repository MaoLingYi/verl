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
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in {"_planned_item_bytes", "_planned_tensor_bytes"}
    ]
    namespace = {"torch": SimpleNamespace(_utils=SimpleNamespace(_element_size=lambda dtype: dtype))}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(SOURCE), "exec"), namespace)
    items = [
        SimpleNamespace(tensor_data=SimpleNamespace(size=(2, 3), properties=SimpleNamespace(dtype=4))),
        SimpleNamespace(tensor_data=SimpleNamespace(size=(5,), properties=SimpleNamespace(dtype=2))),
        SimpleNamespace(tensor_data=None),
    ]
    assert namespace["_planned_tensor_bytes"](items) == (34, 24)


def test_file_buckets_respect_cap_and_isolate_oversize_tensor():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"), filename=str(SOURCE))
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"_planned_item_bytes", "_bounded_file_buckets"}
    ]
    namespace = {"torch": SimpleNamespace(_utils=SimpleNamespace(_element_size=lambda dtype: dtype))}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(SOURCE), "exec"), namespace)
    items = [
        SimpleNamespace(name=name, tensor_data=SimpleNamespace(size=(size,), properties=SimpleNamespace(dtype=1)))
        for name, size in (("a", 60), ("b", 40), ("c", 70), ("large", 250), ("d", 30))
    ]
    buckets = namespace["_bounded_file_buckets"](items, 100)

    assert [[item.name for item in bucket] for bucket in buckets] == [
        ["a", "b"], ["c"], ["large"], ["d"]
    ]
    for bucket in buckets:
        size = sum(item.tensor_data.size[0] for item in bucket)
        assert size <= 100 or (len(bucket) == 1 and size > 100)


def test_writer_finishes_each_file_bucket_before_starting_the_next():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"), filename=str(SOURCE))
    wanted = {
        "_planned_item_bytes", "_planned_tensor_bytes", "_bounded_file_buckets",
        "_TelemetryFileSystemWriter",
    }
    nodes = [
        node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in wanted
    ]
    writes = []

    class Future:
        def set_result(self, value):
            self.value = value

        def wait(self):
            return self.value

    class FileSystemWriter:
        def __init__(self, path, **kwargs):
            self.path = path
            self.fs = SimpleNamespace(concat_path=lambda root, name: f"{root}/{name}")

        def _write_data(self, planner, file_queue):
            entry = file_queue.get_nowait()
            assert file_queue.empty()
            writes.append([item.name for item in entry[2]])
            result = Future()
            result.set_result([entry[1]])
            return result

    fake_torch = SimpleNamespace(
        _utils=SimpleNamespace(_element_size=lambda dtype: dtype),
        futures=SimpleNamespace(Future=Future),
    )
    namespace = {
        "FileSystemWriter": FileSystemWriter,
        "SerializationFormat": SimpleNamespace(TORCH_SAVE="torch_save"),
        "_SYNC_DCP_COPY_AHEAD_BYTES": 64,
        "_SYNC_DCP_BUCKET_BYTES": 100,
        "queue": __import__("queue"),
        "torch": fake_torch,
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), "exec"), namespace)
    items = [
        SimpleNamespace(name=name, tensor_data=SimpleNamespace(size=(size,), properties=SimpleNamespace(dtype=1)))
        for name, size in (("a", 60), ("b", 40), ("c", 70), ("large", 250), ("d", 30))
    ]
    events = []
    writer = namespace["_TelemetryFileSystemWriter"]("/checkpoint", lambda stage, extra: events.append((stage, extra)))
    result = writer.write_data(
        SimpleNamespace(items=items, storage_data=SimpleNamespace(prefix="__0_")),
        planner=object(),
    )

    assert writes == [["a", "b"], ["c"], ["large"], ["d"]]
    assert result.wait() == ["__0_0.distcp", "__0_1.distcp", "__0_2.distcp", "__0_3.distcp"]
    before = [extra for stage, extra in events if stage == "checkpoint_dcp_before_bucket_write"]
    after = [extra for stage, extra in events if stage == "checkpoint_dcp_after_bucket_write"]
    assert [item["checkpoint_dcp_bucket_index"] for item in before] == [0, 1, 2, 3]
    assert before == after


def test_bounded_strategy_uses_single_thread_streaming_writer_and_mcore_planner(monkeypatch):
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"), filename=str(SOURCE))
    wanted = {
        "_planned_item_bytes", "_planned_tensor_bytes", "_bounded_file_buckets",
        "_TelemetryFileSystemWriter", "_bounded_sync_torch_dist_strategy",
    }
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
        "SerializationFormat": SimpleNamespace(TORCH_SAVE="torch_save"),
        "_SYNC_DCP_COPY_AHEAD_BYTES": 64 * 1024**2,
        "_SYNC_DCP_BUCKET_BYTES": 256 * 1024**2,
        "queue": __import__("queue"),
        "torch": SimpleNamespace(_utils=SimpleNamespace(_element_size=lambda _: 4)),
        "torch_dist_save": lambda state, **kwargs: events.append(("save", state, kwargs)),
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), "exec"), namespace)
    callback = lambda stage, extra=None: events.append((stage, extra))
    strategy = namespace["_bounded_sync_torch_dist_strategy"](callback)
    strategy.save({"model": "shard"}, "/checkpoint")

    assert strategy.thread_count == 1
    assert (
        "writer", "/checkpoint",
        {
            "single_file_per_rank": False,
            "thread_count": 1,
            "per_thread_copy_ahead": 64 * 1024**2,
        },
    ) in events
    assert any(event[0] == "planner" for event in events)
    assert any(event[0] == "save" for event in events)
    assert ("checkpoint_dcp_after_mcore_translation", None) in events


def test_bounded_writer_produces_a_loadable_dcp(tmp_path):
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"), filename=str(SOURCE))
    nodes = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef))
        and node.name in {
            "_planned_item_bytes", "_planned_tensor_bytes", "_bounded_file_buckets",
            "_TelemetryFileSystemWriter",
        }
    ]
    namespace = {
        "FileSystemWriter": FileSystemWriter,
        "SerializationFormat": SimpleNamespace(TORCH_SAVE="torch_save"),
        "_SYNC_DCP_COPY_AHEAD_BYTES": 64 * 1024**2,
        "_SYNC_DCP_BUCKET_BYTES": 20,
        "queue": __import__("queue"),
        "torch": torch,
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), "exec"), namespace)

    expected = {
        "large": torch.arange(32, dtype=torch.float32),
        "small_a": torch.arange(4, dtype=torch.float32),
        "small_b": torch.arange(4, dtype=torch.float32) + 10,
    }
    writer = namespace["_TelemetryFileSystemWriter"](tmp_path, None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        torch_dist_save(expected, storage_writer=writer)
        restored = {key: torch.empty_like(value) for key, value in expected.items()}
        torch_dist_load(restored, checkpoint_id=tmp_path)

    assert all(torch.equal(restored[key], value) for key, value in expected.items())
    assert (tmp_path / ".metadata").is_file()
    assert len(list(tmp_path.glob("*.distcp"))) > 1


def test_sync_selects_bounded_strategy_while_async_keeps_mcore_default():
    source = SOURCE.read_text(encoding="utf-8")
    save = source[source.index("def save_dist_checkpointing(") : source.index("def load_dist_checkpointing(")]
    assert "if async_save" in save
    assert "else _bounded_sync_torch_dist_strategy(stage_callback)" in save
    assert "FullyParallelSaveStrategyWrapper" in save
    assert "async_sharded_save=async_save" in save
    assert '"checkpoint_dcp_before_strategy_save"' in save
    assert '"checkpoint_dcp_after_strategy_save"' in save
    assert "self.serialization_format = SerializationFormat.TORCH_SAVE" in SOURCE.read_text(encoding="utf-8")
