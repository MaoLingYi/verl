import ast
import gc
from pathlib import Path
from types import SimpleNamespace
import weakref

import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]


class DDP:
    def __init__(self, data, grad):
        self.buffers = [SimpleNamespace(param_data=data, grad_data=grad)]
        self.expert_parallel_buffers = []
        self.module = SimpleNamespace(parameters=lambda: [])


class DeviceTensor:
    def __init__(self, values):
        self.values = values
        self.device = SimpleNamespace(type="cuda")
        self.capacity = values.numel()
        self.data = self

    def storage(self):
        return SimpleNamespace(size=lambda: self.capacity, resize_=lambda size: setattr(self, "capacity", size))

    def cpu(self):
        return SimpleNamespace(pin_memory=lambda: self.values.detach().clone())

    def copy_(self, source, **kwargs):
        with torch.no_grad():
            self.values.copy_(source)

    def zero_(self):
        self.values.zero_()


def helpers(events, *, real_cuda=False):
    path = ROOT / "verl/utils/megatron_utils.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = {"load_megatron_model_to_gpu", "offload_megatron_model_to_cpu",
             "release_megatron_model_cpu_data", "megatron_model_cpu_data_bytes"}
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    for node in nodes:
        node.decorator_list = []
    device = torch.cuda if real_cuda else SimpleNamespace(
        synchronize=lambda: events.append("synchronize"), empty_cache=lambda: None)
    torch_api = torch if real_cuda else SimpleNamespace(
        _C=SimpleNamespace(_host_emptyCache=lambda: events.append("empty_host_cache")))
    namespace = {"DDP": DDP, "torch": torch_api, "gc": gc, "get_torch_device": lambda: device, "get_device_id": lambda: 0}
    namespace["logger"] = SimpleNamespace(warning=lambda message, *args: events.append(("warning", message % args)))
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


@pytest.mark.parametrize("reclaim_fails", [False, True])
def test_pinned_cache_reclaim_is_best_effort_after_correct_backing_release(reclaim_fails):
    events = []
    api = helpers(events)
    data = DeviceTensor(torch.tensor([1., 2., 3.]))
    model = DDP(data, DeviceTensor(torch.zeros(3)))
    api["offload_megatron_model_to_cpu"]([model])
    reference = weakref.ref(data.cpu_data)
    api["load_megatron_model_to_gpu"]([model])
    pointer, expected = data.values.data_ptr(), data.values.clone()
    calls = []

    def empty_host_cache():
        assert not hasattr(data, "cpu_data")
        assert events[-1] == "synchronize"
        calls.append("empty_host_cache")
        if reclaim_fails:
            raise RuntimeError("private allocator API failed")

    api["torch"]._C._host_emptyCache = empty_host_cache
    assert api["release_megatron_model_cpu_data"]([model]) == 12
    assert reference() is None
    assert api["megatron_model_cpu_data_bytes"]([model]) == 0
    assert data.values.data_ptr() == pointer
    torch.testing.assert_close(data.values, expected, rtol=0, atol=0)
    warnings = [event for event in events if isinstance(event, tuple) and event[0] == "warning"]
    assert len(warnings) == int(reclaim_fails)
    if reclaim_fails:
        assert "private allocator API failed" in warnings[0][1]
    event_count = len(events)
    assert api["release_megatron_model_cpu_data"]([model]) == 0
    assert calls == ["empty_host_cache"]
    assert len(events) == event_count


def test_loaded_backing_release_preserves_model_gradient_and_rebuilds_next_offload():
    events = []
    api = helpers(events)
    values = torch.nn.Parameter(torch.tensor([1., 2., 3.]))
    data = DeviceTensor(values)
    model = DDP(data, DeviceTensor(torch.zeros(3)))
    optimizer = torch.optim.SGD([values], lr=0.1)
    for _ in range(3):
        api["offload_megatron_model_to_cpu"]([model])
        reference = weakref.ref(data.cpu_data)
        expected = data.cpu_data.clone()
        api["load_megatron_model_to_gpu"]([model])
        pointer = values.data_ptr()
        assert api["release_megatron_model_cpu_data"]([model]) == 12
        assert reference() is None
        assert events[-2:] == ["synchronize", "empty_host_cache"]
        assert values.data_ptr() == pointer
        torch.testing.assert_close(values, expected, rtol=0, atol=0)
        assert api["megatron_model_cpu_data_bytes"]([model]) == 0
        event_count = len(events)
        assert api["release_megatron_model_cpu_data"]([model]) == 0
        assert len(events) == event_count
        optimizer.zero_grad()
        values.square().sum().backward()
        torch.testing.assert_close(values.grad, 2 * expected)
        optimizer.step()
        torch.testing.assert_close(values, expected * 0.8)


def test_release_validates_all_buffers_and_failed_sync_keeps_source_owners():
    events = []
    api = helpers(events)
    models = [DDP(DeviceTensor(torch.zeros(2)), DeviceTensor(torch.zeros(2))) for _ in range(2)]
    for model in models:
        api["offload_megatron_model_to_cpu"]([model])
    api["load_megatron_model_to_gpu"]([models[0]])
    with pytest.raises(RuntimeError, match="complete GPU onload"):
        api["release_megatron_model_cpu_data"](models)
    assert all(hasattr(model.buffers[0].param_data, "cpu_data") for model in models)

    api["load_megatron_model_to_gpu"]([models[1]])
    device = SimpleNamespace(synchronize=lambda: (_ for _ in ()).throw(RuntimeError("copy failed")))
    api["get_torch_device"] = lambda: device
    with pytest.raises(RuntimeError, match="copy failed"):
        api["release_megatron_model_cpu_data"](models)
    assert all(hasattr(model.buffers[0].param_data, "cpu_data") for model in models)


@pytest.mark.parametrize("eu_enabled", [True, False])
def test_actor_update_releases_loaded_backing_before_cache_allocation_only_for_eu(eu_enabled):
    events = []
    api = helpers(events)
    model = DDP(DeviceTensor(torch.ones(2)), DeviceTensor(torch.zeros(2)))
    api["offload_megatron_model_to_cpu"]([model])
    path = ROOT / "verl/workers/megatron_workers.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "ActorRolloutRefWorker")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "update_actor")
    method.decorator_list = []
    end = next(i for i, node in enumerate(method.body)
               if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "micro_batch_size"
                                                     for t in node.targets))
    method.body = method.body[:end]
    method.body.append(ast.Return(value=None))

    def load_model(models):
        events.append("model_onload")
        api["load_megatron_model_to_gpu"](models)

    def release(models):
        events.append("backing_release")
        return api["release_megatron_model_cpu_data"](models)

    namespace = {
        "DataProto": object, "logger": None, "torch": torch, "_GIB": 1024**3,
        "load_megatron_model_to_gpu": load_model,
        "release_megatron_model_cpu_data": release,
        "megatron_model_cpu_data_bytes": api["megatron_model_cpu_data_bytes"],
        "log_gpu_memory_usage": lambda *args, **kwargs: None,
        "log_eu_derpo_memory": lambda stage, *args, **kwargs: events.append(stage),
        "_best_effort_release_host_allocator_after_copy_param_restore": lambda: events.append("host_trim") or {},
    }
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), str(path), "exec"), namespace)
    worker = SimpleNamespace(
        _is_actor=True, _is_offload_param=True, actor_module=[model], actor_optimizer=object(),
        _restore_actor_optimizer_copy_params_for_update=lambda: events.append("selective_restore"),
        _preserve_hdo_optimizer_residency=lambda: True, _load_actor_optimizer_for_update=lambda: False,
        config=SimpleNamespace(actor=SimpleNamespace(eu_derpo=SimpleNamespace(enabled=eu_enabled))),
    )
    namespace["update_actor"](worker, SimpleNamespace(meta_info={}))
    if eu_enabled:
        assert events.index("selective_restore") < events.index("model_onload") < events.index("backing_release")
        assert events.index("empty_host_cache") < events.index("host_trim") < events.index("after_actor_load")
        assert api["megatron_model_cpu_data_bytes"]([model]) == 0
    else:
        assert "backing_release" not in events
        assert api["megatron_model_cpu_data_bytes"]([model]) == 8


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA pinned-host allocator")
def test_real_cuda_backing_release_and_rebuild():
    api = helpers([], real_cuda=True)
    data = torch.arange(16, device="cuda", dtype=torch.float32)
    expected = data.clone()
    active_optimizer_state = torch.full((16,), 7.0, pin_memory=True)
    active_pointer = active_optimizer_state.data_ptr()
    model = DDP(data, torch.zeros_like(data))
    for _ in range(3):
        api["offload_megatron_model_to_cpu"]([model])
        reference = weakref.ref(data.cpu_data)
        api["load_megatron_model_to_gpu"]([model])
        gpu_pointer = data.data_ptr()
        assert api["release_megatron_model_cpu_data"]([model]) == 64
        assert reference() is None
        assert data.data_ptr() == gpu_pointer
        assert active_optimizer_state.data_ptr() == active_pointer
        torch.testing.assert_close(active_optimizer_state, torch.full((16,), 7.0), rtol=0, atol=0)
        torch.testing.assert_close(data, expected, rtol=0, atol=0)
        data.add_(1)
        expected.add_(1)
