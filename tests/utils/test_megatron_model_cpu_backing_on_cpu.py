import ast
import copy
from contextlib import nullcontext
import gc
from pathlib import Path
from types import MethodType, SimpleNamespace
import weakref

import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]


def pinned_cpu_fixture(value):
    # Only the CUDA pinned-allocation facility is substituted in these CPU tests.
    value.is_pinned = lambda: True
    return value


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
        return SimpleNamespace(pin_memory=lambda: pinned_cpu_fixture(self.values.detach().clone()))

    def __getattr__(self, name):
        return getattr(self.values, name)

    def detach(self):
        return self.values.detach()

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
        Tensor=torch.Tensor, _C=SimpleNamespace(_host_emptyCache=lambda: events.append("empty_host_cache")))
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


@pytest.mark.parametrize("resume_step", [0, 100])
def test_hybrid_phase_offload_reuses_one_direct_destination_across_weight_sync(resume_step):
    api = helpers([])
    allocated = []

    def allocate(data, *, device, pin_memory):
        assert device == "cpu" and pin_memory is True
        result = pinned_cpu_fixture(torch.empty_like(data.values))
        allocated.append(result.data_ptr())
        return result

    api["torch"].empty_like = allocate
    reference = torch.nn.Parameter(torch.tensor([1., 2., 3.]))
    uninterrupted = torch.optim.Adam([reference], lr=0.01)
    for _ in range(resume_step):
        uninterrupted.zero_grad()
        reference.square().sum().backward()
        uninterrupted.step()
    param = torch.nn.Parameter(reference.detach().clone())
    data = DeviceTensor(param)
    model = DDP(data, DeviceTensor(torch.zeros_like(param)))
    data.cpu = lambda: pytest.fail("materialized an intermediate pageable CPU tensor")
    optimizer = torch.optim.Adam([param], lr=0.01)
    if resume_step:
        optimizer.load_state_dict(copy.deepcopy(uninterrupted.state_dict()))
    for step in (1, 2):
        uninterrupted.zero_grad()
        reference.square().sum().backward()
        uninterrupted.step()
        optimizer.zero_grad()
        param.square().sum().backward()
        optimizer.step()
        state = optimizer.state[param]
        moment_ptr = state["exp_avg"].data_ptr()
        expected = param.detach().clone()
        # First phase transition after optimizer update creates exactly one host destination.
        api["offload_megatron_model_to_cpu"]([model], reuse_pinned_backing=True)
        assert data.capacity == 0 and len(allocated) == step
        pointer = data.cpu_data.data_ptr()
        torch.testing.assert_close(data.cpu_data, expected)
        api["load_megatron_model_to_gpu"]([model], load_grad=False)
        # Make stale/copy-skipping behavior observable without a real GPU.
        data.cpu_data.fill_(float("nan"))
        api["offload_megatron_model_to_cpu"]([model], reuse_pinned_backing=True)
        assert len(allocated) == step and data.cpu_data.data_ptr() == pointer
        torch.testing.assert_close(data.cpu_data, expected)
        assert state["step"].item() == resume_step + step and state["exp_avg"].data_ptr() == moment_ptr
        torch.testing.assert_close(param, reference, rtol=0, atol=0)
        for key in ("exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(state[key], uninterrupted.state[reference][key], rtol=0, atol=0)
        api["load_megatron_model_to_gpu"]([model])
        assert api["release_megatron_model_cpu_data"]([model]) == expected.numel() * expected.element_size()


@pytest.mark.parametrize("enabled,version,backend,expected", [
    (True, "1.8.3", "ram_nfs_hybrid", True),
    (True, "1.8.3", "nfs_direct", False),
    (True, "1.8.2", "cpu_pageable", False),
    (False, "1.8.3", "ram_nfs_hybrid", False),
])
def test_only_hybrid_uses_reusable_backing_at_rollout_phase_boundaries(enabled, version, backend, expected):
    path = ROOT / "verl/workers/megatron_workers.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    worker = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "ActorRolloutRefWorker")
    methods = {node.name: node for node in worker.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    name = "_offload_actor_model_for_rollout"
    for stage in ("init_model", "load_checkpoint", "rollout_mode",
                  "_finish_actor_update_residency", "_release_checkpoint_residency_hold"):
        assert any(isinstance(node, ast.Call) and getattr(node.func, "attr", None) == name
                   for node in ast.walk(methods[stage]))
    calls = []
    namespace = {"offload_megatron_model_to_cpu": lambda model, **kwargs: calls.append((model, kwargs))}
    exec(compile(ast.Module(body=[methods[name]], type_ignores=[]), str(path), "exec"), namespace)
    cfg = SimpleNamespace(enabled=enabled, version=version, cache_backend=backend)
    namespace[name](SimpleNamespace(config=SimpleNamespace(actor=SimpleNamespace(eu_derpo=cfg))), worker)
    assert calls == [(worker, {"reuse_pinned_backing": expected})]


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
        config=SimpleNamespace(actor=SimpleNamespace(eu_derpo=SimpleNamespace(enabled=eu_enabled),
                                                     policy_loss=SimpleNamespace(loss_mode="vanilla"))),
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
@pytest.mark.parametrize("reuse_backing", [False, True])
def test_real_cuda_backing_release_and_rebuild(reuse_backing):
    api = helpers([], real_cuda=True)
    data = torch.arange(16, device="cuda", dtype=torch.float32)
    expected = data.clone()
    active_optimizer_state = torch.full((16,), 7.0, pin_memory=True)
    active_pointer = active_optimizer_state.data_ptr()
    model = DDP(data, torch.zeros_like(data))
    for _ in range(3):
        api["offload_megatron_model_to_cpu"]([model], reuse_pinned_backing=reuse_backing)
        pointer = data.cpu_data.data_ptr()
        api["load_megatron_model_to_gpu"]([model], load_grad=False)
        api["offload_megatron_model_to_cpu"]([model], reuse_pinned_backing=reuse_backing)
        if reuse_backing:
            assert pointer == data.cpu_data.data_ptr() and data.cpu_data.is_pinned()
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


@pytest.mark.parametrize("existing_backing", [False, True])
def test_cold_phase_offload_allocation_ledger_and_bitwise_equal(existing_backing):
    """Real production offload; CPU storage spy substitutes only CUDA/pinned allocation."""
    ledgers = []
    results = []
    for reuse in (False, True):
        ledger = []
        api = helpers([])
        values = torch.arange(512, dtype=torch.float32).to(torch.bfloat16)
        payload_bytes = values.numel() * values.element_size()

        class CopySource(DeviceTensor):
            def cpu(self):
                pageable = self.values.clone()
                ledger.append(("pageable", pageable.numel() * pageable.element_size()))
                def pin():
                    pinned = pageable.clone()
                    ledger.append(("pinned_fixture", pinned.numel() * pinned.element_size()))
                    return pinned
                return SimpleNamespace(pin_memory=pin)

        data = CopySource(values)
        original = pinned_cpu_fixture(torch.zeros_like(values)) if existing_backing else None
        if original is not None:
            data.cpu_data = original
        api["torch"].empty_like = lambda source, **kwargs: (
            ledger.append(("direct_pinned_fixture", payload_bytes))
            or pinned_cpu_fixture(torch.empty_like(source.values)))
        model = DDP(data, DeviceTensor(torch.zeros_like(values)))
        api["offload_megatron_model_to_cpu"]([model], reuse_pinned_backing=reuse)
        assert data.capacity == 0
        assert data.cpu_data.dtype == torch.bfloat16
        if reuse and original is not None:
            assert data.cpu_data is original
        results.append(data.cpu_data.view(torch.int16).clone())
        ledgers.append(ledger)
    torch.testing.assert_close(*results, rtol=0, atol=0)
    assert ledgers[0] == [("pageable", payload_bytes), ("pinned_fixture", payload_bytes)]
    assert ledgers[1] == ([] if existing_backing else [("direct_pinned_fixture", payload_bytes)])


@pytest.mark.parametrize("existing_backing", [False, True])
def test_failed_direct_copy_retains_device_authority_before_offload(existing_backing):
    api = helpers([])
    data = DeviceTensor(torch.arange(8, dtype=torch.float32))
    model = DDP(data, DeviceTensor(torch.zeros(8)))
    original = data.values.clone()
    destination = pinned_cpu_fixture(torch.zeros(8))
    def failed_copy(source, **kwargs):
        assert kwargs == {"non_blocking": False}
        raise RuntimeError("D2H failed")
    destination.copy_ = failed_copy
    if existing_backing:
        data.cpu_data = destination
    api["torch"].empty_like = lambda *args, **kwargs: destination
    with pytest.raises(RuntimeError, match="D2H failed"):
        api["offload_megatron_model_to_cpu"]([model], reuse_pinned_backing=True)
    assert data.capacity == 8 and model.buffers[0].grad_data.capacity == 8
    torch.testing.assert_close(data.values, original, rtol=0, atol=0)
    assert hasattr(data, "cpu_data") == existing_backing
    if existing_backing:
        assert data.cpu_data is destination


@pytest.mark.parametrize("invalid", ["shape", "dtype", "device", "grad", "unpinned", "offset", "storage", "stride", "object"])
def test_retained_backing_metadata_rejected_before_any_buffer_copy(invalid):
    api = helpers([])
    first = DeviceTensor(torch.arange(8, dtype=torch.float32))
    second = DeviceTensor(torch.arange(8, dtype=torch.float32) + 10)
    first.cpu_data = pinned_cpu_fixture(torch.zeros(8))
    copies = []
    first.cpu_data.copy_ = lambda *args, **kwargs: copies.append(True)
    if invalid == "shape":
        backing = torch.zeros(1, 8)  # copy_ would broadcast the source's shape (8,) into (1, 8).
    elif invalid == "dtype":
        backing = torch.zeros(8, dtype=torch.float64)
    elif invalid == "device":
        backing = torch.empty(8, device="meta")
    elif invalid == "grad":
        backing = torch.zeros(8, requires_grad=True)
    elif invalid == "offset":
        backing = torch.zeros(9)[1:]
    elif invalid == "storage":
        backing = torch.zeros(16)[:8]
    elif invalid == "stride":
        backing = torch.zeros(16)[::2]
    elif invalid == "object":
        backing = object()
    else:
        backing = torch.zeros(8)
    if isinstance(backing, torch.Tensor) and invalid != "unpinned":
        pinned_cpu_fixture(backing)
    second.cpu_data = backing
    models = [DDP(first, DeviceTensor(torch.zeros(8))), DDP(second, DeviceTensor(torch.zeros(8)))]
    with pytest.raises(RuntimeError, match="Invalid retained"):
        api["offload_megatron_model_to_cpu"](models, reuse_pinned_backing=True)
    assert copies == []
    assert first.capacity == second.capacity == 8
    assert all(model.buffers[0].grad_data.capacity == 8 for model in models)
    assert second.cpu_data is backing


def test_distinct_gpu_buffers_cannot_share_one_retained_cpu_destination():
    api = helpers([])
    first = DeviceTensor(torch.arange(8, dtype=torch.float32))
    second = DeviceTensor(torch.arange(8, dtype=torch.float32) + 10)
    shared = pinned_cpu_fixture(torch.zeros(8))
    first.cpu_data = second.cpu_data = shared
    models = [DDP(first, DeviceTensor(torch.zeros(8))), DDP(second, DeviceTensor(torch.zeros(8)))]
    with pytest.raises(RuntimeError, match="multiple buffer owners"):
        api["offload_megatron_model_to_cpu"](models, reuse_pinned_backing=True)
    assert first.capacity == second.capacity == 8
    assert torch.count_nonzero(shared) == 0


def test_all_init_and_checkpoint_actor_recold_paths_use_scoped_helper():
    tree = ast.parse((ROOT / "verl/workers/megatron_workers.py").read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "ActorRolloutRefWorker")
    methods = {node.name: node for node in cls.body if isinstance(node, ast.FunctionDef)}
    for name, expected_calls in (("init_model", 1), ("load_checkpoint", 6),
                                 ("compute_log_prob", 1), ("save_checkpoint", 1)):
        calls = [node for node in ast.walk(methods[name]) if isinstance(node, ast.Call)]
        assert sum(getattr(node.func, "attr", None) == "_offload_actor_model_for_rollout" for node in calls) == expected_calls
        assert not any(isinstance(node.func, ast.Name) and node.func.id == "offload_megatron_model_to_cpu"
                       and any(isinstance(arg, ast.Attribute) and arg.attr == "actor_module" for arg in node.args)
                       for node in calls)

    direct = [node for node in ast.walk(cls) if isinstance(node, ast.Call)
              and isinstance(node.func, ast.Name) and node.func.id == "offload_megatron_model_to_cpu"]
    # Actor helper delegates once, while Ref remains on its unchanged default path.
    assert sorted(ast.unparse(call.args[0]) for call in direct) == ["models", "self.ref_module", "self.ref_module"]


@pytest.mark.parametrize("version,backend,reuse", [("1.8.3", "ram_nfs_hybrid", True),
                                                  ("1.8.3", "nfs_direct", False),
                                                  ("1.7", "cpu_pageable", False)])
def test_compute_log_prob_actual_worker_uses_scoped_backing_helper(version, backend, reuse):
    path = ROOT / "verl/workers/megatron_workers.py"
    cls = next(node for node in ast.parse(path.read_text(encoding="utf-8")).body
               if isinstance(node, ast.ClassDef) and node.name == "ActorRolloutRefWorker")
    names = {"compute_log_prob", "_offload_actor_model_for_rollout"}
    functions = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in names]
    for node in functions:
        node.decorator_list = []
    class Proto:
        def __init__(self, tensors=None, meta_info=None):
            self.batch = tensors or {}
            self.meta_info = meta_info or {}
        @classmethod
        def from_dict(cls, **kwargs):
            return cls(**kwargs)
        def to(self, device):
            assert device == "cpu"
            return self
    events = []
    expected = torch.tensor([[1., 2.]])
    namespace = {
        "DataProto": Proto, "nullcontext": nullcontext,
        "load_megatron_model_to_gpu": lambda model, **kw: events.append(("load", model, kw)),
        "offload_megatron_model_to_cpu": lambda model, **kw: events.append(("offload", model, kw)),
        "log_gpu_memory_usage": lambda *args, **kw: None,
        "aggressive_empty_cache": lambda **kw: events.append(("cache", kw)), "logger": None,
        "policy_prepass_tensors": lambda out, ent, eu: {"old_log_probs": out},
    }
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), namespace)
    worker = SimpleNamespace(
        _is_actor=True, _is_offload_param=True, actor_module=object(), enable_routing_replay=False,
        config=SimpleNamespace(
            actor=SimpleNamespace(eu_derpo=SimpleNamespace(enabled=True, version=version, cache_backend=backend),
                                  router_replay=SimpleNamespace(mode="disabled")),
            rollout=SimpleNamespace(log_prob_micro_batch_size_per_gpu=1, log_prob_max_token_len_per_gpu=4096,
                                    log_prob_use_dynamic_bsz=False, temperature=1.)),
        actor=SimpleNamespace(compute_log_prob=lambda **kw: (expected, torch.zeros_like(expected), None)),
    )
    worker._offload_actor_model_for_rollout = MethodType(namespace["_offload_actor_model_for_rollout"], worker)
    result = namespace["compute_log_prob"](worker, Proto())
    assert result.batch["old_log_probs"] is expected
    assert [event[0] for event in events] == ["load", "offload", "cache"]
    assert events[1] == ("offload", worker.actor_module, {"reuse_pinned_backing": reuse})


def benchmark_cpu_backing_fixture(reuse, size_mib):
    """CPU ownership/copy measurement; does not measure CUDA or pinned allocators."""
    import json
    import platform
    import time
    import psutil
    torch.set_num_threads(1)
    process = psutil.Process()
    samples = []
    allocations = []
    def sample():
        memory = process.memory_full_info()
        samples.append({"rss": memory.rss, "pss": getattr(memory, "pss", None),
                        "process_peak_rss": getattr(memory, "peak_wset", None)})
    api = helpers([])
    count = size_mib * 1024**2 // 2
    class Source(DeviceTensor):
        def cpu(self):
            pageable = self.values.clone()
            allocations.append({"kind": "pageable", "bytes": pageable.numel() * 2})
            sample()
            def pin():
                result = pageable.clone()
                allocations.append({"kind": "pinned_CPU_fixture", "bytes": result.numel() * 2})
                sample()
                return result
            return SimpleNamespace(pin_memory=pin)
    data = Source(torch.full((count,), 3., dtype=torch.bfloat16))
    data.cpu_data = pinned_cpu_fixture(torch.zeros_like(data.values))
    model = DDP(data, DeviceTensor(torch.zeros(1)))
    original_pointer = data.cpu_data.data_ptr()
    sample()
    started = time.perf_counter()
    api["offload_megatron_model_to_cpu"]([model], reuse_pinned_backing=reuse)
    elapsed = time.perf_counter() - started
    sample()
    assert bool(torch.all(data.cpu_data == 3.))
    result = {"backend": "CPU_fixture_not_CUDA_pinned", "platform": platform.system(),
              "reuse": reuse, "payload_bytes": count * 2, "allocations": allocations,
              "new_storage_bytes": sum(item["bytes"] for item in allocations),
              "copy_calls": 1 if reuse else 2, "copy_bytes": count * 2 * (1 if reuse else 2),
              "existing_backing_reused": data.cpu_data.data_ptr() == original_pointer,
              "rss_before": samples[0]["rss"], "rss_after": samples[-1]["rss"],
              "observed_rss_peak": max(item["rss"] for item in samples),
              "pss_before": samples[0]["pss"], "pss_after": samples[-1]["pss"],
              "process_peak_rss": samples[-1]["process_peak_rss"], "elapsed_seconds": elapsed}
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Real production offload with explicit CPU-only CUDA/pin fixtures")
    parser.add_argument("--reuse", action="store_true")
    parser.add_argument("--size-mib", type=int, default=64)
    args = parser.parse_args()
    if not 1 <= args.size_mib <= 256:
        parser.error("this developer CPU fixture uses 1..256 MiB")
    benchmark_cpu_backing_fixture(args.reuse, args.size_mib)
