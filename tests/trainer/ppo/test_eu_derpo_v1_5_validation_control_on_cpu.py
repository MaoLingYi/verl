import ast
import asyncio
import copy
import importlib.util
import inspect
from pathlib import Path
from types import MethodType, SimpleNamespace

import torch


ROOT = Path(__file__).parents[3]
SELECTOR_PATH = ROOT / "verl/workers/rollout/sglang_rollout/eu_derpo_v15.py"
WORKER_PATH = ROOT / "verl/workers/megatron_workers.py"
TRAINER_PATH = ROOT / "verl/trainer/ppo/ray_trainer.py"
SERVER_PATH = ROOT / "verl/workers/rollout/sglang_rollout/async_sglang_server.py"
ADAPTER_PATH = ROOT / "verl/workers/rollout/sglang_rollout/sglang_rollout.py"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _method(path, class_name, method_name, namespace, *, strip_imports=False):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    method = copy.deepcopy(next(node for node in cls.body if getattr(node, "name", None) == method_name))
    method.decorator_list = []
    if strip_imports:
        method.body = [node for node in method.body if not isinstance(node, (ast.Import, ast.ImportFrom))]
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[method_name]


selector = _load("eu_derpo_v15_validation_control_under_test", SELECTOR_PATH)


def test_running_event_loop_awaits_worker_control_without_nested_loop():
    calls = []

    class Rollout:
        async def set_eu_derpo_v15_validation_mode(self, enabled, actor_version, state_version):
            assert asyncio.get_running_loop().is_running()
            calls.append((enabled, actor_version, state_version))

    worker = SimpleNamespace(
        config=SimpleNamespace(actor=SimpleNamespace(eu_derpo=SimpleNamespace(enabled=True, version="1.5"))),
        actor=SimpleNamespace(
            _eu_derpo_optimizer_generation=2,
            eu_derpo_utility_state=SimpleNamespace(version=2),
        ),
        rollout=Rollout(),
    )
    fake_torch = SimpleNamespace(distributed=SimpleNamespace(get_rank=lambda: 1))
    method = _method(
        WORKER_PATH,
        "ActorRolloutRefWorker",
        "set_eu_derpo_v15_validation_mode",
        {"torch": fake_torch, "logger": SimpleNamespace(warning=lambda *args: None)},
    )
    asyncio.run(method(worker, True))
    assert calls == [(True, 2, 2)]


def test_control_payload_is_python_metadata_and_never_calls_weight_update():
    requests = []

    class Request:
        def __init__(self, server_args):
            self.server_args = server_args

    class Tokenizer:
        async def set_internal_state(self, request):
            requests.append(request)
            return [True]

    server = SimpleNamespace(node_rank=0, tokenizer_manager=Tokenizer())
    method = _method(
        SERVER_PATH,
        "SGLangHttpServer",
        "set_eu_derpo_v15_validation_mode",
        {
            "SetInternalStateReq": Request,
            "logger": SimpleNamespace(warning=lambda *args: None),
            "VALIDATION_MODE": selector.VALIDATION_MODE,
            "ACTOR_VERSION": selector.ACTOR_VERSION,
            "STATE_VERSION": selector.STATE_VERSION,
        },
        strip_imports=True,
    )
    asyncio.run(method(server, True, 3, 3))
    assert len(requests) == 1
    assert all(type(value) in {bool, int} for value in requests[0].server_args.values())
    assert not any(torch.is_tensor(value) for value in requests[0].server_args.values())


def test_validation_on_bypasses_v15_selector_for_natural_routing():
    source = inspect.getsource(selector.install_sglang_patch)
    natural = source.index("if state.validation_mode:")
    natural_return = source.index("return topk_module.select_experts(", natural)
    exploration = source.index("selected, weights, mode = select_v15_routes(", natural_return)
    assert natural < natural_return < exploration


def test_validation_off_restores_utility_history_not_bootstrap():
    state = selector._RolloutUtilityState(seed=1234)
    state.state_version = state.actor_version = 2
    state.set_validation_mode(True, actor_version=2, utility_state_version=2)
    state.set_validation_mode(False, actor_version=2, utility_state_version=2)
    logits = torch.arange(128, dtype=torch.float32)[None]
    _, _, mode = selector.select_v15_routes(
        logits,
        layer_id=0,
        state_version=state.state_version,
        mu=state.mu,
        sigma=state.sigma,
        noise=torch.zeros((1, 16)),
    )
    assert mode == "utility_history"


def test_toggle_preserves_actor_and_utility_versions():
    state = selector._RolloutUtilityState(seed=1234)
    state.state_version = state.actor_version = 4
    before = (state.actor_version, state.state_version)
    response = selector._apply_validation_control(
        state,
        {
            selector.VALIDATION_MODE: True,
            selector.ACTOR_VERSION: 4,
            selector.STATE_VERSION: 4,
        },
    )
    selector._apply_validation_control(
        state,
        {
            selector.VALIDATION_MODE: False,
            selector.ACTOR_VERSION: 4,
            selector.STATE_VERSION: 4,
        },
    )
    assert before == (state.actor_version, state.state_version)
    assert response[selector.ACTOR_VERSION] == response[selector.STATE_VERSION] == 4


def test_validation_exception_restores_exploration_in_finally():
    toggles = []
    trainer = SimpleNamespace(
        config=SimpleNamespace(
            actor_rollout_ref=SimpleNamespace(
                actor=SimpleNamespace(eu_derpo=SimpleNamespace(enabled=True, version="1.5"))
            )
        ),
        actor_rollout_wg=SimpleNamespace(
            set_eu_derpo_v15_validation_mode=lambda enabled: toggles.append(enabled)
        ),
    )

    def fail(_self, _merged):
        raise RuntimeError("validation failed")

    trainer._validate_impl = MethodType(fail, trainer)
    method = _method(TRAINER_PATH, "RayPPOTrainer", "_validate", {})
    try:
        method(trainer)
    except RuntimeError as error:
        assert str(error) == "validation failed"
    else:
        raise AssertionError("validation failure was unexpectedly swallowed")
    assert toggles == [True, False]


def test_normal_weight_update_path_is_unchanged():
    source = ADAPTER_PATH.read_text(encoding="utf-8")
    update = source[source.index("    async def update_weights(") :]
    control = source[
        source.index("    async def set_eu_derpo_v15_validation_mode(") : source.index("    async def update_weights(")
    ]
    assert "await sgl_update_weights(" in update
    assert "update_weights" not in control


def test_non_v15_validation_does_not_enter_control_path():
    toggles = []
    trainer = SimpleNamespace(
        config=SimpleNamespace(
            actor_rollout_ref=SimpleNamespace(
                actor=SimpleNamespace(eu_derpo=SimpleNamespace(enabled=True, version="1.4"))
            )
        ),
        actor_rollout_wg=SimpleNamespace(
            set_eu_derpo_v15_validation_mode=lambda enabled: toggles.append(enabled)
        ),
    )
    trainer._validate_impl = MethodType(lambda _self, merged: ("natural", merged), trainer)
    method = _method(TRAINER_PATH, "RayPPOTrainer", "_validate", {})
    assert method(trainer, True) == ("natural", True)
    assert toggles == []


def test_natural_validation_returns_before_route_capture():
    source = inspect.getsource(selector.install_sglang_patch)
    natural = source.index("if state.validation_mode:")
    natural_return = source.index("return topk_module.select_experts(", natural)
    capture = source.index("_capture_dispatched_routes(", natural_return)
    assert natural < natural_return < capture


def test_worker_group_contract_is_async_blocking_dispatch():
    tree = ast.parse(WORKER_PATH.read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "ActorRolloutRefWorker")
    method = next(
        node for node in cls.body if getattr(node, "name", None) == "set_eu_derpo_v15_validation_mode"
    )
    assert isinstance(method, ast.AsyncFunctionDef)
    decorator = method.decorator_list[0]
    assert isinstance(decorator, ast.Call)
    assert all(keyword.arg != "blocking" for keyword in decorator.keywords)
    assert "run_until_complete" not in ast.unparse(method)
    assert "update_weights" not in ast.unparse(method)
