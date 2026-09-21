from pathlib import Path
import sys
import types

import torch


MODULE = Path(__file__).parents[3] / "verl/utils/megatron/router_replay_patch.py"


def _load_router_replay(monkeypatch):
    names = (
        "megatron",
        "megatron.core",
        "megatron.core.transformer",
        "megatron.core.transformer.moe",
        "megatron.core.transformer.moe.moe_utils",
        "megatron.core.transformer.moe.token_dispatcher",
        "megatron.core.transformer.moe.router",
        "megatron.core.transformer.transformer_config",
    )
    modules = {name: types.ModuleType(name) for name in names}
    for name in names[:4]:
        modules[name].__path__ = []
    moe_utils = modules["megatron.core.transformer.moe.moe_utils"]
    moe_utils.apply_router_token_dropping = lambda *args, **kwargs: None
    moe_utils.compute_routing_scores_for_aux_loss = lambda *args, **kwargs: None
    moe_utils.group_limited_topk = lambda *args, **kwargs: None
    modules["megatron.core.transformer.moe.token_dispatcher"].MoEAlltoAllTokenDispatcher = type(
        "MoEAlltoAllTokenDispatcher", (), {}
    )
    modules["megatron.core.transformer.moe.router"].TopKRouter = type("TopKRouter", (), {})
    modules["megatron.core.transformer.transformer_config"].TransformerConfig = type(
        "TransformerConfig", (), {}
    )
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    module = types.ModuleType("router_replay_counts_under_test")
    source = "from __future__ import annotations\n" + MODULE.read_text(encoding="utf-8")
    exec(compile(source, str(MODULE), "exec"), module.__dict__)
    return module


def test_full_and_partial_replay_expected_counts_and_clear(monkeypatch):
    module = _load_router_replay(monkeypatch)
    replay = module.RouterReplay
    replay.router_instances = []
    full = replay()
    partial = replay()
    full.set_target_indices(torch.zeros((2, 3, 8), dtype=torch.int32))
    partial.set_target_indices(
        torch.zeros((2, 3, 8), dtype=torch.int32),
        token_mask=torch.tensor([[True, False, True], [False, True, False]]),
    )
    full.replay_matched = full.replay_compared = torch.tensor(6)
    partial.replay_matched = partial.replay_compared = torch.tensor(3)
    assert replay.replay_match_counts() == (9, 9)
    replay.clear_global_indices()
    for router in replay.router_instances:
        assert router.target_topk_idx is None
        assert router.replay_expected is None
        assert router.replay_backward_list == []
