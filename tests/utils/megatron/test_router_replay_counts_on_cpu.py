from pathlib import Path
import sys
import types

import pytest
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


def _run_replay(module, target, mask):
    replay = module.RouterReplay()
    replay.set_target_indices(target, token_mask=mask, retain_for_backward=mask is None)
    replay.set_router_replay_action(module.RouterReplayAction.REPLAY_FORWARD)
    logits = torch.arange(target.shape[0] * 128, dtype=torch.float32).reshape(target.shape[0], 128)
    _, routing_map = module._patched_topk_routing_with_score_function(
        logits, 8, False, None, None, "softmax", None, False, replay, None
    )
    return replay, routing_map


def test_valid_replay_padding_recompute_and_tp_replication(monkeypatch):
    module = _load_router_replay(monkeypatch)
    target = torch.tensor(
        [[1, 2, 3, 4, 5, 6, 7, 8], [0, 0, 0, 0, 0, 0, 0, 0]], dtype=torch.int64
    )
    mask = torch.tensor([True, False])
    tp0, routing_map = _run_replay(module, target, mask)
    tp1, _ = _run_replay(module, target.clone(), mask.clone())

    assert torch.equal(tp0.replayed_topk_idx[mask], target[mask])
    assert routing_map[mask].sum() == 8
    assert torch.equal(tp0.replayed_topk_idx, tp1.replayed_topk_idx)
    forward_ids = tp0.replayed_topk_idx.clone()
    next_target = target.clone()
    next_target[0] = torch.tensor([9, 10, 11, 12, 13, 14, 15, 16])
    tp0.set_target_indices(next_target, token_mask=mask, retain_for_backward=False)
    tp0.set_router_replay_action(module.RouterReplayAction.REPLAY_BACKWARD)
    logits = torch.randn(2, 128)
    module._patched_topk_routing_with_score_function(
        logits, 8, False, None, None, "softmax", None, False, tp0, None
    )
    assert torch.equal(tp0.replayed_topk_idx, forward_ids)
    assert torch.equal(tp0.replayed_target_topk_idx[mask], target[mask])
    assert torch.equal(tp0.replayed_token_mask, mask)


@pytest.mark.parametrize(
    "bad, message",
    [
        ([1, 2, 3, 4, 5, 6, 7, 7], "invalid valid-replay route"),
        ([1, 2, 3, 4, 5, 6, 7, 128], "invalid valid-replay route"),
        ([-1, 2, 3, 4, 5, 6, 7, 8], "invalid valid-replay route"),
    ],
)
def test_invalid_valid_replay_ids_hard_fail(monkeypatch, bad, message):
    module = _load_router_replay(monkeypatch)
    with pytest.raises(RuntimeError, match=message):
        _run_replay(module, torch.tensor([bad]), torch.tensor([True]))
