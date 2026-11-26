"""Exercise the Megatron training loss closure without starting MCore, Ray or SGLang."""

from types import SimpleNamespace

import pytest
import torch

from tests.trainer.ppo.test_dppo_tv_on_cpu import load_dppo_registry
from tests.utils.r3_locked_dataproto_cpu import function


ACTOR = "verl/workers/actor/megatron_actor.py"


def test_megatron_loss_closure_uses_behavior_and_shifted_training_logprob(monkeypatch):
    import sys

    registry, config = load_dppo_registry()
    from verl.trainer.ppo.dppo import DPPOStats

    # Exclude the unrelated rollout-correction diagnostics from this isolated CPU call.
    monkeypatch.setitem(sys.modules, "verl.trainer.ppo.rollout_corr_helper", SimpleNamespace(
        compute_rollout_corr_metrics_from_logprobs=lambda **kwargs: {},
    ))
    assert config.policy_loss.loss_mode == "dppo_binary_tv"
    config.policy_loss.get = lambda key, default=None: getattr(config.policy_loss, key, default)
    actor_config = SimpleNamespace(
        policy_loss=config.policy_loss, global_batch_info=config.global_batch_info,
        loss_agg_mode="token-mean", entropy_coeff=0.0, use_kl_loss=False,
        router_shift_weighting=SimpleNamespace(enabled=False), eu_derpo=SimpleNamespace(enabled=False),
    )
    loss_func = function(ACTOR, "loss_func", {
        "torch": torch, "forward_only": False, "calculate_entropy": False,
        "self": SimpleNamespace(config=actor_config), "get_policy_loss_fn": registry.__getitem__,
        "dppo_stats": DPPOStats(), "append_to_dict": dict.update,
    })
    # At sequence positions 1 and 2 the actor predicts the two response tokens.
    training = torch.tensor([[0.9, 0.4, 0.4, 0.1]]).log().requires_grad_()
    old = training[:, 1:3].detach().clone().requires_grad_()
    behavior = torch.tensor([[0.5, 0.05]]).log().requires_grad_()
    data = dict(responses=torch.tensor([[12, 2]]), response_mask=torch.ones(1, 2),
                advantages=torch.ones(1, 2), old_log_probs=old, rollout_log_probs=behavior)
    loss, _ = loss_func({"log_probs": training}, data, {})
    loss.backward()
    torch.testing.assert_close(training.grad, torch.tensor([[0.0, -0.4, 0.0, 0.0]]))
    assert behavior.grad is None and old.grad is None
    del data["rollout_log_probs"]
    with pytest.raises(KeyError, match="rollout_log_probs"):
        loss_func({"log_probs": training}, data, {})


def test_real_dataproto_reorder_actor_selection_and_microbatch_alignment():
    pytest.importorskip("tensordict")
    pytest.importorskip("ray")
    from verl.protocol import DataProto

    ids = torch.arange(12).reshape(3, 4)
    behavior = -0.1 * (1 + ids[:, -2:])
    data = DataProto.from_dict(tensors={
        "input_ids": ids, "responses": ids[:, -2:], "response_mask": torch.ones(3, 2),
        "old_log_probs": torch.zeros_like(behavior), "rollout_log_probs": behavior,
        "attention_mask": torch.ones_like(ids), "position_ids": ids,
        "advantages": torch.ones_like(behavior), "routed_experts": ids.to(torch.uint8),
    })
    config = SimpleNamespace(
        eu_derpo=SimpleNamespace(enabled=False), use_kl_loss=False, router_replay=SimpleNamespace(mode="R3"),
        ppo_epochs=1, ppo_mini_batch_size=3, data_loader_seed=42, shuffle=False,
    )
    owner = SimpleNamespace(config=config, eu_derpo_observer=None, router_shift_observer=None)
    data.reorder(torch.tensor([2, 0, 1]))
    iterator = function(ACTOR, "make_minibatch_iterator", {})(owner, data)
    minibatch, = iterator
    for row, microbatch in zip([2, 0, 1], minibatch.batch.split(1)):
        torch.testing.assert_close(microbatch["responses"][0], ids[row, -2:])
        torch.testing.assert_close(microbatch["rollout_log_probs"][0], behavior[row])
        assert torch.count_nonzero(microbatch["old_log_probs"]) == 0
