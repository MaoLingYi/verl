import importlib.util
import ast
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import torch

module_path = Path(__file__).parents[3] / "verl" / "trainer" / "ppo" / "router_shift_weighting.py"
spec = importlib.util.spec_from_file_location("router_shift_weighting_under_test", module_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
adjust_log_ratio_with_router_shift = module.adjust_log_ratio_with_router_shift


class ActorConfig:
    def __init__(self, clip_low=3e-4, clip_high=4e-4):
        self.clip_ratio = clip_low
        self.clip_ratio_low = clip_low
        self.clip_ratio_high = clip_high
        self.global_batch_info = {}
        self.router_shift_weighting = SimpleNamespace(gamma_min=0.8)


def _masked_mean(value, mask):
    mask = mask.to(value.dtype)
    return (value * mask).sum() / mask.sum()


def _seq_mean_token_mean(loss_mat, loss_mask, loss_agg_mode, **_):
    assert loss_agg_mode == "seq-mean-token-mean"
    mask = loss_mask.to(loss_mat.dtype)
    return ((loss_mat * mask).sum(-1) / mask.sum(-1).clamp_min(1)).mean()


def _load_gspo():
    source = (module_path.parent / "core_algos.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "compute_policy_loss_gspo")
    function.decorator_list = []
    namespace = {
        "torch": torch,
        "Any": Any,
        "Optional": Optional,
        "ActorConfig": ActorConfig,
        "adjust_log_ratio_with_router_shift": adjust_log_ratio_with_router_shift,
        "agg_loss": _seq_mean_token_mean,
        "verl_F": SimpleNamespace(masked_mean=_masked_mean),
    }
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future, function], type_ignores=[])), str(module_path), "exec"), namespace)
    return namespace["compute_policy_loss_gspo"]


compute_policy_loss_gspo = _load_gspo()


def _gspo(log_prob, old_log_prob, advantages, mask, gamma=None):
    return compute_policy_loss_gspo(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=mask,
        config=ActorConfig(),
        router_shift_gamma=gamma,
    )


def test_disabled_and_gamma_one_are_identity():
    log_ratio = torch.tensor([[math.log(1.1), math.log(0.9)]])
    disabled, disabled_weight = adjust_log_ratio_with_router_shift(log_ratio, None, gamma_min=0.8)
    weighted, _ = adjust_log_ratio_with_router_shift(log_ratio, torch.ones_like(log_ratio), gamma_min=0.8)
    assert disabled_weight is None
    torch.testing.assert_close(disabled, log_ratio)
    torch.testing.assert_close(weighted, log_ratio)


def test_router_shift_rescales_before_clipping_and_applies_floor():
    log_ratio = torch.tensor([[math.log(1.1), math.log(1.1)]])
    adjusted, weight = adjust_log_ratio_with_router_shift(
        log_ratio, torch.tensor([[0.9, 0.5]]), gamma_min=0.8
    )
    torch.testing.assert_close(torch.exp(adjusted), torch.tensor([[1.1 * 0.9, 1.1 * 0.8]]))
    torch.testing.assert_close(weight, torch.tensor([[0.9, 0.8]]))


def test_weight_is_detached_policy_gradient_and_masked_mean_remain_valid():
    log_ratio = torch.tensor([[math.log(1.1), math.log(1.1)]], requires_grad=True)
    gamma = torch.tensor([[0.9, 0.5]], requires_grad=True)
    adjusted, weight = adjust_log_ratio_with_router_shift(log_ratio, gamma, gamma_min=0.8)
    mask = torch.tensor([[True, False]])
    loss = -torch.exp(adjusted)[mask].mean()
    loss.backward()
    assert gamma.grad is None
    assert log_ratio.grad is not None
    assert weight[mask].mean().item() == torch.tensor(0.9).item()


def test_router_shift_rejects_misaligned_or_invalid_gamma():
    log_ratio = torch.zeros((1, 2))
    for gamma in (torch.ones(2), torch.tensor([[1.0, 0.0]]), torch.tensor([[1.0, float("nan")]])):
        try:
            adjust_log_ratio_with_router_shift(log_ratio, gamma, gamma_min=0.8)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid router-shift gamma was accepted")


def test_vanilla_grpo_integration_keeps_ppo_kl_raw_and_weights_before_clipping():
    core_source = (module_path.parent / "core_algos.py").read_text(encoding="utf-8")
    ppo_kl = core_source.index("ppo_kl = verl_F.masked_mean(-negative_approx_kl")
    adjust = core_source.index("negative_approx_kl, _ = adjust_log_ratio_with_router_shift")
    ratio = core_source.index("ratio = torch.exp(negative_approx_kl)", adjust)
    clipping = core_source.index("pg_losses2 = -advantages * torch.clamp", ratio)
    assert ppo_kl < adjust < ratio < clipping


def test_actor_uses_detached_prepass_only_when_weighting_is_enabled():
    actor_source = (
        module_path.parents[2] / "workers" / "actor" / "megatron_actor.py"
    ).read_text(encoding="utf-8")
    enabled = actor_source.index("if self.config.router_shift_weighting.enabled:", actor_source.index("def update_policy"))
    no_grad = actor_source.index("with torch.no_grad():", enabled)
    prepass = actor_source.index("forward_only=True", no_grad)
    training = actor_source.index("metric_micro_batch = self.forward_backward_batch", prepass)
    assert enabled < no_grad < prepass < training
    assert 'policy_loss_kwargs["router_shift_gamma"] = data["router_shift_gamma"]' in actor_source
    assert 'loss_mode not in {"vanilla", "gspo"}' in actor_source


def test_alignment_scope_clears_replay_before_current_prepass_and_update():
    actor_source = (
        module_path.parents[2] / "workers" / "actor" / "megatron_actor.py"
    ).read_text(encoding="utf-8")
    update = actor_source.index("def update_policy")
    scope = actor_source.index("replay_current_update = self.config.router_replay.should_replay_current_update", update)
    replay = actor_source.index("RouterReplay.set_global_router_replay_action", scope)
    natural = actor_source.index("RouterReplay.clear_global_router_replay_action()", replay)
    prepass = actor_source.index("forward_only=True", natural)
    training = actor_source.index("metric_micro_batch = self.forward_backward_batch", prepass)
    assert scope < replay < natural < prepass < training


def test_gspo_rs_disabled_and_gamma_one_match_loss_metrics_and_gradient():
    old = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    advantages = torch.tensor([[1.0, 1.0, 1.0], [-1.0, -1.0, -1.0]])
    mask = torch.tensor([[True, True, False], [True, True, True]])

    results = []
    for gamma in (None, torch.ones_like(old)):
        current = torch.tensor([[1e-4, -1e-4, 9.0], [2e-4, 2e-4, -1e-4]], requires_grad=True)
        loss, metrics = _gspo(current, old, advantages, mask, gamma)
        loss.backward()
        results.append((loss.detach(), metrics, current.grad.detach()))

    torch.testing.assert_close(results[0][0], results[1][0])
    torch.testing.assert_close(results[0][2], results[1][2])
    assert results[0][1] == results[1][1]


def test_gspo_rs_changes_ratio_before_masked_sequence_mean_and_detaches_gamma():
    old = torch.zeros((2, 3))
    current = torch.zeros((2, 3), requires_grad=True)
    gamma = torch.tensor([[0.9, 0.8, 0.1], [0.8, 0.8, 0.8]], requires_grad=True)
    mask = torch.tensor([[True, True, False], [True, True, True]])
    advantages = torch.ones_like(old)

    loss, metrics = _gspo(current, old, advantages, mask, gamma)
    loss.backward()

    expected_ratios = torch.tensor([(0.9 * 0.8) ** 0.5, 0.8])
    expected_loss = -expected_ratios.mean()
    torch.testing.assert_close(loss, expected_loss)
    assert gamma.grad is None
    assert current.grad is not None
    assert metrics["actor/ppo_kl"] == 0.0
    assert metrics["actor/pg_clipfrac"] == 0.0


def test_gspo_uses_sequence_clipping_with_paper_bounds_not_token_clipping():
    old = torch.zeros((1, 2))
    mask = torch.ones_like(old, dtype=torch.bool)
    advantages = torch.ones_like(old)
    current = torch.tensor([[math.log(1.001), math.log(0.9996)]], requires_grad=True)

    loss, metrics = _gspo(current, old, advantages, mask)
    sequence_ratio = math.sqrt(1.001 * 0.9996)
    torch.testing.assert_close(loss, torch.tensor(-min(sequence_ratio, 1.0004)))
    assert metrics["actor/pg_clipfrac"] == 0.0


def test_gspo_uses_asymmetric_paper_clip_bounds_at_sequence_level():
    mask = torch.ones((2, 2), dtype=torch.bool)
    old = torch.zeros((2, 2))
    current = torch.tensor([[math.log(1.001), math.log(1.001)], [math.log(0.999), math.log(0.999)]])
    advantages = torch.tensor([[1.0, 1.0], [-1.0, -1.0]])

    loss, metrics = _gspo(current, old, advantages, mask)

    torch.testing.assert_close(loss, torch.tensor((-1.0004 + 0.9997) / 2))
    assert metrics["actor/pg_clipfrac"] == 1.0
