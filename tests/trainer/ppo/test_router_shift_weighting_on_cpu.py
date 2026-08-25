import importlib.util
import math
from pathlib import Path

import torch

module_path = Path(__file__).parents[3] / "verl" / "trainer" / "ppo" / "router_shift_weighting.py"
spec = importlib.util.spec_from_file_location("router_shift_weighting_under_test", module_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
adjust_log_ratio_with_router_shift = module.adjust_log_ratio_with_router_shift


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
