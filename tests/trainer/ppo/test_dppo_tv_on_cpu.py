"""CPU autograd checks for the sampled-token DPPO production registry entry."""

from __future__ import annotations

import ast
import importlib.util
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


ROOT = Path(__file__).resolve().parents[3]


def _source_functions(path, names, namespace):
    """Execute unmodified production definitions when the optional trainer stack is absent."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    exec(compile(tree, str(path), "exec"), namespace)


def load_dppo_registry():
    modes = ("vanilla", "dppo_tv", "dppo_binary_tv")
    if importlib.util.find_spec("tensordict") is not None:
        from verl.trainer.ppo.core_algos import get_policy_loss_fn
    else:
        # The actual sampler loss is dependency-light. Only its surrounding veRL package imports TensorDict.
        name = "verl.trainer.ppo.dppo"
        spec = importlib.util.spec_from_file_location(name, ROOT / "verl/trainer/ppo/dppo.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        sys.modules[name] = module
        functional = {"torch": torch}
        _source_functions(ROOT / "verl/utils/torch_functional.py", {"masked_sum", "masked_mean"}, functional)
        namespace = {"POLICY_LOSS_REGISTRY": {}, "torch": torch, "verl_F": SimpleNamespace(**functional),
                     "AlgoConfig": type("AlgoConfig", (), {})}
        _source_functions(
            ROOT / "verl/trainer/ppo/core_algos.py",
            {"register_policy_loss", "get_policy_loss_fn", "agg_loss", *(f"compute_policy_loss_{mode}" for mode in modes)},
            namespace,
        )
        get_policy_loss_fn = namespace["get_policy_loss_fn"]
    config = SimpleNamespace(
        global_batch_info={},
        policy_loss=SimpleNamespace(loss_mode="dppo_binary_tv", dppo_delta_tv=0.2, dppo_importance_cap=5.0),
    )
    return {mode: get_policy_loss_fn(mode) for mode in modes}, config


def test_independent_dispatch_and_baseline_cliphigher_backward():
    registry, config = load_dppo_registry()
    assert registry["dppo_tv"].__name__ == "compute_policy_loss_dppo_tv"
    assert registry["dppo_binary_tv"].__name__ == "compute_policy_loss_dppo_binary_tv"
    config.policy_loss.loss_mode = "vanilla"
    config.clip_ratio, config.clip_ratio_low, config.clip_ratio_high = 0.2, 0.2, 0.27
    config.get = vars(config).get
    ratios = torch.tensor([[1.34, 1.24, 0.70, 0.90]], dtype=torch.double)
    old = torch.full_like(ratios, -2.0)
    current = (old + ratios.log()).requires_grad_()
    advantage = current.new_tensor([[1, 1, -1, -1]])
    loss, _ = registry["vanilla"](old, current, advantage, torch.ones_like(ratios), config=config)
    loss.backward()
    torch.testing.assert_close(current.grad, current.new_tensor([[0, -1.24, 0, 0.90]]) / 4)


def test_behavior_anchored_directional_gradient():
    registry, config = load_dppo_registry()
    config.policy_loss.dppo_delta_tv = 0.25  # Exact binary boundary: 0.5 - 0.25.
    current = torch.tensor(
        [[0.8, 0.125, 0.125, 0.75, 0.5, 0.5], [0.75, 0.0625, 1.0, 1.0, 1.0, 1.0]], dtype=torch.double
    ).log()
    behavior = torch.tensor(
        [[0.25, 0.625, 0.625, 0.125, 0.25, 0.5], [0.125, 1.0, 0.5, 1.0, 1.0, 1.0]], dtype=torch.double
    ).log()
    current[1, 2:] = torch.tensor([-10000.0, math.nan, math.inf, -math.inf])
    behavior[1, 1] = -10000.0
    behavior[1, 3:] = math.nan
    current.requires_grad_()
    behavior.requires_grad_()
    advantage = torch.tensor(
        [[2, -3, 2, -2, 0.5, -2], [0, 3, 4, math.nan, math.inf, -math.inf]],
        dtype=torch.double, requires_grad=True,
    )
    valid = torch.tensor([[1, 1, 1, 1, 1, 1], [1, 1, 1, 0, 0, 0]])
    old = torch.full_like(current, math.nan, requires_grad=True)
    loss, _ = registry["dppo_binary_tv"](old, current, advantage, valid, config=config, rollout_log_probs=behavior)
    assert torch.isfinite(loss)
    loss.backward()
    # By hand: blocked +/-; inward +/- (second is capped); equality; ratio=1;
    # zero advantage; extreme capped ratio; underflow; nonfinite padding.
    numerator = current.new_tensor([[0, 0, -0.4, 10, -1, 2], [0, -15, 0, 0, 0, 0]])
    torch.testing.assert_close(current.grad, numerator / 9)
    assert behavior.grad is None and advantage.grad is None and old.grad is None


def test_microbatch_scaling_and_token_weighted_metrics():
    registry, config = load_dppo_registry()
    from verl.trainer.ppo.dppo import DPPOStats

    stats = DPPOStats()
    config.global_batch_info = {"batch_num_tokens": 8, "dp_size": 2}
    config.policy_loss.dppo_importance_cap = 3.0
    for p, b, advantage, mask, expected in [
        ([0.5, 0.9, 0.9], [0.5, 0.1, 0.1], [1, -1, -1], [1, 0, 0], [-0.25, 0, 0]),
        ([0.5, 0.75, 0.05], [0.25, 0.1, 0.5], [1, -1, -1], [1, 1, 1], [0, 0.75, 0]),
    ]:
        current = torch.tensor([p], dtype=torch.double).log().requires_grad_()
        loss, _ = registry["dppo_binary_tv"](
            None, current, torch.tensor([advantage]), torch.tensor([mask]), config=config,
            rollout_log_probs=torch.tensor([b], dtype=torch.double).log(), dppo_stats=stats,
        )
        loss.backward()
        torch.testing.assert_close(current.grad, current.new_tensor([expected]))
    assert stats.metrics() == pytest.approx({
        "dppo/valid_token_count": 4, "dppo/binary_tv_mean": 1.35 / 4, "dppo/outward_fraction": 0.5,
        "dppo/mask_fraction": 0.5, "dppo/mask_fraction_positive": 0.5, "dppo/mask_fraction_negative": 0.5,
        "dppo/importance_weight_mean": 6.1 / 4, "dppo/importance_cap_fraction": 0.25,
    })


def test_empty_response_and_invalid_anchors():
    registry, config = load_dppo_registry()
    loss_fn = registry["dppo_binary_tv"]
    current = torch.tensor([[math.nan]], requires_grad=True)
    valid = torch.zeros_like(current)
    loss, _ = loss_fn(None, current, current.detach(), valid, config=config, rollout_log_probs=current.detach())
    loss.backward()
    assert loss.item() == 0 and current.grad.item() == 0
    with pytest.raises(ValueError, match="rollout_log_probs"):
        loss_fn(torch.zeros_like(current), current, valid, valid, config=config)
    with pytest.raises(ValueError, match="importance"):
        loss_fn(None, current, valid, valid, config=config, rollout_log_probs=current, rollout_is_weights=valid)
