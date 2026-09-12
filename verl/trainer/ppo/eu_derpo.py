"""Frozen EU-DERPO V1.2 tensor math; no distributed or model dependencies."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class ClusterStatistics:
    count: torch.Tensor
    rho: torch.Tensor
    divergence: torch.Tensor
    mask: torch.Tensor
    advantage: torch.Tensor

    @property
    def active(self) -> torch.Tensor:
        return self.count > 0


def _finite(name: str, value: torch.Tensor) -> torch.Tensor:
    value = value.float()
    if not torch.isfinite(value).all():
        raise FloatingPointError(f"EU-DERPO {name} contains NaN or Inf")
    return value


def response_advantage(advantages: torch.Tensor, response_mask: torch.Tensor) -> torch.Tensor:
    advantages = _finite("advantages", advantages)
    mask = response_mask.bool()
    count = mask.sum(1)
    if (count == 0).any():
        raise ValueError("EU-DERPO requires at least one valid response token per sample")
    result = (advantages * mask).sum(1) / count
    deviation = ((advantages - result[:, None]).abs() * mask).amax(1)
    if (deviation > 1.0e-6).any():
        raise ValueError("EU-DERPO V1.2 requires one response-level GRPO advantage A_i")
    return result.detach()


def cluster_statistics(
    current_logp: torch.Tensor,
    behavior_logp: torch.Tensor | None,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    routes: torch.Tensor,
    num_experts: int,
    delta_e: float | None,
    diagnostics_only: bool = False,
) -> ClusterStatistics:
    if behavior_logp is None:
        raise ValueError("EU-DERPO requires rollout_log_probs; old-policy recompute is not a behavior fallback")
    current = _finite("current sampled-token logprob", current_logp)
    behavior = _finite("rollout behavior sampled-token logprob", behavior_logp)
    if current.shape != behavior.shape or current.shape != response_mask.shape:
        raise ValueError("current, rollout behavior, and response mask shapes must match")
    if routes.ndim != 4 or routes.shape[:2] != current.shape:
        raise ValueError("routes must have shape [batch, response, local_layers, topk]")
    routes = routes.detach()
    if routes.numel() and (routes.min() < 0 or routes.max() >= num_experts):
        raise ValueError("route expert id is out of range")
    if routes.shape[-1] > 1 and (routes.sort(-1).values.diff(dim=-1) == 0).any():
        raise ValueError("EU-DERPO requires unique selected token-expert edges")
    if not diagnostics_only and delta_e is None:
        raise ValueError("EU-DERPO requires explicit delta_e")

    batch, tokens, layers, topk = routes.shape
    mask = response_mask.bool()
    advantage = response_advantage(advantages, mask)
    count = torch.zeros((batch, layers, num_experts), dtype=torch.float32, device=current.device)
    log_sum = torch.zeros_like(count)
    divergence_sum = torch.zeros_like(count)
    log_ratio = current - behavior
    binary_tv = (behavior.exp() - current.exp()).abs()
    batch_offset = torch.arange(batch, device=current.device)[:, None, None] * num_experts
    valid = mask[:, :, None].expand(batch, tokens, topk)
    for layer in range(layers):
        edge = routes[:, :, layer, :].long()
        flat_index = (edge + batch_offset).reshape(-1)
        valid_flat = valid.reshape(-1).float()
        count[:, layer].view(-1).scatter_add_(0, flat_index, valid_flat)
        log_values = log_ratio[:, :, None].expand_as(edge).reshape(-1) * valid_flat
        div_values = binary_tv[:, :, None].expand_as(edge).reshape(-1) * valid_flat
        log_sum[:, layer].view(-1).scatter_add_(0, flat_index, log_values)
        divergence_sum[:, layer].view(-1).scatter_add_(0, flat_index, div_values)

    active = count > 0
    safe_count = count.masked_fill(~active, 1.0)
    rho = (log_sum / safe_count).exp().masked_fill(~active, 0.0)
    divergence = (divergence_sum / safe_count).masked_fill(~active, 0.0)
    if diagnostics_only:
        dppo_mask = active
    else:
        adv = advantage[:, None, None]
        outward = (((adv > 0) & (rho > 1)) | ((adv < 0) & (rho < 1))) & (divergence > delta_e)
        dppo_mask = active & ~outward
    return ClusterStatistics(count, _finite("rho_E", rho), _finite("D_E", divergence), dppo_mask.detach(), advantage)


def edppo_token_coefficients(
    stats: ClusterStatistics,
    routes: torch.Tensor,
    response_mask: torch.Tensor,
    total_active_per_sample: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    routes = routes.detach()
    batch, tokens, layers, _ = routes.shape
    active_total = stats.active.sum((1, 2)).float() if total_active_per_sample is None else total_active_per_sample.float()
    if (active_total <= 0).any():
        raise ValueError("EU-DERPO sample has no active Expert cluster")
    coefficient = torch.zeros((batch, tokens), dtype=torch.float32, device=routes.device)
    local_objective = torch.zeros((), dtype=torch.float32, device=routes.device)
    for layer in range(layers):
        edge = routes[:, :, layer, :].long()
        count = stats.count[:, layer].gather(1, edge.reshape(batch, -1)).reshape_as(edge)
        count = count.masked_fill(count == 0, 1.0)
        rho = stats.rho[:, layer].gather(1, edge.reshape(batch, -1)).reshape_as(edge)
        keep = stats.mask[:, layer].gather(1, edge.reshape(batch, -1)).reshape_as(edge)
        # Megatron's legacy pipeline schedule averages the returned loss over microbatches.
        scale = stats.advantage[:, None, None] * keep / (active_total[:, None, None] * count)
        coefficient += (scale * rho.detach()).sum(-1)
        local_objective += (
            stats.advantage[:, None] * stats.mask[:, layer] * stats.rho[:, layer] / active_total[:, None]
        ).sum() / batch
    coefficient *= response_mask.bool()
    return _finite("E-DPPO token coefficient", coefficient).detach(), local_objective


def centered_routing_utility(actual_alpha: torch.Tensor, sensitivity: torch.Tensor) -> torch.Tensor:
    """V1.2 atomic credit: u_R = s - sum(alpha * s), detached after main backward."""
    alpha = _finite("actual selected-softmax alpha", actual_alpha)
    sensitivity = _finite("E-DPPO alpha sensitivity", sensitivity)
    if alpha.shape != sensitivity.shape:
        raise ValueError("EU-DERPO alpha and sensitivity shapes must match")
    if (alpha <= 0).any():
        raise FloatingPointError("EU-DERPO selected alpha must be strictly positive")
    if not torch.isclose(alpha.sum(-1), torch.ones_like(alpha[..., 0]), rtol=2e-6, atol=2e-6).all():
        raise FloatingPointError("EU-DERPO actual selected-softmax alpha does not sum to one")
    baseline = (alpha * sensitivity).sum(-1, keepdim=True)
    return _finite("relative routing utility", sensitivity - baseline).detach()


def aime_accuracy_values(reward_extra_info: dict[str, list], scores: list[float]) -> list[float]:
    """Return the AIME scorer's explicit correctness field; never infer it from reward sign."""
    if "acc" not in reward_extra_info:
        raise RuntimeError("AIME24 validation history requires the scorer's explicit acc field")
    accuracy = [float(value) for value in reward_extra_info["acc"]]
    if len(accuracy) != len(scores):
        raise RuntimeError("AIME24 acc and reward lengths differ")
    if any(not torch.isfinite(torch.tensor(value)) or value not in (0.0, 1.0) for value in accuracy):
        raise RuntimeError("AIME24 acc must be finite binary correctness")
    return accuracy


def append_aime_history(path: str, record: dict[str, object]) -> None:
    path = os.path.abspath(os.path.expanduser(path))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")


def audit_optimizer_minibatches(expanded_batch: int, epochs: int, normalized_minibatch: int) -> int:
    if min(expanded_batch, epochs, normalized_minibatch) <= 0:
        raise ValueError("EU-DERPO optimizer mini-batch sizes and epochs must be positive")
    if expanded_batch % normalized_minibatch:
        raise RuntimeError("EU-DERPO expanded batch is not divisible by the normalized optimizer mini-batch")
    return expanded_batch // normalized_minibatch * epochs


def validate_prompt_groups(prompt_group: torch.Tensor, rollout_n: int) -> torch.Tensor:
    groups = prompt_group.detach().cpu().long()
    if groups.ndim != 1 or groups.numel() == 0 or rollout_n <= 0 or (groups < 0).any():
        raise RuntimeError("EU-DERPO prompt-group metadata is invalid")
    counts = torch.bincount(groups)
    if (counts == 0).any() or not torch.all(counts == rollout_n):
        raise RuntimeError(f"EU-DERPO prompt groups are incomplete: counts={counts.tolist()}, rollout_n={rollout_n}")
    return counts


def router_auxiliary_loss(
    hidden: torch.Tensor,
    router_weight: torch.Tensor,
    selected_experts: torch.Tensor,
    edge_coefficient: torch.Tensor,
) -> torch.Tensor:
    logits = F.linear(hidden.detach(), router_weight)
    selected_log_q = logits.float().gather(-1, selected_experts.detach().long()).log_softmax(-1)
    coefficient = _finite("utility edge coefficient", edge_coefficient).detach()
    if coefficient.shape != selected_log_q.shape:
        raise ValueError("utility edge coefficient and selected Router probability shapes differ")
    return -(coefficient * selected_log_q).sum()


def validate_recompute_hook_count(actual: int, expected: int) -> None:
    if actual != expected:
        raise RuntimeError(f"EU-DERPO recompute Router hook count mismatch: {actual} != {expected}")


def validate_recompute_edge_counts(actual: torch.Tensor, expected: torch.Tensor) -> None:
    if actual.shape != expected.shape or not torch.equal(actual.detach().long(), expected.detach().long()):
        raise RuntimeError("EU-DERPO recompute duplicated or dropped routing-utility edges")


def assert_same_routes(prepass: torch.Tensor, main: torch.Tensor, valid: torch.Tensor | None = None) -> dict[str, object]:
    if prepass.shape != main.shape:
        raise RuntimeError(f"EU-DERPO route shape mismatch: {prepass.shape} != {main.shape}")
    equal = prepass.detach().sort(-1).values.eq(main.detach().sort(-1).values)
    if valid is not None:
        valid = valid.detach().bool()
        if valid.shape == prepass.shape[:-2]:
            valid = valid[..., None].expand(prepass.shape[:-1])
        if valid.shape != prepass.shape[:-1]:
            raise RuntimeError("EU-DERPO route valid mask shape mismatch")
    else:
        valid = torch.ones(prepass.shape[:-1], dtype=torch.bool, device=prepass.device)
    selected_equal = equal[valid]
    if selected_equal.numel() == 0:
        raise RuntimeError("EU-DERPO route equality has no valid response/action edge")
    metrics = {
        "route_equal_fraction": selected_equal.float().mean().item(),
        "route_mismatch_count": int((~selected_equal).sum().item()),
        "layer_mismatch_count": ((~equal) & valid[..., None]).sum(dim=(0, 1, 3)).tolist(),
    }
    if metrics["route_mismatch_count"]:
        raise RuntimeError(f"EU-DERPO prepass/main natural route mismatch: {metrics}")
    return metrics


def aggregate_edge_utility(
    edge_utility: torch.Tensor, routes: torch.Tensor, response_mask: torch.Tensor, num_experts: int
) -> tuple[torch.Tensor, torch.Tensor]:
    utility = _finite("edge utility", edge_utility)
    routes = routes.detach()
    if utility.shape != routes.shape:
        raise ValueError("edge utility and routes must have identical [batch, response, layers, topk] shape")
    batch, tokens, layers, topk = routes.shape
    sums = torch.zeros((batch, layers, num_experts), dtype=torch.float32, device=utility.device)
    counts = torch.zeros_like(sums)
    offset = torch.arange(batch, device=utility.device)[:, None, None] * num_experts
    valid = response_mask.bool()[:, :, None].expand(batch, tokens, topk).reshape(-1).float()
    for layer in range(layers):
        index = (routes[:, :, layer].long() + offset).reshape(-1)
        counts[:, layer].view(-1).scatter_add_(0, index, valid)
        sums[:, layer].view(-1).scatter_add_(0, index, utility[:, :, layer].reshape(-1) * valid)
    return sums, counts


def normalize_group_utility(
    utility_sum: torch.Tensor,
    utility_count: torch.Tensor,
    prompt_group: torch.Tensor,
    eps_u: float,
    min_group_size: int,
    min_std: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    active = utility_count > 0
    utility = utility_sum / utility_count.masked_fill(~active, 1.0)
    normalized = torch.zeros_like(utility)
    skipped = torch.zeros_like(active)
    for group in prompt_group.detach().unique():
        members = prompt_group == group
        for layer in range(utility.shape[1]):
            for expert in range(utility.shape[2]):
                selected = members & active[:, layer, expert]
                values = utility[selected, layer, expert]
                if values.numel() < min_group_size:
                    skipped[selected, layer, expert] = True
                    continue
                std = values.float().std(unbiased=False)
                if not torch.isfinite(std) or std < min_std:
                    skipped[selected, layer, expert] = True
                    continue
                normalized[selected, layer, expert] = (values - values.mean()) / (std + eps_u)
    return _finite("normalized utility", normalized).detach(), skipped.detach()


def distribution_stats(values: torch.Tensor, quantiles: tuple[float, ...]) -> dict[str, float]:
    values = _finite("diagnostic values", values).flatten()
    if values.numel() == 0:
        return {}
    result = {
        "mean": values.mean().item(),
        "std": values.std(unbiased=False).item(),
        "min": values.min().item(),
        "max": values.max().item(),
    }
    result.update({f"p{int(q * 100)}": torch.quantile(values, q).item() for q in quantiles})
    return result
