"""Frozen EU-DERPO V1.2 tensor math; no distributed or model dependencies."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import torch
import torch.nn.functional as F


def policy_prepass_tensors(
    log_probs: torch.Tensor, entropys: torch.Tensor, eu_derpo_enabled: bool
) -> dict[str, torch.Tensor]:
    """Preserve legacy three-logprob plumbing; actual F remains EU's current-policy authority."""
    tensors = {"old_log_probs": log_probs, "entropys": entropys}
    if eu_derpo_enabled:
        tensors["current_log_probs"] = log_probs.clone()
    return tensors


@dataclass
class ClusterStatistics:
    count: torch.Tensor
    rho: torch.Tensor
    divergence: torch.Tensor
    mask: torch.Tensor
    advantage: torch.Tensor
    rho_raw: torch.Tensor | None = None
    divergence_engine: torch.Tensor | None = None
    divergence_raw: torch.Tensor | None = None
    delta_engine: torch.Tensor | None = None
    delta_update: torch.Tensor | None = None
    delta_total: torch.Tensor | None = None
    rho_align: torch.Tensor | None = None
    divergence_align: torch.Tensor | None = None

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


def dppo_tv_valid_mask(
    behavior_log_prob: torch.Tensor,
    current_log_prob: torch.Tensor,
    advantages: torch.Tensor,
    delta_low: float,
    delta_high: float,
) -> torch.Tensor:
    """Return the detached token-level DPPO-Binary-TV credit mask."""

    probability = current_log_prob.exp()
    behavior_probability = behavior_log_prob.exp()
    valid_positive = (probability - behavior_probability) <= delta_high
    valid_negative = (probability - behavior_probability) >= -delta_low
    return torch.where(advantages > 0, valid_positive, valid_negative).detach()


def cluster_statistics(
    current_logp: torch.Tensor,
    behavior_logp: torch.Tensor | None,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    routes: torch.Tensor,
    num_experts: int,
    delta_e: float | None,
    diagnostics_only: bool = False,
    aligned_old_logp: torch.Tensor | None = None,
    optimization_anchor: str = "rollout",
) -> ClusterStatistics:
    if behavior_logp is None:
        raise ValueError("EU-DERPO requires rollout_log_probs; old-policy recompute is not a behavior fallback")
    current = _finite("current sampled-token logprob", current_logp)
    behavior = _finite("rollout behavior sampled-token logprob", behavior_logp)
    if current.shape != behavior.shape or current.shape != response_mask.shape:
        raise ValueError("current, rollout behavior, and response mask shapes must match")
    aligned_old = None
    if aligned_old_logp is not None:
        aligned_old = _finite("aligned-old sampled-token logprob", aligned_old_logp)
        if aligned_old.shape != current.shape:
            raise ValueError("aligned-old, current, and rollout behavior logprob shapes must match")
    if optimization_anchor not in {"rollout", "aligned_old"}:
        raise ValueError("EU-DERPO optimization_anchor must be rollout or aligned_old")
    if optimization_anchor == "aligned_old" and aligned_old is None:
        raise ValueError("EU-DERPO aligned-old optimization requires aligned_old_logp")
    if routes.ndim != 4 or routes.shape[:2] != current.shape:
        raise ValueError("routes must have shape [batch, response, local_layers, topk]")
    routes = routes.detach()
    mask = response_mask.bool()
    if routes.numel() and (routes.min() < 0 or routes.max() >= num_experts):
        raise ValueError("route expert id is out of range")
    if routes.shape[-1] > 1:
        duplicate = routes.sort(-1).values.diff(dim=-1).eq(0).any(dim=-1) & mask[:, :, None]
        if duplicate.any():
            duplicate_count = int(duplicate.sum().item())
            first = int(duplicate.flatten().max(dim=0).indices.item())
            sample, remainder = divmod(first, duplicate.shape[1] * duplicate.shape[2])
            token, layer = divmod(remainder, duplicate.shape[2])
            raise ValueError(
                "EU-DERPO requires unique selected token-expert edges: "
                f"duplicate_valid_positions={duplicate_count}, "
                f"first=(sample={sample}, token={token}, layer={layer}), "
                f"routes={routes[sample, token, layer].tolist()}"
            )
    if not diagnostics_only and delta_e is None:
        raise ValueError("EU-DERPO requires explicit delta_e")

    batch, tokens, layers, topk = routes.shape
    advantage = response_advantage(advantages, mask)
    count = torch.zeros((batch, layers, num_experts), dtype=torch.float32, device=current.device)
    behavior_log_sum = torch.zeros_like(count)
    behavior_divergence_sum = torch.zeros_like(count)
    engine_log_sum = align_log_sum = None
    engine_divergence_sum = align_divergence_sum = None
    if aligned_old is not None:
        engine_log_sum = torch.zeros_like(count)
        align_log_sum = torch.zeros_like(count)
        engine_divergence_sum = torch.zeros_like(count)
        align_divergence_sum = torch.zeros_like(count)
    behavior_log_ratio = current - behavior
    behavior_binary_tv = (behavior.exp() - current.exp()).abs()
    engine_log_ratio = align_log_ratio = None
    engine_binary_tv = align_binary_tv = None
    if aligned_old is not None:
        engine_log_ratio = aligned_old - behavior
        align_log_ratio = current - aligned_old
        engine_binary_tv = (behavior.exp() - aligned_old.exp()).abs()
        align_binary_tv = (aligned_old.exp() - current.exp()).abs()
    valid = mask[:, :, None].expand(batch, tokens, topk).reshape(batch, -1).float()
    for layer in range(layers):
        edge = routes[:, :, layer, :].long()
        index = edge.reshape(batch, -1)
        count[:, layer].scatter_add_(1, index, valid)
        accumulators = [
            (behavior_log_sum, behavior_log_ratio),
            (behavior_divergence_sum, behavior_binary_tv),
        ]
        if aligned_old is not None:
            accumulators.extend((
                (engine_log_sum, engine_log_ratio),
                (align_log_sum, align_log_ratio),
                (engine_divergence_sum, engine_binary_tv),
                (align_divergence_sum, align_binary_tv),
            ))
        for target, values in accumulators:
            edge_values = values[:, :, None].expand_as(edge).reshape(batch, -1) * valid
            target[:, layer].scatter_add_(1, index, edge_values)

    active = count > 0
    safe_count = count.masked_fill(~active, 1.0)
    delta_behavior = (behavior_log_sum / safe_count).masked_fill(~active, 0.0)
    rho_behavior = delta_behavior.exp().masked_fill(~active, 0.0)
    divergence_behavior = (behavior_divergence_sum / safe_count).masked_fill(~active, 0.0)
    if aligned_old is None:
        rho = rho_behavior
        divergence = divergence_behavior
        if diagnostics_only:
            dppo_mask = active
        else:
            adv = advantage[:, None, None]
            outward = (((adv > 0) & (rho > 1)) | ((adv < 0) & (rho < 1))) & (divergence > delta_e)
            dppo_mask = active & ~outward
        # Keep the V1.2.1/default-off path allocation- and return-compatible.
        return ClusterStatistics(
            count,
            _finite("rho_E", rho),
            _finite("D_E", divergence),
            dppo_mask.detach(),
            advantage,
        )
    else:
        delta_engine = (engine_log_sum / safe_count).masked_fill(~active, 0.0)
        delta_align = (align_log_sum / safe_count).masked_fill(~active, 0.0)
        if not torch.allclose(delta_behavior[active], (delta_align + delta_engine)[active], rtol=2e-6, atol=2e-6):
            raise RuntimeError("EU-DERPO delta decomposition failed")
        rho_align = delta_align.exp().masked_fill(~active, 0.0)
        divergence_engine = (engine_divergence_sum / safe_count).masked_fill(~active, 0.0)
        divergence_align = (align_divergence_sum / safe_count).masked_fill(~active, 0.0)
        if optimization_anchor == "aligned_old":
            rho, divergence = rho_align, divergence_align
        else:
            rho, divergence = rho_behavior, divergence_behavior
    if diagnostics_only:
        dppo_mask = active
    else:
        adv = advantage[:, None, None]
        outward = (((adv > 0) & (rho > 1)) | ((adv < 0) & (rho < 1))) & (divergence > delta_e)
        dppo_mask = active & ~outward
    return ClusterStatistics(
        count,
        _finite("rho_E", rho),
        _finite("D_E", divergence),
        dppo_mask.detach(),
        advantage,
        _finite("rho_E_beh", rho_behavior),
        _finite("D_eng", divergence_engine),
        _finite("D_beh", divergence_behavior),
        _finite("Delta_eng", delta_engine),
        _finite("Delta_align", delta_align),
        _finite("Delta_beh", delta_behavior),
        _finite("rho_E_align", rho_align),
        _finite("D_align", divergence_align),
    )


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


def local_rms_routing_utility(
    actual_alpha: torch.Tensor, sensitivity: torch.Tensor, eps_u: float = 1.0e-6
) -> torch.Tensor:
    """V1.5 local RMS-normalized u^R on one selected support."""

    utility = centered_routing_utility(actual_alpha, sensitivity)
    alpha = actual_alpha.detach().float()
    scale = (alpha * utility.square()).sum(-1, keepdim=True).add(float(eps_u)).sqrt()
    return _finite("local RMS-normalized relative routing utility", utility / scale).detach()


class UtilityHistoryState:
    """Detached response-level utility state used by the next rollout."""

    def __init__(self, n, s, q, mu, sigma, version):
        self.n = n
        self.s = s
        self.q = q
        self.mu = mu
        self.sigma = sigma
        self.version = int(version)

    def state_dict(self) -> dict[str, object]:
        return {
            "n": self.n.detach().cpu(),
            "s": self.s.detach().cpu(),
            "q": self.q.detach().cpu(),
            "mu": self.mu.detach().cpu(),
            "sigma": self.sigma.detach().cpu(),
            "version": int(self.version),
        }

    @classmethod
    def from_state_dict(cls, values: dict[str, object]) -> "UtilityHistoryState":
        state = cls(
            *(torch.as_tensor(values[name]).detach().float().cpu() for name in ("n", "s", "q", "mu", "sigma")),
            version=int(values["version"]),
        )
        validate_utility_history_state(state)
        return state


def utility_history_from_response_observations(
    response_sum: torch.Tensor,
    response_count: torch.Tensor,
    *,
    version: int,
) -> UtilityHistoryState:
    """Reduce response×layer×expert token sums to frozen N/S/Q moments."""

    if response_sum.shape != response_count.shape or response_sum.ndim != 3:
        raise ValueError("V1.5 response utility tensors must share [response, layer, expert] shape")
    observed = response_count > 0
    response_utility = response_sum / response_count.clamp_min(1)
    n = observed.sum(0).float()
    s = (response_utility * observed).sum(0)
    q = (response_utility.square() * observed).sum(0)
    return utility_history_from_moments(n, s, q, version=version)


def utility_history_from_moments(
    n: torch.Tensor, s: torch.Tensor, q: torch.Tensor, *, version: int
) -> UtilityHistoryState:
    """Convert detached global N/S/Q moments to frozen mu/sigma state."""

    if n.shape != s.shape or n.shape != q.shape or tuple(n.shape) != (48, 128):
        raise ValueError("V1.5 N/S/Q moments must be [48, 128]")
    n, s, q = n.float(), s.float(), q.float()
    mu = torch.where(n > 0, s / n.clamp_min(1), torch.zeros_like(s))
    numerator = (q - s.square() / n.clamp_min(1)).clamp_min(0)
    variance = torch.where(n >= 2, numerator / (n - 1).clamp_min(1), torch.ones_like(numerator))
    sigma = variance.sqrt()
    state = UtilityHistoryState(
        n.detach().cpu(), s.detach().cpu(), q.detach().cpu(), mu.detach().cpu(), sigma.detach().cpu(), int(version)
    )
    validate_utility_history_state(state)
    return state


def validate_utility_history_state(state: UtilityHistoryState) -> None:
    for name in ("n", "s", "q", "mu", "sigma"):
        value = getattr(state, name)
        if tuple(value.shape) != (48, 128) or not torch.isfinite(value).all():
            raise ValueError(f"V1.5 utility state {name} must be finite [48, 128]")
    if (state.n < 0).any() or (state.sigma < 0).any() or state.version < 0:
        raise ValueError("V1.5 utility state has invalid counts, dispersion, or version")


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
    local_groups = groups - groups[0]
    unique = torch.unique_consecutive(local_groups)
    expected = torch.arange(unique.numel(), dtype=torch.long)
    if not torch.equal(unique, expected) or not torch.equal(local_groups, local_groups.sort().values):
        raise RuntimeError("EU-DERPO prompt groups must be contiguous and ordered on each dense-DP rank")
    counts = torch.bincount(local_groups)
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


def route_match_fraction(expected: torch.Tensor, actual: torch.Tensor, valid: torch.Tensor) -> float:
    """Set-wise Top-K route match on valid response/action positions without asserting equality."""
    if expected.shape != actual.shape or valid.shape != expected.shape[:2]:
        raise RuntimeError("EU-DERPO route-match shape contract failed")
    selected = valid.detach().bool()[:, :, None].expand(expected.shape[:-1])
    equal = expected.detach().sort(-1).values.eq(actual.detach().sort(-1).values).all(-1)
    if not selected.any():
        raise RuntimeError("EU-DERPO route-match probe has no valid response/action position")
    return equal[selected].float().mean().item()


def prepare_v13_rollout_routes(
    routes: torch.Tensor,
    response_mask: torch.Tensor,
    response_length: int,
    *,
    num_layers: int = 48,
    topk: int = 8,
    num_experts: int = 128,
) -> tuple[torch.Tensor, dict[str, int]]:
    """Release the full R3 payload in favor of one response-only uint8 diagnostic copy."""
    # A sampled-token logprob at response offset t is produced by the router at
    # the preceding input position (last prompt token for t=0).
    response_routes = routes[:, -response_length - 1 : -1]
    if response_routes.ndim != 4 or response_routes.shape[2:] != (num_layers, topk):
        raise RuntimeError(
            f"EU-DERPO V1.3 rollout route shape must be [batch,response,{num_layers},{topk}], "
            f"got {response_routes.shape}"
        )
    valid = response_mask.detach().bool()
    if valid.shape != response_routes.shape[:2]:
        raise RuntimeError("EU-DERPO V1.3 rollout route mask shape mismatch")
    selected = response_routes[valid]
    if selected.numel() == 0 or selected.min() < 0 or selected.max() >= num_experts:
        raise RuntimeError(f"EU-DERPO V1.3 rollout routes require valid global expert ids in [0,{num_experts - 1}]")
    if selected.sort(-1).values.diff(dim=-1).eq(0).any():
        raise RuntimeError("EU-DERPO V1.3 rollout Top-K contains duplicate expert ids")
    compact = response_routes.to(torch.uint8).contiguous()
    return compact, {
        "rollout_route_min_expert_id": int(selected.min().item()),
        "rollout_route_max_expert_id": int(selected.max().item()),
        "rollout_route_topk_width": compact.shape[-1],
        "rollout_route_layer_count": compact.shape[-2],
        "rollout_route_valid_count": int(valid.sum().item()),
        "rollout_route_host_bytes": compact.numel() * compact.element_size(),
    }


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
    valid = response_mask.bool()[:, :, None].expand(batch, tokens, topk).reshape(batch, -1).float()
    for layer in range(layers):
        index = routes[:, :, layer].long().reshape(batch, -1)
        counts[:, layer].scatter_add_(1, index, valid)
        sums[:, layer].scatter_add_(1, index, utility[:, :, layer].reshape(batch, -1) * valid)
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
