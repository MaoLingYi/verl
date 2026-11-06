"""V1.8 fixed-support expert-cluster surrogate (V18-6 through V18-16).

Raw log-arithmetic ratio controls direction; each token retains its own cap5
gradient. Neither this statistic nor sampled-token TV is a full-policy bound.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from verl.trainer.ppo.eu_derpo import ClusterStatistics


@dataclass
class TokenAwareStatistics(ClusterStatistics):
    token_weight: torch.Tensor | None = None
    token_surrogate: torch.Tensor | None = None
    log_arithmetic_ratio: torch.Tensor | None = None


def capped_surrogate(log_ratio):
    """H5(exp(delta)), stable even when exp(delta) would overflow."""
    boundary = math.log(5.0)
    return torch.where(log_ratio <= boundary, log_ratio.clamp(max=boundary).exp(),
                       5.0 * (1.0 + log_ratio - boundary))


@torch.no_grad()
def cluster_statistics(current_logp, behavior_logp, advantages, response_mask, routes,
                       num_experts, delta_e, diagnostics_only=False,
                       aligned_old_logp=None, optimization_anchor="rollout"):
    if behavior_logp is None or aligned_old_logp is not None or optimization_anchor != "rollout":
        raise ValueError("V1.8 requires actual rollout behavior, without old recompute")
    if diagnostics_only or delta_e is None or not math.isfinite(delta_e) or delta_e < 0:
        raise ValueError("V1.8 requires the live expert Binary-TV mask")
    if not (current_logp.shape == behavior_logp.shape == advantages.shape == response_mask.shape):
        raise ValueError("V1.8 response tensor shapes differ")
    if routes.ndim != 4 or routes.shape[:2] != current_logp.shape or routes.shape[-1] < 1:
        raise ValueError("routes must be [response, token, layer, topk]")
    valid = response_mask.bool()
    dtype = torch.float64 if current_logp.dtype == torch.float64 else torch.float32
    current, behavior, advantage = [x.detach().to(dtype).masked_fill(~valid, 0)
                                  for x in (current_logp, behavior_logp, advantages)]
    if not all(torch.isfinite(x).all() for x in (current, behavior, advantage)):
        raise ValueError("V1.8 non-finite valid token")
    if (current > 0).any() or (behavior > 0).any():
        raise ValueError("V1.8 sampled-token logprobs must be <= 0")
    lengths = valid.sum(1)
    if (lengths == 0).any():
        raise ValueError("V1.8 empty response")
    a = advantage.sum(1) / lengths
    if ((advantage - a[:, None]).masked_fill(~valid, 0).abs() > 1e-6).any():
        raise ValueError("V1.8 requires response-level advantage")
    routes = routes.detach().masked_fill(~valid[:, :, None, None], 0)
    if routes.dtype not in (torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64):
        raise ValueError("V1.8 routes must be integer expert IDs")
    if (routes < 0).any() or (routes >= num_experts).any():
        raise ValueError("V1.8 expert ID out of range")
    if (routes.sort(-1).values.diff(dim=-1).eq(0).any(-1) & valid[:, :, None]).any():
        raise ValueError("V1.8 duplicate selected expert")
    batch, tokens, layers, topk = routes.shape
    count = current.new_zeros(batch, layers, num_experts)
    divergence = torch.zeros_like(count)
    log_mean = torch.zeros_like(count)
    delta = current - behavior
    if not torch.isfinite(delta).all():
        raise ValueError("V1.8 log-ratio overflow")
    tv = (current.exp() - behavior.exp()).abs()
    edge_valid = valid[:, :, None].expand(batch, tokens, topk).reshape(batch, -1)
    for layer in range(layers):
        index = routes[:, :, layer].long().reshape(batch, -1)
        values = delta[:, :, None].expand(batch, tokens, topk).reshape(batch, -1)
        count[:, layer].scatter_add_(1, index, edge_valid.to(dtype))
        maxima = current.new_full((batch, num_experts), -torch.inf)
        maxima.scatter_reduce_(1, index, values.masked_fill(~edge_valid, -torch.inf), reduce="amax")
        maxima = maxima.masked_fill(count[:, layer] == 0, 0)
        sums = torch.zeros_like(maxima)
        shifted = (values - maxima.gather(1, index)).masked_fill(~edge_valid, -torch.inf)
        sums.scatter_add_(1, index, shifted.exp())
        log_mean[:, layer] = (maxima + sums.clamp_min(torch.finfo(dtype).tiny).log()
                             - count[:, layer].clamp_min(1).log())
        edge_tv = tv[:, :, None].expand(batch, tokens, topk).reshape(batch, -1)
        divergence[:, layer].scatter_add_(1, index, edge_tv * edge_valid)
    active = count > 0
    divergence /= count.clamp_min(1)
    log_mean.masked_fill_(~active, 0)
    outward = active & (((a[:, None, None] > 0) & (log_mean > 0)) |
                        ((a[:, None, None] < 0) & (log_mean < 0)))
    keep = active & ~(outward & (divergence > delta_e))
    # Saturate only the diagnostic exp; all direction decisions use raw log_mean.
    rho = log_mean.clamp(max=math.log(torch.finfo(dtype).max)-1).exp().masked_fill(~active, 0)
    return TokenAwareStatistics(count, rho, divergence, keep, a,
        directional_outward=outward, token_weight=delta.clamp(max=math.log(5)).exp(),
        token_surrogate=capped_surrogate(delta), log_arithmetic_ratio=log_mean)


@torch.no_grad()
def edppo_token_coefficients(stats, routes, response_mask, total_active_per_sample=None):
    """gamma excludes batch scaling: the MB=1 Megatron schedule supplies it once."""
    active_total = (stats.active.sum((1, 2)) if total_active_per_sample is None
                    else total_active_per_sample).to(stats.count.dtype)
    if (active_total <= 0).any():
        raise ValueError("V1.8 response has no active clusters")
    valid = response_mask.bool()
    routes = routes.detach().masked_fill(~valid[:, :, None, None], 0)
    batch, tokens, layers, _ = routes.shape
    structural = stats.count.new_zeros(batch, tokens)
    for layer in range(layers):
        edge = routes[:, :, layer].long()
        per_cluster = stats.mask[:, layer] / stats.count[:, layer].clamp_min(1)
        structural += per_cluster.gather(1, edge.reshape(batch, -1)).reshape_as(edge).sum(-1)
    structural *= (stats.advantage / active_total)[:, None] * valid
    coefficient = structural * stats.token_weight
    objective = (structural * stats.token_surrogate).sum() / batch
    if not torch.isfinite(coefficient).all() or not torch.isfinite(objective):
        raise FloatingPointError("V1.8 non-finite coefficient/objective")
    return coefficient.detach(), objective.detach()
