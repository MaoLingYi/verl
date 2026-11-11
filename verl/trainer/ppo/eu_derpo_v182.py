"""V1.8.2: only the main objective uses pre-mask token-count cluster weights."""
import torch
from verl.trainer.ppo.eu_derpo_v18 import cluster_statistics, capped_surrogate


@torch.no_grad()
def structural_coefficients(stats, routes, response_mask):
    """Per-response A/Z times the number of kept memberships at each token."""
    membership_total = stats.count.sum((1, 2))
    if (membership_total <= 0).any():
        raise ValueError("V1.8.2 response has no active clusters")
    valid = response_mask.bool()
    routes = routes.detach().masked_fill(~valid[:, :, None, None], 0)
    batch, tokens, layers, _ = routes.shape
    structural = stats.count.new_zeros(batch, tokens)
    for layer in range(layers):
        edge = routes[:, :, layer].long()
        per_cluster = stats.mask[:, layer].to(stats.count.dtype)
        structural += per_cluster.gather(1, edge.reshape(batch, -1)).reshape_as(edge).sum(-1)
    structural *= (stats.advantage / membership_total)[:, None] * valid
    return structural


@torch.no_grad()
def edppo_token_coefficients(stats, routes, response_mask):
    """gamma excludes batch scaling: the MB=1 Megatron schedule supplies it once."""
    structural = structural_coefficients(stats, routes, response_mask)
    batch = routes.shape[0]
    coefficient = structural * stats.token_weight
    objective = (structural * stats.token_surrogate).sum() / batch
    if not torch.isfinite(coefficient).all() or not torch.isfinite(objective):
        raise FloatingPointError("V1.8.2 non-finite coefficient/objective")
    return coefficient.detach(), objective.detach()
