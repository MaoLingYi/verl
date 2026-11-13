"""Global valid-response-token mean on the V1.8.2 expert-cluster objective."""
import torch

from verl.trainer.ppo.eu_derpo_v18 import TokenAwareStatistics, cluster_statistics

from verl.trainer.ppo.eu_derpo_v182 import structural_coefficients


def _numerator_weights(stats, routes, response_mask):
    lengths = response_mask.bool().sum(dim=1, keepdim=True)
    return structural_coefficients(stats, routes, response_mask) * lengths


@torch.no_grad()
def schedule_coefficients(stats, routes, response_mask, global_tokens, microbatches, dense_dp):
    if global_tokens <= 0:
        raise ValueError("V1.8.3 requires a nonempty accepted token population")
    weights = _numerator_weights(stats, routes, response_mask)
    gamma = weights * stats.token_weight * (microbatches * dense_dp / global_tokens)
    if not torch.isfinite(gamma).all():
        raise FloatingPointError("V1.8.3 non-finite schedule coefficient")
    return gamma


@torch.no_grad()
def objective_numerator(stats, routes, response_mask):
    weights = _numerator_weights(stats, routes, response_mask)
    numerator = (weights * stats.token_surrogate.masked_fill(~response_mask.bool(), 0)).sum(dtype=torch.float64)
    if not torch.isfinite(numerator):
        raise FloatingPointError("V1.8.3 non-finite objective numerator")
    return numerator

@torch.no_grad()
def post_backward_statistics(rows, behavior, advantages, response_mask, num_experts, delta, diagnostics_only, anchor):
    """Consume CPU actual-F rows in minibatch order; a response contains every member of each cluster."""
    batch = len(response_mask)
    fields = ("count", "rho", "divergence", "mask", "advantage", "directional_outward", "log_arithmetic_ratio")
    compact = {}
    numerator = torch.zeros((), dtype=torch.float64)
    seen = 0
    for row, (routes, current) in enumerate(rows):
        span = slice(row, row + 1)
        routes = routes.unsqueeze(0)
        mask = response_mask[span].cpu()
        stats = cluster_statistics(current.unsqueeze(0), behavior[span].cpu(), advantages[span].cpu(),
                                   mask, routes, num_experts, delta, diagnostics_only, optimization_anchor=anchor)
        numerator += objective_numerator(stats, routes, mask)
        for name in fields:
            value = getattr(stats, name)
            if row == 0:
                compact[name] = value.new_empty((batch, *value.shape[1:]))
            compact[name][span].copy_(value)
        seen += 1
    if seen != batch or not seen:
        raise ValueError("V1.8.3 actual-F cache rows differ from response batch")
    return TokenAwareStatistics(**compact), numerator
