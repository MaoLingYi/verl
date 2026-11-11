"""Global valid-response-token mean on the V1.8.2 expert-cluster objective."""
import torch

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
