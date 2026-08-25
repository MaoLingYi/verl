from __future__ import annotations

import torch


def adjust_log_ratio_with_router_shift(
    log_ratio: torch.Tensor,
    gamma: torch.Tensor | None,
    gamma_min: float,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Apply detached RSPO-style router-shift weighting before base-policy clipping."""
    if gamma is None:
        return log_ratio, None
    if gamma.shape != log_ratio.shape:
        raise ValueError("router-shift gamma must match the token log-ratio shape")
    if not 0.0 < gamma_min <= 1.0:
        raise ValueError("router-shift gamma_min must be in (0, 1]")
    gamma = gamma.detach().float()
    if not torch.isfinite(gamma).all() or torch.any(gamma <= 0) or torch.any(gamma > 1.0 + 1e-6):
        raise ValueError("router-shift gamma must be finite and in (0, 1]")
    weight = gamma.clamp_min(gamma_min)
    return log_ratio + weight.log(), weight
