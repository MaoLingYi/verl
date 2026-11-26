"""Sampled-token Binary-TV DPPO (arXiv:2602.04879v3, Eqs. 12, 13 and 23)."""

import math

import torch


class DPPOStats:
    """Accumulate valid-token sums across microbatches, then reduce over dense DP."""

    def __init__(self):
        self.sums = None

    @torch.no_grad()
    def add(self, valid, tv, outward, blocked, advantages, weight, capped):
        positive = valid & (advantages > 0)
        negative = valid & (advantages < 0)
        sums = torch.stack((
            valid.sum(), (tv * valid).sum(), (outward & valid).sum(), blocked.sum(),
            (blocked & positive).sum(), positive.sum(), (blocked & negative).sum(), negative.sum(),
            (weight * valid).sum(), (capped & valid).sum(),
        )).double()
        if self.sums is None:
            self.sums = sums
        else:
            self.sums += sums

    def metrics(self, group=None):
        if self.sums is None:
            return {}
        sums = self.sums.clone()
        if group is not None:
            torch.distributed.all_reduce(sums, group=group)
        n, tv, outward, blocked, pos_block, pos, neg_block, neg, weight, capped = sums.tolist()
        denom = max(n, 1)
        return {
            "dppo/valid_token_count": n,
            "dppo/binary_tv_mean": tv / denom,
            "dppo/outward_fraction": outward / denom,
            "dppo/mask_fraction": blocked / denom,
            "dppo/mask_fraction_positive": pos_block / max(pos, 1),
            "dppo/mask_fraction_negative": neg_block / max(neg, 1),
            "dppo/importance_weight_mean": weight / denom,
            "dppo/importance_cap_fraction": capped / denom,
        }


def binary_tv_token_losses(log_prob, behavior, advantages, response_mask, delta, cap, stats):
    if behavior is None:
        raise ValueError("dppo_binary_tv requires SGLang rollout_log_probs")
    if not (log_prob.shape == behavior.shape == advantages.shape == response_mask.shape):
        raise ValueError("DPPO inputs must have aligned response-token shapes")
    valid = response_mask.bool()
    dtype = torch.float64 if log_prob.dtype == torch.float64 else torch.float32
    current = log_prob.to(dtype).masked_fill(~valid, 0)
    with torch.no_grad():
        anchor = behavior.to(dtype).masked_fill(~valid, 0)
        adv = advantages.to(dtype).masked_fill(~valid, 0)
        if not (torch.isfinite(current) & (current <= 0) & torch.isfinite(anchor)
                & (anchor <= 0) & torch.isfinite(adv)).all():
            raise ValueError("DPPO requires finite valid advantages and finite nonpositive log probabilities")
        log_ratio = current - anchor
        tv = (current.exp() - anchor.exp()).abs()
        outward = ((adv > 0) & (log_ratio > 0)) | ((adv < 0) & (log_ratio < 0))
        blocked = outward & (tv > delta)
        log_cap = math.log(cap)
        weight = log_ratio.clamp(max=log_cap).exp()
        coefficient = (weight * adv).masked_fill(blocked, 0)
        stats.add(valid, tv, outward, blocked, adv, weight, log_ratio > log_cap)
    return -coefficient * current
