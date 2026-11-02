"""Token DPPO, arXiv:2602.04879v3 Eq.12–14,23 and Appendix E.

No old-policy fallback, PPO clipping, or additional rollout IS correction.
"""
import math
import torch


def token_terms(log_prob, behavior, advantages, response_mask, divergence, delta, cap):
    if behavior is None:
        raise ValueError("DPPO requires real rollout_log_probs; old recompute is not a behavior anchor")
    if divergence not in {"tv", "kl"} or not math.isfinite(delta) or delta < 0:
        raise ValueError("invalid DPPO divergence/delta")
    if not math.isfinite(cap) or cap <= 0:
        raise ValueError("DPPO importance cap must be finite and positive")
    if not (log_prob.shape == behavior.shape == advantages.shape == response_mask.shape):
        raise ValueError("DPPO tensors must have aligned response-token shapes")
    valid = response_mask.bool()
    dtype = torch.float64 if log_prob.dtype == torch.float64 else torch.float32
    current = log_prob.to(dtype).masked_fill(~valid, 0)
    with torch.no_grad():
        b = behavior.to(dtype).masked_fill(~valid, 0)
        a = advantages.to(dtype).masked_fill(~valid, 0)
        if not (torch.isfinite(current[valid]).all() and torch.isfinite(b[valid]).all()
                and torch.isfinite(a[valid]).all()):
            raise ValueError("DPPO non-finite valid response token")
        if (current[valid] > 0).any() or (b[valid] > 0).any():
            raise ValueError("DPPO log probabilities must be <= 0")
        diff = current.detach() - b
        pb, pt = b.exp(), current.detach().exp()
        tv = (pb - pt).abs()
        # Clamp only the Bernoulli KL endpoints; TV and IS retain the original probabilities.
        eps = torch.finfo(dtype).eps
        pb_safe, pt_safe = pb.clamp(eps, 1-eps), pt.clamp(eps, 1-eps)
        kl = (pb_safe * (pb_safe.log() - pt_safe.log()) +
              (1-pb_safe) * (torch.log1p(-pb_safe) - torch.log1p(-pt_safe))).clamp_min(0)
        d = tv if divergence == "tv" else kl
        outward = ((a > 0) & (diff > 0)) | ((a < 0) & (diff < 0))
        blocked = outward & (d > delta) & valid
        # Cap in log space before exp: exact min(rho,C), without overflow.
        weight = diff.clamp(max=math.log(cap)).exp()
        coefficient = a * weight * (~blocked) * valid
        ratio = diff.clamp(max=math.log(torch.finfo(dtype).max)-1).exp()
    return -coefficient * current, dict(tv=tv, kl=kl, ratio=ratio, blocked=blocked,
        outward=outward & valid, capped=(diff > math.log(cap)) & valid,
        positive=(a > 0) & valid, negative=(a < 0) & valid, valid=valid)


class DPPOStats:
    """Token-weighted DP statistics; bounded histograms approximate P95 within 1/4096 TV."""
    def __init__(self):
        self.sums = self.maxima = self.hist = None

    @torch.no_grad()
    def add(self, terms):
        t = terms; valid = t["valid"]
        vals = [valid.sum(), t["tv"][valid].double().sum(), t["kl"][valid].double().sum(),
                t["ratio"][valid].double().sum(), t["blocked"].sum(),
                (t["blocked"] & t["positive"]).sum(), t["positive"].sum(),
                (t["blocked"] & t["negative"]).sum(), t["negative"].sum(),
                t["outward"].sum(), t["capped"].sum()]
        sums = torch.stack([x.double() for x in vals])
        maxima = torch.stack([t[k][valid].max() if valid.any() else sums.new_zeros(())
                              for k in ("tv", "kl", "ratio")]).double()
        hist = torch.stack([torch.histc(t[k][valid].float(), bins=4096, min=0, max=limit).double()
                            for k,limit in (("tv",1), ("kl",40))])
        if self.sums is None:
            self.sums, self.maxima, self.hist = sums, maxima, hist
        else:
            self.sums += sums; self.maxima = torch.maximum(self.maxima, maxima); self.hist += hist

    @torch.no_grad()
    def metrics(self, delta, divergence, group=None):
        if self.sums is None:
            return {}
        if group is not None:
            torch.distributed.all_reduce(self.sums, group=group)
            torch.distributed.all_reduce(self.hist, group=group)
            torch.distributed.all_reduce(self.maxima, op=torch.distributed.ReduceOp.MAX, group=group)
        s = self.sums.cpu().tolist(); m = self.maxima.cpu().tolist(); n = max(s[0],1)
        result = dict(mask_fraction=s[4]/n, mask_fraction_positive=s[5]/max(s[6],1),
            mask_fraction_negative=s[7]/max(s[8],1), outward_fraction=s[9]/n,
            importance_cap_fraction=s[10]/n, ratio_mean=s[3]/n, ratio_max=m[2],
            delta=delta, finite_valid_token_count=s[0])
        for i,k,limit in ((0,"tv",1), (1,"kl",40)):
            if k == "kl" and divergence != "kl":
                continue
            cdf = self.hist[i].cumsum(0)
            index = int(torch.searchsorted(cdf, cdf[-1]*0.95).clamp(max=4095))
            result.update({f"binary_{k}_mean":s[i+1]/n, f"binary_{k}_max":m[i],
                           f"binary_{k}_p95":min((index+1)*limit/4096,m[i]) if s[0] else 0})
        return {"dppo/"+k:v for k,v in result.items()}


def validate_behavior_sampling(params, return_logprob):
    if not return_logprob:
        raise ValueError("DPPO requires SGLang behavior logprobs")
    # AIME validation uses 0.7/0.95 and never supplies a policy-gradient anchor.
    if params.get("temperature", 1.0) == 1.0:
        if params.get("top_p", 1.0) != 1.0 or params.get("top_k", -1) not in (-1, 0):
            raise ValueError("DPPO training behavior requires untruncated sampling")


def aime32_metrics(uids, accuracy):
    from collections import defaultdict
    groups = defaultdict(list)
    for uid, score in zip(uids, accuracy):
        groups[uid].append(float(score))
    if len(uids) != len(accuracy) or len(groups) != 30 or any(len(x) != 32 for x in groups.values()):
        raise ValueError("DPPO AIME24 requires exactly 30 questions x 32 samples")
    if any(score not in (0.,1.) for score in accuracy):
        raise ValueError("AIME accuracy must be binary")
    return {"avg_at_32":sum(accuracy)/960,
            "empirical_pass_at_32":sum(any(x) for x in groups.values())/30}
