"""CPU reference and two independent-process RSS probes for the V1.8.3 post-backward path."""
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))
from test_eu_derpo_v1_8_execution_on_cpu import v18, v183


def inputs(batch, tokens, layers, topk):
    lengths = torch.linspace(1, tokens, batch).long()
    mask = torch.arange(tokens) < lengths[:, None]
    deltas = torch.tensor([.25, 1., 2., 4.9999, 5., 5.0001, 20.]).log().repeat(tokens//7+1)[:tokens]
    behavior = torch.full((batch, tokens), -5.)
    current = (behavior + deltas).masked_fill(~mask, torch.nan)
    adv = (torch.arange(batch) % 3 - 1.)[:, None].expand_as(current)
    # Independent CPU allocations, as in the actual-F route cache.
    base = torch.arange(tokens)[:,None,None] % 7 + torch.arange(layers)[None,:,None] + torch.arange(topk)
    routes = {i: (base+i).remainder(128).byte() for i in range(batch)}
    return routes, current, behavior, adv, mask


def compare_or_measure(mode, data):
    routes, current, behavior, adv, mask = data
    if mode == "whole":
        stacked = torch.stack(list(routes.values()))
        stats = v18.cluster_statistics(current,behavior,adv,mask,stacked,128,.012)
        return stats,v183.objective_numerator(stats,stacked,mask)
    rows = ((routes.pop(i), current[i]) for i in range(len(current)))
    return v183.post_backward_statistics(rows,behavior,adv,mask,128,.012,False,"rollout")


def test_whole_vs_complete_response():
    data = inputs(7,513,5,4)
    routes,z,b,a,mask = data
    full,objective = compare_or_measure("whole",data)
    original = routes.copy()
    compact,part = compare_or_measure("rows",data)
    assert not routes
    for key in ("count","rho","divergence","mask","advantage","directional_outward","log_arithmetic_ratio"):
        torch.testing.assert_close(getattr(compact,key),getattr(full,key),rtol=0,atol=0)
    assert torch.equal(full.divergence[full.directional_outward],compact.divergence[compact.directional_outward])
    torch.testing.assert_close(part,objective,rtol=1e-13,atol=1e-13)
    # Compare the per-row capped signals and their gradient to a shared native gate.
    gate = torch.tensor([.2,-.1],requires_grad=True)
    logp = (gate.softmax(0)*torch.tensor([.3,.8])).sum().log() + z.masked_fill(~mask,0)
    gamma = v183.schedule_coefficients(full,torch.stack(list(original.values())),mask,int(mask.sum()),7,1)
    row_gamma = []
    for i in range(len(z)):
        r = original[i][None]
        row = slice(i,i+1)
        stats = v18.cluster_statistics(z[row],b[row],a[row],mask[row],r,128,.012)
        assert torch.equal(stats.token_weight,full.token_weight[row])
        row_gamma.append(v183.schedule_coefficients(stats,r,mask[row],int(mask.sum()),7,1))
    split = torch.cat(row_gamma)
    assert torch.equal(split,gamma)
    grad = lambda g: torch.autograd.grad(-(g*logp).sum()/7,gate,retain_graph=True)[0]
    assert torch.equal(grad(gamma),grad(split)) and grad(gamma).abs().sum()>0


@pytest.mark.parametrize("bad",[128,0])
def test_last_row_validation(bad):
    data = inputs(3,32,2,2)
    data[0][2][-1,-1] = torch.tensor([0,bad])
    with pytest.raises(ValueError,match="expert"):
        compare_or_measure("rows",data)


if __name__ == "__main__":
    import psutil
    mode = sys.argv[1]
    assert mode in ("whole","rows")
    torch.set_num_threads(1)
    data = inputs(32,4096,24,8)
    proc = psutil.Process()
    before = proc.memory_info().rss
    stats,q = compare_or_measure(mode,data)
    if sys.platform == "win32":
        peak = proc.memory_info().peak_wset
    else:
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
    print(f"mode={mode} uint8[32,4096,24,8] baseline_rss={before} peak_rss={peak} live_route_owners={len(data[0])} Q={float(q):.12g}")
