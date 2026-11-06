"""Independent fixed-mask scalar, analytic, finite-difference and actual gradients."""
import math
import torch
import pytest
import importlib.util
import sys
from pathlib import Path
ROOT = Path(__file__).parents[3]
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module
v17 = load("verl.trainer.ppo.eu_derpo", "verl/trainer/ppo/eu_derpo.py")
v18 = load("v18_math", "verl/trainer/ppo/eu_derpo_v18.py")


def fixture(a=1.):
    b = torch.tensor([[.001, .01, .04, .02, .03]], dtype=torch.float64).log()
    z = b + torch.tensor([[20., 4., .25, 2., .5]], dtype=torch.float64).log()
    # Two overlapping layers, unequal cluster sizes, expert boundary 127.
    routes = torch.tensor([[[[0, 127], [1, 2]], [[0, 1], [1, 2]],
                            [[0, 1], [0, 2]], [[1, 127], [0, 1]],
                            [[0, 127], [0, 2]]]])
    mask = torch.tensor([[1, 1, 1, 1, 0]], dtype=torch.bool)
    return z.requires_grad_(), b, torch.full_like(z, a), mask, routes


def clusters(routes, mask):
    for i in range(routes.shape[0]):
        row = []
        for layer in range(routes.shape[2]):
            for expert in routes[i, mask[i], layer].unique():
                tokens = mask[i] & (routes[i, :, layer] == expert).any(-1)
                row.append((layer, int(expert), tokens))
        yield i, row


def reference(z, b, a, mask, routes, frozen_masks):
    result = (z * 0).sum()
    analytic = torch.zeros_like(z)
    for i, row in clusters(routes, mask):
        for layer, expert, tokens in row:
            if not frozen_masks[i, layer, expert]:
                continue
            ratio = (z[i, tokens] - b[i, tokens]).exp()
            # Independent piecewise H, safe on this finite test domain.
            h = torch.where(ratio <= 5, ratio, 5 + 5 * torch.log(ratio / 5))
            scale = a[i, 0] / (len(row) * int(tokens.sum()) * z.shape[0])
            result = result + scale * h.sum()
            analytic[i, tokens] += scale * ratio.detach().clamp(max=5)
    return result, analytic


@pytest.mark.parametrize('a', [1., -1., 0.])
@pytest.mark.parametrize('delta', [0., .012, .03, 1.])
def test_four_way_gradients(a, delta):
    z, b, adv, mask, routes = fixture(a)
    stats = v18.cluster_statistics(z, b, adv, mask, routes, 128, delta)
    gamma, objective = v18.edppo_token_coefficients(stats, routes, mask)
    assert not gamma.requires_grad and not stats.mask.requires_grad
    scalar, expected = reference(z, b, adv, mask, routes, stats.mask)
    torch.testing.assert_close(objective, scalar.detach(), rtol=1e-13, atol=1e-13)
    auto = torch.autograd.grad(scalar, z)[0]
    detached = torch.autograd.grad((gamma * z).sum() / z.shape[0], z)[0]
    fd = torch.zeros_like(z)
    eps = 1e-5
    for t in range(z.shape[1]):
        plus, minus = z.detach().clone(), z.detach().clone()
        plus[0, t] += eps; minus[0, t] -= eps
        fd[0, t] = (reference(plus,b,adv,mask,routes,stats.mask)[0] -
                    reference(minus,b,adv,mask,routes,stats.mask)[0]) / (2*eps)
    for value in (auto, detached, fd):
        torch.testing.assert_close(value, expected, atol=2e-10, rtol=2e-8)
    absolute = (fd-expected).abs().max().item()
    relative = ((fd-expected).abs()/expected.abs().clamp_min(1e-10)).max().item()
    print(f'V18 a={a} delta={delta}: FD_MAX_ABS={absolute:.3e} FD_MAX_REL={relative:.3e} '
          f'AUTOGRAD_MAX_ABS={(auto-expected).abs().max():.3e}')


@pytest.mark.parametrize('ratio', [2., 5*(1-1e-6), 5., 5*(1+1e-6), 20.])
def test_cap_derivative_nonzero(ratio):
    d = torch.tensor(math.log(ratio), dtype=torch.float64, requires_grad=True)
    gradient = torch.autograd.grad(v18.capped_surrogate(d), d)[0]
    assert gradient.item() == pytest.approx(min(ratio,5), abs=1e-12)


def test_raw_arithmetic_outward_not_geometric_or_capped():
    b = torch.full((1, 10), -10., dtype=torch.float64)
    ratio = torch.tensor([[20.] + [.01]*9], dtype=torch.float64)
    z = b + ratio.log(); mask = torch.ones_like(z,dtype=torch.bool)
    routes = torch.zeros(1,10,1,1,dtype=torch.long)
    stats = v18.cluster_statistics(z,b,torch.ones_like(z),mask,routes,1,1.)
    old = v17.cluster_statistics(z,b,torch.ones_like(z),mask,routes,1,1.)
    assert stats.directional_outward.item() and not old.directional_outward.item()
    assert ratio.clamp(max=5).mean() < 1  # Capping direction would reverse it.
    gamma,_ = v18.edppo_token_coefficients(stats,routes,mask)
    torch.testing.assert_close(gamma,ratio.clamp(max=5)/10)


def test_tv_equality_and_partial_overlap_retained():
    b = torch.tensor([[.01,.01]],dtype=torch.float64).log()
    z = torch.tensor([[.04,.01]],dtype=torch.float64).log()
    mask=torch.ones_like(z,dtype=torch.bool); a=torch.ones_like(z)
    routes=torch.tensor([[[[0,1]],[[1,2]]]])
    stats=v18.cluster_statistics(z,b,a,mask,routes,3,1.)
    threshold=stats.divergence[0,0,1].item()
    stats=v18.cluster_statistics(z,b,a,mask,routes,3,threshold)
    assert not stats.mask[0,0,0] and stats.mask[0,0,1]
    gamma,_=v18.edppo_token_coefficients(stats,routes,mask)
    assert gamma[0,0] == pytest.approx(4/6)  # blocked singleton, kept overlap
    all_blocked=v18.cluster_statistics(z[:,:1],b[:,:1],a[:,:1],mask[:,:1],routes[:,:1],3,0.)
    assert not v18.edppo_token_coefficients(all_blocked,routes[:,:1],mask[:,:1])[0].any()


def test_padding_extreme_ratio_fp32_and_unit_mass():
    z,b,a,mask,routes=fixture()
    z=z.detach().float(); b=b.float(); a=a.float()
    z[0,-1]=float('nan'); b[0,-1]=float('inf'); a[0,-1]=float('nan'); routes[0,-1]=-1
    b[0,0]=-10000.
    stats=v18.cluster_statistics(z,b,a,mask,routes,128,1.)
    gamma,_=v18.edppo_token_coefficients(stats,routes,mask)
    assert torch.isfinite(gamma).all() and gamma[0,-1]==0 and gamma.dtype==torch.float32
    b=z.clone()
    stats=v18.cluster_statistics(z,b,a,mask,routes,128,1.)
    gamma,_=v18.edppo_token_coefficients(stats,routes,mask)
    torch.testing.assert_close(gamma.sum(),torch.tensor(1.))


def test_native_selected_router_gradient_lambda_zero():
    router=torch.tensor([.2,-.4,.7],dtype=torch.float64,requires_grad=True)
    support=torch.tensor([0,2])
    alpha=router[support].softmax(0)
    expert_prob=torch.tensor([[.02,.04],[.05,.01]],dtype=torch.float64)
    z=(expert_prob@alpha).log()[None,:]
    b=torch.full_like(z,math.log(.01)); mask=torch.ones_like(z,dtype=torch.bool)
    routes=support.expand(1,2,1,2)
    stats=v18.cluster_statistics(z,b,torch.ones_like(z),mask,routes,3,1.)
    gamma,_=v18.edppo_token_coefficients(stats,routes,mask)
    loss=-(gamma*z).sum()
    sensitivity=torch.autograd.grad(loss,alpha,retain_graph=True)[0]
    actual=torch.autograd.grad(loss,router)[0]
    expected=alpha.detach()*(sensitivity-(alpha.detach()*sensitivity).sum())
    torch.testing.assert_close(actual[support],expected)
    assert actual[support].abs().min()>0 and actual[1]==0


def test_mctr_uses_arithmetic_population_lag_and_restore():
    z,b,a,mask,routes=fixture()
    controller=v17.P95MCTRController(.012,.0027,.012,.018)
    stats=v18.cluster_statistics(z,b,a,mask,routes,128,controller.delta_current)
    candidates=stats.divergence[stats.directional_outward]
    before=controller.delta_current
    assert before==.012
    q=v17.exact_p95(candidates)
    controller.update(q,candidates.numel())
    assert controller.delta_current == pytest.approx((1-.0027)*before+.0027*min(max(q,.012),.018))
    restored=v17.P95MCTRController(.012,.0027,.012,.018)
    restored.load_state_dict(controller.state_dict())
    assert restored.state_dict()==controller.state_dict()
    restored.update(None,0)
    assert restored.delta_current==controller.delta_current
