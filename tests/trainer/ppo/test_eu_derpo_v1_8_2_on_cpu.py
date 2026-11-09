"""Production V1.8.2 against independent fixed-support cluster sums."""
import importlib.util
import sys
from pathlib import Path
import pytest
import torch

ROOT=Path(__file__).parents[3]
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module
    spec.loader.exec_module(module)
    return module
old=load("v182_reference_fixture",Path(__file__).with_name("test_eu_derpo_v1_8_on_cpu.py"))
sys.modules["verl.trainer.ppo.eu_derpo_v18"]=old.v18
new=load("v182_math",ROOT/"verl/trainer/ppo/eu_derpo_v182.py")


def reference(z,b,a,mask,routes,keep):
    scalar=z.masked_fill(~mask,0).sum()*0
    simplified=scalar
    analytic=torch.zeros_like(z)
    for i,row in old.clusters(routes,mask):
        total=sum(int(tokens.sum()) for _,_,tokens in row)
        for layer,expert,tokens in row:
            if not keep[i,layer,expert]: continue
            r=(z[i,tokens]-b[i,tokens]).exp()
            h=torch.where(r<=5,r,5+5*(r/5).log())
            n=int(tokens.sum());scale=a[i,0]/(total*z.shape[0])
            scalar=scalar+a[i,0]*(n/total)*h.mean()/z.shape[0]
            simplified=simplified+scale*h.sum()
            analytic[i,tokens]+=scale*r.detach().clamp(max=5)
    return scalar,simplified,analytic


@pytest.mark.parametrize("a",[1.,-1.,0.])
@pytest.mark.parametrize("delta",[0.,.012,.03,1.])
def test_four_way(a,delta):
    z,b,adv,mask,routes=old.fixture(a)
    stats=new.cluster_statistics(z,b,adv,mask,routes,128,delta)
    gamma,obj=new.edppo_token_coefficients(stats,routes,mask)
    scalar,simple,expected=reference(z,b,adv,mask,routes,stats.mask)
    torch.testing.assert_close(obj,scalar.detach(),rtol=1e-13,atol=1e-13)
    torch.testing.assert_close(simple,scalar,rtol=1e-13,atol=1e-13)
    auto=torch.autograd.grad(scalar,z)[0]
    actual=torch.autograd.grad((gamma*z).sum()/len(z),z)[0]
    assert not gamma.requires_grad
    fd=torch.zeros_like(z);eps=1e-5
    for t in range(z.shape[1]):
        plus=z.detach().clone();minus=z.detach().clone()
        plus[0,t]+=eps;minus[0,t]-=eps
        fd[0,t]=(reference(plus,b,adv,mask,routes,stats.mask)[0]-reference(minus,b,adv,mask,routes,stats.mask)[0])/(2*eps)
    for value in [auto,actual,fd]:torch.testing.assert_close(value,expected,rtol=2e-8,atol=2e-10)
    print(f"V182 FD_ABS={(fd-expected).abs().max():.3e} FD_REL={((fd-expected).abs()/expected.abs().clamp_min(1e-10)).max():.3e}")


def test_shared_statistics_and_unweighted_mctr():
    z,b,a,m,r=old.fixture()
    left=old.v18.cluster_statistics(z,b,a,m,r,128,.012)
    right=new.cluster_statistics(z,b,a,m,r,128,.012)
    assert new.cluster_statistics is old.v18.cluster_statistics
    assert new.capped_surrogate is old.v18.capped_surrogate
    for key in vars(left):
        if isinstance(vars(left)[key],torch.Tensor):torch.testing.assert_close(vars(left)[key],vars(right)[key],rtol=0,atol=0)
    values=left.divergence[left.directional_outward]
    torch.testing.assert_close(values,right.divergence[right.directional_outward],rtol=0,atol=0)
    # Exact same observations enter the unchanged production controller.
    states=[]
    for stats in [left,right]:
        controller=old.v17.P95MCTRController(.012,.0027,.012,.018)
        controller.update(old.v17.exact_p95(stats.divergence[stats.directional_outward]),int(stats.directional_outward.sum()))
        states.append(controller.state_dict())
        assert old.v17.exact_p95(stats.divergence[stats.directional_outward]) == old.v17.exact_p95(values)

    assert states[0]==states[1]


def test_ten_ninety_premask_denominator_and_response_mean():
    b=torch.full((2,100),.001,dtype=torch.float64).log();z=b.clone().requires_grad_()
    m=torch.ones_like(b,dtype=torch.bool);m[1,10:]=False
    r=torch.zeros(2,100,1,1,dtype=torch.long);r[0,10:]=127
    a=torch.ones_like(z)
    s=new.cluster_statistics(z,b,a,m,r,128,1.)
    g,obj=new.edppo_token_coefficients(s,r,m)
    torch.testing.assert_close(g[0],torch.full_like(g[0],.01))
    torch.testing.assert_close(g[1,:10],torch.full_like(g[1,:10],.1))
    assert s.count.sum().item()==110
    s.mask[0,0,127]=False
    g,obj=new.edppo_token_coefficients(s,r,m)
    assert g[0,10:].eq(0).all()
    torch.testing.assert_close(g[0,:10],torch.full_like(g[0,:10],.01))
    torch.testing.assert_close(g.sum(-1),torch.tensor([.1,1.],dtype=g.dtype))
    torch.testing.assert_close(obj,torch.tensor(.55,dtype=obj.dtype))
    s.mask.zero_();g,obj=new.edppo_token_coefficients(s,r,m)
    assert g.eq(0).all() and obj==0


def test_degree_identity_partial_overlap_and_padding_extreme():
    z,b,a,m,r=old.fixture()
    z=z.detach();z[0,0]=-1;b[0,0]=-10000
    z[~m]=torch.nan;b[~m]=torch.nan;r[~m]=-1
    s=new.cluster_statistics(z,b,a,m,r,128,1.)
    assert s.count.sum()==m.sum()*r.shape[2]*r.shape[3]
    s.mask.fill_(True)
    g,_=new.edppo_token_coefficients(s,r,m)
    torch.testing.assert_close(g[m],s.token_weight[m]/m.sum())
    assert g[0,0]>0 and g[~m].eq(0).all()
    s.mask[0,0,0]=False
    partial,_=new.edppo_token_coefficients(s,r,m)
    assert 0<partial[0,0]<g[0,0]
    h=partial/s.token_weight
    torch.testing.assert_close(h.sum(),(s.mask*s.count).sum()/s.count.sum())
    assert torch.isfinite(s.log_arithmetic_ratio).all() and torch.isfinite(partial).all()


@pytest.mark.parametrize("ratio",[.5,2.,5*(1-1e-6),5.,5*(1+1e-6),20.])
def test_cap_actual_new_function(ratio):
    b=torch.tensor([[-8.]],dtype=torch.float64)
    z=(b+torch.tensor(ratio,dtype=torch.float64).log()).requires_grad_()
    m=torch.ones_like(z,dtype=torch.bool);r=torch.tensor([[[[127]]]])
    s=new.cluster_statistics(z,b,torch.ones_like(z),m,r,128,1.)
    g,_=new.edppo_token_coefficients(s,r,m)
    actual=torch.autograd.grad(-(g*z).sum(),z)[0]
    torch.testing.assert_close(actual,torch.full_like(actual,-min(ratio,5)))


def test_native_router_gradient_from_new_main_objective():
    logits=torch.tensor([.2,-.3],dtype=torch.float64,requires_grad=True)
    z=(logits.softmax(-1)*torch.tensor([.2,.8],dtype=torch.float64)).sum().log().reshape(1,1)
    b=torch.full_like(z,-2);m=torch.ones_like(z,dtype=torch.bool);r=torch.tensor([[[[0,1]]]])
    s=new.cluster_statistics(z,b,torch.ones_like(z),m,r,2,1.)
    g,_=new.edppo_token_coefficients(s,r,m)
    grad=torch.autograd.grad(-(g*z).sum(),logits)[0]
    assert torch.isfinite(grad).all() and grad.abs().sum()>0


def test_variable_degree_general_count_contract():
    # Synthetic ragged support encoded with an inactive sentinel cluster; the
    # production Top-K collector remains fixed-degree and is not extended here.
    from types import SimpleNamespace
    routes=torch.tensor([[[[0,1]],[[0,2]],[[0,2]]]])
    count=torch.tensor([[[3.,1.,0.]]],dtype=torch.float64)
    stats=SimpleNamespace(count=count,mask=count>0,advantage=torch.ones(1,dtype=torch.float64),
                          token_weight=torch.ones(1,3,dtype=torch.float64),
                          token_surrogate=torch.ones(1,3,dtype=torch.float64))
    gamma,_=new.edppo_token_coefficients(stats,routes,torch.ones(1,3,dtype=torch.bool))
    torch.testing.assert_close(gamma,torch.tensor([[.5,.25,.25]],dtype=torch.float64))
    assert count.sum()==4 and not torch.allclose(gamma,torch.full_like(gamma,1/3))
