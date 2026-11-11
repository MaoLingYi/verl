"""Frozen-support reference versus production coefficients, log numerator, and MCore schedule."""
import ast
from types import SimpleNamespace as NS
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))

import pytest
import torch
from test_eu_derpo_v1_8_execution_on_cpu import (
    ROOT, actor_loss, compile_function, eu, mcore_schedule, v18, v182, v183,
)


def fixture(equal=False):
    lens = torch.tensor([4, 2, 3, 1, 3, 2, 2, 4] if not equal else [4]*8)
    mask = torch.arange(4) < lens[:, None]
    behavior = torch.full((8, 4), -6., dtype=torch.float64)
    ratios = torch.tensor([.25, 5-1e-4, 5+1e-4, 20.], dtype=torch.float64)
    z = (behavior + ratios.log()).masked_fill(~mask, torch.nan).requires_grad_()
    advantages = torch.tensor([1., -1., 0., 1., -2., 0., .5, 1.])[:, None].expand_as(z)
    routes = torch.tensor([[[0, 1]], [[1, 2]], [[0, 2]], [[1, 2]]]).expand(8,-1,-1,-1)
    return z, behavior, advantages, mask, routes.masked_fill(~mask[:,:,None,None], -1)


def reference(z, b, a, mask, routes, keep):
    """Enumerate memberships separately from the vectorized scatter/gather implementation."""
    objective = z.masked_fill(~mask, 0).sum()*0
    grad = torch.zeros_like(z)
    for i in range(len(z)):
        memberships = [(layer, expert, (routes[i,:,layer] == expert).any(-1) & mask[i])
                       for layer in range(routes.shape[2]) for expert in range(3)]
        total = sum(int(s.sum()) for _, _, s in memberships)
        for layer, expert, s in memberships:
            if not keep[i,layer,expert] or not s.any():
                continue
            ratio = (z[i,s]-b[i,s]).exp()
            h = torch.where(ratio <= 5, ratio, 5+5*(ratio/5).log())
            scale = a[i,0].item()*int(mask[i].sum()) / (total*int(mask.sum()))
            objective = objective + scale*h.sum()
            grad[i,s] += scale*ratio.detach().clamp(max=5)
    return objective, grad


@pytest.mark.parametrize("decision", ["keep", "partial", "blocked"])
def test_global_scalar_and_gradients(decision):
    z,b,a,m,r = fixture()
    stats = v18.cluster_statistics(z,b,a,m,r,3,.012)
    if decision == "keep":
        stats.mask = stats.active
    elif decision == "blocked":
        stats.mask.zero_()
    else:
        assert stats.mask.any() and (stats.active & ~stats.mask).any()
    target, analytic = reference(z,b,a,m,r,stats.mask)
    gamma = v183.schedule_coefficients(stats,r,m,int(m.sum()),1,1)
    assert not gamma.requires_grad
    loss = -(gamma*z.masked_fill(~m,0)).sum()
    torch.testing.assert_close(v183.objective_numerator(stats,r,m)/m.sum(),target)
    torch.testing.assert_close(torch.autograd.grad(loss,z)[0],-analytic)
    torch.testing.assert_close(torch.autograd.grad(target,z)[0],analytic)
    fd = torch.zeros_like(z)
    for i,t in m.nonzero().tolist():
        plus,minus = z.detach().clone(),z.detach().clone()
        plus[i,t] += 1e-6
        minus[i,t] -= 1e-6
        fd[i,t] = (reference(plus,b,a,m,r,stats.mask)[0]-reference(minus,b,a,m,r,stats.mask)[0])/2e-6
    error = (fd-analytic).abs()
    print(f"{decision}: FD_ABS={error.max():.6g} FD_REL={(error/analytic.abs().clamp_min(1e-10)).max():.6g}")
    torch.testing.assert_close(analytic,fd,rtol=1e-7,atol=1e-9)


@pytest.mark.parametrize("sizes", [[8],[4,4],[1,7],[2,2,2,2],[1,2,1,4]])
@pytest.mark.parametrize("equal", [True,False])
def test_actual_schedule_and_loss(monkeypatch,sizes,equal):
    z,b,a,m,r = fixture(equal)
    stats = v18.cluster_statistics(z,b,a,m,r,3,.012)
    _, grad = reference(z,b,a,m,r,stats.mask)
    schedule = mcore_schedule(monkeypatch)
    config = NS(calculate_per_token_loss=False,timers=None)
    start = 0
    for local in sizes:
        for i in range(start,start+local):
            row = slice(i,i+1)
            def closure(logp):
                return actor_loss(logp,b[row],a[row],m[row],r[row],.012,"1.8.3",
                                  counts=(int(m.sum()),local,len(sizes)))
            loss,_ = schedule(None,z[row],closure,config,None,False,local,[],
                              cp_group_size=1,is_last_stage=True)
            (loss/len(sizes)).backward()
        start += local
    torch.testing.assert_close(z.grad,-grad,atol=5e-8,rtol=5e-7)


def test_actual_actor_statistics_and_controller():
    z,b,a,m,r = fixture()
    def stats(version):
        return actor_loss(z,b,a,m,r,.012,version,counts=(int(m.sum()),8,1),return_stats=True)
    old,new = stats("1.8.2"),stats("1.8.3")
    for key in ["count","active","log_arithmetic_ratio","divergence","directional_outward","mask","token_weight"]:
        assert torch.equal(getattr(old,key),getattr(new,key))
    states = []
    for result in (old,new):
        candidates = result.divergence[result.directional_outward]
        controller = eu.P95MCTRController(.012,.0027,.012,.018)
        states.append(controller.update(eu.exact_p95(candidates),len(candidates)))
    assert states[0] == states[1]


def test_length_mass_and_native_router_gradient():
    gate = torch.tensor([.2,-.3],dtype=torch.float64,requires_grad=True)
    z = (gate.softmax(-1)*gate.new_tensor([.2,.8])).sum().log().expand(2,4000)
    m = torch.ones_like(z,dtype=torch.bool)
    m[0,1000:] = False
    r = torch.zeros(2,4000,1,1,dtype=torch.long)
    stats = v18.cluster_statistics(z,z.detach(),torch.ones_like(z),m,r,1,.012)
    old,_ = v182.edppo_token_coefficients(stats,r,m)
    new = v183.schedule_coefficients(stats,r,m,5000,1,1)
    torch.testing.assert_close(old.sum(1)/2,z.new_tensor([.5,.5]))
    torch.testing.assert_close(new.sum(1),z.new_tensor([.2,.8]))
    assert torch.autograd.grad(-(new*z).sum(),gate)[0].abs().sum()>0


def test_variable_degree():
    count = torch.tensor([[[3.,1.,0.]]])
    routes = torch.tensor([[[[0,1]],[[0,2]],[[0,2]]]])
    # Inactive index 2 marks absent membership in this synthetic ragged fixture.
    mask = torch.ones(1,3,dtype=torch.bool)
    stats = NS(count=count,mask=count>0,advantage=torch.ones(1),token_weight=torch.ones(1,3))
    result = v183.schedule_coefficients(stats,routes,mask,10,1,1)
    torch.testing.assert_close(result,torch.tensor([[.15,.075,.075]]))


def test_driver_counts_before_dispatch():
    class Batch:
        def __len__(self):
            return self.batch["response_mask"].shape[0]
    batch = Batch()
    batch.batch = {"response_mask": fixture()[3]}
    batch.meta_info = {}
    trainer = NS(use_legacy_worker_impl="enable",
                 config=NS(actor_rollout_ref=NS(rollout=NS(multi_turn=NS(enable=False),temperature=1.),
                                                actor=NS(eu_derpo=NS(enabled=True,version="1.8.3")))),
                 actor_rollout_wg=NS(update_actor=lambda data: data.meta_info.copy()))
    path = ROOT/"verl/trainer/ppo/ray_trainer.py"
    node = next(n for n in ast.walk(ast.parse(path.read_text(encoding="utf8")))
                if isinstance(n,ast.FunctionDef) and n.name=="_update_actor")
    metadata = compile_function(node,{},path)(trainer,batch)
    assert metadata["v183_global_tokens"]==21 and metadata["v183_global_responses"]==8

def test_actor_objective_log_sums_shards_before_division(monkeypatch):
    z,b,a,m,r = fixture()
    all_stats = v18.cluster_statistics(z,b,a,m,r,3,.012)
    target,_ = reference(z,b,a,m,r,all_stats.mask)
    numerators = []
    for shard in [slice(0,1),slice(1,None)]:
        stats = v18.cluster_statistics(z[shard],b[shard],a[shard],m[shard],r[shard],3,.012)
        numerators.append(v183.objective_numerator(stats,r[shard],m[shard]))
    path = ROOT/"verl/workers/actor/megatron_actor.py"
    tree = ast.parse(path.read_text(encoding="utf8"))
    body = next(n.body for n in ast.walk(tree) if isinstance(getattr(n,"body",None),list)
                and any(ast.unparse(s).startswith("objective_device =") for s in n.body))
    start = next(i for i,s in enumerate(body) if ast.unparse(s).startswith("objective_device ="))
    end = next(i for i,s in enumerate(body) if ast.unparse(s).startswith("metrics['actor/eu_derpo/objective_edppo']"))
    remote = iter([numerators[1], 0.])
    def reduce(tensor,group):
        assert group == "dense-dp"
        tensor.add_(next(remote))
    monkeypatch.setattr(torch.distributed,"all_reduce",reduce)
    namespace = dict(global_token_mean=True, torch=torch, get_device_id=lambda:"cpu",
                     mpu=NS(get_data_parallel_group=lambda:"dense-dp",get_data_parallel_world_size=lambda:2),
                     objective=numerators[0], data=NS(meta_info={"v183_global_tokens":int(m.sum()),
                     "v183_global_responses":len(z)}), metrics={}, auxiliary_metrics={}, auxiliary_objective=torch.tensor(0.))
    exec(compile(ast.Module(body=body[start:end+1],type_ignores=[]),str(path),"exec"),namespace)
    assert namespace["metrics"]["actor/eu_derpo/objective_edppo"][0] == pytest.approx(target.item())
