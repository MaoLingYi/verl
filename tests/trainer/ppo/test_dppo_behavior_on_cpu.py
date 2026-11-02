"""CPU math tests execute production definitions without importing Ray/Megatron.

AST loading isolates optional runtime imports; it is not a distributed/GPU test.
"""
import ast
import importlib.util
import math
import types
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional
import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location("dppo_math", ROOT / "verl/trainer/ppo/dppo.py")
dppo = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(dppo)


def definitions():
    names = {"agg_loss", "_compute_policy_loss_dppo", "compute_policy_loss_dppo_tv",
             "compute_policy_loss_dppo_kl", "compute_grpo_outcome_advantage", "compute_policy_loss_vanilla"}
    tree = ast.parse((ROOT / "verl/trainer/ppo/core_algos.py").read_text(encoding="utf-8"))
    nodes = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            node.decorator_list = []
            for item in ast.walk(node):
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    item.returns = None
                if isinstance(item, ast.arg): item.annotation = None
            nodes.append(node)
    F = types.SimpleNamespace(masked_sum=lambda x,m:(x*m).sum(),
        masked_mean=lambda x,m:(x*m).sum()/m.sum())
    ns = dict(torch=torch, np=np, Optional=Optional, verl_F=F, defaultdict=defaultdict)
    exec(compile(ast.Module(nodes, type_ignores=[]), "production_core_algos", "exec"), ns)
    return ns

CORE = definitions()
CONFIG = types.SimpleNamespace(policy_loss=types.SimpleNamespace(dppo_delta_tv=.2,
    dppo_delta_kl=.05,dppo_importance_cap=5.),global_batch_info={})


def call(current, behavior, a, mask, kind="tv", old=None, config=CONFIG):
    # Inject only the actual pure production module, leaving optional framework imports unloaded.
    from unittest.mock import patch
    with patch.dict(sys.modules, {"verl.trainer.ppo.dppo":dppo}):
        return CORE["compute_policy_loss_dppo_"+kind](old, current, a, mask,
            config=config, rollout_log_probs=behavior)


@pytest.mark.parametrize("kind",["tv","kl"])
def test_hand_calculation_and_exact_gradient(kind):
    pb=torch.tensor([[.1,.8,.8,.1,.2,.2,.001,.5]],dtype=torch.float64)
    pt=torch.tensor([[.4,.4,.9,.05,.2,.2,.01,.5]],dtype=torch.float64)
    a=torch.tensor([[1.,-1.,-1.,1.,0.,1.,1.,-1.]])
    current=pt.log().requires_grad_(); mask=torch.ones_like(a);mask[0,-1]=0
    loss,terms=dppo.token_terms(current,pb.log(),a,mask,kind,.2 if kind=="tv" else .05,5.)
    tv=(pb-pt).abs();kl=pb*(pb/pt).log()+(1-pb)*((1-pb)/(1-pt)).log()
    torch.testing.assert_close(terms["tv"],tv)
    torch.testing.assert_close(terms["kl"][mask.bool()],kl[mask.bool()])
    divergence=tv if kind=="tv" else kl
    blocked=(((a>0)&(pt>pb))|((a<0)&(pt<pb))) & (divergence>(.2 if kind=="tv" else .05))
    expected=-a*torch.minimum(pt/pb,torch.tensor(5.))*~blocked*mask
    loss.sum().backward();torch.testing.assert_close(current.grad,expected)
    assert current.grad[0,0]==0 and current.grad[0,1]==0
    assert current.grad[0,2]>0 and current.grad[0,3]<0
    assert current.grad[0,4]==0 and current.grad[0,6].item()==pytest.approx(-5) and current.grad[0,7]==0
    assert not terms["blocked"].requires_grad


@pytest.mark.parametrize("kind",["tv","kl"])
def test_delta_equality_and_rho_equality(kind):
    c=torch.tensor([[.5,.25]],dtype=torch.float64).log().requires_grad_()
    b=torch.tensor([[.25,.25]],dtype=torch.float64).log();a=torch.ones_like(b);m=torch.ones_like(b)
    _,t=dppo.token_terms(c,b,a,m,kind,1.,5)
    threshold=t[kind][0,0].item()
    _,exact=dppo.token_terms(c,b,a,m,kind,threshold,5)
    assert not exact["blocked"].any()
    _,below=dppo.token_terms(c,b,a,m,kind,threshold/2,5)
    assert below["blocked"].tolist()==[[True,False]]


@pytest.mark.parametrize("kind",["tv","kl"])
def test_behavior_anchor_old_recompute_invariance(kind):
    b=torch.tensor([[.1,.8]],dtype=torch.float64).log()
    x=torch.tensor([[.4,.4]],dtype=torch.float64).log().requires_grad_()
    a=torch.tensor([[1.,-1.]]);m=torch.ones_like(a)
    l1,t1=call(x,b,a,m,kind,old=b+1)
    l2,t2=call(x,b,a,m,kind,old=b-4)
    torch.testing.assert_close(l1,l2);assert t1==t2
    g1=torch.autograd.grad(l1,x,retain_graph=True)[0]
    g2=torch.autograd.grad(l2,x)[0];torch.testing.assert_close(g1,g2)
    assert t1["dppo/mask_fraction"]==1
    with pytest.raises(ValueError, match="real rollout"):
        call(x,None,a,m,kind,old=b)


@pytest.mark.parametrize("kind",["tv","kl"])
@pytest.mark.parametrize("zero_adv",[True,False])
def test_extreme_endpoints_padding_finite_and_zero_adv(kind,zero_adv):
    c=torch.tensor([[0.,-1e-300,-10000.,float("nan")]],dtype=torch.float64,requires_grad=True)
    b=torch.tensor([[-10000.,0.,-1e-300,float("nan")]],dtype=torch.float64)
    a=torch.full_like(b,0 if zero_adv else 1);m=torch.tensor([[1.,1.,1.,0.]])
    loss,stats=call(c,b,a,m,kind);loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(c.grad).all()
    assert all(math.isfinite(x) for x in stats.values())
    assert stats["dppo/finite_valid_token_count"]==3
    if zero_adv: assert torch.count_nonzero(c.grad)==0
    empty,_=call(c,b,a,torch.zeros_like(m),kind);assert empty==0


def test_grpo_group_eight_no_std_normalization():
    rewards=torch.arange(16,dtype=torch.float64).reshape(16,1)
    mask=torch.ones_like(rewards);index=np.repeat([0,1],8)
    adv,_=CORE["compute_grpo_outcome_advantage"](rewards,mask,index,norm_adv_by_std_in_grpo=False)
    expected=torch.cat([torch.arange(8)-3.5]*2).reshape(16,1).double()
    torch.testing.assert_close(adv,expected)


def test_full_response_microbatch_and_global_token_aggregation():
    c=torch.tensor([[.4,.5],[.3,.6]],dtype=torch.float64).log().requires_grad_()
    b=torch.full_like(c,.3).log();a=torch.ones_like(c);m=torch.tensor([[1.,0.],[1.,1.]])
    losses,_=dppo.token_terms(c,b,a,m,"tv",1.,5.)
    agg=CORE["agg_loss"]
    full=agg(losses,m,"token-mean")
    micro=sum(agg(losses[i:i+1],m[i:i+1],"token-mean",batch_num_tokens=3) for i in range(2))
    torch.testing.assert_close(full,micro)
    # Historical Megatron contract: mean of per-microbatch token means (unequal lengths differ).
    legacy=sum(agg(losses[i:i+1],m[i:i+1],"token-mean") for i in range(2))/2
    assert not torch.isclose(legacy,full)


def test_statistics_token_weighted_across_microbatches():
    x=torch.tensor([[.1,.3],[.5,.9]]).log();b=torch.full_like(x,.2).log()
    a=torch.tensor([[1.,-1.],[-1.,1.]]);m=torch.tensor([[1.,0.],[1.,1.]])
    _,all_terms=dppo.token_terms(x,b,a,m,"tv",.2,5)
    one=dppo.DPPOStats();one.add(all_terms)
    many=dppo.DPPOStats()
    for i in range(2):
        _,t=dppo.token_terms(x[i:i+1],b[i:i+1],a[i:i+1],m[i:i+1],"tv",.2,5);many.add(t)
    assert one.metrics(.2,"tv")==pytest.approx(many.metrics(.2,"tv"))


def test_invalid_values_fail_fast():
    x=torch.zeros(1,1);m=torch.ones_like(x)
    for delta,cap in [(-1,5),(.2,0),(.2,float("inf"))]:
        with pytest.raises(ValueError): dppo.token_terms(x,x,m,m,"tv",delta,cap)
    with pytest.raises(ValueError,match="non-finite"):
        dppo.token_terms(x+float("nan"),x,m,m,"tv",.2,5)


def test_aime32_is_not_empirical_pass32():
    uids=np.repeat(np.arange(30),32).tolist();acc=[0.]*960;acc[0]=1
    result=dppo.aime32_metrics(uids,acc)
    assert result["avg_at_32"]==1/960
    assert result["empirical_pass_at_32"]==1/30
    with pytest.raises(ValueError):dppo.aime32_metrics(uids[:-1],acc[:-1])


def test_sampling_and_additional_is_fail_fast():
    dppo.validate_behavior_sampling(dict(temperature=1.,top_p=1.,top_k=-1),True)
    with pytest.raises(ValueError):dppo.validate_behavior_sampling({},False)
    with pytest.raises(ValueError):dppo.validate_behavior_sampling(dict(temperature=1.,top_p=.9),True)
    from unittest.mock import patch
    x=torch.full((1,1),-.5)
    with patch.dict(sys.modules,{"verl.trainer.ppo.dppo":dppo}):
        with pytest.raises(ValueError,match="additional rollout IS"):
            CORE["compute_policy_loss_dppo_tv"](x,x,x,x,config=CONFIG,rollout_log_probs=x,rollout_is_weights=x)


@pytest.mark.parametrize("enabled",[False,True])
def test_graph_sleep_wake_independent_of_replay(enabled):
    import asyncio
    from unittest.mock import AsyncMock
    source=ROOT/"verl/workers/rollout/sglang_rollout/sglang_rollout.py"
    tree=ast.parse(source.read_text(encoding="utf-8"))
    cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=="ServerAdapter")
    methods=[x for x in cls.body if isinstance(x,ast.AsyncFunctionDef) and x.name in {"resume","release"}]
    for node in methods:
        node.decorator_list=[];node.returns=None
        for arg in node.args.args:arg.annotation=None
    ns={};exec(compile(ast.Module(methods,type_ignores=[]),str(source),"exec"),ns)
    engine=types.SimpleNamespace(resume_memory_occupation=AsyncMock(),release_memory_occupation=AsyncMock())
    actor=types.SimpleNamespace(_init_server_adapter=AsyncMock(),_engine=engine,
        config=types.SimpleNamespace(free_cache_engine=True,cuda_graph_memory_saver=enabled,
                                     enable_rollout_routing_replay=False),
        device_mesh={"infer_tp":types.SimpleNamespace(get_local_rank=lambda:0)})
    asyncio.run(ns["release"](actor));asyncio.run(ns["resume"](actor,["weights"]))
    asyncio.run(ns["resume"](actor,["kv_cache"]))
    expected=["kv_cache","weights"]+(["cuda_graph"] if enabled else [])
    assert engine.release_memory_occupation.call_args.kwargs["tags"]==expected
    assert engine.resume_memory_occupation.call_args_list[0].kwargs["tags"]==["weights"]
    assert engine.resume_memory_occupation.call_args_list[1].kwargs["tags"]==["kv_cache"]+(["cuda_graph"] if enabled else [])


def test_legacy_algorithm_definitions_unchanged():
    import subprocess
    before=subprocess.check_output(["git","-C",str(ROOT),"show",
        "753a21ac22cb349951f76988cbe3c38c481cf9dc:verl/trainer/ppo/core_algos.py"],text=True,encoding="utf-8")
    after=(ROOT/"verl/trainer/ppo/core_algos.py").read_text(encoding="utf-8")
    names={"compute_grpo_outcome_advantage","compute_policy_loss_vanilla","compute_policy_loss_gspo",
           "compute_policy_loss_gmpo","compute_policy_loss_cispo","compute_policy_loss_bypass_mode"}
    def nodes(s):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(s).body
                         if isinstance(n,ast.FunctionDef) and n.name in names}
    assert nodes(before)==nodes(after)


def test_megatron_real_loss_call_uses_rollout_anchor():
    from unittest.mock import patch
    source=ROOT/"verl/workers/actor/megatron_actor.py"
    tree=ast.parse(source.read_text(encoding="utf-8"))
    init=next(n for n in ast.walk(tree) if isinstance(n,ast.Assign)
              and any(isinstance(t,ast.Name) and t.id=="policy_loss_kwargs" for t in n.targets))
    branch=next(n for n in ast.walk(tree) if isinstance(n,ast.If) and isinstance(n.test,ast.Compare)
                and isinstance(n.test.left,ast.Name) and n.test.left.id=="loss_mode")
    call_node=next(n for n in ast.walk(tree) if isinstance(n,ast.Assign) and isinstance(n.value,ast.Call)
                  and isinstance(n.value.func,ast.Name) and n.value.func.id=="policy_loss_fn")
    current=torch.tensor([[.4,.4]],dtype=torch.float64).log().requires_grad_()
    behavior=torch.tensor([[.1,.8]],dtype=torch.float64).log()
    ns=dict(loss_mode="dppo_tv",self=types.SimpleNamespace(config=CONFIG,_dppo_stats=None),
        data={"rollout_log_probs":behavior}, old_log_prob=behavior+3, log_prob=current,
        advantages=torch.tensor([[1.,-1.]]),response_mask=torch.ones(1,2),
        loss_agg_mode="token-mean",rollout_is_weights=None,policy_loss_fn=CORE["compute_policy_loss_dppo_tv"])
    program=compile(ast.Module([init,branch,call_node],type_ignores=[]),str(source),"exec")
    with patch.dict(sys.modules,{"verl.trainer.ppo.dppo":dppo}):
        exec(program,ns)
        assert ns["pg_metrics"]["dppo/mask_fraction"]==1
        ns["old_log_prob"]=behavior-8;exec(program,ns)
        assert ns["pg_metrics"]["dppo/mask_fraction"]==1
        ns["data"]={}
        with pytest.raises(ValueError,match="real rollout"):exec(program,ns)
