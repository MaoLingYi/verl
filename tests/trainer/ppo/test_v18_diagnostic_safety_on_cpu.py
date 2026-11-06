import ast
import math
from types import SimpleNamespace as NS
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from test_eu_derpo_v1_8_on_cpu import v17, v18, ROOT
import torch
import pytest


def diagnostic(version, rho, log_ratio):
    path=ROOT/"verl/workers/actor/megatron_actor.py"
    tree=ast.parse(path.read_text(encoding="utf-8"))
    loop=next(n for n in ast.walk(tree) if isinstance(n,ast.For) and isinstance(n.iter,ast.Tuple)
              and any(isinstance(x,ast.Constant) and x.value=="rho_e" for x in ast.walk(n.iter)))
    branch=next(n for n in ast.walk(tree) if isinstance(n,ast.If) and isinstance(n.test,ast.Compare)
                and ast.unparse(n.test)=="self.config.eu_derpo.version in {'1.7', '1.8'}"
                and any(isinstance(x,ast.Name) and x.id=="ratio_diagnostic" for x in ast.walk(n)))
    gather=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="_gather_diagnostic_values")
    ns={"get_device_id":lambda:"cpu","torch":torch,"mpu":NS(get_pipeline_model_parallel_world_size=lambda:1,get_data_parallel_world_size=lambda:1,get_pipeline_model_parallel_group=lambda:None,get_data_parallel_group=lambda:None),
        "self":NS(config=NS(eu_derpo=NS(version=version))),"active":torch.ones_like(rho,dtype=torch.bool),
        "stats_eu":NS(rho=rho,log_arithmetic_ratio=log_ratio,divergence=torch.zeros_like(rho)),
        "utility_count":torch.ones_like(rho),"utility_sum":torch.ones_like(rho),
        "cluster_utility":torch.ones_like(rho),"normalized":torch.ones_like(rho),
        "dppo":NS(log_quantiles=(.5,.95)),"distribution_stats":v17.distribution_stats,"metrics":{}}
    # Real gather executes without distributed initialization in this CPU case.
    exec(compile(ast.Module(body=[gather,loop,branch],type_ignores=[]),str(path),"exec"),ns)
    return ns["metrics"]


@pytest.mark.parametrize("value",[2.,1.25e38])
def test_actual_diagnostic_loops(value):
    rho=torch.full((1,1,8),value)
    old=v17.distribution_stats(rho,(.95,))
    if value>1e30: assert not math.isfinite(old["mean"])
    result=diagnostic("1.8",rho,rho.log())
    assert not any("rho_e" in name for name in result)
    assert all(math.isfinite(x[0]) for x in result.values())
    assert sum(name.endswith("log_arithmetic_ratio_mean") for name in result)==1
    legacy=diagnostic("1.7",torch.full_like(rho,2),rho.log())
    assert legacy["actor/eu_derpo/rho_e_mean"]==[2.]
    assert legacy["actor/eu_derpo/rho_e_beh_mean"]==[2.]


def test_diagnostics_do_not_mutate_main_math():
    z=torch.full((1,4),-2.,requires_grad=True); b=torch.full_like(z,-1000.)
    mask=torch.ones_like(z,dtype=torch.bool); routes=torch.zeros(1,4,1,1,dtype=torch.long)
    stats=v18.cluster_statistics(z,b,torch.ones_like(z),mask,routes,1,1.)
    before=[x.clone() for x in (stats.directional_outward,stats.mask,stats.token_weight)]
    gamma,_=v18.edppo_token_coefficients(stats,routes,mask)
    grad=torch.autograd.grad(-(gamma*z).sum(),z,retain_graph=True)[0]
    diagnostic("1.8",stats.rho,stats.log_arithmetic_ratio)
    after,_=v18.edppo_token_coefficients(stats,routes,mask)
    for x,y in zip(before,(stats.directional_outward,stats.mask,stats.token_weight)): assert torch.equal(x,y)
    assert torch.equal(after,gamma) and stats.token_weight.eq(5).all()
    assert torch.equal(torch.autograd.grad(-(after*z).sum(),z)[0],grad)
