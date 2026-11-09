"""Execute production loss/trainer AST on CPU; no Ray, Megatron model or GPU launch."""
import ast
from collections import defaultdict
import importlib.util
import os
from pathlib import Path
import sys
from types import SimpleNamespace, ModuleType

import numpy as np
import pytest
import torch


ROOT = Path(__file__).parents[3]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


eu = load("verl.trainer.ppo.eu_derpo", ROOT / "verl/trainer/ppo/eu_derpo.py")
v18 = load("v18_execution_math", ROOT / "verl/trainer/ppo/eu_derpo_v18.py")
sys.modules["verl.trainer.ppo.eu_derpo_v18"] = v18
v182 = load("v182_execution_math", ROOT / "verl/trainer/ppo/eu_derpo_v182.py")
legacy = load("v18_fit_harness", Path(__file__).with_name("test_eu_derpo_v1_7_on_cpu.py"))


def compile_function(node, namespace, filename):
    node.decorator_list = []
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), node], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(filename), "exec"), namespace)
    return namespace[node.name]


def actor_loss(log_prob, behavior, advantage, mask, routes, delta, version="1.8"):
    path = ROOT / "verl/workers/actor/megatron_actor.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    branch = next(node for node in ast.walk(tree) if isinstance(node, ast.If)
                  and ast.unparse(node.test) == "self.config.eu_derpo.enabled"
                  and any(isinstance(n, ast.Name) and n.id == "eu_stats" for n in ast.walk(node)))
    cfg = SimpleNamespace(version=version, expert_cluster_dppo=SimpleNamespace(delta_e=delta, diagnostics_only=False),
                          behavior_expert_is=SimpleNamespace(behavior_source="rollout"))
    recorded = []
    observer = SimpleNamespace(num_experts=3, record_main_logprobs=lambda *args: recorded.append(args))
    def wrong_version(*args, **kwargs):
        pytest.fail("V1.8 entered geometric V1.7 math")
    namespace = dict(torch=torch, eu_derpo_v18=v18, eu_derpo_v182=v182, cluster_statistics=wrong_version,
                     edppo_token_coefficients=wrong_version,
                     self=SimpleNamespace(config=SimpleNamespace(eu_derpo=cfg), eu_derpo_observer=observer,
                                          mctr_controller=object(), _eu_derpo_delta_current=delta),
                     log_prob=log_prob, data={"rollout_log_probs": behavior}, advantages=advantage,
                     response_mask=mask, eu_derpo_f_routes=routes, eu_derpo_sample_ids=torch.tensor([0]),
                     device=torch.device("cpu"))
    exec(compile(ast.Module(body=branch.body, type_ignores=[]), str(path), "exec"), namespace)
    assert recorded and not namespace["coefficient"].requires_grad
    return namespace["pg_loss"], namespace["pg_metrics"]


def mcore_schedule(monkeypatch):
    default = ROOT.parent / "pp1_tp4_ep8_core_sources/megatron_core"
    source = Path(os.environ.get("MCORE_SOURCE_ROOT", default)) / "pipeline_parallel/schedules.py"
    if not source.exists():
        pytest.skip("set MCORE_SOURCE_ROOT to the installed/fixed MCore source tree")
    tree = ast.parse(source.read_text(encoding="utf-8"))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "forward_step_calc_loss")
    stub = ModuleType("megatron.core.transformer.multi_token_prediction")
    stub.MTPLossAutoScaler = object
    monkeypatch.setitem(sys.modules, stub.__name__, stub)
    return compile_function(node, {"torch": torch}, source)


@pytest.mark.parametrize("version", ["1.8", "1.8.2"])
@pytest.mark.parametrize("dp_size", [1, 4])
def test_actual_actor_backward_and_mcore_response_scaling(monkeypatch, dp_size, version):
    schedule = mcore_schedule(monkeypatch)
    batch, tokens = 8, 3
    behavior = torch.full((batch, tokens), .01).log()
    ratios = torch.tensor([[.25, 4., 20.], [2., .5, 1.]]).repeat(4, 1)
    current = (behavior + ratios.log()).requires_grad_()
    advantages = torch.tensor([1., -1., 0., 2., -2., 1., -1., .5])[:, None].expand(-1, tokens)
    mask = torch.ones(batch, tokens, dtype=torch.bool)
    mask[::2, -1] = False
    routes = torch.tensor([[[[0, 1]], [[1, 2]], [[0, 2]]]]).expand(batch, -1, -1, -1)
    config = SimpleNamespace(calculate_per_token_loss=False, timers=None)
    local_batch = batch // dp_size
    for i in range(batch):
        loss, _ = schedule(None, current[i:i+1],
            lambda z: actor_loss(z, behavior[i:i+1], advantages[i:i+1], mask[i:i+1], routes[i:i+1], .012, version),
            config, None, False, local_batch, [], cp_group_size=1, is_last_stage=True)
        # Dense DDP average; MCore applies the same effective 1/DP to expert gradients.
        (loss / dp_size).backward()
    expected = torch.zeros_like(current)
    for i in range(batch):
        clusters = [(routes[i, :, 0] == e).any(-1) & mask[i] for e in range(3)]
        clusters = [support for support in clusters if support.any()]
        for support in clusters:
            r = ratios[i, support]
            a = advantages[i, 0]
            outward = (a > 0 and r.mean() > 1) or (a < 0 and r.mean() < 1)
            tv = (current.detach()[i, support].exp() - behavior[i, support].exp()).abs().mean()
            if not (outward and tv > .012):
                denominator = (sum(int(s.sum()) for s in clusters) if version == "1.8.2"
                               else len(clusters) * support.sum())
                expected[i, support] -= a * r.clamp(max=5) / (batch * denominator)
    torch.testing.assert_close(current.grad, expected, rtol=2e-6, atol=2e-7)
    assert current.grad[~mask].eq(0).all()


@pytest.mark.parametrize("version", ["1.8", "1.8.2"])
def test_v18_fit_uses_existing_ds_before_real_unnormalized_grpo(monkeypatch, version):
    trainer, run, updates, seen, logs = legacy.fit_harness(monkeypatch, version, [24, 40], target=64)
    trainer.config.algorithm.filter_groups.enable = True
    trainer.config.algorithm.norm_adv_by_std_in_grpo = False
    # The existing harness exposes the actual fit globals via its closure.
    namespace = next(cell.cell_contents for cell in run.__closure__ if isinstance(cell.cell_contents, dict))
    path = ROOT / "verl/trainer/ppo/core_algos.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "compute_grpo_outcome_advantage")
    grpo = compile_function(node, {"torch": torch, "np": np, "defaultdict": defaultdict}, path)
    original = namespace["compute_advantage"]
    verified = []
    def advantage(data, **kwargs):
        assert kwargs["norm_adv_by_std_in_grpo"] is False
        original(data, **kwargs)
        rewards = data.batch["token_level_rewards"]
        actual, _ = grpo(rewards, data.batch["response_mask"], data.non_tensor_batch["uid"], norm_adv_by_std_in_grpo=False)
        scores = rewards.sum(-1)
        for uid in set(data.non_tensor_batch["uid"]):
            rows = torch.tensor(data.non_tensor_batch["uid"] == uid)
            expected = (scores[rows] - scores[rows].mean())[:, None] * data.batch["response_mask"][rows]
            torch.testing.assert_close(actual[rows], expected)
        verified.append(actual)
        return data
    namespace["compute_advantage"] = advantage
    run()
    assert len(updates) == len(seen) == len(verified) == 1
    assert len(updates[0]) == 512
    assert logs[0]["dynamic_sampling/generation_rounds"] == 2
    assert logs[0]["dynamic_sampling/accepted_groups"] == 64
    assert "old_log_probs" not in updates[0].batch


def test_v182_actual_actor_masks_nan_padding_before_loss():
    z=torch.tensor([[-3.,float("nan")]],requires_grad=True)
    b=torch.tensor([[-4.,float("nan")]])
    mask=torch.tensor([[True,False]])
    routes=torch.tensor([[[[0,1]],[[-1,-1]]]])
    loss,_=actor_loss(z,b,torch.ones_like(z),mask,routes,1.,"1.8.2")
    assert torch.isfinite(loss)
    loss.backward()
    assert z.grad[0,1]==0 and z.grad[0,0]!=0
