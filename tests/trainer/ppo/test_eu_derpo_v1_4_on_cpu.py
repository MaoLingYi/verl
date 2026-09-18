import importlib.util
from pathlib import Path

import torch


ROOT = Path(__file__).parents[3]
SPEC = importlib.util.spec_from_file_location("eu_derpo_v14_under_test", ROOT / "verl/trainer/ppo/eu_derpo.py")
eu = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(eu)


def stats(behavior_probability, aligned_probability, current_probability=0.60):
    current = torch.log(torch.tensor([[current_probability]]))
    behavior = torch.log(torch.tensor([[behavior_probability]]))
    aligned = torch.log(torch.tensor([[aligned_probability]]))
    routes = torch.tensor([[[[7]]]], dtype=torch.uint8)
    mask = torch.ones_like(current, dtype=torch.bool)
    result = eu.cluster_statistics(
        current,
        behavior,
        torch.ones_like(current),
        mask,
        routes,
        128,
        0.02,
        aligned_old_logp=aligned,
        optimization_anchor="rollout",
    )
    _, objective = eu.edppo_token_coefficients(result, routes, mask)
    return result, objective


def active_value(tensor, result):
    return tensor[result.active].item()


def test_case_a_aligned_old_changes_diagnostics_but_not_production():
    first, first_objective = stats(0.50, 0.40)
    second, second_objective = stats(0.50, 0.55)
    torch.testing.assert_close(first.rho, second.rho)
    torch.testing.assert_close(first.divergence, second.divergence)
    assert torch.equal(first.mask, second.mask)
    torch.testing.assert_close(first_objective, second_objective)
    assert not torch.equal(first.delta_engine, second.delta_engine)
    assert not torch.equal(first.delta_update, second.delta_update)
    assert not torch.equal(first.rho_align, second.rho_align)
    assert not torch.equal(first.divergence_align, second.divergence_align)


def test_case_b_rollout_behavior_changes_production_and_can_change_mask():
    outward, outward_objective = stats(0.50, 0.55)
    inward, inward_objective = stats(0.59, 0.55)
    assert not torch.equal(outward.rho, inward.rho)
    assert not torch.equal(outward.divergence, inward.divergence)
    assert not torch.equal(outward.mask, inward.mask)
    assert not torch.equal(outward_objective, inward_objective)


def test_case_c_production_rho_is_theta_over_behavior():
    result, _ = stats(0.30, 0.50, current_probability=0.60)
    assert abs(active_value(result.rho, result) - 2.0) < 1.0e-6
    assert abs(active_value(result.rho_align, result) - 1.2) < 1.0e-6


def test_case_d_production_binary_tv_is_behavior_vs_current():
    result, _ = stats(0.30, 0.50, current_probability=0.60)
    assert abs(active_value(result.divergence, result) - 0.30) < 1.0e-6
    assert abs(active_value(result.divergence_align, result) - 0.10) < 1.0e-6


def test_case_e_behavior_delta_decomposes_into_align_plus_engine():
    result, _ = stats(0.30, 0.50, current_probability=0.60)
    torch.testing.assert_close(
        result.delta_total[result.active],
        result.delta_update[result.active] + result.delta_engine[result.active],
    )


def test_v14_actor_selects_anchor_by_versioned_behavior_source():
    actor = (ROOT / "verl/workers/actor/megatron_actor.py").read_text(encoding="utf-8")
    assert actor.count("optimization_anchor=self.config.eu_derpo.behavior_expert_is.behavior_source") == 2
    assert 'self.config.eu_derpo.version in {"1.3", "1.4"}' in actor
