import importlib.util
from pathlib import Path

import torch


ROOT = Path(__file__).parents[3]
SPEC = importlib.util.spec_from_file_location("eu_derpo_v13_under_test", ROOT / "verl" / "trainer" / "ppo" / "eu_derpo.py")
eu = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(eu)


def test_v13_uses_aligned_old_for_optimization_and_keeps_raw_diagnostics():
    current = torch.log(torch.tensor([[0.60, 0.30]]))
    aligned_old = torch.log(torch.tensor([[0.50, 0.25]]))
    rollout = torch.log(torch.tensor([[0.40, 0.20]]))
    routes = torch.tensor([[[[0, 1]], [[0, 1]]]], dtype=torch.uint8)
    mask = torch.ones_like(current, dtype=torch.bool)
    stats = eu.cluster_statistics(
        current,
        rollout,
        torch.ones_like(current),
        mask,
        routes,
        128,
        0.02,
        aligned_old_logp=aligned_old,
        optimization_anchor="aligned_old",
    )
    active = stats.active
    expected_update = (current - aligned_old).mean().exp()
    expected_raw = (current - rollout).mean().exp()
    torch.testing.assert_close(stats.rho[active], expected_update.expand_as(stats.rho[active]))
    torch.testing.assert_close(stats.rho_raw[active], expected_raw.expand_as(stats.rho_raw[active]))
    torch.testing.assert_close(
        stats.delta_total[active],
        stats.delta_update[active] + stats.delta_engine[active],
    )
    assert torch.all(stats.divergence[active] == (aligned_old.exp() - current.exp()).abs().mean())
    assert torch.all(stats.divergence_engine[active] == (rollout.exp() - aligned_old.exp()).abs().mean())


def test_v13_route_compaction_is_response_only_uint8_and_accepts_expert_127():
    full = torch.zeros((1, 5, 48, 8), dtype=torch.int64)
    full[:, -3:-1] = torch.tensor([0, 1, 2, 3, 4, 5, 6, 127])
    compact, metrics = eu.prepare_v13_rollout_routes(full, torch.tensor([[1, 1]]), 2)
    assert compact.shape == (1, 2, 48, 8)
    assert compact.dtype == torch.uint8
    assert metrics["rollout_route_max_expert_id"] == 127
    assert metrics["rollout_route_host_bytes"] == compact.numel()


def test_v13_route_compaction_rejects_out_of_range_before_uint8_cast():
    full = torch.zeros((1, 2, 48, 8), dtype=torch.int64)
    full[:, 0, ..., -1] = 256
    try:
        eu.prepare_v13_rollout_routes(full, torch.tensor([[1]]), 1)
    except RuntimeError as error:
        assert "global expert ids" in str(error)
    else:
        raise AssertionError("expert id 256 must not wrap to uint8 zero")


def test_rollout_current_match_is_diagnostic_not_an_equality_assertion():
    rollout = torch.tensor([[[[0, 1]], [[2, 3]]]], dtype=torch.uint8)
    current = torch.tensor([[[[1, 0]], [[2, 4]]]], dtype=torch.uint8)
    assert eu.route_match_fraction(rollout, current, torch.tensor([[1, 1]])) == 0.5
