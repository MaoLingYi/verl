import ast
import importlib.util
import math
import socket
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import pytest
import torch
import torch.multiprocessing as mp


ROOT = Path(__file__).parents[3]
SPEC = importlib.util.spec_from_file_location("eu_derpo_v16_under_test", ROOT / "verl/trainer/ppo/eu_derpo.py")
eu = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(eu)


def controller(delta=0.012, beta=0.0027):
    return eu.P95MCTRController(delta, beta, 0.012, 0.018)


def _distributed_p95_worker(rank, world_size, init_method, result_dir):
    torch.distributed.init_process_group("gloo", init_method=init_method, rank=rank, world_size=world_size)
    shards = (torch.tensor([0.01, 0.011, 0.012]), torch.tensor([0.013, 0.014, 0.030]))
    q_raw, count = eu.distributed_exact_p95(shards[rank])
    state = eu.P95MCTRController(0.012, 0.0027, 0.012, 0.018)
    delta_next = state.update(q_raw, count)["mctr/delta_next"]
    torch.save((q_raw, count, delta_next), Path(result_dir) / f"rank-{rank}.pt")
    torch.distributed.destroy_process_group()


def _production_mctr_helper(mpu):
    actor_path = ROOT / "verl/workers/actor/megatron_actor.py"
    tree = ast.parse(actor_path.read_text(encoding="utf-8"), filename=str(actor_path))
    function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_global_mctr_outward_p95"
    )
    namespace = {
        "torch": torch,
        "mpu": mpu,
        "distributed_exact_p95": eu.distributed_exact_p95,
        "get_device_id": lambda: torch.device("cpu"),
    }
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, function], type_ignores=[]))
    exec(compile(module, str(actor_path), "exec"), namespace)
    return namespace["_global_mctr_outward_p95"]


def _production_ownership_worker(rank, world_size, init_method, result_dir):
    torch.distributed.init_process_group("gloo", init_method=init_method, rank=rank, world_size=world_size)
    dp_groups = [
        torch.distributed.new_group([0, 2]),
        torch.distributed.new_group([1, 3]),
    ]
    tp_groups = [
        torch.distributed.new_group([0, 1]),
        torch.distributed.new_group([2, 3]),
    ]
    tp_rank = rank % 2
    dp_rank = rank // 2
    mpu = SimpleNamespace(
        get_pipeline_model_parallel_world_size=lambda: 1,
        get_tensor_model_parallel_rank=lambda: tp_rank,
        get_data_parallel_group=lambda: dp_groups[tp_rank],
        get_tensor_model_parallel_world_size=lambda: 2,
        get_tensor_model_parallel_src_rank=lambda: dp_rank * 2,
        get_tensor_model_parallel_group=lambda: tp_groups[dp_rank],
    )
    shard = (
        torch.tensor([0.010, 0.011, 0.012])
        if dp_rank == 0 else torch.tensor([0.013, 0.014, 0.030])
    )
    q_raw, count = _production_mctr_helper(mpu)(shard)
    state = controller()
    delta_next = state.update(q_raw, count)["mctr/delta_next"]
    torch.save((q_raw, count, delta_next), Path(result_dir) / f"production-rank-{rank}.pt")
    torch.distributed.destroy_process_group()


def test_controller_clips_before_ema_and_holds_at_minimum():
    low = controller()
    metrics = low.update(0.010, 4)
    assert metrics["mctr/q_outward_p95_clipped"] == 0.012
    assert metrics["mctr/delta_next"] == 0.012

    high = controller()
    metrics = high.update(0.030, 4)
    assert metrics["mctr/q_outward_p95_clipped"] == 0.018
    assert metrics["mctr/delta_next"] == pytest.approx(0.0120162)
    assert controller(beta=1).update(0.018, 4)["mctr/delta_next"] == 0.018


def test_v16_config_freezes_bounds_without_affecting_v14():
    source_path = ROOT / "verl/workers/config/actor.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    names = {"BehaviorExpertISConfig", "ExpertClusterDPPOConfig", "RoutingUtilityConfig", "MCTRConfig", "EUDERPOConfig"}
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name in names]

    @dataclass
    class BaseConfig:
        pass

    namespace = {"dataclass": dataclass, "field": field, "Optional": Optional, "BaseConfig": BaseConfig}
    exec(compile(ast.Module(body=classes, type_ignores=[]), str(source_path), "exec"), namespace)
    common = {
        "enabled": True,
        "partial_rollout_old_alignment": True,
        "behavior_expert_is": namespace["BehaviorExpertISConfig"](enabled=True),
        "expert_cluster_dppo": namespace["ExpertClusterDPPOConfig"](enabled=True, delta_e=0.012),
        "routing_utility": namespace["RoutingUtilityConfig"](enabled=True, lambda_u=0.15, diagnostics=True),
    }
    namespace["EUDERPOConfig"](
        version="1.6", **common,
        mctr=namespace["MCTRConfig"](enabled=True, beta=1, delta_min=0.012, delta_max=0.018),
    )
    for bounds in ((0.011, 0.018), (0.012, 0.019)):
        with pytest.raises(ValueError, match="freezes MCTR bounds"):
            namespace["EUDERPOConfig"](
                version="1.6", **common,
                mctr=namespace["MCTRConfig"](enabled=True, beta=0.0027, delta_min=bounds[0], delta_max=bounds[1]),
            )
    namespace["EUDERPOConfig"](version="1.4", **common)


def test_controller_ema_invalid_and_empty_fallbacks():
    mid = controller(0.015)
    assert mid.update(0.018, 2)["mctr/delta_next"] == pytest.approx(0.0150081)
    before = mid.delta_current
    invalid = mid.update(float("nan"), 2)
    assert invalid["mctr/delta_next"] == before
    assert invalid["mctr/fallback_invalid_q"] == 1.0
    empty = mid.update(None, 0)
    assert empty["mctr/delta_next"] == before
    assert empty["mctr/fallback_invalid_q"] == 1.0


@pytest.mark.parametrize("delta", [0.0119, 0.0181])
def test_delta0_outside_frozen_envelope_fails_before_first_step(delta):
    with pytest.raises(ValueError, match="bounded delta_initial"):
        controller(delta)


def _stats(delta):
    current = torch.log(torch.tensor([[0.516]], requires_grad=True))
    behavior = torch.log(torch.tensor([[0.500]]))
    aligned = torch.log(torch.tensor([[0.505]]))
    routes = torch.tensor([[[[7]]]], dtype=torch.uint8)
    mask = torch.ones_like(current, dtype=torch.bool)
    result = eu.cluster_statistics(
        current.detach(), behavior, torch.ones_like(current), mask, routes, 128, delta,
        aligned_old_logp=aligned, optimization_anchor="rollout",
    )
    _, objective = eu.edppo_token_coefficients(result, routes, mask)
    return result, objective


def test_one_step_lag_and_fixed_delta_regression():
    state = controller()
    delta_k = state.delta_current
    first, objective_first = _stats(delta_k)
    assert first.directional_outward[first.active].item()
    assert not first.mask[first.active].item()
    state.update(0.018, 1)
    repeated, objective_repeated = _stats(delta_k)
    assert torch.equal(first.mask, repeated.mask)
    torch.testing.assert_close(objective_first, objective_repeated)
    assert state.delta_current > delta_k


def _complete_fixed_delta_path(delta):
    current = torch.log(torch.tensor([[0.52, 0.48], [0.54, 0.46]]))
    behavior = torch.log(torch.tensor([[0.50, 0.50], [0.50, 0.50]]))
    aligned = torch.log(torch.tensor([[0.51, 0.49], [0.52, 0.48]]))
    advantages = torch.ones_like(current)
    response_mask = torch.ones_like(current, dtype=torch.bool)
    routes = torch.tensor([[[[0, 1]], [[1, 2]]], [[[0, 2]], [[1, 2]]]])
    stats = eu.cluster_statistics(
        current,
        behavior,
        advantages,
        response_mask,
        routes,
        3,
        delta,
        aligned_old_logp=aligned,
        optimization_anchor="rollout",
    )
    _, j_edppo = eu.edppo_token_coefficients(stats, routes, response_mask)

    alpha = torch.tensor([[[[0.6, 0.4]], [[0.3, 0.7]]], [[[0.55, 0.45]], [[0.2, 0.8]]]])
    sensitivity = torch.tensor([[[[0.2, -0.1]], [[-0.4, 0.3]]], [[[0.1, -0.2]], [[0.5, -0.3]]]])
    u_r = eu.centered_routing_utility(alpha, sensitivity)
    utility_sum, utility_count = eu.aggregate_edge_utility(u_r, routes, response_mask, 3)
    u_hat, _ = eu.normalize_group_utility(
        utility_sum, utility_count, torch.tensor([0, 0]), 1.0e-6, 2, 1.0e-6
    )
    selected_u_hat = torch.stack(
        [u_hat[:, layer].gather(1, routes[:, :, layer].reshape(2, -1)).reshape(2, 2, 2)
         for layer in range(routes.shape[2])],
        dim=2,
    )
    selected_gate = torch.stack(
        [stats.mask[:, layer].gather(1, routes[:, :, layer].reshape(2, -1)).reshape(2, 2, 2)
         for layer in range(routes.shape[2])],
        dim=2,
    )
    edge_coefficient = selected_u_hat * selected_gate
    hidden = torch.arange(16, dtype=torch.float32).reshape(4, 4) / 10
    router_weight = torch.arange(12, dtype=torch.float32).reshape(3, 4) / 20
    j_u = eu.router_auxiliary_loss(
        hidden,
        router_weight,
        routes.reshape(4, 2),
        edge_coefficient.reshape(4, 2),
    )
    return {
        "rho_E": stats.rho,
        "D_E": stats.divergence,
        "M_E": stats.mask,
        "J_E": j_edppo,
        "s": sensitivity,
        "uR": u_r,
        "UR_sum": utility_sum,
        "UR_count": utility_count,
        "Uhat": u_hat,
        "J_U": j_u,
    }


def test_v16_frozen_delta_is_complete_v14_math_regression():
    fixed_v14 = _complete_fixed_delta_path(0.012)
    mctr = controller()
    frozen_v16 = _complete_fixed_delta_path(mctr.delta_current)
    assert fixed_v14.keys() == frozen_v16.keys()
    for name in fixed_v14:
        if fixed_v14[name].dtype == torch.bool:
            assert torch.equal(fixed_v14[name], frozen_v16[name]), name
        else:
            torch.testing.assert_close(fixed_v14[name], frozen_v16[name], msg=name)


def test_controller_and_candidate_path_are_detached():
    result, _ = _stats(0.012)
    candidates = result.divergence[result.directional_outward].detach()
    assert candidates.grad_fn is None
    q = eu.exact_p95(candidates)
    state = controller()
    state.update(q, candidates.numel())
    assert isinstance(state.delta_current, float)


def test_exact_global_p95_is_not_average_of_local_p95s_and_ownership_is_unique():
    shards = [torch.tensor([0.01, 0.011, 0.012]), torch.tensor([0.013, 0.014, 0.030])]
    exact = eu.exact_p95(torch.cat(shards))
    local_average = sum(eu.exact_p95(shard) for shard in shards) / len(shards)
    assert exact == pytest.approx(torch.quantile(torch.cat(shards), 0.95).item())
    assert not math.isclose(exact, local_average)
    tp_replicas = [(0, shard) for shard in shards] + [(1, shard.clone()) for shard in shards]
    owned = torch.cat([shard for tp_rank, shard in tp_replicas if tp_rank == 0])
    assert owned.numel() == sum(shard.numel() for shard in shards)
    assert eu.exact_p95(owned) == exact


def test_distributed_global_p95_matches_exact_reference_on_every_rank(tmp_path):
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    init_method = f"tcp://127.0.0.1:{port}"
    mp.spawn(_distributed_p95_worker, args=(2, init_method, str(tmp_path)), nprocs=2, join=True)
    results = [torch.load(tmp_path / f"rank-{rank}.pt", weights_only=True) for rank in range(2)]
    expected = torch.quantile(torch.tensor([0.01, 0.011, 0.012, 0.013, 0.014, 0.030]), 0.95).item()
    expected_delta = (1 - 0.0027) * 0.012 + 0.0027 * min(expected, 0.018)
    for q_raw, count, delta_next in results:
        assert q_raw == pytest.approx(expected)
        assert count == 6
        assert delta_next == pytest.approx(expected_delta)


def test_production_tp_owner_dense_dp_gather_and_tp_broadcast_are_rank_consistent(tmp_path):
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    init_method = f"tcp://127.0.0.1:{port}"
    mp.spawn(_production_ownership_worker, args=(4, init_method, str(tmp_path)), nprocs=4, join=True)
    results = [
        torch.load(tmp_path / f"production-rank-{rank}.pt", weights_only=True)
        for rank in range(4)
    ]
    unique_distribution = torch.tensor([0.010, 0.011, 0.012, 0.013, 0.014, 0.030])
    expected = torch.quantile(unique_distribution, 0.95).item()
    local_p95_mean = (
        torch.quantile(unique_distribution[:3], 0.95)
        + torch.quantile(unique_distribution[3:], 0.95)
    ).item() / 2
    expected_delta = (1 - 0.0027) * 0.012 + 0.0027 * min(expected, 0.018)
    assert not math.isclose(expected, local_p95_mean)
    for q_raw, count, delta_next in results:
        assert q_raw == pytest.approx(expected)
        assert count == unique_distribution.numel()
        assert delta_next == pytest.approx(expected_delta)


def test_checkpoint_round_trip_restores_delta_step_and_validates_beta():
    original = controller()
    original.update(0.018, 5)
    restored = controller()
    restored.load_state_dict(original.state_dict())
    assert restored.delta_current == original.delta_current
    assert restored.controller_step == original.controller_step
    wrong_beta = eu.P95MCTRController(0.012, 0.01, 0.012, 0.018)
    with pytest.raises(ValueError, match="differs from resolved config"):
        wrong_beta.load_state_dict(original.state_dict())


def test_actor_lifecycle_uses_unique_owner_and_updates_only_after_optimizer_step():
    actor = (ROOT / "verl/workers/actor/megatron_actor.py").read_text(encoding="utf-8")
    helper = actor[actor.index("def _global_mctr_outward_p95") : actor.index("def _append_eu_derpo_scalar_metrics")]
    assert "get_tensor_model_parallel_rank() != 0" in helper
    assert "distributed_exact_p95(owned.to(get_device_id()), mpu.get_data_parallel_group())" in helper
    update = actor[actor.index("def update_policy") :]
    assert update.index("self.actor_optimizer.step()") < update.index("self.mctr_controller.update(")
    loss = actor[actor.index("def loss_func") : actor.index("def update_policy")]
    assert "self._eu_derpo_delta_current if self.mctr_controller is not None" in loss


def test_checkpoint_manager_wires_controller_extra_state():
    source = (ROOT / "verl/utils/checkpoint/megatron_checkpoint_manager.py").read_text(encoding="utf-8")
    assert 'state_dict["eu_derpo_controller"] = extra_state' in source
    assert 'extra_state_loader(state_dict["eu_derpo_controller"])' in source
