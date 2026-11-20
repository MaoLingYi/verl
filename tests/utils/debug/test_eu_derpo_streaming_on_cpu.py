"""CPU graph/reference checks for zero-Hidden, lambda-zero utility collection.

The Router is the existing 128-expert/Top-8 fixture, not a production MCore
kernel. Distributed ownership checks partition real rows without starting CUDA.
"""
from __future__ import annotations

from contextlib import ExitStack
import ast
from dataclasses import dataclass, field
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch
from typing import Optional

import pytest
import torch


ROOT = Path(__file__).parents[3]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


FIXTURE = _load("streaming_observer_fixture", Path(__file__).with_name("test_eu_derpo_observer_on_cpu.py"))


@pytest.fixture
def harness():
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    h = FIXTURE.TestEUDERPOObserver()
    h.setUp()
    names = ["verl.trainer.ppo.eu_derpo_v18", "verl.trainer.ppo.eu_derpo_v182", "verl.trainer.ppo.eu_derpo_v183"]
    previous = {name: sys.modules.get(name) for name in names}
    try:
        for name in names:
            _load(name, ROOT / "verl/trainer/ppo" / (name.rsplit(".", 1)[1] + ".py"))
        yield h
    finally:
        h.tearDown()
        for name, old in previous.items():
            if old is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old
        torch.set_num_threads(previous_threads)


def _inputs(dtype):
    generator = torch.Generator().manual_seed(183)
    lengths = torch.tensor([1, 7, 2, 6, 3, 5, 4, 7])
    response_mask = torch.arange(7)[None, :] < lengths[:, None]
    hidden = torch.randn(8, 9, 4, generator=generator) * 0.125
    hidden[..., 0] = 1.0
    return hidden.to(torch.bfloat16).to(dtype), response_mask


def _new_observer(h, mode, dtype):
    h.observer.close()
    h.model = FIXTURE.FakeModel().to(dtype=dtype)
    with torch.no_grad():
        for router in h.model.routers:
            # All eight responses share experts, including 127; remaining
            # experts have empty clusters. Other features vary the sensitivities.
            router.weight[:, 0] = torch.linspace(-2, 2, 128).to(dtype)
    h.observer = h.observer_class(
        [h.model], FIXTURE.observer_config(), diagnostics=True,
        utility_execution_mode=mode,
        cache_backend="ram_nfs_hybrid" if mode == "streaming_diagnostic" else "cpu_pageable",
    )
    return h.observer


def _run_update(h, mode, dtype, generation, backbone, optimizer):
    observer = h.observer
    rng_before = torch.get_rng_state().clone()
    math = sys.modules["verl.trainer.ppo.eu_derpo"]
    v18 = sys.modules["verl.trainer.ppo.eu_derpo_v18"]
    v183 = sys.modules["verl.trainer.ppo.eu_derpo_v183"]
    hidden, response_mask = _inputs(dtype)
    ids = torch.arange(100, 108)
    groups = torch.full((8,), 73)
    input_ids = torch.ones((8, 9), dtype=torch.long)
    attention = torch.ones_like(input_ids, dtype=torch.bool)
    advantages = torch.tensor([1., -1., .5, 2., -2., .75, 1.5, -.5])[:, None].expand(8, 7)
    expert_value = torch.sin(torch.arange(128, dtype=torch.float32) * .3)
    reference_sum_log_q = torch.zeros((8, 48, 128))
    reference_sum_u = torch.zeros_like(reference_sum_log_q)
    all_stats = []
    optimizer.zero_grad(set_to_none=True)
    observer.start_main_batch(ids, groups, attention, response_mask, 7, generation)
    if mode == "streaming_diagnostic":
        assert observer._hidden_cache is None and observer._file_store is None
        assert observer._support_cache is None

    for row in range(8):
        row_ids = ids[row:row + 1]
        row_input = input_ids[row:row + 1]
        row_attention = attention[row:row + 1]
        row_mask = response_mask[row:row + 1]
        observer.begin_main_microbatch(row_input, row_attention, row_ids, row_mask, 7)

        def model_hidden():
            # Real BF16 actual-F values, including a differentiable upstream
            # parameter. The FP32 test promotes those exact BF16 values.
            return (hidden[row] * backbone).to(torch.bfloat16).to(dtype)

        with torch.no_grad():
            f_hidden = model_hidden()
            f_outputs = h.model(f_hidden)
            f_logp = sum((p * expert_value).sum(-1) for p, _ in f_outputs)[1:8][None, :] / 48
        routes = observer.finish_main_microbatch(row_input, row_attention, row_ids, 7)
        observer.record_main_logprobs(row_ids, routes, f_logp)
        behavior = f_logp - torch.log(torch.tensor([.7, 1.1, .9, 1.2, .8, 1.05, .95]))
        stats = v18.cluster_statistics(f_logp, behavior, advantages[row:row + 1], row_mask, routes, 128, .012)
        all_stats.append(stats)
        coefficients = v183.schedule_coefficients(stats, routes, row_mask, int(response_mask.sum()), 8, 4)
        outputs = h.model(model_hidden())
        for p, _ in outputs:
            p.retain_grad()
        logp = sum((p * expert_value).sum(-1) for p, _ in outputs)[1:8][None, :] / 48
        # MCore divides by the 8 microbatches, then dense-DP averages by 4.
        (-(coefficients * logp).sum() / (8 * 4)).backward()
        for layer, (probabilities, routing_map) in enumerate(outputs):
            for token in row_mask[0].nonzero().flatten().tolist():
                sequence_index = token + 1
                support = routing_map[sequence_index].nonzero().flatten()
                alpha = probabilities[sequence_index, support].detach().float()
                sensitivity = -probabilities.grad[sequence_index, support].float()
                u = sensitivity - (alpha * sensitivity).sum()
                reference_sum_u[row, layer, support] += u
                logits = torch.nn.functional.linear(f_hidden[sequence_index:sequence_index + 1], h.model.routers[layer].weight)
                log_q = logits[0, support].detach().float().log_softmax(-1)
                reference_sum_log_q[row, layer, support] += log_q
    for router in h.model.routers:
        router.weight.main_grad = router.weight.grad.detach().clone()
    observer.wrap_native_finalize(lambda: None)()
    sums, sums_sq, counts, capture = observer.finish_main_batch()
    consumed = list(observer.consume_post_backward_rows(ids))
    assert len(consumed) == 8 and not observer.route_cache and not observer.current_logprob_cache
    torch.testing.assert_close(sums, reference_sum_u, rtol=2e-5, atol=1e-8)
    normalized, skipped = math.normalize_group_utility(sums, counts, groups, 1e-6, 2, 5e-7)
    assert normalized.abs().max() > .1, "fixture must have a real, nonzero teacher"
    assert counts[:, :, 127].sum() > 0
    assert (counts == 0).any()
    assert torch.count_nonzero(normalized[counts == 0]) == 0
    combined = SimpleNamespace(
        count=torch.cat([item.count for item in all_stats]),
        mask=torch.cat([item.mask for item in all_stats]),
    )
    assert combined.mask.any() and ((counts > 0) & ~combined.mask).any()
    active = (combined.count > 0).sum((1, 2)).float()
    main_before = [router.weight.main_grad.clone() for router in h.model.routers]
    backbone_before = backbone.grad.clone()
    with ExitStack() as stack:
        if mode == "streaming_diagnostic":
            stack.enter_context(patch("torch.autograd.grad", side_effect=AssertionError("aux autograd")))
            stack.enter_context(patch.object(h.observer_module, "_reduce_router_auxiliary_grad", side_effect=AssertionError("aux reduce")))
            for router in h.model.routers:
                stack.enter_context(patch.object(router, "_verl_eu_derpo_original_gating", side_effect=AssertionError("Step E gating")))
            metrics = observer.finish_streaming_utility(ids, combined, normalized, active, 0., generation)
            torch.testing.assert_close(observer._sum_log_q, reference_sum_log_q, rtol=2e-6, atol=4e-6)
            assert observer._aux_reduce_count == [0] * 48
            assert observer._main_grad_add_count == [0] * 48
        else:
            metrics = observer.run_router_only_step_e(ids, combined, normalized, active, 0., generation)
    observer.validate_before_optimizer_step(generation)
    assert metrics["utility_router_grad_norm"] == 0
    hidden_metrics = observer.hidden_store_metrics(capture, metrics, 0.)
    if mode == "streaming_diagnostic":
        for key in ("hidden_payload_bytes", "hidden_nas_write_bytes", "hidden_nas_read_bytes", "auxiliary_gradient_add_count"):
            assert hidden_metrics[key] == 0, key
    for router, before in zip(h.model.routers, main_before):
        assert torch.equal(router.weight.main_grad, before)
    assert torch.equal(backbone.grad, backbone_before)
    denominator = 8 * active[:, None, None] * counts.clamp_min(1)
    reference_objective = (combined.mask * normalized * reference_sum_log_q / denominator).double().sum()
    assert abs(metrics["utility_objective"] - reference_objective.item()) < 3e-7
    optimizer.step()
    observer.finish_optimizer_step(generation + 1)
    assert observer._hidden_cache is None and observer._file_store is None
    assert observer._utility_sum is None and observer._support_cache is None
    return dict(sums=sums.clone(), sums_sq=sums_sq.clone(), counts=counts.clone(),
                utility=sums / counts.clamp_min(1), normalized=normalized.clone(), skipped=skipped.clone(),
                gradients=main_before, backbone_gradient=backbone_before,
                weights=[p.detach().clone() for p in h.model.parameters()], backbone=backbone.detach().clone(),
                objective=metrics["utility_objective"], native_norm=metrics["native_router_grad_norm"],
                main_norm=(sum(g.float().square().sum() for g in main_before) + backbone_before.float().square().sum()).sqrt().item(),
                rng_before=rng_before, rng_after=torch.get_rng_state().clone(), capture=capture, stats=all_stats)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_actual_graph_eight_response_teacher_main_gradient_and_two_updates(harness, dtype):
    h = harness
    results = {}
    store = ModuleType("verl.utils.debug.eu_hidden_store")
    store.DirectHiddenStore = Mock(side_effect=AssertionError("HiddenStore construction"))
    store.HybridHiddenStore = Mock(side_effect=AssertionError("HiddenStore construction"))
    store.WRITE_BYTES = 4 * 1024**2
    for mode in ("legacy", "streaming_diagnostic"):
        _new_observer(h, mode, dtype)
        backbone = torch.nn.Parameter(torch.tensor(1., dtype=dtype))
        optimizer = torch.optim.SGD([*h.model.parameters(), backbone], lr=.001)
        controller = sys.modules["verl.trainer.ppo.eu_derpo"].P95MCTRController(.012, .1, .001, .1)
        results[mode] = []
        with patch.dict(sys.modules, {"verl.utils.debug.eu_hidden_store": store}):
            for generation in range(2):
                row = _run_update(h, mode, dtype, generation, backbone, optimizer)
                outward = torch.cat([s.divergence[s.directional_outward] for s in row["stats"]])
                q = torch.quantile(outward.float(), .95).item() if outward.numel() else None
                controller.update(q, outward.numel())
                row["controller"] = controller.state_dict()
                results[mode].append(row)
    store.DirectHiddenStore.assert_not_called()
    store.HybridHiddenStore.assert_not_called()
    for original, streamed in zip(results["legacy"], results["streaming_diagnostic"]):
        for field in ("sums", "sums_sq", "counts", "utility", "normalized", "skipped", "backbone", "backbone_gradient", "rng_before", "rng_after"):
            assert torch.equal(original[field], streamed[field]), field
        for field in ("gradients", "weights"):
            assert all(torch.equal(a, b) for a, b in zip(original[field], streamed[field])), field
        assert original["controller"] == streamed["controller"]
        assert original["native_norm"] == streamed["native_norm"]
        assert original["main_norm"] == streamed["main_norm"]
        assert torch.equal(streamed["rng_before"], streamed["rng_after"])
        error = abs(original["objective"] - streamed["objective"])
        print(f"{dtype} J_U absolute error={error:.9g}; main/native/U/Uhat/update error=0")
        assert error < 3e-7


def test_normalization_empty_singleton_zero_variance_and_threshold(harness):
    normalize = sys.modules["verl.trainer.ppo.eu_derpo"].normalize_group_utility
    counts = torch.zeros(8, 1, 128)
    sums = torch.zeros_like(counts)
    counts[:, 0, 127] = torch.arange(1, 9)
    sums[:, 0, 127] = counts[:, 0, 127] * torch.arange(8)
    counts[:, 0, 126] = 1
    sums[:, 0, 126] = 2
    counts[0, 0, 125] = 1
    sums[0, 0, 125] = 4
    counts[:, 0, 124] = 1
    sums[:, 0, 124] = torch.tensor([0., 2e-7] * 4)
    normalized, skipped = normalize(sums, counts, torch.ones(8, dtype=torch.long), 1e-6, 2, 5e-7)
    assert normalized[:, 0, 127].abs().max() > 1
    assert skipped[:, 0, 126].all() and skipped[0, 0, 125]
    assert skipped[:, 0, 124].all()
    assert not skipped[:, 0, :124].any()  # absent clusters are not denominator members
    assert torch.count_nonzero(normalized[:, :, :127]) == 0


def test_tp_unequal_ownership_and_16k_positions(harness):
    eu = harness.observer_module
    length = 16384
    attention = torch.ones((1, length + 1025), dtype=torch.bool)
    mask = torch.ones((1, length), dtype=torch.bool)
    sample_ids = torch.tensor([999])
    plans = [eu._plan_local_response_rows(attention, mask, sample_ids, length, 2, rank) for rank in (0, 1)]
    assert plans[0]["rows"] != plans[1]["rows"]
    combined = torch.cat([plan["response_position"] for plan in plans])
    assert torch.equal(combined.sort().values, torch.arange(length, dtype=torch.int32))
    assert combined.max() == 16383 and combined.min() == 0
    # Each SP owner contributes its own log-q sum, while globally normalized
    # coefficients are replicated. Summing owner objectives must count each row once.
    token_log_q = torch.linspace(-3., -.1, length)
    local = [token_log_q[plan["response_position"].long()].double().sum() for plan in plans]
    torch.testing.assert_close(sum(local), token_log_q.double().sum(), rtol=0, atol=1e-10)
    observer = _new_observer(harness, "streaming_diagnostic", torch.float32)
    for current_length in (7, length):
        observer.start_main_batch(
            sample_ids, torch.tensor([91]), torch.ones((1, current_length + 1025), dtype=torch.bool),
            torch.ones((1, current_length), dtype=torch.bool), current_length, 0,
        )
        assert observer._hidden_cache is None and observer._support_cache is None and observer._file_store is None
        assert observer._sum_log_q.shape == (1, 48, 128)
        assert observer._sum_log_q.numel() * observer._sum_log_q.element_size() == 24576
        assert observer._cache_plan["hidden"] == 0 and observer._cache_plan["support"] == 0
        observer.clear()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_selected_support_permutation_and_bf16_log_alpha_is_not_log_q(harness, dtype):
    logits = torch.linspace(-3., 4., 128)[None, :].to(dtype)
    support = torch.tensor([[127, 3, 51, 2, 91, 65, 12, 99]])
    coefficients = torch.tensor([[.3, -.2, 1., .5, -.7, .8, 1.2, -.4]])
    permutation = torch.tensor([7, 0, 2, 6, 1, 4, 3, 5])
    _, expected = harness.observer_module._selected_support_loss(logits, support, coefficients, 0.)
    _, permuted = harness.observer_module._selected_support_loss(
        logits, support[:, permutation], coefficients[:, permutation], 0.,
    )
    assert abs((permuted - expected).item()) < 1e-6
    log_q = logits.gather(1, support).float().log_softmax(-1)
    rounded_alpha_log = log_q.exp().to(torch.bfloat16).float().log()
    error = (rounded_alpha_log - log_q).abs().max().item()
    print(f"{dtype} log(BF16 alpha) versus FP32 log_q error={error:.9g}")
    assert error > 1e-4  # using alpha.log would silently change diagnostic semantics


def test_streaming_preserves_recompute_route_rejection(harness):
    h = harness
    _new_observer(h, "streaming_diagnostic", torch.float32)
    h.run_main_backward(recompute_shift=1)
    with pytest.raises(RuntimeError, match="route mismatch"):
        h.observer.finish_main_batch()


def test_streaming_rejects_nonzero_lambda_before_diagnostic(harness):
    h = harness
    _new_observer(h, "streaming_diagnostic", torch.float32)
    h.run_main_backward()
    sums, _, counts, _ = h.observer.finish_main_batch()
    stats = SimpleNamespace(count=counts, mask=counts > 0)
    before = [router.weight.main_grad.clone() for router in h.model.routers]
    with pytest.raises(RuntimeError, match="lambda"):
        h.observer.finish_streaming_utility(h.sample_ids, stats, sums, (counts > 0).sum((1, 2)), .1, 0)
    assert all(torch.equal(router.weight.main_grad, gradient) for router, gradient in zip(h.model.routers, before))


def test_streaming_tp_local_objectives_sum_once_and_reclaim(harness):
    h = harness
    observer = _new_observer(h, "streaming_diagnostic", torch.float32)
    h.run_main_backward()
    _, _, counts, capture = observer.finish_main_batch()
    stats = SimpleNamespace(count=counts, mask=counts > 0)
    normalized = torch.arange(128).float().expand_as(counts) / 128
    active = (counts > 0).sum((1, 2)).float()
    full = observer._sum_log_q.clone()
    first = torch.zeros_like(full)
    first[0] = full[0]  # one valid response token versus the other owner's two
    objectives = []
    with patch("torch.distributed.all_reduce", side_effect=AssertionError("duplicate TP reduction")):
        for local in (first, full - first, full):
            observer._sum_log_q = local
            observer._step_e_started = False
            result = observer.finish_streaming_utility(h.sample_ids, stats, normalized, active, 0., 0)
            objectives.append(result["utility_objective"])
    assert abs(objectives[0] + objectives[1] - objectives[2]) < 2e-7
    with pytest.raises(RuntimeError, match="already consumed"):
        observer.finish_streaming_utility(h.sample_ids, stats, normalized, active, 0., 0)
    memory = ModuleType("verl.utils.memory_utils")
    memory.post_step_e_host_reclaim = Mock(return_value={"reclaimed": True})
    with patch.dict(sys.modules, {"verl.utils.memory_utils": memory}):
        assert observer.reclaim_step_e_payloads() == {"reclaimed": True}
    assert observer._sum_log_q is None and observer._utility_sum is None
    assert observer._provenance is None and not observer._pending_forward_logits
    memory.post_step_e_host_reclaim.assert_called_once()
    observer.validate_before_optimizer_step(0)
    observer.finish_optimizer_step(1)


def test_legacy_nonzero_lambda_still_adds_real_auxiliary_gradient(harness):
    h = harness
    h.run_main_backward()
    _, _, counts, _ = h.observer.finish_main_batch()
    stats = SimpleNamespace(count=counts, mask=counts > 0)
    normalized = torch.arange(128).float().expand_as(counts) / 128
    before = [router.weight.main_grad.clone() for router in h.model.routers]
    metrics = h.observer.run_router_only_step_e(h.sample_ids, stats, normalized, (counts > 0).sum((1, 2)), .1, 0)
    assert metrics["utility_router_grad_norm"] > 0
    assert any(not torch.equal(router.weight.main_grad, gradient) for router, gradient in zip(h.model.routers, before))


def test_actual_config_acceptance_and_rejection_without_hydra_dependency():
    path = ROOT / "verl/workers/config/actor.py"
    names = {"BehaviorExpertISConfig", "ExpertClusterDPPOConfig", "RoutingUtilityConfig", "MCTRConfig", "EUDERPOConfig"}
    tree = ast.parse(path.read_text(encoding="utf8"))
    body = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name in names]
    namespace = dict(dataclass=dataclass, field=field, Optional=Optional, BaseConfig=object)
    # Execute the actual dataclasses and validation method; Hydra and the
    # unrelated worker/device classes are unnecessary for this configuration test.
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), namespace)
    config = namespace["EUDERPOConfig"]
    assert config().utility_execution_mode == "legacy"

    def build(mode="streaming_diagnostic", version="1.8.3", coefficient=0., **extra):
        return config(
            enabled=True, version=version, utility_execution_mode=mode,
            behavior_expert_is=namespace["BehaviorExpertISConfig"](enabled=True),
            expert_cluster_dppo=namespace["ExpertClusterDPPOConfig"](enabled=True, delta_e=.012),
            routing_utility=namespace["RoutingUtilityConfig"](enabled=True, lambda_u=coefficient, diagnostics=True),
            mctr=namespace["MCTRConfig"](enabled=True, beta=.1, delta_min=.012, delta_max=.018),
            **extra,
        )

    assert build(cache_backend="ram_nfs_hybrid").cache_root is None
    assert build(mode="legacy", coefficient=.1).routing_utility.lambda_u == .1
    for coefficient in (.1, -1., float("nan"), float("inf")):
        with pytest.raises(ValueError, match="Streaming Utility"):
            build(coefficient=coefficient)
    with pytest.raises(ValueError, match="Streaming Utility"):
        build(version="1.8.2")
    with pytest.raises(ValueError, match="Unknown"):
        build(mode="typo")
    with pytest.raises(ValueError, match="cache_root"):
        build(mode="legacy", cache_backend="ram_nfs_hybrid")
