from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch


ROOT = Path(__file__).parents[3]
MATH_MODULE = ROOT / "verl" / "trainer" / "ppo" / "eu_derpo.py"
OBSERVER_MODULE = ROOT / "verl" / "utils" / "debug" / "eu_derpo.py"
MISSING = object()


class FakeRouterShiftObserver:
    @staticmethod
    def _unpack_sequence_parallel(records, input_ids, attention_mask):
        if any(record is None for record in records):
            raise RuntimeError("missing fake Router output")
        batch, sequence = input_ids.shape
        return torch.stack(records, dim=1).reshape(batch, sequence, len(records), -1)

    @staticmethod
    def _pack_sequence_parallel(records, input_ids, attention_mask, response_length):
        full = torch.zeros(
            (input_ids.shape[0], input_ids.shape[1], *records.shape[2:]), dtype=records.dtype
        )
        full[:, -response_length - 1 : -1] = records
        return full.reshape(input_ids.numel(), *records.shape[2:])


class TopKRouter(torch.nn.Module):
    def __init__(self, seed):
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.weight = torch.nn.Parameter(torch.randn(128, 4, generator=generator) * 0.1)
        self.topk = 8
        self.route_shift = 0
        self.routing_map_shift = 0
        self.eval_route_shift = 0

    def gating(self, hidden):
        return torch.nn.functional.linear(hidden, self.weight)

    def routing(self, logits):
        logits = logits.reshape(-1, logits.shape[-1])
        if not self.training and self.eval_route_shift:
            logits = logits.roll(self.eval_route_shift, dims=-1)
        if self.route_shift:
            logits = logits.roll(self.route_shift, dims=-1)
        routes = logits.topk(self.topk, dim=-1).indices
        selected_logits = logits.gather(-1, routes)
        alpha = selected_logits.float().softmax(-1)
        probabilities = torch.zeros_like(logits, dtype=alpha.dtype).scatter(-1, routes, alpha)
        routing_map_routes = (routes + self.routing_map_shift) % logits.shape[-1]
        routing_map = torch.zeros_like(logits, dtype=torch.bool).scatter(-1, routing_map_routes, True)
        return probabilities, routing_map

    def forward(self, hidden):
        return self.routing(self.gating(hidden))


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.routers = torch.nn.ModuleList([TopKRouter(index) for index in range(48)])

    def forward(self, hidden):
        return [router(hidden) for router in self.routers]


def observer_config():
    return SimpleNamespace(
        moe_router_score_function="softmax",
        moe_router_pre_softmax=False,
        moe_router_topk_scaling_factor=None,
        moe_router_fusion=False,
        moe_router_load_balancing_type="none",
        moe_z_loss_coeff=None,
        moe_input_jitter_eps=None,
        moe_expert_capacity_factor=None,
        moe_apply_probs_on_input=False,
        tensor_model_parallel_size=2,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=8,
        expert_tensor_parallel_size=1,
        context_parallel_size=1,
        num_layers=48,
        recompute_granularity="full",
        recompute_method="uniform",
        recompute_num_layers=1,
        virtual_pipeline_model_parallel_size=None,
        sequence_parallel=True,
        hidden_dropout=0.0,
        attention_dropout=0.0,
    )


class TestEUDERPOObserver(unittest.TestCase):
    def setUp(self):
        math_spec = importlib.util.spec_from_file_location("eu_derpo_observer_math_under_test", MATH_MODULE)
        math_module = importlib.util.module_from_spec(math_spec)
        assert math_spec.loader is not None
        math_spec.loader.exec_module(math_module)

        parallel_state = types.ModuleType("megatron.core.parallel_state")
        parallel_state.get_data_parallel_world_size = lambda: 4
        parallel_state.get_expert_data_parallel_world_size = lambda: 1
        parallel_state.get_tensor_model_parallel_world_size = lambda: 1
        parallel_state.get_pipeline_model_parallel_world_size = lambda: 1
        parallel_state.get_pipeline_model_parallel_rank = lambda: 0
        parallel_state.get_tensor_model_parallel_rank = lambda: 0
        parallel_state.get_data_parallel_rank = lambda: 0
        parallel_state.get_expert_model_parallel_rank = lambda: 0
        parallel_state.get_tensor_model_parallel_group = lambda: "tp"
        parallel_state.get_pipeline_model_parallel_group = lambda: "pp"
        parallel_state.get_data_parallel_group = lambda: "dp"

        modules = {
            "verl": types.ModuleType("verl"),
            "verl.trainer": types.ModuleType("verl.trainer"),
            "verl.trainer.ppo": types.ModuleType("verl.trainer.ppo"),
            "verl.trainer.ppo.eu_derpo": math_module,
            "verl.utils": types.ModuleType("verl.utils"),
            "verl.utils.debug": types.ModuleType("verl.utils.debug"),
            "verl.utils.debug.router_shift": types.ModuleType("verl.utils.debug.router_shift"),
            "megatron": types.ModuleType("megatron"),
            "megatron.core": types.ModuleType("megatron.core"),
            "megatron.core.parallel_state": parallel_state,
        }
        modules["verl.utils.debug.router_shift"].RouterShiftObserver = FakeRouterShiftObserver
        modules["megatron.core"].parallel_state = parallel_state
        self.saved_modules = {name: sys.modules.get(name, MISSING) for name in modules}
        sys.modules.update(modules)

        observer_spec = importlib.util.spec_from_file_location("eu_derpo_observer_under_test", OBSERVER_MODULE)
        observer_module = importlib.util.module_from_spec(observer_spec)
        assert observer_spec.loader is not None
        observer_spec.loader.exec_module(observer_module)
        self.observer_module = observer_module
        self.parallel_state = parallel_state
        self.all_reduce_patcher = mock.patch(
            "torch.distributed.all_reduce", side_effect=self._simulate_identical_dp_gradients
        )
        self.all_reduce_patcher.start()
        self.observer_class = observer_module.EUDERPOObserver
        self.model = FakeModel()
        self.observer = self.observer_class([self.model], observer_config(), diagnostics=True)
        self.sample_ids = torch.tensor([17, 23], dtype=torch.int64)
        self.input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
        self.attention_mask = torch.ones_like(self.input_ids, dtype=torch.bool)
        self.response_mask = torch.tensor([[True, False], [True, True]])
        self.hidden = torch.tensor(
            [
                [[0.1, 0.2, 0.3, 0.4], [0.4, 0.1, -0.2, 0.3], [0.2, -0.1, 0.5, 0.3]],
                [[-0.3, 0.2, 0.6, 0.1], [0.7, -0.2, 0.1, 0.4], [0.3, 0.8, -0.4, 0.2]],
            ]
        )

    def enable_route_attribution(self):
        self.observer.close()
        self.observer = self.observer_class(
            [self.model], observer_config(), diagnostics=True, route_attribution=True
        )

    def test_v15_router_grad_norm_rejects_frozen_router(self):
        self.observer.close()
        self.observer = self.observer_class(
            [self.model], observer_config(), version="1.5"
        )
        for router in self.model.routers:
            router.weight.main_grad = torch.ones_like(router.weight)
        expected = (48 * 128 * 4) ** 0.5
        self.assertAlmostEqual(self.observer.v15_router_grad_norm(), expected, places=5)
        for router in self.model.routers:
            router.weight.main_grad.zero_()
        self.assertEqual(self.observer.v15_router_grad_norm(), 0.0)
        with self.assertRaisesRegex(RuntimeError, "smoke requires a nonzero"):
            self.observer.v15_router_grad_norm(require_nonzero=True)

    def tearDown(self):
        self.observer.close()
        self.all_reduce_patcher.stop()
        for name, module in self.saved_modules.items():
            if module is MISSING:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module

    @staticmethod
    def _simulate_identical_dp_gradients(tensor, op=None, group=None):
        if group == "dp":
            tensor.mul_(4)

    def run_prepass(self, training=True):
        self.observer.start_prepass_batch()
        self.observer.begin_prepass_microbatch()
        previous_mode = self.model.training
        self.model.train(training)
        with torch.no_grad():
            self.model(self.hidden.reshape(-1, 4))
        self.model.train(previous_mode)
        self.observer.finish_prepass_microbatch(
            self.input_ids, self.attention_mask, self.sample_ids, response_length=2
        )
        self.observer.finish_prepass_batch()

    def set_route_behavior(self, route_shift=0, routing_map_shift=0):
        for router in self.model.routers:
            router.route_shift = route_shift
            router.routing_map_shift = routing_map_shift

    def run_main_backward(
        self,
        forward_shift=0,
        recompute_shift=0,
        recompute_map_shift=0,
        swap_cache=False,
        prepass_training=True,
    ):
        self.run_prepass(training=prepass_training)
        self.observer.start_main_batch(
            self.sample_ids,
            torch.tensor([4, 9], dtype=torch.int64),
            self.attention_mask,
            self.response_mask,
            2,
            0,
        )
        self.assertFalse(self.observer._utility_sum[:, 0].is_contiguous())
        expert_weight = torch.arange(128, dtype=torch.float32)
        for row, sample_id in enumerate(self.sample_ids):
            ids = sample_id.reshape(1)
            input_ids = self.input_ids[row : row + 1]
            attention_mask = self.attention_mask[row : row + 1]
            response_mask = self.response_mask[row : row + 1]
            hidden = self.hidden[row]
            self.observer.begin_main_microbatch(input_ids, attention_mask, ids, response_mask, 2)
            self.set_route_behavior(route_shift=forward_shift)
            with torch.no_grad():
                self.model(hidden)
            f_routes = self.observer.finish_main_microbatch(input_ids, attention_mask, ids, 2)
            if swap_cache:
                self.observer.route_cache[int(sample_id)] = (
                    self.observer.route_cache[int(sample_id)].long() + 1
                ).remainder(128).to(torch.uint8)
            self.observer.record_main_logprobs(ids, f_routes, torch.zeros((1, 2)))
            self.set_route_behavior(route_shift=recompute_shift, routing_map_shift=recompute_map_shift)
            outputs = self.model(hidden)
            sum((probabilities * expert_weight).sum() for probabilities, _ in outputs).backward()
        self.set_route_behavior()
        for router in self.model.routers:
            router.weight.main_grad = router.weight.grad.detach().clone()
        self.observer.wrap_native_finalize(lambda: None)()

    def finish_main(self):
        utility_sum, utility_sum_sq, utility_count, metrics = self.observer.finish_main_batch()
        self.assertEqual(metrics["gradient_hook_count"], 96)
        self.assertEqual(metrics["expected_gradient_hook_count"], 96)
        self.assertEqual(metrics["forward_recompute_mismatch_count"], 0)
        self.assertEqual(metrics["route_mismatch_count"], 0)
        self.assertEqual(metrics["first_route_mismatch"], {})
        self.assertEqual(utility_count.sum((0, 2)).tolist(), [24.0] * 48)
        self.assertTrue(torch.isfinite(utility_sum).all())
        self.assertTrue(torch.isfinite(utility_sum_sq).all())
        return utility_sum, utility_count, metrics

    def run_auxiliary(self, utility_count):
        stats = SimpleNamespace(count=utility_count, mask=utility_count > 0)
        normalized = (utility_count > 0).float()
        total_active = self.response_mask.sum(-1).float()
        self.observer.start_aux_batch(self.sample_ids, stats, normalized, total_active, 0.05)
        for row, sample_id in enumerate(self.sample_ids):
            ids = sample_id.reshape(1)
            self.observer.begin_aux_microbatch(
                self.input_ids[row : row + 1],
                self.attention_mask[row : row + 1],
                ids,
                self.response_mask[row : row + 1],
                2,
            )
            with torch.no_grad():
                self.model(self.hidden[row].reshape(3, 1, 4))
            self.observer.finish_aux_microbatch()
        return self.observer.finish_aux_batch()

    def test_invalid_alpha_flag_is_sticky_and_preserves_buffer_identity(self):
        self.observer.start_main_batch(
            self.sample_ids,
            torch.tensor([4, 9], dtype=torch.int64),
            self.attention_mask,
            self.response_mask,
            2,
            0,
        )
        flag = self.observer._invalid_flag
        pointer = flag.data_ptr()
        alpha = torch.full((2, 8), 0.125)
        self.observer._mark_invalid_alpha(alpha, alpha.clone())
        self.assertEqual(flag.item(), 0)
        invalid = alpha.clone()
        invalid[0, 0] = 0.0
        self.observer._mark_invalid_alpha(invalid, invalid)
        self.assertEqual(flag.item(), 1)
        self.observer._mark_invalid_alpha(alpha, alpha.clone())
        self.assertEqual(flag.item(), 1)
        self.observer._mark_invalid_alpha(invalid, invalid)
        self.assertEqual(flag.item(), 1)
        self.assertIs(self.observer._invalid_flag, flag)
        self.assertEqual(flag.data_ptr(), pointer)
        self.assertEqual(flag.shape, torch.Size([]))
        self.assertEqual(flag.dtype, torch.int32)
        self.assertEqual(flag.device.type, "cpu")
        self.assertFalse(flag.requires_grad)

    def test_full_prepass_main_recompute_auxiliary_and_reset_lifecycle(self):
        self.run_main_backward()
        self.assertFalse(self.observer._pending_alpha)
        self.assertTrue(all(not queue for queue in self.observer._pending_recompute))
        _, utility_count, metrics = self.finish_main()
        self.assertEqual(metrics["route_mismatch_count"], 0)
        self.assertTrue(metrics["route_phase_execution"]["P"]["model_training"])
        self.assertFalse(metrics["route_phase_execution"]["P"]["grad_enabled"])
        self.assertTrue(metrics["route_phase_execution"]["F"]["model_training"])
        self.assertFalse(metrics["route_phase_execution"]["F"]["grad_enabled"])
        self.assertTrue(metrics["route_phase_execution"]["R"]["model_training"])
        self.assertTrue(metrics["route_phase_execution"]["R"]["grad_enabled"])
        self.assertGreater(metrics["alpha_min"], 0.0)
        auxiliary_metrics = self.run_auxiliary(utility_count)
        self.assertTrue(torch.isfinite(torch.tensor(auxiliary_metrics["utility_objective"])))
        self.assertIsNone(self.observer.mode)
        self.assertFalse(self.observer.route_cache)
        self.assertIsNone(self.observer._invalid_flag)
        self.observer.start_prepass_batch()
        self.assertEqual(self.observer.mode, "prepass")
        self.assertFalse(self.observer.route_cache)
        self.assertTrue(all(not queue for queue in self.observer._pending_recompute))

    def test_hook_count_mismatch_still_fails_fast(self):
        self.run_main_backward()
        self.observer._main_hook_count -= 1
        with self.assertRaisesRegex(RuntimeError, "hook count"):
            self.observer.finish_main_batch()

    def test_native_finalize_wrapper_marks_only_after_success_and_rejects_second_call(self):
        self.observer._native_finalize_count = 0
        self.observer._native_finalize_completed = False
        calls = []
        wrapped = self.observer.wrap_native_finalize(lambda: calls.append("done"))
        wrapped()
        self.assertEqual(calls, ["done"])
        self.assertEqual(self.observer._native_finalize_count, 1)
        self.assertTrue(self.observer._native_finalize_completed)
        with self.assertRaisesRegex(RuntimeError, "more than once"):
            wrapped()

    def test_native_finalize_failure_is_not_marked_complete(self):
        self.observer._native_finalize_count = 0
        self.observer._native_finalize_completed = False

        def fail():
            raise ValueError("native failure")

        with self.assertRaisesRegex(ValueError, "native failure"):
            self.observer.wrap_native_finalize(fail)()
        self.assertEqual(self.observer._native_finalize_count, 0)
        self.assertFalse(self.observer._native_finalize_completed)

    def test_optimizer_generation_and_parameter_version_are_frozen_until_step_e(self):
        self.observer.start_main_batch(
            self.sample_ids,
            torch.tensor([4, 9], dtype=torch.int64),
            self.attention_mask,
            self.response_mask,
            2,
            7,
        )
        with self.assertRaisesRegex(RuntimeError, "optimizer generation"):
            self.observer._assert_parameters_unchanged(8)
        with torch.no_grad():
            self.model.routers[0].weight.add_(0)
        with self.assertRaisesRegex(RuntimeError, "Parameter version"):
            self.observer._assert_parameters_unchanged(7)

    def test_edge_count_mismatch_still_fails_fast(self):
        self.run_main_backward()
        self.observer._edge_count_by_layer[0] += 1
        with self.assertRaisesRegex(RuntimeError, "routing-utility edges"):
            self.observer.finish_main_batch()

    def test_prepass_divergence_is_not_an_eu_semantic_failure(self):
        self.run_main_backward(forward_shift=1, recompute_shift=1)
        _, _, metrics = self.finish_main()
        self.assertEqual(metrics["forward_recompute_mismatch_count"], 0)

    def test_eval_mode_prepass_divergence_is_not_an_eu_semantic_failure(self):
        for router in self.model.routers:
            router.eval_route_shift = 1
        self.run_main_backward(prepass_training=False)
        _, _, metrics = self.finish_main()
        self.assertEqual(metrics["forward_recompute_mismatch_count"], 0)

    def test_forward_recompute_divergence_is_attributed_separately(self):
        self.run_main_backward(recompute_shift=1)
        with self.assertRaises(RuntimeError) as caught:
            self.observer.finish_main_batch()
        message = str(caught.exception)
        self.assertNotIn("'forward_recompute_mismatch_count': 0", message)
        self.assertNotIn("'route_mismatch_count': 0", message)

    def test_sample_mapping_corruption_still_fails_fast(self):
        with self.assertRaisesRegex(RuntimeError, "same-F route snapshot"):
            self.run_main_backward(swap_cache=True)

    def test_actual_f_cache_has_all_global_layers_and_expert_ids(self):
        self.run_main_backward()
        routes = self.observer.routes_for(self.sample_ids)
        self.assertEqual(routes.shape, (2, 2, 48, 8))
        self.assertGreaterEqual(routes.min().item(), 0)
        self.assertLessEqual(routes.max().item(), 127)
        self.assertEqual(self.observer.current_logprobs_for(self.sample_ids).shape, (2, 2))

    def test_v121_router_only_step_e_uses_actual_f_cache_without_new_routing(self):
        self.run_main_backward()
        _, utility_count, _ = self.finish_main()
        stats = SimpleNamespace(count=utility_count, mask=utility_count > 0)
        normalized = (utility_count > 0).float()
        total_active = self.response_mask.sum(-1).float()
        before = [router.weight.main_grad.clone() for router in self.model.routers]
        original_routing = [router.routing for router in self.model.routers]
        for router in self.model.routers:
            router.routing = mock.Mock(side_effect=AssertionError("routing invoked"))
        with mock.patch.object(self.model, "forward", side_effect=AssertionError("full A invoked")), mock.patch(
            "torch.topk", side_effect=AssertionError("new Top-K invoked")
        ), mock.patch.object(torch.Tensor, "topk", side_effect=AssertionError("Tensor.topk invoked")):
            try:
                metrics = self.observer.run_router_only_step_e(
                    self.sample_ids, stats, normalized, total_active, 0.05, 0
                )
            finally:
                for router, routing in zip(self.model.routers, original_routing):
                    router.routing = routing
        self.assertEqual(metrics["full_auxiliary_transformer_forward_count"], 0)
        self.assertEqual(metrics["step_e_natural_topk_call_count"], 0)
        self.assertEqual(metrics["aux_reduce_count_by_layer"], [1] * 48)
        self.assertEqual(metrics["main_grad_add_count_by_layer"], [1] * 48)
        self.assertTrue(metrics["all_cache_layers_consumed"])
        self.assertTrue(
            all(not torch.equal(old, router.weight.main_grad) for old, router in zip(before, self.model.routers))
        )
        self.observer.validate_before_optimizer_step(0)
        cleanup = self.observer.finish_optimizer_step(1)
        self.assertEqual(cleanup, {"cache_clear_count": 1, "orphan_cache_count": 0})
        self.assertIsNone(self.observer._hidden_cache)

    def test_saved_tensor_and_routing_map_must_match_within_invocation(self):
        with self.assertRaisesRegex(RuntimeError, "saved-tensor Top-K differs from routing-map Top-K"):
            self.run_main_backward(recompute_map_shift=1)

    def test_dp4_auxiliary_gradient_is_sum_divided_by_four_after_tp(self):
        local_gradients = [torch.tensor([value, value + 1.0]) for value in range(4)]
        global_sum = torch.stack(local_gradients).sum(0)
        calls = []

        def all_reduce(tensor, op=None, group=None):
            calls.append((group, op))
            if group == "dp":
                tensor.copy_(global_sum)

        self.parallel_state.get_tensor_model_parallel_world_size = lambda: 2
        with mock.patch("torch.distributed.all_reduce", side_effect=all_reduce):
            reduced = []
            for gradient in local_gradients:
                value = gradient.clone()
                self.observer_module._reduce_router_auxiliary_grad(value)
                reduced.append(value)

        for value in reduced:
            torch.testing.assert_close(value, global_sum / 4)
        self.assertEqual(calls, [("tp", None), ("dp", torch.distributed.ReduceOp.SUM)] * 4)

    def test_dp1_auxiliary_gradient_has_no_collective(self):
        self.parallel_state.get_tensor_model_parallel_world_size = lambda: 1
        self.parallel_state.get_data_parallel_world_size = lambda: 1
        gradient = torch.tensor([1.0, 2.0])
        with mock.patch("torch.distributed.all_reduce") as all_reduce:
            self.observer_module._reduce_router_auxiliary_grad(gradient)
        all_reduce.assert_not_called()
        torch.testing.assert_close(gradient, torch.tensor([1.0, 2.0]))

    def test_main_grad_adds_each_averaged_auxiliary_gradient_once(self):
        self.run_main_backward()
        _, utility_count, _ = self.finish_main()
        before = self.model.routers[0].weight.main_grad.clone()
        reduced = []
        original = self.observer_module._reduce_router_auxiliary_grad

        def capture(gradient):
            original(gradient)
            reduced.append(gradient.clone())

        with mock.patch.object(self.observer_module, "_reduce_router_auxiliary_grad", side_effect=capture):
            self.run_auxiliary(utility_count)

        self.assertEqual(len(reduced), 96)
        torch.testing.assert_close(
            self.model.routers[0].weight.main_grad - before,
            reduced[0] + reduced[48],
        )

    def test_mcore_3d_gating_logits_are_flattened_without_reselecting_routes(self):
        for sequence, batch in ((2, 3), (3, 1), (1, 3)):
            with self.subTest(sequence=sequence, batch=batch):
                hidden = torch.randn(sequence, batch, 4)
                weight = torch.nn.Parameter(torch.randn(16, 4))
                raw_logits = torch.nn.functional.linear(hidden, weight)
                flat = raw_logits.view(-1, 16)
                selected = flat.topk(3, dim=-1).indices
                selected_probs = flat.gather(-1, selected).float().softmax(-1)
                probs = torch.zeros_like(flat).scatter(-1, selected, selected_probs)
                routing_map = torch.zeros_like(flat, dtype=torch.bool).scatter(-1, selected, True)
                actual = self.observer_class._selected(routing_map, 3)

                if sequence > 1 and batch > 1:
                    with self.assertRaisesRegex(RuntimeError, "same number of dimensions"):
                        raw_logits.gather(-1, actual)

                logits, actual = self.observer_module._canonicalize_router_auxiliary_inputs(
                    hidden, raw_logits, probs, routing_map, actual, 3, 16
                )
                selected_logits = logits.float().gather(-1, actual)
                log_alpha = selected_logits.log_softmax(-1)
                actual_alpha = probs.gather(-1, actual)
                self.assertEqual(logits.shape, (sequence * batch, 16))
                self.assertEqual(actual.shape, (sequence * batch, 3))
                self.assertEqual(selected_logits.shape, (sequence * batch, 3))
                self.assertEqual(log_alpha.shape, (sequence * batch, 3))
                self.assertEqual(actual_alpha.shape, (sequence * batch, 3))
                torch.testing.assert_close(log_alpha.exp(), actual_alpha)
                torch.testing.assert_close(log_alpha.exp().sum(-1), torch.ones(sequence * batch))
                objective = -(torch.ones_like(log_alpha) * log_alpha).sum()
                auxiliary_grad = torch.autograd.grad(objective, weight)[0]
                self.assertTrue(torch.isfinite(objective))
                self.assertTrue(torch.isfinite(auxiliary_grad).all())

    def test_auxiliary_shape_contract_reports_all_shapes(self):
        hidden = torch.randn(2, 3, 4)
        raw_logits = torch.randn(2, 3, 16)
        probs = torch.randn(5, 16)
        routing_map = torch.zeros(6, 16, dtype=torch.bool)
        actual = torch.zeros(6, 3, dtype=torch.long)
        with self.assertRaisesRegex(RuntimeError, "hidden=.*raw_logits=.*flattened_logits=.*probs=.*routing_map=.*actual=.*topk=3, num_experts=16"):
            self.observer_module._canonicalize_router_auxiliary_inputs(
                hidden, raw_logits, probs, routing_map, actual, 3, 16
            )

    def test_auxiliary_route_mismatch_still_fails_fast(self):
        self.run_main_backward()
        _, utility_count, _ = self.finish_main()
        self.set_route_behavior(route_shift=1)
        try:
            with self.assertRaisesRegex(RuntimeError, "auxiliary natural route differs"):
                self.run_auxiliary(utility_count)
        finally:
            self.set_route_behavior()

    def test_route_attribution_classifies_equal_order_only_and_set_mismatch(self):
        compare = self.observer_module._route_attribution_comparison
        equal = compare(torch.tensor([[1, 2, 3, 4]]), torch.tensor([[1, 2, 3, 4]]))
        self.assertTrue(equal["ordered_equal"].item())
        self.assertTrue(equal["set_equal"].item())
        self.assertFalse(equal["order_only"].item())
        self.assertFalse(equal["set_mismatch"].item())

        order_only = compare(torch.tensor([[1, 2, 3, 4]]), torch.tensor([[2, 1, 3, 4]]))
        self.assertFalse(order_only["ordered_equal"].item())
        self.assertTrue(order_only["set_equal"].item())
        self.assertTrue(order_only["order_only"].item())

        mismatch = compare(torch.tensor([[1, 2, 3, 4]]), torch.tensor([[1, 2, 3, 5]]))
        self.assertFalse(mismatch["set_equal"].item())
        self.assertEqual(mismatch["intersection"].item(), 3)
        self.assertTrue(mismatch["set_mismatch"].item())

    def test_route_attribution_detects_semantic_token_packing_permutation(self):
        routes = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
        # sample, group, physical row, response token, original token, valid, packed ordinal
        f_metadata = torch.tensor([[17, 4, 0, 10, 20, 1, 30], [17, 4, 1, 11, 21, 1, 31]])
        a_metadata = f_metadata.flip(0).clone()
        a_metadata[:, 2] = torch.tensor([0, 1])
        result = self.observer_module._route_attribution_comparison(
            routes, routes, f_metadata, a_metadata
        )
        self.assertTrue(result["set_equal"].all())
        self.assertFalse(result["semantic_equal"].any())

    def test_route_attribution_summary_finds_first_divergence(self):
        summary = self.observer_module._route_attribution_summary(
            torch.tensor([3, 3, 3]), torch.tensor([0, 0, 1]), torch.tensor([0, 1, 2])
        )
        self.assertEqual(summary["first_order_only_layer"], 2)
        self.assertEqual(summary["first_set_mismatch_layer"], 1)
        self.assertEqual(summary["set_mismatch_by_layer"], [0, 1, 2])

    def test_route_attribution_is_disabled_by_default(self):
        self.run_main_backward()
        self.finish_main()
        self.assertFalse(self.observer.route_attribution)
        self.assertFalse(self.observer._ordered_route_cache)
        self.assertIsNone(self.observer._aux_attribution_counts)

    def test_route_attribution_reports_detail_and_still_fails_fast(self):
        self.enable_route_attribution()
        self.run_main_backward()
        _, utility_count, _ = self.finish_main()
        self.set_route_behavior(route_shift=1)
        try:
            with self.assertRaisesRegex(RuntimeError, "GENUINE_SET_MISMATCH"):
                self.run_auxiliary(utility_count)
        finally:
            self.set_route_behavior()
        detail = self.observer._aux_first_set_mismatch
        self.assertEqual(detail["layer_index"], 0)
        self.assertEqual(detail["classification"], "GENUINE_SET_MISMATCH")
        self.assertLess(detail["intersection_count"], 8)
        self.assertEqual(len(detail["A_top_ids"]), 12)
        self.assertEqual(detail["response_token_F"], detail["response_token_A"])
        self.assertEqual(detail["original_token_position_F"], detail["original_token_position_A"])


if __name__ == "__main__":
    unittest.main()
