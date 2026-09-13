from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace

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
        return records.reshape(input_ids.shape[0] * response_length, *records.shape[2:])


class TopKRouter(torch.nn.Module):
    def __init__(self, seed):
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.weight = torch.nn.Parameter(torch.randn(128, 4, generator=generator) * 0.1)
        self.topk = 8

    def gating(self, hidden):
        return torch.nn.functional.linear(hidden, self.weight)

    def routing(self, logits):
        routes = logits.topk(self.topk, dim=-1).indices
        selected_logits = logits.gather(-1, routes)
        alpha = selected_logits.float().softmax(-1)
        probabilities = torch.zeros_like(logits, dtype=alpha.dtype).scatter(-1, routes, alpha)
        routing_map = torch.zeros_like(logits, dtype=torch.bool).scatter(-1, routes, True)
        return probabilities, routing_map

    def forward(self, hidden):
        return self.routing(self.gating(hidden))


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.routers = torch.nn.ModuleList([TopKRouter(index) for index in range(12)])

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
        pipeline_model_parallel_size=4,
        expert_model_parallel_size=2,
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
        parallel_state.get_data_parallel_world_size = lambda: 1
        parallel_state.get_expert_data_parallel_world_size = lambda: 1
        parallel_state.get_tensor_model_parallel_world_size = lambda: 1
        parallel_state.get_pipeline_model_parallel_world_size = lambda: 1
        parallel_state.get_pipeline_model_parallel_rank = lambda: 0
        parallel_state.get_tensor_model_parallel_group = lambda: None
        parallel_state.get_pipeline_model_parallel_group = lambda: None

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

    def tearDown(self):
        self.observer.close()
        for name, module in self.saved_modules.items():
            if module is MISSING:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module

    def run_prepass(self):
        self.observer.start_prepass_batch()
        self.observer.begin_prepass_microbatch()
        with torch.no_grad():
            self.model(self.hidden.reshape(-1, 4))
        self.observer.finish_prepass_microbatch(
            self.input_ids, self.attention_mask, self.sample_ids, response_length=2
        )
        self.observer.finish_prepass_batch()

    def run_main_backward(self):
        self.run_prepass()
        self.observer.start_main_batch(self.sample_ids)
        self.assertFalse(self.observer._utility_sum[:, 0].is_contiguous())
        expert_weight = torch.arange(128, dtype=torch.float32)
        for row, sample_id in enumerate(self.sample_ids):
            ids = sample_id.reshape(1)
            input_ids = self.input_ids[row : row + 1]
            attention_mask = self.attention_mask[row : row + 1]
            response_mask = self.response_mask[row : row + 1]
            hidden = self.hidden[row, :2]
            self.observer.begin_main_microbatch(input_ids, attention_mask, ids, response_mask, 2)
            with torch.no_grad():
                self.model(hidden)
            self.observer.finish_main_microbatch()
            outputs = self.model(hidden)
            sum((probabilities * expert_weight).sum() for probabilities, _ in outputs).backward()
        for router in self.model.routers:
            router.weight.main_grad = router.weight.grad.detach().clone()

    def finish_main(self):
        utility_sum, utility_sum_sq, utility_count, metrics = self.observer.finish_main_batch()
        self.assertEqual(metrics["gradient_hook_count"], 24)
        self.assertEqual(metrics["expected_gradient_hook_count"], 24)
        self.assertEqual(utility_count.sum((0, 2)).tolist(), [24.0] * 12)
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
                self.model(self.hidden[row, :2])
            self.observer.finish_aux_microbatch()
        return self.observer.finish_aux_batch()

    def test_invalid_alpha_flag_is_sticky_and_preserves_buffer_identity(self):
        self.observer.start_main_batch(self.sample_ids)
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

    def test_edge_count_mismatch_still_fails_fast(self):
        self.run_main_backward()
        self.observer._edge_count_by_layer[0] += 1
        with self.assertRaisesRegex(RuntimeError, "routing-utility edges"):
            self.observer.finish_main_batch()


if __name__ == "__main__":
    unittest.main()
