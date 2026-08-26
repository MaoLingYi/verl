import ast
import importlib.util
import math
import random
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch


class _DataProto:
    def __init__(self, batch):
        self.batch = batch


class TopKRouter(torch.nn.Module):
    """MCore 0.16-shaped router: gating is a bound method, not a child module."""

    def __init__(self, num_experts=4, topk=2):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.arange(num_experts * 3, dtype=torch.float32).reshape(num_experts, 3) / (num_experts * 3)
        )
        self.topk = topk

    def gating(self, inputs):
        output = torch.nn.functional.linear(inputs, self.weight)
        self.last_gating_output = output
        return output

    def forward(self, inputs):
        logits = self.gating(inputs)
        replay = getattr(self, "router_replay", None)
        if replay is not None and replay.router_replay_action.value == "replay_forward":
            indices = replay.target_topk_idx
        else:
            indices = torch.topk(logits, self.topk, dim=-1).indices
        routing_map = torch.zeros_like(logits, dtype=torch.bool).scatter(-1, indices, True)
        return torch.softmax(logits, dim=-1), routing_map


def _router_config(**overrides):
    values = {
        "virtual_pipeline_model_parallel_size": None,
        "tensor_model_parallel_size": 1,
        "sequence_parallel": False,
        "moe_router_score_function": "softmax",
        "moe_router_pre_softmax": True,
    }
    values.update(overrides)
    return types.SimpleNamespace(**values)


module_path = Path(__file__).parents[3] / "verl" / "utils" / "debug" / "metrics.py"


def _load_metrics_module():
    protocol = types.ModuleType("verl.protocol")
    protocol.DataProto = _DataProto
    modules = {"verl": types.ModuleType("verl"), "verl.protocol": protocol}
    spec = importlib.util.spec_from_file_location("tim_metrics_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, modules):
        spec.loader.exec_module(module)
    return module


metrics_module = _load_metrics_module()
router_module_path = module_path.with_name("router_shift.py")
router_spec = importlib.util.spec_from_file_location("router_shift_under_test", router_module_path)
router_module = importlib.util.module_from_spec(router_spec)
router_spec.loader.exec_module(router_module)


def _parallel_modules(pp_size=1, dp_size=1):
    state = types.ModuleType("megatron.core.parallel_state")
    state.get_pipeline_model_parallel_world_size = lambda: pp_size
    state.get_data_parallel_world_size = lambda: dp_size
    state.get_pipeline_model_parallel_group = lambda: "pp"
    state.get_data_parallel_group = lambda: "dp"
    core = types.ModuleType("megatron.core")
    core.parallel_state = state
    return {"megatron": types.ModuleType("megatron"), "megatron.core": core, "megatron.core.parallel_state": state}


def _legacy_router_shift(stage_deltas, layer_counts, topk, masks):
    """Test-only reference: separate elementwise PP SUMs followed by per-microbatch metrics."""
    global_deltas, gammas = [], []
    totals = torch.zeros(3)
    for mb, mask in enumerate(masks):
        delta = sum(stage[mb] for stage in stage_deltas)
        gamma = torch.exp(-delta / (sum(layer_counts) * topk))
        valid = gamma[mask]
        totals += torch.tensor([valid.sum().item(), (valid < 0.8).sum().item(), valid.numel()])
        global_deltas.append(delta)
        gammas.append(gamma)
    return global_deltas, gammas, totals


class TestTimMetrics(unittest.TestCase):
    def test_module_loading_does_not_pollute_sys_modules(self):
        before = {name: sys.modules.get(name) for name in ("verl", "verl.protocol")}

        _load_metrics_module()

        for name, previous in before.items():
            if previous is None:
                self.assertNotIn(name, sys.modules)
            else:
                self.assertIs(sys.modules[name], previous)

    def test_uses_response_mask_and_detaches(self):
        rollout_log_probs = torch.full((1, 3), -3.0, requires_grad=True)
        actor_log_probs = torch.tensor(
            [[-3.0 + math.log(4.0), -3.0 - math.log(4.0), -1.0]], requires_grad=True
        )
        data = _DataProto(
            {
                "rollout_log_probs": rollout_log_probs,
                "old_log_probs": actor_log_probs,
                "response_mask": torch.tensor([[1, 1, 0]]),
                "responses": torch.zeros((1, 3)),
            }
        )

        metrics = metrics_module.calculate_debug_metrics(data)

        self.assertAlmostEqual(metrics["diag/tim/kl_k3"], 1.125, places=6)
        self.assertAlmostEqual(metrics["diag/tim/logprob_abs_mean"], math.log(4.0), places=6)
        self.assertEqual(metrics["diag/tim/extreme_frac_tau2"], 1.0)
        self.assertIsNone(rollout_log_probs.grad)
        self.assertIsNone(actor_log_probs.grad)

    def test_bfloat16_inputs_use_finite_fp32_diagnostics(self):
        rollout = torch.tensor([[-3.0, -3.0, -3.0]], dtype=torch.bfloat16, requires_grad=True)
        actor = torch.tensor(
            [[-3.0 + math.log(4.0), -3.0 - math.log(4.0), -1.0]],
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        data = _DataProto(
            {
                "rollout_log_probs": rollout,
                "old_log_probs": actor,
                "response_mask": torch.tensor([[1, 1, 0]]),
                "responses": torch.zeros((1, 3)),
            }
        )

        metrics = metrics_module.calculate_debug_metrics(data)

        for name in ("diag/tim/kl_k3", "diag/tim/logprob_abs_mean", "diag/tim/extreme_frac_tau2"):
            self.assertTrue(math.isfinite(metrics[name]))
        self.assertAlmostEqual(metrics["diag/tim/kl_k3"], 1.125, delta=0.02)
        self.assertAlmostEqual(metrics["diag/tim/logprob_abs_mean"], math.log(4.0), delta=0.01)
        self.assertIsNone(rollout.grad)
        self.assertIsNone(actor.grad)

    def test_loss_mask_fallback(self):
        data = _DataProto(
            {
                "rollout_log_probs": torch.tensor([[-3.0, -3.0]]),
                "old_log_probs": torch.tensor([[-3.0 + math.log(4.0), -1.0]]),
                "loss_mask": torch.tensor([[1, 0]]),
                "responses": torch.zeros((1, 2)),
            }
        )

        metrics = metrics_module.calculate_debug_metrics(data)

        self.assertAlmostEqual(metrics["diag/tim/logprob_abs_mean"], math.log(4.0), places=6)

    def test_attention_mask_is_sliced_to_response(self):
        data = _DataProto(
            {
                "rollout_log_probs": torch.tensor([[-3.0, -3.0]]),
                "old_log_probs": torch.tensor([[-3.0 + math.log(4.0), -1.0]]),
                "attention_mask": torch.tensor([[1, 1, 1, 0]]),
                "responses": torch.zeros((1, 2)),
            }
        )

        metrics = metrics_module.calculate_debug_metrics(data)

        self.assertAlmostEqual(metrics["diag/tim/logprob_abs_mean"], math.log(4.0), places=6)

    def test_no_mask_uses_legacy_all_token_fallback(self):
        data = _DataProto(
            {
                "rollout_log_probs": torch.tensor([[-3.0, -3.0]]),
                "old_log_probs": torch.tensor([[-3.0 + math.log(4.0), -3.0]]),
                "responses": torch.zeros((1, 2)),
            }
        )

        metrics = metrics_module.calculate_debug_metrics(data)

        self.assertAlmostEqual(metrics["diag/tim/logprob_abs_mean"], math.log(4.0) / 2, places=6)

    def test_scatter_selection_is_deterministic_bounded_and_rng_free(self):
        with (
            patch.object(torch, "randperm", side_effect=AssertionError("torch RNG used")),
            patch.object(np.random, "choice", side_effect=AssertionError("NumPy RNG used")),
            patch.object(random, "sample", side_effect=AssertionError("Python RNG used")),
        ):
            for count, expected in ((17, 17), (4096, 4096), (5000, 4096)):
                valid = torch.arange(count)
                first = metrics_module.deterministic_even_indices(valid, 4096)
                second = metrics_module.deterministic_even_indices(valid, 4096)
                self.assertTrue(torch.equal(first, second))
                self.assertEqual(first.numel(), expected)
                self.assertEqual(torch.unique(first).numel(), expected)
                self.assertEqual((first[0].item(), first[-1].item()), (0, count - 1))

    def test_scatter_cadence(self):
        formal = [step for step in range(1, 371) if metrics_module.should_save_tim_scatter(step, 370, 25, True)]
        self.assertEqual(formal, [*range(25, 351, 25), 370])
        for final_step in (1, 8, 20):
            self.assertEqual(
                [step for step in range(1, final_step + 1) if metrics_module.should_save_tim_scatter(step, final_step, 25, True)],
                [final_step],
            )

    def test_scatter_npz_schema_mask_probabilities_and_sequence_ratio(self):
        rollout = torch.tensor([[-1.0, -2.0, -3.0]])
        training = torch.tensor([[-0.5, -2.5, -2.0]])
        mask = torch.tensor([[1, 1, 0]])
        data = _DataProto(
            {
                "rollout_log_probs": rollout,
                "old_log_probs": training,
                "response_mask": mask,
                "responses": torch.zeros((1, 3)),
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            metrics_module.calculate_debug_metrics(
                data,
                scatter_config={
                    "enabled": True,
                    "interval": 25,
                    "max_token_pairs": 4096,
                    "save_final": True,
                    "required": True,
                    "dir": directory,
                },
                global_step=8,
                total_training_steps=8,
                precision="bfloat16",
                r3_enabled=True,
            )
            output = Path(directory) / "s000008.npz"
            self.assertTrue(output.is_file())
            self.assertFalse((Path(directory) / ".tmp000008.npz").exists())
            with np.load(output) as snapshot:
                self.assertEqual(set(snapshot.files), {
                    "token_inference_logprob", "token_training_logprob",
                    "token_inference_prob", "token_training_prob",
                    "token_batch_row", "token_position", "seq_length", "seq_log_ratio",
                    "seq_mean_log_ratio", "global_step", "precision", "r3_enabled", "schema_version",
                })
                np.testing.assert_allclose(snapshot["token_inference_logprob"], [-1.0, -2.0])
                np.testing.assert_allclose(snapshot["token_training_logprob"], [-0.5, -2.5])
                np.testing.assert_allclose(snapshot["token_inference_prob"], np.exp([-1.0, -2.0]))
                np.testing.assert_allclose(snapshot["token_training_prob"], np.exp([-0.5, -2.5]))
                np.testing.assert_array_equal(snapshot["token_batch_row"], [0, 0])
                np.testing.assert_array_equal(snapshot["token_position"], [0, 1])
                np.testing.assert_array_equal(snapshot["seq_length"], [2])
                np.testing.assert_allclose(snapshot["seq_log_ratio"], [0.0])
                np.testing.assert_allclose(snapshot["seq_mean_log_ratio"], [0.0])
                for key in ("token_inference_logprob", "token_training_logprob", "token_inference_prob", "token_training_prob", "seq_log_ratio", "seq_mean_log_ratio"):
                    self.assertEqual(snapshot[key].dtype, np.float32)
                for key in ("token_batch_row", "token_position", "seq_length"):
                    self.assertEqual(snapshot[key].dtype, np.int32)
                self.assertEqual(snapshot["global_step"].dtype, np.int64)
                self.assertEqual(snapshot["precision"].item(), "bfloat16")
                self.assertTrue(snapshot["r3_enabled"].item())
                self.assertEqual(snapshot["schema_version"].item(), 1)

    def test_scatter_fails_on_invalid_required_data_or_path(self):
        config = {
            "enabled": True, "interval": 25, "max_token_pairs": 4096,
            "save_final": True, "required": True, "dir": "missing-tim-directory",
        }
        with self.assertRaises(ValueError):
            metrics_module.save_tim_scatter_sidecar(
                torch.zeros((1, 2)), torch.zeros((1, 3)), torch.ones((1, 2)), config,
                global_step=1, total_training_steps=1, precision="bfloat16", r3_enabled=False,
            )
        with self.assertRaises(ValueError):
            metrics_module.save_tim_scatter_sidecar(
                torch.zeros((1, 2)), torch.zeros((1, 2)), torch.zeros((1, 2)), config,
                global_step=1, total_training_steps=1, precision="bfloat16", r3_enabled=False,
            )
        with self.assertRaises(FileNotFoundError):
            metrics_module.save_tim_scatter_sidecar(
                torch.zeros((1, 2)), torch.zeros((1, 2)), torch.ones((1, 2)), config,
                global_step=1, total_training_steps=1, precision="bfloat16", r3_enabled=False,
            )

    def test_scatter_uses_existing_tensors_without_extra_forward_or_full_cpu_copy(self):
        source = module_path.read_text(encoding="utf-8")
        trainer = (module_path.parents[2] / "trainer/ppo/ray_trainer.py").read_text(encoding="utf-8")
        self.assertNotIn("rollout_log_probs.cpu()", source)
        self.assertNotIn("training_log_probs.cpu()", source)
        self.assertLess(source.index("index_select(0, selected)"), source.index("rollout_selected.cpu()"))
        self.assertEqual(trainer.count("old_log_prob, old_log_prob_mfu = self._compute_old_log_prob(batch)"), 1)


class TestRouterShiftContract(unittest.TestCase):
    def test_baseline_ep2_uses_global_expert_mask(self):
        logits = torch.zeros((1, 128))
        global_indices = torch.tensor([[0, 63, 64, 127]])
        routing_map = torch.zeros_like(logits, dtype=torch.bool).scatter(-1, global_indices, True)

        indices, _ = router_module.RouterShiftObserver.selected_old_router_log_probs(
            logits, routing_map, topk=4
        )

        self.assertEqual(set(indices[0].tolist()), set(global_indices[0].tolist()))

    def test_r3_ep2_hook_uses_exact_global_replay_indices(self):
        router = TopKRouter()
        router.topk = 2
        router.router_replay = types.SimpleNamespace(
            router_replay_action=types.SimpleNamespace(value="replay_forward"),
            target_topk_idx=torch.tensor([[3, 3]]),
        )
        observer = router_module.RouterShiftObserver([router], _router_config())
        observer.start_old_batch()
        observer.begin_old_microbatch()

        _, collapsed_map = router(torch.tensor([[1.0, 2.0, 3.0]]))

        self.assertEqual(collapsed_map.sum().item(), 1)
        self.assertEqual(observer._old_indices[0].tolist(), [[3, 3]])

    def test_ep2_replay_indices_are_global_not_local(self):
        logits = torch.zeros((1, 128))
        global_indices = torch.tensor([[0, 64, 127]], dtype=torch.int64)

        indices, _ = router_module.RouterShiftObserver.selected_old_router_log_probs(
            logits, global_indices, topk=3
        )

        self.assertEqual(indices.tolist(), [[0, 64, 127]])

    def test_streamed_reduction_exposes_per_sample_gamma_and_retains_old_cache(self):
        observer = router_module.RouterShiftObserver([TopKRouter()], _router_config())
        observer.start_current_batch()
        observer.old_cache = {0: (torch.zeros(1), torch.zeros(1))}
        observer._current_microbatches = [
            (torch.tensor([[0.0, 2.0]]), torch.tensor([[True, False]]), torch.tensor([3])),
            (torch.tensor([[1.0, 3.0]]), torch.tensor([[True, True]]), torch.tensor([7])),
        ]
        all_local = torch.cat([item[0] for item in observer._current_microbatches])
        all_mask = torch.cat([item[1] for item in observer._current_microbatches])
        gamma = torch.exp(-all_local / (len(observer.routers) * observer.topk))[all_mask]

        parallel_state = types.ModuleType("megatron.core.parallel_state")
        parallel_state.get_pipeline_model_parallel_world_size = lambda: 1
        parallel_state.get_data_parallel_world_size = lambda: 1
        core = types.ModuleType("megatron.core")
        core.parallel_state = parallel_state
        with patch.dict(
            sys.modules,
            {
                "megatron": types.ModuleType("megatron"),
                "megatron.core": core,
                "megatron.core.parallel_state": parallel_state,
            },
        ):
            totals = observer.finish_current_batch(need_gamma_by_sample=True)

        self.assertAlmostEqual(totals["gamma_sum"], gamma.sum().item(), places=6)
        self.assertEqual(totals["clip_sum"], (gamma < 0.8).float().sum().item())
        self.assertEqual(totals["token_count"], gamma.numel())
        torch.testing.assert_close(totals["gamma_by_sample"][3], torch.exp(-torch.tensor([0.0, 2.0]) / 2))
        torch.testing.assert_close(totals["gamma_by_sample"][7], torch.exp(-torch.tensor([1.0, 3.0]) / 2))
        self.assertEqual(observer._current_microbatches, [])
        self.assertNotEqual(observer.old_cache, {})
        observer.clear_old_cache()
        self.assertEqual(observer.old_cache, {})

    def test_finished_microbatch_releases_gpu_intermediates(self):
        observer = router_module.RouterShiftObserver([TopKRouter()], _router_config())
        observer.start_current_batch()
        observer._current_indices = torch.ones(1)
        observer._current_old_log_probs = torch.ones(1)
        observer._current_abs_sums = [torch.ones(1)]
        observer._current_sample_ids = torch.tensor([5])
        observer._capturing = True
        observer._unpack_sequence_parallel = lambda *_args: torch.arange(3.0).reshape(1, 3, 1, 1)

        with patch.object(torch.Tensor, "cpu", side_effect=AssertionError("microbatch D2H")):
            observer.finish_current_microbatch(
                torch.zeros((1, 3)),
                torch.ones((1, 3), dtype=torch.bool),
                torch.tensor([[True, True]]),
                response_length=2,
            )

        self.assertEqual(observer._current_microbatches[0][0].device.type, "cpu")
        self.assertEqual(observer._current_microbatches[0][1].device.type, "cpu")
        self.assertEqual(observer._current_microbatches[0][2].tolist(), [5])
        self.assertIsNone(observer._current_indices)
        self.assertIsNone(observer._current_old_log_probs)
        self.assertIsNone(observer._current_abs_sums)
        self.assertIsNone(observer._current_sample_ids)

    def test_zero_delta_and_masked_padding(self):
        values = torch.zeros((1, 2, 3, 4), dtype=torch.bfloat16)
        metrics = metrics_module.calculate_router_shift_metrics(values, values, torch.tensor([[1, 0]]))

        self.assertEqual(metrics["diag/router_shift/ratio_mean"], 1.0)
        self.assertEqual(metrics["diag/router_shift/clipfrac_gamma_0_8"], 0.0)

    def test_pure_math_uses_old_experts_and_response_mask(self):
        old = torch.zeros((1, 2, 2, 2), requires_grad=True)
        current = torch.tensor(
            [[[[0.0, 0.0], [math.log(4.0), math.log(4.0)]], [[10.0, 10.0], [10.0, 10.0]]]],
            requires_grad=True,
        )

        metrics = metrics_module.calculate_router_shift_metrics(old, current, torch.tensor([[1, 0]]))

        self.assertAlmostEqual(metrics["diag/router_shift/ratio_mean"], 0.5, places=6)
        self.assertEqual(metrics["diag/router_shift/clipfrac_gamma_0_8"], 1.0)
        self.assertIsNone(old.grad)
        self.assertIsNone(current.grad)

    def test_current_full_logits_cover_old_expert_outside_current_topk(self):
        old_logits = torch.tensor([[5.0, 4.0, 0.0, -1.0]])
        old_map = torch.tensor([[1, 1, 0, 0]], dtype=torch.bool)
        old_indices, old_selected = router_module.RouterShiftObserver.selected_old_router_log_probs(
            old_logits, old_map, topk=2
        )
        current_logits = torch.tensor([[0.0, -1.0, 5.0, 4.0]], dtype=torch.bfloat16)

        diff = router_module.RouterShiftObserver.current_abs_diff_sum(
            current_logits, old_indices, old_selected, invalid_flag=torch.zeros((), dtype=torch.bool)
        )
        reference = (
            torch.log_softmax(current_logits.float(), dim=-1).gather(-1, old_indices) - old_selected
        ).abs().sum(-1)

        self.assertTrue(torch.isfinite(diff).all())
        torch.testing.assert_close(diff, reference)
        gamma = torch.exp(-diff / 2)
        self.assertGreater(gamma.item(), 0.0)
        self.assertLess(gamma.item(), 1.0)
        self.assertEqual(set(old_indices[0].tolist()), {0, 1})

    def test_qwen3_post_softmax_uses_full_router_distribution(self):
        old_logits = torch.tensor([[5.0, 4.0, 0.0, -1.0]])
        old_map = torch.tensor([[1, 1, 0, 0]], dtype=torch.bool)
        old_indices, old_selected = router_module.RouterShiftObserver.selected_old_router_log_probs(
            old_logits, old_map, topk=2, pre_softmax=False
        )
        current_logits = torch.tensor([[0.0, -2.0, 5.0, 4.0]], dtype=torch.bfloat16)

        diff = router_module.RouterShiftObserver.current_abs_diff_sum(
            current_logits, old_indices, old_selected, pre_softmax=False,
            invalid_flag=torch.zeros((), dtype=torch.bool),
        )
        reference = (
            torch.log_softmax(current_logits.float(), dim=-1).gather(-1, old_indices)
            - torch.log_softmax(old_logits.float(), dim=-1).gather(-1, old_indices)
        ).abs().sum(-1)
        gamma = torch.exp(-diff / 2)

        self.assertTrue(torch.isfinite(diff).all())
        torch.testing.assert_close(diff, reference)
        self.assertGreater(gamma.item(), 0.0)
        self.assertLess(gamma.item(), 1.0)

    def test_qwen3_post_softmax_identical_routers_have_gamma_one(self):
        old_logits = torch.tensor([[5.0, 4.0, 0.0, -1.0]])
        routing_map = torch.tensor([[1, 1, 0, 0]], dtype=torch.bool)
        old_indices, old_selected = router_module.RouterShiftObserver.selected_old_router_log_probs(
            old_logits, routing_map, topk=2, pre_softmax=False
        )
        current_logits = old_logits.to(torch.bfloat16)

        diff = router_module.RouterShiftObserver.current_abs_diff_sum(
            current_logits, old_indices, old_selected, pre_softmax=False,
            invalid_flag=torch.zeros((), dtype=torch.bool),
        )
        reference = (
            torch.log_softmax(current_logits.float(), dim=-1).gather(-1, old_indices) - old_selected
        ).abs().sum(-1)

        torch.testing.assert_close(diff, reference)
        torch.testing.assert_close(torch.exp(-diff / 2), torch.ones_like(diff))

    def test_qwen3_post_softmax_observer_is_diagnostic_only(self):
        baseline = TopKRouter()
        observed = TopKRouter()
        observed.load_state_dict(baseline.state_dict())
        observer = router_module.RouterShiftObserver(
            [observed], _router_config(moe_router_pre_softmax=False)
        )
        observer.start_old_batch()
        observer.begin_old_microbatch()
        baseline_input = torch.tensor([[1.0, 2.0, 3.0]], requires_grad=True)
        observed_input = baseline_input.detach().clone().requires_grad_(True)

        baseline_output = baseline(baseline_input)[0]
        observed_output = observed(observed_input)[0]
        baseline_output[:, 0].sum().backward()
        observed_output[:, 0].sum().backward()

        torch.testing.assert_close(observed_output, baseline_output)
        torch.testing.assert_close(observed_input.grad, baseline_input.grad)
        torch.testing.assert_close(observed.weight.grad, baseline.weight.grad)
        self.assertFalse(observer._old_log_probs[0].requires_grad)

    def test_router_data_validation_rejects_invalid_map_range_and_nonfinite_values(self):
        logits = torch.tensor([[1.0, 0.0]])
        with self.assertRaisesRegex(RuntimeError, "top-k routing map"):
            router_module.RouterShiftObserver.selected_old_router_log_probs(
                logits, torch.tensor([[1, 1]], dtype=torch.bool), topk=1
            )
        with self.assertRaisesRegex(RuntimeError, "expert indices"):
            router_module.RouterShiftObserver.selected_old_router_log_probs(
                logits, torch.tensor([[2]]), topk=1
            )
        with self.assertRaisesRegex(RuntimeError, "non-finite"):
            router_module.RouterShiftObserver.selected_old_router_log_probs(
                torch.tensor([[float("nan"), 0.0]]), torch.tensor([[1]]), topk=1
            )

    def test_pp4_aggregates_delta_before_exponentiation(self):
        local = [
            torch.tensor([0.0]),
            torch.tensor([math.log(2.0)]),
            torch.tensor([math.log(4.0)]),
            torch.tensor([math.log(8.0)]),
        ]

        gamma = router_module.RouterShiftObserver.aggregate_partials(local, [1, 1, 1, 1], topk=1)

        expected = torch.exp(-sum(local) / 4)
        wrong = sum(torch.exp(-value) for value in local) / 4
        torch.testing.assert_close(gamma, expected)
        self.assertFalse(torch.allclose(gamma, wrong))

    def test_sample_ids_restore_records_after_minibatch_reordering(self):
        observer = router_module.RouterShiftObserver([TopKRouter(num_experts=8, topk=1)], _router_config())
        observer.start_current_batch()
        observer.old_cache = {
            3: (torch.tensor([[[3]]], dtype=torch.uint8), torch.tensor([[[0.3]]])),
            7: (torch.tensor([[[7]]], dtype=torch.uint8), torch.tensor([[[0.7]]])),
        }
        observer._pack_sequence_parallel = lambda records, *_args: records.squeeze(1)

        observer.begin_current_microbatch(
            torch.zeros((2, 1)),
            torch.ones((2, 1), dtype=torch.bool),
            torch.tensor([7, 3]),
            response_length=1,
        )

        self.assertTrue(torch.equal(observer._current_indices[0, :, 0], torch.tensor([7, 3])))
        torch.testing.assert_close(observer._current_old_log_probs[0, :, 0], torch.tensor([0.7, 0.3]))

    def test_observer_hooks_do_not_change_router_output_or_gradients(self):
        baseline = TopKRouter()
        observed = TopKRouter()
        observed.load_state_dict(baseline.state_dict())
        parameter_count = sum(parameter.numel() for parameter in observed.parameters())
        observer = router_module.RouterShiftObserver([observed], _router_config())
        observer.start_old_batch()
        observer.begin_old_microbatch()
        baseline_input = torch.tensor([[1.0, 2.0, 3.0]], requires_grad=True)
        observed_input = baseline_input.detach().clone().requires_grad_(True)

        baseline_output = baseline(baseline_input)[0]
        observed_output = observed(observed_input)[0]
        baseline_output[:, 0].sum().backward()
        observed_output[:, 0].sum().backward()

        torch.testing.assert_close(observed_output, baseline_output)
        torch.testing.assert_close(observed_input.grad, baseline_input.grad)
        torch.testing.assert_close(observed.weight.grad, baseline.weight.grad)
        self.assertEqual(sum(parameter.numel() for parameter in observed.parameters()), parameter_count)
        self.assertFalse(observer._old_log_probs[0].requires_grad)

    def test_old_cache_preserves_fp32_zero_and_tiny_router_shift(self):
        router = TopKRouter()
        observer = router_module.RouterShiftObserver([router], _router_config())
        observer.start_old_batch()
        observer.begin_old_microbatch()

        router(torch.tensor([[1.0, 2.0, 3.0]]))
        old_logits = router.last_gating_output.detach().clone()
        old_indices = observer._old_indices[0]
        cached_old_log_probs = observer._old_log_probs[0]

        self.assertEqual(old_indices.dtype, torch.uint8)
        self.assertEqual(cached_old_log_probs.dtype, torch.float32)
        zero_diff = observer.current_abs_diff_sum(
            old_logits, old_indices, cached_old_log_probs, invalid_flag=torch.zeros((), dtype=torch.bool)
        )
        torch.testing.assert_close(zero_diff, torch.zeros_like(zero_diff), atol=0, rtol=0)
        torch.testing.assert_close(torch.exp(-zero_diff / router.topk), torch.ones_like(zero_diff))

        current_logits = old_logits.clone()
        current_logits[0, old_indices[0, 0].item()] += 1e-3
        measured = observer.current_abs_diff_sum(
            current_logits, old_indices, cached_old_log_probs, invalid_flag=torch.zeros((), dtype=torch.bool)
        )
        fp32_reference = observer.current_abs_diff_sum(
            current_logits,
            old_indices,
            torch.log_softmax(old_logits.float(), dim=-1).gather(-1, old_indices.long()),
            invalid_flag=torch.zeros((), dtype=torch.bool),
        )
        self.assertGreater(measured.item(), 0)
        torch.testing.assert_close(measured, fp32_reference, atol=1e-7, rtol=1e-5)

    def test_method_gating_wrap_is_instance_scoped_and_not_nested(self):
        observed = TopKRouter()
        untouched = TopKRouter()
        original = observed.gating
        untouched_original = untouched.gating

        observer = router_module.RouterShiftObserver([observed], _router_config())
        wrapped = observed.__dict__["gating"]

        self.assertEqual(observed._verl_router_shift_original_gating, original)
        self.assertNotIn("gating", untouched.__dict__)
        self.assertEqual(untouched.gating, untouched_original)
        returned_logits = observed.gating(torch.ones((1, 3)))
        self.assertIs(returned_logits, observed.last_gating_output)
        with self.assertRaisesRegex(RuntimeError, "already attached"):
            router_module.RouterShiftObserver([observed], _router_config())
        self.assertIs(observed.__dict__["gating"], wrapped)

        observer.close()
        self.assertEqual(observed.gating, original)
        self.assertFalse(hasattr(observed, "_verl_router_shift_original_gating"))

    def test_router_semantics_compatibility_gate(self):
        with self.assertRaisesRegex(ValueError, "score_function='softmax'"):
            router_module.RouterShiftObserver(
                [TopKRouter()], _router_config(moe_router_score_function="sigmoid")
            )
        with self.assertRaisesRegex(ValueError, "boolean moe_router_pre_softmax"):
            router_module.RouterShiftObserver(
                [TopKRouter()], _router_config(moe_router_pre_softmax=None)
            )

    def test_real_mcore_016_gating_api_when_available(self):
        try:
            from megatron.core.transformer.moe.router import TopKRouter as MCoreTopKRouter
        except ImportError:
            self.skipTest("Megatron-Core is not installed")

        self.assertTrue(callable(MCoreTopKRouter.gating))
        self.assertFalse(isinstance(MCoreTopKRouter.gating, torch.nn.Module))

    def test_diagnostics_default_disabled_and_collective_is_post_schedule(self):
        root = module_path.parents[3]
        config_source = (root / "verl" / "workers" / "config" / "actor.py").read_text()
        actor_source = (root / "verl" / "workers" / "actor" / "megatron_actor.py").read_text()
        trainer_source = (root / "verl" / "trainer" / "ppo" / "ray_trainer.py").read_text()

        self.assertIn("enabled: bool = False", config_source)
        self.assertIn("if self.config.router_shift_diagnostics.enabled:", actor_source)
        self.assertIn('torch.arange(\n                                    len(batch)', trainer_source)
        schedule_end = actor_source.index("# loss_reduces contains the stats returned from loss_func")
        aggregate = actor_source.index("self.router_shift_observer.finish_current_batch(")
        self.assertLess(schedule_end, aggregate)
        observer_source = router_module_path.read_text()
        self.assertIn("-response_length - 1 : -1", observer_source)
        self.assertIn("get_pipeline_model_parallel_group()", observer_source)
        self.assertIn("get_data_parallel_group()", observer_source)

    def test_current_dataflow_records_indices_but_not_router_scores(self):
        root = module_path.parents[3]
        replay_source = (root / "verl" / "utils" / "megatron" / "router_replay_patch.py").read_text()
        worker_source = (root / "verl" / "workers" / "megatron_workers.py").read_text()

        self.assertIn("def record_indices", replay_source)
        self.assertNotIn("def record_log_probs", replay_source)
        self.assertIn('self.enable_routing_replay = self.router_replay.mode != "disabled"', worker_source)
        self.assertIn('tensors = {"ref_log_prob": output} if is_lora else {"old_log_probs": output}', worker_source)
        self.assertNotIn('output.batch["old_router_log_probs"]', worker_source)


class TestRouterShiftP0(unittest.TestCase):
    def _observer(self, layers=1, topk=2):
        observer = router_module.RouterShiftObserver(
            [TopKRouter(num_experts=128, topk=topk) for _ in range(layers)], _router_config()
        )
        self.addCleanup(observer.close)
        observer.start_current_batch()
        return observer

    def _run_batched(self, stages, counts, topk, masks, ids, need_gamma):
        observer = self._observer(counts[0], topk)
        observer._current_microbatches = list(zip(stages[0], masks, ids))
        reduced = []

        def pp_sum(payload, group):
            self.assertEqual(group, "pp")
            self.assertFalse(payload.requires_grad)
            for count, stage in zip(counts[1:], stages[1:]):
                payload.add_(torch.cat([torch.tensor([float(count), 0.0]), *[delta.flatten() for delta in stage]]))
            reduced.append(payload.clone())

        with patch.dict(sys.modules, _parallel_modules(pp_size=len(stages))):
            with patch.object(torch.distributed, "all_reduce", side_effect=pp_sum) as collective:
                result = observer.finish_current_batch(need_gamma_by_sample=need_gamma)
        self.assertEqual(collective.call_count, 1)
        self._assert_current_cleared(observer)
        return result, reduced[0][2:]

    def _assert_current_cleared(self, observer):
        self.assertEqual(observer._current_microbatches, [])
        self.assertEqual(observer._current_seen_sample_ids, set())
        self.assertIsNone(observer._current_invalid_flag)
        self.assertIsNone(observer._current_indices)
        self.assertIsNone(observer._current_old_log_probs)
        self.assertIsNone(observer._current_abs_sums)
        self.assertIsNone(observer._current_sample_ids)
        self.assertIsNone(observer.mode)
        self.assertFalse(observer._capturing)

    def test_legacy_vs_batched_pp4_logits_masks_and_noncontiguous_sample_order(self):
        generator = torch.Generator().manual_seed(20260826)
        counts, topk = [1, 2, 3, 2], 8
        ids = [torch.tensor([17, 2]), torch.tensor([91]), torch.tensor([5])]
        masks = [
            torch.tensor([[1, 1, 1, 1, 1, 1, 1], [1, 1, 0, 0, 0, 0, 0]], dtype=torch.bool),
            torch.tensor([[1, 1, 1, 0, 0]], dtype=torch.bool),
            torch.tensor([[0, 0, 0, 0]], dtype=torch.bool),
        ]
        legacy_stages, optimized_stages = [], []
        for count in counts:
            legacy, optimized = [], []
            for mask in masks:
                shape = (count, *mask.shape, 128)
                old = torch.randn(shape, generator=generator)
                current = old + torch.randn(shape, generator=generator) * 0.6
                selected_indices = old.topk(topk, dim=-1).indices
                old_selected = old.log_softmax(-1).gather(-1, selected_indices)
                current_selected = current.log_softmax(-1).gather(-1, selected_indices)
                legacy.append((current_selected - old_selected).abs().sum(-1).sum(0))
                flag = torch.zeros((), dtype=torch.bool)
                per_layer = [
                    router_module.RouterShiftObserver.current_abs_diff_sum(
                        current[layer].reshape(-1, 128).requires_grad_(),
                        selected_indices[layer].reshape(-1, topk),
                        old_selected[layer].reshape(-1, topk).requires_grad_(),
                        invalid_flag=flag,
                    ).reshape(mask.shape)
                    for layer in range(count)
                ]
                self.assertFalse(flag.item())
                self.assertTrue(all(not value.requires_grad for value in per_layer))
                optimized.append(torch.stack(per_layer).sum(0))
            legacy_stages.append(legacy)
            optimized_stages.append(optimized)
        deltas, gammas, totals = _legacy_router_shift(legacy_stages, counts, topk, masks)
        result, actual_delta = self._run_batched(optimized_stages, counts, topk, masks, ids, True)
        torch.testing.assert_close(actual_delta, torch.cat([delta.flatten() for delta in deltas]))
        self.assertEqual(list(result["gamma_by_sample"]), [17, 2, 91, 5])
        for sample_ids, gamma in zip(ids, gammas):
            for sample_id, row in zip(sample_ids.tolist(), gamma):
                torch.testing.assert_close(result["gamma_by_sample"][sample_id], row)
        torch.testing.assert_close(torch.tensor(result["gamma_sum"]), totals[0])
        self.assertEqual(result["clip_sum"], totals[1].item())
        self.assertEqual(result["token_count"], totals[2].item())
        torch.testing.assert_close(torch.tensor(result["gamma_sum"] / result["token_count"]), totals[0] / totals[2])
        torch.testing.assert_close(torch.tensor(result["clip_sum"] / result["token_count"]), totals[1] / totals[2])

    def test_384_separate_reductions_equal_one_batched_reduction_diagnostics_only(self):
        generator = torch.Generator().manual_seed(7)
        stages = [[torch.rand((1, 4), generator=generator) * 100 for _ in range(384)] for _ in range(4)]
        masks = [torch.tensor([[True, True, False, True]]) for _ in range(384)]
        ids = [torch.tensor([i * 7 + 2]) for i in range(384)]
        deltas, _, totals = _legacy_router_shift(stages, [12] * 4, 8, masks)
        host_reads = []
        original_tolist = torch.Tensor.tolist

        def read_totals(tensor):
            host_reads.append(tuple(tensor.shape))
            return original_tolist(tensor)

        with (
            patch.object(router_module.RouterShiftObserver, "_sample_ids", side_effect=AssertionError("finalize IDs")),
            patch.object(torch.Tensor, "item", side_effect=AssertionError("finalize item")),
            patch.object(torch.Tensor, "tolist", read_totals),
            patch.object(torch.Tensor, "cpu", side_effect=AssertionError("diagnostics-only gamma D2H")),
        ):
            result, actual_delta = self._run_batched(stages, [12] * 4, 8, masks, ids, False)
        self.assertEqual(host_reads, [(4,)])
        self.assertEqual(set(result), {"gamma_sum", "clip_sum", "token_count"})
        torch.testing.assert_close(actual_delta, torch.cat([delta.flatten() for delta in deltas]))
        torch.testing.assert_close(torch.tensor(result["gamma_sum"]), totals[0], atol=1e-3, rtol=1e-5)
        self.assertEqual(result["clip_sum"], totals[1].item())
        self.assertEqual(result["token_count"], totals[2].item())

    def test_threshold_point_eight_preserves_strict_comparison(self):
        threshold_delta = -torch.log(torch.tensor(0.8)) * (48 * 8)
        delta = torch.stack([
            threshold_delta - 0.1,
            torch.nextafter(threshold_delta, torch.tensor(float("-inf"))),
            threshold_delta,
            torch.nextafter(threshold_delta, torch.tensor(float("inf"))),
            threshold_delta + 0.1,
        ]).reshape(1, -1)
        stages = [[delta / 4] for _ in range(4)]
        masks, ids = [torch.ones_like(delta, dtype=torch.bool)], [torch.tensor([17])]
        _, legacy_gamma, totals = _legacy_router_shift(stages, [12] * 4, 8, masks)
        result, _ = self._run_batched(stages, [12] * 4, 8, masks, ids, True)
        gamma = result["gamma_by_sample"][17]
        torch.testing.assert_close(gamma, legacy_gamma[0][0])
        self.assertEqual(result["clip_sum"], totals[1].item())
        self.assertFalse((gamma[0] < 0.8).item())
        self.assertTrue((gamma[-1] < 0.8).item())
        # A one-ULP delta perturbation can cross the discontinuous threshold; do not widen it.
        self.assertTrue(torch.equal(gamma < 0.8, legacy_gamma[0][0] < 0.8))

    def test_current_layer_has_no_host_tensor_reads_and_defers_nan_inf(self):
        for bad in (float("nan"), float("inf"), float("-inf")):
            observer = self._observer()
            observer._capturing = True
            observer._current_indices = torch.tensor([[[0, 127]]], dtype=torch.uint8)
            observer._current_old_log_probs = torch.zeros((1, 1, 2))
            observer._current_abs_sums = [None]
            logits = torch.zeros((1, 128), requires_grad=True)
            with torch.no_grad():
                logits[0, 5] = bad
            with (
                patch.object(torch.Tensor, "__bool__", side_effect=AssertionError("host bool")),
                patch.object(torch.Tensor, "item", side_effect=AssertionError("host item")),
                patch.object(torch.Tensor, "cpu", side_effect=AssertionError("D2H")),
                patch.object(torch.Tensor, "tolist", side_effect=AssertionError("host list")),
            ):
                observer._observe_gating(0, logits)
            self.assertTrue(observer._current_invalid_flag.item())
            self.assertFalse(observer._current_invalid_flag.requires_grad)
            self.assertFalse(observer._current_abs_sums[0].requires_grad)
            # Even invalid logits outside response positions must make finalization fail.
            observer._current_microbatches = [(torch.zeros((1, 2)), torch.ones((1, 2), dtype=torch.bool), torch.tensor([17]))]
            with patch.dict(sys.modules, _parallel_modules()):
                with self.assertRaisesRegex(RuntimeError, "non-finite"):
                    observer.finish_current_batch()
            self._assert_current_cleared(observer)
            self.assertEqual(observer.old_cache, {})

    def test_invalid_old_routes_and_cache_values_are_rejected_before_current_pass(self):
        for index in (-1, 128, 255):
            with self.assertRaisesRegex(RuntimeError, "expert indices"):
                router_module.RouterShiftObserver.selected_old_router_log_probs(
                    torch.zeros((1, 128)), torch.tensor([[0, index]]), topk=2
                )
        for bad_indices, bad_probs in ((True, False), (False, True)):
            observer = self._observer()
            observer.start_old_batch()
            observer.begin_old_microbatch()
            indices = torch.zeros((1, 3, 1, 2), dtype=torch.uint8)
            probs = torch.zeros((1, 3, 1, 2))
            if bad_indices:
                indices[0, 0, 0, 0] = 128
            if bad_probs:
                probs[0, 0, 0, 0] = float("nan")
            with patch.object(observer, "_unpack_sequence_parallel", side_effect=[indices, probs]):
                with self.assertRaisesRegex(RuntimeError, "old cache"):
                    observer.finish_old_microbatch(torch.zeros((1, 3)), torch.ones((1, 3)), torch.tensor([17]), 2)
            self.assertEqual(observer.old_cache, {})

    def test_weighting_uses_one_bulk_cpu_copy_and_preserves_mapping(self):
        observer = self._observer()
        observer._current_microbatches = [
            (torch.tensor([[0.0, 1.0], [2.0, 3.0]]), torch.ones((2, 2), dtype=torch.bool), torch.tensor([17, 2])),
            (torch.tensor([[4.0, 5.0]]), torch.ones((1, 2), dtype=torch.bool), torch.tensor([91])),
        ]
        copies = []
        original_cpu = torch.Tensor.cpu

        def record_copy(tensor, *args, **kwargs):
            copies.append(tensor.numel())
            return original_cpu(tensor, *args, **kwargs)

        with patch.dict(sys.modules, _parallel_modules()), patch.object(torch.Tensor, "cpu", record_copy):
            result = observer.finish_current_batch(need_gamma_by_sample=True)
        self.assertEqual(copies, [6])
        self.assertEqual(list(result["gamma_by_sample"]), [17, 2, 91])
        torch.testing.assert_close(torch.stack(list(result["gamma_by_sample"].values())), torch.exp(-torch.arange(6.0).reshape(3, 2) / 2))

    def test_uint8_cache_upper_bound_does_not_wrap_at_256_experts(self):
        observer = router_module.RouterShiftObserver([TopKRouter(num_experts=256)], _router_config())
        self.addCleanup(observer.close)
        indices, selected = observer.selected_old_router_log_probs(
            torch.zeros((1, 256)), torch.tensor([[0, 255]], dtype=torch.uint8), topk=2
        )
        observer.start_old_batch()
        observer.begin_old_microbatch()
        restored_indices = indices.to(torch.uint8).reshape(1, 1, 1, 2).expand(1, 3, 1, 2)
        restored_probs = selected.reshape(1, 1, 1, 2).expand(1, 3, 1, 2)
        with patch.object(observer, "_unpack_sequence_parallel", side_effect=[restored_indices, restored_probs]):
            observer.finish_old_microbatch(torch.zeros((1, 3)), torch.ones((1, 3)), torch.tensor([17]), 2)
        self.assertEqual(observer.old_cache[17][0][0, 0].tolist(), [0, 255])

    def test_remote_invalid_status_reaches_pp_and_dp_before_host_raise(self):
        original_tolist = torch.Tensor.tolist
        for bad_group in ("pp", "dp"):
            observer = self._observer()
            observer._current_microbatches = [
                (torch.zeros((1, 2)), torch.ones((1, 2), dtype=torch.bool), torch.tensor([17]))
            ]
            groups = []

            def reduce(payload, group):
                groups.append(group)
                if group == bad_group:
                    payload[1 if group == "pp" else 3] += 1

            def read_totals(tensor):
                groups.append("host_read")
                self.assertEqual(tuple(tensor.shape), (4,))
                return original_tolist(tensor)

            with patch.dict(sys.modules, _parallel_modules(pp_size=4, dp_size=2)):
                with (
                    patch.object(torch.distributed, "all_reduce", side_effect=reduce),
                    patch.object(torch.Tensor, "item", side_effect=AssertionError("finalize item")),
                    patch.object(torch.Tensor, "tolist", read_totals),
                ):
                    with self.assertRaisesRegex(RuntimeError, "non-finite"):
                        observer.finish_current_batch()
            self.assertEqual(groups, ["pp", "dp", "host_read"])
            self._assert_current_cleared(observer)

    def test_begin_current_microbatch_rejects_intra_and_cross_microbatch_duplicates(self):
        observer = self._observer(topk=1)
        observer.old_cache[17] = (torch.zeros((1, 1, 1), dtype=torch.uint8), torch.zeros((1, 1, 1)))
        observer._pack_sequence_parallel = lambda records, *_args: records.squeeze(1)
        with self.assertRaisesRegex(RuntimeError, "duplicate"):
            observer.begin_current_microbatch(torch.zeros((2, 1)), torch.ones((2, 1)), torch.tensor([17, 17]), 1)
        self.assertEqual(observer._current_seen_sample_ids, set())
        observer.begin_current_microbatch(torch.zeros((1, 1)), torch.ones((1, 1)), torch.tensor([17]), 1)
        with self.assertRaisesRegex(RuntimeError, "duplicate"):
            observer.begin_current_microbatch(torch.zeros((1, 1)), torch.ones((1, 1)), torch.tensor([17]), 1)
        observer.start_current_batch()
        self.assertEqual(observer._current_seen_sample_ids, set())
        observer.begin_current_microbatch(torch.zeros((1, 1)), torch.ones((1, 1)), torch.tensor([17]), 1)

    def test_batch_lifecycle_empty_mask_collective_failure_and_abort_retry(self):
        observer = self._observer()
        for failure in ("empty", "collective", "abort"):
            observer.start_current_batch()
            observer.old_cache[17] = (torch.zeros(1), torch.zeros(1))
            observer._current_microbatches = [(torch.ones((1, 2)), torch.zeros((1, 2), dtype=torch.bool), torch.tensor([17]))]
            if failure == "abort":
                actor = types.SimpleNamespace(router_shift_observer=observer)

                @router_module.clear_router_shift_on_error
                def fail(_actor):
                    raise KeyboardInterrupt("abort")

                with self.assertRaises(KeyboardInterrupt):
                    fail(actor)
            else:
                with patch.dict(sys.modules, _parallel_modules(pp_size=4 if failure == "collective" else 1)):
                    with patch.object(torch.distributed, "all_reduce", side_effect=RuntimeError("collective failure")):
                        with self.assertRaises(RuntimeError):
                            observer.finish_current_batch()
            self._assert_current_cleared(observer)
            self.assertEqual(observer.old_cache, {})
        observer.start_current_batch()
        self.assertFalse(observer._current_invalid_flag.item())
        observer._current_microbatches = [(torch.zeros((1, 1)), torch.ones((1, 1), dtype=torch.bool), torch.tensor([17]))]
        with patch.dict(sys.modules, _parallel_modules()):
            self.assertEqual(observer.finish_current_batch()["gamma_sum"], 1.0)
        self._assert_current_cleared(observer)

    def test_restart_and_close_discard_pending_cuda_state(self):
        observer = self._observer()
        observer._current_microbatches.append((torch.ones(2), torch.ones(2), torch.tensor([17])))
        observer._current_invalid_flag.fill_(True)
        observer._current_indices = torch.ones(2)
        observer.start_current_batch()
        self.assertEqual(observer._current_microbatches, [])
        self.assertIsNone(observer._current_indices)
        self.assertFalse(observer._current_invalid_flag.item())
        observer._pending_old_log_probs[0] = torch.ones(2)
        observer.close()
        self._assert_current_cleared(observer)
        self.assertEqual(observer._pending_old_log_probs, {})

    def test_actor_keeps_cpu_ids_after_reorder_before_h2d(self):
        source = (module_path.parents[2] / "workers/actor/megatron_actor.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        forward = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "forward_step")
        start = next(i for i, node in enumerate(forward.body) if isinstance(node, ast.Assign) and ast.unparse(node.value) == "next(batch_iter)")
        # Execute the actual actor prefix with a fake device transfer, not a copied implementation.
        prefix = ast.Module(body=forward.body[start:start + 4], type_ignores=[])

        class Batch(dict):
            def to(self, device):
                result = Batch(self)
                result["router_shift_sample_ids"] = "unread CUDA metadata"
                return result

            def contiguous(self):
                return self

        ids = torch.tensor([17, 2, 91, 5])
        routes = torch.arange(16).reshape(4, 2, 2)
        observer = self._observer(topk=1)
        observer.old_cache = {i: (torch.tensor([[[i]]], dtype=torch.uint8), torch.tensor([[[i / 100]]])) for i in ids.tolist()}
        observer._pack_sequence_parallel = lambda records, *_args: records.squeeze(1)
        order = [2, 0, 3, 1]
        batch = Batch(router_shift_sample_ids=ids[order], routed_experts=routes[order])
        env = {"batch_iter": iter([batch]), "self": types.SimpleNamespace(router_shift_observer=observer), "get_device_id": lambda: "cuda"}
        exec(compile(prefix, "actor_cpu_metadata_prefix", "exec"), env)
        self.assertEqual(env["router_shift_sample_ids"].tolist(), [91, 17, 5, 2])
        torch.testing.assert_close(env["batch"]["routed_experts"], routes[order])
        observer.begin_current_microbatch(torch.zeros((4, 1)), torch.ones((4, 1)), env["router_shift_sample_ids"], 1)
        self.assertEqual(observer._current_indices[0, :, 0].tolist(), [91, 17, 5, 2])
        with self.assertRaisesRegex(RuntimeError, "duplicate"):
            observer.begin_current_microbatch(torch.zeros((4, 1)), torch.ones((4, 1)), env["router_shift_sample_ids"], 1)

    def test_current_observer_preserves_output_routes_gradients_and_optimizer_step(self):
        baseline, observed = TopKRouter(), TopKRouter()
        observed.load_state_dict(baseline.state_dict())
        replay_indices = torch.tensor([[3, 3], [1, 2]])
        queue = [replay_indices.clone()]
        for router in (baseline, observed):
            router.router_replay = types.SimpleNamespace(
                router_replay_action=types.SimpleNamespace(value="replay_forward"),
                target_topk_idx=replay_indices,
                replay_backward_list=queue,
            )
        observer = router_module.RouterShiftObserver([observed], _router_config(moe_router_pre_softmax=False))
        self.addCleanup(observer.close)
        inputs = torch.tensor([[1.0, 2.0, 3.0], [0.5, 1.0, -1.0]])
        old_logits = torch.nn.functional.linear(inputs, observed.weight).detach()
        observer.start_current_batch()
        observer._capturing = True
        observer._current_indices = replay_indices.unsqueeze(0)
        observer._current_old_log_probs = old_logits.log_softmax(-1).gather(-1, replay_indices).unsqueeze(0)
        observer._current_abs_sums = [None]
        left, right = inputs.clone().requires_grad_(), inputs.clone().requires_grad_()
        left_output, left_routes = baseline(left)
        right_output, right_routes = observed(right)
        left_loss, right_loss = left_output[:, 0].sum(), right_output[:, 0].sum()
        left_loss.backward()
        right_loss.backward()
        torch.testing.assert_close(right_output, left_output)
        torch.testing.assert_close(right_routes, left_routes)
        torch.testing.assert_close(right_loss, left_loss)
        torch.testing.assert_close(right.grad, left.grad)
        torch.testing.assert_close(observed.weight.grad, baseline.weight.grad)
        self.assertFalse(observer._current_abs_sums[0].requires_grad)
        for router in (baseline, observed):
            torch.optim.SGD(router.parameters(), lr=0.01).step()
        torch.testing.assert_close(observed.weight, baseline.weight)
        self.assertIs(observed.router_replay.replay_backward_list, queue)
        torch.testing.assert_close(queue[0], replay_indices)
        observer._capturing = False  # backward recompute must not recapture statistics
        previous = observer._current_abs_sums[0]
        observed(inputs)
        self.assertIs(observer._current_abs_sums[0], previous)


if __name__ == "__main__":
    unittest.main()
