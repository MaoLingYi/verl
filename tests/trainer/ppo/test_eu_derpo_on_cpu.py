from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import torch


MODULE = Path(__file__).parents[3] / "verl" / "trainer" / "ppo" / "eu_derpo.py"
SPEC = importlib.util.spec_from_file_location("eu_derpo_under_test", MODULE)
eu = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(eu)


class TestEUDERPOMath(unittest.TestCase):
    def setUp(self):
        self.current = torch.log(torch.tensor([[0.4, 0.2, 0.3], [0.1, 0.5, 0.25]]))
        self.behavior = torch.log(torch.tensor([[0.2, 0.3, 0.3], [0.2, 0.4, 0.25]]))
        self.mask = torch.ones(2, 3, dtype=torch.bool)
        self.advantages = torch.tensor([[1.0] * 3, [-1.0] * 3])
        self.routes = torch.tensor(
            [
                [[[0, 1]], [[0, 2]], [[1, 2]]],
                [[[0, 2]], [[0, 1]], [[1, 2]]],
            ]
        )

    def test_behavior_anchor_is_required(self):
        with self.assertRaisesRegex(ValueError, "rollout_log_probs"):
            eu.cluster_statistics(self.current, None, self.advantages, self.mask, self.routes, 3, 0.1)

    def test_aime_history_requires_explicit_binary_accuracy(self):
        self.assertEqual(eu.aime_accuracy_values({"acc": [True, False]}, [-1.0, 1.0]), [1.0, 0.0])
        with self.assertRaisesRegex(RuntimeError, "explicit acc"):
            eu.aime_accuracy_values({}, [1.0])
        with self.assertRaisesRegex(RuntimeError, "finite binary"):
            eu.aime_accuracy_values({"acc": [0.5]}, [1.0])

    def test_aime_history_appends_one_record_per_line(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "aime24_validation_history.jsonl")
            eu.append_aime_history(path, {"global_step": 50})
            eu.append_aime_history(path, {"global_step": 100})
            records = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]
        self.assertEqual(records, [{"global_step": 50}, {"global_step": 100}])

    def test_actual_optimizer_minibatch_count_and_complete_prompt_groups(self):
        self.assertEqual(eu.audit_optimizer_minibatches(384, 1, 384), 1)
        with self.assertRaisesRegex(RuntimeError, "not divisible"):
            eu.audit_optimizer_minibatches(383, 1, 384)
        counts = eu.validate_prompt_groups(torch.arange(48).repeat_interleave(8), 8)
        self.assertTrue(torch.equal(counts, torch.full((48,), 8)))
        with self.assertRaisesRegex(RuntimeError, "incomplete"):
            eu.validate_prompt_groups(torch.tensor([0, 0, 1]), 2)

    def test_expert_is_and_binary_tv_use_the_same_current_cluster(self):
        stats = eu.cluster_statistics(
            self.current, self.behavior, self.advantages, self.mask, self.routes, 3, 0.1
        )
        token_ids = torch.tensor([0, 1])
        expected_rho = torch.exp((self.current[0, token_ids] - self.behavior[0, token_ids]).mean())
        expected_tv = (self.behavior[0, token_ids].exp() - self.current[0, token_ids].exp()).abs().mean()
        torch.testing.assert_close(stats.rho[0, 0, 0], expected_rho)
        torch.testing.assert_close(stats.divergence[0, 0, 0], expected_tv)

    def test_directional_mask_all_cases(self):
        current = torch.log(torch.tensor([[0.6], [0.2], [0.6], [0.2], [0.6], [0.2], [0.6], [0.2]]))
        behavior = torch.log(torch.tensor([[0.3], [0.4], [0.3], [0.4], [0.3], [0.4], [0.55], [0.25]]))
        advantage = torch.tensor([[1.0], [1.0], [-1.0], [-1.0], [0.0], [0.0], [1.0], [-1.0]])
        routes = torch.zeros((8, 1, 1, 1), dtype=torch.long)
        stats = eu.cluster_statistics(current, behavior, advantage, torch.ones_like(current, dtype=torch.bool), routes, 1, 0.1)
        self.assertEqual(stats.mask[:, 0, 0].tolist(), [False, True, True, False, True, True, True, True])

    def test_no_ratio_clipping_in_edppo_coefficient(self):
        current = torch.tensor([[3.0]])
        behavior = torch.tensor([[-3.0]])
        routes = torch.zeros((1, 1, 1, 1), dtype=torch.long)
        stats = eu.cluster_statistics(current, behavior, torch.ones_like(current), torch.ones_like(current, dtype=torch.bool), routes, 1, 100.0)
        coefficient, _ = eu.edppo_token_coefficients(stats, routes, torch.ones_like(current, dtype=torch.bool))
        torch.testing.assert_close(coefficient[0, 0], torch.tensor(6.0).exp())

    def test_edge_uniqueness_and_overlap(self):
        duplicate = self.routes.clone()
        duplicate[0, 0, 0] = torch.tensor([1, 1])
        with self.assertRaisesRegex(ValueError, "unique"):
            eu.cluster_statistics(self.current, self.behavior, self.advantages, self.mask, duplicate, 3, 0.1)
        utility = torch.arange(self.routes.numel(), dtype=torch.float32).reshape_as(self.routes)
        sums, counts = eu.aggregate_edge_utility(utility, self.routes, self.mask, 3)
        self.assertEqual(counts[0, 0].tolist(), [2.0, 2.0, 2.0])
        self.assertEqual(sums[0, 0].tolist(), [2.0, 5.0, 8.0])

    def test_group_normalization_is_per_expert_and_skips_sparse_or_constant(self):
        sums = torch.tensor([[[1.0, 5.0, 2.0]], [[3.0, 5.0, 0.0]], [[100.0, 5.0, 0.0]]])
        counts = torch.tensor([[[1.0, 1.0, 1.0]], [[1.0, 1.0, 0.0]], [[0.0, 1.0, 0.0]]])
        groups = torch.tensor([7, 7, 7])
        normalized, skipped = eu.normalize_group_utility(sums, counts, groups, 1e-6, 2, 1e-6)
        torch.testing.assert_close(normalized[:2, 0, 0], torch.tensor([-1.0, 1.0]), atol=2e-6, rtol=0)
        self.assertEqual(normalized[:, 0, 1].tolist(), [0.0, 0.0, 0.0])
        self.assertTrue(skipped[:, 0, 1].all())
        self.assertEqual(normalized[:, 0, 2].tolist(), [0.0, 0.0, 0.0])
        self.assertTrue(skipped[0, 0, 2])
        self.assertFalse(normalized.requires_grad)

    def test_v12_centered_routing_utility_and_alpha_fail_fast(self):
        sensitivity = torch.tensor([[0.2, -0.3]])
        alpha = torch.tensor([[0.25, 0.75]])
        utility = eu.centered_routing_utility(alpha, sensitivity)
        torch.testing.assert_close(utility, sensitivity - (alpha * sensitivity).sum(-1, keepdim=True))
        torch.testing.assert_close((alpha * utility).sum(-1), torch.zeros(1), atol=1e-7, rtol=0)
        with self.assertRaises(FloatingPointError):
            eu.centered_routing_utility(torch.tensor([[0.0, 1.0]]), sensitivity)

    def test_route_mismatch_fails_fast(self):
        metrics = eu.assert_same_routes(self.routes, self.routes.clone())
        self.assertEqual(metrics["route_equal_fraction"], 1.0)
        changed = self.routes.clone()
        changed[0, 0, 0, 0] = 2
        with self.assertRaisesRegex(RuntimeError, "route mismatch"):
            eu.assert_same_routes(self.routes, changed)

    def test_route_equality_ignores_invalid_prompt_and_padding(self):
        changed = self.routes.clone()
        changed[:, 0] = 2 - changed[:, 0]
        valid = torch.ones(2, 3, dtype=torch.bool)
        valid[:, 0] = False
        metrics = eu.assert_same_routes(self.routes, changed, valid)
        self.assertEqual(metrics["route_mismatch_count"], 0)
        changed[0, 1, 0, 0] = 2
        with self.assertRaisesRegex(RuntimeError, "route mismatch"):
            eu.assert_same_routes(self.routes, changed, valid)

    def test_router_auxiliary_is_router_only_and_stop_gradient(self):
        hidden = torch.randn(4, 5, requires_grad=True)
        backbone_scale = torch.tensor(2.0, requires_grad=True)
        hidden_from_backbone = hidden * backbone_scale
        router = torch.randn(7, 5, requires_grad=True)
        indices = torch.tensor([[0, 2], [1, 3], [2, 4], [3, 5]])
        coefficient = torch.randn(4, 2, requires_grad=True)
        loss = eu.router_auxiliary_loss(hidden_from_backbone, router, indices, coefficient)
        loss.backward()
        self.assertIsNotNone(router.grad)
        self.assertGreater(router.grad.norm().item(), 0)
        self.assertIsNone(hidden.grad)
        self.assertIsNone(backbone_scale.grad)
        self.assertIsNone(coefficient.grad)

    def test_recompute_hook_count_cannot_double(self):
        eu.validate_recompute_hook_count(12, 12)
        eu.validate_recompute_edge_counts(torch.tensor([8, 16]), torch.tensor([8, 16]))
        with self.assertRaisesRegex(RuntimeError, "hook count mismatch"):
            eu.validate_recompute_hook_count(24, 12)
        with self.assertRaisesRegex(RuntimeError, "duplicated or dropped"):
            eu.validate_recompute_edge_counts(torch.tensor([16, 32]), torch.tensor([8, 16]))


if __name__ == "__main__":
    unittest.main()
