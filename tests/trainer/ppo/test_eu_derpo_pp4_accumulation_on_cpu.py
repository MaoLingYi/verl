from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import torch


MODULE = Path(__file__).parents[3] / "verl" / "trainer" / "ppo" / "eu_derpo.py"
SPEC = importlib.util.spec_from_file_location("eu_derpo_pp4_accumulation_under_test", MODULE)
eu = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(eu)


def reference_clusters(current, behavior, mask, routes, experts):
    batch, tokens, layers, topk = routes.shape
    count = torch.zeros((batch, layers, experts))
    log_sum = torch.zeros_like(count)
    divergence_sum = torch.zeros_like(count)
    for b in range(batch):
        for t in range(tokens):
            if not mask[b, t]:
                continue
            log_ratio = current[b, t] - behavior[b, t]
            binary_tv = (behavior[b, t].exp() - current[b, t].exp()).abs()
            for layer in range(layers):
                for k in range(topk):
                    expert = routes[b, t, layer, k]
                    count[b, layer, expert] += 1
                    log_sum[b, layer, expert] += log_ratio
                    divergence_sum[b, layer, expert] += binary_tv
    active = count > 0
    denominator = count.masked_fill(~active, 1)
    rho = (log_sum / denominator).exp().masked_fill(~active, 0)
    divergence = (divergence_sum / denominator).masked_fill(~active, 0)
    return count, rho, divergence


def reference_utility(utility, mask, routes, experts):
    batch, tokens, layers, topk = routes.shape
    sums = torch.zeros((batch, layers, experts))
    counts = torch.zeros_like(sums)
    for b in range(batch):
        for t in range(tokens):
            if not mask[b, t]:
                continue
            for layer in range(layers):
                for k in range(topk):
                    expert = routes[b, t, layer, k]
                    sums[b, layer, expert] += utility[b, t, layer, k]
                    counts[b, layer, expert] += 1
    return sums, counts


class TestEUDERPOPP4Accumulation(unittest.TestCase):
    def setUp(self):
        self.batch, self.tokens, self.layers, self.topk, self.experts = 2, 3, 12, 8, 128
        self.routes = torch.empty((self.batch, self.tokens, self.layers, self.topk), dtype=torch.long)
        for b in range(self.batch):
            for t in range(self.tokens):
                for layer in range(self.layers):
                    start = (37 * b + 19 * t + 11 * layer) % self.experts
                    self.routes[b, t, layer] = (torch.arange(self.topk) + start) % self.experts
        self.mask = torch.tensor([[True, True, False], [True, False, True]])
        self.current = torch.tensor([[-0.2, -0.4, -0.6], [-0.3, -0.5, -0.7]])
        self.behavior = torch.tensor([[-0.5, -0.1, -0.8], [-0.6, -0.9, -0.2]])
        self.advantages = torch.tensor([[1.0] * self.tokens, [-1.0] * self.tokens])
        self.reference = reference_clusters(
            self.current, self.behavior, self.mask, self.routes, self.experts
        )

    def statistics(self, routes=None):
        return eu.cluster_statistics(
            self.current,
            self.behavior,
            self.advantages,
            self.mask,
            self.routes if routes is None else routes,
            self.experts,
            0.05,
        )

    def test_pp4_cluster_statistics_does_not_crash(self):
        stats = self.statistics()
        self.assertEqual(stats.count.shape, (2, 12, 128))

    def test_per_layer_destination_slice_is_non_contiguous(self):
        accumulator = torch.zeros((self.batch, self.layers, self.experts))
        self.assertFalse(accumulator[:, 1].is_contiguous())
        with self.assertRaises(RuntimeError):
            accumulator[:, 1].view(-1)

    def test_cluster_counts_match_python_reference(self):
        torch.testing.assert_close(self.statistics().count, self.reference[0])

    def test_cluster_rho_matches_python_reference(self):
        torch.testing.assert_close(self.statistics().rho, self.reference[1])

    def test_binary_tv_divergence_matches_python_reference(self):
        torch.testing.assert_close(self.statistics().divergence, self.reference[2])

    def test_multilayer_utility_matches_python_reference(self):
        utility = torch.arange(self.routes.numel(), dtype=torch.float32).reshape_as(self.routes) / 100
        actual_sums, actual_counts = eu.aggregate_edge_utility(
            utility, self.routes, self.mask, self.experts
        )
        expected_sums, expected_counts = reference_utility(utility, self.mask, self.routes, self.experts)
        torch.testing.assert_close(actual_sums, expected_sums)
        torch.testing.assert_close(actual_counts, expected_counts)

    def test_padding_contributes_no_cluster_or_utility_edges(self):
        routes = self.routes.clone()
        routes[~self.mask] = 0
        stats = self.statistics(routes)
        expected = reference_clusters(self.current, self.behavior, self.mask, routes, self.experts)
        torch.testing.assert_close(stats.count, expected[0])
        self.assertEqual(stats.count.sum().item(), self.mask.sum().item() * self.layers * self.topk)
        utility = torch.ones_like(routes, dtype=torch.float32)
        utility[~self.mask] = 1_000_000
        sums, counts = eu.aggregate_edge_utility(utility, routes, self.mask, self.experts)
        expected_sums, expected_counts = reference_utility(utility, self.mask, routes, self.experts)
        torch.testing.assert_close(sums, expected_sums)
        torch.testing.assert_close(counts, expected_counts)

    def test_valid_duplicate_still_fails_fast(self):
        routes = self.routes.clone()
        routes[0, 0, 4, 1] = routes[0, 0, 4, 0]
        with self.assertRaisesRegex(ValueError, "unique selected token-expert edges"):
            self.statistics(routes)

    def test_local_layers_and_batches_are_isolated(self):
        routes = torch.empty((2, 2, 12, 8), dtype=torch.long)
        for layer in range(12):
            routes[:, :, layer] = torch.arange(layer * 8, layer * 8 + 8)
        mask = torch.tensor([[True, False], [True, False]])
        values = torch.ones((2, 2, 12, 8))
        sums, counts = eu.aggregate_edge_utility(values, routes, mask, 128)
        for b in range(2):
            torch.testing.assert_close(counts[b, 0, :8], torch.ones(8))
            torch.testing.assert_close(counts[b, 0, 8:16], torch.zeros(8))
            torch.testing.assert_close(counts[b, 1, :8], torch.zeros(8))
            torch.testing.assert_close(counts[b, 1, 8:16], torch.ones(8))
            torch.testing.assert_close(sums[b], counts[b])


if __name__ == "__main__":
    unittest.main()
