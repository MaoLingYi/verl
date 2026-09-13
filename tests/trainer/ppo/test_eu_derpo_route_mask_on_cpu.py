from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import torch


MODULE = Path(__file__).parents[3] / "verl" / "trainer" / "ppo" / "eu_derpo.py"
SPEC = importlib.util.spec_from_file_location("eu_derpo_route_mask_under_test", MODULE)
eu = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(eu)


class TestEUDERPORouteMask(unittest.TestCase):
    @staticmethod
    def statistics(routes, mask, current=None, behavior=None):
        batch, tokens = mask.shape
        current = torch.zeros((batch, tokens)) if current is None else current
        behavior = torch.zeros_like(current) if behavior is None else behavior
        advantages = torch.ones_like(current)
        return eu.cluster_statistics(current, behavior, advantages, mask, routes, 16, 100.0)

    def test_valid_unique_routes_pass(self):
        routes = torch.tensor([[[[1, 2, 3, 4, 5, 6, 7, 8]]]])
        stats = self.statistics(routes, torch.tensor([[True]]))
        self.assertEqual(stats.count.sum().item(), 8)

    def test_padding_duplicate_placeholder_is_ignored(self):
        routes = torch.tensor([[[[1, 2, 3, 4, 5, 6, 7, 8]], [[0, 0, 0, 0, 0, 0, 0, 0]]]])
        stats = self.statistics(routes, torch.tensor([[True, False]]))
        self.assertEqual(stats.count.sum().item(), 8)
        self.assertEqual(stats.count[0, 0, 0].item(), 0)

    def test_valid_duplicate_fails_fast_with_diagnostic(self):
        routes = torch.tensor([[[[1, 1, 3, 4, 5, 6, 7, 8]]]])
        with self.assertRaisesRegex(
            ValueError,
            r"duplicate_valid_positions=1, first=\(sample=0, token=0, layer=0\).*routes=\[1, 1, 3, 4, 5, 6, 7, 8\]",
        ):
            self.statistics(routes, torch.tensor([[True]]))

    def test_mixed_valid_and_padding_batch_excludes_padding_edges(self):
        routes = torch.tensor(
            [
                [
                    [[1, 2, 3, 4, 5, 6, 7, 8]],
                    [[0, 0, 0, 0, 0, 0, 0, 0]],
                    [[8, 9, 10, 11, 12, 13, 14, 15]],
                ]
            ]
        )
        stats = self.statistics(routes, torch.tensor([[True, False, True]]))
        self.assertEqual(stats.count.sum().item(), 16)
        self.assertEqual(stats.count[0, 0, 0].item(), 0)

    def test_valid_duplicate_hidden_among_padding_still_fails(self):
        routes = torch.tensor(
            [
                [
                    [[1, 2, 3, 4, 5, 6, 7, 8]],
                    [[0, 0, 0, 0, 0, 0, 0, 0]],
                    [[9, 9, 10, 11, 12, 13, 14, 15]],
                ]
            ]
        )
        with self.assertRaisesRegex(ValueError, r"first=\(sample=0, token=2, layer=0\)"):
            self.statistics(routes, torch.tensor([[True, False, True]]))

    def test_valid_unique_cluster_numerics_are_unchanged(self):
        routes = torch.tensor(
            [[[[0, 1, 2, 3, 4, 5, 6, 7]], [[0, 1, 2, 3, 4, 5, 6, 7]]]]
        )
        current = torch.log(torch.tensor([[0.4, 0.2]]))
        behavior = torch.log(torch.tensor([[0.2, 0.1]]))
        stats = self.statistics(routes, torch.tensor([[True, True]]), current, behavior)
        torch.testing.assert_close(stats.count[0, 0, :8], torch.full((8,), 2.0))
        torch.testing.assert_close(stats.count[0, 0, 8:], torch.zeros(8))
        torch.testing.assert_close(stats.rho[0, 0, :8], torch.full((8,), 2.0))
        torch.testing.assert_close(stats.divergence[0, 0, :8], torch.full((8,), 0.15))


if __name__ == "__main__":
    unittest.main()
