from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).parents[3]
MATH_MODULE = ROOT / "verl" / "trainer" / "ppo" / "eu_derpo.py"
ACTOR_MODULE = ROOT / "verl" / "workers" / "actor" / "megatron_actor.py"
SPEC = importlib.util.spec_from_file_location("eu_derpo_f_source_math_under_test", MATH_MODULE)
eu = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(eu)


class TestEUDERPOFSingleSource(unittest.TestCase):
    def setUp(self):
        self.current_f = torch.tensor([[-0.2, -0.4]], requires_grad=True)
        self.behavior = torch.tensor([[-1.0, -1.0]])
        self.old = torch.tensor([[-8.0, -8.0]])
        self.advantages = torch.ones((1, 2))
        self.mask = torch.ones((1, 2), dtype=torch.bool)
        self.f_routes = torch.tensor([[[[0]], [[0]]]])
        self.p_routes = torch.tensor([[[[1]], [[1]]]])

    def test_same_f_detached_statistics_multiply_same_f_graph(self):
        stats = eu.cluster_statistics(
            self.current_f.detach(), self.behavior, self.advantages, self.mask, self.f_routes, 2, 0.05
        )
        coefficient, _ = eu.edppo_token_coefficients(stats, self.f_routes, self.mask)
        self.assertFalse(stats.rho.requires_grad)
        loss = -(coefficient * self.current_f).sum()
        loss.backward()
        torch.testing.assert_close(self.current_f.grad, -coefficient)

    def test_p_routes_and_old_logprob_do_not_define_f_coefficient(self):
        f_stats = eu.cluster_statistics(
            self.current_f.detach(), self.behavior, self.advantages, self.mask, self.f_routes, 2, 0.05
        )
        p_stats = eu.cluster_statistics(
            self.old, self.behavior, self.advantages, self.mask, self.p_routes, 2, 0.05
        )
        f_coefficient, _ = eu.edppo_token_coefficients(f_stats, self.f_routes, self.mask)
        p_coefficient, _ = eu.edppo_token_coefficients(p_stats, self.p_routes, self.mask)
        self.assertFalse(torch.equal(f_coefficient, p_coefficient))
        self.assertGreater(f_stats.count[0, 0, 0].item(), 0)
        self.assertEqual(f_stats.count[0, 0, 1].item(), 0)

    def test_production_timing_contract_is_loss_closure_after_f(self):
        source = ACTOR_MODULE.read_text(encoding="utf-8")
        loss_start = source.index("def loss_func(")
        forward_start = source.index("def forward_step(", loss_start)
        finish_f = source.index("eu_derpo_observer.finish_main_microbatch(", forward_start)
        capture = source.index("eu_derpo_f_routes=eu_derpo_f_routes", finish_f)
        self.assertLess(finish_f, capture)
        loss_source = source[loss_start:forward_start]
        self.assertIn("log_prob.detach()", loss_source)
        self.assertIn('data.get("rollout_log_probs")', loss_source)
        self.assertIn("eu_derpo_f_routes", loss_source)
        self.assertNotIn('data["current_log_probs"]', loss_source)
        self.assertNotIn('data["eu_derpo_token_coeff"]', source)

    def test_dense_dp_collectives_only_reduce_logging_metrics(self):
        source = ACTOR_MODULE.read_text(encoding="utf-8")
        update = source[source.index("def update_policy(") :]
        normalize = update.index("normalize_group_utility(")
        first_dp_collective = update.index("mpu.get_data_parallel_world_size()")
        self.assertGreater(first_dp_collective, normalize)
        self.assertNotIn("get_data_parallel", update[:normalize])


if __name__ == "__main__":
    unittest.main()
