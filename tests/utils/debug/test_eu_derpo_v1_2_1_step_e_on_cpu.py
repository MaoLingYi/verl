from __future__ import annotations

import importlib.util
import io
import sys
import types
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

import torch


ROOT = Path(__file__).parents[3]
MATH_MODULE = ROOT / "verl" / "trainer" / "ppo" / "eu_derpo.py"
OBSERVER_MODULE = ROOT / "verl" / "utils" / "debug" / "eu_derpo.py"


def load_module():
    math_spec = importlib.util.spec_from_file_location("eu_derpo_v121_math", MATH_MODULE)
    math_module = importlib.util.module_from_spec(math_spec)
    assert math_spec.loader is not None
    math_spec.loader.exec_module(math_module)

    router_shift = types.ModuleType("verl.utils.debug.router_shift")
    router_shift.RouterShiftObserver = object
    modules = {
        "verl": types.ModuleType("verl"),
        "verl.trainer": types.ModuleType("verl.trainer"),
        "verl.trainer.ppo": types.ModuleType("verl.trainer.ppo"),
        "verl.trainer.ppo.eu_derpo": math_module,
        "verl.utils": types.ModuleType("verl.utils"),
        "verl.utils.debug": types.ModuleType("verl.utils.debug"),
        "verl.utils.debug.router_shift": router_shift,
    }
    saved = {name: sys.modules.get(name) for name in modules}
    sys.modules.update(modules)
    try:
        spec = importlib.util.spec_from_file_location("eu_derpo_v121_observer", OBSERVER_MODULE)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        for name, previous in saved.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


class TestEUDERPOV121StepE(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.eu = load_module()

    def test_selected_support_q_matches_reference_and_sign(self):
        logits = torch.tensor([[0.2, -0.1, 0.7, 1.2]], requires_grad=True)
        support = torch.tensor([[3, 0]], dtype=torch.uint8)
        coefficient = torch.tensor([[0.4, -0.2]])
        loss, objective = self.eu._selected_support_loss(logits, support, coefficient, 0.05)
        reference_log_q = torch.log_softmax(logits[:, [3, 0]].float(), -1)
        torch.testing.assert_close(objective, (coefficient * reference_log_q).sum())
        torch.testing.assert_close(loss, -0.05 * objective)

    def test_support_permutation_preserves_objective_and_router_gradient(self):
        generator = torch.Generator().manual_seed(5)
        hidden = torch.randn(7, 5, generator=generator)
        weight = torch.randn(11, 5, generator=generator, requires_grad=True)
        support = torch.stack([torch.randperm(11, generator=generator)[:4] for _ in range(7)])
        coefficient = torch.randn(7, 4, generator=generator)

        def evaluate(ids, coeff):
            logits = torch.nn.functional.linear(hidden.detach(), weight)
            loss, objective = self.eu._selected_support_loss(logits, ids, coeff, 0.05)
            return objective, torch.autograd.grad(loss, weight, create_graph=False)[0]

        objective, gradient = evaluate(support, coefficient)
        permutation = torch.tensor([2, 0, 3, 1])
        permuted_objective, permuted_gradient = evaluate(
            support[:, permutation], coefficient[:, permutation]
        )
        torch.testing.assert_close(permuted_objective, objective)
        torch.testing.assert_close(permuted_gradient, gradient)

    def test_v12_full_a_and_v121_cached_f_gradient_are_equivalent_when_states_match(self):
        generator = torch.Generator().manual_seed(19)
        hidden_f = torch.randn(9, 6, generator=generator)
        hidden_a = hidden_f.clone()
        support_f = torch.stack([torch.randperm(13, generator=generator)[:4] for _ in range(9)])
        support_a = support_f.clone()
        coefficient = torch.randn(9, 4, generator=generator)
        initial = torch.randn(13, 6, generator=generator)

        old_weight = initial.clone().requires_grad_()
        old_logits = torch.nn.functional.linear(hidden_a.detach(), old_weight)
        old_log_q = old_logits.gather(-1, support_a).float().log_softmax(-1)
        old_objective = (coefficient.detach() * old_log_q).sum()
        old_gradient = torch.autograd.grad(-0.05 * old_objective, old_weight)[0]

        new_weight = initial.clone().requires_grad_()
        new_logits = torch.nn.functional.linear(hidden_f.detach(), new_weight)
        new_loss, new_objective = self.eu._selected_support_loss(
            new_logits, support_f.to(torch.uint8), coefficient, 0.05
        )
        new_gradient = torch.autograd.grad(new_loss, new_weight)[0]
        torch.testing.assert_close(new_objective, old_objective)
        torch.testing.assert_close(new_gradient, old_gradient)
        main = torch.randn_like(new_gradient)
        torch.testing.assert_close(main + new_gradient, main + old_gradient)

    def test_lambda_scales_once_and_hidden_is_detached(self):
        hidden_leaf = torch.randn(6, 3, requires_grad=True)
        backbone_scale = torch.tensor(2.0, requires_grad=True)
        hidden = hidden_leaf * backbone_scale
        support = torch.tensor([[0, 2]] * 6, dtype=torch.uint8)
        coefficient = torch.randn(6, 2)
        initial_weight = torch.randn(4, 3)
        norms = []
        for value in (0.0, 0.05, 0.1):
            weight = initial_weight.clone().requires_grad_()
            logits = torch.nn.functional.linear(hidden.detach(), weight)
            loss, _ = self.eu._selected_support_loss(logits, support, coefficient, value)
            norms.append(torch.autograd.grad(loss, weight, create_graph=False)[0].norm())
        self.assertEqual(norms[0].item(), 0.0)
        torch.testing.assert_close(norms[2], norms[1] * 2, rtol=2e-5, atol=2e-5)
        self.assertIsNone(hidden_leaf.grad)
        self.assertIsNone(backbone_scale.grad)

    def test_local_response_plan_is_valid_only_and_shared(self):
        attention = torch.tensor(
            [[False, True, True, True, True, True], [True, True, True, True, False, False]]
        )
        response = torch.tensor([[True, False, True], [True, True, False]])
        sample_ids = torch.tensor([17, 23])
        plan0 = self.eu._plan_local_response_rows(attention, response, sample_ids, 3, 2, 0)
        plan1 = self.eu._plan_local_response_rows(attention, response, sample_ids, 3, 2, 1)
        self.assertEqual(plan0["sample_id"].dtype, torch.int64)
        self.assertEqual(plan0["sample_row"].dtype, torch.int32)
        self.assertEqual(plan0["response_position"].dtype, torch.int32)
        self.assertEqual(plan0["rows"] + plan1["rows"], int(response.sum()))
        combined = sorted(
            zip(
                torch.cat((plan0["sample_id"], plan1["sample_id"])).tolist(),
                torch.cat((plan0["response_position"], plan1["response_position"])).tolist(),
            )
        )
        self.assertEqual(combined, [(17, 0), (17, 2), (23, 0), (23, 1)])

    def test_local_response_plan_handles_heavy_sp_skew(self):
        attention = torch.ones((1, 12), dtype=torch.bool)
        response = torch.tensor([[True] * 8])
        sample_ids = torch.tensor([127])
        first = self.eu._plan_local_response_rows(attention, response, sample_ids, 8, 2, 0)
        second = self.eu._plan_local_response_rows(attention, response, sample_ids, 8, 2, 1)
        self.assertEqual(first["rows"] + second["rows"], 8)
        self.assertNotEqual(first["rows"], second["rows"])

    def test_cache_budget_and_64_mib_staging_cap(self):
        budget = self.eu._cache_byte_plan(98_304, 48, 2048, 8)
        self.assertEqual(budget["hidden"], 18 * 1024**3)
        self.assertEqual(budget["support"], 36 * 1024**2)
        self.assertEqual(budget["metadata"], 1_572_864)
        rows = self.eu._staging_chunk_rows(64, 2048, 8)
        self.assertEqual(rows, 16_225)
        self.assertLessEqual(rows * (2048 * 2 + 8 + 8 * 4), 64 * 1024**2)
        self.assertGreater((rows + 1) * (2048 * 2 + 8 + 8 * 4), 64 * 1024**2)
        logits = torch.zeros((1, 128), requires_grad=True)
        support = torch.tensor([[0, 127]], dtype=torch.uint8)
        loss, _ = self.eu._selected_support_loss(logits, support, torch.ones((1, 2)), 0.05)
        self.assertTrue(torch.isfinite(loss))

    def test_node_ram_decision_uses_sum_and_safety_margin(self):
        required = [10, 20, 30, 40, 50, 60, 70, 80]
        passed = self.eu._node_ram_decision(required, mem_available=1_000, safety_margin=100)
        self.assertTrue(passed["passed"])
        self.assertEqual(passed["node_required"], sum(required))
        failed = self.eu._node_ram_decision(required, mem_available=459, safety_margin=100)
        self.assertFalse(failed["passed"])

    def test_low_node_ram_headroom_warns_without_blocking_cache_allocation(self):
        observer = object.__new__(self.eu.EUDERPOObserver)
        observer.routers = [types.SimpleNamespace(weight=torch.empty(0))]
        output = io.StringIO()
        with (
            mock.patch.object(self.eu.torch.cuda, "is_available", return_value=False),
            mock.patch.object(self.eu.torch.distributed, "is_initialized", return_value=False),
            mock.patch.object(self.eu, "_read_mem_available", return_value=40),
            mock.patch.object(self.eu, "HOST_RAM_SAFETY_MARGIN_BYTES", 32),
            redirect_stdout(output),
        ):
            decision = observer._coordinated_ram_preflight(16)
        self.assertFalse(decision["passed"])
        self.assertIn("WARNING: EU-DERPO node RAM headroom low; continuing", output.getvalue())
        self.assertIn("EU-DERPO host RAM preflight", output.getvalue())

    def test_error_decorator_releases_cache_references(self):
        class Observer:
            cleared = False

            def clear(self):
                self.cleared = True

        class Actor:
            eu_derpo_observer = Observer()

            @self.eu.clear_eu_derpo_on_error
            def fail(self):
                raise RuntimeError("boom")

        actor = Actor()
        with self.assertRaisesRegex(RuntimeError, "boom"):
            actor.fail()
        self.assertTrue(actor.eu_derpo_observer.cleared)

    def test_coefficient_builder_joins_by_expert_id(self):
        stats = types.SimpleNamespace(
            count=torch.tensor([[[2.0, 4.0, 5.0, 1.0]]]),
            mask=torch.tensor([[[True, False, True, True]]]),
        )
        normalized = torch.tensor([[[0.5, 10.0, -2.0, 3.0]]])
        rows = torch.tensor([0, 0])
        support = torch.tensor([[3, 0], [2, 1]], dtype=torch.uint8)
        coefficient = self.eu._build_edge_coefficients(
            stats, normalized, torch.tensor([2.0]), rows, 0, support, 1
        )
        expected = torch.tensor([[1.5, 0.125], [-0.2, 0.0]])
        torch.testing.assert_close(coefficient, expected)
        stats.count[0, 0, 3] = 0
        with self.assertRaisesRegex(RuntimeError, "missing from cluster statistics"):
            self.eu._build_edge_coefficients(
                stats, normalized, torch.tensor([2.0]), rows, 0, support, 1
            )


if __name__ == "__main__":
    unittest.main()
