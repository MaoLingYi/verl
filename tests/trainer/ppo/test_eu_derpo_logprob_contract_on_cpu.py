from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).parents[3]


def load_module(name: str, path: Path, postpone_annotations: bool = False):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    if postpone_annotations:
        source = "from __future__ import annotations\n" + path.read_text(encoding="utf-8")
        exec(compile(source, str(path), "exec"), module.__dict__)
    else:
        spec.loader.exec_module(module)
    return module


eu = load_module("eu_derpo_contract_under_test", ROOT / "verl" / "trainer" / "ppo" / "eu_derpo.py")


def load_rollout_correction():
    names = (
        "verl",
        "verl.utils",
        "verl.utils.torch_functional",
        "verl.protocol",
        "verl.trainer",
        "verl.trainer.config",
        "verl.trainer.config.algorithm",
        "verl.workers",
        "verl.workers.config",
        "verl.workers.config.actor",
    )
    saved = {name: sys.modules.get(name) for name in names}
    modules = {name: types.ModuleType(name) for name in names}

    def masked_sum(values, mask, axis=None):
        return (values * mask).sum(dim=axis)

    def masked_mean(values, mask, axis=None):
        return masked_sum(values, mask, axis) / mask.sum(dim=axis).clamp(min=1)

    functional = modules["verl.utils.torch_functional"]
    functional.masked_sum = masked_sum
    functional.masked_mean = masked_mean
    functional.distributed_masked_mean = masked_mean

    class DataProto:
        @classmethod
        def from_dict(cls, tensors):
            return types.SimpleNamespace(batch=tensors)

    modules["verl.protocol"].DataProto = DataProto
    modules["verl.trainer.config.algorithm"].RolloutCorrectionConfig = dict
    modules["verl.workers.config.actor"].PolicyLossConfig = dict
    sys.modules.update(modules)
    try:
        return load_module(
            "rollout_corr_contract_under_test",
            ROOT / "verl" / "trainer" / "ppo" / "rollout_corr_helper.py",
            postpone_annotations=True,
        )
    finally:
        for name, previous in saved.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


corr = load_rollout_correction()


class SyntheticBatch:
    def __init__(self, tensors):
        self.batch = dict(tensors)

    def union(self, other):
        self.batch.update(other.batch)
        return self


class TestThreeLogprobContract(unittest.TestCase):
    def setUp(self):
        self.old = torch.tensor([[-2.0, -4.0]])
        self.current = torch.tensor([[-0.2, -0.4]])
        self.rollout = torch.tensor([[-1.0, -1.0]])
        self.mask = torch.ones_like(self.old, dtype=torch.bool)

    def test_three_fields_exist(self):
        fields = eu.policy_prepass_tensors(self.old, torch.zeros_like(self.old), True)
        fields["rollout_log_probs"] = self.rollout
        self.assertTrue({"old_log_probs", "current_log_probs", "rollout_log_probs"} <= fields.keys())

    def test_old_policy_field_does_not_alias_current_or_rollout(self):
        fields = eu.policy_prepass_tensors(self.old, torch.zeros_like(self.old), True)
        fields["rollout_log_probs"] = self.rollout
        self.assertEqual(fields["old_log_probs"].data_ptr(), self.old.data_ptr())
        self.assertNotEqual(fields["old_log_probs"].data_ptr(), fields["current_log_probs"].data_ptr())
        self.assertNotEqual(fields["old_log_probs"].data_ptr(), fields["rollout_log_probs"].data_ptr())

    def test_generic_rollout_correction_reads_old_policy_field(self):
        batch = SyntheticBatch(
            {
                "old_log_probs": self.old,
                "current_log_probs": torch.full_like(self.current, 50.0),
                "rollout_log_probs": self.rollout,
                "response_mask": self.mask,
            }
        )
        _, metrics = corr.compute_rollout_correction_and_add_to_batch(batch, {})
        self.assertAlmostEqual(metrics["rollout_corr/kl"], 2.0)
        torch.testing.assert_close(batch.batch["response_mask"], self.mask)

    def test_expert_is_reads_current_over_rollout_and_ignores_old(self):
        routes = torch.zeros((1, 2, 1, 1), dtype=torch.long)
        advantages = torch.ones_like(self.current)

        def expert_is(old):
            batch = {
                "old_log_probs": old,
                "current_log_probs": self.current,
                "rollout_log_probs": self.rollout,
            }
            return eu.cluster_statistics(
                batch["current_log_probs"],
                batch["rollout_log_probs"],
                advantages,
                self.mask,
                routes,
                1,
                100.0,
            ).rho

        expected = torch.exp((self.current - self.rollout).mean())
        torch.testing.assert_close(expert_is(self.old)[0, 0, 0], expected)
        torch.testing.assert_close(expert_is(torch.full_like(self.old, -100.0))[0, 0, 0], expected)

    def test_legacy_disabled_prepass_is_unchanged(self):
        fields = eu.policy_prepass_tensors(self.old, torch.zeros_like(self.old), False)
        self.assertEqual(set(fields), {"old_log_probs", "entropys"})
        self.assertEqual(fields["old_log_probs"].data_ptr(), self.old.data_ptr())


if __name__ == "__main__":
    unittest.main()
