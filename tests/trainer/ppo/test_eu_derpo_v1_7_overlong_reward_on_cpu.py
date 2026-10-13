"""Execute pinned native reward code and the V1.7 fit AST with CPU fakes."""

import ast
import asyncio
import gc
import importlib.util
import inspect
import sys
import types
import weakref
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf


ROOT = Path(__file__).parents[3]
spec = importlib.util.spec_from_file_location("v17_overlong_harness", Path(__file__).with_name("test_eu_derpo_v1_7_on_cpu.py"))
harness = importlib.util.module_from_spec(spec)
spec.loader.exec_module(harness)
math_dapo = harness.load("v17_overlong_math", "verl/utils/reward_score/math_dapo.py")


def execute_symbols(path, names, namespace):
    source = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    nodes = [node for node in source.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[])), path, "exec"), namespace)


@pytest.fixture
def reward_loader(monkeypatch):
    # Replace only the Ray-dependent base initialization; run native registry/loader/managers/scorer.
    class CPUBase:
        def __init__(self, config, tokenizer, compute_score):
            self.config, self.tokenizer, self.compute_score = config, tokenizer, compute_score
            self.loop = asyncio.get_running_loop()

    package = types.ModuleType("v17_overlong_scores")
    package.math_dapo = math_dapo
    monkeypatch.setitem(sys.modules, package.__name__, package)
    monkeypatch.setitem(sys.modules, package.__name__ + ".math_dapo", math_dapo)
    namespace = {"__package__": package.__name__, "inspect": inspect,
                 "RewardManagerBase": CPUBase, "REWARD_MANAGER": {}}
    execute_symbols("verl/utils/reward_score/__init__.py", {"default_compute_score"}, namespace)
    execute_symbols("verl/experimental/reward_loop/reward_manager/registry.py",
                    {"register", "get_reward_manager_cls"}, namespace)
    for name, filename in (("NaiveRewardManager", "naive"), ("DAPORewardManager", "dapo")):
        execute_symbols(f"verl/experimental/reward_loop/reward_manager/{filename}.py", {name}, namespace)
    module = types.ModuleType("verl.experimental.reward_loop.reward_manager")
    module.get_reward_manager_cls = namespace["get_reward_manager_cls"]
    monkeypatch.setitem(sys.modules, module.__name__, module)
    execute_symbols("verl/trainer/ppo/reward.py", {"get_custom_reward_fn", "load_reward_manager"}, namespace)
    return namespace["load_reward_manager"]


class Sample:
    def __init__(self, length, correct, source="math_dapo"):
        self.batch = {
            "responses": torch.full((16384,), int(correct), dtype=torch.long),
            "attention_mask": torch.cat((torch.ones(length + 2, dtype=torch.long),
                                         torch.zeros(16384 - length, dtype=torch.long))),
        }
        self.non_tensor_batch = {"data_source": source, "reward_model": {"ground_truth": "7"},
                                 "extra_info": {"index": 42, "tag": "math"}}

    def __len__(self):
        return 1

    def __getitem__(self, index):
        assert index == 0
        return self


def config(name, enabled):
    return OmegaConf.create({"reward": {
        "reward_manager": {"source": "register", "name": name},
        "custom_reward_function": {"path": None},
        "reward_kwargs": {"max_resp_len": 16384, "overlong_buffer_cfg": {
            "enable": enabled, "len": 4096, "penalty_factor": 1., "log": True,
        }},
    }})


TOKENIZER = types.SimpleNamespace(decode=lambda ids, **kwargs: "Answer: 7" if ids[0] else "Answer: 8")


@pytest.mark.parametrize("length,penalty", [(10, 0.), (12288, 0.), (12289, -1 / 4096),
                                           (14336, -.5), (16384, -1.)])
@pytest.mark.parametrize("correct", [False, True])
def test_native_dapo_additive_boundaries_preserve_original_acc(reward_loader, length, penalty, correct):
    async def run():
        manager = reward_loader(config("dapo", True), TOKENIZER)
        return await manager.run_single(Sample(length, correct))

    output = asyncio.run(run())
    base_reward = 1. if correct else -1.
    assert float(output["reward_score"]) == pytest.approx(base_reward + penalty)
    assert output["reward_extra_info"]["score"] == base_reward
    assert output["reward_extra_info"]["acc"] == correct
    assert float(output["reward_extra_info"]["overlong_reward"]) == pytest.approx(penalty)
    assert bool(output["reward_extra_info"]["overlong"]) == (penalty < 0)


@pytest.mark.parametrize("source", ["math_dapo", "aime24"])
@pytest.mark.parametrize("correct", [False, True])
@pytest.mark.parametrize("length", [10, 16384])
def test_native_dapo_disabled_has_naive_reward_and_extra_info_parity(reward_loader, source, correct, length):
    async def run():
        old = await reward_loader(config("naive", False), TOKENIZER).run_single(Sample(length, correct, source))
        new = await reward_loader(config("dapo", False), TOKENIZER).run_single(Sample(length, correct, source))
        return old, new

    old, new = asyncio.run(run())
    assert new == old


@pytest.mark.parametrize("correct", [False, True])
@pytest.mark.parametrize("length", [10, 12288])
def test_short_shaping_matches_naive_except_added_logging_fields(reward_loader, correct, length):
    async def run():
        old = await reward_loader(config("naive", False), TOKENIZER).run_single(Sample(length, correct))
        new = await reward_loader(config("dapo", True), TOKENIZER).run_single(Sample(length, correct))
        return old, new

    old, new = asyncio.run(run())
    assert float(new["reward_score"]) == old["reward_score"]
    assert {key: value for key, value in new["reward_extra_info"].items()
            if key not in {"overlong", "overlong_reward"}} == old["reward_extra_info"]


async def shaped_batch(reward_loader):
    patterns = [[0] * 8, [1] * 8, harness.MIXED]
    raw = harness.batch(patterns, length=16384)
    manager = reward_loader(config("dapo", True), TOKENIZER)
    outputs = []
    for row, correct in enumerate(np.asarray(patterns).reshape(-1)):
        length = 16384 if row % 2 else 12288
        outputs.append(await manager.run_single(Sample(length, correct)))
        raw.batch["response_mask"][row, length:] = 0
        raw.batch["attention_mask"][row, length + 1:] = 0
        raw.batch["rm_scores"][row, length - 1] = float(outputs[-1]["reward_score"])
    raw.non_tensor_batch.update({key: np.asarray([item["reward_extra_info"][key] for item in outputs])
                                for key in outputs[0]["reward_extra_info"]})
    raw.meta_info["reward_extra_keys"] = list(outputs[0]["reward_extra_info"])
    return raw


def test_ds_uses_unshaped_acc_despite_shaped_reward_variance_and_releases_raw(reward_loader):
    raw = asyncio.run(shaped_batch(reward_loader))
    rewards = raw.batch["rm_scores"].sum(-1).reshape(3, 8)
    assert rewards[0].unique().tolist() == [-2., -1.]
    assert rewards[1].unique().tolist() == [0., 1.]
    refs = [weakref.ref(raw), weakref.ref(raw.batch["responses"]), weakref.ref(raw.batch["rm_scores"]),
            weakref.ref(raw.non_tensor_batch["acc"]), weakref.ref(raw.non_tensor_batch["overlong_reward"])]
    collector = harness.ds.PromptGroupSampler(1, 8, "acc", 10, 16384)
    accepted = collector.add(raw, metric_values=raw.non_tensor_batch["acc"])
    assert accepted.non_tensor_batch["uid"].tolist() == ["g2"] * 8
    assert accepted.non_tensor_batch["acc"].tolist() == harness.MIXED
    assert len(accepted) == 8 and not collector.batches
    del raw, rewards
    gc.collect()
    assert all(ref() is None for ref in refs)
    metrics = collector.metrics(accepted)
    assert metrics["dynamic_sampling/rejected_all_zero"] == 1
    assert metrics["dynamic_sampling/rejected_all_one"] == 1
    assert metrics["length/raw/hit_16k_rate"] == .5
    assert metrics["length/accepted/hit_16k_rate"] == .5


def test_actual_fit_logs_overlong_scalars_only_for_accepted_groups(monkeypatch, reward_loader):
    raw = asyncio.run(shaped_batch(reward_loader))
    trainer, run, updates, _, logs = harness.fit_harness(monkeypatch, "1.7", [raw], target=1)
    trainer.config.data.max_response_length = 16384
    run()
    assert len(updates) == 1 and len(updates[0]) == 8
    assert logs[0]["overlong/count"] == 4
    assert logs[0]["overlong/rate"] == .5
    assert logs[0]["overlong_reward/mean"] == -.5
    assert logs[0]["overlong_reward/min"] == -1.
    assert logs[0]["length/accepted/max"] == 16384
    assert "old_log_probs" not in updates[0].batch
