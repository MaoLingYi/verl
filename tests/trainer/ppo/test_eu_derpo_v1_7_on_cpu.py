"""CPU production-control-flow tests; TensorDict/Ray are replaced, not launched."""

import ast
import contextlib
import gc
import importlib.util
import sys
import types
import weakref
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf


ROOT = Path(__file__).parents[3]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ds = load("v17_sampling_test", "verl/trainer/ppo/dynamic_sampling.py")
eu = load("v17_math_test", "verl/trainer/ppo/eu_derpo.py")


class TensorDict(dict):
    """Only the CPU TensorDict operations used by the real DataProto methods."""

    def __init__(self, source, batch_size=None, device=None):
        super().__init__(source)
        self.batch_size = batch_size or (next(iter(source.values())).shape[0],)
        self.device = torch.device("cpu")

    def __len__(self):
        return self.batch_size[0]

    def chunk(self, chunks, dim=0):
        return [TensorDict({key: value.chunk(chunks, dim)[i] for key, value in self.items()}) for i in range(chunks)]

    def select(self, *keys):
        return TensorDict({key: self[key] for key in keys})

    @classmethod
    def __torch_function__(cls, function, types_, args=(), kwargs=None):
        if function is torch.cat:
            batches = args[0]
            return cls({key: torch.cat([batch[key] for batch in batches], dim=0) for key in batches[0]})
        return NotImplemented


class Proto:
    def __init__(self, batch, non_tensor_batch=None, meta_info=None):
        self.batch = batch
        self.non_tensor_batch = non_tensor_batch or {}
        self.meta_info = meta_info or {}

    def __len__(self):
        return self.batch.batch_size[0]

    def is_padding_enabled(self):
        return False

    @classmethod
    def from_single_dict(cls, batch):
        return batch

    def union(self, other):
        self.batch.update(other.batch)
        self.non_tensor_batch.update(other.non_tensor_batch)
        self.meta_info.update(other.meta_info)
        if hasattr(other, "correctness"):
            self.correctness = other.correctness
        return self

    def make_iterator(self, mini_batch_size, epochs, seed, dataloader_kwargs):
        assert len(self) == mini_batch_size and epochs == 1
        assert not dataloader_kwargs["shuffle"]
        return iter([self])


# Exercise exact production row selection, concat, repeat and DP chunk methods.
tree = ast.parse((ROOT / "verl/protocol.py").read_text(encoding="utf-8"))
protocol = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "DataProto")
methods = [node for node in protocol.body if isinstance(node, ast.FunctionDef) and node.name in {
    "select", "select_idxs", "concat", "repeat", "chunk",
}]
namespace = {
    "np": np, "torch": torch, "TensorDict": TensorDict,
    "list_of_dict_to_dict_of_list": lambda list_of_dict: {
        key: [item[key] for item in list_of_dict] for key in list_of_dict[0]
    },
}
future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
shell = ast.ClassDef(name="ProductionProtoMethods", bases=[], keywords=[], body=methods, decorator_list=[])
exec(compile(ast.fix_missing_locations(ast.Module(body=[future, shell], type_ignores=[])), "protocol", "exec"), namespace)
for name in ("select", "select_idxs", "concat", "repeat", "chunk"):
    setattr(Proto, name, namespace["ProductionProtoMethods"].__dict__[name])


def batch(patterns, prefix="g", length=8):
    count = len(patterns) * 8
    return Proto(
        TensorDict({
            "responses": torch.arange(count).unsqueeze(1).expand(count, length).clone(),
            "response_mask": torch.ones(count, length, dtype=torch.long),
            "attention_mask": torch.ones(count, length + 1, dtype=torch.long),
            "rollout_log_probs": torch.full((count, length), -2.),
            "rm_scores": torch.zeros(count, length),
        }),
        {"uid": np.repeat([f"{prefix}{i}" for i in range(len(patterns))], 8),
         "acc": np.asarray(patterns).reshape(-1),
         "payload": np.array([{"sample": i} for i in range(count)], dtype=object)},
        {"temperature": 1., "reward_extra_keys": ["acc"], "global_token_num": [length + 1] * count},
    )


MIXED = [0, 0, 1, 0, 1, 0, 1, 0]


def test_whole_groups_uid_order_and_rejected_storage_release():
    raw = batch([[0] * 8, [1] * 8, MIXED, [1, 1, 1, 0, 1, 1, 1, 1]])
    refs = [weakref.ref(raw), weakref.ref(raw.batch["responses"]), weakref.ref(raw.non_tensor_batch["acc"])]
    collector = ds.PromptGroupSampler(2, 8, "acc", 10, 8192)
    accepted = collector.add(raw, metric_values=raw.non_tensor_batch["acc"])
    assert accepted.non_tensor_batch["uid"].tolist() == ["g2"] * 8 + ["g3"] * 8
    assert accepted.batch["responses"][:, 0].tolist() == list(range(16, 32))
    assert "eu_derpo_sample_ids" not in accepted.batch
    assert "routed_experts" not in accepted.batch
    assert not collector.batches
    del raw
    gc.collect()
    assert all(ref() is None for ref in refs)
    metrics = collector.metrics(accepted)
    assert metrics["dynamic_sampling/rejected_all_zero"] == 1
    assert metrics["dynamic_sampling/rejected_all_one"] == 1
    assert metrics["dynamic_sampling/accepted_groups"] == 2


def test_refill_40_50_50_takes_first_128_groups_and_dp4_locality():
    collector = ds.PromptGroupSampler(128, 8, "acc", 3, 8192)
    result = None
    for round_, count in enumerate((40, 50, 50)):
        raw = batch([MIXED] * count + [[0] * 8], prefix=f"r{round_}:")
        ref = weakref.ref(raw.batch["responses"])
        result = collector.add(raw, metric_values=raw.non_tensor_batch["acc"])
        del raw
        gc.collect()
        assert ref() is None
    assert len(result) == 1024
    expected = [f"r0:{i}" for i in range(40)] + [f"r1:{i}" for i in range(50)] + [f"r2:{i}" for i in range(38)]
    assert result.non_tensor_batch["uid"].tolist() == np.repeat(expected, 8).tolist()
    mapping = {uid: i for i, uid in enumerate(expected)}
    result.batch["eu_derpo_prompt_group"] = torch.tensor([mapping[uid] for uid in result.non_tensor_batch["uid"]])
    result.batch["eu_derpo_sample_ids"] = torch.arange(1024)
    for rank, shard in enumerate(result.chunk(4)):
        counts = eu.validate_prompt_groups(shard.batch["eu_derpo_prompt_group"], 8)
        assert counts.tolist() == [8] * 32
        assert len(shard) == 256
        assert shard.batch["eu_derpo_sample_ids"].tolist() == list(range(rank * 256, (rank + 1) * 256))
    assert eu.audit_optimizer_minibatches(1024 // 4, 1, 128 * 8 // 4) == 1
    assert collector.metrics(result)["dynamic_sampling/surplus_groups"] == 12


def test_max_rounds_failure_releases_accepted_accumulator():
    collector = ds.PromptGroupSampler(128, 8, "acc", 2, 8192)
    raw = batch([MIXED] * 40)
    assert collector.add(raw, metric_values=raw.non_tensor_batch["acc"]) is None
    with pytest.raises(RuntimeError, match="accepted 90/128"):
        raw = batch([MIXED] * 50, prefix="next")
        collector.add(raw, metric_values=raw.non_tensor_batch["acc"])
    assert not collector.batches


@pytest.mark.parametrize("rows", [7, 5])
def test_partial_group_is_rejected(rows):
    raw = batch([MIXED]).select_idxs(np.arange(rows))
    with pytest.raises(RuntimeError, match="complete prompt groups"):
        ds.PromptGroupSampler(1, 8, "acc", 10, 8192).add(raw, metric_values=raw.non_tensor_batch["acc"])


def test_raw_and_accepted_length_quantiles_are_separate():
    raw = batch([[0] * 8, MIXED], length=8192)
    raw.batch["response_mask"][8:, 1024:] = 0
    collector = ds.PromptGroupSampler(1, 8, "acc", 10, 8192)
    accepted = collector.add(raw, metric_values=raw.non_tensor_batch["acc"])
    metrics = collector.metrics(accepted)
    assert metrics["length/raw/max"] == 8192
    assert metrics["length/raw/hit_8k_rate"] == .5
    for name in ("mean", "p50", "p90", "p95", "max"):
        assert metrics[f"length/accepted/{name}"] == 1024
    assert metrics["length/accepted/hit_8k_rate"] == 0


def test_behavior_anchor_and_current_route_math_are_unchanged():
    current = torch.tensor([[-2., -3.]], requires_grad=True)
    behavior = current.detach() - .01
    routes = torch.tensor([[[[0, 127]], [[1, 127]]]], dtype=torch.uint8)
    mask = torch.ones_like(current, dtype=torch.bool)
    stats = eu.cluster_statistics(current.detach(), behavior, torch.ones_like(current), mask, routes, 128, .012)
    old_stats = eu.cluster_statistics(
        current.detach(), behavior, torch.ones_like(current), mask, routes, 128, .012,
        aligned_old_logp=behavior - .1, optimization_anchor="rollout",
    )
    new_coefficient, objective = eu.edppo_token_coefficients(stats, routes, mask)
    old_coefficient, old_objective = eu.edppo_token_coefficients(old_stats, routes, mask)
    torch.testing.assert_close(new_coefficient, old_coefficient, rtol=0, atol=0)
    torch.testing.assert_close(objective, old_objective, rtol=0, atol=0)
    grad = torch.autograd.grad(-(new_coefficient * current).sum(), current)[0]
    torch.testing.assert_close(grad, -new_coefficient)
    changed = eu.cluster_statistics(current.detach(), behavior - .5, torch.ones_like(current), mask, routes, 128, .012)
    assert not torch.equal(stats.mask, changed.mask)
    assert stats.divergence_engine is None and stats.delta_engine is None
    with pytest.raises(RuntimeError, match="natural route mismatch"):
        eu.assert_same_routes(routes, routes.flip(1), mask)


def test_utility_normalization_is_same_prompt_only():
    groups = torch.arange(2).repeat_interleave(8)
    utility = torch.cat((torch.arange(8.), torch.arange(8.) + 100.)).reshape(16, 1, 1)
    normalized, _ = eu.normalize_group_utility(utility, torch.ones_like(utility), groups, 1e-6, 2, 1e-6)
    torch.testing.assert_close(normalized[:8], normalized[8:])


def test_real_actor_minibatch_planner_needs_no_old_or_rollout_routes():
    source = ast.parse((ROOT / "verl/workers/actor/megatron_actor.py").read_text(encoding="utf-8"))
    actor_cls = next(node for node in source.body if isinstance(node, ast.ClassDef) and node.name == "MegatronPPOActor")
    planner = next(node for node in actor_cls.body if isinstance(node, ast.FunctionDef) and node.name == "make_minibatch_iterator")
    ns = {
        "audit_optimizer_minibatches": eu.audit_optimizer_minibatches,
        "validate_prompt_groups": eu.validate_prompt_groups,
    }
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future, planner], type_ignores=[])), "actor.planner", "exec"), ns)
    actor = types.SimpleNamespace(
        eu_derpo_observer=object(), router_shift_observer=None,
        config=OmegaConf.create({
            "eu_derpo": {"version": "1.7"}, "use_kl_loss": False, "router_replay": {"mode": "disabled"},
            "ppo_epochs": 1, "ppo_mini_batch_size": 128 * 8 // 4, "shuffle": False, "data_loader_seed": 42,
        }),
    )
    for rank in range(4):
        data = batch([MIXED] * 32)
        data.batch.update({
            "input_ids": data.batch["responses"], "position_ids": torch.zeros(256, 8, dtype=torch.long),
            "advantages": torch.ones(256, 8), "eu_derpo_sample_ids": torch.arange(rank * 256, (rank + 1) * 256),
            "eu_derpo_prompt_group": torch.arange(rank * 32, (rank + 1) * 32).repeat_interleave(8),
        })
        data.meta_info["eu_derpo_rollout_n"] = 8
        minibatches = list(ns["make_minibatch_iterator"](actor, data))
        assert len(minibatches) == 1 and len(minibatches[0]) == 256
        assert "old_log_probs" not in minibatches[0].batch
        assert "routed_experts" not in minibatches[0].batch
        assert minibatches[0].meta_info["eu_derpo_expected_optimizer_minibatches"] == 1


def fit_harness(monkeypatch, version, rounds, *, steps=1, target=128, batches_per_epoch=None, epochs=10):
    """Run the actual fit AST with CPU rollout/actor spies, never import Ray/Megatron."""
    log = []
    tracking = types.ModuleType("verl.utils.tracking")
    tracking.Tracking = lambda **kwargs: types.SimpleNamespace(log=lambda data, step: log.append(data))
    monkeypatch.setitem(sys.modules, "verl.utils.tracking", tracking)

    @contextlib.contextmanager
    def timer(name, timing, **kwargs):
        yield
        timing[name] = timing.get(name, 0.) + 1.

    consumed = []
    seeds = []
    syncs = []

    class Loader:
        sampler = types.SimpleNamespace(set_epoch=lambda epoch: seeds.append(epoch))
        traversals = 0
        resume_offset = 0

        def __len__(self):
            return batches_per_epoch or len(rounds)

        def __iter__(self):
            self.traversals += 1
            start, self.resume_offset = self.resume_offset, 0
            for index in range(start, len(self)):
                specification = rounds[index]
                count = target if isinstance(specification, int) else len(specification) // 8
                ids = list(range(index * count, (index + 1) * count))
                consumed.append(ids)
                yield Proto(TensorDict({"prompts": torch.tensor(ids).unsqueeze(1)}),
                            {"multi_modal_inputs": np.array([{}] * count, dtype=object)})

    config = OmegaConf.create({
        "trainer": {"project_name": "cpu", "experiment_name": "v17", "logger": [], "total_epochs": epochs,
                    "val_before_train": False, "balance_batch": False, "critic_warmup": 0, "test_freq": -1,
                    "esi_redundant_time": 0, "save_freq": -1},
        "actor_rollout_ref": {"actor": {"eu_derpo": {"enabled": True, "version": version}, "shuffle": False},
                              "rollout": {"n": 8, "temperature": 1., "skip_rollout": False}},
        "algorithm": {"adv_estimator": "grpo", "use_kl_in_reward": False, "gamma": 1., "lam": 1.,
                      "filter_groups": {"enable": version == "1.7", "metric": "acc", "max_num_gen_batches": 3}},
        "data": {"train_batch_size": target, "max_response_length": 8},
        "global_profiler": {"steps": None, "profile_continuous_steps": False},
    })
    advantages_seen = []
    updates = []

    def advantage(data, **kwargs):
        advantages_seen.append(data.non_tensor_batch["uid"].copy())
        assert len(data) == target * 8
        assert eu.validate_prompt_groups(data.batch["eu_derpo_prompt_group"], 8).tolist() == [8] * target
        return data

    def update(data):
        gc.collect()
        assert all(ref() is None for ref in trainer.raw_payload_refs)
        updates.append(data)
        return types.SimpleNamespace(meta_info={"metrics": {}})

    outputs = iter(rounds)

    def generate(prompts):
        gc.collect()
        assert all(ref() is None for ref in trainer.raw_payload_refs)
        generated = next(outputs)
        if isinstance(generated, int):
            generated = batch([MIXED] * generated + [[0] * 8] * (target - generated))
            trainer.raw_payload_refs.extend([
                weakref.ref(generated),
                *[weakref.ref(value) for value in generated.batch.values()],
                *[weakref.ref(value) for value in generated.non_tensor_batch.values()],
            ])
        # Reward extraction owns correctness; the raw production batch has no acc field.
        generated.correctness = generated.non_tensor_batch.pop("acc")
        generated.non_tensor_batch["uid"] = prompts.non_tensor_batch["uid"]
        generated.meta_info["timing"] = {"generate_sequences": 1.}
        return generated

    trainer = types.SimpleNamespace(
        config=config, total_training_steps=steps, train_dataloader=Loader(), train_dataset=object(),
        actor_rollout_wg=types.SimpleNamespace(), use_rm=False, use_critic=False, use_reference_policy=False,
        checkpoint_manager=types.SimpleNamespace(update_weights=lambda step: syncs.append((step, len(updates))), sleep_replicas=lambda: None),
        async_rollout_manager=types.SimpleNamespace(generate_sequences=generate),
        resource_pool_manager=types.SimpleNamespace(get_n_gpus=lambda: 8),
        _load_checkpoint=lambda: None, _start_profiling=lambda *args: None, _stop_profiling=lambda *args: None,
        _get_gen_batch=lambda data: data, _should_save_checkpoint_this_step=lambda **kwargs: False,
        _update_actor=update,
        raw_payload_refs=[], consumed=consumed, seeds=seeds, syncs=syncs,
    )
    trainer._compute_old_log_prob = lambda data: pytest.fail("old aligned recompute was called")
    def extract(data):
        if hasattr(data, "correctness"):
            assert "acc" not in data.non_tensor_batch
        extras = {
            key: data.correctness if key == "acc" and hasattr(data, "correctness") else data.non_tensor_batch[key]
            for key in data.meta_info["reward_extra_keys"]
        }
        return data.batch["rm_scores"], extras

    source = ast.parse((ROOT / "verl/trainer/ppo/ray_trainer.py").read_text(encoding="utf-8"))
    cls = next(node for node in source.body if isinstance(node, ast.ClassDef) and node.name == "RayPPOTrainer")
    fit = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "fit")
    ns = {
        "torch": torch, "np": np, "uuid": __import__("uuid"), "DataProto": Proto,
        "PromptGroupSampler": ds.PromptGroupSampler, "marked_timer": timer,
        "tqdm": lambda **kwargs: types.SimpleNamespace(update=lambda *args: None, close=lambda: None),
        "AdvantageEstimator": types.SimpleNamespace(REMAX="remax", GDPO="gdpo"),
        "extract_reward": extract, "compute_advantage": advantage,
        "should_save_ckpt_esi": lambda **kwargs: False, "reduce_metrics": lambda data: data,
        "compute_data_metrics": lambda **kwargs: {}, "compute_timing_metrics": lambda **kwargs: {"timing/" + key: value for key, value in kwargs["timing_raw"].items()},
        "compute_throughout_metrics": lambda **kwargs: {}, "compute_variance_proxy_metrics": lambda **kwargs: {},
        "AbstractCurriculumSampler": type("UnusedSampler", (), {}), "pprint": lambda *args: None,
    }
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future, fit], type_ignores=[])), "ray_trainer.fit", "exec"), ns)
    return trainer, lambda: ns["fit"](trainer), updates, advantages_seen, log


def test_real_fit_refills_before_ids_advantage_actor_and_bypasses_old(monkeypatch):
    rounds = [40, 50, 50]
    trainer, run, updates, seen, logs = fit_harness(monkeypatch, "1.7", rounds)
    run()
    assert len(updates) == len(seen) == 1
    assert "old_log_probs" not in updates[0].batch
    assert "current_log_probs" not in updates[0].batch
    assert "routed_experts" not in updates[0].batch
    assert logs[0]["dynamic_sampling/accepted_groups"] == 128
    assert logs[0]["dynamic_sampling/generation_rounds"] == 3
    assert trainer.global_steps == 2


def test_real_fit_stops_at_round_budget_before_advantage_or_update(monkeypatch):
    trainer, run, updates, seen, _ = fit_harness(monkeypatch, "1.7", [40, 50])
    trainer.config.algorithm.filter_groups.max_num_gen_batches = 2
    with pytest.raises(RuntimeError, match="exhausted 2 generation rounds"):
        run()
    assert not updates and not seen


def test_v16_fit_still_calls_old_recompute(monkeypatch):
    trainer, run, updates, seen, _ = fit_harness(monkeypatch, "1.6", [batch([MIXED] * 128)])
    trainer._compute_old_log_prob = lambda data: (_ for _ in ()).throw(RuntimeError("V16 old replay reached"))
    # Supply the fields read just before the existing old forward.
    trainer.config.actor_rollout_ref.actor.router_shift_diagnostics = {"enabled": False}
    trainer.config.actor_rollout_ref.actor.router_shift_weighting = {"enabled": False}
    with pytest.raises(RuntimeError, match="V16 old replay reached"):
        run()
    assert not updates and not seen


def test_real_fit_iterator_continues_after_optimizer_update(monkeypatch):
    raw = [batch([MIXED] * 4, prefix=f"b{i}:") for i in range(4)]
    trainer, run, updates, _, logs = fit_harness(monkeypatch, "1.7", raw, steps=2, target=8)
    run()
    assert trainer.consumed == [list(range(i, i + 4)) for i in (0, 4, 8, 12)]
    assert trainer.train_dataloader.traversals == 1
    assert trainer.seeds == [0]
    assert len(updates) == 2
    assert [entry["dynamic_sampling/generation_rounds"] for entry in logs] == [2, 2]
    assert [entry["timing/generate_sequences"] for entry in logs] == [2., 2.]
    assert [entry["timing/step"] for entry in logs] == [2., 2.]
    assert trainer.syncs == [(0, 0), (1, 1), (2, 2)]


def test_real_fit_partial_collector_and_rounds_cross_epoch(monkeypatch):
    trainer, run, updates, _, logs = fit_harness(monkeypatch, "1.7", [40, 50, 50], batches_per_epoch=2)
    run()
    assert len(updates) == 1 and len(updates[0]) == 1024
    assert trainer.train_dataloader.traversals == 2
    assert trainer.seeds == [0, 1]
    assert logs[0]["dynamic_sampling/generation_rounds"] == 3
    assert logs[0]["dynamic_sampling/accepted_generated_groups"] == 140
    assert logs[0]["dynamic_sampling/raw_data_epoch_start"] == 0
    assert logs[0]["dynamic_sampling/raw_data_epoch_end"] == 1
    assert logs[0]["timing/generate_sequences"] == 3.
    assert logs[0]["timing/step"] == 3.
    assert trainer.syncs == [(0, 0), (1, 1)]
    ids = updates[0].batch["prompts"][:, 0].reshape(128, 8)[:, 0].tolist()
    assert ids == list(range(40)) + list(range(128, 178)) + list(range(38))


def test_real_fit_round_budget_is_not_reset_at_epoch_boundary(monkeypatch):
    trainer, run, updates, _, _ = fit_harness(monkeypatch, "1.7", [40, 50], batches_per_epoch=1)
    trainer.config.algorithm.filter_groups.max_num_gen_batches = 2
    with pytest.raises(RuntimeError, match="exhausted 2 generation rounds: accepted 90/128"):
        run()
    assert trainer.seeds == [0, 1]
    assert not updates
    assert trainer.syncs == [(0, 0)]


def test_real_fit_exact_five_updates_across_three_raw_epochs(monkeypatch):
    trainer, run, updates, _, logs = fit_harness(
        monkeypatch, "1.7", [128] * 5, steps=5, batches_per_epoch=2,
    )
    run()
    assert len(updates) == len(logs) == 5
    assert [entry["training/epoch"] for entry in logs] == [0, 0, 1, 1, 2]
    assert [entry["training/global_step"] for entry in logs] == [1, 2, 3, 4, 5]
    assert trainer.train_dataloader.traversals == 3
    assert trainer.syncs == [(0, 0)] + [(i, i) for i in range(1, 6)]


def test_real_fit_reward_output_acc_without_prefilled_batch_acc(monkeypatch):
    raw = batch([[0] * 8, [1] * 8, MIXED, MIXED])
    trainer, run, updates, _, logs = fit_harness(monkeypatch, "1.7", [raw], target=2)
    run()
    assert len(updates[0]) == 16
    assert logs[0]["dynamic_sampling/rejected_all_zero"] == 1
    assert logs[0]["dynamic_sampling/rejected_all_one"] == 1
    assert updates[0].non_tensor_batch["acc"].tolist() == MIXED * 2


def test_real_fit_missing_reward_metric_has_no_batch_or_reward_fallback(monkeypatch):
    trainer, run, updates, _, _ = fit_harness(monkeypatch, "1.7", [128])
    trainer.config.algorithm.filter_groups.metric = "missing_correctness"
    with pytest.raises(ValueError, match="reward metric 'missing_correctness' is missing"):
        run()
    assert not updates


def test_real_fit_epoch_safety_ceiling_reports_incomplete_updates(monkeypatch):
    trainer, run, updates, _, _ = fit_harness(monkeypatch, "1.7", [128], steps=2, epochs=1)
    with pytest.raises(RuntimeError, match="completed 1/2 optimizer updates"):
        run()
    assert len(updates) == 1


def test_v16_does_not_enter_v17_sampler_even_when_filter_config_enabled(monkeypatch):
    trainer, run, updates, _, _ = fit_harness(monkeypatch, "1.6", [batch([MIXED] * 128)])
    trainer.config.algorithm.filter_groups.enable = True
    trainer.config.algorithm.filter_groups.metric = "missing_correctness"
    trainer.config.actor_rollout_ref.actor.router_shift_diagnostics = {"enabled": False}
    trainer.config.actor_rollout_ref.actor.router_shift_weighting = {"enabled": False}
    trainer._compute_old_log_prob = lambda data: (_ for _ in ()).throw(RuntimeError("V16 old replay reached"))
    with pytest.raises(RuntimeError, match="V16 old replay reached"):
        run()
    assert not updates and trainer.seeds == []


@pytest.mark.parametrize("version,enabled,algorithm", [
    ("1.4", True, "grpo"), ("1.6", True, "grpo"),
    ("1.7", False, "grpo"), ("1.7", False, "gspo"), ("1.7", False, "gmpo"),
])
def test_sampler_is_scoped_to_enabled_eu_v17(monkeypatch, version, enabled, algorithm):
    trainer, run, _, _, _ = fit_harness(monkeypatch, version, [128])
    trainer.config.actor_rollout_ref.actor.eu_derpo.enabled = enabled
    trainer.config.algorithm.adv_estimator = algorithm
    trainer.config.algorithm.filter_groups.enable = True
    trainer.config.algorithm.filter_groups.metric = "missing_correctness"
    trainer.config.actor_rollout_ref.actor.router_shift_diagnostics = {"enabled": False}
    trainer.config.actor_rollout_ref.actor.router_shift_weighting = {"enabled": False}
    trainer._compute_old_log_prob = lambda data: (_ for _ in ()).throw(RuntimeError("legacy old path reached"))
    with pytest.raises(RuntimeError, match="legacy old path reached"):
        run()
    assert trainer.seeds == []


def test_real_fit_restored_loader_position_is_not_reseeded(monkeypatch):
    trainer, run, updates, _, logs = fit_harness(
        monkeypatch, "1.7", [128, 128], steps=3, batches_per_epoch=2,
    )

    def restore():
        trainer.global_steps = 1
        trainer.raw_data_epoch = 3
        trainer._raw_dataloader_resumed = True
        trainer.train_dataloader.resume_offset = 1

    trainer._load_checkpoint = restore
    run()
    assert len(updates) == 2
    assert trainer.consumed == [list(range(128, 256)), list(range(128))]
    assert trainer.seeds == [4]
    assert [entry["training/epoch"] for entry in logs] == [3, 4]
    assert [entry["training/global_step"] for entry in logs] == [2, 3]


def test_explicit_resume_250_runs_updates_251_through_absolute_target_370(monkeypatch):
    trainer, run, updates, _, logs = fit_harness(
        monkeypatch, "1.7", [1] * 120, steps=370, target=1, batches_per_epoch=1, epochs=130,
    )

    def restore():
        trainer.global_steps = 250
        trainer.raw_data_epoch = 0
        trainer._raw_dataloader_resumed = True

    trainer._load_checkpoint = restore
    run()
    assert len(updates) == 120
    assert [entry["training/global_step"] for entry in logs] == list(range(251, 371))


@pytest.mark.parametrize("version", ["1.6", "1.7"])
def test_production_dataloader_checkpoint_blocks_preserve_state_and_raw_epoch(version):
    source = ast.parse((ROOT / "verl/trainer/ppo/ray_trainer.py").read_text(encoding="utf-8"))
    cls = next(node for node in source.body if isinstance(node, ast.ClassDef) and node.name == "RayPPOTrainer")
    save = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "_save_checkpoint")
    start = next(i for i, node in enumerate(save.body) if isinstance(node, ast.Assign)
                 and any(isinstance(target, ast.Name) and target.id == "dataloader_state_dict" for target in node.targets))
    load = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "_load_checkpoint")
    block = next(node for node in load.body if isinstance(node, ast.If) and isinstance(node.test, ast.Call)
                 and ast.unparse(node.test.func) == "os.path.exists")
    saved = {}
    restored = {}
    trainer = types.SimpleNamespace(
        config=OmegaConf.create({"actor_rollout_ref": {"actor": {"eu_derpo": {"enabled": True, "version": version}}}}),
        raw_data_epoch=7,
        train_dataloader=types.SimpleNamespace(state_dict=lambda: {"iterator_position": 2},
                                             load_state_dict=lambda state: restored.update(state)),
    )
    namespace = {"self": trainer, "dataloader_local_path": "unused",
                 "torch": types.SimpleNamespace(save=lambda state, path: saved.update(state),
                                                load=lambda *args, **kwargs: dict(saved))}
    exec(compile(ast.Module(body=save.body[start:start + 3], type_ignores=[]), "save.data", "exec"), namespace)
    if version == "1.7":
        assert saved["eu_derpo_raw_data_epoch"] == 7
        trainer.raw_data_epoch = 0
    else:
        assert saved == {"iterator_position": 2}
    exec(compile(ast.Module(body=block.body, type_ignores=[]), "load.data", "exec"), namespace)
    assert restored == {"iterator_position": 2}
    if version == "1.7":
        assert trainer.raw_data_epoch == 7 and trainer._raw_dataloader_resumed


def test_real_fit_ds_refill_keeps_replicas_awake_until_collector_ready(monkeypatch):
    trainer, run, updates, _, _ = fit_harness(monkeypatch, "1.7", [40, 50, 50])
    events = []
    awake = False
    collected = False
    original_generate = trainer.async_rollout_manager.generate_sequences
    original_update = trainer._update_actor
    original_weights = trainer.checkpoint_manager.update_weights
    original_add = ds.PromptGroupSampler.add

    def update_weights(step):
        nonlocal awake
        original_weights(step)
        awake = True
        events.append(f"update_weights:{step}")

    def generate(prompts):
        assert awake
        events.append("generate")
        return original_generate(prompts)

    def add(collector, data, *, metric_values):
        nonlocal collected
        accepted = original_add(collector, data, metric_values=metric_values)
        assert awake and "sleep" not in events
        collected = accepted is not None
        if collected:
            assert collector.accepted_groups == 128 and len(accepted) == 1024
        return accepted

    def sleep():
        nonlocal awake
        assert awake and collected and not updates
        assert events.count("generate") == 3
        awake = False
        events.append("sleep")

    def actor_update(data):
        assert not awake
        events.append("actor_update")
        return original_update(data)

    monkeypatch.setattr(ds.PromptGroupSampler, "add", add)
    trainer.async_rollout_manager.generate_sequences = generate
    trainer.checkpoint_manager.sleep_replicas = sleep
    trainer.checkpoint_manager.update_weights = update_weights
    trainer._update_actor = actor_update
    run()
    assert len(updates) == 1 and awake
    assert events == [
        "update_weights:0", "generate", "generate", "generate",
        "sleep", "actor_update", "update_weights:1",
    ]
