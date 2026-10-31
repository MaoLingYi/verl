"""Execute production path expressions and startup control flow without GPU imports."""
import ast
from pathlib import Path
from types import SimpleNamespace as N
import pytest
from tests.trainer.test_r3_taskrunner_startup_on_cpu import execute, ActorDeath

ROOT = Path(__file__).resolve().parents[2]

@pytest.mark.parametrize("path", [None, "/sft/compat-tokenizer"])
def test_dataset_tokenizer_and_processor_path(path):
    tree = ast.parse((ROOT / "verl/trainer/main_ppo.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "TaskRunner")
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "run")
    assignment = next(n for n in fn.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "local_path" for t in n.targets))
    class Config(dict):
        __getattr__ = dict.__getitem__
    model = Config(path="/sft/weights", tokenizer_path=path, use_shm=False)
    ns = {"config": N(actor_rollout_ref=N(model=model)), "copy_to_local": lambda p, **kw: p}
    exec(compile(ast.Module(body=[assignment], type_ignores=[]), "routing", "exec"), ns)
    assert ns["local_path"] == (path or model.path)
    calls = [n for n in ast.walk(fn) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in {"hf_tokenizer", "hf_processor"}]
    assert len(calls) == 2 and all(ast.unparse(c.args[0]) == "local_path" for c in calls)

@pytest.mark.parametrize("path", [None, "/sft/compat-tokenizer"])
def test_sglang_weights_and_tokenizer_separate(path):
    tree = ast.parse((ROOT / "verl/workers/rollout/sglang_rollout/async_sglang_server.py").read_text())
    args = next(n.value for n in ast.walk(tree) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "args" for t in n.targets) and isinstance(n.value, ast.Dict))
    pairs = {k.value: v for k, v in zip(args.keys, args.values) if isinstance(k, ast.Constant)}
    ns = {"self": N(model_config=N(local_path="/sft/weights", local_tokenizer_path=path))}
    def value(k): return eval(compile(ast.Expression(pairs[k]), k, "eval"), ns)
    assert value("model_path") == "/sft/weights"
    assert value("tokenizer_path") == (path or "/sft/weights")
    assert value("enable_memory_saver") is True

@pytest.mark.parametrize("ready_errors", [(), (ActorDeath("startup"),)])
def test_sft_startup_bounded_and_run_never_retries(ready_errors):
    failure = ActorDeath("after optimizer might change state")
    events, options, error = execute(experiment="grpo_r3_qwen3_30b_a3b_sft_b64_n8_16k_pp1_tp2_ep8_etp1", ready_errors=ready_errors, run_error=failure)
    assert error is failure
    assert sum(event == "run" for event, _ in events) == 1
    assert sum(event == "create" for event, _ in events) == 1 + len(ready_errors)
    assert options == [{"max_restarts": 0, "max_task_retries": 0}]


def observation_api():
    tree = ast.parse((ROOT / "verl/trainer/ppo/ray_trainer.py").read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "RayPPOTrainer")
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_observe_sft_initial_rollout")
    ns = {"json": __import__("json")}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "observe", "exec"), ns)
    return ns[fn.name]


def observation_fixture(tmp_path):
    import torch
    experiment = "grpo_r3_qwen3_30b_a3b_sft_b64_n8_16k_pp1_tp2_ep8_etp1"
    parent = tmp_path / (experiment + "_recompute_full_uniform1") / "trial"
    class Config(dict):
        __getattr__ = dict.__getitem__
    owner = N(config=N(trainer=Config(sft_init_observation_dir=str(parent / "initial_rollouts"),
                    experiment_name=experiment, validation_data_dir=str(parent / "validation_generations"))),
              global_steps=1, tokenizer=N(decode=lambda ids, **kw: str(ids)))
    data = {"response_mask": torch.tensor([[1, 1, 0], [1, 1, 1]]),
            "responses": torch.tensor([[7, 8, 0], [9, 10, 11]]), "routed_experts": object()}
    batch = N(batch=data, non_tensor_batch={"uid": ["same", "same"]})
    reward = torch.tensor([[0., 1.], [0., -1.]])
    return owner, batch, reward, parent / "initial_rollouts"


@pytest.mark.parametrize("step", [-1, 0, 1, 2, 3])
def test_observation_step_scope_and_readonly_streaming(tmp_path, step):
    import torch
    import json
    observe = observation_api()
    owner, batch, reward, target = observation_fixture(tmp_path)
    original = {k: v.clone() for k, v in batch.batch.items() if isinstance(v, torch.Tensor)}
    original_reward = reward.clone(); routes = batch.batch["routed_experts"]
    owner.global_steps = step
    result = observe(owner, batch, reward)
    assert result == ("PASS" if step in (1, 2) else "DISABLED")
    if step in (1, 2):
        rows = [json.loads(x) for x in (target / f"step{step}.jsonl").read_text().splitlines()]
        assert [r["length"] for r in rows] == [2, 3]
        assert [r["token_ids"] for r in rows] == [[7, 8], [9, 10, 11]]
        assert [r["cap_hit"] for r in rows] == [False, True]
        assert [r["reward"] for r in rows] == [1., -1.]
        assert json.loads((target / f"step{step}.complete.json").read_text())["status"] == "PASS"
    else: assert not target.exists()
    assert all(torch.equal(batch.batch[k], original[k]) for k in original)
    assert batch.batch["routed_experts"] is routes and torch.equal(reward, original_reward)


@pytest.mark.parametrize("change", ["base", "no_directory", "foreign_directory"])
def test_observation_never_enables_base_or_foreign_directory(tmp_path, change):
    owner, batch, reward, target = observation_fixture(tmp_path)
    if change == "base": owner.config.trainer["experiment_name"] = "base"
    elif change == "no_directory": owner.config.trainer["sft_init_observation_dir"] = None
    else: owner.config.trainer["sft_init_observation_dir"] = str(tmp_path / "foreign")
    assert observation_api()(owner, batch, reward).startswith("DISABLED")
    assert not target.exists()


@pytest.mark.parametrize("failure", ["conflict", "write", "cuda"])
def test_observation_io_failure_is_visible_without_retrying_updates(tmp_path, monkeypatch, capsys, failure):
    owner, batch, reward, target = observation_fixture(tmp_path)
    target.mkdir(parents=True)
    if failure == "conflict": (target / "step1.jsonl").write_text("old")
    elif failure == "write":
        original_open = Path.open
        def denied(path, *args, **kwargs):
            if path.name == "step1.jsonl": raise PermissionError("fixture disk failure")
            return original_open(path, *args, **kwargs)
        monkeypatch.setattr(Path, "open", denied)
    else:
        def cuda_error(*args, **kwargs): raise RuntimeError("CUDA original fatal")
        owner.tokenizer.decode = cuda_error
        with pytest.raises(RuntimeError, match="CUDA original fatal"): observation_api()(owner, batch, reward)
        assert not (target / "step1.complete.json").exists()
        return
    assert observation_api()(owner, batch, reward) == "FAILED_IO"
    assert "SFT_ROLLOUT_OBSERVATION=FAILED_IO" in capsys.readouterr().out
    assert not (target / "step1.complete.json").exists()
    if failure == "conflict": assert (target / "step1.jsonl").read_text() == "old"


def test_observation_never_cpu_copies_whole_responses(tmp_path):
    import torch
    owner, batch, reward, _ = observation_fixture(tmp_path)
    class SampleOnlyTensor(torch.Tensor):
        def cpu(self, *args, **kwargs):
            assert self.ndim == 1, "whole response batch copied"
            return super().cpu(*args, **kwargs)
    batch.batch["responses"] = batch.batch["responses"].as_subclass(SampleOnlyTensor)
    assert observation_api()(owner, batch, reward) == "PASS"
