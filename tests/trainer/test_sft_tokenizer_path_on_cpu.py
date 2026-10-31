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


def test_initial_rollout_observation_is_bounded_and_does_not_touch_routes(tmp_path):
    import json
    import torch
    tree = ast.parse((ROOT / "verl/trainer/ppo/ray_trainer.py").read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "RayPPOTrainer")
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_observe_sft_initial_rollout")
    ns = {"json": json}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "observe", "exec"), ns)
    data = {"response_mask": torch.tensor([[1, 1, 0], [1, 1, 1]]), "responses": torch.tensor([[7, 8, 0], [9, 10, 11]])}
    original = {k: v.clone() for k, v in data.items()}
    batch = N(batch=data, non_tensor_batch={"uid": ["same", "same"]})
    owner = N(config=N(trainer={"sft_init_observation_dir": str(tmp_path)}), global_steps=1, tokenizer=N(decode=lambda ids, **kw: str(ids)))
    ns[fn.name](owner, batch, torch.tensor([[0., 1.], [0., -1.]]))
    rows = [json.loads(x) for x in (tmp_path / "step1.jsonl").read_text().splitlines()]
    assert [r["length"] for r in rows] == [2, 3]
    assert [r["cap_hit"] for r in rows] == [False, True]
    assert [r["reward"] for r in rows] == [1., -1.]
    assert all(torch.equal(data[k], original[k]) for k in data)
    owner.global_steps = 3
    ns[fn.name](owner, batch, None)
    assert not (tmp_path / "step3.jsonl").exists()
