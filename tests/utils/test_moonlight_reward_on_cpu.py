"""Run the production scorer and RewardLoop methods on synthetic CPU batches."""
import ast
import asyncio
from collections import defaultdict
from functools import partial
import inspect
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as NS

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
VERSION = "answer_boxed_final_v2"


@pytest.fixture
def dispatch(monkeypatch):
    # Import the real CPU modules without starting veRL's distributed package initializer.
    for name in ("verl", "verl.utils"):
        package = ModuleType(name)
        package.__path__ = [str(ROOT.joinpath(*name.split(".")))]
        monkeypatch.setitem(sys.modules, name, package)
    from verl.utils.reward_score import default_compute_score
    return default_compute_score


@pytest.mark.parametrize("text,gt,pred,correct", [
    ("Answer: 42", "42", "42", True),
    (r"Hence, the checksum of the encoded message is $\boxed{7}$.", "7", "7", True),
    (r"The required length of the cable is \boxed{\sqrt{3}}.", r"\sqrt{3}", r"\sqrt{3}", True),
    (r"Hence, the candidate length of the cable is \boxed{7}.", "7", "[INVALID]", False),
    (r"Therefore, the checksum if the flag is set is \boxed{7}.", "7", "[INVALID]", False),
    (r"Hence, the value is 8 or the result is \boxed{7}.", "7", "[INVALID]", False),
    ("The two processes can use either resource. Answer: 7", "7", "7", True),
    ("We compared two possible choices and proved uniqueness.\n\nAnswer: 7", "7", "7", True),
    ("This is the least possible positive cost.\nAnswer: 7", "7", "7", True),
    ("Suppose the cost is positive.\nThus, we obtain $\\boxed{7}$.", "7", "[INVALID]", False),
    ("For example.\n\\boxed{7}", "7", "[INVALID]", False),
    ("### Answer:\nThe cost is seven credits.\n\nAnswer: 7", "7", "7", True),
    (r"Therefore, the maximum value is \boxed{42}.", "42", "42", True),
    (r"The required value is \boxed{42}.", "42", "42", True),
    (r"Thus, we obtain $\boxed{42}$.", "42", "42", True),
    ("A calculation ends here. Thus, we obtain **$\\boxed{42}$**. I hope this helps!", "42", "42", True),
    ("Therefore, the maximum value is \\boxed{42}. But the other case is still open.", "42", "[INVALID]", False),
    ("The candidate result is\n\\boxed{42}", "42", "[INVALID]", False),
    ("Possible answers:\n\\boxed{41}\n\\boxed{42}", "42", "[INVALID]", False),
    ("Answer: 41\nThe alternative is\n\\boxed{42}", "42", "[INVALID]", False),
    ("If x is positive,\nthen the result is\n\\boxed{42}", "42", "[INVALID]", False),
    ("For example, the required value is \\boxed{42}.", "42", "[INVALID]", False),
    ("If positive,\nThus, we obtain $\\boxed{42}$.", "42", "[INVALID]", False),
    ("Possible answers:\n\\boxed{41}.\n\\boxed{42}.\nFinal answer: 42", "42", "42", True),
    ("Answer: 41\nCorrection: Answer: 42", "42", "42", True),
    ("\\boxed{41}\n\\boxed{42}", "42", "[INVALID]", False),
    ("\\boxed{42}\n\\boxed{42}", "42", "42", True),
    ("Final answer: 41\nFinal answer: 42", "42", "[INVALID]", False),
    ("Answer: 42\nThanks.", "42", "42", True),
    (r"Answer: \boxed{\{1,2\}}", r"\{1,2\}", r"\{1,2\}", True),
    ("Answer: (0,1]", "(0,1]", "(0,1]", True),
    ("Answer: [0,1)", "[0,1)", "[0,1)", True),
    ("Answer: (1,2,3)", "(1,2,3)", "(1,2,3)", True),
    ("Final answer: \\[\\boxed{\\frac{1}{2} +\n\\sqrt{3}}\\]", r"\frac{1}{2}+\sqrt{3}", r"\frac{1}{2}+\sqrt{3}", True),
    ("We reject the first candidate. The required value is \\boxed{42}.", "42", "42", True),
    ("Answer: (1,2,3]", "(1,2,3]", "[INVALID]", False),
    ("Answer: [0,1", "[0,1]", "[INVALID]", False),
    ("Answer: 4\n2", "42", "[INVALID]", False),
    ("Final answer: $41$", "42", "41", False),
    (r"\boxed{042}", "42", "42", True),
    (r"Answer: $\boxed{42}$", "42", "42", True),
    ("**Final answer: 042**", "42", "42", True),
    ("**Answer:** $42$", "42", "42", True),
    ("\\[\n\\boxed{42}\n\\]", "42", "42", True),
    ("\\boxed{42}\nAnswer: 41", "42", "41", False),
    ("\\boxed{41}\nCorrection: Answer: 42", "42", "42", True),
    ("<think>\\boxed{42}</think>\nAnswer: 41", "42", "41", False),
    ("<think>Answer: 42\n\\boxed{42}", "42", "[INVALID]", False),
    ("Answer: 41 or 42", "42", "[INVALID]", False),
    ("Answer: 42\nFinal answer:", "42", "[INVALID]", False),
    ("Answer: 41. Actually, the final answer is: 42.", "42", "42", True),
    ("Answer: 41\nAnswer: 42", "42", "[INVALID]", False),
    ("Answer: 42\nAnswer: 42", "42", "42", True),
    ("Suppose the answer is:\n\\boxed{42}", "42", "[INVALID]", False),
    ("If the candidate is:\nAnswer: 42", "42", "[INVALID]", False),
    ("Answer: 41\nor\n\\boxed{42}", "42", "[INVALID]", False),
    ("Either:\nAnswer: 41\nAlternatively:\nAnswer: 42", "42", "[INVALID]", False),
    ("An intermediate example:\n\n\\boxed{42}", "42", "[INVALID]", False),
    ("Answer: 4 2", "42", "[INVALID]", False),
    (r"\boxed{\frac{1}{", "1", "[INVALID]", False),
    ("try 42\ntry 42", "42", "[INVALID]", False),
    (r"Answer: \frac12", r"\frac{1}{2}", r"\frac{1}{2}", True),
    (r"\boxed{\sqrt{2}}", r"\sqrt2", r"\sqrt{2}", True),
    ("Final answer: " + "x+" * 175 + "1", "x+" * 175 + "1", "x+" * 175 + "1", True),
])
def test_dispatch_final_submission(dispatch, text, gt, pred, correct):
    assert dispatch("aime24", text, gt, math_dapo_version=VERSION) == {
        "score": 1. if correct else -1., "acc": correct, "pred": pred}


def test_conflict_diagnostics_distinguish_resolution(dispatch):
    from verl.utils.reward_score import answer_boxed_final_v2 as protocol, math_dapo
    norm = math_dapo.normalize_final_answer
    assert protocol.extract("Answer: 41\nCorrection: Answer: 42", norm) == ("42", "answer", True)
    assert protocol.extract("\\boxed{41}\n\\boxed{42}", norm) == (None, "unresolved_conflict", True)
    assert protocol.extract("Possible answers:\n\\boxed{41}\n\\boxed{42}", norm) == (None, "nonfinal_context", True)
    assert protocol.extract(r"Thus, we obtain $\boxed{42}$.", norm) == ("42", "inline_boxed", False)


def method(path, name, ns):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    fn = next(n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name)
    fn.decorator_list = []
    exec(compile(ast.Module([fn], []), path, "exec", flags=__import__("__future__").annotations.compiler_flag), ns)
    return ns[name]


def test_two_formats_flow_through_worker_and_group_advantage(dispatch):
    from omegaconf import OmegaConf
    from verl.utils.reward_score import math_dapo

    assert not dispatch("math_dapo", r"\boxed{42}", "42")["acc"]
    assert math_dapo.compute_score(r"\boxed{42}", "42", strict_box_verify=True)["acc"]
    with pytest.raises(ValueError, match="legacy"):
        dispatch("aime", "Answer: 42", "42", math_dapo_version=VERSION, strict_box_verify=True)
    ns = dict(inspect=inspect, partial=partial, torch=torch, defaultdict=defaultdict)
    reward = "verl/trainer/ppo/reward.py"
    method(reward, "_call_with_kwargs", ns)
    load = method(reward, "get_custom_reward_fn", ns)
    worker_score = method("verl/experimental/reward_loop/reward_loop.py", "compute_score", ns)
    run_single = method("verl/experimental/reward_loop/reward_manager/naive.py", "run_single", ns)
    extract = method(reward, "extract_reward", ns)
    advantage = method("verl/trainer/ppo/core_algos.py", "compute_grpo_outcome_advantage", ns)
    cfg = OmegaConf.create({"reward": {"custom_reward_function": {
        "path": "pkg://verl.utils.reward_score", "name": "default_compute_score",
        "reward_kwargs": {"math_dapo_version": VERSION}}}})
    strings = ["Answer: 42", r"\boxed{42}", "Answer: 41"]
    async def simulate(validate, fn):
        mgr = NS(loop=asyncio.get_running_loop(), tokenizer=NS(decode=lambda ids, **_: strings[ids.item()]),
                 reward_router_address=None, compute_score=fn, is_async_reward_score=False)
        mgr.run_single = partial(run_single, mgr)
        scores = []
        for i in range(3):
            item = NS(batch={"responses": torch.tensor([i, 0]), "attention_mask": torch.tensor([1, 1, 0])},
                      non_tensor_batch={"data_source": "math_dapo", "reward_model": {"ground_truth": "42"},
                                        "extra_info": {"validate": validate}})
            scores.append((await worker_score(NS(config=cfg, reward_manager=mgr), [item]))["reward_score"])
        return scores

    scores = asyncio.run(simulate(False, load(cfg)))
    assert scores == asyncio.run(simulate(True, load(cfg))) == [1., 1., -1.]
    legacy = asyncio.run(simulate(False, dispatch))
    assert legacy == [1., -1., -1.]
    # Execute the AgentLoop's actual placement of scalar rewards at the last response token.
    tree = ast.parse((ROOT / "verl/experimental/agent_loop/agent_loop.py").read_text(encoding="utf-8"))
    place = next(n for n in ast.walk(tree) if isinstance(n, ast.If) and
                 ast.unparse(n.test) == "all((score is not None for score in scores))")
    for values, expected in [(scores, 1), (legacy, -1)]:
        mask = torch.tensor([[1, 0]] * 3)
        batch = {}
        exec(compile(ast.Module([place], []), "agent_loop", "exec"), dict(torch=torch, scores=values, batch=batch,
             prompt_ids=mask[:, :1], attention_mask=torch.cat([mask[:, :1], mask], 1), response_mask=mask))
        tensor, _ = extract(NS(batch=batch, meta_info={}))
        adv, _ = advantage(tensor, mask, ["p"] * 3)
        assert adv[1, 0].sign().item() == expected
        assert not adv[:, 1].any()
