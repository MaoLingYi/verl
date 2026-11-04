"""Execute real RewardLoop initialization methods without Ray/GPU imports."""
import ast
import asyncio
from pathlib import Path
from types import SimpleNamespace as N
import pytest
from omegaconf import DictConfig, OmegaConf

ROOT = Path(__file__).resolve().parents[2]


def worker_class(tokenizer, copied):
    tree = ast.parse((ROOT/'verl/experimental/reward_loop/reward_loop.py').read_text(encoding='utf-8'))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'RewardLoopWorker')
    cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in ('__init__', '_init_reward_fn')]
    def copy(path):
        copied.append(path)
        return path
    ns = dict(DictConfig=DictConfig, copy_to_local=copy, hf_tokenizer=tokenizer,
              load_reward_manager=lambda config, tok, **kw: N(tokenizer=tok))
    exec(compile(ast.Module(body=[cls], type_ignores=[]), 'real-reward-loop', 'exec'), ns)
    return ns['RewardLoopWorker']


@pytest.mark.parametrize('path', ['/sft/compat', None, ''])
def test_real_worker_constructor_selects_explicit_tokenizer_or_base_fallback(path):
    copied = []
    expected = path or '/base/hf'
    token = N(decode=lambda ids, **kw: 'Answer: 2')
    def load(selected, **kwargs):
        assert selected == expected
        assert kwargs == {'trust_remote_code': True}
        return token
    config = OmegaConf.create({'actor_rollout_ref': {'model': {'path': '/base/hf', 'tokenizer_path': path}},
                               'reward': {'reward_model': {'enable': False}}})
    worker = worker_class(load, copied)(config)
    assert copied == [expected]
    assert worker.input_tokenizer is token and worker.reward_manager.tokenizer is token
    assert config.actor_rollout_ref.model.path == '/base/hf'


def test_sft_bad_config_and_initialization_error_are_not_swallowed():
    for path in (None, '/invalid/compat'):
        copied = []
        def load(selected, **kwargs):
            if selected == '/sft/raw':
                special_tokens = ['original metadata']
                special_tokens.keys()  # Original 4.57.1 failure boundary, not a real Transformers test.
            raise FileNotFoundError(selected)
        config = OmegaConf.create({'actor_rollout_ref': {'model': {'path': '/sft/raw', 'tokenizer_path': path}},
                                   'reward': {'reward_model': {'enable': False}}})
        with pytest.raises(AttributeError if path is None else FileNotFoundError):
            worker_class(load, copied)(config)
        assert copied == [path or '/sft/raw']


def test_real_naive_reward_decode_text_reaches_unchanged_scorer():
    import torch
    tree = ast.parse((ROOT/'verl/experimental/reward_loop/reward_manager/naive.py').read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'NaiveRewardManager')
    method = next(n for n in cls.body if isinstance(n, ast.AsyncFunctionDef) and n.name == 'run_single')
    ns = {'DataProto': object}
    exec(compile(ast.Module(body=[method], type_ignores=[]), 'real-naive-reward', 'exec'), ns)
    seen = []
    def decode(ids, **kw):
        assert ids.tolist() == [7, 8] and kw == {'skip_special_tokens': True}
        return '<think>work</think>\nAnswer: 2'
    def score(**kw): seen.append(kw['solution_str']); return 1.0
    item = N(batch={'responses': torch.tensor([7, 8, 0]), 'attention_mask': torch.tensor([1, 1, 1, 0])},
             non_tensor_batch={'data_source': 'fixture', 'reward_model': {'ground_truth': '2'}})
    class Batch:
        def __len__(self): return 1
        def __getitem__(self, index): return item
    async def run():
        owner = N(loop=asyncio.get_running_loop(), tokenizer=N(decode=decode), compute_score=score,
                  is_async_reward_score=False, reward_router_address=None)
        return await ns['run_single'](owner, Batch())
    assert asyncio.run(run())['reward_score'] == 1.0
    assert seen == ['<think>work</think>\nAnswer: 2']


def test_real_transformers4571_compatible_worker_when_assets_available():
    import os
    transformers = pytest.importorskip('transformers')
    if transformers.__version__ != '4.57.1': pytest.skip('requires frozen Transformers 4.57.1')
    path = os.environ.get('SFT_COMPAT_TOKENIZER_TEST_PATH')
    if not path: pytest.skip('real audited SFT tokenizer asset is not available locally')
    tokenizer_tree = ast.parse((ROOT/'verl/utils/tokenizer.py').read_text())
    functions = [n for n in tokenizer_tree.body if isinstance(n, ast.FunctionDef)
                 and n.name in ('hf_tokenizer', 'set_pad_token_id')]
    ns = {'warnings': __import__('warnings')}
    exec(compile(ast.Module(body=functions, type_ignores=[]), 'real-hf-tokenizer', 'exec'), ns)
    config = OmegaConf.create({'actor_rollout_ref': {'model': {'path': '/must-not-load-weights', 'tokenizer_path': path}},
                               'reward': {'reward_model': {'enable': False}}})
    copied = []
    worker = worker_class(ns['hf_tokenizer'], copied)(config)
    dataset_tokenizer = ns['hf_tokenizer'](path, trust_remote_code=False)
    assert copied == [path]
    assert worker.input_tokenizer.get_vocab() == dataset_tokenizer.get_vocab()
    assert worker.input_tokenizer.chat_template == dataset_tokenizer.chat_template
    assert (worker.input_tokenizer.eos_token_id, worker.input_tokenizer.pad_token_id) == (151645, 151643)
    ids = dataset_tokenizer.encode('<think>work</think>\nAnswer: 2', add_special_tokens=False)
    assert worker.input_tokenizer.decode(ids, skip_special_tokens=True) == dataset_tokenizer.decode(ids, skip_special_tokens=True)
