import importlib.util
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).parents[3]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


eu = _load("eu_derpo_v15_math_under_test", ROOT / "verl/trainer/ppo/eu_derpo.py")
selector = _load(
    "eu_derpo_v15_selector_under_test",
    ROOT / "verl/workers/rollout/sglang_rollout/eu_derpo_v15.py",
)


def test_v15_selector_transition_and_original_logit_weights():
    logits = torch.arange(128, dtype=torch.float32)[None]
    noise = torch.zeros((1, 16))
    bootstrap_ids, bootstrap_alpha, mode = selector.select_v15_routes(
        logits, layer_id=0, state_version=0, noise=noise
    )
    assert mode == "esrl_bootstrap"
    assert torch.equal(bootstrap_ids[0, :4], torch.tensor([127, 126, 125, 124], dtype=torch.int32))

    mu = torch.zeros((48, 128))
    sigma = torch.zeros_like(mu)
    candidates = torch.arange(108, 124)
    mu[0, candidates] = torch.arange(16, dtype=torch.float32).flip(0)
    history_ids, history_alpha, mode = selector.select_v15_routes(
        logits, layer_id=0, state_version=1, mu=mu, sigma=sigma, noise=noise
    )
    assert mode == "utility_history"
    assert set(history_ids[0, 4:].tolist()) == {108, 109, 110, 111}
    expected_alpha = logits.gather(-1, history_ids.long()).softmax(-1)
    assert torch.allclose(history_alpha, expected_alpha)
    assert not torch.equal(history_ids[:, 4:], bootstrap_ids[:, 4:])


def test_v15_loss_sign_jacobian_and_local_rms():
    z = torch.tensor([[0.2, -0.1, 0.7]], requires_grad=True)
    alpha = z.softmax(-1)
    alpha.retain_grad()
    s = torch.tensor([[1.5, -0.25, 0.75]])
    loss = -(alpha * s).sum()
    loss.backward()
    recovered_s = -alpha.grad
    utility = eu.centered_routing_utility(alpha.detach(), recovered_s)
    assert torch.allclose(z.grad, -(alpha.detach() * utility), atol=1.0e-6)

    normalized = eu.local_rms_routing_utility(alpha.detach(), recovered_s, 1.0e-6)
    rms = (alpha.detach() * normalized.square()).sum(-1)
    assert torch.all(rms < 1.0)
    assert torch.allclose((alpha.detach() * normalized).sum(-1), torch.zeros(1), atol=1.0e-6)


def test_v15_response_level_moments_and_cold_start():
    response_sum = torch.zeros((3, 48, 128))
    response_count = torch.zeros_like(response_sum)
    response_sum[0, 0, 7], response_count[0, 0, 7] = 2.0, 2.0
    response_sum[1, 0, 7], response_count[1, 0, 7] = 6.0, 2.0
    response_sum[2, 0, 8], response_count[2, 0, 8] = -2.0, 1.0
    state = eu.utility_history_from_response_observations(response_sum, response_count, version=1)
    assert state.n[0, 7] == 2
    assert state.s[0, 7] == 4
    assert state.q[0, 7] == 10
    assert state.mu[0, 7] == 2
    assert torch.allclose(state.sigma[0, 7], torch.sqrt(torch.tensor(2.0)))
    assert state.n[0, 8] == 1 and state.mu[0, 8] == -2 and state.sigma[0, 8] == 1
    assert state.n[0, 9] == 0 and state.mu[0, 9] == 0 and state.sigma[0, 9] == 1
    assert all(not value.requires_grad for value in (state.n, state.s, state.q, state.mu, state.sigma))


def test_v15_valid_credit_excludes_masked_and_zero_advantage_tokens():
    behavior = torch.log(torch.tensor([[0.2, 0.2, 0.2, 0.2]]))
    current = torch.log(torch.tensor([[0.21, 0.9, 0.19, 0.1]]))
    advantage = torch.tensor([[1.0, 1.0, 0.0, -1.0]])
    action_mask = torch.tensor([[True, True, True, False]])
    dppo = eu.dppo_tv_valid_mask(behavior, current, advantage, 0.2, 0.2)
    valid = dppo & action_mask & advantage.ne(0)
    assert torch.equal(valid, torch.tensor([[True, False, False, False]]))
