"""EU-DERPO V1.5 rollout routing for the pinned SGLang 0.5.9 stack.

The pure selector is kept independent of SGLang so its frozen mathematics can be
tested on CPU.  ``install_sglang_patch`` is called only in scheduler children
when the V1.5 rollout feature flag is enabled.
"""

from __future__ import annotations

import logging
import math
import os
from collections.abc import Iterable

import torch


logger = logging.getLogger(__name__)

STATE_PREFIX = "__verl_eu_derpo_v15."
STATE_MU = f"{STATE_PREFIX}mu"
STATE_SIGMA = f"{STATE_PREFIX}sigma"
STATE_VERSION = f"{STATE_PREFIX}state_version"
ACTOR_VERSION = f"{STATE_PREFIX}actor_version"
VALIDATION_MODE = f"{STATE_PREFIX}validation_mode"


def _validate_selector_support(anchors, candidates, explored, selected) -> None:
    """Enforce the frozen V1.5 ID contract at the selector boundary."""
    expected = (
        (anchors, 4, "anchors"),
        (candidates, 16, "candidates"),
        (explored, 4, "explored"),
        (selected, 8, "selected"),
    )
    for value, width, name in expected:
        if value.ndim != 2 or value.shape[-1] != width:
            raise RuntimeError(f"EU-DERPO V1.5 {name} must have shape [tokens, {width}]")
        if value.numel() and ((value < 0).any() or (value >= 128).any()):
            raise RuntimeError(f"EU-DERPO V1.5 {name} contains an out-of-range Expert ID")
        if value.numel() and not (value.sort(-1).values.diff(dim=-1) > 0).all():
            raise RuntimeError(f"EU-DERPO V1.5 {name} contains duplicate Expert IDs")
    if (anchors.unsqueeze(-1) == candidates.unsqueeze(-2)).any():
        raise RuntimeError("EU-DERPO V1.5 anchors and candidates overlap")
    if not (explored.unsqueeze(-1) == candidates.unsqueeze(-2)).any(-1).all():
        raise RuntimeError("EU-DERPO V1.5 explored Experts are not a subset of candidates")
    if (anchors.unsqueeze(-1) == explored.unsqueeze(-2)).any():
        raise RuntimeError("EU-DERPO V1.5 anchors and explored Experts overlap")


def _capture_dispatched_routes(capturer, layer_id: int, dispatched_logical_ids: torch.Tensor) -> None:
    """Capture the exact logical IDs that feed SGLang's physical dispatch mapping."""
    capturer.capture(layer_id=layer_id, topk_ids=dispatched_logical_ids)


def select_v15_routes(
    router_logits: torch.Tensor,
    *,
    layer_id: int,
    state_version: int,
    mu: torch.Tensor | None = None,
    sigma: torch.Tensor | None = None,
    noise: torch.Tensor | None = None,
    sigma_min: float = 0.0,
    sigma_max: float = 0.4,
) -> tuple[torch.Tensor, torch.Tensor, str]:
    """Return global Expert IDs and current-logit selected-softmax weights.

    Iteration one is represented by ``state_version == 0``.  From version one
    onward the Router logits only form the anchor and candidate gate; utility
    history alone ranks the sixteen candidates.
    """

    if router_logits.ndim != 2 or router_logits.shape[-1] != 128:
        raise ValueError("EU-DERPO V1.5 requires [tokens, 128] Router logits")
    if not 0 <= int(layer_id) < 48:
        raise ValueError("EU-DERPO V1.5 requires a global MoE layer in [0, 48)")

    logits = router_logits.float()
    top20 = logits.topk(20, dim=-1, sorted=True).indices
    anchors, candidates = top20[:, :4], top20[:, 4:]
    if noise is None:
        noise = torch.randn(candidates.shape, dtype=torch.float32, device=logits.device)
    else:
        noise = noise.to(device=logits.device, dtype=torch.float32)
        if noise.shape != candidates.shape:
            raise ValueError("EU-DERPO V1.5 noise must have shape [tokens, 16]")

    if int(state_version) == 0:
        probability = logits.softmax(-1)
        entropy = -(probability * probability.clamp_min(torch.finfo(probability.dtype).tiny).log()).sum(-1)
        normalized_entropy = entropy / math.log(128)
        esrl_sigma = sigma_min + (sigma_max - sigma_min) * (1 - normalized_entropy)
        exploration_score = logits.gather(-1, candidates) + esrl_sigma[:, None] * noise
        mode = "esrl_bootstrap"
    else:
        if mu is None or sigma is None or tuple(mu.shape) != (48, 128) or tuple(sigma.shape) != (48, 128):
            raise ValueError("EU-DERPO V1.5 utility state must be [48, 128]")
        layer_mu = mu[layer_id].to(device=logits.device, dtype=torch.float32)
        layer_sigma = sigma[layer_id].to(device=logits.device, dtype=torch.float32)
        exploration_score = layer_mu[candidates] + layer_sigma[candidates] * noise
        mode = "utility_history"

    explored = candidates.gather(-1, exploration_score.topk(4, dim=-1, sorted=True).indices)
    selected = torch.cat((anchors, explored), dim=-1)
    _validate_selector_support(anchors, candidates, explored, selected)
    weights = logits.gather(-1, selected).softmax(-1)
    return selected.to(torch.int32), weights, mode


class _RolloutUtilityState:
    def __init__(self, seed: int):
        self.seed = int(seed)
        self.mu = torch.zeros((48, 128), dtype=torch.float32)
        self.sigma = torch.ones((48, 128), dtype=torch.float32)
        self.state_version = 0
        self.actor_version = 0
        self.validation_mode = False
        self.last_selector_log = None
        self._generators: dict[str, torch.Generator] = {}
        self._rng_version = 0
        self._pending: dict[str, torch.Tensor] = {}

    def generator(self, device: torch.device) -> torch.Generator:
        key = str(device)
        generator = self._generators.get(key)
        if generator is None:
            generator = torch.Generator(device=device)
            generator.manual_seed(self.seed + 1_000_003 * self._rng_version)
            self._generators[key] = generator
        return generator

    def update(self, values: dict[str, torch.Tensor]) -> None:
        if VALIDATION_MODE in values:
            self.validation_mode = bool(values.pop(VALIDATION_MODE).item())
            logger.warning("EU_DERPO_V15_VALIDATION natural_routing=%d", int(self.validation_mode))
        if not values:
            return
        self._pending.update(values)
        required = {STATE_MU, STATE_SIGMA, STATE_VERSION, ACTOR_VERSION}
        if not required.issubset(self._pending):
            return
        values = {name: self._pending.pop(name) for name in required}
        # The payload arrives on the scheduler's rollout device.  Clone it
        # once per iteration so every layer can gather locally without a
        # per-token/per-layer host-to-device transfer.
        mu = values[STATE_MU].detach().float().clone()
        sigma = values[STATE_SIGMA].detach().float().clone()
        state_version = int(values[STATE_VERSION].item())
        actor_version = int(values[ACTOR_VERSION].item())
        if tuple(mu.shape) != (48, 128) or tuple(sigma.shape) != (48, 128):
            raise ValueError("EU-DERPO V1.5 rollout utility payload must be [48, 128]")
        if not torch.isfinite(mu).all() or not torch.isfinite(sigma).all() or (sigma < 0).any():
            raise ValueError("EU-DERPO V1.5 rollout utility payload is invalid")
        if state_version != actor_version:
            raise ValueError(
                f"EU-DERPO V1.5 actor/state version mismatch: actor={actor_version}, state={state_version}"
            )
        if state_version < self.state_version:
            raise ValueError("EU-DERPO V1.5 utility state version moved backwards")
        self.mu, self.sigma = mu, sigma
        self.state_version, self.actor_version = state_version, actor_version
        self._rng_version = state_version
        self._generators.clear()


def install_sglang_patch(*, seed: int = 1234) -> None:
    """Install the feature-scoped selector and state loader in one scheduler."""

    from sglang.srt.eplb.expert_location_dispatch import topk_ids_logical_to_physical
    from sglang.srt.layers.moe import topk as topk_module
    from sglang.srt.layers.moe.routed_experts_capturer import get_global_experts_capturer
    from sglang.srt.layers.moe.topk import StandardTopKOutput, TopK
    from sglang.srt.models.qwen3_moe import Qwen3MoeForCausalLM

    if getattr(TopK, "_verl_eu_derpo_v15_installed", False):
        return
    state = _RolloutUtilityState(seed)
    original_load_weights = Qwen3MoeForCausalLM.load_weights

    def load_weights(model, weights: Iterable[tuple[str, torch.Tensor]]):
        materialized = list(weights)
        utility = {name: tensor for name, tensor in materialized if name.startswith(STATE_PREFIX)}
        normal = ((name, tensor) for name, tensor in materialized if not name.startswith(STATE_PREFIX))
        if utility:
            state.update(utility)
            if not state._pending:
                logger.warning(
                    "EU_DERPO_V15_STATE_SYNC actor_version=%d utility_state_version=%d",
                    state.actor_version,
                    state.state_version,
                )
        return original_load_weights(model, normal)

    def forward(self, hidden_states, router_logits, *, num_token_non_padded=None, expert_location_dispatch_info=None):
        if state.validation_mode:
            return topk_module.select_experts(
                hidden_states=hidden_states,
                layer_id=self.layer_id,
                router_logits=router_logits,
                topk_config=self.topk_config,
                num_token_non_padded=num_token_non_padded,
                expert_location_dispatch_info=expert_location_dispatch_info,
            )
        config = self.topk_config
        if (
            config.top_k != 8
            or config.use_grouped_topk
            or config.num_fused_shared_experts != 0
            or not config.renormalize
            or config.scoring_func != "softmax"
            or config.correction_bias is not None
            or config.routed_scaling_factor is not None
            or self.layer_id is None
        ):
            raise ValueError("EU-DERPO V1.5 encountered unsupported SGLang TopK semantics")
        noise = torch.randn(
            (router_logits.shape[0], 16),
            dtype=torch.float32,
            device=router_logits.device,
            generator=state.generator(router_logits.device),
        )
        selected, weights, mode = select_v15_routes(
            router_logits,
            layer_id=int(self.layer_id),
            state_version=state.state_version,
            mu=state.mu,
            sigma=state.sigma,
            noise=noise,
        )
        logical_selected = selected.clone()
        selected = topk_ids_logical_to_physical(selected, expert_location_dispatch_info)
        topk_module._mask_topk_ids_padded_region(selected, num_token_non_padded)
        topk_module._mask_topk_ids_padded_region(logical_selected, num_token_non_padded)
        if (capturer := get_global_experts_capturer()) is not None:
            _capture_dispatched_routes(capturer, int(self.layer_id), logical_selected)
        selector_log = (mode, state.actor_version, state.state_version)
        if int(self.layer_id) == 0 and state.last_selector_log != selector_log:
            logger.warning(
                "EU_DERPO_V15_SELECTOR mode=%s actor_version=%d utility_state_version=%d",
                mode,
                state.actor_version,
                state.state_version,
            )
            state.last_selector_log = selector_log
        return StandardTopKOutput(weights, selected, router_logits)

    Qwen3MoeForCausalLM.load_weights = load_weights
    TopK.forward_cuda = forward
    TopK.forward_native = forward
    TopK._verl_eu_derpo_v15_installed = True


def run_scheduler_process_v15(*args, **kwargs):
    """Spawn-safe SGLang scheduler entrypoint used by veRL's HTTP server."""

    seed = int(os.environ.get("VERL_EU_DERPO_V15_ROLLOUT_SEED", "1234"))
    install_sglang_patch(seed=seed)
    from sglang.srt.entrypoints.engine import run_scheduler_process

    return run_scheduler_process(*args, **kwargs)
