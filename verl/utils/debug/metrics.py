# Copyright 2025 Individual Contributor: TomQunChaoA
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import logging
import math
import os
from pathlib import Path

import numpy as np
import torch

from verl.protocol import DataProto

logger = logging.getLogger(__file__)


def should_save_tim_scatter(global_step: int, total_training_steps: int, interval: int, save_final: bool) -> bool:
    if global_step < 1 or total_training_steps < 1 or interval < 1:
        raise ValueError("TIM scatter step and interval values must be positive")
    return global_step % interval == 0 or (save_final and global_step == total_training_steps)


def deterministic_even_indices(valid_flat_indices: torch.Tensor, max_token_pairs: int) -> torch.Tensor:
    if valid_flat_indices.ndim != 1 or valid_flat_indices.numel() == 0 or max_token_pairs < 1:
        raise ValueError("TIM scatter selection requires valid indices and a positive limit")
    count = min(valid_flat_indices.numel(), max_token_pairs)
    if count == valid_flat_indices.numel():
        return valid_flat_indices
    if count == 1:
        return valid_flat_indices[:1]
    positions = torch.arange(count, dtype=torch.int64, device=valid_flat_indices.device)
    positions = positions * (valid_flat_indices.numel() - 1) // (count - 1)
    return valid_flat_indices.index_select(0, positions)


def save_tim_scatter_sidecar(
    rollout_log_probs: torch.Tensor,
    training_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    config,
    *,
    global_step: int,
    total_training_steps: int,
    precision: str,
    r3_enabled: bool,
) -> Path | None:
    if not config or not config.get("enabled", False):
        return None
    interval = int(config.get("interval", 0))
    max_token_pairs = int(config.get("max_token_pairs", 0))
    save_final = config.get("save_final") is True
    if config.get("required") is not True:
        raise ValueError("enabled TIM scatter output must be required")
    if not should_save_tim_scatter(global_step, total_training_steps, interval, save_final):
        return None
    if (
        rollout_log_probs.ndim != 2
        or rollout_log_probs.shape != training_log_probs.shape
        or response_mask.shape != rollout_log_probs.shape
    ):
        raise ValueError("TIM scatter log-prob tensors and response mask must have identical shapes")
    if rollout_log_probs.device != training_log_probs.device or response_mask.device != rollout_log_probs.device:
        raise ValueError("TIM scatter log-prob tensors and response mask must be on the same device")

    mask = response_mask.bool()
    valid_flat_indices = mask.reshape(-1).nonzero(as_tuple=False).squeeze(1)
    selected = deterministic_even_indices(valid_flat_indices, max_token_pairs)
    width = rollout_log_probs.shape[1]
    rollout_selected = rollout_log_probs.reshape(-1).index_select(0, selected).detach().float()
    training_selected = training_log_probs.reshape(-1).index_select(0, selected).detach().float()

    seq_length = mask.sum(dim=1)
    if (seq_length == 0).any():
        raise ValueError("TIM scatter requires every response to contain at least one valid token")
    seq_log_ratio = torch.where(
        mask,
        training_log_probs.float() - rollout_log_probs.float(),
        torch.zeros((), dtype=torch.float32, device=mask.device),
    ).sum(dim=1)
    seq_mean_log_ratio = seq_log_ratio / seq_length.float()

    output_dir = Path(str(config.get("dir", "")))
    if not output_dir.is_dir():
        raise FileNotFoundError(f"TIM scatter directory does not exist: {output_dir}")
    final_path = output_dir / f"s{global_step:06d}.npz"
    temporary_path = output_dir / f".tmp{global_step:06d}.npz"
    np.savez_compressed(
        temporary_path,
        token_inference_logprob=rollout_selected.cpu().numpy().astype(np.float32, copy=False),
        token_training_logprob=training_selected.cpu().numpy().astype(np.float32, copy=False),
        token_inference_prob=rollout_selected.exp().cpu().numpy().astype(np.float32, copy=False),
        token_training_prob=training_selected.exp().cpu().numpy().astype(np.float32, copy=False),
        token_batch_row=(selected // width).to(torch.int32).cpu().numpy(),
        token_position=(selected % width).to(torch.int32).cpu().numpy(),
        seq_length=seq_length.to(torch.int32).cpu().numpy(),
        seq_log_ratio=seq_log_ratio.cpu().numpy().astype(np.float32, copy=False),
        seq_mean_log_ratio=seq_mean_log_ratio.cpu().numpy().astype(np.float32, copy=False),
        global_step=np.asarray(global_step, dtype=np.int64),
        precision=np.asarray(precision),
        r3_enabled=np.asarray(r3_enabled, dtype=np.bool_),
        schema_version=np.asarray(1, dtype=np.int32),
    )
    os.replace(temporary_path, final_path)
    return final_path


def calculate_token_list_diff(tensor1: torch.Tensor, tensor2: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # verify inputs
    if tensor1.numel() == 0 or tensor2.numel() == 0:
        return torch.zeros(tensor1.shape[0], dtype=torch.long, device=tensor1.device)
    if tensor1.shape != tensor2.shape or mask.shape != tensor1.shape or mask.shape != tensor2.shape:
        print(
            f"<WARN> dim of tensor1, tensor2, mask is not equal, {(tensor1.shape)=},{(tensor2.shape)=}, {(mask.shape)=}"
        )
        return torch.ones_like(tensor1)
    # transfer to same device
    if tensor2.device != tensor1.device:
        tensor2 = tensor2.to(tensor1.device)
    if mask.device != tensor1.device:
        mask = mask.to(tensor1.device)

    # calculate diff
    diff_mask = tensor1 != tensor2

    valid_diff_mask = diff_mask & (mask == 1)

    diff_counts = valid_diff_mask.sum(dim=1)

    return diff_counts


def pearson_correlation_coefficient(tensor1: torch.Tensor, tensor2: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # implemention of https://arxiv.org/pdf/2506.13585
    if tensor1.shape != tensor2.shape or mask.shape != tensor1.shape or mask.shape != tensor2.shape:
        return 0
    mt1 = torch.masked_select(tensor1, mask)
    mt2 = torch.masked_select(tensor2, mask)
    result = torch.corrcoef(torch.stack([mt1, mt2], dim=0))
    return result[0][1].detach().item()


def calculate_log_prob_diff(log_probs1: torch.Tensor, log_probs2: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    full_diff = torch.abs(log_probs1 - log_probs2)
    return torch.masked_select(full_diff, mask)


@torch.no_grad()
def calculate_router_shift_metrics(
    old_router_log_probs: torch.Tensor,
    current_router_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
) -> dict:
    """Calculate RSPO router-shift diagnostics from aligned [batch, response, layer, top-k] tensors."""
    if old_router_log_probs.shape != current_router_log_probs.shape or old_router_log_probs.ndim != 4:
        raise ValueError("router log-prob tensors must have identical [batch, response, layer, top-k] shapes")
    if response_mask.shape != old_router_log_probs.shape[:2]:
        raise ValueError("router response mask must have [batch, response] shape")

    mask = response_mask.bool()
    if not mask.any():
        raise ValueError("router-shift diagnostics require at least one valid response token")

    delta = (current_router_log_probs.float() - old_router_log_probs.float()).abs().mean(dim=-1).mean(dim=-1)
    gamma = torch.exp(-delta)[mask]
    return {
        "diag/router_shift/ratio_mean": gamma.mean().item(),
        "diag/router_shift/clipfrac_gamma_0_8": (gamma < 0.8).float().mean().item(),
    }


@torch.no_grad()
def calculate_debug_metrics(
    data: DataProto,
    *,
    scatter_config=None,
    global_step: int | None = None,
    total_training_steps: int | None = None,
    precision: str = "unknown",
    r3_enabled: bool = False,
) -> dict:
    """
    calculate rollout vs actor logprobs diff, for debugging purpose

    Args:
        data: DataProto
            the data batch to calculate
            rollout_log_probs: log_probs record when rollout forward tokens
            old_log_probs or current_log_probs: sampled-token logprobs from the actor prepass
            loss_mask or attention_mask: to mask unrelated token
            responses: the response tokens, for calculating size
    Returns:
        dict: metrics
            "training/rollout_probs_diff_valid": 1->input is valid, 0->input is invalid
            "training/rollout_probs_diff_max": max value of logprob diff of rollout vs. actor
            "training/rollout_probs_diff_mean": mean value of logprob diff of rollout vs. actor
            "training/rollout_probs_diff_std": std value of logprob diff of rollout vs. actor
            "training/rollout_actor_probs_pearson_corr": logprob's pearson corrcoef of rollout vs. actor, reference to https://arxiv.org/pdf/2506.13585
    """

    rollout_old_log_probs = data.batch["rollout_log_probs"]
    actor_old_log_probs = (
        data.batch["current_log_probs"] if "current_log_probs" in data.batch else data.batch["old_log_probs"]
    )
    if "response_mask" in data.batch:
        logger.debug("response mask found, use it to mask log probs")
        log_prob_mask = data.batch["response_mask"]
    elif "loss_mask" in data.batch:
        log_prob_mask = data.batch["loss_mask"]
    elif "attention_mask" in data.batch:
        log_prob_mask = data.batch["attention_mask"]
    else:
        logger.warning(f"no mask info found, use all log probs, {(data.batch.keys())=}")
        log_prob_mask = torch.ones_like(rollout_old_log_probs)
    responses = data.batch["responses"]
    response_length = responses.size(1)

    response_mask = log_prob_mask[:, -response_length:]
    # calculate pearson corrcoef
    actor_probs = torch.exp(actor_old_log_probs)
    rollout_probs = torch.exp(rollout_old_log_probs)
    response_mask_bool = response_mask.bool()
    if (
        actor_old_log_probs.shape != rollout_old_log_probs.shape
        or response_mask_bool.shape != actor_old_log_probs.shape
    ):
        raise ValueError("TIM log-prob tensors and response mask must have identical shapes")
    if not response_mask_bool.any():
        raise ValueError("TIM diagnostics require at least one valid response token")

    # Samples come from the inference policy. For r = pi_train / pi_infer,
    # exp(log(r)) - 1 - log(r) is the paper's sampled-token k3 estimator.
    log_ratio = actor_old_log_probs - rollout_old_log_probs
    masked_log_ratio = torch.masked_select(log_ratio, response_mask_bool).float()
    tim_kl_k3 = torch.mean(torch.expm1(masked_log_ratio) - masked_log_ratio)
    tim_logprob_abs_mean = torch.mean(torch.abs(masked_log_ratio))
    tim_extreme_frac_tau2 = torch.mean((torch.abs(masked_log_ratio) > math.log(2.0)).float())

    pearson_corrcoef = pearson_correlation_coefficient(actor_probs, rollout_probs, response_mask_bool)
    rollout_probs_diff = calculate_log_prob_diff(actor_probs, rollout_probs, response_mask_bool)
    metrics = {
        "training/rollout_probs_diff_valid": 1,
        "training/rollout_probs_diff_max": torch.max(rollout_probs_diff).detach().item(),
        "training/rollout_probs_diff_mean": torch.mean(rollout_probs_diff).detach().item(),
        "training/rollout_probs_diff_std": torch.std(rollout_probs_diff).detach().item(),
        "training/rollout_actor_probs_pearson_corr": pearson_corrcoef,
        "diag/tim/kl_k3": tim_kl_k3.item(),
        "diag/tim/logprob_abs_mean": tim_logprob_abs_mean.item(),
        "diag/tim/extreme_frac_tau2": tim_extreme_frac_tau2.item(),
    }
    if scatter_config and scatter_config.get("enabled", False):
        if "response_mask" not in data.batch:
            raise ValueError("TIM scatter requires the explicit response_mask tensor")
        if global_step is None or total_training_steps is None:
            raise ValueError("TIM scatter requires global and final training steps")
        save_tim_scatter_sidecar(
            rollout_old_log_probs,
            actor_old_log_probs,
            response_mask_bool,
            scatter_config,
            global_step=global_step,
            total_training_steps=total_training_steps,
            precision=precision,
            r3_enabled=r3_enabled,
        )
    return metrics
