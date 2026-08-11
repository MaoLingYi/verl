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

import logging
import math

import torch

from verl.protocol import DataProto

logger = logging.getLogger(__file__)


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
def calculate_debug_metrics(data: DataProto) -> dict:
    """
    calculate rollout vs actor logprobs diff, for debugging purpose

    Args:
        data: DataProto
            the data batch to calculate
            rollout_log_probs: log_probs record when rollout forward tokens
            old_log_probs(actor log probs): log_probs record when actor forward tokens
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
    actor_old_log_probs = data.batch["old_log_probs"]
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
    return {
        "training/rollout_probs_diff_valid": 1,
        "training/rollout_probs_diff_max": torch.max(rollout_probs_diff).detach().item(),
        "training/rollout_probs_diff_mean": torch.mean(rollout_probs_diff).detach().item(),
        "training/rollout_probs_diff_std": torch.std(rollout_probs_diff).detach().item(),
        "training/rollout_actor_probs_pearson_corr": pearson_corrcoef,
        "diag/tim/kl_k3": tim_kl_k3.item(),
        "diag/tim/logprob_abs_mean": tim_logprob_abs_mean.item(),
        "diag/tim/extreme_frac_tau2": tim_extreme_frac_tau2.item(),
    }
