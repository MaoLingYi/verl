"""Bounded UID-group collection for the EU-DERPO V1.7 legacy trainer.

Uses the legacy verl-recipe/dapo/dapo_ray_trainer.py filtering/refill pattern;
keeps only complete correctness groups and never retains raw generation batches.
"""

from copy import deepcopy

import numpy as np


def response_length_metrics(lengths, prefix, response_limit):
    values = np.asarray(lengths, dtype=np.int64)
    return {
        f"{prefix}/mean": float(values.mean()),
        f"{prefix}/p50": float(np.quantile(values, 0.50)),
        f"{prefix}/p90": float(np.quantile(values, 0.90)),
        f"{prefix}/p95": float(np.quantile(values, 0.95)),
        f"{prefix}/max": int(values.max()),
        f"{prefix}/hit_8k_rate": float((values >= response_limit).mean()),
    }


class PromptGroupSampler:
    """One optimizer batch: accepted payloads plus scalar counts/lengths only."""

    def __init__(self, target_groups, rollout_n, metric, max_num_gen_batches, response_limit):
        self.target_groups = target_groups
        self.rollout_n = rollout_n
        self.metric = metric
        self.max_num_gen_batches = max_num_gen_batches
        if max_num_gen_batches <= 0:
            raise ValueError("Dynamic Sampling requires a positive max_num_gen_batches")
        self.response_limit = response_limit
        self.batches = []
        self.accepted_groups = 0
        self.generated_groups = 0
        self.mixed_groups = 0
        self.rejected_all_zero = 0
        self.rejected_all_one = 0
        self.rounds = 0
        self.raw_lengths = []

    def add(self, batch, *, metric_values):
        uids = batch.non_tensor_batch["uid"]
        if len(uids) != len(batch):
            raise RuntimeError("Dynamic Sampling prompt UID count differs from response count")
        accuracy = np.asarray(metric_values)
        if accuracy.shape != (len(batch),) or not np.isin(accuracy, (0, 1)).all():
            raise ValueError("Dynamic Sampling requires one binary correctness value per response")
        groups = {}
        for row, uid in enumerate(uids):
            groups.setdefault(uid, []).append(row)
        accepted_rows = []
        remaining = self.target_groups - self.accepted_groups
        for rows in groups.values():
            if len(rows) != self.rollout_n or rows != list(range(rows[0], rows[0] + self.rollout_n)):
                raise RuntimeError("Dynamic Sampling requires contiguous complete prompt groups")
            correct = int(accuracy[rows].sum())
            self.rejected_all_zero += int(correct == 0)
            self.rejected_all_one += int(correct == self.rollout_n)
            if 0 < correct < self.rollout_n:
                self.mixed_groups += 1
                if remaining > 0:
                    accepted_rows.extend(rows)
                    remaining -= 1
        self.generated_groups += len(groups)
        self.rounds += 1
        self.raw_lengths.extend(batch.batch["response_mask"].sum(-1).detach().cpu().tolist())
        if accepted_rows:
            selected = batch.select_idxs(np.asarray(accepted_rows, dtype=np.int64))
            # select_idxs shares meta_info; raw token counts differ between refill rounds.
            selected.meta_info = {
                key: deepcopy(value)
                for key, value in batch.meta_info.items()
                if key not in {"global_token_num", "images_seqlens", "metrics", "timing"}
            }
            selected.non_tensor_batch = deepcopy(selected.non_tensor_batch)
            self.batches.append(selected)
            self.accepted_groups += len(accepted_rows) // self.rollout_n
        if self.accepted_groups < self.target_groups:
            if self.rounds >= self.max_num_gen_batches:
                self.batches.clear()
                raise RuntimeError(
                    f"Dynamic Sampling exhausted {self.rounds} generation rounds: "
                    f"accepted {self.accepted_groups}/{self.target_groups} complete groups"
                )
            return None
        result = self.batches[0] if len(self.batches) == 1 else type(batch).concat(self.batches)
        self.batches.clear()
        return result

    def metrics(self, accepted_batch):
        metrics = {
            "dynamic_sampling/generated_groups": self.generated_groups,
            "dynamic_sampling/accepted_groups": self.accepted_groups,
            "dynamic_sampling/accepted_generated_groups": self.mixed_groups,
            "dynamic_sampling/surplus_groups": self.mixed_groups - self.accepted_groups,
            "dynamic_sampling/rejected_all_zero": self.rejected_all_zero,
            "dynamic_sampling/rejected_all_one": self.rejected_all_one,
            "dynamic_sampling/acceptance_rate": self.mixed_groups / self.generated_groups,
            "dynamic_sampling/generation_rounds": self.rounds,
        }
        metrics.update(response_length_metrics(self.raw_lengths, "length/raw", self.response_limit))
        lengths = accepted_batch.batch["response_mask"].sum(-1).detach().cpu().tolist()
        metrics.update(response_length_metrics(lengths, "length/accepted", self.response_limit))
        return metrics
