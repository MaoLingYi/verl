from __future__ import annotations

from functools import wraps
from types import MethodType

import torch


def clear_router_shift_on_error(method):
    """Release observer and router-replay state if an actor forward/update is aborted."""
    @wraps(method)
    def wrapped(actor, *args, **kwargs):
        try:
            return method(actor, *args, **kwargs)
        except BaseException:
            if actor.router_shift_observer is not None:
                actor.router_shift_observer.clear_old_cache()
            if getattr(actor, "enable_routing_replay", False):
                from verl.utils.megatron.router_replay_patch import RouterReplay

                RouterReplay.clear_global_router_replay_action()
                RouterReplay.clear_global_indices()
            raise

    return wrapped


class RouterShiftObserver:
    """Observer-only router diagnostics for existing Megatron old/current forwards."""

    def __init__(self, models, tf_config):
        if tf_config.virtual_pipeline_model_parallel_size is not None:
            raise ValueError("router-shift diagnostics do not support virtual pipeline parallelism")
        if tf_config.tensor_model_parallel_size > 1 and not tf_config.sequence_parallel:
            raise ValueError("router-shift diagnostics require sequence parallelism when TP > 1")
        if getattr(tf_config, "moe_expert_capacity_factor", None) is not None:
            raise ValueError("router-shift diagnostics require capacity-free top-k routing")
        if getattr(tf_config, "moe_router_score_function", None) != "softmax":
            raise ValueError("router-shift diagnostics require moe_router_score_function='softmax'")
        self.pre_softmax = getattr(tf_config, "moe_router_pre_softmax", None)
        if not isinstance(self.pre_softmax, bool):
            raise ValueError("router-shift diagnostics require boolean moe_router_pre_softmax")

        self.tf_config = tf_config
        self.routers = [
            module
            for model in models
            for module in model.modules()
            if module.__class__.__name__ == "TopKRouter" and hasattr(module, "gating")
        ]
        if not self.routers:
            raise ValueError("router-shift diagnostics require at least one TopKRouter")
        self.topk = self.routers[0].topk
        if any(router.topk != self.topk for router in self.routers):
            raise ValueError("router-shift diagnostics require a uniform router top-k")
        self.num_experts = self.routers[0].weight.shape[0]
        if any(router.weight.shape[0] != self.num_experts for router in self.routers):
            raise ValueError("router-shift diagnostics require a uniform expert count")

        self.mode = None
        self._capturing = False
        self.old_cache = {}
        self._pending_old_log_probs = {}
        self._old_indices = None
        self._old_log_probs = None
        self._current_indices = None
        self._current_old_log_probs = None
        self._current_abs_sums = None
        self._current_sample_ids = None
        self._current_microbatches = []
        self._current_invalid_flag = None
        self._current_seen_sample_ids = set()
        self._handles = []
        for index, router in enumerate(self.routers):
            self._wrap_gating(router, index)
            self._handles.append(router.register_forward_hook(self._make_router_hook(index)))

    def _wrap_gating(self, router, index):
        if hasattr(router, "_verl_router_shift_original_gating"):
            raise RuntimeError("router-shift diagnostics are already attached to this router")
        original_gating = router.gating

        def wrapped_gating(_router, *args, **kwargs):
            output = original_gating(*args, **kwargs)
            self._observe_gating(index, output.detach())
            return output

        router._verl_router_shift_original_gating = original_gating
        router.gating = MethodType(wrapped_gating, router)

    def close(self):
        self.clear_old_cache()
        for router in self.routers:
            original_gating = getattr(router, "_verl_router_shift_original_gating", None)
            if original_gating is not None:
                del router.gating
                del router._verl_router_shift_original_gating
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    @staticmethod
    @torch.no_grad()
    def selected_old_router_log_probs(logits, routing_map, topk, pre_softmax=True):
        flat_logits = logits.detach().float().reshape(-1, logits.shape[-1])
        flat_routing = routing_map.detach().reshape(-1, routing_map.shape[-1])
        if flat_routing.dtype == torch.bool:
            if flat_logits.shape != flat_routing.shape or not torch.all(flat_routing.sum(dim=-1) == topk):
                raise RuntimeError("router-shift diagnostics received invalid top-k routing map")
            indices = torch.topk(flat_routing.to(torch.uint8), topk, dim=-1).indices
        elif flat_routing.dtype in (torch.uint8, torch.int32, torch.int64):
            indices = flat_routing.long()
            if (
                flat_routing.shape != (flat_logits.shape[0], topk)
                or torch.any(indices < 0)
                or torch.any(indices >= flat_logits.shape[-1])
            ):
                raise RuntimeError("router-shift diagnostics received invalid top-k expert indices")
        else:
            raise RuntimeError("router-shift diagnostics received unsupported routing data")
        if not torch.isfinite(flat_logits).all():
            raise RuntimeError("router-shift diagnostics received non-finite router logits")
        selected = torch.log_softmax(flat_logits, dim=-1).gather(-1, indices)
        return indices, selected

    @staticmethod
    @torch.no_grad()
    def current_abs_diff_sum(
        logits, old_indices, old_selected_log_probs, pre_softmax=True, *, invalid_flag: torch.Tensor
    ):
        """Read trusted old-cache data; defer dynamic CUDA validation until the batch boundary."""
        flat_logits = logits.detach().float().reshape(-1, logits.shape[-1])
        if (
            old_indices.shape != old_selected_log_probs.shape
            or old_indices.ndim != 2
            or old_indices.shape[0] != flat_logits.shape[0]
            or old_indices.dtype not in (torch.uint8, torch.int32, torch.int64)
        ):
            raise RuntimeError("router-shift diagnostics received invalid old expert indices")
        # No CUDA bool is consumed by Python in the current-layer hot path.
        # Old index bounds / finite log-probs were checked before cache insertion.
        invalid_flag.logical_or_(~torch.isfinite(flat_logits).all())
        current_selected = torch.log_softmax(flat_logits, dim=-1).gather(-1, old_indices.long())
        delta = (current_selected - old_selected_log_probs.float()).abs().sum(dim=-1)
        invalid_flag.logical_or_(~torch.isfinite(delta).all())
        return delta

    @torch.no_grad()
    def _observe_gating(self, index, output):
        if self._capturing and self.mode == "old":
            self._pending_old_log_probs[index] = output
        elif self._capturing and self.mode == "current":
            if output.shape[-1] != self.num_experts or self._current_indices.shape[-1] != self.topk:
                raise RuntimeError("router-shift current router dimensions do not match the old cache")
            self._current_abs_sums[index] = self.current_abs_diff_sum(
                output,
                self._current_indices[index],
                self._current_old_log_probs[index],
                self.pre_softmax,
                invalid_flag=self._current_invalid_flag,
            )

    def _make_router_hook(self, index):
        def hook(module, _inputs, output):
            if not self._capturing or self.mode != "old":
                return
            logits = self._pending_old_log_probs.pop(index)
            routing_data = output[1]
            replay = getattr(module, "router_replay", None)
            if (
                replay is not None
                and getattr(getattr(replay, "router_replay_action", None), "value", None) == "replay_forward"
                and replay.target_topk_idx is not None
            ):
                routing_data = replay.target_topk_idx
            indices, selected = self.selected_old_router_log_probs(
                logits, routing_data, self.topk, self.pre_softmax
            )
            if logits.shape[-1] > 256:
                raise RuntimeError("router-shift diagnostics require expert indices representable as uint8")
            self._old_indices[index] = indices.to(torch.uint8)
            self._old_log_probs[index] = selected.to(torch.float32)

        return hook

    def start_old_batch(self):
        self.clear_old_cache()
        self.mode = "old"
        self._capturing = False

    def begin_old_microbatch(self):
        self._capturing = True
        self._pending_old_log_probs.clear()
        self._old_indices = [None] * len(self.routers)
        self._old_log_probs = [None] * len(self.routers)

    @staticmethod
    def _sample_ids(sample_ids: torch.Tensor, batch_size: int) -> list[int]:
        if sample_ids.device.type != "cpu" or sample_ids.dtype != torch.int64 or sample_ids.shape != (batch_size,):
            raise RuntimeError("router-shift sample ids must be CPU int64 row metadata")
        return sample_ids.tolist()

    @torch.no_grad()
    def finish_old_microbatch(self, input_ids, attention_mask, sample_ids, response_length):
        indices = self._unpack_sequence_parallel(self._old_indices, input_ids, attention_mask)
        log_probs = self._unpack_sequence_parallel(self._old_log_probs, input_ids, attention_mask)
        indices = indices[:, -response_length - 1 : -1].cpu()
        log_probs = log_probs[:, -response_length - 1 : -1].cpu()
        # Cache owns detached CPU records, treated as immutable by the current pass.
        # Validate once here (after TP layout restoration), not 12 times per current microbatch.
        expected_shape = (input_ids.shape[0], response_length, len(self.routers), self.topk)
        if (
            indices.shape != expected_shape
            or log_probs.shape != expected_shape
            or indices.dtype != torch.uint8
            or log_probs.dtype != torch.float32
        ):
            raise RuntimeError("router-shift old cache has invalid shape or dtype")
        if torch.any(indices.int() >= self.num_experts):
            raise RuntimeError("router-shift old cache has invalid expert indices")
        if not torch.isfinite(log_probs).all():
            raise RuntimeError("router-shift old cache has non-finite log probabilities")
        for row, sample_id in enumerate(self._sample_ids(sample_ids, input_ids.shape[0])):
            if sample_id in self.old_cache:
                raise RuntimeError(f"duplicate router-shift sample id: {sample_id}")
            self.old_cache[sample_id] = (indices[row], log_probs[row])
        self._old_indices = None
        self._old_log_probs = None
        self._pending_old_log_probs.clear()
        self._capturing = False

    def finish_old_batch(self):
        self.mode = None
        self._capturing = False

    def start_current_batch(self):
        self._reset_current_batch()
        self.mode = "current"
        self._current_invalid_flag = torch.zeros((), dtype=torch.bool, device=self.routers[0].weight.device)

    @torch.no_grad()
    def begin_current_microbatch(self, input_ids, attention_mask, sample_ids, response_length):
        ids = self._sample_ids(sample_ids, input_ids.shape[0])
        if len(set(ids)) != len(ids) or self._current_seen_sample_ids.intersection(ids):
            raise RuntimeError("duplicate current router-shift sample id")
        self._current_seen_sample_ids.update(ids)
        self._capturing = True
        try:
            records = [self.old_cache[sample_id] for sample_id in ids]
        except KeyError as exc:
            raise RuntimeError(f"missing old router record for sample id {exc.args[0]}") from exc
        old_indices = torch.stack([record[0] for record in records])
        old_log_probs = torch.stack([record[1] for record in records])
        self._current_indices = self._pack_sequence_parallel(
            old_indices, input_ids, attention_mask, response_length
        ).permute(1, 0, 2)
        self._current_old_log_probs = self._pack_sequence_parallel(
            old_log_probs, input_ids, attention_mask, response_length
        ).permute(1, 0, 2)
        self._current_abs_sums = [None] * len(self.routers)
        self._current_sample_ids = sample_ids.detach()

    @torch.no_grad()
    def finish_current_microbatch(self, input_ids, attention_mask, response_mask, response_length):
        local = self._unpack_sequence_parallel(
            [value.unsqueeze(-1) for value in self._current_abs_sums], input_ids, attention_mask
        )
        local = local[:, -response_length - 1 : -1, :, 0].sum(dim=-1)
        if local.shape != response_mask.shape or local.device != response_mask.device:
            raise RuntimeError("router-shift delta and response mask shapes/devices differ")
        # local is a fresh reduction; the actor never mutates response_mask in-place.
        # Keep only detached token statistics, not logits or model activations.
        self._current_microbatches.append(
            (local.detach(), response_mask.detach().bool(), self._current_sample_ids)
        )
        self._current_indices = None
        self._current_old_log_probs = None
        self._current_abs_sums = None
        self._current_sample_ids = None
        self._capturing = False

    @torch.no_grad()
    def finish_current_batch(self, need_gamma_by_sample: bool = False):
        try:
            return self._finish_current_batch(need_gamma_by_sample)
        except BaseException:
            self.clear_old_cache()
            raise
        finally:
            self._reset_current_batch()

    def _finish_current_batch(self, need_gamma_by_sample: bool):
        self.mode = None
        self._capturing = False
        if not self._current_microbatches:
            raise RuntimeError("router-shift diagnostics require at least one current microbatch")
        device = self.routers[0].weight.device
        rows = [] if need_gamma_by_sample else None
        for local, mask, sample_ids in self._current_microbatches:
            if (
                local.ndim != 2
                or local.shape != mask.shape
                or local.dtype != torch.float32
                or mask.dtype != torch.bool
                or local.device != device
                or mask.device != device
            ):
                raise RuntimeError("router-shift delta and response mask shapes/dtypes/devices differ")
            # Execution already validates sample identity in begin_current_microbatch.
            # Only weighting needs a second traversal to build the CPU gamma mapping.
            if rows is not None:
                rows.extend((sample_id, local.shape[1]) for sample_id in self._sample_ids(sample_ids, local.shape[0]))
        if rows is not None and len({sample_id for sample_id, _ in rows}) != len(rows):
            raise RuntimeError("duplicate current router-shift sample id")

        # Flattening also handles different response widths without padding/reordering rows.
        # Layer count and the 0/1 error flag are exactly representable in FP32 at this scale.
        payload = torch.cat([
            torch.tensor([float(len(self.routers))], dtype=torch.float32, device=device),
            self._current_invalid_flag.reshape(1).float(),
            *[local.reshape(-1) for local, _, _ in self._current_microbatches],
        ])
        mask = torch.cat([mask.reshape(-1) for _, mask, _ in self._current_microbatches])
        self._current_microbatches.clear()

        from megatron.core import parallel_state as mpu

        if mpu.get_pipeline_model_parallel_world_size() > 1:
            group = mpu.get_pipeline_model_parallel_group()
            torch.distributed.all_reduce(payload, group=group)

        # Aggregate delta across ALL PP layers BEFORE exponentiating, never average stage gammas.
        gamma = torch.exp(-payload[2:] / (payload[0] * self.topk))
        invalid = (payload[1] != 0) | ~torch.isfinite(payload[2:]).all()
        totals = torch.stack([
            torch.where(mask, gamma, 0.0).sum(),
            ((gamma < 0.8) & mask).sum(dtype=torch.float32),
            mask.sum(dtype=torch.float32),
            invalid.float(),
        ])
        # Propagate invalid status before any host-side raise so peers follow the same collectives.
        if mpu.get_data_parallel_world_size() > 1:
            torch.distributed.all_reduce(totals, group=mpu.get_data_parallel_group())
        gamma_sum, clip_sum, token_count, invalid = totals.tolist()
        if invalid != 0:
            raise RuntimeError("router-shift diagnostics received non-finite current router data")
        if token_count == 0:
            raise RuntimeError("router-shift diagnostics require at least one valid response token")
        result = {
            "gamma_sum": gamma_sum,
            "clip_sum": clip_sum,
            "token_count": int(token_count),
        }
        if need_gamma_by_sample:
            gamma_cpu = gamma.detach().cpu()  # One bulk copy, only for RS weighting.
            gamma_by_sample = {}
            offset = 0
            for sample_id, width in rows:
                gamma_by_sample[sample_id] = gamma_cpu[offset : offset + width]
                offset += width
            result["gamma_by_sample"] = gamma_by_sample
        return result

    def _reset_current_batch(self):
        self.mode = None
        self._capturing = False
        self._current_microbatches.clear()
        self._current_seen_sample_ids.clear()
        self._current_invalid_flag = None
        self._current_indices = None
        self._current_old_log_probs = None
        self._current_abs_sums = None
        self._current_sample_ids = None

    def clear_old_cache(self):
        self._reset_current_batch()
        self.old_cache.clear()
        self._pending_old_log_probs.clear()
        self._old_indices = None
        self._old_log_probs = None

    @staticmethod
    def aggregate_partials(local_abs_diff_sums, local_layer_counts, topk):
        total = torch.stack(local_abs_diff_sums).sum(dim=0).float()
        layer_count = sum(local_layer_counts)
        return torch.exp(-total / (layer_count * topk))

    @staticmethod
    def _pack_sequence_parallel(records, input_ids, attention_mask, response_length):
        if input_ids.is_nested:
            raise ValueError("router-shift diagnostics currently require padded Megatron inputs")
        batch_size, seq_len = attention_mask.shape[:2]
        full = torch.zeros(
            (batch_size, seq_len, *records.shape[2:]), dtype=records.dtype, device=attention_mask.device
        )
        full[:, -response_length - 1 : -1] = records.to(attention_mask.device)

        from megatron.core.tensor_parallel import scatter_to_sequence_parallel_region
        from verl.models.mcore.util import preprocess_packed_seqs

        packed, _ = preprocess_packed_seqs(full, attention_mask, pre_process=True)
        return scatter_to_sequence_parallel_region(packed.squeeze(0).contiguous())

    @staticmethod
    def _unpack_sequence_parallel(records, input_ids, attention_mask):
        if any(record is None for record in records):
            raise RuntimeError("router-shift observer did not receive every local router output")
        if input_ids.is_nested:
            raise ValueError("router-shift diagnostics currently require padded Megatron inputs")

        from megatron.core.tensor_parallel import gather_from_sequence_parallel_region
        from verl.models.mcore.util import postprocess_packed_seqs, preprocess_packed_seqs

        packed = torch.stack(records, dim=1)
        packed = gather_from_sequence_parallel_region(packed, tensor_parallel_output_grad=False).unsqueeze(0)
        batch_size, seq_len = attention_mask.shape[:2]
        _, packed_seq_params = preprocess_packed_seqs(input_ids, attention_mask, pre_process=True)
        return postprocess_packed_seqs(
            packed, packed_seq_params, attention_mask, batch_size, seq_len, post_process=True
        )
