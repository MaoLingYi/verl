from types import MethodType

import torch


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

        self.mode = None
        self._capturing = False
        self.old_cache = {}
        self._pending_old_log_probs = {}
        self._old_indices = None
        self._old_log_probs = None
        self._current_indices = None
        self._current_old_log_probs = None
        self._current_abs_sums = None
        self._current_microbatches = []
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
        for router in self.routers:
            original_gating = getattr(router, "_verl_router_shift_original_gating", None)
            if original_gating is not None:
                del router.gating
                del router._verl_router_shift_original_gating
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    @staticmethod
    def selected_old_router_log_probs(logits, routing_map, topk, pre_softmax=True):
        flat_logits = logits.detach().float().reshape(-1, logits.shape[-1])
        flat_map = routing_map.detach().reshape(-1, routing_map.shape[-1]).bool()
        if flat_logits.shape != flat_map.shape or not torch.all(flat_map.sum(dim=-1) == topk):
            raise RuntimeError("router-shift diagnostics received invalid top-k routing map")
        if not torch.isfinite(flat_logits).all():
            raise RuntimeError("router-shift diagnostics received non-finite router logits")
        indices = torch.topk(flat_map.to(torch.uint8), topk, dim=-1).indices
        selected = torch.log_softmax(flat_logits, dim=-1).gather(-1, indices)
        return indices, selected

    @staticmethod
    def current_abs_diff_sum(logits, old_indices, old_selected_log_probs, pre_softmax=True):
        flat_logits = logits.detach().float().reshape(-1, logits.shape[-1])
        if (
            old_indices.shape != old_selected_log_probs.shape
            or old_indices.ndim != 2
            or old_indices.shape[0] != flat_logits.shape[0]
            or old_indices.dtype not in (torch.uint8, torch.int32, torch.int64)
            or torch.any(old_indices < 0)
            or torch.any(old_indices >= flat_logits.shape[-1])
        ):
            raise RuntimeError("router-shift diagnostics received invalid old expert indices")
        if not torch.isfinite(flat_logits).all():
            raise RuntimeError("router-shift diagnostics received non-finite router logits")
        if not torch.isfinite(old_selected_log_probs).all():
            raise RuntimeError("router-shift diagnostics received non-finite old router log probabilities")
        current_selected = torch.log_softmax(flat_logits, dim=-1).gather(-1, old_indices.long())
        return (current_selected - old_selected_log_probs.float()).abs().sum(dim=-1)

    def _observe_gating(self, index, output):
        if self._capturing and self.mode == "old":
            self._pending_old_log_probs[index] = output
        elif self._capturing and self.mode == "current":
            self._current_abs_sums[index] = self.current_abs_diff_sum(
                output,
                self._current_indices[index],
                self._current_old_log_probs[index],
                self.pre_softmax,
            )

    def _make_router_hook(self, index):
        def hook(_module, _inputs, output):
            if not self._capturing or self.mode != "old":
                return
            logits = self._pending_old_log_probs.pop(index)
            indices, selected = self.selected_old_router_log_probs(
                logits, output[1], self.topk, self.pre_softmax
            )
            if logits.shape[-1] > 256:
                raise RuntimeError("router-shift diagnostics require expert indices representable as uint8")
            self._old_indices[index] = indices.to(torch.uint8)
            self._old_log_probs[index] = selected.to(torch.float32)

        return hook

    def start_old_batch(self):
        self.old_cache.clear()
        self.mode = "old"
        self._capturing = False

    def begin_old_microbatch(self):
        self._capturing = True
        self._pending_old_log_probs.clear()
        self._old_indices = [None] * len(self.routers)
        self._old_log_probs = [None] * len(self.routers)

    def finish_old_microbatch(self, input_ids, attention_mask, sample_ids, response_length):
        indices = self._unpack_sequence_parallel(self._old_indices, input_ids, attention_mask)
        log_probs = self._unpack_sequence_parallel(self._old_log_probs, input_ids, attention_mask)
        indices = indices[:, -response_length - 1 : -1].cpu()
        log_probs = log_probs[:, -response_length - 1 : -1].cpu()
        for row, sample_id in enumerate(sample_ids.tolist()):
            if sample_id in self.old_cache:
                raise RuntimeError(f"duplicate router-shift sample id: {sample_id}")
            self.old_cache[sample_id] = (indices[row], log_probs[row])
        self._capturing = False

    def finish_old_batch(self):
        self.mode = None
        self._capturing = False

    def start_current_batch(self):
        self.mode = "current"
        self._capturing = False
        self._current_microbatches = []

    def begin_current_microbatch(self, input_ids, attention_mask, sample_ids, response_length):
        self._capturing = True
        try:
            records = [self.old_cache[sample_id] for sample_id in sample_ids.tolist()]
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

    def finish_current_microbatch(self, input_ids, attention_mask, response_mask, response_length):
        local = self._unpack_sequence_parallel(
            [value.unsqueeze(-1) for value in self._current_abs_sums], input_ids, attention_mask
        )
        local = local[:, -response_length - 1 : -1, :, 0].sum(dim=-1)
        self._current_microbatches.append((local, response_mask.bool()))
        self._capturing = False

    def finish_current_batch(self):
        self.mode = None
        self._capturing = False
        local_abs_sum = torch.cat([item[0] for item in self._current_microbatches], dim=0)
        response_mask = torch.cat([item[1] for item in self._current_microbatches], dim=0)
        layer_count = torch.tensor(float(len(self.routers)), device=local_abs_sum.device)

        from megatron.core import parallel_state as mpu

        if mpu.get_pipeline_model_parallel_world_size() > 1:
            group = mpu.get_pipeline_model_parallel_group()
            torch.distributed.all_reduce(local_abs_sum, group=group)
            torch.distributed.all_reduce(layer_count, group=group)

        gamma = torch.exp(-local_abs_sum.float() / (layer_count * self.topk))[response_mask]
        if gamma.numel() == 0:
            raise RuntimeError("router-shift diagnostics require at least one valid response token")
        totals = torch.stack(
            [gamma.sum(), (gamma < 0.8).float().sum(), gamma.new_tensor(float(gamma.numel()))]
        )
        if mpu.get_data_parallel_world_size() > 1:
            torch.distributed.all_reduce(totals, group=mpu.get_data_parallel_group())
        return {
            "gamma_sum": totals[0].item(),
            "clip_sum": totals[1].item(),
            "token_count": int(totals[2].item()),
        }

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
