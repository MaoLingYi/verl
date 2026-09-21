"""Memory-safe Megatron observer for frozen EU-DERPO V1.2.1."""

from __future__ import annotations

from collections import deque
from functools import wraps
import math
import os
import time
from types import MethodType

import torch

from verl.trainer.ppo.eu_derpo import (
    centered_routing_utility,
    local_rms_routing_utility,
    validate_recompute_edge_counts,
    validate_recompute_hook_count,
)
from verl.utils.debug.router_shift import RouterShiftObserver


GIB = 1024**3
HOST_RAM_SAFETY_MARGIN_BYTES = 32 * GIB


def _selected_support_loss(logits, support, coefficient, lambda_u):
    """Return -lambda*J_U and J_U on the fixed actual-F selected support."""
    flat = logits.reshape(-1, logits.shape[-1])
    ids = support.to(device=flat.device, dtype=torch.long)
    weights = coefficient.detach().to(device=flat.device, dtype=torch.float32)
    if ids.shape != weights.shape or ids.shape[0] != flat.shape[0]:
        raise RuntimeError(
            f"EU-DERPO Step E shape mismatch: logits={tuple(flat.shape)}, "
            f"support={tuple(ids.shape)}, coefficient={tuple(weights.shape)}"
        )
    selected_logits = flat.gather(-1, ids)
    log_q = selected_logits.float().log_softmax(-1)
    objective = (weights * log_q).sum()
    return -float(lambda_u) * objective, objective


def _cache_byte_plan(rows, layers, hidden_size, topk):
    rows, layers, hidden_size, topk = map(int, (rows, layers, hidden_size, topk))
    return {
        "hidden": rows * layers * hidden_size * 2,
        "support": rows * layers * topk,
        "metadata": rows * (8 + 4 + 4),
    }


def _staging_chunk_rows(staging_mib, hidden_size, topk):
    bytes_per_row = int(hidden_size) * 2 + int(topk) + int(topk) * 4
    rows = int(staging_mib) * 1024**2 // bytes_per_row
    if rows < 1:
        raise ValueError("EU-DERPO Step E staging cap cannot hold one row")
    return rows


def _node_ram_decision(required_by_rank, mem_available, safety_margin):
    node_required = sum(int(value) for value in required_by_rank)
    mem_available = int(mem_available)
    safety_margin = int(safety_margin)
    return {
        "required_by_rank": [int(value) for value in required_by_rank],
        "node_required": node_required,
        "mem_available": mem_available,
        "safety_margin": safety_margin,
        "passed": mem_available >= node_required + safety_margin,
    }


def _read_mem_available():
    with open("/proc/meminfo", encoding="ascii") as stream:
        for line in stream:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    raise RuntimeError("EU-DERPO cannot read MemAvailable from /proc/meminfo")


def _plan_local_response_rows(attention_mask, response_mask, sample_ids, response_length, tp_size, tp_rank):
    """Plan MB1 sequence-parallel valid-response rows without allocating hidden cache."""
    attention = attention_mask.detach().cpu().bool()
    response = response_mask.detach().cpu().bool()
    ids = sample_ids.detach().cpu().long().reshape(-1)
    batch, sequence = attention.shape
    if response.shape != (batch, int(response_length)) or ids.numel() != batch:
        raise RuntimeError("EU-DERPO cache planner received inconsistent batch shapes")
    if len(set(ids.tolist())) != batch:
        raise RuntimeError("EU-DERPO cache planner received duplicate sample IDs")
    if not 0 <= int(tp_rank) < int(tp_size):
        raise RuntimeError("EU-DERPO cache planner received invalid TP rank")

    planned = []
    spans = {}
    cursor = 0
    response_start = sequence - int(response_length) - 1
    if response_start < 0:
        raise RuntimeError("EU-DERPO response does not fit padded sequence")
    for sample_row, sample_id in enumerate(ids.tolist()):
        full_valid = torch.zeros(sequence, dtype=torch.bool)
        full_position = torch.full((sequence,), -1, dtype=torch.int32)
        full_valid[response_start : sequence - 1] = response[sample_row]
        full_position[response_start : sequence - 1] = torch.arange(response_length, dtype=torch.int32)
        packed_valid = full_valid[attention[sample_row]]
        packed_position = full_position[attention[sample_row]]
        padded_length = ((packed_valid.numel() + tp_size - 1) // tp_size) * tp_size
        local_length = padded_length // tp_size
        local_start = tp_rank * local_length
        local_end = local_start + local_length
        if padded_length > packed_valid.numel():
            padding = padded_length - packed_valid.numel()
            packed_valid = torch.nn.functional.pad(packed_valid, (0, padding), value=False)
            packed_position = torch.nn.functional.pad(packed_position, (0, padding), value=-1)
        positions = packed_position[local_start:local_end][packed_valid[local_start:local_end]]
        start, end = cursor, cursor + positions.numel()
        spans[int(sample_id)] = (start, end)
        cursor = end
        planned.append((sample_row, sample_id, positions))

    sample_row_table = torch.empty(cursor, dtype=torch.int32)
    sample_id_table = torch.empty(cursor, dtype=torch.int64)
    response_position_table = torch.empty(cursor, dtype=torch.int32)
    for sample_row, sample_id, positions in planned:
        start, end = spans[int(sample_id)]
        sample_row_table[start:end] = sample_row
        sample_id_table[start:end] = sample_id
        response_position_table[start:end] = positions
    return {
        "rows": cursor,
        "spans": spans,
        "sample_row": sample_row_table,
        "sample_id": sample_id_table,
        "response_position": response_position_table,
    }


def _build_edge_coefficients(stats, normalized, active_total, sample_rows, layer, support, batch_size):
    rows = sample_rows.long()
    ids = support.long()
    count = stats.count[rows, layer].gather(1, ids)
    if (count <= 0).any():
        raise RuntimeError("EU-DERPO actual-F support edge is missing from cluster statistics")
    keep = stats.mask[rows, layer].gather(1, ids)
    utility = normalized[rows, layer].gather(1, ids)
    denominator = float(batch_size) * active_total[rows, None].float() * count.masked_fill(count == 0, 1.0)
    return (keep.float() * utility.float() / denominator).detach()


def clear_eu_derpo_on_error(method):
    @wraps(method)
    def wrapped(actor, *args, **kwargs):
        try:
            return method(actor, *args, **kwargs)
        except BaseException:
            if actor.eu_derpo_observer is not None:
                actor.eu_derpo_observer.clear()
            raise

    return wrapped


def _reduce_router_auxiliary_grad(auxiliary_grad):
    from megatron.core import parallel_state as mpu

    if mpu.get_tensor_model_parallel_world_size() > 1:
        torch.distributed.all_reduce(auxiliary_grad, group=mpu.get_tensor_model_parallel_group())
    dense_dp_size = mpu.get_data_parallel_world_size()
    if dense_dp_size > 1:
        torch.distributed.all_reduce(
            auxiliary_grad, op=torch.distributed.ReduceOp.SUM, group=mpu.get_data_parallel_group()
        )
        auxiliary_grad.div_(dense_dp_size)


def _canonicalize_router_auxiliary_inputs(hidden, raw_logits, probs, routing_map, actual, topk, num_experts):
    logits = raw_logits.reshape(-1, raw_logits.shape[-1]) if raw_logits.ndim else raw_logits
    actual = actual.long()
    valid = (
        logits.ndim == probs.ndim == routing_map.ndim == actual.ndim == 2
        and logits.shape[0] == probs.shape[0] == routing_map.shape[0] == actual.shape[0]
        and logits.shape[-1] == probs.shape[-1] == routing_map.shape[-1] == num_experts
        and actual.shape[-1] == topk
        and actual.dtype == torch.long
        and actual.numel() > 0
        and actual.min().item() >= 0
        and actual.max().item() < num_experts
    )
    if not valid:
        raise RuntimeError(
            "EU-DERPO Router auxiliary shape contract failed: "
            f"hidden={tuple(hidden.shape)}, raw_logits={tuple(raw_logits.shape)}, "
            f"flattened_logits={tuple(logits.shape)}, probs={tuple(probs.shape)}, "
            f"routing_map={tuple(routing_map.shape)}, actual={tuple(actual.shape)}, "
            f"topk={topk}, num_experts={num_experts}"
        )
    return logits, actual


@torch.no_grad()
def _route_attribution_comparison(expected_ordered, actual_ordered, expected_metadata=None, actual_metadata=None):
    """Classify route rows without changing the production route acceptance rule."""
    expected = expected_ordered.long()
    actual = actual_ordered.long()
    ordered_equal = expected.eq(actual).all(-1)
    expected_sorted = expected.sort(-1).values
    actual_sorted = actual.sort(-1).values
    set_equal = expected_sorted.eq(actual_sorted).all(-1)
    semantic_equal = torch.ones_like(set_equal)
    if expected_metadata is not None and actual_metadata is not None:
        # local physical-row fields are intentionally excluded from semantic identity.
        semantic_columns = (0, 1, 3, 4, 5, 6)
        semantic_equal = expected_metadata[..., semantic_columns].eq(
            actual_metadata[..., semantic_columns]
        ).all(-1)
    intersection = (expected[:, :, None] == actual[:, None, :]).any(-1).sum(-1)
    return {
        "ordered_equal": ordered_equal,
        "set_equal": set_equal,
        "semantic_equal": semantic_equal,
        "intersection": intersection,
        "order_only": ~ordered_equal & set_equal,
        "set_mismatch": ~set_equal,
    }


def _route_attribution_summary(valid, order_only, set_mismatch):
    fraction = set_mismatch.float() / valid.clamp_min(1)
    order_layers = (order_only > 0).nonzero(as_tuple=False)
    set_layers = (set_mismatch > 0).nonzero(as_tuple=False)
    return {
        "valid_by_layer": valid.cpu().tolist(),
        "order_only_by_layer": order_only.cpu().tolist(),
        "set_mismatch_by_layer": set_mismatch.cpu().tolist(),
        "set_mismatch_fraction_by_layer": fraction.cpu().tolist(),
        "first_order_only_layer": int(order_layers[0, 0].item()) if order_layers.numel() else None,
        "first_set_mismatch_layer": int(set_layers[0, 0].item()) if set_layers.numel() else None,
    }


class EUDERPOObserver:
    """Capture actual-F state and apply deferred Router-only utility gradients."""

    def __init__(self, models, tf_config, diagnostics=False, route_attribution=False, staging_mib=64, version="1.2.1", eps_u=1.0e-6):
        required = {
            "moe_router_score_function": "softmax",
            "moe_router_pre_softmax": False,
            "moe_router_topk_scaling_factor": None,
            "moe_router_fusion": False,
            "moe_router_load_balancing_type": "none",
            "moe_z_loss_coeff": None,
            "moe_input_jitter_eps": None,
            "moe_expert_capacity_factor": None,
            "moe_apply_probs_on_input": False,
            "tensor_model_parallel_size": 2,
            "pipeline_model_parallel_size": 1,
            "expert_model_parallel_size": 8,
            "expert_tensor_parallel_size": 1,
            "context_parallel_size": 1,
            "num_layers": 48,
            "recompute_granularity": "full",
            "recompute_method": "uniform",
            "recompute_num_layers": 1,
        }
        mismatches = {key: (getattr(tf_config, key, None), expected) for key, expected in required.items() if getattr(tf_config, key, None) != expected}
        if mismatches:
            raise ValueError(f"EU-DERPO unsupported Router semantics: {mismatches}")
        if tf_config.virtual_pipeline_model_parallel_size is not None:
            raise ValueError("EU-DERPO does not support virtual pipeline parallelism")
        if tf_config.tensor_model_parallel_size > 1 and not tf_config.sequence_parallel:
            raise ValueError("EU-DERPO requires sequence parallelism when TP > 1")
        if getattr(tf_config, "hidden_dropout", 0.0) != 0 or getattr(tf_config, "attention_dropout", 0.0) != 0:
            raise ValueError("EU-DERPO F/recompute route identity requires zero model dropout")
        from megatron.core import parallel_state as mpu

        if mpu.get_data_parallel_world_size() != 4 or mpu.get_expert_data_parallel_world_size() != 1:
            raise ValueError("EU-DERPO PP1 integration requires dense DP=4 and Expert DP=1")

        self.tf_config = tf_config
        self.version = str(version)
        self.v15 = self.version == "1.5"
        self.eps_u = float(eps_u)
        self.diagnostics = bool(diagnostics)
        self.route_attribution = bool(route_attribution)
        self.routers = [
            module
            for model in models
            for module in model.modules()
            if module.__class__.__name__ == "TopKRouter" and hasattr(module, "gating")
        ]
        if not self.routers:
            raise ValueError("EU-DERPO requires at least one TopKRouter")
        self.topk = self.routers[0].topk
        self.num_experts = self.routers[0].weight.shape[0]
        self.hidden_size = self.routers[0].weight.shape[1]
        self.staging_mib = int(staging_mib)
        if self.topk != 8 or self.num_experts != 128:
            raise ValueError("EU-DERPO V1.2.1 Qwen3 recipe requires 128 Experts and Top-K=8")
        if any(
            router.topk != self.topk or tuple(router.weight.shape) != (self.num_experts, self.hidden_size)
            for router in self.routers
        ):
            raise ValueError("EU-DERPO requires uniform local Router shapes")
        if len(self.routers) != 48:
            raise ValueError("EU-DERPO PP1 requires exactly 48 local MoE layers")

        self.mode = None
        self.route_cache = {}
        self._ordered_route_cache = {}
        self._route_metadata_cache = {}
        self.current_logprob_cache = {}
        self._credit_mask = None
        self._prepass_route_cache = {}
        self._handles = []
        self._pending_alpha = {}
        self._pending_recompute = [deque() for _ in self.routers]
        self._active = None
        self._forward_routes = None
        self._forward_ordered_routes = None
        self._main_route_metadata = None
        self._prepass_routes = None
        self._prepass_ids = None
        self._sample_row = {}
        self._sample_ids_by_row = None
        self._prompt_group_by_sample = {}
        self._utility_sum = None
        self._utility_sum_sq = None
        self._utility_count = None
        self._main_hook_count = 0
        self._main_hook_count_by_layer = None
        self._expected_main_hook_count = 0
        self._edge_count_by_layer = None
        self._route_mismatch_by_layer = None
        self._forward_recompute_mismatch_by_layer = None
        self._route_compare_by_layer = None
        self._first_route_mismatch = {}
        self._phase_execution = {}
        self._alpha_summary = []
        self._alpha_min = None
        self._invalid_flag = None
        self._center_residual_max = None
        self._aux_stats = None
        self._aux_attribution = None
        self._aux_attribution_counts = None
        self._aux_first_set_mismatch = None
        self._aux_objective = 0.0
        self._native_router_grad_sq = 0.0
        self._utility_router_grad_sq = 0.0
        self._router_grad_dot = 0.0
        self._hidden_cache = None
        self._support_cache = None
        self._provenance = None
        self._planned_spans = None
        self._cache_cursor_by_layer = None
        self._cache_consumed_by_layer = None
        self._cache_plan = None
        self._cache_metrics = None
        self._native_finalize_count = 0
        self._native_finalize_completed = False
        self._optimizer_generation = None
        self._parameter_snapshot = None
        self._step_e_started = False
        self._full_auxiliary_transformer_forward_count = 0
        self._step_e_natural_topk_call_count = 0
        self._step_e_router_forward_call_count = 0
        self._step_e_routing_call_count = 0
        self._step_e_dispatch_count = 0
        self._d2h_seconds = 0.0
        self._h2d_seconds = 0.0
        self._aux_reduce_count = None
        self._main_grad_add_count = None
        self._step_e_complete = False
        for index, router in enumerate(self.routers):
            self._remember_gating(router)
            self._wrap_routing(router, index)
            self._handles.append(router.register_forward_hook(self._make_router_hook(index)))

    def _remember_gating(self, router):
        if hasattr(router, "_verl_eu_derpo_original_gating") or hasattr(router, "_verl_router_shift_original_gating"):
            raise RuntimeError("EU-DERPO Router hooks cannot be combined with Router Shift hooks")
        router._verl_eu_derpo_original_gating = router.gating

    def _wrap_routing(self, router, index):
        original = router.routing

        def wrapped_routing(_router, *args, **kwargs):
            if self.mode != "main" or not torch.is_grad_enabled():
                return original(*args, **kwargs)
            record = {"alpha": None, "routes": None, "ready": False}

            def pack(tensor):
                grad_fn = type(tensor.grad_fn).__name__ if tensor.grad_fn is not None else ""
                selected_shape = (args[0].numel() // self.num_experts, self.topk)
                if tensor.shape == selected_shape and tensor.dtype == torch.int64:
                    routes = tensor.detach()
                    if record["routes"] is None:
                        record["routes"] = routes
                    elif not torch.equal(record["routes"], routes):
                        raise RuntimeError("EU-DERPO actual-alpha saved Top-K indices disagree")
                elif tensor.shape == selected_shape and tensor.dtype == torch.float32 and grad_fn.startswith("SoftmaxBackward"):
                    if record["alpha"] is not None:
                        raise RuntimeError("EU-DERPO found multiple selected-softmax alpha nodes")
                    record["alpha"] = tensor.detach()
                    tensor.register_hook(lambda grad, item=record: self._alpha_grad(item, grad))
                return tensor

            with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
                output = original(*args, **kwargs)
            if record["alpha"] is None or record["routes"] is None:
                raise RuntimeError("EU-DERPO could not capture actual selected-softmax alpha/Top-K nodes")
            self._pending_alpha[index] = record
            return output

        router._verl_eu_derpo_original_routing = original
        router.routing = MethodType(wrapped_routing, router)

    @staticmethod
    def _selected(routing_map: torch.Tensor, topk: int) -> torch.Tensor:
        flat = routing_map.detach().reshape(-1, routing_map.shape[-1])
        if flat.dtype != torch.bool:
            raise RuntimeError("EU-DERPO received an invalid natural Top-K routing map")
        selected = flat.nonzero(as_tuple=False)[:, 1]
        if selected.numel() != flat.shape[0] * topk:
            raise RuntimeError("EU-DERPO received an invalid natural Top-K routing map")
        return selected.reshape(flat.shape[0], topk)

    def _make_router_hook(self, index):
        def hook(module, inputs, output):
            if self.mode == "prepass":
                self._record_execution("P", module)
                self._prepass_routes[index] = self._selected(output[1], self.topk).to(torch.uint8)
                return
            if self.mode == "main":
                self._observe_main(index, module, inputs[0], output)
                return
            if self.mode == "aux":
                self._observe_aux(index, module, inputs[0], output)

        return hook

    def _record_execution(self, phase, module):
        if phase not in self._phase_execution:
            self._phase_execution[phase] = {
                "model_training": module.training,
                "grad_enabled": torch.is_grad_enabled(),
                "inference_mode": torch.is_inference_mode_enabled(),
                "autocast_enabled": torch.is_autocast_enabled(),
                "weight_dtype": str(module.weight.dtype),
                "weight_device": str(module.weight.device),
                "weight_version": module.weight._version,
                "weight_data_ptr": module.weight.data_ptr(),
            }

    @torch.no_grad()
    def _mark_invalid_alpha(self, fp32_alpha, router_alpha):
        rtol = max(2.0e-5, 2 * torch.finfo(router_alpha.dtype).eps)
        invalid = (
            ~torch.isfinite(fp32_alpha).all()
            | (fp32_alpha <= 0).any()
            | ~torch.isclose(fp32_alpha, router_alpha.float(), rtol=rtol, atol=2.0e-6).all()
        )
        self._invalid_flag.bitwise_or_(invalid.to(torch.int32))

    @staticmethod
    def _ids(sample_ids: torch.Tensor, batch_size: int) -> list[int]:
        if sample_ids.device.type != "cpu" or sample_ids.dtype != torch.int64 or sample_ids.shape != (batch_size,):
            raise RuntimeError("EU-DERPO sample ids must be CPU int64 row metadata")
        return sample_ids.tolist()

    def start_prepass_batch(self):
        self._prepass_route_cache.clear()
        self.mode = "prepass"

    def begin_prepass_microbatch(self):
        self._prepass_routes = [None] * len(self.routers)

    @torch.no_grad()
    def finish_prepass_microbatch(self, input_ids, attention_mask, sample_ids, response_length):
        routes = RouterShiftObserver._unpack_sequence_parallel(self._prepass_routes, input_ids, attention_mask)
        routes = routes[:, -response_length - 1 : -1].cpu()
        expected = (input_ids.shape[0], response_length, len(self.routers), self.topk)
        if routes.shape != expected or routes.dtype != torch.uint8:
            raise RuntimeError(f"EU-DERPO prepass route shape/dtype mismatch: {routes.shape}, {routes.dtype}")
        for row, sample_id in enumerate(self._ids(sample_ids, input_ids.shape[0])):
            if sample_id in self._prepass_route_cache:
                raise RuntimeError(f"duplicate EU-DERPO sample id: {sample_id}")
            self._prepass_route_cache[sample_id] = routes[row]
        self._prepass_routes = None

    def finish_prepass_batch(self):
        if not self._prepass_route_cache:
            raise RuntimeError("EU-DERPO prepass captured no routes")
        self.mode = None

    def routes_for(self, sample_ids: torch.Tensor) -> torch.Tensor:
        try:
            return torch.stack([self.route_cache[int(sample_id)] for sample_id in sample_ids.tolist()])
        except KeyError as exc:
            raise RuntimeError(f"missing EU-DERPO actual-F route for sample {exc.args[0]}") from exc

    def _ordered_routes_for(self, sample_ids: torch.Tensor) -> torch.Tensor:
        try:
            return torch.stack([self._ordered_route_cache[int(sample_id)] for sample_id in sample_ids.tolist()])
        except KeyError as exc:
            raise RuntimeError(f"missing EU-DERPO attribution route for sample {exc.args[0]}") from exc

    @staticmethod
    def _ordered_selected(probs, selected):
        order = probs.detach().float().gather(-1, selected).argsort(-1, descending=True)
        return selected.gather(-1, order)

    def _route_metadata(self, input_ids, attention_mask, sample_ids, response_mask, response_length):
        ids = self._ids(sample_ids, input_ids.shape[0])
        device = attention_mask.device
        batch, sequence = attention_mask.shape
        response = torch.arange(response_length, dtype=torch.int64, device=device)[None, :].expand(batch, -1)
        original = response + sequence - response_length - 1
        lengths = attention_mask.sum(-1, dtype=torch.int64)
        tp_size = self.tf_config.tensor_model_parallel_size
        padded = lengths + (tp_size - lengths.remainder(tp_size)).remainder(tp_size)
        starts = torch.cat((torch.zeros(1, dtype=torch.int64, device=device), padded.cumsum(0)[:-1]))
        valid_ordinal = attention_mask.long().cumsum(-1).gather(1, original).sub(1)
        packed_ordinal = starts[:, None] + valid_ordinal
        sample = torch.tensor(ids, dtype=torch.int64, device=device)[:, None].expand(-1, response_length)
        groups = torch.tensor(
            [self._prompt_group_by_sample[value] for value in ids], dtype=torch.int64, device=device
        )[:, None].expand(-1, response_length)
        local_row = torch.arange(batch, dtype=torch.int64, device=device)[:, None].expand(-1, response_length)
        valid = response_mask.to(device).long()
        return torch.stack((sample, groups, local_row, response, original, valid, packed_ordinal), dim=-1)

    def current_logprobs_for(self, sample_ids: torch.Tensor) -> torch.Tensor:
        try:
            return torch.stack([self.current_logprob_cache[int(sample_id)] for sample_id in sample_ids.tolist()])
        except KeyError as exc:
            raise RuntimeError(f"missing EU-DERPO actual-F current logprob for sample {exc.args[0]}") from exc

    @staticmethod
    def _router_parameters(router):
        parameters = dict(router.named_parameters(recurse=False))
        if set(parameters) != {"weight"}:
            raise RuntimeError(
                f"EU-DERPO V1.2.1 supports the frozen Router parameter set {{weight}}, got {sorted(parameters)}"
            )
        return tuple(parameters.items())

    def _coordinated_ram_preflight(self, local_required):
        device = (
            torch.device("cuda", torch.cuda.current_device())
            if torch.cuda.is_available()
            else self.routers[0].weight.device
        )
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            if torch.distributed.get_world_size() != 8:
                raise RuntimeError("EU-DERPO V1.2.1 node RAM preflight requires one world8 node")
            local = torch.tensor([int(local_required)], dtype=torch.int64, device=device)
            gathered = [torch.zeros_like(local) for _ in range(8)]
            torch.distributed.all_gather(gathered, local)
            required_by_rank = [int(item.item()) for item in gathered]
            payload = torch.zeros(2, dtype=torch.int64, device=device)
            if torch.distributed.get_rank() == 0:
                available = _read_mem_available()
                decision = _node_ram_decision(
                    required_by_rank, available, HOST_RAM_SAFETY_MARGIN_BYTES
                )
                payload[0] = available
                payload[1] = int(decision["passed"])
            torch.distributed.broadcast(payload, src=0)
            decision = _node_ram_decision(
                required_by_rank, int(payload[0].item()), HOST_RAM_SAFETY_MARGIN_BYTES
            )
            if bool(payload[1].item()) != decision["passed"]:
                raise RuntimeError("EU-DERPO node RAM preflight broadcast was inconsistent")
        else:
            try:
                available = _read_mem_available()
            except (OSError, RuntimeError):
                available = (1 << 63) - 1
            decision = _node_ram_decision(
                [local_required], available, HOST_RAM_SAFETY_MARGIN_BYTES
            )
        if not decision["passed"]:
            raise MemoryError(f"EU-DERPO node RAM preflight failed: {decision}")
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            print(
                "EU-DERPO host RAM preflight: "
                f"MemAvailable_before_actor_update={decision['mem_available']} "
                f"EU_cache_required_bytes={local_required} "
                f"node_EU_cache_required_bytes={decision['node_required']}",
                flush=True,
            )
        return decision

    def wrap_native_finalize(self, original_finalize):
        if original_finalize is None:
            raise RuntimeError("EU-DERPO requires MCore native finalize_model_grads")

        @wraps(original_finalize)
        def wrapped(*args, **kwargs):
            if self._native_finalize_count != 0 or self._native_finalize_completed:
                raise RuntimeError("EU-DERPO native finalize_model_grads was invoked more than once")
            result = original_finalize(*args, **kwargs)
            self._native_finalize_count += 1
            self._native_finalize_completed = True
            return result

        return wrapped

    def _assert_native_finalize(self):
        if self._native_finalize_count != 1 or not self._native_finalize_completed:
            raise RuntimeError(
                "EU-DERPO Router-only Step E requires one completed native finalize_model_grads"
            )

    def start_main_batch(
        self,
        sample_ids: torch.Tensor,
        prompt_groups: torch.Tensor,
        attention_mask: torch.Tensor,
        response_mask: torch.Tensor,
        response_length: int,
        optimizer_generation: int,
    ):
        ids = self._ids(sample_ids, sample_ids.numel())
        if len(set(ids)) != len(ids):
            raise RuntimeError("duplicate EU-DERPO main sample id")
        self.route_cache.clear()
        self._ordered_route_cache.clear()
        self._route_metadata_cache.clear()
        self.current_logprob_cache.clear()
        self.mode = "main"
        self._sample_row = {sample_id: row for row, sample_id in enumerate(ids)}
        groups = self._ids(prompt_groups, sample_ids.numel())
        self._prompt_group_by_sample = dict(zip(ids, groups))
        device = self.routers[0].weight.device
        from megatron.core import parallel_state as mpu

        if getattr(self.tf_config, "hidden_size", self.hidden_size) != self.hidden_size:
            raise RuntimeError("EU-DERPO Router hidden-size contract changed")
        if self.hidden_size == 2048 and self.routers[0].weight.dtype != torch.bfloat16:
            raise RuntimeError("EU-DERPO V1.2.1 production Router must remain BF16")
        plan = _plan_local_response_rows(
            attention_mask,
            response_mask,
            sample_ids,
            response_length,
            mpu.get_tensor_model_parallel_world_size(),
            mpu.get_tensor_model_parallel_rank(),
        )
        byte_plan = _cache_byte_plan(plan["rows"], len(self.routers), self.hidden_size, self.topk)
        ram = {"node_required": 0, "mem_available": 0, "safety_margin": 0}
        if not self.v15:
            ram = self._coordinated_ram_preflight(sum(byte_plan.values()))
        self._hidden_cache = None if self.v15 else [
            torch.empty((plan["rows"], self.hidden_size), dtype=torch.bfloat16)
            for _ in self.routers
        ]
        self._support_cache = None if self.v15 else [
            torch.empty((plan["rows"], self.topk), dtype=torch.uint8)
            for _ in self.routers
        ]
        if not self.v15 and any(tensor.is_pinned() for tensor in self._hidden_cache + self._support_cache):
            raise RuntimeError("EU-DERPO full cache must use pageable CPU memory")
        if self.v15:
            byte_plan = {"hidden": 0, "support": 0, "metadata": 0}
        self._provenance = {
            key: plan[key] for key in ("sample_row", "sample_id", "response_position")
        }
        self._planned_spans = plan["spans"]
        self._cache_cursor_by_layer = [0] * len(self.routers)
        self._cache_consumed_by_layer = [0] * len(self.routers)
        self._cache_plan = byte_plan
        self._cache_metrics = {
            "planned_valid_rows": plan["rows"],
            "hidden_cache_bytes": byte_plan["hidden"],
            "support_cache_bytes": byte_plan["support"],
            "semantic_metadata_bytes": byte_plan["metadata"],
            "pageable_cache_peak_bytes": sum(byte_plan.values()),
            "eu_cache_required_bytes": sum(byte_plan.values()),
            "node_required_cache_bytes": ram["node_required"],
            "mem_available_before_allocation_bytes": ram["mem_available"],
            "mem_available_before_actor_update_bytes": ram["mem_available"],
            "host_safety_margin_bytes": ram["safety_margin"],
        }
        self._native_finalize_count = 0
        self._native_finalize_completed = False
        self._optimizer_generation = int(optimizer_generation)
        self._parameter_snapshot = []
        for router in self.routers:
            snapshot = []
            for name, parameter in self._router_parameters(router):
                snapshot.append((name, parameter, id(parameter), parameter._version))
            self._parameter_snapshot.append(tuple(snapshot))
        self._step_e_started = False
        self._d2h_seconds = 0.0
        self._h2d_seconds = 0.0
        self._aux_reduce_count = [0] * len(self.routers)
        self._main_grad_add_count = [0] * len(self.routers)
        self._step_e_complete = False
        self._full_auxiliary_transformer_forward_count = 0
        self._step_e_natural_topk_call_count = 0
        self._step_e_router_forward_call_count = 0
        self._step_e_routing_call_count = 0
        self._step_e_dispatch_count = 0
        self._sample_ids_by_row = sample_ids.to(device)
        self._credit_mask = (
            torch.zeros((len(ids), response_length), dtype=torch.bool, device=device)
            if self.v15
            else None
        )
        shape = (len(ids), len(self.routers), self.num_experts)
        self._utility_sum = torch.zeros(shape, dtype=torch.float32, device=device)
        self._utility_sum_sq = torch.zeros_like(self._utility_sum) if self.diagnostics else None
        self._utility_count = torch.zeros_like(self._utility_sum)
        self._main_hook_count = 0
        self._main_hook_count_by_layer = torch.zeros(len(self.routers), dtype=torch.int64, device=device)
        self._expected_main_hook_count = len(ids) * len(self.routers)
        self._edge_count_by_layer = torch.zeros(len(self.routers), dtype=torch.int64, device=device)
        self._route_mismatch_by_layer = torch.zeros(len(self.routers), dtype=torch.int64, device=device)
        self._forward_recompute_mismatch_by_layer = torch.zeros_like(self._route_mismatch_by_layer)
        self._route_compare_by_layer = torch.zeros_like(self._route_mismatch_by_layer)
        self._first_route_mismatch = {
            phase: {
                "found": torch.zeros((), dtype=torch.bool, device=device),
                "sample_row": torch.full((), -1, dtype=torch.int64, device=device),
                "sample_id": torch.full((), -1, dtype=torch.int64, device=device),
                "response_token": torch.full((), -1, dtype=torch.int64, device=device),
                "global_layer": torch.full((), -1, dtype=torch.int64, device=device),
                "expected": torch.full((self.topk,), -1, dtype=torch.int64, device=device),
                "actual": torch.full((self.topk,), -1, dtype=torch.int64, device=device),
                "topk_intersection": torch.full((), -1, dtype=torch.int64, device=device),
            }
            for phase in ("forward_recompute",)
        }
        self._alpha_summary.clear()
        self._alpha_min = torch.tensor(float("inf"), dtype=torch.float32, device=device)
        self._invalid_flag = torch.zeros((), dtype=torch.int32, device=device)
        self._center_residual_max = torch.zeros((), dtype=torch.float32, device=device)
        for queue in self._pending_recompute:
            queue.clear()

    @torch.no_grad()
    def begin_main_microbatch(self, input_ids, attention_mask, sample_ids, response_mask, response_length):
        ids = self._ids(sample_ids, input_ids.shape[0])
        if len(ids) != 1:
            raise RuntimeError("EU-DERPO V1.2.1 frozen cache planner requires microbatch=1")
        rows = torch.tensor([self._sample_row[sample_id] for sample_id in ids], dtype=torch.int64)[:, None]
        rows = rows.expand(-1, response_length)
        packed_rows = RouterShiftObserver._pack_sequence_parallel(rows[..., None], input_ids, attention_mask, response_length).squeeze(-1)
        packed_valid = RouterShiftObserver._pack_sequence_parallel(response_mask[..., None], input_ids, attention_mask, response_length).squeeze(-1).bool()
        tokens = torch.arange(response_length, dtype=torch.int64)[None, :].expand(input_ids.shape[0], -1)
        packed_tokens = RouterShiftObserver._pack_sequence_parallel(
            tokens[..., None], input_ids, attention_mask, response_length
        ).squeeze(-1)
        start, end = self._planned_spans[ids[0]]
        actual_positions = packed_tokens[packed_valid].detach().cpu().to(torch.int32)
        expected_positions = self._provenance["response_position"][start:end]
        if not torch.equal(actual_positions, expected_positions):
            raise RuntimeError(
                "EU-DERPO planned response rows differ from actual packed SP-local rows"
            )
        if not torch.equal(
            packed_rows[packed_valid].detach().cpu().to(torch.int32),
            self._provenance["sample_row"][start:end],
        ):
            raise RuntimeError("EU-DERPO planned sample rows differ from actual packed rows")
        self._active = (packed_rows.long(), packed_valid, packed_tokens.long(), start, end)
        self._forward_routes = [None] * len(self.routers)
        if self.route_attribution:
            self._forward_ordered_routes = [None] * len(self.routers)
            self._main_route_metadata = self._route_metadata(
                input_ids, attention_mask, sample_ids, response_mask, response_length
            ).cpu()

    @torch.no_grad()
    def finish_main_microbatch(self, input_ids, attention_mask, sample_ids, response_length):
        routes = RouterShiftObserver._unpack_sequence_parallel(
            self._forward_routes, input_ids, attention_mask
        )[:, -response_length - 1 : -1]
        if self.route_attribution:
            ordered_routes = RouterShiftObserver._unpack_sequence_parallel(
                self._forward_ordered_routes, input_ids, attention_mask
            )[:, -response_length - 1 : -1]
        expected = (input_ids.shape[0], response_length, len(self.routers), self.topk)
        if routes.shape != expected or routes.dtype != torch.uint8:
            raise RuntimeError(f"EU-DERPO actual-F route shape/dtype mismatch: {routes.shape}, {routes.dtype}")
        if self.route_attribution and (ordered_routes.shape != expected or ordered_routes.dtype != torch.uint8):
            raise RuntimeError("EU-DERPO attribution ordered-route shape/dtype mismatch")
        for row, sample_id in enumerate(self._ids(sample_ids, input_ids.shape[0])):
            if sample_id in self.route_cache:
                raise RuntimeError(f"duplicate EU-DERPO actual-F sample id: {sample_id}")
            self.route_cache[sample_id] = routes[row].cpu()
            if self.route_attribution:
                self._ordered_route_cache[sample_id] = ordered_routes[row].cpu()
                self._route_metadata_cache[sample_id] = self._main_route_metadata[row]
        self._active = None
        self._forward_routes = None
        self._forward_ordered_routes = None
        self._main_route_metadata = None
        return routes

    @torch.no_grad()
    def record_main_logprobs(self, sample_ids, routes, current_logprobs, credit_mask=None):
        ids = self._ids(sample_ids, current_logprobs.shape[0])
        cached = self.routes_for(sample_ids)
        if not torch.equal(cached.long().sort(-1).values, routes.detach().cpu().long().sort(-1).values):
            raise RuntimeError("EU-DERPO loss closure did not receive its same-F route snapshot")
        if current_logprobs.ndim != 2 or not torch.isfinite(current_logprobs).all():
            raise RuntimeError("EU-DERPO actual-F current logprob is missing or non-finite")
        for row, sample_id in enumerate(ids):
            if sample_id in self.current_logprob_cache:
                raise RuntimeError(f"duplicate EU-DERPO actual-F current logprob: {sample_id}")
            self.current_logprob_cache[sample_id] = current_logprobs[row].detach().float().cpu()
        if self.v15:
            if credit_mask is None or credit_mask.shape != current_logprobs.shape:
                raise RuntimeError("EU-DERPO V1.5 requires an aligned token credit mask")
            rows = torch.tensor([self._sample_row[item] for item in ids], dtype=torch.long, device=credit_mask.device)
            self._credit_mask[rows] = credit_mask.detach().bool()

    def _observe_main(self, index, module, hidden, output):
        actual_set = self._selected(output[1], self.topk)
        if torch.is_grad_enabled() and output[0].requires_grad:
            self._record_execution("R", module)
            if not self._pending_recompute[index]:
                raise RuntimeError("EU-DERPO recompute has no matching original forward")
            context, forward_routes = self._pending_recompute[index].popleft()
            record = self._pending_alpha.pop(index)
            actual = record["routes"]
            rows, valid, tokens = context
            self._check_same_invocation(index, actual, actual_set, rows, tokens, valid)
            self._check_route("forward_recompute", index, forward_routes, actual_set, rows, tokens, valid)
            alpha = record["alpha"]
            actual_alpha = output[0].gather(-1, actual)
            self._mark_invalid_alpha(alpha, actual_alpha)
            selected = alpha[valid]
            if selected.numel():
                self._alpha_min.copy_(torch.minimum(self._alpha_min, selected.detach().min()))
                if self.diagnostics:
                    self._alpha_summary.append(selected.reshape(-1, self.topk)[:32].detach().clone())
            record.update(layer=index, rows=rows, tokens=tokens, valid=valid, ready=True)
        else:
            self._record_execution("F", module)
            if self._active is None:
                raise RuntimeError("EU-DERPO original forward has no microbatch context")
            rows, valid, tokens, start, end = self._active
            flat_hidden = hidden.detach().reshape(-1, hidden.shape[-1])
            if flat_hidden.shape != (actual_set.shape[0], self.hidden_size):
                raise RuntimeError(
                    f"EU-DERPO actual-F hidden shape mismatch: {tuple(flat_hidden.shape)}"
                )
            if end - start != int(valid.sum().item()):
                raise RuntimeError("EU-DERPO actual-F cache span cardinality mismatch")
            if not self.v15 and self._cache_cursor_by_layer[index] != start:
                raise RuntimeError("EU-DERPO actual-F hidden cache write cursor mismatch")
            selected_support = actual_set[valid]
            if selected_support.numel() and (
                selected_support.min().item() < 0
                or selected_support.max().item() >= self.num_experts
                or not (selected_support.sort(-1).values.diff(dim=-1) > 0).all()
            ):
                raise RuntimeError("EU-DERPO actual-F support IDs are invalid or duplicated")
            if not self.v15:
                copy_started = time.perf_counter()
                self._hidden_cache[index][start:end].copy_(
                    flat_hidden[valid].to(device="cpu", dtype=torch.bfloat16), non_blocking=False
                )
                self._support_cache[index][start:end].copy_(
                    selected_support.to(device="cpu", dtype=torch.uint8), non_blocking=False
                )
                self._d2h_seconds += time.perf_counter() - copy_started
                self._cache_cursor_by_layer[index] = end
            self._forward_routes[index] = actual_set.detach().to(torch.uint8)
            if self.route_attribution:
                self._forward_ordered_routes[index] = self._ordered_selected(output[0], actual_set).to(torch.uint8)
            self._pending_recompute[index].append(((rows, valid, tokens), actual_set.detach()))

    @torch.no_grad()
    def _check_route(self, phase, index, expected, actual, rows, tokens, valid):
        differences = ~expected.long().sort(-1).values.eq(actual.long().sort(-1).values)
        mismatch = differences[valid].sum()
        if phase != "forward_recompute":
            raise RuntimeError(f"unsupported EU-DERPO route comparison phase: {phase}")
        self._forward_recompute_mismatch_by_layer[index] += mismatch
        self._route_mismatch_by_layer[index] += mismatch
        self._route_compare_by_layer[index] += valid.sum() * self.topk
        self._remember_first_route_mismatch(phase, index, expected, actual, rows, tokens, differences.any(-1) & valid)

    @torch.no_grad()
    def _remember_first_route_mismatch(self, phase, index, expected, actual, rows, tokens, mismatch):
        record = self._first_route_mismatch[phase]
        packed = mismatch.to(torch.int64).argmax()
        take = ~record["found"] & mismatch.any()
        expected_route = expected[packed].long()
        actual_route = actual[packed].long()
        sample_row = rows[packed]
        from megatron.core import parallel_state as mpu

        values = {
            "sample_row": sample_row,
            "sample_id": self._sample_ids_by_row[sample_row],
            "response_token": tokens[packed],
            "global_layer": torch.as_tensor(
                mpu.get_pipeline_model_parallel_rank() * len(self.routers) + index,
                dtype=torch.int64,
                device=expected.device,
            ),
            "expected": expected_route,
            "actual": actual_route,
            "topk_intersection": (expected_route[:, None] == actual_route[None, :]).any(-1).sum(),
        }
        for key, value in values.items():
            record[key].copy_(torch.where(take, value, record[key]))
        record["found"].bitwise_or_(take)

    def _first_route_metrics(self):
        result = {}
        for phase, record in self._first_route_mismatch.items():
            if record["found"].item():
                result[phase] = {
                    key: value.tolist() if value.ndim else value.item()
                    for key, value in record.items()
                    if key != "found"
                }
        return result

    @torch.no_grad()
    def _check_same_invocation(self, index, saved, routing_map, rows, tokens, valid):
        differences = ~saved.long().sort(-1).values.eq(routing_map.long().sort(-1).values)
        mismatch = differences.any(-1)
        if mismatch.any():
            packed = mismatch.nonzero(as_tuple=False)[0, 0]
            raise RuntimeError(
                "EU-DERPO saved-tensor Top-K differs from routing-map Top-K in one Router invocation: "
                f"sample={int(rows[packed].item())}, response_token={int(tokens[packed].item())}, "
                f"valid_response={bool(valid[packed].item())}, local_layer={index}, "
                f"saved={saved[packed].long().tolist()}, "
                f"routing_map={routing_map[packed].long().tolist()}"
            )

    @torch.no_grad()
    def _check_aux_route(self, index, expected, actual, valid, expected_ordered=None, actual_ordered=None):
        mismatch = (~expected.long().sort(-1).values.eq(actual.long().sort(-1).values))[valid].sum()
        self._route_mismatch_by_layer[index] += mismatch
        if not self.route_attribution:
            return None
        expected_metadata, actual_metadata = self._aux_attribution[1:]
        comparison = _route_attribution_comparison(
            expected_ordered, actual_ordered, expected_metadata, actual_metadata
        )
        self._aux_attribution_counts[0, index] += valid.sum()
        self._aux_attribution_counts[1, index] += (comparison["order_only"] & valid).sum()
        self._aux_attribution_counts[2, index] += (comparison["set_mismatch"] & valid).sum()
        return comparison

    @torch.no_grad()
    def _alpha_grad(self, record, grad):
        if not record["ready"]:
            raise RuntimeError("EU-DERPO actual-alpha gradient fired before route metadata was ready")
        layer, routes, rows, valid = record["layer"], record["routes"], record["rows"], record["valid"]
        sensitivity = -grad.float()
        utility = (
            local_rms_routing_utility(record["alpha"], sensitivity, self.eps_u)
            if self.v15
            else centered_routing_utility(record["alpha"], sensitivity)
        )
        center_residual = (record["alpha"].float() * utility).sum(-1).abs().max()
        self._center_residual_max.copy_(torch.maximum(self._center_residual_max, center_residual))
        self._invalid_flag.bitwise_or_((center_residual > 2.0e-5).to(torch.int32))
        self._invalid_flag.bitwise_or_((~torch.isfinite(utility).all()).to(torch.int32))
        if self.v15:
            credit = self._credit_mask.to(rows.device)[rows, record["tokens"]]
            valid = valid & credit
        edge_valid = valid[:, None].expand_as(routes).reshape(-1)
        flat_rows = rows[:, None].expand_as(routes).reshape(-1)[edge_valid]
        flat_routes = routes.reshape(-1)[edge_valid]
        flat_layers = torch.full_like(flat_rows, layer)
        indices = (flat_rows, flat_layers, flat_routes)
        selected_utility = utility.reshape(-1)[edge_valid]
        self._utility_sum.index_put_(indices, selected_utility, accumulate=True)
        if self._utility_sum_sq is not None:
            self._utility_sum_sq.index_put_(indices, selected_utility.square(), accumulate=True)
        self._utility_count.index_put_(indices, torch.ones_like(selected_utility), accumulate=True)
        self._edge_count_by_layer[layer] += edge_valid.sum()
        self._main_hook_count += 1
        self._main_hook_count_by_layer[layer] += 1

    @torch.no_grad()
    def _gather_variable(self, values, group):
        if torch.distributed.get_world_size(group) == 1:
            return values
        size = torch.tensor([values.shape[0]], dtype=torch.int64, device=values.device)
        sizes = [torch.zeros_like(size) for _ in range(torch.distributed.get_world_size(group))]
        torch.distributed.all_gather(sizes, size, group=group)
        maximum = max(item.item() for item in sizes)
        padded = torch.zeros((maximum, *values.shape[1:]), dtype=values.dtype, device=values.device)
        padded[: values.shape[0]] = values
        gathered = [torch.empty_like(padded) for _ in sizes]
        torch.distributed.all_gather(gathered, padded, group=group)
        return torch.cat([item[: length.item()] for item, length in zip(gathered, sizes, strict=True)])

    @torch.no_grad()
    def finish_main_batch(self):
        self._assert_native_finalize()
        if self._pending_alpha:
            raise RuntimeError("EU-DERPO has unconsumed actual-alpha records")
        if any(self._pending_recompute):
            raise RuntimeError("EU-DERPO backward missed a checkpoint recompute")
        layer_mismatch = self._global_layer_mismatch()
        forward_recompute_layer_mismatch = self._global_layer_values(self._forward_recompute_mismatch_by_layer)
        mismatch = int(layer_mismatch.sum().item())
        forward_recompute_mismatch = int(forward_recompute_layer_mismatch.sum().item())
        compared = int(self._global_layer_values(self._route_compare_by_layer).sum().item())
        invalid = self._global_invalid()
        metrics = {
            "route_mismatch_count": mismatch,
            "route_equal_fraction": (compared - mismatch) / compared if compared else 0.0,
            "layer_mismatch_count": layer_mismatch.cpu().tolist(),
            "forward_recompute_mismatch_count": forward_recompute_mismatch,
            "forward_recompute_equal_fraction": (
                (compared - forward_recompute_mismatch) / compared if compared else 0.0
            ),
            "route_match_currentF_checkpointR": (
                (compared - forward_recompute_mismatch) / compared if compared else 0.0
            ),
            "forward_recompute_layer_mismatch_count": forward_recompute_layer_mismatch.cpu().tolist(),
            "first_route_mismatch": self._first_route_metrics(),
            "route_phase_execution": self._phase_execution.copy(),
            "gradient_hook_count": self._main_hook_count,
            "expected_gradient_hook_count": self._expected_main_hook_count,
            "gradient_hook_count_by_layer": self._main_hook_count_by_layer.cpu().tolist(),
            "utility_edge_count_by_layer": self._edge_count_by_layer.cpu().tolist(),
            "weighted_center_max_abs": self._center_residual_max.item(),
            "native_finalize_count": self._native_finalize_count,
            "native_finalize_completed": float(self._native_finalize_completed),
            "captured_valid_rows": 0 if self.v15 else self._cache_plan["hidden"] // (
                len(self.routers) * self.hidden_size * 2
            ),
            "d2h_capture_seconds": self._d2h_seconds,
            **self._cache_metrics,
        }
        if not self.v15 and any(
            cursor != self._cache_metrics["planned_valid_rows"] for cursor in self._cache_cursor_by_layer
        ):
            raise RuntimeError("EU-DERPO planned/captured/final hidden cache rows differ")
        if mismatch or forward_recompute_mismatch:
            raise RuntimeError(f"EU-DERPO actual-F/recompute natural route mismatch: {metrics}")
        if invalid:
            raise FloatingPointError("EU-DERPO actual alpha or relative routing utility is invalid")
        hook_invalid = torch.tensor(
            int(
                self._main_hook_count != self._expected_main_hook_count
                or not torch.all(self._main_hook_count_by_layer == len(self._sample_row))
            ),
            dtype=torch.int32,
            device=self._utility_sum.device,
        )
        from megatron.core import parallel_state as mpu

        if mpu.get_tensor_model_parallel_world_size() > 1:
            torch.distributed.all_reduce(hook_invalid, op=torch.distributed.ReduceOp.MAX, group=mpu.get_tensor_model_parallel_group())
        if mpu.get_pipeline_model_parallel_world_size() > 1:
            torch.distributed.all_reduce(hook_invalid, op=torch.distributed.ReduceOp.MAX, group=mpu.get_pipeline_model_parallel_group())
        if hook_invalid.item():
            raise RuntimeError("EU-DERPO recompute Router hook count differs on at least one rank")
        validate_recompute_hook_count(self._main_hook_count, self._expected_main_hook_count)
        expected_edges = self._utility_count.sum((0, 2)).long()
        validate_recompute_edge_counts(self._edge_count_by_layer, expected_edges)

        if mpu.get_tensor_model_parallel_world_size() > 1:
            torch.distributed.all_reduce(self._utility_sum, group=mpu.get_tensor_model_parallel_group())
            if self._utility_sum_sq is not None:
                torch.distributed.all_reduce(self._utility_sum_sq, group=mpu.get_tensor_model_parallel_group())
            torch.distributed.all_reduce(self._utility_count, group=mpu.get_tensor_model_parallel_group())
        if self._alpha_summary:
            alpha = torch.cat(self._alpha_summary, dim=0)
            if mpu.get_tensor_model_parallel_world_size() > 1:
                torch.distributed.all_reduce(
                    self._alpha_min, op=torch.distributed.ReduceOp.MIN, group=mpu.get_tensor_model_parallel_group()
                )
                alpha = self._gather_variable(alpha, mpu.get_tensor_model_parallel_group())
            if mpu.get_pipeline_model_parallel_world_size() > 1:
                torch.distributed.all_reduce(
                    self._alpha_min, op=torch.distributed.ReduceOp.MIN, group=mpu.get_pipeline_model_parallel_group()
                )
                alpha = self._gather_variable(alpha, mpu.get_pipeline_model_parallel_group())
            metrics.update({
                "alpha_min": self._alpha_min.item(),
                "alpha_p1": torch.quantile(alpha, 0.01).item(),
                "alpha_p5": torch.quantile(alpha, 0.05).item(),
                "alpha_median": torch.quantile(alpha, 0.5).item(),
                "router_entropy_sampled": (-(alpha * alpha.log()).sum(-1)).mean().item(),
            })
        self.mode = None
        utility_sum_sq = self._utility_sum_sq.cpu() if self._utility_sum_sq is not None else None
        return self._utility_sum.cpu(), utility_sum_sq, self._utility_count.cpu(), metrics

    @torch.no_grad()
    def _global_invalid(self):
        from megatron.core import parallel_state as mpu

        invalid = self._invalid_flag.clone()
        if mpu.get_tensor_model_parallel_world_size() > 1:
            torch.distributed.all_reduce(invalid, op=torch.distributed.ReduceOp.MAX, group=mpu.get_tensor_model_parallel_group())
        if mpu.get_pipeline_model_parallel_world_size() > 1:
            torch.distributed.all_reduce(invalid, op=torch.distributed.ReduceOp.MAX, group=mpu.get_pipeline_model_parallel_group())
        return bool(invalid.item())

    @torch.no_grad()
    def _global_layer_mismatch(self):
        return self._global_layer_values(self._route_mismatch_by_layer)

    @torch.no_grad()
    def _global_layer_values(self, values):
        from megatron.core import parallel_state as mpu

        result = torch.zeros(
            len(self.routers) * mpu.get_pipeline_model_parallel_world_size(),
            dtype=torch.int64,
            device=values.device,
        )
        offset = mpu.get_pipeline_model_parallel_rank() * len(self.routers)
        result[offset : offset + len(self.routers)] = values
        if mpu.get_tensor_model_parallel_world_size() > 1:
            torch.distributed.all_reduce(result, group=mpu.get_tensor_model_parallel_group())
        if mpu.get_pipeline_model_parallel_world_size() > 1:
            torch.distributed.all_reduce(result, group=mpu.get_pipeline_model_parallel_group())
        return result

    def _assert_parameters_unchanged(self, optimizer_generation):
        if int(optimizer_generation) != self._optimizer_generation:
            raise RuntimeError("EU-DERPO optimizer generation changed between F and Step E")
        for router, snapshot in zip(self.routers, self._parameter_snapshot):
            current = dict(self._router_parameters(router))
            for name, parameter, identity, version in snapshot:
                if current[name] is not parameter or id(current[name]) != identity:
                    raise RuntimeError("EU-DERPO Router Parameter object changed between F and Step E")
                if parameter._version != version:
                    raise RuntimeError("EU-DERPO Router Parameter version changed between F and Step E")

    def run_router_only_step_e(
        self,
        sample_ids,
        stats,
        normalized_utility,
        total_active,
        lambda_u,
        optimizer_generation,
    ):
        """Replay only Router gating on detached actual-F cache and merge its gradient."""
        self._assert_native_finalize()
        self._assert_parameters_unchanged(optimizer_generation)
        forbidden_counts = (
            self._full_auxiliary_transformer_forward_count,
            self._step_e_natural_topk_call_count,
            self._step_e_router_forward_call_count,
            self._step_e_routing_call_count,
            self._step_e_dispatch_count,
        )
        if any(forbidden_counts):
            raise RuntimeError("EU-DERPO V1.2.1 Step E observed a forbidden full-A/routing call")
        if self._step_e_started:
            raise RuntimeError("EU-DERPO Router-only Step E was invoked more than once")
        if self._hidden_cache is None or self._support_cache is None or self._provenance is None:
            raise RuntimeError("EU-DERPO Router-only Step E is missing actual-F cache")
        if not torch.equal(sample_ids.detach().cpu().long(), self._sample_ids_by_row.detach().cpu().long()):
            raise RuntimeError("EU-DERPO Router-only Step E sample ordering changed")
        self._step_e_started = True
        rows = self._cache_metrics["planned_valid_rows"]
        chunk_rows = _staging_chunk_rows(self.staging_mib, self.hidden_size, self.topk)
        allocation_rows = max(1, min(rows, chunk_rows))
        device = self.routers[0].weight.device
        use_pinned = device.type == "cuda"
        hidden_stage = torch.empty(
            (allocation_rows, self.hidden_size), dtype=torch.bfloat16, pin_memory=use_pinned
        )
        support_stage = torch.empty(
            (allocation_rows, self.topk), dtype=torch.uint8, pin_memory=use_pinned
        )
        coefficient_stage = torch.empty(
            (allocation_rows, self.topk), dtype=torch.float32, pin_memory=use_pinned
        )
        pinned_bytes = (
            hidden_stage.numel() * hidden_stage.element_size()
            + support_stage.numel() * support_stage.element_size()
            + coefficient_stage.numel() * coefficient_stage.element_size()
        ) if use_pinned else 0
        if pinned_bytes > self.staging_mib * 1024**2:
            raise RuntimeError("EU-DERPO pinned staging exceeded its configured cap")

        objective_total = 0.0
        native_router_grad_sq = 0.0
        utility_router_grad_sq = 0.0
        router_grad_dot = 0.0
        max_gpu_chunk_bytes = 0
        gpu_memory_before = torch.cuda.memory_allocated(device) if use_pinned else 0
        gpu_memory_peak = gpu_memory_before
        replay_started = time.perf_counter()
        for layer, router in enumerate(self.routers):
            parameters = self._router_parameters(router)
            accumulators = {
                name: torch.zeros_like(parameter, dtype=torch.float32, device=parameter.device)
                for name, parameter in parameters
            }
            cursor = 0
            while cursor < rows:
                end = min(rows, cursor + chunk_rows)
                length = end - cursor
                if self._cache_consumed_by_layer[layer] != cursor:
                    raise RuntimeError("EU-DERPO Step E cache consumption cursor mismatch")
                support = self._support_cache[layer][cursor:end]
                sample_rows = self._provenance["sample_row"][cursor:end]
                coefficient = _build_edge_coefficients(
                    stats,
                    normalized_utility,
                    total_active,
                    sample_rows,
                    layer,
                    support,
                    len(sample_ids),
                )
                hidden_stage[:length].copy_(self._hidden_cache[layer][cursor:end])
                support_stage[:length].copy_(support)
                coefficient_stage[:length].copy_(coefficient)
                transfer_started = time.perf_counter()
                hidden_device = hidden_stage[:length].to(
                    device=device, dtype=router.weight.dtype, non_blocking=False
                )
                support_uint8 = support_stage[:length].to(device=device, non_blocking=False)
                support_long = support_uint8.long()
                coefficient_device = coefficient_stage[:length].to(device=device, non_blocking=False)
                self._h2d_seconds += time.perf_counter() - transfer_started
                if hidden_device.requires_grad or coefficient_device.requires_grad:
                    raise RuntimeError("EU-DERPO Step E hidden/coefficient must be detached")
                with torch.enable_grad():
                    logits = router._verl_eu_derpo_original_gating(hidden_device)
                    loss, objective = _selected_support_loss(
                        logits, support_long, coefficient_device, lambda_u
                    )
                    gradients = torch.autograd.grad(
                        loss,
                        tuple(parameter for _, parameter in parameters),
                        retain_graph=False,
                        create_graph=False,
                        allow_unused=False,
                    )
                if not torch.isfinite(loss).all() or any(
                    not torch.isfinite(gradient).all() for gradient in gradients
                ):
                    raise FloatingPointError("EU-DERPO Router-only Step E produced a non-finite gradient")
                for (name, _), gradient in zip(parameters, gradients):
                    accumulators[name].add_(gradient.float())
                objective_total += objective.detach().float().item()
                self._cache_consumed_by_layer[layer] = end
                max_gpu_chunk_bytes = max(
                    max_gpu_chunk_bytes,
                    hidden_device.numel() * hidden_device.element_size()
                    + support_uint8.numel() * support_uint8.element_size()
                    + support_long.numel() * support_long.element_size()
                    + coefficient_device.numel() * coefficient_device.element_size()
                    + logits.numel() * logits.element_size(),
                )
                if use_pinned:
                    gpu_memory_peak = max(gpu_memory_peak, torch.cuda.memory_allocated(device))
                cursor = end

            for name, parameter in parameters:
                auxiliary_grad = accumulators[name]
                _reduce_router_auxiliary_grad(auxiliary_grad)
                self._aux_reduce_count[layer] += 1
                main_grad = getattr(parameter, "main_grad", None)
                if main_grad is None or main_grad.shape != auxiliary_grad.shape:
                    raise RuntimeError("EU-DERPO requires matching Megatron main_grad for Router Step E")
                if not torch.isfinite(main_grad).all() or not torch.isfinite(auxiliary_grad).all():
                    raise FloatingPointError("EU-DERPO main or reduced auxiliary Router grad is non-finite")
                with torch.no_grad():
                    before_native = main_grad.clone()
                    before = before_native.float()
                    native_router_grad_sq += before.square().sum().item()
                    utility_router_grad_sq += auxiliary_grad.square().sum().item()
                    router_grad_dot += (before * auxiliary_grad).sum().item()
                    auxiliary_native = auxiliary_grad.to(main_grad.dtype)
                    expected_after = before_native + auxiliary_native
                    main_grad.add_(auxiliary_native)
                    if not torch.isfinite(main_grad).all():
                        raise FloatingPointError("EU-DERPO combined Router main_grad is non-finite")
                    delta = main_grad.float() - before
                    expected_delta = expected_after.float() - before
                    if not torch.equal(delta, expected_delta):
                        raise RuntimeError("EU-DERPO main_grad delta differs from one auxiliary add")
                self._main_grad_add_count[layer] += 1

        if self._native_finalize_count != 1:
            raise RuntimeError("EU-DERPO native finalize count changed during Step E")
        if any(value != rows for value in self._cache_consumed_by_layer):
            raise RuntimeError("EU-DERPO Step E left an orphan or duplicate hidden chunk")
        if any(value != 1 for value in self._aux_reduce_count):
            raise RuntimeError("EU-DERPO auxiliary Router grad reduce count is not one per layer")
        if any(value != 1 for value in self._main_grad_add_count):
            raise RuntimeError("EU-DERPO Router main_grad add count is not one per layer")
        self._step_e_complete = True
        utility_norm = utility_router_grad_sq**0.5
        native_norm = native_router_grad_sq**0.5
        denominator = utility_norm * native_norm
        return {
            "utility_objective": objective_total,
            "native_router_grad_norm": native_norm,
            "utility_router_grad_norm": utility_norm,
            "utility_to_main_router_grad_ratio": utility_norm / (native_norm + 1.0e-12),
            "router_grad_cosine": router_grad_dot / denominator if denominator else 0.0,
            "full_auxiliary_transformer_forward_count": self._full_auxiliary_transformer_forward_count,
            "step_e_natural_topk_call_count": self._step_e_natural_topk_call_count,
            "step_e_router_forward_call_count": self._step_e_router_forward_call_count,
            "step_e_routing_call_count": self._step_e_routing_call_count,
            "step_e_dispatch_count": self._step_e_dispatch_count,
            "hidden_source_actual_f": 1.0,
            "support_source_actual_f": 1.0,
            "native_finalize_count": self._native_finalize_count,
            "aux_reduce_count_by_layer": self._aux_reduce_count.copy(),
            "main_grad_add_count_by_layer": self._main_grad_add_count.copy(),
            "all_cache_layers_consumed": float(all(value == rows for value in self._cache_consumed_by_layer)),
            "utility_edges_consumed": rows * len(self.routers) * self.topk,
            "orphan_cache_count": 0,
            "pinned_peak_bytes": pinned_bytes,
            "step_e_max_chunk_rows": min(rows, chunk_rows),
            "step_e_live_tensor_bytes_lower_bound": max_gpu_chunk_bytes,
            "step_e_gpu_memory_allocated_delta_bytes": max(0, gpu_memory_peak - gpu_memory_before),
            "h2d_replay_seconds": self._h2d_seconds,
            "router_only_step_e_seconds": time.perf_counter() - replay_started,
        }

    def validate_before_optimizer_step(self, optimizer_generation):
        self._assert_native_finalize()
        self._assert_parameters_unchanged(optimizer_generation)
        if self.v15:
            return
        if not self._step_e_complete:
            raise RuntimeError("EU-DERPO optimizer step attempted before Router-only Step E completed")
        if any(value != 1 for value in self._aux_reduce_count + self._main_grad_add_count):
            raise RuntimeError("EU-DERPO Router auxiliary reduce/add ledger is incomplete")

    @torch.no_grad()
    def v15_router_grad_norm(self, require_nonzero: bool = False) -> float:
        """Return the global Router-gradient L2 norm; smoke may require it nonzero."""

        if not self.v15:
            raise RuntimeError("V1.5 Router-gradient audit called for a legacy EU-DERPO version")
        grad_sq = torch.zeros((), dtype=torch.float64, device=self.routers[0].weight.device)
        for router in self.routers:
            main_grad = getattr(router.weight, "main_grad", None)
            if main_grad is None or not torch.isfinite(main_grad).all():
                raise RuntimeError("EU-DERPO V1.5 Router main_grad is missing or non-finite")
            grad_sq.add_(main_grad.double().square().sum())
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(grad_sq)
        norm = grad_sq.sqrt().item()
        if not math.isfinite(norm):
            raise RuntimeError("EU-DERPO V1.5 Router gradient must be finite")
        if require_nonzero and norm <= 0:
            raise RuntimeError("EU-DERPO V1.5 smoke requires a nonzero Router gradient")
        return norm

    def finish_optimizer_step(self, optimizer_generation):
        if int(optimizer_generation) != self._optimizer_generation + 1:
            raise RuntimeError("EU-DERPO optimizer step generation did not advance exactly once")
        self.clear()
        return {"cache_clear_count": 1, "orphan_cache_count": 0}

    def start_aux_batch(self, sample_ids, stats, normalized_utility, total_active, lambda_u):
        self._full_auxiliary_transformer_forward_count += 1
        self.mode = "aux"
        self._aux_stats = (sample_ids.clone(), stats, normalized_utility, total_active.float(), float(lambda_u))
        self._aux_objective = 0.0
        self._native_router_grad_sq = 0.0
        self._utility_router_grad_sq = 0.0
        self._router_grad_dot = 0.0
        if self.route_attribution:
            device = self.routers[0].weight.device
            self._aux_attribution_counts = torch.zeros((3, len(self.routers)), dtype=torch.int64, device=device)
            self._aux_first_set_mismatch = None

    @torch.no_grad()
    def begin_aux_microbatch(self, input_ids, attention_mask, sample_ids, response_mask, response_length):
        all_ids, stats, normalized, active_total, lambda_u = self._aux_stats
        row_by_id = {int(value): row for row, value in enumerate(all_ids.tolist())}
        rows = torch.tensor([row_by_id[value] for value in self._ids(sample_ids, input_ids.shape[0])])
        routes = self.routes_for(sample_ids)
        batch, tokens, layers, _ = routes.shape
        weights = torch.zeros_like(routes, dtype=torch.float32)
        for layer in range(layers):
            edge = routes[:, :, layer].long()
            count = stats.count[rows, layer].gather(1, edge.reshape(batch, -1)).reshape_as(edge)
            keep = stats.mask[rows, layer].gather(1, edge.reshape(batch, -1)).reshape_as(edge)
            utility = normalized[rows, layer].gather(1, edge.reshape(batch, -1)).reshape_as(edge)
            weights[:, :, layer] = keep * utility / (len(all_ids) * active_total[rows, None, None] * count.masked_fill(count == 0, 1.0))
        weights *= response_mask[:, :, None, None]
        packed_routes = RouterShiftObserver._pack_sequence_parallel(routes, input_ids, attention_mask, response_length).permute(1, 0, 2)
        packed_weights = RouterShiftObserver._pack_sequence_parallel(weights, input_ids, attention_mask, response_length).permute(1, 0, 2)
        packed_valid = RouterShiftObserver._pack_sequence_parallel(
            response_mask[..., None], input_ids, attention_mask, response_length
        ).squeeze(-1).bool()
        self._active = (packed_routes, packed_weights, packed_valid, lambda_u)
        if self.route_attribution:
            ordered = self._ordered_routes_for(sample_ids)
            packed_ordered = RouterShiftObserver._pack_sequence_parallel(
                ordered, input_ids, attention_mask, response_length
            ).permute(1, 0, 2)
            expected_metadata = torch.stack(
                [self._route_metadata_cache[int(sample_id)] for sample_id in sample_ids.tolist()]
            )
            actual_metadata = self._route_metadata(
                input_ids, attention_mask, sample_ids, response_mask, response_length
            )
            packed_expected_metadata = RouterShiftObserver._pack_sequence_parallel(
                expected_metadata, input_ids, attention_mask, response_length
            )
            packed_actual_metadata = RouterShiftObserver._pack_sequence_parallel(
                actual_metadata, input_ids, attention_mask, response_length
            )
            self._aux_attribution = (
                packed_ordered,
                packed_expected_metadata,
                packed_actual_metadata,
            )

    def finish_aux_microbatch(self):
        self._active = None
        self._aux_attribution = None

    @torch.no_grad()
    def _remember_aux_attribution(self, index, logits, expected, actual, actual_ordered, valid, comparison):
        mismatch = comparison["set_mismatch"] & valid
        if self._aux_first_set_mismatch is not None or not mismatch.any():
            return
        packed = int(mismatch.nonzero(as_tuple=False)[0, 0].item())
        expected_ordered, expected_metadata, actual_metadata = self._aux_attribution
        f_ordered = expected_ordered[index, packed].long()
        a_ordered = actual_ordered[packed].long()
        f_sorted = f_ordered.sort().values
        a_sorted = a_ordered.sort().values
        f_only = f_sorted[~(f_sorted[:, None] == a_sorted[None, :]).any(-1)]
        a_only = a_sorted[~(a_sorted[:, None] == f_sorted[None, :]).any(-1)]
        topn = min(self.topk + 4, self.num_experts)
        top_scores, top_ids = logits[packed].detach().float().topk(topn)
        top_positions = {int(expert): position + 1 for position, expert in enumerate(top_ids.tolist())}
        f_meta = expected_metadata[packed].long()
        a_meta = actual_metadata[packed].long()
        from megatron.core import parallel_state as mpu

        self._aux_first_set_mismatch = {
            "global_rank": torch.distributed.get_rank() if torch.distributed.is_initialized() else 0,
            "local_rank": int(os.environ.get("LOCAL_RANK", "0")),
            "tp_rank": mpu.get_tensor_model_parallel_rank(),
            "dp_rank": mpu.get_data_parallel_rank(),
            "ep_rank": mpu.get_expert_model_parallel_rank(),
            "layer_index": index,
            "module": type(self.routers[index]).__name__,
            "sample_id": int(a_meta[0].item()),
            "sample_row": self._sample_row.get(int(a_meta[0].item()), -1),
            "local_batch_index_F": int(f_meta[2].item()),
            "local_batch_index_A": int(a_meta[2].item()),
            "prompt_group": int(a_meta[1].item()),
            "original_token_position_F": int(f_meta[4].item()),
            "original_token_position_A": int(a_meta[4].item()),
            "response_token_F": int(f_meta[3].item()),
            "response_token_A": int(a_meta[3].item()),
            "packed_token_ordinal_F": int(f_meta[6].item()),
            "packed_token_ordinal_A": int(a_meta[6].item()),
            "sp_local_token_index": packed,
            "valid_F": bool(f_meta[5].item()),
            "valid_A": bool(a_meta[5].item()),
            "semantic_equal": bool(comparison["semantic_equal"][packed].item()),
            "F_ordered": f_ordered.tolist(),
            "A_ordered": a_ordered.tolist(),
            "F_sorted": f_sorted.tolist(),
            "A_sorted": a_sorted.tolist(),
            "ordered_equal": bool(comparison["ordered_equal"][packed].item()),
            "set_equal": bool(comparison["set_equal"][packed].item()),
            "intersection_count": int(comparison["intersection"][packed].item()),
            "F_only": f_only.tolist(),
            "A_only": a_only.tolist(),
            "F_only_A_top_positions": {
                int(expert): top_positions.get(int(expert)) for expert in f_only.tolist()
            },
            "A_top_ids": top_ids.tolist(),
            "A_top_scores": top_scores.tolist(),
            "A_kth_expert": int(top_ids[self.topk - 1].item()),
            "A_kplus1_expert": int(top_ids[self.topk].item()),
            "A_boundary_margin": float((top_scores[self.topk - 1] - top_scores[self.topk]).item()),
            "classification": (
                "GENUINE_SET_MISMATCH" if comparison["semantic_equal"][packed] else "ALIGNMENT_MISMATCH"
            ),
        }
        print("========== EU-DERPO ROUTE ATTRIBUTION ==========", flush=True)
        for key, value in self._aux_first_set_mismatch.items():
            print(f"{key}={value}", flush=True)
        print("================================================", flush=True)

    def _observe_aux(self, index, module, hidden, output):
        self._step_e_natural_topk_call_count += 1
        self._step_e_router_forward_call_count += 1
        self._step_e_routing_call_count += 1
        self._step_e_dispatch_count += 1
        if self._active is None:
            raise RuntimeError("EU-DERPO auxiliary forward has no microbatch context")
        routes, weights, valid, lambda_u = self._active
        actual = self._selected(output[1], self.topk)
        with torch.enable_grad():
            raw_logits = module._verl_eu_derpo_original_gating(hidden.detach())
            logits, actual = _canonicalize_router_auxiliary_inputs(
                hidden, raw_logits, output[0], output[1], actual, self.topk, self.num_experts
            )
            selected_logits = logits.float().gather(-1, actual)
            log_alpha = selected_logits.log_softmax(-1)
            actual_alpha = output[0].gather(-1, actual)
            if self.route_attribution:
                actual_ordered = self._ordered_selected(output[0], actual)
                comparison = self._check_aux_route(
                    index, routes[index], actual, valid, self._aux_attribution[0][index], actual_ordered
                )
                self._remember_aux_attribution(
                    index, logits, routes[index], actual, actual_ordered, valid, comparison
                )
            else:
                self._check_aux_route(index, routes[index], actual, valid)
            if actual_alpha.shape != log_alpha.shape:
                raise RuntimeError(
                    "EU-DERPO Router auxiliary selected-alpha shape contract failed: "
                    f"hidden={tuple(hidden.shape)}, raw_logits={tuple(raw_logits.shape)}, "
                    f"flattened_logits={tuple(logits.shape)}, probs={tuple(output[0].shape)}, "
                    f"routing_map={tuple(output[1].shape)}, actual={tuple(actual.shape)}, "
                    f"log_alpha={tuple(log_alpha.shape)}, actual_alpha={tuple(actual_alpha.shape)}, "
                    f"topk={self.topk}, num_experts={self.num_experts}"
                )
            self._mark_invalid_alpha(log_alpha.detach().exp(), actual_alpha)
            loss = -lambda_u * (weights[index].to(log_alpha.device) * log_alpha).sum()
            auxiliary_grad = torch.autograd.grad(loss, module.weight, create_graph=False)[0]
        self._invalid_flag.bitwise_or_(
            (~torch.isfinite(loss).all() | ~torch.isfinite(auxiliary_grad).all()).to(torch.int32)
        )
        _reduce_router_auxiliary_grad(auxiliary_grad)
        main_grad = getattr(module.weight, "main_grad", None)
        if main_grad is None:
            raise RuntimeError("EU-DERPO requires Megatron main_grad for Router-only auxiliary update")
        with torch.no_grad():
            if self.diagnostics:
                main = main_grad.float()
                auxiliary = auxiliary_grad.float()
                self._native_router_grad_sq += main.square().sum().item()
                self._utility_router_grad_sq += auxiliary.square().sum().item()
                self._router_grad_dot += (main * auxiliary).sum().item()
            main_grad.add_(auxiliary_grad.to(main_grad.dtype))
            self._aux_objective += (-loss.detach().float() / lambda_u).item()

    def finish_aux_batch(self):
        mismatch = int(self._global_layer_mismatch().sum().item())
        invalid = self._global_invalid()
        attribution = None
        if self.route_attribution:
            valid = self._global_layer_values(self._aux_attribution_counts[0])
            order_only = self._global_layer_values(self._aux_attribution_counts[1])
            set_mismatch = self._global_layer_values(self._aux_attribution_counts[2])
            attribution = _route_attribution_summary(valid, order_only, set_mismatch)
            attribution["first_local_set_mismatch"] = self._aux_first_set_mismatch
            set_fraction = torch.tensor(attribution["set_mismatch_fraction_by_layer"])
            from megatron.core import parallel_state as mpu

            if mpu.get_tensor_model_parallel_rank() == 0:
                print("EU-DERPO route attribution summary", flush=True)
                print("layer | valid | order_only | set_mismatch | fraction", flush=True)
                for layer, values in enumerate(zip(valid.tolist(), order_only.tolist(), set_mismatch.tolist(), set_fraction.tolist())):
                    print(f"{layer} | {values[0]} | {values[1]} | {values[2]} | {values[3]:.8g}", flush=True)
                print(
                    f"first_order_only_layer={attribution['first_order_only_layer']} "
                    f"first_set_mismatch_layer={attribution['first_set_mismatch_layer']}",
                    flush=True,
                )
        if mismatch:
            suffix = f": {attribution}" if attribution is not None else ""
            raise RuntimeError("EU-DERPO auxiliary natural route differs from the main route" + suffix)
        if invalid:
            raise FloatingPointError("EU-DERPO auxiliary alpha or Router gradient is invalid")
        metrics = {"utility_objective": self._aux_objective}
        if attribution is not None:
            metrics["route_attribution"] = attribution
        if self.diagnostics:
            grad_stats = torch.tensor(
                [self._native_router_grad_sq, self._utility_router_grad_sq, self._router_grad_dot],
                dtype=torch.float64,
                device=self.routers[0].weight.device,
            )
            from megatron.core import parallel_state as mpu

            if mpu.get_pipeline_model_parallel_world_size() > 1:
                torch.distributed.all_reduce(grad_stats, group=mpu.get_pipeline_model_parallel_group())
            native_norm = grad_stats[0].sqrt().item()
            utility_norm = grad_stats[1].sqrt().item()
            denominator = native_norm * utility_norm
            metrics.update({
                "native_router_grad_norm": native_norm,
                "utility_router_grad_norm": utility_norm,
                "utility_to_main_router_grad_ratio": utility_norm / (native_norm + 1.0e-12),
                "router_grad_cosine": grad_stats[2].item() / denominator if denominator else 0.0,
            })
        self.mode = None
        self._aux_stats = None
        self.route_cache.clear()
        self._ordered_route_cache.clear()
        self._route_metadata_cache.clear()
        self.current_logprob_cache.clear()
        self._credit_mask = None
        self._prepass_route_cache.clear()
        self._sample_row.clear()
        self._prompt_group_by_sample.clear()
        self._sample_ids_by_row = None
        self._utility_sum = None
        self._utility_sum_sq = None
        self._utility_count = None
        self._edge_count_by_layer = None
        self._main_hook_count_by_layer = None
        self._route_mismatch_by_layer = None
        self._forward_recompute_mismatch_by_layer = None
        self._route_compare_by_layer = None
        self._first_route_mismatch.clear()
        self._phase_execution.clear()
        self._alpha_summary.clear()
        self._alpha_min = None
        self._invalid_flag = None
        self._center_residual_max = None
        self._aux_attribution = None
        self._aux_attribution_counts = None
        self._aux_first_set_mismatch = None
        return metrics

    def clear(self):
        self.mode = None
        self.route_cache.clear()
        self._ordered_route_cache.clear()
        self._route_metadata_cache.clear()
        self.current_logprob_cache.clear()
        self._prepass_route_cache.clear()
        self._pending_alpha.clear()
        for queue in self._pending_recompute:
            queue.clear()
        self._active = None
        self._forward_routes = None
        self._forward_ordered_routes = None
        self._main_route_metadata = None
        self._prepass_routes = None
        self._prepass_ids = None
        self._sample_row.clear()
        self._prompt_group_by_sample.clear()
        self._utility_sum = None
        self._utility_sum_sq = None
        self._utility_count = None
        self._edge_count_by_layer = None
        self._main_hook_count_by_layer = None
        self._route_mismatch_by_layer = None
        self._forward_recompute_mismatch_by_layer = None
        self._route_compare_by_layer = None
        self._sample_ids_by_row = None
        self._first_route_mismatch.clear()
        self._phase_execution.clear()
        self._aux_stats = None
        self._aux_attribution = None
        self._aux_attribution_counts = None
        self._aux_first_set_mismatch = None
        self._alpha_summary.clear()
        self._alpha_min = None
        self._invalid_flag = None
        self._center_residual_max = None
        self._hidden_cache = None
        self._support_cache = None
        self._provenance = None
        self._planned_spans = None
        self._cache_cursor_by_layer = None
        self._cache_consumed_by_layer = None
        self._cache_plan = None
        self._cache_metrics = None
        self._native_finalize_count = 0
        self._native_finalize_completed = False
        self._optimizer_generation = None
        self._parameter_snapshot = None
        self._step_e_started = False
        self._step_e_complete = False
        self._aux_reduce_count = None
        self._main_grad_add_count = None
        self._d2h_seconds = 0.0
        self._h2d_seconds = 0.0
        self._full_auxiliary_transformer_forward_count = 0
        self._step_e_natural_topk_call_count = 0
        self._step_e_router_forward_call_count = 0
        self._step_e_routing_call_count = 0
        self._step_e_dispatch_count = 0

    def close(self):
        self.clear()
        for router in self.routers:
            original = getattr(router, "_verl_eu_derpo_original_gating", None)
            if original is not None:
                del router._verl_eu_derpo_original_gating
            original = getattr(router, "_verl_eu_derpo_original_routing", None)
            if original is not None:
                del router.routing
                del router._verl_eu_derpo_original_routing
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
