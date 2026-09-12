"""Memory-safe Megatron observer for frozen EU-DERPO V1.2."""

from __future__ import annotations

from collections import deque
from functools import wraps
from types import MethodType

import torch

from verl.trainer.ppo.eu_derpo import (
    centered_routing_utility,
    validate_recompute_edge_counts,
    validate_recompute_hook_count,
)
from verl.utils.debug.router_shift import RouterShiftObserver


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


class EUDERPOObserver:
    """Capture natural routes and stream actual-alpha routing credit during main backward."""

    def __init__(self, models, tf_config, diagnostics=False):
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
            "pipeline_model_parallel_size": 4,
            "expert_model_parallel_size": 2,
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
            raise ValueError("EU-DERPO prepass/main route identity requires zero model dropout")
        from megatron.core import parallel_state as mpu

        if mpu.get_data_parallel_world_size() != 1 or mpu.get_expert_data_parallel_world_size() != 1:
            raise ValueError("EU-DERPO V1.2 first integration requires dense DP=1 and Expert DP=1")

        self.tf_config = tf_config
        self.diagnostics = bool(diagnostics)
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
        if self.topk != 8 or self.num_experts != 128:
            raise ValueError("EU-DERPO V1.2 Qwen3 recipe requires 128 Experts and Top-K=8")
        if any(router.topk != self.topk or router.weight.shape[0] != self.num_experts for router in self.routers):
            raise ValueError("EU-DERPO requires uniform local Router shapes")
        if len(self.routers) != 12:
            raise ValueError("EU-DERPO PP4 requires exactly 12 local MoE layers per stage")

        self.mode = None
        self.route_cache = {}
        self._handles = []
        self._pending_alpha = {}
        self._pending_recompute = [deque() for _ in self.routers]
        self._active = None
        self._prepass_routes = None
        self._prepass_ids = None
        self._sample_row = {}
        self._utility_sum = None
        self._utility_sum_sq = None
        self._utility_count = None
        self._main_hook_count = 0
        self._main_hook_count_by_layer = None
        self._expected_main_hook_count = 0
        self._edge_count_by_layer = None
        self._route_mismatch_by_layer = None
        self._route_compare_by_layer = None
        self._alpha_summary = []
        self._alpha_min = None
        self._invalid_flag = None
        self._center_residual_max = None
        self._aux_stats = None
        self._aux_objective = 0.0
        self._native_router_grad_sq = 0.0
        self._utility_router_grad_sq = 0.0
        self._router_grad_dot = 0.0
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
                self._prepass_routes[index] = self._selected(output[1], self.topk).to(torch.uint8)
                return
            if self.mode == "main":
                self._observe_main(index, output)
                return
            if self.mode == "aux":
                self._observe_aux(index, module, inputs[0], output)

        return hook

    @torch.no_grad()
    def _mark_invalid_alpha(self, fp32_alpha, router_alpha):
        rtol = max(2.0e-5, 2 * torch.finfo(router_alpha.dtype).eps)
        invalid = (
            ~torch.isfinite(fp32_alpha).all()
            | (fp32_alpha <= 0).any()
            | ~torch.isclose(fp32_alpha, router_alpha.float(), rtol=rtol, atol=2.0e-6).all()
        )
        self._invalid_flag.maximum_(invalid.to(torch.int32))

    @staticmethod
    def _ids(sample_ids: torch.Tensor, batch_size: int) -> list[int]:
        if sample_ids.device.type != "cpu" or sample_ids.dtype != torch.int64 or sample_ids.shape != (batch_size,):
            raise RuntimeError("EU-DERPO sample ids must be CPU int64 row metadata")
        return sample_ids.tolist()

    def start_prepass_batch(self):
        self.clear()
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
            if sample_id in self.route_cache:
                raise RuntimeError(f"duplicate EU-DERPO sample id: {sample_id}")
            self.route_cache[sample_id] = routes[row]
        self._prepass_routes = None

    def finish_prepass_batch(self):
        if not self.route_cache:
            raise RuntimeError("EU-DERPO prepass captured no routes")
        self.mode = None

    def routes_for(self, sample_ids: torch.Tensor) -> torch.Tensor:
        try:
            return torch.stack([self.route_cache[int(sample_id)] for sample_id in sample_ids.tolist()])
        except KeyError as exc:
            raise RuntimeError(f"missing EU-DERPO prepass route for sample {exc.args[0]}") from exc

    def start_main_batch(self, sample_ids: torch.Tensor):
        ids = self._ids(sample_ids, sample_ids.numel())
        if len(set(ids)) != len(ids):
            raise RuntimeError("duplicate EU-DERPO main sample id")
        self.mode = "main"
        self._sample_row = {sample_id: row for row, sample_id in enumerate(ids)}
        device = self.routers[0].weight.device
        shape = (len(ids), len(self.routers), self.num_experts)
        self._utility_sum = torch.zeros(shape, dtype=torch.float32, device=device)
        self._utility_sum_sq = torch.zeros_like(self._utility_sum) if self.diagnostics else None
        self._utility_count = torch.zeros_like(self._utility_sum)
        self._main_hook_count = 0
        self._main_hook_count_by_layer = torch.zeros(len(self.routers), dtype=torch.int64, device=device)
        self._expected_main_hook_count = len(ids) * len(self.routers)
        self._edge_count_by_layer = torch.zeros(len(self.routers), dtype=torch.int64, device=device)
        self._route_mismatch_by_layer = torch.zeros(len(self.routers), dtype=torch.int64, device=device)
        self._route_compare_by_layer = torch.zeros_like(self._route_mismatch_by_layer)
        self._alpha_summary.clear()
        self._alpha_min = torch.tensor(float("inf"), dtype=torch.float32, device=device)
        self._invalid_flag = torch.zeros((), dtype=torch.int32, device=device)
        self._center_residual_max = torch.zeros((), dtype=torch.float32, device=device)
        for queue in self._pending_recompute:
            queue.clear()

    @torch.no_grad()
    def begin_main_microbatch(self, input_ids, attention_mask, sample_ids, response_mask, response_length):
        ids = self._ids(sample_ids, input_ids.shape[0])
        records = torch.stack([self.route_cache[sample_id] for sample_id in ids])
        packed_routes = RouterShiftObserver._pack_sequence_parallel(records, input_ids, attention_mask, response_length).permute(1, 0, 2)
        rows = torch.tensor([self._sample_row[sample_id] for sample_id in ids], dtype=torch.int64)[:, None]
        rows = rows.expand(-1, response_length)
        packed_rows = RouterShiftObserver._pack_sequence_parallel(rows[..., None], input_ids, attention_mask, response_length).squeeze(-1)
        packed_valid = RouterShiftObserver._pack_sequence_parallel(response_mask[..., None], input_ids, attention_mask, response_length).squeeze(-1).bool()
        self._active = (packed_routes, packed_rows.long(), packed_valid)

    def finish_main_microbatch(self):
        self._active = None

    def _observe_main(self, index, output):
        actual_set = self._selected(output[1], self.topk)
        if torch.is_grad_enabled() and output[0].requires_grad:
            if self._active is not None:
                context = self._active
            elif self._pending_recompute[index]:
                context = self._pending_recompute[index].popleft()
            else:
                raise RuntimeError("EU-DERPO recompute has no matching original forward")
            record = self._pending_alpha.pop(index)
            actual = record["routes"]
            expected, rows, valid = context[0][index], context[1], context[2]
            self._check_route(index, expected, actual, valid)
            alpha = record["alpha"]
            actual_alpha = output[0].gather(-1, actual)
            self._mark_invalid_alpha(alpha, actual_alpha)
            selected = alpha[valid]
            if selected.numel():
                self._alpha_min.minimum_(selected.detach().min())
                if self.diagnostics:
                    self._alpha_summary.append(selected.reshape(-1, self.topk)[:32].detach().clone())
            record.update(layer=index, rows=rows, valid=valid, ready=True)
        else:
            if self._active is None:
                raise RuntimeError("EU-DERPO original forward has no microbatch context")
            self._pending_recompute[index].append(self._active)

    @torch.no_grad()
    def _check_route(self, index, expected, actual, valid):
        mismatch = (~expected.long().sort(-1).values.eq(actual.long().sort(-1).values))[valid].sum()
        self._route_mismatch_by_layer[index] += mismatch
        self._route_compare_by_layer[index] += valid.sum() * self.topk

    @torch.no_grad()
    def _alpha_grad(self, record, grad):
        if not record["ready"]:
            raise RuntimeError("EU-DERPO actual-alpha gradient fired before route metadata was ready")
        layer, routes, rows, valid = record["layer"], record["routes"], record["rows"], record["valid"]
        sensitivity = -grad.float()
        utility = centered_routing_utility(record["alpha"], sensitivity)
        center_residual = (record["alpha"].float() * utility).sum(-1).abs().max()
        self._center_residual_max.maximum_(center_residual)
        self._invalid_flag.maximum_((center_residual > 2.0e-5).to(torch.int32))
        self._invalid_flag.maximum_((~torch.isfinite(utility).all()).to(torch.int32))
        edge_valid = valid[:, None].expand_as(routes).reshape(-1)
        flat_index = (rows[:, None] * self.num_experts + routes).reshape(-1)[edge_valid]
        selected_utility = utility.reshape(-1)[edge_valid]
        self._utility_sum[:, layer].view(-1).scatter_add_(0, flat_index, selected_utility)
        if self._utility_sum_sq is not None:
            self._utility_sum_sq[:, layer].view(-1).scatter_add_(0, flat_index, selected_utility.square())
        self._utility_count[:, layer].view(-1).scatter_add_(0, flat_index, torch.ones_like(selected_utility))
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
        if self._pending_alpha:
            raise RuntimeError("EU-DERPO has unconsumed actual-alpha records")
        if any(self._pending_recompute):
            raise RuntimeError("EU-DERPO backward missed a checkpoint recompute")
        layer_mismatch = self._global_layer_mismatch()
        mismatch = int(layer_mismatch.sum().item())
        compared = int(self._global_layer_values(self._route_compare_by_layer).sum().item())
        invalid = self._global_invalid()
        metrics = {
            "route_mismatch_count": mismatch,
            "route_equal_fraction": (compared - mismatch) / compared if compared else 0.0,
            "layer_mismatch_count": layer_mismatch.cpu().tolist(),
            "gradient_hook_count": self._main_hook_count,
            "expected_gradient_hook_count": self._expected_main_hook_count,
            "gradient_hook_count_by_layer": self._main_hook_count_by_layer.cpu().tolist(),
            "utility_edge_count_by_layer": self._edge_count_by_layer.cpu().tolist(),
            "weighted_center_max_abs": self._center_residual_max.item(),
        }
        if mismatch:
            raise RuntimeError(f"EU-DERPO prepass/main natural route mismatch: {metrics}")
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

    def start_aux_batch(self, sample_ids, stats, normalized_utility, total_active, lambda_u):
        self.mode = "aux"
        self._aux_stats = (sample_ids.clone(), stats, normalized_utility, total_active.float(), float(lambda_u))
        self._aux_objective = 0.0
        self._native_router_grad_sq = 0.0
        self._utility_router_grad_sq = 0.0
        self._router_grad_dot = 0.0

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

    def finish_aux_microbatch(self):
        self._active = None

    def _observe_aux(self, index, module, hidden, output):
        if self._active is None:
            raise RuntimeError("EU-DERPO auxiliary forward has no microbatch context")
        routes, weights, valid, lambda_u = self._active
        actual = self._selected(output[1], self.topk)
        self._check_route(index, routes[index], actual, valid)
        with torch.enable_grad():
            logits = module._verl_eu_derpo_original_gating(hidden.detach())
            log_alpha = logits.float().gather(-1, actual).log_softmax(-1)
            actual_alpha = output[0].gather(-1, actual)
            self._mark_invalid_alpha(log_alpha.detach().exp(), actual_alpha)
            loss = -lambda_u * (weights[index].to(log_alpha.device) * log_alpha).sum()
            auxiliary_grad = torch.autograd.grad(loss, module.weight, create_graph=False)[0]
        self._invalid_flag.maximum_(
            (~torch.isfinite(loss).all() | ~torch.isfinite(auxiliary_grad).all()).to(torch.int32)
        )
        from megatron.core import parallel_state as mpu

        if mpu.get_tensor_model_parallel_world_size() > 1:
            torch.distributed.all_reduce(auxiliary_grad, group=mpu.get_tensor_model_parallel_group())
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
        if mismatch:
            raise RuntimeError("EU-DERPO auxiliary natural route differs from the main route")
        if invalid:
            raise FloatingPointError("EU-DERPO auxiliary alpha or Router gradient is invalid")
        metrics = {"utility_objective": self._aux_objective}
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
                "router_grad_cosine": grad_stats[2].item() / denominator if denominator else 0.0,
            })
        self.mode = None
        self._aux_stats = None
        self.route_cache.clear()
        self._sample_row.clear()
        self._utility_sum = None
        self._utility_sum_sq = None
        self._utility_count = None
        self._edge_count_by_layer = None
        self._main_hook_count_by_layer = None
        self._route_mismatch_by_layer = None
        self._route_compare_by_layer = None
        self._alpha_summary.clear()
        self._alpha_min = None
        self._invalid_flag = None
        self._center_residual_max = None
        return metrics

    def clear(self):
        self.mode = None
        self.route_cache.clear()
        self._pending_alpha.clear()
        for queue in self._pending_recompute:
            queue.clear()
        self._active = None
        self._prepass_routes = None
        self._prepass_ids = None
        self._sample_row.clear()
        self._utility_sum = None
        self._utility_sum_sq = None
        self._utility_count = None
        self._edge_count_by_layer = None
        self._main_hook_count_by_layer = None
        self._route_mismatch_by_layer = None
        self._route_compare_by_layer = None
        self._aux_stats = None
        self._alpha_summary.clear()
        self._alpha_min = None
        self._invalid_flag = None
        self._center_residual_max = None

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
