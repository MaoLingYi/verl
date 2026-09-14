# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
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

import megatron.core
import torch
from megatron.core import dist_checkpointing, mpu
from megatron.core.dist_checkpointing.serialization import (
    get_default_load_sharded_strategy,
    get_default_save_sharded_strategy,
)
from megatron.core.dist_checkpointing.strategies.fully_parallel import (
    FullyParallelLoadStrategyWrapper,
    FullyParallelSaveStrategyWrapper,
)
from packaging import version
from torch.distributed.checkpoint import FileSystemWriter, save as torch_dist_save


_SYNC_DCP_COPY_AHEAD_BYTES = 64 * 1024**2


def _planned_tensor_bytes(plan):
    total = 0
    largest = 0
    for item in plan.items:
        tensor_data = getattr(item, "tensor_data", None)
        if tensor_data is None:
            continue
        size = 1
        for dim in tensor_data.size:
            size *= dim
        size *= torch._utils._element_size(tensor_data.properties.dtype)
        total += size
        largest = max(largest, size)
    return total, largest


class _TelemetryFileSystemWriter(FileSystemWriter):
    def __init__(self, path, stage_callback):
        super().__init__(
            path,
            thread_count=1,
            per_thread_copy_ahead=_SYNC_DCP_COPY_AHEAD_BYTES,
        )
        self._stage_callback = stage_callback

    def _report(self, stage, plan=None):
        if self._stage_callback is None:
            return
        extra = {"checkpoint_copy_ahead_bytes": _SYNC_DCP_COPY_AHEAD_BYTES}
        if plan is not None:
            total, largest = _planned_tensor_bytes(plan)
            extra.update(
                checkpoint_planned_tensor_bytes=total,
                checkpoint_largest_tensor_bytes=largest,
                checkpoint_streaming_target_bytes=_SYNC_DCP_COPY_AHEAD_BYTES + largest,
            )
        self._stage_callback(stage, extra)

    def prepare_local_plan(self, plan):
        result = super().prepare_local_plan(plan)
        self._report("checkpoint_dcp_after_local_plan", result)
        return result

    def prepare_global_plan(self, plans):
        result = super().prepare_global_plan(plans)
        self._report("checkpoint_dcp_after_global_plan")
        return result

    def write_data(self, plan, planner):
        self._report("checkpoint_dcp_before_staging", plan)
        result = super().write_data(plan, planner)
        self._report("checkpoint_dcp_after_write_data", plan)
        return result

    def finish(self, metadata, results):
        self._report("checkpoint_dcp_before_finalize")
        result = super().finish(metadata, results)
        self._report("checkpoint_dcp_after_finalize")
        return result


def _bounded_sync_torch_dist_strategy(stage_callback=None):
    from megatron.core.dist_checkpointing.strategies.torch import (
        MCoreSavePlanner,
        TorchDistSaveShardedStrategy,
        _replace_state_dict_keys_with_sharded_keys,
        mcore_to_pyt_state_dict,
    )

    class BoundedSyncTorchDistSaveStrategy(TorchDistSaveShardedStrategy):
        def save(self, sharded_state_dict, checkpoint_dir):
            sharded_state_dict, _, _ = _replace_state_dict_keys_with_sharded_keys(
                sharded_state_dict, self.keep_only_main_replica
            )
            pyt_state_dict = mcore_to_pyt_state_dict(sharded_state_dict, False)
            if stage_callback is not None:
                stage_callback("checkpoint_dcp_after_mcore_translation", None)
            writer = _TelemetryFileSystemWriter(checkpoint_dir, stage_callback)
            torch_dist_save(
                pyt_state_dict,
                storage_writer=writer,
                planner=MCoreSavePlanner(
                    dedup_replicated_tensors=not self.keep_only_main_replica,
                    flatten_state_dict=False,
                ),
            )

    return BoundedSyncTorchDistSaveStrategy("torch_dist", 1, thread_count=1)


def save_dist_checkpointing(
    sharded_state_dict,
    ckpt_path,
    async_save=False,
    content_metadata=None,
    stage_callback=None,
):
    validate_sharding_integrity = True
    # Get checkpointing strategies
    save_strategy = (
        get_default_save_sharded_strategy("torch_dist")
        if async_save
        else _bounded_sync_torch_dist_strategy(stage_callback)
    )
    save_strategy = FullyParallelSaveStrategyWrapper(
        save_strategy, mpu.get_data_parallel_group(with_context_parallel=True)
    )

    # https://github.com/NVIDIA/Megatron-LM/blob/core_v0.14.0/megatron/core/optimizer/distrib_optimizer.py#L1109-L1123
    mcore_ge_014 = version.parse(megatron.core.__version__) >= version.parse("0.14.0")
    # Save model sharded state dicts
    save_kwargs = dict(
        sharded_strategy=save_strategy,
        async_sharded_save=async_save,
        validate_access_integrity=validate_sharding_integrity,
    )
    if content_metadata is not None:
        if mcore_ge_014:
            save_kwargs["content_metadata"] = content_metadata
    if stage_callback is not None and not async_save:
        stage_callback(
            "checkpoint_dcp_before_strategy_save",
            {
                "checkpoint_copy_ahead_bytes": _SYNC_DCP_COPY_AHEAD_BYTES,
                "checkpoint_writer_thread_count": 1,
            },
        )
    result = dist_checkpointing.save(sharded_state_dict, ckpt_path, **save_kwargs)
    if stage_callback is not None and not async_save:
        stage_callback("checkpoint_dcp_after_strategy_save", None)
    return result


def load_dist_checkpointing(sharded_state_dict, ckpt_dir):
    # Get checkpointing strategies
    load_strategy = get_default_load_sharded_strategy(ckpt_dir)
    load_strategy = FullyParallelLoadStrategyWrapper(
        load_strategy, mpu.get_data_parallel_group(with_context_parallel=True)
    )

    # Fix torch.load weights only error
    try:
        import transformer_engine as te

        torch.serialization.add_safe_globals([torch.optim.AdamW])
        torch.serialization.add_safe_globals([te.pytorch.optimizers.fused_adam.FusedAdam])
    except Exception:
        pass

    # Load model sharded state dicts
    state_dict = dist_checkpointing.load(sharded_state_dict, ckpt_dir, sharded_strategy=load_strategy)

    return state_dict
