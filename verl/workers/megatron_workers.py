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
"""
The main entry point to run the PPO algorithm
"""

import ctypes
import datetime
import gc
import logging
import os
import sys
import time

import psutil
import torch
import torch.distributed
from codetiming import Timer
from omegaconf import DictConfig, OmegaConf

try:
    from verl.workers.engine.mindspeed.transformer_impl import repatch
except ImportError:
    repatch = None

from contextlib import nullcontext

from megatron.core import parallel_state as mpu

from verl import DataProto
from verl.models.mcore import get_mcore_weight_converter
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import Dispatch, make_nd_compute_dataproto_dispatch_fn, register
from verl.trainer.ppo.eu_derpo import policy_prepass_tensors
from verl.utils import hf_tokenizer
from verl.utils.checkpoint.megatron_checkpoint_manager import MegatronCheckpointManager
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug.eu_derpo import HOST_RAM_SAFETY_MARGIN_BYTES
from verl.utils.device import (
    get_device_id,
    get_device_name,
    get_nccl_backend,
    get_torch_device,
    set_expandable_segments,
)
from verl.utils.distributed import set_numa_affinity
from verl.utils.flops_counter import FlopsCounter
from verl.utils.fs import copy_to_local
from verl.utils.megatron.router_replay_patch import RouterReplay, RouterReplayAction, apply_router_replay_patch
from verl.utils.megatron_peft_utils import add_base_layer_suffix, build_peft_config_for_vllm
from verl.utils.megatron_utils import (
    is_megatron_model_offloaded,
    is_megatron_optimizer_offloaded,
    load_megatron_model_to_gpu,
    load_megatron_optimizer,
    load_megatron_optimizer_copy_params_to_gpu,
    megatron_model_cpu_data_bytes,
    offload_megatron_model_to_cpu,
    offload_megatron_optimizer,
    offload_megatron_optimizer_copy_params_to_cpu,
    per_tensor_generator,
    register_megatron_training_hooks,
)
from verl.utils.memory_utils import (
    aggressive_empty_cache,
    estimate_hdo_memory,
    estimate_optimizer_phase_offload_memory,
    log_eu_derpo_memory,
)
from verl.utils.model import get_hf_model_path, load_mcore_dist_weights, load_megatron_gptmodel_weights
from verl.utils.profiler import (
    DistProfiler,
    DistProfilerExtension,
    GPUMemoryLogger,
    ProfilerConfig,
    log_gpu_memory_usage,
    simple_timer,
)
from verl.utils.profiler.performance import reduce_timing, topk_reduce_ratio_min_max
from verl.utils.ray_utils import get_event_loop
from verl.utils.torch_functional import use_original_torch_compile
from verl.workers.actor.megatron_actor import MegatronPPOActor
from verl.workers.config import HFModelConfig, McoreCriticConfig, RolloutConfig
from verl.workers.critic.megatron_critic import MegatronPPOCritic
from verl.workers.rollout import get_rollout_class

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

_GIB = 1024**3
_RESUME_MAX_HOST_MEMORY_PERCENT = 80.0
_RESUME_MIN_HOST_AVAILABLE_BYTES = 200 * _GIB


def _best_effort_release_host_allocator_after_copy_param_restore():
    collected = gc.collect()
    attempted = sys.platform.startswith("linux")
    rc = None
    if attempted:
        try:
            malloc_trim = ctypes.CDLL("libc.so.6").malloc_trim
            malloc_trim.argtypes = [ctypes.c_size_t]
            malloc_trim.restype = ctypes.c_int
            rc = int(malloc_trim(0))
        except (AttributeError, OSError) as error:
            logger.warning("EU-DERPO malloc_trim unavailable after copy-param restore: %r", error)
    return {"gc_collected": collected, "malloc_trim_attempted": int(attempted), "malloc_trim_rc": rc}


_CHECKPOINT_MIN_CUDA_FREE_BYTES = 16 * _GIB
_CHECKPOINT_MAX_CUDA_RESERVED_BYTES = 64 * _GIB


def _validate_actor_metrics_schema(metrics):
    for key, value in metrics.items():
        if isinstance(value, dict):
            raise TypeError(f"actor metric {key!r} has non-reducible value_type=dict")
        if isinstance(value, list):
            for index, item in enumerate(value):
                if isinstance(item, dict):
                    raise TypeError(
                        f"actor metric {key!r} has non-reducible list element "
                        f"index={index} element_type=dict"
                    )


def _optimizer_copy_param_host_decision(required_by_rank, mem_available):
    node_required = sum(int(value) for value in required_by_rank)
    mem_available = int(mem_available)
    return {
        "required_by_rank": [int(value) for value in required_by_rank],
        "node_required": node_required,
        "mem_available": mem_available,
        "safety_margin": HOST_RAM_SAFETY_MARGIN_BYTES,
        "passed": mem_available >= node_required + HOST_RAM_SAFETY_MARGIN_BYTES,
    }


def _is_gpu_adam_distributed_optimizer(optimizer):
    from megatron.core.optimizer import ChainedOptimizer
    from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer
    from transformer_engine.pytorch.optimizers import FusedAdam

    optimizers = optimizer.chained_optimizers if isinstance(optimizer, ChainedOptimizer) else (optimizer,)
    return bool(optimizers) and all(
        isinstance(distributed_optimizer, DistributedOptimizer)
        and isinstance(distributed_optimizer.optimizer, FusedAdam)
        for distributed_optimizer in optimizers
    )


def _log_resume_memory(stage):
    host = psutil.virtual_memory()
    device = get_torch_device()
    gpu_available = device.is_available()
    gpu_allocated = device.memory_allocated() if gpu_available else 0
    gpu_reserved = device.memory_reserved() if gpu_available else 0
    gpu_peak = device.max_memory_allocated() if gpu_available else 0
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    should_log = rank == 0 or os.getenv("VERL_RESUME_MEM_DEBUG_ALL_RANKS") == "1"
    if should_log:
        logger.warning(
            "RESUME_MEM %s host_used_gib=%.2f host_available_gib=%.2f host_percent=%.1f "
            "gpu_allocated_gib=%.2f gpu_reserved_gib=%.2f gpu_max_allocated_gib=%.2f",
            stage,
            host.used / _GIB,
            host.available / _GIB,
            host.percent,
            gpu_allocated / _GIB,
            gpu_reserved / _GIB,
            gpu_peak / _GIB,
        )
    if should_log and (
        host.percent >= _RESUME_MAX_HOST_MEMORY_PERCENT or host.available < _RESUME_MIN_HOST_AVAILABLE_BYTES
    ):
        logger.warning(
            "RESUME_MEM_WARNING host budget observation threshold crossed at %s: "
            "host_used_gib=%.2f host_available_gib=%.2f host_percent=%.1f",
            stage,
            host.used / _GIB,
            host.available / _GIB,
            host.percent,
        )
    return {
        "host_used": host.used,
        "host_available": host.available,
        "host_percent": host.percent,
        "gpu_available": gpu_available,
        "gpu_allocated": gpu_allocated,
        "gpu_reserved": gpu_reserved,
        "gpu_peak_allocated": gpu_peak,
    }


def _reset_resume_peak_memory():
    device = get_torch_device()
    if device.is_available():
        device.reset_peak_memory_stats()


def _normalize_mbridge_qwen2moe_config(hf_config, tf_config):
    architectures = getattr(hf_config, "architectures", None) or ()
    if "Qwen2MoeForCausalLM" not in architectures:
        return
    if not hasattr(tf_config, "moe_shared_expert_gate"):
        raise RuntimeError("Qwen2MoE requires moe_shared_expert_gate support")
    tf_config.moe_shared_expert_gate = True


def set_random_seed(seed, only_rollout=False):
    import random

    import numpy as np
    import torch

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if not only_rollout and get_torch_device().device_count() > 0:
        from megatron.core import tensor_parallel

        tensor_parallel.model_parallel_cuda_manual_seed(seed)
    # FIXME: torch cumsum not support deterministic (used in vllm sampler),
    # https://github.com/pytorch/pytorch/issues/89492
    # torch.use_deterministic_algorithms(True, warn_only=True)
    # os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


class MegatronWorker(Worker):
    def _init_hf_config_and_tf_config(
        self,
        model_path,
        tokenizer_or_path,
        dtype,
        override_model_config,
        override_transformer_config,
        trust_remote_code=False,
        megatron_config=None,
        enable_mtp=False,
    ):
        from transformers import AutoConfig

        from verl.models.mcore import hf_to_mcore_config
        from verl.utils import hf_processor
        from verl.utils.model import update_model_config

        # Step 1: initialize the tokenizer
        self.local_path = copy_to_local(model_path)
        if tokenizer_or_path is None:
            self.tokenizer = hf_tokenizer(self.local_path, trust_remote_code=trust_remote_code)
            self.processor = hf_processor(self.local_path, trust_remote_code=trust_remote_code)
        elif isinstance(tokenizer_or_path, str):
            self.tokenizer = hf_tokenizer(copy_to_local(tokenizer_or_path), trust_remote_code=trust_remote_code)
            self.processor = hf_processor(copy_to_local(tokenizer_or_path), trust_remote_code=trust_remote_code)
        else:
            self.tokenizer = tokenizer_or_path
            self.processor = tokenizer_or_path

        if self.config.model.get("custom_chat_template", None) is not None:
            if self.processor is not None:
                self.processor.chat_template = self.config.model.custom_chat_template
            else:
                self.tokenizer.chat_template = self.config.model.custom_chat_template

        # Step 2: get the hf
        hf_config = AutoConfig.from_pretrained(self.local_path, trust_remote_code=trust_remote_code)

        # Step 3: override the hf config
        override_config_kwargs = {
            "bos_token_id": self.tokenizer.bos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        override_config_kwargs.update(override_model_config.get("model_config", {}))
        self.share_embeddings_and_output_weights = getattr(hf_config, "tie_word_embeddings", False)

        # only actor need enable mtp
        if enable_mtp:
            assert hf_config.num_nextn_predict_layers > 0, "MTP requires at least one nextn_predict_layer"
            assert megatron_config.use_mbridge, "MTP requires use_mbridge to be True"
            override_transformer_config["mtp_loss_scaling_factor"] = self.config.model.mtp.mtp_loss_scaling_factor
        else:
            if hasattr(hf_config, "num_nextn_predict_layers"):
                hf_config.num_nextn_predict_layers = 0

        self.enable_mtp = enable_mtp

        update_model_config(hf_config, override_config_kwargs=override_config_kwargs)
        self.architectures = getattr(hf_config, "architectures", None)
        if self.rank == 0:
            print(f"Model config after override: {hf_config}")

        from verl.models.mcore.config_converter import mapping_string_to_attn_backend

        # todo: remove this line after mcore adopt mbridge 0.15, now for compatibility
        override_transformer_config = mapping_string_to_attn_backend(override_transformer_config)
        fp16 = dtype == torch.float16
        bf16 = dtype == torch.bfloat16
        if fp16:
            assert megatron_config.use_mbridge, "fp16 mode requires use_mbridge to be True"

        self.provider = None
        self.vanilla_bridge = megatron_config.get("vanilla_mbridge", True)
        if megatron_config.use_mbridge:
            if self.vanilla_bridge:
                from verl.models.mcore.mbridge import AutoBridge

                bridge = AutoBridge.from_config(hf_config, dtype=dtype)
                bridge.set_extra_args(**override_transformer_config)
                tf_config = bridge.config
                _normalize_mbridge_qwen2moe_config(hf_config, tf_config)
                tf_config.fp16 = fp16
                tf_config.bf16 = bf16
            else:
                from verl.models.mcore.bridge import AutoBridge

                # Use Megatron-Bridge to convert HF config to Megatron config
                bridge = AutoBridge.from_hf_pretrained(self.local_path, trust_remote_code=trust_remote_code)
                # Get Megatron provider and configure it
                provider = bridge.to_megatron_provider(load_weights=False)

                # In case of invalid overrides, we need to make sure some critical params are set correctly
                provider.params_dtype = dtype

                # Ensure dtype settings propagate to Megatron-Bridge/TE
                provider.fp16 = fp16
                provider.bf16 = bf16

                # Pass distributed info
                provider.tensor_model_parallel_size = megatron_config.tensor_model_parallel_size
                provider.pipeline_model_parallel_size = megatron_config.pipeline_model_parallel_size
                provider.expert_model_parallel_size = megatron_config.expert_model_parallel_size
                provider.expert_tensor_parallel_size = megatron_config.expert_tensor_parallel_size
                provider.virtual_pipeline_model_parallel_size = megatron_config.virtual_pipeline_model_parallel_size
                provider.context_parallel_size = megatron_config.context_parallel_size
                provider.sequence_parallel = megatron_config.sequence_parallel

                # Match verl implementation (need variable_seq_lengths)
                from megatron.core.transformer.enums import AttnBackend

                provider.attention_backend = AttnBackend.flash
                provider.variable_seq_lengths = True
                provider.moe_token_dispatcher_type = "alltoall"
                provider.moe_router_load_balancing_type = "none"

                # Apply transformer config overrides
                for key, value in override_transformer_config.items():
                    setattr(provider, key, value)

                provider.finalize()
                self.provider = provider
                tf_config = None  # Will be set after model creation
            self.bridge = bridge
        else:
            tf_config = hf_to_mcore_config(hf_config, dtype, **override_transformer_config)
            self.bridge = None

        if torch.distributed.get_rank() == 0:
            if tf_config is not None:
                print(f"TF config: {tf_config}")
        self.hf_config = hf_config
        self.tf_config = tf_config

        # Get PEFT config from model.lora if specified
        from verl.workers.config.megatron_peft import get_peft_cls

        self.peft_cls = get_peft_cls(
            model_config=self.config.model, bridge=self.bridge, provider=self.provider, dtype=dtype
        )


class ActorRolloutRefWorker(MegatronWorker, DistProfilerExtension):
    """
    This worker can be instantiated as a standalone actor or a standalone rollout or a standalone reference policy
    or a hybrid engine based on the config.rollout
    """

    def __init__(self, config: DictConfig, role: str, **kwargs):
        Worker.__init__(self)
        self.config = config
        if repatch is not None:
            # NPU MindSpeed patch, will be refactored with MindSpeedEngine.
            repatch(self.config.actor.megatron.get("override_transformer_config", {}))

        self.role = role
        assert self.role in ["actor", "rollout", "ref", "actor_rollout", "actor_rollout_ref"]

        self._is_actor = self.role in ["actor", "actor_rollout", "actor_rollout_ref"]
        self._is_rollout = self.role in ["rollout", "actor_rollout", "actor_rollout_ref"]
        self._is_ref = self.role in ["ref", "actor_rollout_ref"]

        # NOTE(sgm): We utilize colocate WorkerGroup by default.
        # As a result, Workers for different model share the same process.
        # Therefore, we only require one distribute initialization.
        # To utilize different parallel strategy in different models:
        # 1, users should disable WorkerDict; 2.assign different ResourcePool to different models,
        # 3. and apply the following patch in ray==2.10, https://github.com/ray-project/ray/pull/44385
        if not torch.distributed.is_initialized():
            set_numa_affinity()
            rank = int(os.environ["LOCAL_RANK"])
            torch.distributed.init_process_group(
                backend=f"cpu:gloo,{get_device_name()}:{get_nccl_backend()}",
                timeout=datetime.timedelta(seconds=self.config.get("nccl_timeout", 600)),
                init_method=os.environ.get("DIST_INIT_METHOD", None),
            )
            get_torch_device().set_device(rank)

            if self._is_actor or self._is_ref:
                mpu.initialize_model_parallel(
                    tensor_model_parallel_size=self.config.actor.megatron.tensor_model_parallel_size,
                    pipeline_model_parallel_size=self.config.actor.megatron.pipeline_model_parallel_size,
                    virtual_pipeline_model_parallel_size=self.config.actor.megatron.virtual_pipeline_model_parallel_size,
                    use_sharp=False,
                    context_parallel_size=self.config.actor.megatron.context_parallel_size,
                    expert_model_parallel_size=self.config.actor.megatron.expert_model_parallel_size,
                    expert_tensor_parallel_size=self.config.actor.megatron.expert_tensor_parallel_size,
                    nccl_communicator_config_path=None,
                )

        if self._is_actor or self._is_ref:
            is_collect = (
                mpu.get_tensor_model_parallel_rank() == 0
                and mpu.get_pipeline_model_parallel_rank() == mpu.get_pipeline_model_parallel_world_size() - 1
                and mpu.get_context_parallel_rank() == 0
            )
            self._register_dispatch_collect_info(
                mesh_name="actor", dp_rank=mpu.get_data_parallel_rank(), is_collect=is_collect
            )
        only_rollout = self._is_rollout and not self._is_actor

        self.enable_routing_replay = False
        if self._is_actor:
            self.router_replay = self.config.actor.router_replay
            self.enable_routing_replay = self.router_replay.mode != "disabled"

        if self.enable_routing_replay:
            apply_router_replay_patch()

        set_random_seed(seed=self.config.actor.megatron.seed, only_rollout=only_rollout)

        if self._is_actor:
            omega_profiler_config = config.actor.get("profiler", {})
        elif self._is_rollout:
            # NOTE: In colocation mode, rollout config may not take effect (follow the actor config)
            # This is for extendability in AsyncRL cases
            omega_profiler_config = config.rollout.get("profiler", {})
        elif self._is_ref:
            omega_profiler_config = config.ref.get("profiler", {})
        else:
            raise ValueError(
                f"Invalid role {self.role}, should be one of "
                "['actor', 'rollout', 'ref', 'actor_rollout', 'actor_rollout_ref']"
            )
        # omega_profiler_config is DictConfig
        # profiler_config is a ProfilerConfig dataclass
        profiler_config = omega_conf_to_dataclass(omega_profiler_config, dataclass_type=ProfilerConfig)
        if omega_profiler_config.get("tool", None) in ["npu", "nsys", "torch", "torch_memory"]:
            tool_config = omega_conf_to_dataclass(
                omega_profiler_config.get("tool_config", {}).get(omega_profiler_config.get("tool"))
            )
        else:
            tool_config = None
        DistProfilerExtension.__init__(
            self, DistProfiler(rank=self.rank, config=profiler_config, tool_config=tool_config)
        )

        # TODO(sgm): Currently, we only support reference model param offload
        # will support other offload later
        self._is_offload_param = False
        self._is_offload_grad = False
        self._is_offload_optimizer = False
        self._checkpoint_training_residency_held = False
        self._hdo_optimizer_residency_preserved = False
        self._optimizer_copy_params_offloaded_for_rollout = False
        self._optimizer_copy_params_cuda_bytes_before_rollout_offload = 0

        # Initialize LoRA-related attributes (will be updated in _build_rollout if needed)
        self.base_sync_done = False
        self.peft_merge = False

        # normalize config
        if self._is_actor:
            self.config.actor.ppo_mini_batch_size *= self.config.rollout.n
            self.config.actor.ppo_mini_batch_size //= mpu.get_data_parallel_world_size()
            if self.config.actor.get("ppo_micro_batch_size", None):
                self.config.actor.ppo_micro_batch_size //= mpu.get_data_parallel_world_size()
                self.config.rollout.log_prob_micro_batch_size //= mpu.get_data_parallel_world_size()
                self.config.actor.ppo_micro_batch_size_per_gpu = self.config.actor.ppo_micro_batch_size
                self.config.rollout.log_prob_micro_batch_size_per_gpu = self.config.rollout.log_prob_micro_batch_size

            self._is_offload_param = self.config.actor.megatron.get("param_offload", False)
            self._is_offload_grad = self.config.actor.megatron.get("grad_offload", False)
            self._is_offload_optimizer = self.config.actor.megatron.get("optimizer_offload", False)
        elif self._is_ref:
            if self.config.ref.get("log_prob_micro_batch_size", None):
                self.config.ref.log_prob_micro_batch_size //= mpu.get_data_parallel_world_size()
                self.config.ref.log_prob_micro_batch_size_per_gpu = self.config.ref.log_prob_micro_batch_size
            else:
                assert self.config.ref.get("log_prob_micro_batch_size_per_gpu", None) is not None, (
                    "Please note that in the ref policy configuration, `log_prob_micro_batch_size_per_gpu` and "
                    "`log_prob_micro_batch_size` should not be None at the same time."
                )
            self._ref_is_offload_param = self.config.ref.megatron.get("param_offload", False)

    def _build_model_optimizer(
        self, model_path, optim_config, override_model_config, override_transformer_config, override_ddp_config=None
    ):
        from verl.utils.megatron.optimizer import (
            get_megatron_optimizer,
            get_megatron_optimizer_param_scheduler,
            init_megatron_optim_config,
        )
        from verl.utils.megatron_utils import McoreModuleWrapperConfig, make_megatron_module
        from verl.utils.model import get_generation_config, print_model_size

        self._init_hf_config_and_tf_config(
            model_path,
            self.config.model.get("tokenizer_path") or model_path,
            self.dtype,
            override_model_config,
            override_transformer_config,
            self.config.model.get("trust_remote_code", False),
            self.config.actor.megatron if not self._is_ref else self.config.ref.megatron,
            self.config.model.get("mtp", {}).get("enable", False),
        )
        self.generation_config = get_generation_config(
            self.local_path,
            self.config.model.get("trust_remote_code", False),
        )

        if self._is_actor or self._is_rollout:
            wrap_config = McoreModuleWrapperConfig(
                is_value_model=False,  # actor is not value model
                share_embeddings_and_output_weights=self.share_embeddings_and_output_weights,
                wrap_with_ddp=True,
                use_distributed_optimizer=self.config.actor.megatron.use_distributed_optimizer,
            )
            actor_module, updated_tf_config = make_megatron_module(
                wrap_config=wrap_config,
                tf_config=self.tf_config,
                hf_config=self.hf_config,
                bridge=self.bridge,
                provider=self.provider,
                override_model_config=override_model_config,
                override_ddp_config=override_ddp_config,
                peft_cls=self.peft_cls,
                peft_config=self.config.model.get("lora", None),
            )
            self.tf_config = updated_tf_config
            print(f"actor_module: {len(actor_module)}")
            if self.config.actor.load_weight:
                if self.config.actor.megatron.use_dist_checkpointing:
                    load_mcore_dist_weights(
                        actor_module,
                        self.config.actor.megatron.dist_checkpointing_path,
                        is_value_model=False,
                        prefix=self.config.actor.megatron.dist_checkpointing_prefix,
                    )
                else:
                    if self.bridge is not None:
                        local_model_path = get_hf_model_path(self.config)
                        if self.vanilla_bridge:
                            self.bridge.load_weights(actor_module, local_model_path)
                        else:
                            self.bridge.load_hf_weights(actor_module, local_model_path)
                    else:
                        load_megatron_gptmodel_weights(
                            self.config, self.hf_config, actor_module, params_dtype=self.dtype, is_value_model=False
                        )

            if self.rank == 0:
                print_model_size(actor_module[0])
            log_gpu_memory_usage("After MegatronPPOActor init", logger=logger)
        elif self._is_ref:
            wrap_config = McoreModuleWrapperConfig(
                is_value_model=False,  # ref is not value model
                share_embeddings_and_output_weights=self.share_embeddings_and_output_weights,
                wrap_with_ddp=False,
                use_distributed_optimizer=self.config.ref.megatron.use_distributed_optimizer,
            )
            ref_module, updated_tf_config = make_megatron_module(
                wrap_config=wrap_config,
                tf_config=self.tf_config,
                hf_config=self.hf_config,
                bridge=self.bridge,
                provider=self.provider,
                override_model_config=override_model_config,
            )
            self.tf_config = updated_tf_config
            if self.config.ref.load_weight:  # should align with the actor:
                assert self.config.actor.load_weight == self.config.ref.load_weight
                print("load ref weight start")
                if self.config.ref.megatron.use_dist_checkpointing:
                    load_mcore_dist_weights(
                        ref_module,
                        self.config.ref.megatron.dist_checkpointing_path,
                        is_value_model=False,
                        prefix=self.config.ref.megatron.dist_checkpointing_prefix,
                    )
                else:
                    if self.bridge is not None:
                        local_model_path = get_hf_model_path(self.config)
                        if self.vanilla_bridge:
                            self.bridge.load_weights(ref_module, local_model_path)
                        else:
                            self.bridge.load_hf_weights(ref_module, local_model_path)
                    else:
                        load_megatron_gptmodel_weights(
                            self.config, self.hf_config, ref_module, params_dtype=self.dtype, is_value_model=False
                        )
            log_gpu_memory_usage("After ref module init", logger=logger)
            return ref_module, self.hf_config

        # TODO: add more optimizer args into config
        if self._is_actor:
            optim_config_megatron = init_megatron_optim_config(
                optim_config,
                use_distributed_optimizer=wrap_config.use_distributed_optimizer,
                fp16=self.dtype == torch.float16,
            )
            actor_optimizer = get_megatron_optimizer(model=actor_module, config=optim_config_megatron)
            actor_optimizer_scheduler = get_megatron_optimizer_param_scheduler(
                optimizer=actor_optimizer, config=optim_config
            )
        else:
            optim_config = None
            actor_optimizer = None
            actor_optimizer_scheduler = None

        log_gpu_memory_usage("After actor optimizer init", logger=logger)

        register_megatron_training_hooks(actor_module, actor_optimizer)

        return actor_module, actor_optimizer, actor_optimizer_scheduler, self.hf_config, optim_config

    def _build_rollout(self, trust_remote_code=False):
        from torch.distributed.device_mesh import init_device_mesh

        # 1. parse rollout and huggingface model config
        rollout_config: RolloutConfig = omega_conf_to_dataclass(self.config.rollout)
        model_config: HFModelConfig = omega_conf_to_dataclass(self.config.model)

        # 2. build rollout device mesh
        infer_tp = self.config.rollout.tensor_model_parallel_size * self.config.rollout.data_parallel_size
        infer_pp = self.config.rollout.pipeline_model_parallel_size
        infer_world_size = infer_tp * infer_pp
        dp = self.world_size // infer_world_size
        assert self.world_size % infer_world_size == 0, (
            f"rollout world_size: {self.world_size} is not divisible by infer_world_size: {infer_world_size}"
        )
        rollout_device_mesh = init_device_mesh(
            get_device_name(), mesh_shape=(dp, infer_tp, infer_pp), mesh_dim_names=["dp", "infer_tp", "infer_pp"]
        )

        self.rollout_device_mesh = rollout_device_mesh

        is_collect = (
            rollout_device_mesh["infer_tp"].get_local_rank() == 0
            and rollout_device_mesh["infer_pp"].get_local_rank() == 0
        )
        self._register_dispatch_collect_info(
            "rollout", dp_rank=rollout_device_mesh["dp"].get_local_rank(), is_collect=is_collect
        )

        # 4. build rollout model
        log_gpu_memory_usage(f"Before building {self.config.rollout.name} rollout", logger=logger)
        self.rollout = get_rollout_class(rollout_config.name, rollout_config.mode)(
            config=rollout_config, model_config=model_config, device_mesh=rollout_device_mesh
        )
        log_gpu_memory_usage(f"After building {self.config.rollout.name} rollout", logger=logger)

        # Initialize base_sync_done for LoRA
        self.base_sync_done: bool = "dummy" not in self.config.rollout.load_format
        self.peft_merge: bool = model_config.lora.get("merge", False)

        # 5. switch to trainer mode
        # NOTE: It's critical that hybrid engine in trainer mode initially to load checkpoint.
        # For async mode, we can't call run_until_complete here, so we will switch to trainer mode in AgentLoopManager.
        # Note: sync mode is deprecated and rejected in RolloutConfig.__post_init__

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        if self.config.model.get("external_lib", None) is not None:
            # This is used to import external_lib into the huggingface systems
            import importlib

            importlib.import_module(self.config.model.external_lib)

        from verl.utils.torch_dtypes import PrecisionType

        override_model_config = OmegaConf.to_container(OmegaConf.create(self.config.model.get("override_config", {})))
        if self._is_actor:
            override_transformer_config = OmegaConf.to_container(
                OmegaConf.create(self.config.actor.megatron.get("override_transformer_config", {}))
            )
            if self.enable_routing_replay:
                override_transformer_config["enable_routing_replay"] = True
            override_ddp_config = OmegaConf.to_container(
                OmegaConf.create(self.config.actor.megatron.get("override_ddp_config", {}))
            )
        elif self._is_ref:
            override_transformer_config = OmegaConf.to_container(
                OmegaConf.create(self.config.ref.megatron.get("override_transformer_config", {}))
            )
        else:
            override_transformer_config = {}
        self.param_dtype = PrecisionType.to_dtype(self.config.actor.megatron.dtype)
        log_gpu_memory_usage("Before init actor model and optimizer", logger=logger)
        self.dtype = PrecisionType.to_dtype(self.param_dtype)
        if self._is_actor:
            # we need the model for actor and rollout
            optim_config = self.config.actor.optim if self._is_actor else None
            (
                self.actor_module,
                self.actor_optimizer,
                self.actor_optimizer_scheduler,
                self.actor_model_config,
                self.actor_optim_config,
            ) = self._build_model_optimizer(
                model_path=self.config.model.path,
                optim_config=optim_config,
                override_model_config=override_model_config,
                override_transformer_config=override_transformer_config,
                override_ddp_config=override_ddp_config,
            )
            self._validate_hdo_optimizer_residency_config()
            if self.config.actor.eu_derpo.enabled:
                log_eu_derpo_memory("after_optimizer_construction", self.actor_optimizer)
            if self._is_offload_param:
                offload_megatron_model_to_cpu(self.actor_module)
                log_gpu_memory_usage("After offload actor params and grad during init", logger=logger)
            if self._is_offload_optimizer:
                self._offload_actor_optimizer()
                log_gpu_memory_usage("After offload actor optimizer during init", logger=logger)
            if self.config.actor.eu_derpo.enabled:
                log_eu_derpo_memory(
                    "after_phase_offload_init",
                    self.actor_optimizer,
                    megatron_model_cpu_data_bytes(self.actor_module),
                )

        if self._is_actor:
            actor_cfg = omega_conf_to_dataclass(self.config.actor)
            self.actor = MegatronPPOActor(
                config=actor_cfg,
                model_config=self.actor_model_config,
                hf_config=self.hf_config,
                tf_config=self.tf_config,
                actor_module=self.actor_module,
                actor_optimizer=self.actor_optimizer,
                mtp_config=self.config.model.mtp if self.config.model.mtp.enable else None,
            )
            print(f"routing replay layers: {len(RouterReplay.router_instances)}")
            log_gpu_memory_usage("After MegatronPPOActor init", logger=logger)

        if self._is_rollout:
            with use_original_torch_compile():
                self._build_rollout(trust_remote_code=self.config.model.get("trust_remote_code", False))
            log_gpu_memory_usage("After rollout init", logger=logger)

        if self._is_ref:
            self.ref_module, self.ref_model_config = self._build_model_optimizer(
                model_path=self.config.model.path,
                optim_config=None,
                override_model_config=override_model_config,
                override_transformer_config=override_transformer_config,
            )
            log_gpu_memory_usage("After ref model init", logger=logger)
            self.ref_policy = MegatronPPOActor(
                config=self.config.ref,
                model_config=self.ref_model_config,
                hf_config=self.hf_config,
                tf_config=self.tf_config,
                actor_module=self.ref_module,
                actor_optimizer=None,
            )
            if self._ref_is_offload_param:
                offload_megatron_model_to_cpu(self.ref_module)
                log_gpu_memory_usage("After offload ref params during init", logger=logger)

        if self._is_actor:
            self.flops_counter = FlopsCounter(self.actor_model_config)
            self.checkpoint_mananager = MegatronCheckpointManager(
                config=self.config,
                checkpoint_config=self.config.actor.checkpoint,
                model_config=self.actor_model_config,
                transformer_config=self.tf_config,
                role="actor",
                model=self.actor_module,
                arch=self.architectures[0],
                hf_config=self.hf_config,
                param_dtype=self.param_dtype,
                share_embeddings_and_output_weights=self.share_embeddings_and_output_weights,
                processing_class=self.processor if self.processor is not None else self.tokenizer,
                optimizer=self.actor_optimizer,
                optimizer_scheduler=self.actor_optimizer_scheduler,
                use_distributed_optimizer=self.config.actor.megatron.use_distributed_optimizer,
                use_checkpoint_opt_param_scheduler=self.config.actor.optim.use_checkpoint_opt_param_scheduler,
                bridge=self.bridge,
                provider=self.provider,
                use_dist_checkpointing=self.config.actor.megatron.use_dist_checkpointing,
                peft_cls=self.peft_cls,
            )

            self.layer_name_mapping = {
                "qkv_layer_name": "self_attention.linear_qkv.",
                "gate_proj_layer_name": "linear_fc1.",
            }
            self.weight_converter = None
            if not self.config.actor.megatron.use_mbridge:
                self.weight_converter = get_mcore_weight_converter(self.actor_model_config, self.dtype)

        # Free cached GPU memory so colocated vLLM processes can see it via cudaMemGetInfo
        aggressive_empty_cache(force_sync=True)
        log_gpu_memory_usage("After init_model finish", logger=logger)

    async def rollout_mode(self):
        """Context switch hybridengine to rollout mode."""
        aggressive_empty_cache(force_sync=True)
        set_expandable_segments(False)

        if self._is_offload_param:
            load_megatron_model_to_gpu(self.actor.actor_module, load_grad=False)
            log_gpu_memory_usage("After load actor params during rollout_mode", logger=logger)

        # Build peft_config for vLLM LoRA support
        peft_config = None
        do_lora_base_sync = False
        if not self.peft_merge and self.peft_cls is not None:
            peft_config = build_peft_config_for_vllm(self.config.model.get("lora", {}))
            # set sleep level for LoRA adapter weights only sync
            # TODO: make this configurable so that users with small
            # main memory can trade sync time to avoid OOM
            self.rollout.sleep_level = 1

            do_lora_base_sync = (not self.base_sync_done) or (
                self.rollout.sleep_level != 1 and self.config.rollout.free_cache_engine
            )

        if self.bridge is not None:
            if self.vanilla_bridge:
                per_tensor_param = self.bridge.export_weights(self.actor.actor_module)
            elif not self.peft_merge and self.peft_cls is not None:
                # Only export adapter weights
                per_tensor_param = self.bridge.export_adapter_weights(self.actor.actor_module)
            else:
                per_tensor_param = self.bridge.export_hf_weights(self.actor.actor_module)
        else:
            per_tensor_param = per_tensor_generator(
                self.actor.actor_module,
                self.actor_model_config,
                self.weight_converter,
                self.tf_config,
                self.layer_name_mapping,
            )

        update_weights_started = time.perf_counter()
        if self.config.rollout.free_cache_engine:
            await self.rollout.resume(tags=["weights"])
        if self.config.actor.eu_derpo.enabled:
            log_eu_derpo_memory(
                "before_rollout_update_weights",
                self.actor_optimizer,
                megatron_model_cpu_data_bytes(self.actor_module),
                extra={
                    "checkpoint_hold_active": int(self._checkpoint_training_residency_held),
                    "defer_phase_offload_for_checkpoint": 0,
                },
            )
            self._warn_eu_derpo_hdo_gpu_headroom("before_rollout_update_weights")
        if do_lora_base_sync:
            # Base layer sync
            per_tensor_param_lora_base = self.bridge.export_hf_weights(
                self.actor.actor_module, merge_adapter_weights=False
            )
            await self.rollout.update_weights(
                add_base_layer_suffix(per_tensor_param_lora_base, model_type=self.hf_config.model_type),
                peft_config=peft_config,
                base_sync_done=False,
            )

            # Mark base sync as done after first successful sync
            self.base_sync_done = True

        await self.rollout.update_weights(per_tensor_param, peft_config=peft_config, base_sync_done=True)
        if self.config.actor.eu_derpo.enabled:
            log_eu_derpo_memory(
                "after_rollout_update_weights",
                self.actor_optimizer,
                megatron_model_cpu_data_bytes(self.actor_module),
                extra={"update_weights_elapsed_s": time.perf_counter() - update_weights_started},
            )
        if self._is_offload_param:
            offload_megatron_model_to_cpu(self.actor.actor_module)
        cache_released = self._release_actor_cuda_cache_before_rollout_wakeup()
        if not cache_released:
            aggressive_empty_cache(force_sync=True)
        if self.config.rollout.free_cache_engine:
            self._warn_eu_derpo_hdo_gpu_headroom("before_rollout_wakeup")
            wakeup_started = time.perf_counter()
            await self.rollout.resume(tags=["kv_cache"])
            if self.config.actor.eu_derpo.enabled:
                log_eu_derpo_memory(
                    "after_rollout_wakeup",
                    self.actor_optimizer,
                    megatron_model_cpu_data_bytes(self.actor_module),
                    extra={"rollout_wakeup_elapsed_s": time.perf_counter() - wakeup_started},
                )

        set_expandable_segments(True)

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    @GPUMemoryLogger(role="update_actor", logger=logger)
    @DistProfiler.annotate(color="red", role="actor_update")
    def update_actor(self, data: DataProto):
        assert self._is_actor
        self._restore_actor_optimizer_copy_params_for_update()
        eu_enabled = self.config.actor.eu_derpo.enabled
        defer_phase_offload = bool(data.meta_info.get("defer_phase_offload_for_checkpoint", False))
        if eu_enabled:
            log_eu_derpo_memory(
                "before_actor_update",
                self.actor_optimizer,
                megatron_model_cpu_data_bytes(self.actor_module),
            )
        if self._is_offload_param:
            load_megatron_model_to_gpu(self.actor_module)
            log_gpu_memory_usage("After load actor params and grad during update_actor", logger=logger)
        preserve_hdo = self._preserve_hdo_optimizer_residency()
        if self._load_actor_optimizer_for_update():
            log_gpu_memory_usage("After load actor optimizer during update_actor", logger=logger)
        if eu_enabled:
            log_eu_derpo_memory(
                "after_actor_load",
                self.actor_optimizer,
                megatron_model_cpu_data_bytes(self.actor_module),
            )

        micro_batch_size = self.config.actor.ppo_micro_batch_size_per_gpu
        data.meta_info["micro_batch_size"] = micro_batch_size
        dataloader = self.actor.make_minibatch_iterator(data=data)
        with Timer(name="update_policy", logger=None) as timer:
            metrics = self.actor.update_policy(dataloader=dataloader)
        delta_time = timer.last
        global_num_tokens = data.meta_info["global_token_num"]
        images_seqlens = data.meta_info.get("images_seqlens", None)
        estimated_flops, promised_flops = self.flops_counter.estimate_flops(
            global_num_tokens, delta_time, images_seqlens=images_seqlens
        )
        metrics["perf/mfu/actor"] = estimated_flops * self.config.actor.ppo_epochs / promised_flops / self.world_size
        metrics["perf/max_memory_allocated_gb"] = get_torch_device().max_memory_allocated() / (1024**3)
        metrics["perf/max_memory_reserved_gb"] = get_torch_device().max_memory_reserved() / (1024**3)
        metrics["perf/cpu_memory_used_gb"] = psutil.virtual_memory().used / (1024**3)
        from verl.utils.megatron.optimizer import get_megatron_last_lr

        metrics["actor/lr"] = get_megatron_last_lr(self.actor_optimizer)
        self.actor_optimizer_scheduler.step(1)
        if eu_enabled:
            _validate_actor_metrics_schema(metrics)

        # TODO: here, we should return all metrics
        output = DataProto(meta_info={"metrics": metrics})
        output = output.to("cpu")

        self._finish_actor_update_residency(defer_phase_offload, eu_enabled, preserve_hdo)

        aggressive_empty_cache(force_sync=True)
        return output

    def _preserve_hdo_optimizer_residency(self):
        return bool(
            self.config.actor.eu_derpo.enabled
            and self.config.actor.eu_derpo.preserve_hdo_optimizer_residency_between_steps
        )

    def _selective_optimizer_copy_param_offload_enabled(self):
        return bool(
            self.config.actor.eu_derpo.enabled
            and self.config.actor.eu_derpo.offload_optimizer_copy_params_for_rollout
        )

    def _actor_optimizer_residency(self):
        if self._optimizer_copy_params_offloaded_for_rollout:
            return "ROLLOUT_PARTIAL"
        if self._is_offload_optimizer and is_megatron_optimizer_offloaded(self.actor_optimizer):
            return "FULL_OFFLOADED"
        return "TRAINING_RESIDENT"

    def _optimizer_copy_param_host_preflight(self, local_required):
        if torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
            if world_size != 8:
                raise RuntimeError("EU-DERPO optimizer copy-param host preflight requires one world8 node")
            local = torch.tensor([int(local_required)], dtype=torch.int64, device=get_device_name())
            gathered = [torch.zeros_like(local) for _ in range(world_size)]
            torch.distributed.all_gather(gathered, local)
            required_by_rank = [int(item.item()) for item in gathered]
            payload = torch.zeros(2, dtype=torch.int64, device=get_device_name())
            if torch.distributed.get_rank() == 0:
                decision = _optimizer_copy_param_host_decision(
                    required_by_rank, psutil.virtual_memory().available
                )
                payload[0] = decision["mem_available"]
                payload[1] = int(decision["passed"])
            torch.distributed.broadcast(payload, src=0)
            decision = _optimizer_copy_param_host_decision(required_by_rank, int(payload[0].item()))
            if bool(payload[1].item()) != decision["passed"]:
                raise RuntimeError("EU-DERPO optimizer copy-param host preflight broadcast was inconsistent")
        else:
            decision = _optimizer_copy_param_host_decision(
                [local_required], psutil.virtual_memory().available
            )
        if not decision["passed"]:
            raise MemoryError(
                "EU_DERPO_COPY_PARAM_HOST_HEADROOM_INSUFFICIENT "
                f"mem_available_bytes={decision['mem_available']} "
                f"node_copy_param_bytes={decision['node_required']} "
                f"host_safety_margin_bytes={decision['safety_margin']}"
            )
        return decision

    def _complete_optimizer_copy_param_transition(self, stage, failure, error, moved):
        moved_tensors = moved.get("moved_tensors", getattr(error, "moved_tensors", 0))
        moved_bytes = moved.get("moved_bytes", getattr(error, "moved_bytes", 0))
        if failure is not None:
            failure.fill_(int(error is not None))
            torch.distributed.all_reduce(failure, op=torch.distributed.ReduceOp.MAX)
            failed = bool(failure.item())
        else:
            failed = error is not None
        if failed:
            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
            raise RuntimeError(
                "EU_DERPO_COPY_PARAM_RESIDENCY_TRANSITION_FAILED "
                f"stage={stage} rank={rank} moved_tensors={moved_tensors} "
                f"moved_bytes={moved_bytes} local_error={error!r}"
            ) from error

    def _offload_actor_optimizer_copy_params_for_rollout(self):
        if self._optimizer_copy_params_offloaded_for_rollout:
            return False
        before_estimate = estimate_optimizer_phase_offload_memory(self.actor_optimizer)
        local_bytes = before_estimate["optimizer_phase_cuda_copy_param_bytes"]
        if local_bytes <= 0:
            raise RuntimeError("EU-DERPO selective rollout offload found no CUDA optimizer copy params")
        host = self._optimizer_copy_param_host_preflight(local_bytes)
        log_eu_derpo_memory(
            "before_optimizer_copy_param_offload",
            self.actor_optimizer,
            megatron_model_cpu_data_bytes(self.actor_module),
            extra={
                "copy_param_offload_active": 0,
                "optimizer_residency": self._actor_optimizer_residency(),
                "node_copy_param_bytes_to_offload_gib": host["node_required"] / _GIB,
                "host_safety_margin_gib": host["safety_margin"] / _GIB,
                "copy_param_transition_elapsed_s": 0.0,
            },
        )
        started = time.perf_counter()
        failure = (
            torch.zeros(1, dtype=torch.int32, device=get_device_name())
            if torch.distributed.is_initialized()
            else None
        )
        moved = {}
        error = None
        try:
            moved = offload_megatron_optimizer_copy_params_to_cpu(self.actor_optimizer)
        except Exception as caught:
            error = caught
        self._complete_optimizer_copy_param_transition("offload", failure, error, moved)
        device = get_torch_device()
        device.synchronize()
        device.empty_cache()
        self._optimizer_copy_params_cuda_bytes_before_rollout_offload = local_bytes
        self._optimizer_copy_params_offloaded_for_rollout = True
        self._hdo_optimizer_residency_preserved = True
        after_estimate = estimate_optimizer_phase_offload_memory(self.actor_optimizer)
        sanity_error = None
        if (
            after_estimate["optimizer_phase_cuda_copy_param_bytes"] != 0
            or after_estimate["optimizer_phase_cuda_state_bytes"]
            != before_estimate["optimizer_phase_cuda_state_bytes"]
            or moved["moved_bytes"] != local_bytes
        ):
            sanity_error = RuntimeError(
                "EU-DERPO optimizer copy-param offload residency sanity check failed "
                f"before={before_estimate} after={after_estimate} moved={moved}"
            )
        self._complete_optimizer_copy_param_transition("offload_sanity", failure, sanity_error, moved)
        log_eu_derpo_memory(
            "after_optimizer_copy_param_offload",
            self.actor_optimizer,
            megatron_model_cpu_data_bytes(self.actor_module),
            extra={
                "copy_param_offload_active": 1,
                "optimizer_residency": self._actor_optimizer_residency(),
                "copy_param_moved_tensors": moved["moved_tensors"],
                "copy_param_moved_gib": moved["moved_bytes"] / _GIB,
                "copy_param_transition_elapsed_s": time.perf_counter() - started,
            },
        )
        return True

    def _restore_actor_optimizer_copy_params_for_update(self):
        if not self._optimizer_copy_params_offloaded_for_rollout:
            return False
        before_estimate = estimate_optimizer_phase_offload_memory(self.actor_optimizer)
        log_eu_derpo_memory(
            "before_optimizer_copy_param_restore",
            self.actor_optimizer,
            megatron_model_cpu_data_bytes(self.actor_module),
            extra={
                "copy_param_offload_active": 1,
                "optimizer_residency": self._actor_optimizer_residency(),
                "copy_param_transition_elapsed_s": 0.0,
            },
        )
        started = time.perf_counter()
        failure = (
            torch.zeros(1, dtype=torch.int32, device=get_device_name())
            if torch.distributed.is_initialized()
            else None
        )
        moved = {}
        error = None
        try:
            moved = load_megatron_optimizer_copy_params_to_gpu(self.actor_optimizer)
        except Exception as caught:
            error = caught
        self._complete_optimizer_copy_param_transition("restore", failure, error, moved)
        get_torch_device().synchronize()
        after_estimate = estimate_optimizer_phase_offload_memory(self.actor_optimizer)
        expected = self._optimizer_copy_params_cuda_bytes_before_rollout_offload
        sanity_error = None
        if (
            after_estimate["optimizer_phase_cuda_copy_param_bytes"] != expected
            or after_estimate["optimizer_phase_cuda_state_bytes"]
            != before_estimate["optimizer_phase_cuda_state_bytes"]
            or moved["moved_bytes"] != expected
        ):
            sanity_error = RuntimeError(
                "EU-DERPO optimizer copy-param restore residency sanity check failed "
                f"before={before_estimate} after={after_estimate} moved={moved} expected={expected}"
            )
        self._complete_optimizer_copy_param_transition("restore_sanity", failure, sanity_error, moved)
        self._optimizer_copy_params_offloaded_for_rollout = False
        reclaim_before_rss = psutil.Process().memory_info().rss / _GIB
        reclaim_before_available = psutil.virtual_memory().available / _GIB
        reclaim_common = {
            "copy_param_offload_active": 0,
            "optimizer_residency": self._actor_optimizer_residency(),
        }
        log_eu_derpo_memory(
            "before_optimizer_copy_param_host_reclaim",
            self.actor_optimizer,
            megatron_model_cpu_data_bytes(self.actor_module),
            extra={
                **reclaim_common,
                "gc_collected": 0,
                "malloc_trim_attempted": 0,
                "malloc_trim_rc": None,
                "host_reclaim_elapsed_s": 0.0,
                "mem_available_gain_gib": 0.0,
                "rss_drop_gib": 0.0,
            },
        )
        reclaim_started = time.perf_counter()
        reclaim = _best_effort_release_host_allocator_after_copy_param_restore()
        reclaim_elapsed = time.perf_counter() - reclaim_started
        reclaim_after_rss = psutil.Process().memory_info().rss / _GIB
        reclaim_after_available = psutil.virtual_memory().available / _GIB
        log_eu_derpo_memory(
            "after_optimizer_copy_param_host_reclaim",
            self.actor_optimizer,
            megatron_model_cpu_data_bytes(self.actor_module),
            extra={
                **reclaim_common,
                **reclaim,
                "host_reclaim_elapsed_s": reclaim_elapsed,
                "mem_available_gain_gib": reclaim_after_available - reclaim_before_available,
                "rss_drop_gib": reclaim_before_rss - reclaim_after_rss,
            },
        )
        log_eu_derpo_memory(
            "after_optimizer_copy_param_restore",
            self.actor_optimizer,
            megatron_model_cpu_data_bytes(self.actor_module),
            extra={
                "copy_param_offload_active": 0,
                "optimizer_residency": self._actor_optimizer_residency(),
                "copy_param_moved_tensors": moved["moved_tensors"],
                "copy_param_moved_gib": moved["moved_bytes"] / _GIB,
                "copy_param_transition_elapsed_s": time.perf_counter() - started,
            },
        )
        return True

    def _load_actor_optimizer_for_update(self):
        if self._optimizer_copy_params_offloaded_for_rollout:
            raise RuntimeError("ROLLOUT_PARTIAL optimizer reached generic update load")
        if not self._is_offload_optimizer or self._hdo_optimizer_residency_preserved:
            return False
        load_megatron_optimizer(self.actor_optimizer)
        return True

    def _offload_actor_optimizer(self):
        offload_megatron_optimizer(self.actor_optimizer)
        self._hdo_optimizer_residency_preserved = False
        self._optimizer_copy_params_offloaded_for_rollout = False
        self._optimizer_copy_params_cuda_bytes_before_rollout_offload = 0

    def _validate_hdo_optimizer_residency_config(self):
        selective = self._selective_optimizer_copy_param_offload_enabled()
        if selective and not self._preserve_hdo_optimizer_residency():
            raise RuntimeError(
                "EU_DERPO_SELECTIVE_COPY_PARAM_OFFLOAD_REQUIRES_PRESERVED_PARTIAL_HDO"
            )
        if not self._is_actor or not self._preserve_hdo_optimizer_residency():
            return
        optimizer_override = self.config.actor.optim.override_optimizer_config or {}
        estimate = estimate_hdo_memory(self.actor_optimizer)
        if (
            optimizer_override.get("optimizer_cpu_offload") is not True
            or optimizer_override.get("optimizer_offload_fraction") != 0.75
            or not self._is_offload_optimizer
            or estimate["hdo_cpu_param_numel"] <= 0
            or estimate["hdo_gpu_param_numel"] <= 0
        ):
            raise RuntimeError(
                "EU_DERPO_PRESERVE_HDO_REQUIRES_PARTIAL_HDO "
                "optimizer_cpu_offload=true optimizer_offload_fraction=0.75 "
                "and materialized CPU/GPU HDO parameter groups"
            )

    def _warn_eu_derpo_hdo_gpu_headroom(self, stage):
        if not self._hdo_optimizer_residency_preserved:
            return
        device = get_torch_device()
        cuda_free, cuda_total = device.mem_get_info()
        cuda_reserved = device.memory_reserved()
        if (
            cuda_free < _CHECKPOINT_MIN_CUDA_FREE_BYTES
            or cuda_reserved > _CHECKPOINT_MAX_CUDA_RESERVED_BYTES
        ):
            logger.warning(
                "EU-DERPO GPU headroom low: "
                f"stage={stage} rank={torch.distributed.get_rank()} "
                f"allocated_gib={device.memory_allocated() / _GIB:.2f} "
                f"reserved_gib={cuda_reserved / _GIB:.2f} "
                f"free_gib={cuda_free / _GIB:.2f} total_gib={cuda_total / _GIB:.2f}"
            )

    def _release_actor_cuda_cache_before_rollout_wakeup(self):
        eu_derpo = self.config.actor.eu_derpo
        if not (
            self.config.rollout.free_cache_engine
            and eu_derpo.enabled
            and eu_derpo.release_actor_cuda_cache_before_rollout_wakeup
        ):
            return False

        before = log_eu_derpo_memory(
            "before_rollout_cuda_cache_release",
            self.actor_optimizer,
            megatron_model_cpu_data_bytes(self.actor_module),
        )
        started = time.perf_counter()
        device = get_torch_device()
        device.synchronize()
        device.empty_cache()
        cuda_free, _ = device.mem_get_info()
        cuda_allocated = device.memory_allocated()
        cuda_reserved = device.memory_reserved()
        log_eu_derpo_memory(
            "after_rollout_cuda_cache_release",
            self.actor_optimizer,
            megatron_model_cpu_data_bytes(self.actor_module),
            extra={
                "reserved_minus_allocated_before_gib": (
                    before["cuda_reserved_gib"] - before["cuda_allocated_gib"]
                ),
                "reserved_minus_allocated_after_gib": (cuda_reserved - cuda_allocated) / _GIB,
                "cuda_free_gain_gib": cuda_free / _GIB - before["cuda_free_gib"],
                "cache_release_elapsed_s": time.perf_counter() - started,
            },
        )
        return True

    def _finish_actor_update_residency(self, defer_phase_offload, eu_enabled, preserve_hdo=False):
        if defer_phase_offload:
            if self._checkpoint_training_residency_held:
                raise RuntimeError("checkpoint training residency hold is already active")
            self._checkpoint_training_residency_held = True
            if eu_enabled:
                log_eu_derpo_memory(
                    "checkpoint_hold_enter",
                    self.actor_optimizer,
                    megatron_model_cpu_data_bytes(self.actor_module),
                    extra={
                        "checkpoint_hold_active": 1,
                        "defer_phase_offload_for_checkpoint": 1,
                    },
                )
        else:
            transition_started = time.perf_counter()
            before = None
            predicted_host_increment = 0
            selective_copy_param_offload = self._selective_optimizer_copy_param_offload_enabled()
            if eu_enabled:
                estimate = estimate_optimizer_phase_offload_memory(self.actor_optimizer)
                predicted_host_increment = estimate[
                    "optimizer_phase_cuda_copy_param_bytes"
                    if selective_copy_param_offload
                    else "optimizer_phase_cuda_total_bytes"
                ]
                before = log_eu_derpo_memory(
                    "before_actor_residency_transition",
                    self.actor_optimizer,
                    megatron_model_cpu_data_bytes(self.actor_module),
                    extra={"predicted_optimizer_host_increment_gib": predicted_host_increment / _GIB},
                )
            if self._is_offload_param:
                offload_megatron_model_to_cpu(self.actor_module)
                log_gpu_memory_usage("After offload actor params and grad during update_actor", logger=logger)
            if eu_enabled:
                log_eu_derpo_memory(
                    "after_model_offload",
                    self.actor_optimizer,
                    megatron_model_cpu_data_bytes(self.actor_module),
                )
            if selective_copy_param_offload:
                self._offload_actor_optimizer_copy_params_for_rollout()
                logger.warning("optimizer copy params selectively offloaded for rollout")
            elif self._is_offload_optimizer and not preserve_hdo:
                self._offload_actor_optimizer()
                log_gpu_memory_usage("After offload actor optimizer during update_actor", logger=logger)
            elif preserve_hdo:
                self._hdo_optimizer_residency_preserved = True
                logger.warning("optimizer phase offload skipped to preserve native partial HDO residency")

            if eu_enabled:
                after_available = psutil.virtual_memory().available / _GIB
                log_eu_derpo_memory(
                    "after_optimizer_residency_decision",
                    self.actor_optimizer,
                    megatron_model_cpu_data_bytes(self.actor_module),
                    extra={
                        "checkpoint_hold_active": 0,
                        "defer_phase_offload_for_checkpoint": 0,
                        "optimizer_phase_offload_skipped": int(preserve_hdo),
                        "copy_param_offload_active": int(
                            self._optimizer_copy_params_offloaded_for_rollout
                        ),
                        "predicted_optimizer_host_increment_gib": predicted_host_increment / _GIB,
                        "actual_rss_increment_gib": (
                            psutil.Process().memory_info().rss / _GIB - before["rss_gib"]
                        ),
                        "actual_mem_available_drop_gib": before["mem_available_gib"] - after_available,
                        "optimizer_residency_transition_elapsed_s": time.perf_counter() - transition_started,
                    },
                )

    def _release_checkpoint_residency_hold(self):
        if not self._checkpoint_training_residency_held:
            return False
        first_error = None

        def log_reoffload(stage, **extra):
            if self.config.actor.eu_derpo.enabled:
                log_eu_derpo_memory(
                    stage,
                    self.actor_optimizer,
                    megatron_model_cpu_data_bytes(self.actor_module),
                    extra={
                        "checkpoint_hold_active": 1,
                        "defer_phase_offload_for_checkpoint": 1,
                        **extra,
                    },
                )

        skip_optimizer_offload = bool(
            self.config.actor.eu_derpo.enabled
            and self.config.actor.eu_derpo.skip_post_checkpoint_optimizer_offload
        ) or self._preserve_hdo_optimizer_residency()
        selective_copy_param_offload = self._selective_optimizer_copy_param_offload_enabled()
        try:
            log_reoffload("post_ckpt_reoffload_before")
            if self._is_offload_param and not is_megatron_model_offloaded(self.actor_module):
                try:
                    offload_megatron_model_to_cpu(self.actor_module)
                except Exception as error:
                    first_error = error
            log_reoffload("post_ckpt_after_param_offload")
            log_reoffload("post_ckpt_before_optimizer_offload")
            if selective_copy_param_offload:
                try:
                    self._offload_actor_optimizer_copy_params_for_rollout()
                except Exception as error:
                    if first_error is None:
                        first_error = error
            elif (
                self._is_offload_optimizer
                and not skip_optimizer_offload
                and not is_megatron_optimizer_offloaded(self.actor_optimizer)
            ):
                try:
                    self._offload_actor_optimizer()
                except Exception as error:
                    if first_error is None:
                        first_error = error
            elif skip_optimizer_offload:
                if self._preserve_hdo_optimizer_residency():
                    self._hdo_optimizer_residency_preserved = True
                logger.warning("optimizer phase offload skipped by experiment")
            skipped = int(skip_optimizer_offload)
            log_reoffload("post_ckpt_after_optimizer_offload", optimizer_phase_offload_skipped=skipped)
            log_reoffload("post_ckpt_reoffload_done", optimizer_phase_offload_skipped=skipped)
        finally:
            if self.config.actor.eu_derpo.enabled:
                log_eu_derpo_memory(
                    "checkpoint_hold_exit" if first_error is None else "checkpoint_hold_release_failed",
                    self.actor_optimizer,
                    megatron_model_cpu_data_bytes(self.actor_module),
                    extra={
                        "checkpoint_hold_active": int(first_error is not None),
                        "defer_phase_offload_for_checkpoint": int(first_error is not None),
                    },
                )
            self._checkpoint_training_residency_held = first_error is not None
        aggressive_empty_cache(force_sync=True)
        if first_error is not None:
            raise first_error
        return True

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def release_checkpoint_residency_hold(self):
        """Idempotently restore phase-offloaded residency after a checkpoint hold."""
        return self._release_checkpoint_residency_hold()

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="rollout"))
    @GPUMemoryLogger(role="generate_sequences", logger=logger)
    @DistProfiler.annotate(color="red", role="rollout_generate")
    def generate_sequences(self, prompts: DataProto):
        assert self._is_rollout
        prompts = prompts.to(get_device_name())
        meta_info = {
            "eos_token_id": self.generation_config.eos_token_id
            if self.generation_config is not None
            else self.tokenizer.eos_token_id,
            "pad_token_id": self.generation_config.pad_token_id
            if self.generation_config is not None
            else self.tokenizer.pad_token_id,
        }
        prompts.meta_info.update(meta_info)
        preserve_hdo = self._preserve_hdo_optimizer_residency()
        if self.config.actor.eu_derpo.enabled:
            log_eu_derpo_memory(
                "before_generate_sequences",
                self.actor_optimizer,
                megatron_model_cpu_data_bytes(self.actor_module),
            )
        if self._is_offload_optimizer and not preserve_hdo:
            self._offload_actor_optimizer()

        timing_generate = {}
        if self._is_actor:  # For rollout only, we do not switch context.
            loop = get_event_loop()
            loop.run_until_complete(self.rollout_mode())
            log_gpu_memory_usage("After switch to rollout mode", logger=logger)

        with simple_timer("generate_sequences", timing_generate):
            output = self.rollout.generate_sequences(prompts=prompts)
        if self.config.actor.eu_derpo.enabled:
            log_eu_derpo_memory(
                "after_generate_sequences",
                self.actor_optimizer,
                megatron_model_cpu_data_bytes(self.actor_module),
                extra={"generate_sequences_elapsed_s": timing_generate["generate_sequences"]},
            )

        if self._is_actor:
            loop.run_until_complete(self.trainer_mode())
            log_gpu_memory_usage("After switch to trainer mode", logger=logger)

        # We calculate the average timing across all ranks
        # to make sure meta_info["timing"] is the same
        timing_generate_topk_ratio, timing_generate_min, timing_generate_max = topk_reduce_ratio_min_max(
            timing_generate["generate_sequences"]
        )
        timing_generate = reduce_timing(timing_generate)
        timing_generate.update(
            {
                "generation_timing/max": timing_generate_max,
                "generation_timing/min": timing_generate_min,
                "generation_timing/topk_ratio": timing_generate_topk_ratio,
            }
        )
        output.meta_info["timing"] = timing_generate
        output = output.to("cpu")
        # clear kv cache
        aggressive_empty_cache(force_sync=True)
        return output

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    @GPUMemoryLogger(role="compute_ref_log_prob", logger=logger)
    @DistProfiler.annotate(color="olive", role="ref_compute_log_prob")
    def compute_ref_log_prob(self, data: DataProto):
        if self.peft_cls is not None:
            # if is lora, actor without lora applied is the ref
            data.meta_info["is_lora"] = True
            return self.compute_log_prob(data)
        assert self._is_ref
        if self._ref_is_offload_param:
            load_megatron_model_to_gpu(self.ref_module, load_grad=False)
            log_gpu_memory_usage("After load ref params and grad during compute_ref_log_prob", logger=logger)
        micro_batch_size = self.config.ref.log_prob_micro_batch_size_per_gpu
        data.meta_info["micro_batch_size"] = micro_batch_size
        data.meta_info["max_token_len"] = self.config.ref.log_prob_max_token_len_per_gpu
        data.meta_info["use_dynamic_bsz"] = self.config.ref.log_prob_use_dynamic_bsz
        data.meta_info["temperature"] = self.config.rollout.temperature
        output, _, _ = self.ref_policy.compute_log_prob(data=data, calculate_entropy=False)
        output = DataProto.from_dict(tensors={"ref_log_prob": output})
        output = output.to("cpu")
        if self._ref_is_offload_param:
            offload_megatron_model_to_cpu(self.ref_module)
            log_gpu_memory_usage("After offload ref params and grad during compute_ref_log_prob", logger=logger)
        aggressive_empty_cache(force_sync=True)
        return output

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    @GPUMemoryLogger(role="compute_log_prob", logger=logger)
    @DistProfiler.annotate(color="blue", role="actor_compute_log_prob")
    def compute_log_prob(self, data: DataProto):
        assert self._is_actor
        if self._is_offload_param:
            load_megatron_model_to_gpu(self.actor_module, load_grad=False)
            log_gpu_memory_usage("After load actor params and grad during compute_log_prob", logger=logger)
        is_lora = data.meta_info.pop("is_lora", False)
        adapter_ctx = self.peft_cls.disable_adapter(self.actor_module) if is_lora else nullcontext()
        # HybridEngine recomputes the stable old-policy snapshot before the actor update.
        config_source = self.config.ref if is_lora else self.config.rollout
        data.meta_info["micro_batch_size"] = config_source.log_prob_micro_batch_size_per_gpu
        data.meta_info["max_token_len"] = config_source.log_prob_max_token_len_per_gpu
        data.meta_info["use_dynamic_bsz"] = config_source.log_prob_use_dynamic_bsz
        data.meta_info["temperature"] = self.config.rollout.temperature

        if self.enable_routing_replay and self.config.actor.router_replay.mode == "R2":
            RouterReplay.set_global_router_replay_action(RouterReplayAction.RECORD)

        if self.enable_routing_replay and self.config.actor.router_replay.mode == "R3":
            RouterReplay.set_global_router_replay_action(RouterReplayAction.REPLAY_FORWARD)

        with adapter_ctx:
            output, entropys, layers_topk_idx = self.actor.compute_log_prob(data=data, calculate_entropy=not is_lora)
        if is_lora:
            tensors = {"ref_log_prob": output}
        else:
            # Preserve the three-logprob plumbing; EU current-policy authority is captured by actual F.
            tensors = policy_prepass_tensors(output, entropys, self.config.actor.eu_derpo.enabled)
        output = DataProto.from_dict(
            tensors=tensors,
            meta_info={"temperature": self.config.rollout.temperature},
        )
        if self.config.actor.router_replay.mode == "R2":
            output.batch["routed_experts"] = layers_topk_idx

        if self.config.actor.router_replay.mode in ["R2", "R3"]:
            RouterReplay.clear_global_indices()
            RouterReplay.clear_global_router_replay_action()

        output = output.to("cpu")
        # clear kv cache
        if self._is_offload_param:
            offload_megatron_model_to_cpu(self.actor_module)
            log_gpu_memory_usage("After offload actor params and grad during compute_log_prob", logger=logger)
        aggressive_empty_cache(force_sync=True)
        return output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(
        self,
        checkpoint_path,
        hdfs_path=None,
        del_local_after_load=True,
        staged_restore=False,
    ):
        # No checkpoint to load, just offload the model and optimizer to CPU
        if checkpoint_path is None:
            if self._is_offload_param:
                offload_megatron_model_to_cpu(self.actor_module)
            if self._is_offload_optimizer:
                self._offload_actor_optimizer()
            log_gpu_memory_usage("After offload actor params and optimizer during load_checkpoint", logger=logger)
            return

        if staged_restore:
            if not (
                self._is_offload_param
                and self._is_offload_optimizer
                and self.checkpoint_mananager.use_dist_checkpointing
                and self.checkpoint_mananager.should_load_model
                and self.checkpoint_mananager.should_load_optimizer
                and self.checkpoint_mananager.should_load_extra
                and _is_gpu_adam_distributed_optimizer(self.actor_optimizer)
            ):
                raise RuntimeError(
                    "Staged resume requires full model/optimizer/extra DCP loading with param_offload=true, "
                    "optimizer_offload=true, and DistributedOptimizer + TransformerEngine FusedAdam"
                )

            _log_resume_memory("actor_initialized")
            _reset_resume_peak_memory()
            model_before_memory = _log_resume_memory("before_model_load")
            model_loaded_memory = None
            try:
                load_megatron_model_to_gpu(self.actor_module)
                sharded_sd_metadata = self.checkpoint_mananager.load_checkpoint(
                    local_path=checkpoint_path,
                    hdfs_path=hdfs_path,
                    del_local_after_load=False,
                    load_contents=("model", "extra"),
                )
                model_loaded_memory = _log_resume_memory("after_model_load")
            finally:
                offload_megatron_model_to_cpu(self.actor_module)
                aggressive_empty_cache(force_sync=True)
                model_offloaded_memory = _log_resume_memory("after_model_offload")
                if (
                    model_loaded_memory is not None
                    and model_loaded_memory["gpu_available"]
                    and model_offloaded_memory["gpu_allocated"] >= model_loaded_memory["gpu_allocated"]
                ):
                    logger.warning(
                        "RESUME_MEM_WARNING model offload did not reduce allocated GPU memory: "
                        "baseline_gib=%.2f loaded_gib=%.2f offloaded_gib=%.2f",
                        model_before_memory["gpu_allocated"] / _GIB,
                        model_loaded_memory["gpu_allocated"] / _GIB,
                        model_offloaded_memory["gpu_allocated"] / _GIB,
                    )

            _reset_resume_peak_memory()
            optimizer_before_memory = _log_resume_memory("before_optimizer_load")
            optimizer_loaded_memory = None
            try:
                load_megatron_optimizer(self.actor_optimizer)
                self.checkpoint_mananager.load_checkpoint(
                    local_path=checkpoint_path,
                    hdfs_path=hdfs_path,
                    del_local_after_load=del_local_after_load,
                    load_contents=("optimizer",),
                    sharded_sd_metadata=sharded_sd_metadata,
                    stage_callback=_log_resume_memory,
                )
                optimizer_loaded_memory = _log_resume_memory("after_optimizer_load")
            finally:
                self._offload_actor_optimizer()
                aggressive_empty_cache(force_sync=True)
                optimizer_offloaded_memory = _log_resume_memory("after_optimizer_offload")
                if (
                    optimizer_loaded_memory is not None
                    and optimizer_loaded_memory["gpu_available"]
                    and optimizer_offloaded_memory["gpu_allocated"] >= optimizer_loaded_memory["gpu_allocated"]
                ):
                    logger.warning(
                        "RESUME_MEM_WARNING optimizer offload did not reduce allocated GPU memory: "
                        "baseline_gib=%.2f loaded_gib=%.2f offloaded_gib=%.2f",
                        optimizer_before_memory["gpu_allocated"] / _GIB,
                        optimizer_loaded_memory["gpu_allocated"] / _GIB,
                        optimizer_offloaded_memory["gpu_allocated"] / _GIB,
                    )

            _log_resume_memory("restore_complete")
            return

        if self._is_offload_param:
            load_megatron_model_to_gpu(self.actor_module)
        if self._is_offload_optimizer:
            load_megatron_optimizer(self.actor_optimizer)
        try:
            self.checkpoint_mananager.load_checkpoint(
                local_path=checkpoint_path, hdfs_path=hdfs_path, del_local_after_load=del_local_after_load
            )
        finally:
            if self._is_offload_param:
                offload_megatron_model_to_cpu(self.actor_module)
            if self._is_offload_optimizer:
                self._offload_actor_optimizer()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_pretrained_model(self, checkpoint_path, del_local_after_load=True):
        pass

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, checkpoint_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None):
        checkpoint_hold = self._checkpoint_training_residency_held
        copy_params_were_partial = self._optimizer_copy_params_offloaded_for_rollout
        if copy_params_were_partial:
            self._restore_actor_optimizer_copy_params_for_update()
        model_was_offloaded = self._is_offload_param and is_megatron_model_offloaded(self.actor_module)
        optimizer_was_offloaded = self._is_offload_optimizer and is_megatron_optimizer_offloaded(
            self.actor_optimizer
        )
        eu_enabled = self.config.actor.eu_derpo.enabled

        def log_checkpoint_memory(stage, checkpoint_extra=None):
            if eu_enabled:
                extra = {
                    "checkpoint_hold_active": int(self._checkpoint_training_residency_held),
                    "defer_phase_offload_for_checkpoint": int(checkpoint_hold),
                }
                if checkpoint_extra:
                    extra.update(checkpoint_extra)
                log_eu_derpo_memory(
                    stage,
                    self.actor_optimizer,
                    megatron_model_cpu_data_bytes(self.actor_module),
                    extra=extra,
                )

        log_checkpoint_memory("before_checkpoint")
        model_loaded = False
        optimizer_loaded = False
        try:
            if checkpoint_hold and (model_was_offloaded or optimizer_was_offloaded):
                raise RuntimeError("checkpoint residency hold lost training residency before save")
            if model_was_offloaded:
                load_megatron_model_to_gpu(self.actor_module)
                model_loaded = True
            if optimizer_was_offloaded:
                load_megatron_optimizer(self.actor_optimizer)
                optimizer_loaded = True
            log_checkpoint_memory("after_checkpoint_training_residency_load")
            if checkpoint_hold:
                device = get_torch_device()
                cuda_free, cuda_total = device.mem_get_info()
                cuda_reserved = device.memory_reserved()
                local_insufficient = (
                    cuda_free < _CHECKPOINT_MIN_CUDA_FREE_BYTES
                    or cuda_reserved > _CHECKPOINT_MAX_CUDA_RESERVED_BYTES
                )
                failure = torch.tensor(
                    int(local_insufficient), device=get_device_name(), dtype=torch.int32
                )
                torch.distributed.all_reduce(failure, op=torch.distributed.ReduceOp.MAX)
                log_checkpoint_memory("checkpoint_gpu_headroom")
                if failure.item():
                    raise RuntimeError(
                        "CHECKPOINT_GPU_HEADROOM_INSUFFICIENT "
                        f"rank={torch.distributed.get_rank()} "
                        f"allocated_gib={device.memory_allocated() / _GIB:.2f} "
                        f"reserved_gib={cuda_reserved / _GIB:.2f} "
                        f"free_gib={cuda_free / _GIB:.2f} total_gib={cuda_total / _GIB:.2f}"
                    )
            self.checkpoint_mananager.save_checkpoint(
                local_path=checkpoint_path,
                hdfs_path=hdfs_path,
                global_step=global_step,
                max_ckpt_to_keep=max_ckpt_to_keep,
                stage_callback=log_checkpoint_memory if eu_enabled else None,
            )
            torch.distributed.barrier()
        finally:
            if checkpoint_hold:
                self._release_checkpoint_residency_hold()
            else:
                if model_loaded:
                    offload_megatron_model_to_cpu(self.actor_module)
                if optimizer_loaded:
                    self._offload_actor_optimizer()
                elif copy_params_were_partial:
                    self._offload_actor_optimizer_copy_params_for_rollout()
                log_checkpoint_memory("after_checkpoint_reoffload")

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def async_calls_finalize_fn_exec(self, blocking=False):
        from megatron.core.dist_checkpointing.strategies.base import async_calls

        async_calls.maybe_finalize_async_calls(blocking=blocking)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def start_profile(self, **kwargs) -> None:
        """Start profiling for the current rank in the current training step."""
        self.profiler.start(**kwargs)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def stop_profile(self) -> None:
        """Stop profiling for the current rank in the current training step."""
        self.profiler.stop()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def dump_memory_snapshot(self, tag: str = "manual", sub_dir: str = None) -> None:
        """Manually trigger a CUDA memory snapshot dump on all ranks."""
        # Memory snapshot is now handled by the profiler system
        # This method is kept for backward compatibility but delegates to profiler
        if hasattr(self, "profiler") and hasattr(self.profiler, "_impl"):
            try:
                # Try to use the profiler's memory snapshot functionality
                if hasattr(self.profiler._impl, "sampler"):
                    out_dir = OmegaConf.select(self.config, "actor.profiler.save_path") or "."
                    self.profiler._impl.sampler.dump_memory_snapshot(out_dir=out_dir, tag=tag, sub_dir=sub_dir)
            except Exception as e:
                # Log a warning if memory snapshot fails. This might be expected if the profiler doesn't support it.
                logger.warning(f"Failed to dump memory snapshot: {e}")


class AsyncActorRolloutRefWorker(ActorRolloutRefWorker):
    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    async def update_weights(self, global_steps: int = None):
        await self.rollout_mode()
        return True


class CriticWorker(MegatronWorker, DistProfilerExtension):
    def __init__(self, config: McoreCriticConfig):
        Worker.__init__(self)

        omega_profiler_config = config.get("profiler", {})
        profiler_config = omega_conf_to_dataclass(omega_profiler_config, dataclass_type=ProfilerConfig)
        if omega_profiler_config.get("tool", None) in ["npu", "nsys", "torch", "torch_memory"]:
            tool_config = omega_conf_to_dataclass(
                omega_profiler_config.get("tool_config", {}).get(omega_profiler_config.get("tool"))
            )
        else:
            tool_config = None
        DistProfilerExtension.__init__(
            self, DistProfiler(rank=self.rank, config=profiler_config, tool_config=tool_config)
        )
        self.config: McoreCriticConfig = config

        # NOTE(sgm): We utilize colocate WorkerGroup by default.
        # As a result, Workers for different model share the same process.
        # Therefore, we only require one distribute initialization.
        # To utilize different parallel strategy in different models:
        # 1, users should disable WorkerDict; 2.assign different ResourcePool to different models,
        # 3. and apply the following patch in ray==2.10, https://github.com/ray-project/ray/pull/44385
        if not torch.distributed.is_initialized():
            set_numa_affinity()
            rank = int(os.environ["LOCAL_RANK"])
            torch.distributed.init_process_group(
                backend=get_nccl_backend(),
                timeout=datetime.timedelta(seconds=self.config.get("nccl_timeout", 600)),
                init_method=os.environ.get("DIST_INIT_METHOD", None),
            )
            get_torch_device().set_device(rank)

            mpu.initialize_model_parallel(
                tensor_model_parallel_size=self.config.megatron.tensor_model_parallel_size,
                pipeline_model_parallel_size=self.config.megatron.pipeline_model_parallel_size,
                virtual_pipeline_model_parallel_size=self.config.megatron.virtual_pipeline_model_parallel_size,
                use_sharp=False,
                context_parallel_size=self.config.megatron.context_parallel_size,
                expert_model_parallel_size=self.config.megatron.expert_model_parallel_size,
                expert_tensor_parallel_size=self.config.megatron.expert_tensor_parallel_size,
                nccl_communicator_config_path=None,
            )

        is_collect = (
            mpu.get_tensor_model_parallel_rank() == 0
            and mpu.get_pipeline_model_parallel_rank() == mpu.get_pipeline_model_parallel_world_size() - 1
            and mpu.get_context_parallel_rank() == 0
        )
        self._register_dispatch_collect_info(
            mesh_name="critic", dp_rank=mpu.get_data_parallel_rank(), is_collect=is_collect
        )

        set_random_seed(seed=self.config.megatron.seed)

        # set FSDP offload params
        self._is_offload_param = self.config.megatron.param_offload
        self._is_offload_optimizer = self.config.megatron.optimizer_offload

        # normalize config
        self.config.ppo_mini_batch_size *= self.config.rollout_n
        self.config.ppo_mini_batch_size //= mpu.get_data_parallel_world_size()
        if self.config.get("ppo_micro_batch_size", None):
            self.config.ppo_micro_batch_size //= mpu.get_data_parallel_world_size()
            self.config.ppo_micro_batch_size_per_gpu = self.config.ppo_micro_batch_size

        # TODO(sgm): support critic model offload

    def _build_critic_model_optimizer(
        self, model_path, optim_config, override_model_config, override_transformer_config, override_ddp_config
    ):
        from verl.utils.megatron.optimizer import (
            get_megatron_optimizer,
            get_megatron_optimizer_param_scheduler,
            init_megatron_optim_config,
        )
        from verl.utils.megatron_utils import McoreModuleWrapperConfig, make_megatron_module
        from verl.utils.model import print_model_size

        self._init_hf_config_and_tf_config(
            model_path,
            self.config.model.get("tokenizer_path") or model_path,
            self.dtype,
            override_model_config,
            override_transformer_config,
            self.config.model.get("trust_remote_code", False),
            self.config.megatron,
        )

        wrap_config = McoreModuleWrapperConfig(
            is_value_model=True,  # critic is value model
            share_embeddings_and_output_weights=False,
            wrap_with_ddp=True,
            use_distributed_optimizer=self.config.megatron.use_distributed_optimizer,
        )
        critic_module, updated_tf_config = make_megatron_module(
            wrap_config=wrap_config,
            tf_config=self.tf_config,
            hf_config=self.hf_config,
            bridge=self.bridge,
            provider=self.provider,
            override_model_config=override_model_config,
            override_ddp_config=override_ddp_config,
            peft_cls=self.peft_cls,
            peft_config=self.config.model.get("lora", None),
        )
        self.tf_config = updated_tf_config
        # note that here critic_module will be a list to be compatible with the construction of interleaved pp (vpp).
        # but here, we do not use pp (vpp) yet. For simplicity, we remove the list
        # critic_module = nn.ModuleList(critic_module)

        if self.config.load_weight:
            t0 = time.time()
            if self.config.megatron.use_dist_checkpointing:
                load_mcore_dist_weights(
                    critic_module,
                    self.config.megatron.dist_checkpointing_path,
                    is_value_model=True,
                    prefix=self.config.megatron.dist_checkpointing_prefix,
                )
            else:
                if self.bridge is not None:
                    local_model_path = get_hf_model_path(self.config)
                    if self.vanilla_bridge:
                        self.bridge.load_weights(critic_module, local_model_path)
                    else:
                        self.bridge.load_hf_weights(
                            critic_module, local_model_path, allowed_mismatched_params=["output_layer.weight"]
                        )
                else:
                    load_megatron_gptmodel_weights(
                        self.config, self.hf_config, critic_module, params_dtype=self.dtype, is_value_model=True
                    )
            t1 = time.time()
            if torch.distributed.get_rank() == 0:
                print(f"critic load_weight time: {t1 - t0}")
        if self.rank == 0:
            print_model_size(critic_module[0])

        # TODO: add more optimizer args into config
        optim_config_megatron = init_megatron_optim_config(
            optim_config,
            use_distributed_optimizer=wrap_config.use_distributed_optimizer,
            fp16=self.dtype == torch.float16,
        )
        critic_optimizer = get_megatron_optimizer(model=critic_module, config=optim_config_megatron)
        critic_optimizer_scheduler = get_megatron_optimizer_param_scheduler(
            optimizer=critic_optimizer, config=optim_config
        )
        get_torch_device().empty_cache()

        register_megatron_training_hooks(critic_module, critic_optimizer)

        return critic_module, critic_optimizer, critic_optimizer_scheduler, self.hf_config, optim_config

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        # create critic

        from verl.utils.torch_dtypes import PrecisionType

        if self.config.model.get("external_lib", None) is not None:
            # This is used to import external_lib into the huggingface systems
            import importlib

            importlib.import_module(self.config.model.external_lib)
        override_model_config = OmegaConf.to_container(OmegaConf.create(self.config.model.get("override_config", {})))
        override_transformer_config = OmegaConf.to_container(
            OmegaConf.create(self.config.megatron.get("override_transformer_config", {}))
        )
        override_ddp_config = OmegaConf.to_container(
            OmegaConf.create(self.config.megatron.get("override_ddp_config", {}))
        )
        self.param_dtype = PrecisionType.to_dtype(self.config.megatron.dtype)
        self.dtype = PrecisionType.to_dtype(self.param_dtype)
        (
            self.critic_module,
            self.critic_optimizer,
            self.critic_optimizer_scheduler,
            self.critic_model_config,
            critic_optimizer_config,
        ) = self._build_critic_model_optimizer(
            model_path=self.config.model.path,
            optim_config=self.config.optim,
            override_model_config=override_model_config,
            override_transformer_config=override_transformer_config,
            override_ddp_config=override_ddp_config,
        )
        if self._is_offload_param:
            offload_megatron_model_to_cpu(self.critic_module)
        if self._is_offload_optimizer:
            offload_megatron_optimizer(self.critic_optimizer)

        self.critic = MegatronPPOCritic(
            config=self.config,
            model_config=self.critic_model_config,
            hf_config=self.hf_config,
            tf_config=self.tf_config,
            critic_module=self.critic_module,
            critic_optimizer=self.critic_optimizer,
            critic_optimizer_config=critic_optimizer_config,
        )
        self.flops_counter = FlopsCounter(self.critic_model_config)
        self.checkpoint_mananager = MegatronCheckpointManager(
            config=self.config,
            checkpoint_config=self.config.checkpoint,
            model_config=self.critic_model_config,
            transformer_config=self.tf_config,
            role="critic",
            model=self.critic_module,
            arch=self.architectures[0],
            hf_config=self.hf_config,
            param_dtype=self.param_dtype,
            share_embeddings_and_output_weights=False,
            processing_class=self.processor if self.processor is not None else self.tokenizer,
            optimizer=self.critic_optimizer,
            optimizer_scheduler=self.critic_optimizer_scheduler,
            use_distributed_optimizer=self.config.megatron.use_distributed_optimizer,
            use_checkpoint_opt_param_scheduler=self.config.optim.use_checkpoint_opt_param_scheduler,
            bridge=self.bridge,
            provider=self.provider,
            use_dist_checkpointing=self.config.megatron.use_dist_checkpointing,
            peft_cls=self.peft_cls,
        )

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="critic"))
    @DistProfiler.annotate(color="cyan", role="compute_values")
    def compute_values(self, data: DataProto):
        micro_batch_size = self.config.ppo_micro_batch_size_per_gpu
        data.meta_info["micro_batch_size"] = micro_batch_size
        data.meta_info["max_token_len"] = self.config.forward_max_token_len_per_gpu
        data.meta_info["use_dynamic_bsz"] = self.config.use_dynamic_bsz
        data = data.to(get_device_id())
        if self._is_offload_param:
            load_megatron_model_to_gpu(self.critic_module)
        values = self.critic.compute_values(data=data)
        output = DataProto.from_dict(tensors={"values": values})
        output = output.to("cpu")
        if self._is_offload_param:
            offload_megatron_model_to_cpu(self.critic_module)
        return output

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="critic"))
    @DistProfiler.annotate(color="pink", role="critic_update")
    def update_critic(self, data: DataProto):
        data = data.to(get_device_id())

        if self._is_offload_param:
            load_megatron_model_to_gpu(self.critic_module)
        if self._is_offload_optimizer:
            load_megatron_optimizer(self.critic_optimizer)

        dataloader = self.critic.make_minibatch_iterator(data)
        with Timer(name="update_critic", logger=None) as timer:
            metrics = self.critic.update_critic(dataloader=dataloader)
        delta_time = timer.last
        global_num_tokens = data.meta_info["global_token_num"]
        estimated_flops, promised_flops = self.flops_counter.estimate_flops(global_num_tokens, delta_time)
        metrics["perf/mfu/critic"] = estimated_flops * self.config.ppo_epochs / promised_flops / self.world_size
        from verl.utils.megatron.optimizer import get_megatron_last_lr

        metrics["critic/lr"] = get_megatron_last_lr(self.critic_optimizer)
        self.critic_optimizer_scheduler.step(1)

        output = DataProto(batch=None, meta_info={"metrics": metrics})

        if self._is_offload_param:
            offload_megatron_model_to_cpu(self.critic_module)
        if self._is_offload_optimizer:
            offload_megatron_optimizer(self.critic_optimizer)
        output = output.to("cpu")
        return output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, checkpoint_path, hdfs_path=None, del_local_after_load=True):
        if self._is_offload_param:
            load_megatron_model_to_gpu(self.critic_module)
        self.checkpoint_mananager.load_checkpoint(
            local_path=checkpoint_path, hdfs_path=hdfs_path, del_local_after_load=del_local_after_load
        )
        if self._is_offload_param:
            offload_megatron_model_to_cpu(self.critic_module)
        if self._is_offload_optimizer:
            offload_megatron_optimizer(self.critic_optimizer)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, checkpoint_path, hdfs_path=None, global_steps=0, max_ckpt_to_keep=None):
        if self._is_offload_param:
            load_megatron_model_to_gpu(self.critic_module)
        self.checkpoint_mananager.save_checkpoint(
            local_path=checkpoint_path, hdfs_path=hdfs_path, global_step=global_steps, max_ckpt_to_keep=max_ckpt_to_keep
        )
        if self._is_offload_param:
            offload_megatron_model_to_cpu(self.critic_module)
