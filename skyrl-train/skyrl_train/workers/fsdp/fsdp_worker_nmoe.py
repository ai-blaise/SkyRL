"""NMoE-specific FSDP Workers for SkyRL RL Training.

This module provides NMoE-specialized versions of the FSDP workers that use
NMoEModelWrapper instead of HFModelWrapper for policy, reference, and critic models.

Key differences from standard FSDP workers:
- Uses NMoEModelWrapper for model wrapping
- Uses NMoEWeightExtractor for EP-aware weight extraction
- Supports expert cache refresh after optimizer steps
- Handles nmoe checkpoint format
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property
from typing import Iterator, List, Optional, TYPE_CHECKING

import torch
import torch.nn as nn

try:
    from torch.distributed.tensor import DTensor
except ImportError:
    from torch.distributed._tensor import DTensor

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from skyrl_train.training_batch import TrainingInputBatch, TrainingOutputBatch


# =============================================================================
# Local copies of weight sync classes to avoid ray dependency at module load.
# The weight_sync package imports ray through broadcast_strategy.py in __init__.py
# =============================================================================

@dataclass
class WeightChunk:
    """Represents one or more model parameters to be transferred.

    This is a local copy to avoid importing from weight_sync package which
    triggers ray import through broadcast_strategy.
    """

    names: List[str]
    dtypes: List[str]
    shapes: List[List[int]]
    tensors: List[torch.Tensor]

    def __post_init__(self):
        """Validate that all input lists have the same length."""
        lengths = [len(self.names), len(self.dtypes), len(self.shapes), len(self.tensors)]
        if len(set(lengths)) != 1:
            raise ValueError(
                f"All lists must have the same length. Got names={len(self.names)}, "
                f"dtypes={len(self.dtypes)}, shapes={len(self.shapes)}, tensors={len(self.tensors)}"
            )

    def __len__(self) -> int:
        """Return the number of parameters in this chunk."""
        return len(self.names)

    @cached_property
    def total_numel(self) -> int:
        """Calculate total number of elements across all tensors."""
        return sum(t.numel() for t in self.tensors)

    @cached_property
    def total_size_bytes(self) -> int:
        """Calculate total memory footprint in bytes."""
        return sum(t.numel() * t.element_size() for t in self.tensors)


class WeightExtractor(ABC):
    """Extracts weights from training backend models.

    This is a local copy to avoid importing from weight_sync package which
    triggers ray import through broadcast_strategy.
    """

    @abstractmethod
    def extract_weights(self, dtype: torch.dtype) -> Iterator[WeightChunk]:
        """Extract weights from the model as WeightChunk objects."""
        ...


def _fsdp_version(model) -> int:
    """Get FSDP version (1 or 2) for a model.

    This is a simplified version that avoids importing the full fsdp_utils module.
    """
    try:
        from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
        if isinstance(model, FSDP):
            return 1 if hasattr(model, '_handle') else 2
    except ImportError:
        pass
    return 2  # Default to FSDP2 behavior


class NMoEFSDPWeightExtractor(WeightExtractor):
    """Extracts weights from FSDP-sharded nmoe models.

    This extractor handles nmoe models wrapped in FSDP, supporting:
    - EP-aware expert weight extraction
    - Module-grouped chunking for efficient transfer
    - Both expert and dense parameter handling
    """

    def __init__(
        self,
        model: torch.nn.Module,
        group_by_module: bool = False,
        batch_size_threshold_gb: float = 0.0,
    ):
        self.model = model
        self.group_by_module = group_by_module
        self.batch_size_threshold_gb = batch_size_threshold_gb

    def extract_weights(self, dtype: torch.dtype):
        """Extract weights from FSDP-wrapped nmoe model."""
        from torch.distributed.fsdp.api import ShardedStateDictConfig, StateDictType
        from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
        from skyrl_train.weight_sync.weight_extractor_utils import yield_module_grouped_chunks

        # Configure state_dict type for FSDP v1
        if _fsdp_version(self.model) == 1:
            FSDP.set_state_dict_type(
                self.model,
                state_dict_type=StateDictType.SHARDED_STATE_DICT,
                state_dict_config=ShardedStateDictConfig(),
            )

        # Get state dict (handles FSDP sharding)
        params = self.model.state_dict()

        if not self.group_by_module:
            # Simple path: yield one chunk per parameter
            for name, param in params.items():
                tensor = self._gather_tensor(param).to(dtype).detach().contiguous()
                yield WeightChunk(
                    names=[name],
                    dtypes=[str(dtype)],
                    shapes=[list(tensor.shape)],
                    tensors=[tensor],
                )
        else:
            for chunk in yield_module_grouped_chunks(
                params=params,
                dtype=dtype,
                gather_tensor_fn=self._gather_tensor,
                get_shape_fn=lambda name, param, tensor: list(tensor.shape),
                batch_size_threshold_gb=self.batch_size_threshold_gb,
            ):
                yield chunk

    def _gather_tensor(self, param: torch.Tensor) -> torch.Tensor:
        """Gather sharded tensor into full tensor."""
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            return param.to(device, non_blocking=True).full_tensor() if isinstance(param, DTensor) else param
        return param.full_tensor() if isinstance(param, DTensor) else param


def _get_policy_worker_base():
    """Lazily import PolicyWorkerBase to avoid ray dependency at module load."""
    from skyrl_train.workers.worker import PolicyWorkerBase
    return PolicyWorkerBase


def _get_ref_worker_base():
    """Lazily import RefWorkerBase to avoid ray dependency at module load."""
    from skyrl_train.workers.worker import RefWorkerBase
    return RefWorkerBase


class NMoEFSDPPolicyWorkerMixin:
    """Mixin providing NMoE-specific functionality for policy workers.

    This mixin adds:
    - NMoE model initialization using NMoEModelWrapper
    - Expert cache refresh support
    - NMoE-specific weight extraction
    """

    def init_nmoe_model(self, model_path, num_training_steps: int = None):
        """Initialize nmoe model with FSDP strategy."""
        from skyrl_train.model_wrapper_nmoe import NMoEModelWrapper
        from skyrl_train.distributed.fsdp_strategy import FSDPStrategy
        from skyrl_train.distributed.fsdp_utils import get_init_weight_context_manager

        assert self.cfg.trainer.strategy in ("fsdp", "fsdp2")
        strategy = FSDPStrategy(
            fsdp_config=self.cfg.trainer.policy.fsdp_config,
            optimizer_config=self.cfg.trainer.policy.optimizer_config,
            model_config=self.cfg.trainer.policy.model,
            fsdp_strategy=self.cfg.trainer.strategy,
            seed=self.cfg.trainer.seed,
            micro_train_batch_size_per_gpu=self.cfg.trainer.micro_train_batch_size_per_gpu,
            num_training_steps=num_training_steps,
        )
        strategy.setup_distributed()
        self.strategy = strategy

        # Get LoRA configuration
        lora_config = self.cfg.trainer.policy.model.lora
        self._is_lora = lora_config.rank > 0

        # Update per-gpu mini batch size based on device mesh
        self._normalize_mini_batch_size()

        # Get nmoe config
        gradient_checkpointing = self.cfg.trainer.gradient_checkpointing

        # Create NMoEModelWrapper
        init_context = get_init_weight_context_manager(
            use_meta_tensor=False,
            mesh=self.strategy.device_mesh
        )

        with init_context():
            wrapped_model = NMoEModelWrapper(
                model_path,
                temperature=1.0,
                use_torch_compile=self.cfg.trainer.policy.use_torch_compile,
                gradient_checkpointing=gradient_checkpointing,
                # LoRA configuration
                lora_rank=lora_config.rank,
                lora_alpha=lora_config.alpha,
                lora_dropout=lora_config.dropout,
                lora_target_modules=lora_config.target_modules,
                lora_exclude_modules=lora_config.exclude_modules,
                lora_init_method=getattr(lora_config, 'init_method', 'gaussian'),
                lora_include_experts=getattr(lora_config, 'include_experts', False),
            )

        self.model, self.optimizer, self.scheduler = strategy.prepare(
            (wrapped_model, None, None),
        )
        assert (
            self.optimizer is not None and self.scheduler is not None
        ), "FSDP preparation should create optimizer and scheduler"

        # Initialize weight extractor for nmoe
        # Lazy import to avoid ray dependency at module load
        from skyrl_train.weight_sync.cuda_ipc_strategy import CudaIpcTransferStrategy

        group_by_module = self._transfer_strategy_cls is CudaIpcTransferStrategy
        self.weight_extractor = NMoEFSDPWeightExtractor(
            self.model.model,
            group_by_module=group_by_module,
            batch_size_threshold_gb=(
                self.cfg.generator.weight_transfer_threshold_cuda_ipc_GB if group_by_module else 0.0
            ),
        )

        # Track if model uses quantized experts
        self._uses_quantized_experts = wrapped_model.uses_quantized_experts

        logger.info(f"[NMoE] Initialized FSDP policy worker with model from {model_path}")

    def _refresh_expert_caches(self):
        """Refresh expert caches after optimizer step (for FP8/NVFP4 models)."""
        if hasattr(self.model, 'refresh_expert_caches'):
            self.model.refresh_expert_caches()
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'refresh_expert_caches'):
            self.model.model.refresh_expert_caches()


class NMoEFSDPCriticWorkerMixin:
    """Mixin providing NMoE-specific functionality for critic workers.

    This mixin adds:
    - NMoE critic model initialization using NMoECriticWrapper
    - Expert cache refresh support for PPO training
    - Value head parameter handling
    """

    def init_nmoe_critic_model(self, model_path, num_training_steps: int = None):
        """Initialize nmoe critic model with FSDP strategy."""
        from skyrl_train.model_wrapper_nmoe import NMoECriticWrapper
        from skyrl_train.distributed.fsdp_strategy import FSDPStrategy
        from skyrl_train.distributed.fsdp_utils import get_init_weight_context_manager

        assert self.cfg.trainer.strategy in ("fsdp", "fsdp2")
        strategy = FSDPStrategy(
            fsdp_config=self.cfg.trainer.critic.fsdp_config,
            optimizer_config=self.cfg.trainer.critic.optimizer_config,
            fsdp_strategy=self.cfg.trainer.strategy,
            seed=self.cfg.trainer.seed,
            micro_train_batch_size_per_gpu=self.cfg.trainer.micro_train_batch_size_per_gpu,
            num_training_steps=num_training_steps,
        )
        strategy.setup_distributed()
        self.strategy = strategy

        # Update per-gpu mini batch size based on device mesh
        self._normalize_mini_batch_size()

        # Get config options
        gradient_checkpointing = self.cfg.trainer.gradient_checkpointing
        value_head_prefix = getattr(self.cfg.trainer.algorithm, 'value_head_prefix', 'value_head')

        # Determine if we should initialize value head weights
        # If critic and policy share the same base model, init value head
        init_value_head = (
            self.cfg.trainer.policy.model.path == self.cfg.trainer.critic.model.path
        )

        # Create NMoECriticWrapper
        init_context = get_init_weight_context_manager(
            use_meta_tensor=False,
            mesh=self.strategy.device_mesh
        )

        with init_context():
            critic = NMoECriticWrapper(
                model_path,
                value_head_prefix=value_head_prefix,
                init_value_head=init_value_head,
                gradient_checkpointing=gradient_checkpointing,
            )

        self.model, self.optimizer, self.scheduler = strategy.prepare(
            (critic, None, None),
        )
        assert (
            self.optimizer is not None and self.scheduler is not None
        ), "FSDP preparation should create optimizer and scheduler"

        # Track if model uses quantized experts
        self._uses_quantized_experts = critic.uses_quantized_experts

        logger.info(f"[NMoE Critic] Initialized FSDP critic worker with model from {model_path}")

    def _refresh_expert_caches(self):
        """Refresh expert caches after optimizer step (for FP8/NVFP4 models)."""
        if hasattr(self.model, 'refresh_expert_caches'):
            self.model.refresh_expert_caches()
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'refresh_expert_caches'):
            self.model.model.refresh_expert_caches()


class NMoEFSDPRefWorkerMixin:
    """Mixin providing NMoE-specific functionality for reference workers."""

    def init_nmoe_ref_model(self, model_path):
        """Initialize frozen nmoe reference model."""
        from skyrl_train.model_wrapper_nmoe import NMoEModelWrapper
        from skyrl_train.distributed.fsdp_strategy import FSDPStrategy
        from skyrl_train.distributed.fsdp_utils import get_init_weight_context_manager

        assert self.cfg.trainer.strategy in ("fsdp", "fsdp2")
        strategy = FSDPStrategy(
            fsdp_config=self.cfg.trainer.ref.fsdp_config,
            fsdp_strategy=self.cfg.trainer.strategy,
            seed=self.cfg.trainer.seed,
            micro_train_batch_size_per_gpu=self.cfg.trainer.micro_train_batch_size_per_gpu,
        )
        strategy.setup_distributed()
        self.strategy = strategy

        init_context = get_init_weight_context_manager(
            use_meta_tensor=False,
            mesh=self.strategy.device_mesh
        )

        with init_context():
            wrapped_model = NMoEModelWrapper(
                model_path,
                temperature=1.0,
                use_torch_compile=False,
                gradient_checkpointing=False,
            )
            wrapped_model.freeze_for_reference()

        self.model = strategy.prepare(wrapped_model)
        self.model.eval()

        logger.info(f"[NMoE] Initialized FSDP reference worker with frozen model from {model_path}")


def get_nmoe_workers():
    """Get NMoE-specific worker classes.

    Returns:
        Tuple of (PolicyWorker, CriticWorker, RefWorker) for FSDP-based NMoE training.

    Note: Imports ray lazily to avoid import errors when ray is not installed.
    """
    import ray
    import torch.distributed

    from skyrl_train.workers.worker import PolicyWorkerBase, CriticWorkerBase, RefWorkerBase
    from skyrl_train.distributed.fsdp_strategy import FSDPStrategy
    from skyrl_train.distributed.fsdp_utils import fsdp_version
    from skyrl_train.training_batch import TrainingInputBatch, TrainingOutputBatch

    class NMoEFSDPPolicyWorkerBase(NMoEFSDPPolicyWorkerMixin, PolicyWorkerBase):
        """FSDP Policy Worker for NMoE models."""

        def offload_to_cpu(self, pin_memory=True, non_blocking=True, offload_optimizer=True, offload_model=True):
            self._set_numa_affinity(torch.distributed.get_rank() % torch.cuda.device_count())
            self.strategy.offload_to_cpu(
                self.model, self.optimizer, pin_memory, non_blocking, offload_optimizer, offload_model
            )

        def backload_to_gpu(self, non_blocking=True, backload_optimizer=True, backload_model=True):
            self.strategy.backload_to_gpu(self.model, self.optimizer, non_blocking, backload_optimizer, backload_model)

        def init_model(self, model_path, num_training_steps: int = None):
            """Initialize nmoe model with FSDP strategy."""
            self.init_nmoe_model(model_path, num_training_steps)

        def optim_step(self):
            """Perform optimizer step with expert cache refresh for quantized models.

            For FP8/NVFP4 models, the expert weight caches need to be refreshed after
            the optimizer updates the master weights.
            """
            grad_norm = super().optim_step()

            # Refresh expert caches for quantized models
            if getattr(self, '_uses_quantized_experts', False):
                self._refresh_expert_caches()

            return grad_norm

        async def broadcast_to_inference_engines(self, inference_engine_client):
            """Broadcast weights to inference engines.

            Supports both full weight sync and LoRA adapter sync.
            """
            from skyrl_train.utils import str_to_torch_dtype

            use_prefix_cache = self.cfg.generator.enable_prefix_caching
            generator_dtype = str_to_torch_dtype(self.cfg.generator.model_dtype)
            cache_reset_task = None
            if use_prefix_cache and torch.distributed.get_rank() == 0:
                cache_reset_task = inference_engine_client.reset_prefix_cache()

            torch.cuda.empty_cache()

            if self._is_lora:
                # LoRA mode: sync only LoRA adapter weights
                await self._save_and_sync_lora_adapters(inference_engine_client)
            else:
                # Full weight sync
                await self._weight_transfer_sender.send_chunks(
                    self.weight_extractor.extract_weights(generator_dtype)
                )

            if cache_reset_task is not None:
                await cache_reset_task
            torch.cuda.empty_cache()
            torch.distributed.barrier()

        async def _save_and_sync_lora_adapters(self, inference_engine_client):
            """Save LoRA adapters and sync to inference engines.

            This method extracts LoRA weights from the FSDP-wrapped model,
            saves them in PEFT format, and updates the inference engine.
            """
            import os
            import io
            from safetensors.torch import save_file
            from skyrl_train.model_wrapper_nmoe import collect_nmoe_lora_params
            from skyrl_train.utils import str_to_torch_dtype

            # Get LoRA sync path from config
            lora_sync_path = self.cfg.trainer.policy.model.lora.lora_sync_path

            # Collect LoRA parameters from the model
            # Handle FSDP-wrapped models
            if hasattr(self.model, 'model'):
                base_model = self.model.model
            else:
                base_model = self.model

            lora_params = collect_nmoe_lora_params(base_model)

            if not lora_params:
                logger.warning("[NMoE LoRA] No LoRA parameters found, falling back to full weight sync")
                await self._weight_transfer_sender.send_chunks(
                    self.weight_extractor.extract_weights(
                        str_to_torch_dtype(self.cfg.generator.model_dtype)
                    )
                )
                return

            # Save on rank 0 only
            if torch.distributed.get_rank() == 0:
                os.makedirs(lora_sync_path, exist_ok=True)

                # Save adapter weights
                save_file(lora_params, os.path.join(lora_sync_path, "adapter_model.safetensors"))

                # Save adapter config
                adapter_config = {
                    "r": self.cfg.trainer.policy.model.lora.rank,
                    "lora_alpha": self.cfg.trainer.policy.model.lora.alpha,
                    "lora_dropout": self.cfg.trainer.policy.model.lora.dropout,
                    "target_modules": self.cfg.trainer.policy.model.lora.target_modules,
                    "bias": "none",
                    "task_type": "CAUSAL_LM",
                }
                import json
                with io.open(os.path.join(lora_sync_path, "adapter_config.json"), "w", encoding="utf-8") as f:
                    json.dump(adapter_config, f, indent=2)

                logger.info(f"[NMoE LoRA] Saved {len(lora_params)} adapter parameters to {lora_sync_path}")

                # Update inference engine with LoRA weights
                try:
                    from sglang.srt.managers.schedule_batch import LoraLoadRequest
                    lora_request = LoraLoadRequest(lora_path=lora_sync_path)
                    await inference_engine_client.update_named_weights(lora_request)
                    logger.info("[NMoE LoRA] Updated inference engine with LoRA adapters")
                except ImportError:
                    logger.warning(
                        "[NMoE LoRA] SGLang LoRA support not available, "
                        "inference engine may not have updated weights"
                    )

            # Ensure all ranks wait for sync to complete
            torch.distributed.barrier()

        def _set_pad_token_id(self, pad_token_id):
            """Set pad token ID on the model config."""
            if hasattr(self.model, 'model') and hasattr(self.model.model, 'config'):
                self.model.model.config.pad_token_id = pad_token_id

        def forward(self, data: TrainingInputBatch) -> TrainingOutputBatch:
            """Run forward pass on data in inference mode."""
            output = super().forward(data)
            if self._world_size > 1 and fsdp_version(self.model.model) == 1:
                self.model.model._handle.reshard(True)
            return output

    class NMoEFSDPCriticWorkerBase(NMoEFSDPCriticWorkerMixin, CriticWorkerBase):
        """FSDP Critic Worker for NMoE models.

        Used for PPO training with value baseline estimation.
        """

        def offload_to_cpu(self, pin_memory=True, non_blocking=True, offload_optimizer=True, offload_model=True):
            self._set_numa_affinity(torch.distributed.get_rank() % torch.cuda.device_count())
            self.strategy.offload_to_cpu(
                self.model, self.optimizer, pin_memory, non_blocking, offload_optimizer, offload_model
            )

        def backload_to_gpu(self, non_blocking=True, backload_optimizer=True, backload_model=True):
            self.strategy.backload_to_gpu(self.model, self.optimizer, non_blocking, backload_optimizer, backload_model)

        def init_model(self, model_path, num_training_steps: int = None):
            """Initialize nmoe critic model with FSDP strategy."""
            self.init_nmoe_critic_model(model_path, num_training_steps)

        def optim_step(self):
            """Perform optimizer step with expert cache refresh for quantized models."""
            grad_norm = super().optim_step()

            # Refresh expert caches for quantized models
            if getattr(self, '_uses_quantized_experts', False):
                self._refresh_expert_caches()

            return grad_norm

        def forward(self, data: TrainingInputBatch) -> TrainingOutputBatch:
            """Run forward pass on data in inference mode."""
            output = super().forward(data)
            if self._world_size > 1 and fsdp_version(self.model.model) == 1:
                self.model.model._handle.reshard(True)
            return output

    class NMoEFSDPRefWorkerBase(NMoEFSDPRefWorkerMixin, RefWorkerBase):
        """FSDP Reference Worker for NMoE models."""

        def offload_to_cpu(self, pin_memory=True, non_blocking=True, **kwargs):
            self._set_numa_affinity(torch.distributed.get_rank() % torch.cuda.device_count())
            self.strategy.offload_to_cpu(self.model, None, pin_memory, non_blocking)

        def backload_to_gpu(self, non_blocking=True, **kwargs):
            self.strategy.backload_to_gpu(self.model, None, non_blocking)

        def init_model(self, model_path):
            """Initialize frozen nmoe reference model."""
            self.init_nmoe_ref_model(model_path)

        def forward(self, data: TrainingInputBatch) -> TrainingOutputBatch:
            """Run forward pass on data in inference mode."""
            output = super().forward(data)
            if self._world_size > 1 and fsdp_version(self.model.model) == 1:
                self.model.model._handle.reshard(True)
            return output

    NMoEPolicyWorker = ray.remote(num_gpus=1)(NMoEFSDPPolicyWorkerBase)
    NMoECriticWorker = ray.remote(num_gpus=1)(NMoEFSDPCriticWorkerBase)
    NMoERefWorker = ray.remote(num_gpus=1)(NMoEFSDPRefWorkerBase)

    return NMoEPolicyWorker, NMoECriticWorker, NMoERefWorker
