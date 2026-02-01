"""
NMoE Megatron Worker for SkyRL.

This module provides Ray-based workers for training NMoE models using the
Megatron parallelism infrastructure with RDEP expert dispatch.
"""

from __future__ import annotations

import os
from datetime import timedelta
from typing import Any, Dict, List, Optional, Tuple

import ray
import torch
import torch.nn as nn
from torch import distributed as dist
from transformers import AutoTokenizer

from skyrl_train.distributed.megatron.nmoe_megatron_strategy import (
    NMoEMegatronConfig,
    NMoEMegatronStrategy,
    create_nmoe_megatron_strategy,
)
from skyrl_train.distributed.dispatch import MeshRank
from skyrl_train.training_batch import TrainingOutputBatch
from skyrl_train.workers.worker import PolicyWorkerBase, RefWorkerBase, CriticWorkerBase
from skyrl_train.workers.worker_utils import BatchIterator, reduce_metrics
from skyrl_train.utils.utils import str_to_torch_dtype
from skyrl_train.utils.constants import SKYRL_WORKER_NCCL_TIMEOUT_IN_S
from skyrl_train.weight_sync import WeightExtractor, WeightChunk

try:
    import megatron.core.parallel_state as mpu
    from megatron.core.optimizer import DistributedOptimizer
    from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler
    MEGATRON_AVAILABLE = True
except ImportError:
    MEGATRON_AVAILABLE = False
    mpu = None

try:
    from nmoe.config import Config as NMoEModelConfig
    from nmoe.model import Transformer as NMoETransformer
    from nmoe.rdep import Rdep
    NMOE_AVAILABLE = True
except ImportError:
    NMOE_AVAILABLE = False


class NMoEMegatronWeightExtractor(WeightExtractor):
    """Extracts weights from NMoE models in Megatron parallel setting.

    Handles both shared weights and expert weights, accounting for expert
    parallelism when extracting weights for SGLang inference engines.
    """

    def __init__(
        self,
        model: nn.Module,
        enable_bucketing: bool = False,
        bucket_size_threshold_GB: float = 1.0,
        training_dtype: torch.dtype = torch.bfloat16,
    ):
        self.model = model
        self.enable_bucketing = enable_bucketing
        self.bucket_size_threshold_GB = bucket_size_threshold_GB
        self.training_dtype = training_dtype

        # Unwrap model
        self._unwrapped_model = model
        while hasattr(self._unwrapped_model, "module"):
            self._unwrapped_model = self._unwrapped_model.module

        # Categorize parameters
        self._shared_params = {}
        self._expert_params = {}
        for name, param in self._unwrapped_model.named_parameters():
            if self._is_expert_param(name):
                self._expert_params[name] = param
            else:
                self._shared_params[name] = param

    def _is_expert_param(self, name: str) -> bool:
        """Check if parameter is an expert parameter."""
        expert_patterns = [".experts.", ".ffn.W1", ".ffn.W2", ".ffn.W3", "moe_layer"]
        return any(pattern in name for pattern in expert_patterns)

    def iter_chunks(self) -> List[WeightChunk]:
        """Iterate over weight chunks for sync."""
        chunks = []

        # Add shared parameters as a single chunk
        if self._shared_params:
            chunks.append(WeightChunk(
                name="shared_weights",
                params=self._shared_params,
            ))

        # Add expert parameters grouped by layer
        expert_by_layer = {}
        for name, param in self._expert_params.items():
            # Extract layer index from name (e.g., blocks.0.ffn.W1 -> 0)
            parts = name.split(".")
            for i, p in enumerate(parts):
                if p.isdigit():
                    layer_idx = int(p)
                    if layer_idx not in expert_by_layer:
                        expert_by_layer[layer_idx] = {}
                    expert_by_layer[layer_idx][name] = param
                    break

        for layer_idx, params in sorted(expert_by_layer.items()):
            chunks.append(WeightChunk(
                name=f"experts_layer_{layer_idx}",
                params=params,
            ))

        return chunks


class NMoEMegatronModelWrapper(nn.Module):
    """Wrapper for NMoE model with Megatron integration.

    This wrapper:
    1. Manages the NMoE Transformer model
    2. Integrates with Megatron parallel state
    3. Provides forward/generate interfaces compatible with SkyRL
    """

    def __init__(
        self,
        model_config: NMoEModelConfig,
        strategy: NMoEMegatronStrategy,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()

        self.model_config = model_config
        self.strategy = strategy
        self.gradient_checkpointing = gradient_checkpointing

        # Create the NMoE Transformer
        self.actor_module = NMoETransformer(model_config)

        # Set up RDEP instances for each MoE layer
        self._setup_rdep_instances()

    def _setup_rdep_instances(self) -> None:
        """Set up RDEP instances for MoE layers."""
        # RDEP instances are managed by the strategy
        for i, block in enumerate(self.actor_module.blocks):
            if hasattr(block, "moe") and block.moe is not None:
                # Register this MoE layer with the strategy
                rdep = self.strategy.get_rdep_instance(
                    name=f"moe_layer_{i}",
                    dim=self.model_config.dim,
                    n_local_experts=self.model_config.n_routed_experts // self.strategy._ep_world_size,
                    topk=self.model_config.n_activated_experts,
                )
                # Attach RDEP to the MoE layer
                if hasattr(block.moe, "set_rdep"):
                    block.moe.set_rdep(rdep)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with NMoE model."""
        # Get logits from the model
        logits = self.actor_module(input_ids)

        output = {"logits": logits}

        # Compute loss if labels provided
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss(reduction="none")
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
            output["loss"] = loss.view(shift_labels.shape)

        return output

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        do_sample: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        """Generate tokens autoregressively."""
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        generated = input_ids

        for _ in range(max_new_tokens):
            # Get logits for next token
            logits = self.actor_module(generated)
            next_token_logits = logits[:, -1, :]

            # Apply temperature
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature

            # Apply top-k filtering
            if top_k is not None and top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float("-inf")

            # Apply top-p (nucleus) filtering
            if top_p is not None and top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = float("-inf")

            # Sample or argmax
            if do_sample:
                probs = torch.softmax(next_token_logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1)
            else:
                next_tokens = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            # Append to generated sequence
            generated = torch.cat([generated, next_tokens], dim=-1)

        return generated


class NMoEMegatronWorkerMixin:
    """Mixin providing common functionality for NMoE Megatron workers."""

    def init_nmoe_megatron_model(
        self,
        model_path: str,
        nmoe_config: NMoEMegatronConfig,
        seed: int = 42,
        is_lora: bool = False,
        gradient_checkpointing: bool = False,
    ) -> Tuple[NMoEMegatronModelWrapper, NMoEMegatronStrategy]:
        """Initialize NMoE model with Megatron strategy.

        Args:
            model_path: Path to model checkpoint
            nmoe_config: NMoE Megatron configuration
            seed: Random seed
            is_lora: Whether to use LoRA
            gradient_checkpointing: Enable gradient checkpointing

        Returns:
            Tuple of (model_wrapper, strategy)
        """
        # Create strategy
        strategy = NMoEMegatronStrategy(
            config=nmoe_config,
            seed=seed,
            is_lora=is_lora,
        )

        # Initialize distributed
        strategy.setup_distributed(
            timeout=timedelta(seconds=SKYRL_WORKER_NCCL_TIMEOUT_IN_S)
        )

        # Load model configuration
        model_config = self._load_nmoe_config(model_path)

        # Create model wrapper
        model_wrapper = NMoEMegatronModelWrapper(
            model_config=model_config,
            strategy=strategy,
            gradient_checkpointing=gradient_checkpointing,
        )

        # Load checkpoint if exists
        if os.path.exists(model_path):
            strategy.load_checkpoint(
                model=model_wrapper,
                ckpt_dir=model_path,
                load_module_strict=False,
            )

        return model_wrapper, strategy

    def _load_nmoe_config(self, model_path: str) -> NMoEModelConfig:
        """Load NMoE model configuration from path."""
        import json

        config_path = os.path.join(model_path, "nmoe_config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config_dict = json.load(f)
            model_config_dict = config_dict.get("model_config", config_dict)
            return NMoEModelConfig(**model_config_dict)

        # Fallback to rd.pt
        rd_path = os.path.join(model_path, "rd.pt")
        if os.path.exists(rd_path):
            checkpoint = torch.load(rd_path, map_location="cpu")
            config_dict = checkpoint.get("config", {})
            return NMoEModelConfig(**config_dict)

        raise FileNotFoundError(f"No NMoE config found at {model_path}")

    def create_weight_extractor(
        self,
        model: nn.Module,
        enable_bucketing: bool = False,
    ) -> NMoEMegatronWeightExtractor:
        """Create weight extractor for inference sync."""
        return NMoEMegatronWeightExtractor(
            model=model,
            enable_bucketing=enable_bucketing,
        )


@ray.remote
class NMoEMegatronPolicyWorker(PolicyWorkerBase, NMoEMegatronWorkerMixin):
    """Ray actor for NMoE policy training with Megatron parallelism."""

    def __init__(
        self,
        worker_args: Dict[str, Any],
        mesh_rank: MeshRank,
    ):
        super().__init__(worker_args, mesh_rank)

        self.nmoe_config = NMoEMegatronConfig(
            tensor_model_parallel_size=worker_args.get("tp_size", 1),
            pipeline_model_parallel_size=worker_args.get("pp_size", 1),
            context_parallel_size=worker_args.get("cp_size", 1),
            expert_model_parallel_size=worker_args.get("ep_size", 1),
            expert_tensor_parallel_size=worker_args.get("etp_size", 1),
            rdep_profile=worker_args.get("rdep_profile", "bf16"),
            rdep_capacity=worker_args.get("rdep_capacity", 65536),
        )

    def init_model(self) -> None:
        """Initialize the NMoE policy model."""
        self.model, self.strategy = self.init_nmoe_megatron_model(
            model_path=self.worker_args["model_path"],
            nmoe_config=self.nmoe_config,
            seed=self.worker_args.get("seed", 42),
            is_lora=self.worker_args.get("is_lora", False),
            gradient_checkpointing=self.worker_args.get("gradient_checkpointing", False),
        )

        # Move to GPU
        self.model = self.model.cuda()
        self.model.train()

        # Set up weight extractor for inference sync
        self.weight_extractor = self.create_weight_extractor(
            model=self.model,
            enable_bucketing=self.worker_args.get("enable_bucketing", False),
        )

    def compute_log_probs(
        self,
        batch: TrainingOutputBatch,
        return_entropy: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Compute log probabilities for the batch."""
        input_ids = batch.sequences.cuda()
        attention_mask = batch.attention_mask.cuda()

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        logits = outputs["logits"]

        # Compute log probs
        log_probs = torch.log_softmax(logits, dim=-1)
        action_log_probs = torch.gather(
            log_probs[:, :-1, :],
            dim=-1,
            index=input_ids[:, 1:].unsqueeze(-1),
        ).squeeze(-1)

        result = {"action_log_probs": action_log_probs}

        if return_entropy:
            probs = torch.softmax(logits[:, :-1, :], dim=-1)
            entropy = -torch.sum(probs * log_probs[:, :-1, :], dim=-1)
            result["entropy"] = entropy

        return result

    def backward(self, loss: torch.Tensor) -> None:
        """Perform backward pass."""
        self.strategy.backward(loss, self.model, self.optimizer)

    def optimizer_step(self) -> Optional[torch.Tensor]:
        """Perform optimizer step."""
        return self.strategy.optimizer_step(
            optimizer=self.optimizer,
            model=self.model,
            scheduler=self.scheduler,
        )

    def save_checkpoint(self, ckpt_dir: str) -> None:
        """Save checkpoint."""
        self.strategy.save_checkpoint(
            model=self.model,
            ckpt_dir=ckpt_dir,
            node_local_rank=self.mesh_rank.local_rank,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            tokenizer=self.tokenizer,
        )

    def get_weight_chunks(self) -> List[WeightChunk]:
        """Get weight chunks for inference sync."""
        return self.weight_extractor.iter_chunks()

    def offload_to_cpu(
        self,
        pin_memory: bool = True,
        non_blocking: bool = True,
        offload_optimizer: bool = True,
        offload_model: bool = True,
    ) -> None:
        """Offload model and optimizer to CPU memory."""
        self.strategy.offload_to_cpu(
            model=self.model,
            optimizer=self.optimizer if offload_optimizer else None,
            pin_memory=pin_memory,
            non_blocking=non_blocking,
            offload_optimizer=offload_optimizer,
            offload_model=offload_model,
        )

    def backload_to_gpu(
        self,
        non_blocking: bool = True,
        backload_optimizer: bool = True,
        backload_model: bool = True,
    ) -> None:
        """Reload model and optimizer back to GPU."""
        self.strategy.backload_to_gpu(
            model=self.model,
            optimizer=self.optimizer if backload_optimizer else None,
            non_blocking=non_blocking,
            backload_optimizer=backload_optimizer,
            backload_model=backload_model,
        )


@ray.remote
class NMoEMegatronRefWorker(RefWorkerBase, NMoEMegatronWorkerMixin):
    """Ray actor for NMoE reference model with Megatron parallelism."""

    def __init__(
        self,
        worker_args: Dict[str, Any],
        mesh_rank: MeshRank,
    ):
        super().__init__(worker_args, mesh_rank)

        self.nmoe_config = NMoEMegatronConfig(
            tensor_model_parallel_size=worker_args.get("tp_size", 1),
            pipeline_model_parallel_size=worker_args.get("pp_size", 1),
            context_parallel_size=worker_args.get("cp_size", 1),
            expert_model_parallel_size=worker_args.get("ep_size", 1),
            expert_tensor_parallel_size=worker_args.get("etp_size", 1),
            rdep_profile=worker_args.get("rdep_profile", "bf16"),
            rdep_capacity=worker_args.get("rdep_capacity", 65536),
        )

    def init_model(self) -> None:
        """Initialize the NMoE reference model."""
        self.model, self.strategy = self.init_nmoe_megatron_model(
            model_path=self.worker_args["model_path"],
            nmoe_config=self.nmoe_config,
            seed=self.worker_args.get("seed", 42),
        )

        # Move to GPU and set to eval mode
        self.model = self.model.cuda()
        self.model.eval()

    @torch.no_grad()
    def compute_ref_log_probs(
        self,
        batch: TrainingOutputBatch,
    ) -> Dict[str, torch.Tensor]:
        """Compute reference log probabilities."""
        input_ids = batch.sequences.cuda()
        attention_mask = batch.attention_mask.cuda()

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        logits = outputs["logits"]
        log_probs = torch.log_softmax(logits, dim=-1)
        action_log_probs = torch.gather(
            log_probs[:, :-1, :],
            dim=-1,
            index=input_ids[:, 1:].unsqueeze(-1),
        ).squeeze(-1)

        return {"ref_log_probs": action_log_probs}

    def offload_to_cpu(
        self,
        pin_memory: bool = True,
        non_blocking: bool = True,
        **kwargs,
    ) -> None:
        """Offload reference model to CPU memory."""
        self.strategy.offload_to_cpu(
            model=self.model,
            optimizer=None,
            pin_memory=pin_memory,
            non_blocking=non_blocking,
            offload_optimizer=False,
            offload_model=True,
        )

    def backload_to_gpu(self, non_blocking: bool = True, **kwargs) -> None:
        """Reload reference model back to GPU."""
        self.strategy.backload_to_gpu(
            model=self.model,
            optimizer=None,
            non_blocking=non_blocking,
            backload_optimizer=False,
            backload_model=True,
        )


@ray.remote
class NMoEMegatronCriticWorker(CriticWorkerBase, NMoEMegatronWorkerMixin):
    """Ray actor for NMoE critic model with Megatron parallelism."""

    def __init__(
        self,
        worker_args: Dict[str, Any],
        mesh_rank: MeshRank,
    ):
        super().__init__(worker_args, mesh_rank)

        self.nmoe_config = NMoEMegatronConfig(
            tensor_model_parallel_size=worker_args.get("tp_size", 1),
            pipeline_model_parallel_size=worker_args.get("pp_size", 1),
            context_parallel_size=worker_args.get("cp_size", 1),
            expert_model_parallel_size=worker_args.get("ep_size", 1),
            expert_tensor_parallel_size=worker_args.get("etp_size", 1),
            rdep_profile=worker_args.get("rdep_profile", "bf16"),
            rdep_capacity=worker_args.get("rdep_capacity", 65536),
        )

    def init_model(self) -> None:
        """Initialize the NMoE critic model."""
        self.model, self.strategy = self.init_nmoe_megatron_model(
            model_path=self.worker_args["model_path"],
            nmoe_config=self.nmoe_config,
            seed=self.worker_args.get("seed", 42),
            gradient_checkpointing=self.worker_args.get("gradient_checkpointing", False),
        )

        # Add value head
        self._add_value_head()

        # Move to GPU
        self.model = self.model.cuda()
        self.model.train()

    def _add_value_head(self) -> None:
        """Add value head to the model for PPO critic."""
        hidden_size = self.model.model_config.dim
        self.value_head = nn.Linear(hidden_size, 1, bias=False).cuda()

        # Initialize with small weights
        nn.init.normal_(self.value_head.weight, std=0.02)

    def compute_values(
        self,
        batch: TrainingOutputBatch,
    ) -> Dict[str, torch.Tensor]:
        """Compute value estimates for the batch."""
        input_ids = batch.sequences.cuda()
        attention_mask = batch.attention_mask.cuda()

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            # Get hidden states from the model
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        # Get last hidden state (before LM head)
        # This requires modifying the forward to return hidden states
        hidden_states = outputs.get("hidden_states", outputs["logits"])

        # Compute values
        values = self.value_head(hidden_states).squeeze(-1)

        return {"values": values}

    def backward(self, loss: torch.Tensor) -> None:
        """Perform backward pass."""
        self.strategy.backward(loss, self.model, self.optimizer)

    def optimizer_step(self) -> Optional[torch.Tensor]:
        """Perform optimizer step."""
        return self.strategy.optimizer_step(
            optimizer=self.optimizer,
            model=self.model,
            scheduler=self.scheduler,
        )

    def offload_to_cpu(
        self,
        pin_memory: bool = True,
        non_blocking: bool = True,
        offload_optimizer: bool = True,
        offload_model: bool = True,
    ) -> None:
        """Offload critic model and optimizer to CPU memory."""
        self.strategy.offload_to_cpu(
            model=self.model,
            optimizer=self.optimizer if offload_optimizer else None,
            pin_memory=pin_memory,
            non_blocking=non_blocking,
            offload_optimizer=offload_optimizer,
            offload_model=offload_model,
        )

    def backload_to_gpu(
        self,
        non_blocking: bool = True,
        backload_optimizer: bool = True,
        backload_model: bool = True,
    ) -> None:
        """Reload critic model and optimizer back to GPU."""
        self.strategy.backload_to_gpu(
            model=self.model,
            optimizer=self.optimizer if backload_optimizer else None,
            non_blocking=non_blocking,
            backload_optimizer=backload_optimizer,
            backload_model=backload_model,
        )


def get_nmoe_megatron_workers(
    worker_args: Dict[str, Any],
    mesh_rank: MeshRank,
) -> Tuple[
    NMoEMegatronPolicyWorker,
    NMoEMegatronRefWorker,
    NMoEMegatronCriticWorker,
]:
    """Factory function to create all NMoE Megatron workers.

    Args:
        worker_args: Configuration dictionary for workers
        mesh_rank: Distributed mesh rank information

    Returns:
        Tuple of (policy_worker, ref_worker, critic_worker)
    """
    policy_worker = NMoEMegatronPolicyWorker.remote(worker_args, mesh_rank)
    ref_worker = NMoEMegatronRefWorker.remote(worker_args, mesh_rank)
    critic_worker = NMoEMegatronCriticWorker.remote(worker_args, mesh_rank)

    return policy_worker, ref_worker, critic_worker
