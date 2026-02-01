"""NMoE Weight Extractor for EP-aware weight extraction.

This module provides the NMoEWeightExtractor class for extracting weights
from nmoe models in a way that handles Expert Parallelism (EP) correctly.

Key features:
- Separates expert and dense parameters using nmoe's param_sets()
- Handles EP-sharded expert weights with proper gathering
- Integrates with SkyRL's weight sync infrastructure
- Supports weight sync to SGLang inference engine
"""

from typing import Dict, Iterator, List, Optional, Callable, Any, Tuple
import torch
import torch.distributed as dist
import torch.nn as nn

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

# Import from specific modules to avoid pulling in ray/other heavy deps
from skyrl_train.weight_sync.base import WeightChunk
from skyrl_train.weight_sync.weight_extractor import WeightExtractor
from skyrl_train.weight_sync.weight_extractor_utils import yield_module_grouped_chunks


class NMoEWeightExtractor(WeightExtractor):
    """Extracts weights from nmoe models with EP-awareness.

    This extractor handles the unique structure of nmoe MoE models where:
    - Expert weights (W1, W3, W2) may be sharded across EP ranks
    - Dense weights (attention, embeddings, etc.) are not EP-sharded
    - Router weights need special handling for load balancing

    The extractor uses nmoe's built-in param_sets() method to correctly
    identify and separate expert from dense parameters.

    Args:
        model: The nmoe model (wrapped in NMoEModelWrapper or raw Transformer)
        ep_group: Expert parallel process group (None for single GPU)
        ep_rank: This rank's position in the EP group (0 for single GPU)
        ep_world_size: Total number of EP ranks (1 for single GPU)
        batch_size_threshold_gb: Batch weight chunks up to this size for transfer

    Example:
        >>> from skyrl_train.model_wrapper_nmoe import NMoEModelWrapper
        >>> wrapper = NMoEModelWrapper(model)
        >>> extractor = NMoEWeightExtractor(wrapper)
        >>> for chunk in extractor.extract_weights(torch.bfloat16):
        ...     # Transfer chunk to inference engine
        ...     pass
    """

    def __init__(
        self,
        model: nn.Module,
        ep_group: Optional[dist.ProcessGroup] = None,
        ep_rank: int = 0,
        ep_world_size: int = 1,
        batch_size_threshold_gb: float = 0.5,
    ):
        self.model = model
        self.ep_group = ep_group
        self.ep_rank = ep_rank
        self.ep_world_size = ep_world_size
        self.batch_size_threshold_gb = batch_size_threshold_gb

        # Get the underlying nmoe model if wrapped
        self._nmoe_model = self._unwrap_model(model)

        # Cache parameter categorization
        self._expert_param_names: Optional[set] = None
        self._categorize_parameters()

    def _unwrap_model(self, model: nn.Module) -> nn.Module:
        """Unwrap model to get the underlying nmoe Transformer."""
        # Handle NMoEModelWrapper
        if hasattr(model, 'model'):
            return model.model
        # Handle HFModelWrapper pattern
        if hasattr(model, '_nmoe_model'):
            return model._nmoe_model
        return model

    def _categorize_parameters(self) -> None:
        """Categorize parameters into expert and dense sets."""
        self._expert_param_names = set()

        # Use nmoe's param_sets() if available
        if hasattr(self._nmoe_model, 'param_sets'):
            expert_params, _ = self._nmoe_model.param_sets()
            expert_ids = {id(p) for p in expert_params}

            for name, param in self._nmoe_model.named_parameters():
                if id(param) in expert_ids:
                    self._expert_param_names.add(name)
        else:
            # Fallback: identify expert params by name patterns
            for name, _ in self._nmoe_model.named_parameters():
                # nmoe expert weight patterns
                if any(pattern in name for pattern in ['W1', 'W2', 'W3']):
                    if 'ffn' in name or 'moe' in name.lower():
                        self._expert_param_names.add(name)

        logger.info(f"[NMoEWeightExtractor] Found {len(self._expert_param_names)} expert parameters")

    def _is_expert_param(self, name: str) -> bool:
        """Check if a parameter is an expert parameter."""
        return name in self._expert_param_names

    def _gather_expert_tensor(self, tensor: torch.Tensor, name: str) -> torch.Tensor:
        """Gather expert tensor from EP ranks.

        For EP-sharded expert weights, we need to gather the full weights
        from all EP ranks. Each rank holds n_local = n_experts / ep_world_size.

        Args:
            tensor: Local expert tensor [n_local, ...]
            name: Parameter name (for logging)

        Returns:
            Full expert tensor [n_experts, ...] on rank 0,
            or the original tensor if EP is not used
        """
        if self.ep_world_size == 1 or self.ep_group is None:
            return tensor

        # Expert weights have shape [n_local, dim1, dim2]
        # Gather to get [n_experts, dim1, dim2]
        if len(tensor.shape) < 2:
            # Not an expert weight tensor
            return tensor

        n_local = tensor.shape[0]
        n_experts = n_local * self.ep_world_size

        # Allocate gather buffer on rank 0
        if self.ep_rank == 0:
            full_shape = (n_experts,) + tensor.shape[1:]
            gathered = torch.empty(full_shape, dtype=tensor.dtype, device=tensor.device)
        else:
            gathered = None

        # Gather all expert weights to rank 0
        gather_list = None
        if self.ep_rank == 0:
            gather_list = [torch.empty_like(tensor) for _ in range(self.ep_world_size)]

        dist.gather(tensor, gather_list, dst=0, group=self.ep_group)

        if self.ep_rank == 0:
            # Concatenate gathered weights
            gathered = torch.cat(gather_list, dim=0)
            return gathered
        else:
            # Non-rank-0 returns empty tensor (will be filtered out)
            return tensor

    def _gather_tensor_fn(self, param: torch.Tensor, name: str) -> torch.Tensor:
        """Gather tensor from distributed ranks.

        This handles both EP gathering for expert params and
        any other distributed gathering needed.
        """
        if self._is_expert_param(name):
            return self._gather_expert_tensor(param, name)
        else:
            # Dense parameters are not EP-sharded
            return param

    def extract_weights(self, dtype: torch.dtype) -> Iterator[WeightChunk]:
        """Extract weights from the model as WeightChunk objects.

        Handles EP-aware extraction:
        - Expert weights are gathered from EP ranks
        - Dense weights are extracted directly
        - Weights are converted to the target dtype

        Args:
            dtype: Target dtype for inference (e.g., torch.bfloat16)

        Yields:
            WeightChunk objects containing model parameters
        """
        # Get all named parameters
        params = dict(self._nmoe_model.named_parameters())

        # For EP, only rank 0 yields the full weights
        # Other ranks participate in gather but don't yield
        is_primary = self.ep_rank == 0 or self.ep_world_size == 1

        def gather_fn(param_tuple: Tuple[str, torch.Tensor]) -> torch.Tensor:
            """Gather function for yield_module_grouped_chunks."""
            name, param = param_tuple
            return self._gather_tensor_fn(param.data, name)

        def shape_fn(param_name: str, param: Any, tensor: torch.Tensor) -> List[int]:
            """Get shape for the parameter."""
            return list(tensor.shape)

        if is_primary:
            # Wrap params with names for the gather function
            named_params = {name: (name, param) for name, param in params.items()}

            yield from yield_module_grouped_chunks(
                params=named_params,
                dtype=dtype,
                gather_tensor_fn=gather_fn,
                get_shape_fn=shape_fn,
                batch_size_threshold_gb=self.batch_size_threshold_gb,
            )
        else:
            # Non-primary ranks participate in gather operations but don't yield
            for name, param in params.items():
                if self._is_expert_param(name):
                    # Participate in gather
                    self._gather_expert_tensor(param.data, name)

    def extract_expert_weights_only(self, dtype: torch.dtype) -> Iterator[WeightChunk]:
        """Extract only expert weights.

        This is useful for partial weight sync when only MoE experts
        have been updated (e.g., during LoRA training of experts only).

        Args:
            dtype: Target dtype for inference

        Yields:
            WeightChunk objects containing only expert parameters
        """
        expert_params = {}
        for name, param in self._nmoe_model.named_parameters():
            if self._is_expert_param(name):
                expert_params[name] = (name, param)

        is_primary = self.ep_rank == 0 or self.ep_world_size == 1

        def gather_fn(param_tuple: Tuple[str, torch.Tensor]) -> torch.Tensor:
            name, param = param_tuple
            return self._gather_tensor_fn(param.data, name)

        def shape_fn(param_name: str, param: Any, tensor: torch.Tensor) -> List[int]:
            return list(tensor.shape)

        if is_primary:
            yield from yield_module_grouped_chunks(
                params=expert_params,
                dtype=dtype,
                gather_tensor_fn=gather_fn,
                get_shape_fn=shape_fn,
                batch_size_threshold_gb=self.batch_size_threshold_gb,
            )
        else:
            for name, param in expert_params.values():
                self._gather_expert_tensor(param.data, name)

    def extract_dense_weights_only(self, dtype: torch.dtype) -> Iterator[WeightChunk]:
        """Extract only dense (non-expert) weights.

        This is useful for partial weight sync when only dense layers
        have been updated.

        Args:
            dtype: Target dtype for inference

        Yields:
            WeightChunk objects containing only dense parameters
        """
        dense_params = {}
        for name, param in self._nmoe_model.named_parameters():
            if not self._is_expert_param(name):
                dense_params[name] = param

        def gather_fn(param: torch.Tensor) -> torch.Tensor:
            # Dense params don't need EP gathering
            return param.data

        def shape_fn(param_name: str, param: Any, tensor: torch.Tensor) -> List[int]:
            return list(tensor.shape)

        yield from yield_module_grouped_chunks(
            params=dense_params,
            dtype=dtype,
            gather_tensor_fn=gather_fn,
            get_shape_fn=shape_fn,
            batch_size_threshold_gb=self.batch_size_threshold_gb,
        )

    def get_weight_stats(self) -> Dict[str, Any]:
        """Get statistics about the model's weights.

        Returns:
            Dict containing:
                - n_expert_params: Number of expert parameters
                - n_dense_params: Number of dense parameters
                - expert_numel: Total elements in expert params
                - dense_numel: Total elements in dense params
                - ep_world_size: EP world size
        """
        expert_numel = 0
        dense_numel = 0
        n_expert = 0
        n_dense = 0

        for name, param in self._nmoe_model.named_parameters():
            numel = param.numel()
            if self._is_expert_param(name):
                expert_numel += numel
                n_expert += 1
            else:
                dense_numel += numel
                n_dense += 1

        # Adjust expert numel for EP sharding
        if self.ep_world_size > 1:
            expert_numel *= self.ep_world_size

        return {
            'n_expert_params': n_expert,
            'n_dense_params': n_dense,
            'expert_numel': expert_numel,
            'dense_numel': dense_numel,
            'total_numel': expert_numel + dense_numel,
            'ep_world_size': self.ep_world_size,
            'expert_size_gb': expert_numel * 2 / (1024**3),  # Assuming bf16
            'dense_size_gb': dense_numel * 2 / (1024**3),
        }


def create_nmoe_weight_extractor(
    model: nn.Module,
    ep_group: Optional[dist.ProcessGroup] = None,
    batch_size_threshold_gb: float = 0.5,
) -> NMoEWeightExtractor:
    """Factory function to create NMoEWeightExtractor.

    This automatically detects EP settings from the process group.

    Args:
        model: The nmoe model
        ep_group: Expert parallel process group (auto-detected if None)
        batch_size_threshold_gb: Batch weight chunks up to this size

    Returns:
        Configured NMoEWeightExtractor instance
    """
    # Auto-detect EP settings
    if ep_group is not None:
        ep_rank = dist.get_rank(ep_group)
        ep_world_size = dist.get_world_size(ep_group)
    elif dist.is_initialized():
        # Default to world group if no EP group specified
        # In practice, EP group should be explicitly provided
        ep_rank = dist.get_rank()
        ep_world_size = dist.get_world_size()
        logger.warning(
            "[NMoEWeightExtractor] No EP group specified, using world group. "
            "This may not be correct for multi-dimensional parallelism."
        )
    else:
        ep_rank = 0
        ep_world_size = 1

    return NMoEWeightExtractor(
        model=model,
        ep_group=ep_group,
        ep_rank=ep_rank,
        ep_world_size=ep_world_size,
        batch_size_threshold_gb=batch_size_threshold_gb,
    )
