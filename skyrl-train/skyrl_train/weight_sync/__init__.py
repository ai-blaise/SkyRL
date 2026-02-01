"""Weight synchronization abstractions for distributed RL training."""

from typing import Type, TYPE_CHECKING

if TYPE_CHECKING:
    from omegaconf import DictConfig

from .base import WeightChunk, WeightUpdateRequest, LoraLoadRequest
from .weight_extractor import WeightExtractor
from .weight_loader import WeightLoader
from .transfer_strategy import (
    WeightTransferStrategy,
    WeightTransferSender,
    WeightTransferReceiver,
    WeightSyncInitInfo,
)
from .broadcast_strategy import (
    BroadcastInitInfo,
    BroadcastTransferStrategy,
    BroadcastWeightTransferSender,
    BroadcastWeightTransferReceiver,
    BroadcastWeightUpdateRequest,
)
from .cuda_ipc_strategy import (
    CudaIpcInitInfo,
    CudaIpcTransferStrategy,
    CudaIpcWeightTransferSender,
    CudaIpcWeightTransferReceiver,
    CudaIpcWeightUpdateRequest,
)
from .checkpoint_engine_strategy import (
    CheckpointEngineInitInfo,
    CheckpointEngineTransferStrategy,
    CheckpointEngineWeightTransferSender,
    CheckpointEngineWeightTransferReceiver,
    CheckpointEngineWeightUpdateRequest,
    CHECKPOINT_ENGINE_AVAILABLE,
)


def get_transfer_strategy_cls(cfg: "DictConfig") -> Type[WeightTransferStrategy]:
    """Get the appropriate transfer strategy class based on config.

    Strategy selection priority:
    1. checkpoint-engine: If weight_sync_backend is "checkpoint_engine" and available
    2. CUDA IPC: If weight_sync_backend is "nccl" and colocate_all is True
    3. Auto: Automatically select best strategy based on placement and TP size
    4. Broadcast: Default fallback for cross-node scenarios

    Performance notes for TP > 1:
    - CUDA IPC is significantly faster than broadcast for TP > 1 because it
      transfers weights directly via GPU memory handles without process group ops.
    - Broadcast with TP > 1 requires coordinating multiple process group operations
      across all TP workers, which adds latency.
    - Recommendation: Use colocate_all=True with nccl backend for best performance
      when training and inference are on the same node.

    Args:
        cfg: Configuration object containing generator and trainer settings.

    Returns:
        The strategy class to use for weight transfer.
    """
    from loguru import logger

    backend = cfg.generator.weight_sync_backend
    colocate_all = cfg.trainer.placement.colocate_all
    tp_size = cfg.generator.inference_engine_tensor_parallel_size

    # Check for checkpoint-engine
    if backend == "checkpoint_engine":
        if CHECKPOINT_ENGINE_AVAILABLE:
            return CheckpointEngineTransferStrategy
        else:
            logger.warning(
                "checkpoint_engine backend requested but checkpoint-engine is not installed. "
                "Falling back to broadcast. Install with: pip install checkpoint-engine"
            )
            return BroadcastTransferStrategy

    # Auto mode: automatically select best strategy
    if backend == "auto":
        if colocate_all:
            logger.info(
                "Auto weight sync: selecting CUDA IPC (colocate_all=True, same-node optimization)"
            )
            return CudaIpcTransferStrategy
        else:
            if tp_size > 1:
                logger.warning(
                    f"Auto weight sync: selecting Broadcast with TP={tp_size}. "
                    f"For better performance, consider setting colocate_all=True to enable CUDA IPC. "
                    f"Broadcast with TP > 1 requires multiple process group operations per weight sync."
                )
            else:
                logger.info(
                    "Auto weight sync: selecting Broadcast (cross-node or non-colocated)"
                )
            return BroadcastTransferStrategy

    # Check for CUDA IPC (same-node optimization)
    if backend == "nccl" and colocate_all:
        return CudaIpcTransferStrategy

    # Broadcast path - warn about TP > 1 performance
    if tp_size > 1 and not colocate_all:
        logger.warning(
            f"Using Broadcast weight sync with TP={tp_size} and colocate_all=False. "
            f"This configuration may have suboptimal performance. "
            f"For faster weight sync with TP > 1, consider: "
            f"(1) Setting colocate_all=True to enable CUDA IPC, or "
            f"(2) Using checkpoint_engine backend for disk-based sync."
        )
    elif tp_size > 1 and colocate_all and backend != "nccl":
        logger.info(
            f"Using Broadcast weight sync with TP={tp_size}. "
            f"Note: CUDA IPC would be faster but requires weight_sync_backend='nccl'. "
            f"Current backend: '{backend}'."
        )

    # Default to broadcast
    return BroadcastTransferStrategy


__all__ = [
    "WeightChunk",
    "WeightExtractor",
    "WeightLoader",
    "WeightUpdateRequest",
    "LoraLoadRequest",
    "BroadcastWeightUpdateRequest",
    "CudaIpcWeightUpdateRequest",
    "CheckpointEngineWeightUpdateRequest",
    "WeightTransferStrategy",
    "WeightTransferSender",
    "WeightTransferReceiver",
    "WeightSyncInitInfo",
    "BroadcastInitInfo",
    "CudaIpcInitInfo",
    "CheckpointEngineInitInfo",
    "BroadcastTransferStrategy",
    "BroadcastWeightTransferSender",
    "BroadcastWeightTransferReceiver",
    "CudaIpcTransferStrategy",
    "CudaIpcWeightTransferSender",
    "CudaIpcWeightTransferReceiver",
    "CheckpointEngineTransferStrategy",
    "CheckpointEngineWeightTransferSender",
    "CheckpointEngineWeightTransferReceiver",
    "CHECKPOINT_ENGINE_AVAILABLE",
    "get_transfer_strategy_cls",
]
