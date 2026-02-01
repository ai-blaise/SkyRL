"""Checkpoint-engine based weight transfer strategy.

This module implements the checkpoint-engine transfer strategy for synchronizing model
weights from training workers to inference engines using MoonshotAI's checkpoint-engine.

Checkpoint-engine provides:
- ParameterServer-based weight distribution
- Broadcast mode for synchronous updates (fastest)
- P2P mode for dynamic instances using RDMA/mooncake
- Pipeline execution with overlapping communication

Requirements:
- checkpoint-engine package: pip install checkpoint-engine
- For RDMA support: mooncake-transfer-engine

References:
- https://github.com/MoonshotAI/checkpoint-engine
- https://github.com/sgl-project/sglang/issues/10464
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable, Iterator, Optional, Tuple, Dict, Any, List

import torch
from loguru import logger

from skyrl_train.weight_sync.base import WeightChunk, WeightUpdateRequest
from skyrl_train.weight_sync.transfer_strategy import (
    WeightSyncInitInfo,
    WeightTransferStrategy,
    WeightTransferSender,
    WeightTransferReceiver,
)

if TYPE_CHECKING:
    from omegaconf import DictConfig
    from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient

# Check if checkpoint-engine is available
try:
    from checkpoint_engine import ParameterServer
    from checkpoint_engine.worker import Worker as CkptWorker
    CHECKPOINT_ENGINE_AVAILABLE = True
except ImportError:
    CHECKPOINT_ENGINE_AVAILABLE = False
    ParameterServer = None
    CkptWorker = None


@dataclass
class CheckpointEngineInitInfo(WeightSyncInitInfo):
    """Initialization info for checkpoint-engine based weight transfer."""

    # ParameterServer configuration
    ps_host: str
    ps_port: int
    num_workers: int
    model_dtype_str: str

    # Transfer mode: "broadcast" or "p2p"
    transfer_mode: str = "broadcast"

    # Worker configuration (for receiver side)
    worker_rank: int = 0

    use_overlapped_weight_sync: bool = False
    """Whether to use overlapped weight sync when supported.

    When enabled, weight transfer happens in the background while inference
    engines continue generation. Only the final weight application requires
    a brief pause.
    """

    @staticmethod
    def strategy_type() -> type:
        """Return the strategy class for this init info type."""
        return CheckpointEngineTransferStrategy

    def for_engine(self, engine_index: int, tp_size: int, pp_size: int) -> "CheckpointEngineInitInfo":
        """Return init_info with worker_rank adjusted for this engine.

        Args:
            engine_index: Index of the engine (0-based).
            tp_size: Tensor parallel size of the engine.
            pp_size: Pipeline parallel size of the engine.

        Returns:
            CheckpointEngineInitInfo with adjusted worker_rank.
        """
        from dataclasses import replace
        cumulative_offset = engine_index * tp_size * pp_size
        return replace(self, worker_rank=cumulative_offset)


@dataclass
class CheckpointEngineWeightUpdateRequest(WeightUpdateRequest):
    """Request for checkpoint-engine based weight transfer.

    Contains metadata for weights that will be transferred via checkpoint-engine.
    """

    # Bucket ID for this update (for tracking)
    bucket_id: Optional[str] = None


class CheckpointEngineWeightTransferSender(WeightTransferSender):
    """Sends weights via checkpoint-engine ParameterServer.

    Uses checkpoint-engine's broadcast or P2P mode to distribute weights
    to inference engines efficiently.
    """

    def __init__(
        self,
        init_info: CheckpointEngineInitInfo,
        inference_client: "InferenceEngineClient",
        parameter_server: Optional[Any] = None,
    ) -> None:
        """Initialize the checkpoint-engine sender.

        Args:
            init_info: CheckpointEngineInitInfo containing config-derived args.
            inference_client: Client for coordinating with inference engines.
            parameter_server: Optional pre-created ParameterServer instance.
        """
        if not CHECKPOINT_ENGINE_AVAILABLE:
            raise ImportError(
                "checkpoint-engine is not installed. "
                "Install with: pip install checkpoint-engine"
            )

        self._init_info = init_info
        self._inference_client = inference_client

        # Create or use provided ParameterServer
        if parameter_server is not None:
            self._ps = parameter_server
        else:
            self._ps = ParameterServer(
                host=init_info.ps_host,
                port=init_info.ps_port,
                num_workers=init_info.num_workers,
            )
            logger.info(
                f"Created checkpoint-engine ParameterServer at "
                f"{init_info.ps_host}:{init_info.ps_port} with {init_info.num_workers} workers"
            )

    async def send_chunks(
        self,
        chunks: Iterable[WeightChunk],
        weight_version: Optional[str] = None,
    ) -> None:
        """Send chunks via checkpoint-engine.

        Args:
            chunks: Iterable of WeightChunk objects to send.
            weight_version: Optional version identifier for tracking.
        """
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0

        for chunk in chunks:
            # Collect weights from chunk
            weights_dict: Dict[str, torch.Tensor] = {}
            for name, tensor, shape in zip(chunk.names, chunk.tensors, chunk.shapes):
                weights_dict[name] = tensor.detach()

            # Only rank 0 coordinates the update
            if rank == 0:
                # Notify inference engines about incoming update
                request = CheckpointEngineWeightUpdateRequest(
                    names=chunk.names,
                    dtypes=[self._init_info.model_dtype_str] * len(chunk.names),
                    shapes=chunk.shapes,
                    weight_version=weight_version,
                    bucket_id=f"bucket_{weight_version or 'default'}",
                )

                # Use ParameterServer to broadcast weights
                if self._init_info.transfer_mode == "broadcast":
                    # Broadcast mode: fastest for synchronous updates
                    self._ps.update_weights(weights_dict, ranks=None)
                else:
                    # P2P mode: for dynamic instances
                    # Get list of worker ranks from inference client
                    worker_ranks = list(range(self._init_info.num_workers))
                    self._ps.update_weights(weights_dict, ranks=worker_ranks)

                # Notify inference engines that update is complete
                # Use overlapped sync if enabled and supported by the backend
                if (
                    self._init_info.use_overlapped_weight_sync
                    and self._inference_client.supports_overlapped_weight_sync()
                ):
                    await self._inference_client.overlapped_weight_sync(request)
                else:
                    await self._inference_client.update_named_weights(request)

            # Synchronize all training ranks
            if torch.distributed.is_initialized():
                torch.distributed.barrier()

    def teardown(self) -> None:
        """Shutdown the ParameterServer."""
        if hasattr(self, "_ps") and self._ps is not None:
            try:
                self._ps.shutdown()
            except Exception as e:
                logger.warning(f"Error shutting down ParameterServer: {e}")


class CheckpointEngineWeightTransferReceiver(WeightTransferReceiver):
    """Receives weights via checkpoint-engine Worker.

    Connects to the ParameterServer and receives weights via broadcast or P2P.
    """

    def __init__(
        self,
        init_info: CheckpointEngineInitInfo,
        worker: Optional[Any] = None,
    ) -> None:
        """Initialize the checkpoint-engine receiver.

        Args:
            init_info: CheckpointEngineInitInfo from the sender.
            worker: Optional pre-created Worker instance.
        """
        if not CHECKPOINT_ENGINE_AVAILABLE:
            raise ImportError(
                "checkpoint-engine is not installed. "
                "Install with: pip install checkpoint-engine"
            )

        self._init_info = init_info

        # Create or use provided Worker
        if worker is not None:
            self._worker = worker
        else:
            self._worker = CkptWorker(
                ps_host=init_info.ps_host,
                ps_port=init_info.ps_port,
                rank=init_info.worker_rank,
            )
            logger.info(
                f"Created checkpoint-engine Worker (rank={init_info.worker_rank}) "
                f"connecting to {init_info.ps_host}:{init_info.ps_port}"
            )

    def receive_weights(
        self, request: CheckpointEngineWeightUpdateRequest
    ) -> Iterator[Tuple[str, torch.Tensor]]:
        """Receive weights via checkpoint-engine.

        Args:
            request: Weight update request with names, dtypes, shapes.

        Yields:
            Tuples of (parameter_name, tensor) for each weight.
        """
        from skyrl_train.utils import str_to_torch_dtype

        # Get weights from the worker
        weights = self._worker.get_weights(request.names)

        for name, dtype_str, shape in zip(request.names, request.dtypes, request.shapes):
            tensor = weights.get(name)
            if tensor is None:
                raise RuntimeError(f"Weight '{name}' not received from checkpoint-engine")

            # Verify dtype matches
            expected_dtype = str_to_torch_dtype(dtype_str)
            if tensor.dtype != expected_dtype:
                tensor = tensor.to(expected_dtype)

            # Verify shape matches
            if list(tensor.shape) != list(shape):
                raise RuntimeError(
                    f"Shape mismatch for '{name}': expected {shape}, got {tensor.shape}"
                )

            yield name, tensor

    def teardown(self) -> None:
        """Disconnect the Worker from ParameterServer."""
        if hasattr(self, "_worker") and self._worker is not None:
            try:
                self._worker.disconnect()
            except Exception as e:
                logger.warning(f"Error disconnecting Worker: {e}")


class CheckpointEngineTransferStrategy(WeightTransferStrategy):
    """Factory for checkpoint-engine based weight transfer.

    This strategy uses MoonshotAI's checkpoint-engine for efficient distributed
    weight synchronization, particularly optimized for large-scale multi-node
    deployments.

    Benefits over broadcast strategy:
    - Pipeline execution with overlapping communication
    - RDMA support via mooncake-transfer-engine
    - Optimized for trillion-parameter scale models
    - Both broadcast and P2P modes available

    All methods are static - no instance state needed.
    """

    @staticmethod
    def is_available() -> bool:
        """Check if checkpoint-engine is available."""
        return CHECKPOINT_ENGINE_AVAILABLE

    @staticmethod
    def create_init_info(cfg: "DictConfig") -> CheckpointEngineInitInfo:
        """Create init info with all config-derived args.

        Args:
            cfg: Configuration object containing generator settings.

        Returns:
            CheckpointEngineInitInfo containing all args needed for sender/receiver creation.
        """
        import socket
        import ray

        # Get ParameterServer host/port
        ps_host = getattr(cfg.generator, "checkpoint_engine_host", None)
        if ps_host is None:
            ps_host = ray._private.services.get_node_ip_address()

        ps_port = getattr(cfg.generator, "checkpoint_engine_port", None)
        if ps_port is None:
            with socket.socket() as sock:
                sock.bind(("", 0))
                ps_port = sock.getsockname()[1]

        # Calculate number of workers
        num_inference_engines = cfg.generator.num_inference_engines
        tp_size = cfg.generator.inference_engine_tensor_parallel_size
        pp_size = cfg.generator.inference_engine_pipeline_parallel_size
        dp_size = cfg.generator.inference_engine_data_parallel_size
        num_workers = num_inference_engines * tp_size * pp_size * dp_size

        # Get transfer mode
        transfer_mode = getattr(cfg.generator, "checkpoint_engine_mode", "broadcast")

        return CheckpointEngineInitInfo(
            ps_host=ps_host,
            ps_port=ps_port,
            num_workers=num_workers,
            model_dtype_str=cfg.generator.model_dtype,
            transfer_mode=transfer_mode,
            override_existing_receiver=cfg.generator.override_existing_update_group == "enable",
            use_overlapped_weight_sync=cfg.generator.get("use_overlapped_weight_sync", False),
        )

    @staticmethod
    def create_sender(
        init_info: CheckpointEngineInitInfo,
        inference_client: "InferenceEngineClient",
    ) -> CheckpointEngineWeightTransferSender:
        """Create a checkpoint-engine sender.

        Args:
            init_info: CheckpointEngineInitInfo containing config-derived args.
            inference_client: Client for coordinating with inference engines.

        Returns:
            A configured CheckpointEngineWeightTransferSender instance.
        """
        return CheckpointEngineWeightTransferSender(
            init_info=init_info,
            inference_client=inference_client,
        )

    @staticmethod
    def create_receiver(
        init_info: CheckpointEngineInitInfo,
    ) -> CheckpointEngineWeightTransferReceiver:
        """Create a checkpoint-engine receiver.

        Args:
            init_info: CheckpointEngineInitInfo from the sender.

        Returns:
            A configured CheckpointEngineWeightTransferReceiver instance.
        """
        return CheckpointEngineWeightTransferReceiver(init_info=init_info)
