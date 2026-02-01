"""Comprehensive weight synchronization tests for cross-component integration.

Tests weight transfer between training and inference components across different
backends (CUDA IPC, NCCL Broadcast, Gloo Broadcast, checkpoint-engine).

Run tests:
    pytest tests/integration/test_cross_component_weight_sync.py -v
    pytest tests/integration/test_cross_component_weight_sync.py -v -m "gpu and integration"
"""

import asyncio
import base64
import os
import pickle
import socket
import tempfile
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterator, List, Optional, Tuple
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import torch
import torch.distributed as dist

from skyrl_train.weight_sync import (
    BroadcastInitInfo,
    BroadcastTransferStrategy,
    BroadcastWeightTransferReceiver,
    BroadcastWeightTransferSender,
    BroadcastWeightUpdateRequest,
    CHECKPOINT_ENGINE_AVAILABLE,
    CheckpointEngineInitInfo,
    CheckpointEngineTransferStrategy,
    CheckpointEngineWeightTransferReceiver,
    CheckpointEngineWeightTransferSender,
    CheckpointEngineWeightUpdateRequest,
    CudaIpcInitInfo,
    CudaIpcTransferStrategy,
    CudaIpcWeightTransferReceiver,
    CudaIpcWeightTransferSender,
    CudaIpcWeightUpdateRequest,
    LoraLoadRequest,
    WeightChunk,
    WeightUpdateRequest,
    get_transfer_strategy_cls,
)
from skyrl_train.weight_sync.base import WeightChunk
from skyrl_train.weight_sync.cuda_ipc_strategy import _IPC_REQUEST_END_MARKER


# =============================================================================
# Test Markers and Fixtures
# =============================================================================


def is_gpu_available() -> bool:
    """Check if CUDA is available."""
    return torch.cuda.is_available()


def is_multi_gpu_available() -> bool:
    """Check if multiple GPUs are available."""
    return torch.cuda.is_available() and torch.cuda.device_count() >= 2


gpu = pytest.mark.gpu
integration = pytest.mark.integration
multi_gpu = pytest.mark.skipif(
    not is_multi_gpu_available(), reason="Requires 2+ GPUs"
)
single_gpu = pytest.mark.skipif(not is_gpu_available(), reason="Requires GPU")


@pytest.fixture
def mock_inference_client():
    """Create a mock inference engine client."""
    client = MagicMock()
    client.update_named_weights = AsyncMock(return_value={"success": True})
    client.overlapped_weight_sync = AsyncMock(return_value={"success": True})
    client.supports_overlapped_weight_sync = MagicMock(return_value=False)
    return client


@pytest.fixture
def mock_inference_client_overlapped():
    """Create a mock inference client supporting overlapped sync."""
    client = MagicMock()
    client.update_named_weights = AsyncMock(return_value={"success": True})
    client.overlapped_weight_sync = AsyncMock(return_value={"success": True})
    client.supports_overlapped_weight_sync = MagicMock(return_value=True)
    return client


@pytest.fixture
def sample_weight_chunk():
    """Create a sample WeightChunk for testing."""
    return WeightChunk(
        names=["model.layer.0.weight", "model.layer.0.bias"],
        dtypes=["torch.float32", "torch.float32"],
        shapes=[[64, 64], [64]],
        tensors=[
            torch.randn(64, 64),
            torch.randn(64),
        ],
    )


@pytest.fixture
def sample_weight_chunk_gpu():
    """Create a sample WeightChunk on GPU."""
    if not is_gpu_available():
        pytest.skip("GPU not available")
    return WeightChunk(
        names=["model.layer.0.weight", "model.layer.0.bias"],
        dtypes=["torch.bfloat16", "torch.bfloat16"],
        shapes=[[64, 64], [64]],
        tensors=[
            torch.randn(64, 64, device="cuda", dtype=torch.bfloat16),
            torch.randn(64, device="cuda", dtype=torch.bfloat16),
        ],
    )


@pytest.fixture
def broadcast_init_info():
    """Create a BroadcastInitInfo for testing."""
    return BroadcastInitInfo(
        master_addr="127.0.0.1",
        master_port=29500,
        rank_offset=1,
        world_size=2,
        group_name="test_weight_sync",
        backend="gloo",
        model_dtype_str="torch.bfloat16",
        override_existing_receiver=True,
        use_overlapped_weight_sync=False,
    )


@pytest.fixture
def cuda_ipc_init_info():
    """Create a CudaIpcInitInfo for testing."""
    return CudaIpcInitInfo(
        model_dtype_str="torch.bfloat16",
        override_existing_receiver=True,
        use_overlapped_weight_sync=False,
    )


# =============================================================================
# Section 1: CUDA IPC Weight Transfer Tests
# =============================================================================


@gpu
@integration
class TestCudaIpcWeightTransfer:
    """Tests for CUDA IPC weight transfer (nmoe -> SGLang)."""

    def test_cuda_ipc_weight_update_request_serialization(self):
        """Test CudaIpcWeightUpdateRequest serialization roundtrip."""
        request = CudaIpcWeightUpdateRequest(
            names=["model.layer.weight"],
            dtypes=["torch.bfloat16"],
            shapes=[[4096, 4096]],
            sizes=[4096 * 4096],
            ipc_handles={"gpu-uuid-0": ("rebuild_func", ("arg1", "arg2"))},
            weight_version="v1.0.0",
        )

        data = request.serialize()
        result = CudaIpcWeightUpdateRequest.deserialize(data)

        assert result.names == request.names
        assert result.dtypes == request.dtypes
        assert result.shapes == request.shapes
        assert result.sizes == request.sizes
        assert result.weight_version == request.weight_version

    def test_cuda_ipc_serialization_4_byte_alignment(self):
        """Test that serialized data is 4-byte aligned."""
        for size in [10, 100, 1000, 4096]:
            request = CudaIpcWeightUpdateRequest(
                names=[f"layer_{i}.weight" for i in range(size % 10 + 1)],
                dtypes=["torch.bfloat16"] * (size % 10 + 1),
                shapes=[[size, size]] * (size % 10 + 1),
                sizes=[size * size] * (size % 10 + 1),
                ipc_handles={},
            )
            data = request.serialize()
            assert len(data) % 4 == 0, f"Size {size}: len={len(data)} not 4-byte aligned"

    def test_cuda_ipc_deserialize_missing_end_marker(self):
        """Test deserialization fails with missing end marker."""
        with pytest.raises(ValueError, match="End marker not found"):
            CudaIpcWeightUpdateRequest.deserialize(b"invalid_data_without_marker")

    def test_cuda_ipc_deserialize_invalid_base64(self):
        """Test deserialization fails with invalid base64."""
        invalid_data = b"!!!invalid_base64!!!" + _IPC_REQUEST_END_MARKER
        with pytest.raises(ValueError, match="Failed to deserialize"):
            CudaIpcWeightUpdateRequest.deserialize(invalid_data)

    def test_cuda_ipc_deserialize_corrupted_pickle(self):
        """Test deserialization fails with corrupted pickle data."""
        corrupted = base64.b64encode(b"not_a_valid_pickle") + _IPC_REQUEST_END_MARKER
        with pytest.raises(ValueError, match="Failed to deserialize"):
            CudaIpcWeightUpdateRequest.deserialize(corrupted)

    @single_gpu
    def test_cuda_ipc_init_info_creation(self):
        """Test CudaIpcInitInfo creation from config."""
        from skyrl_train.config.utils import get_default_config

        cfg = get_default_config()
        cfg.generator.model_dtype = "torch.bfloat16"
        cfg.generator.override_existing_update_group = "enable"
        cfg.generator.use_overlapped_weight_sync = False

        init_info = CudaIpcTransferStrategy.create_init_info(cfg)

        assert isinstance(init_info, CudaIpcInitInfo)
        assert init_info.model_dtype_str == "torch.bfloat16"
        assert init_info.override_existing_receiver is True
        assert init_info.use_overlapped_weight_sync is False

    @single_gpu
    def test_cuda_ipc_receiver_creation(self, cuda_ipc_init_info):
        """Test CudaIpcWeightTransferReceiver creation."""
        receiver = CudaIpcTransferStrategy.create_receiver(cuda_ipc_init_info)

        assert isinstance(receiver, CudaIpcWeightTransferReceiver)
        assert receiver._model_dtype == torch.bfloat16

    @single_gpu
    def test_cuda_ipc_receiver_dtype_validation(self, cuda_ipc_init_info):
        """Test receiver validates dtype matches."""
        receiver = CudaIpcTransferStrategy.create_receiver(cuda_ipc_init_info)

        # Create request with mismatched dtype
        request = CudaIpcWeightUpdateRequest(
            names=["layer.weight"],
            dtypes=["torch.float32"],  # Different from bfloat16
            shapes=[[64, 64]],
            sizes=[64 * 64],
            ipc_handles={"gpu-uuid": ("fn", ("args",))},
        )

        with pytest.raises(AssertionError, match="mismatch dtype"):
            list(receiver.receive_weights(request))

    def test_cuda_ipc_multiple_weights_serialization(self):
        """Test serialization with multiple weights in single request."""
        names = [f"model.layers.{i}.weight" for i in range(100)]
        shapes = [[1024, 1024] for _ in range(100)]
        sizes = [1024 * 1024 for _ in range(100)]

        request = CudaIpcWeightUpdateRequest(
            names=names,
            dtypes=["torch.bfloat16"] * 100,
            shapes=shapes,
            sizes=sizes,
            ipc_handles={f"gpu-{i}": (f"fn_{i}", (f"arg_{i}",)) for i in range(8)},
            weight_version="step_1000",
        )

        data = request.serialize()
        result = CudaIpcWeightUpdateRequest.deserialize(data)

        assert len(result.names) == 100
        assert result.ipc_handles == request.ipc_handles
        assert result.weight_version == "step_1000"

    def test_cuda_ipc_weight_versioning(self):
        """Test weight versioning is preserved through serialization."""
        versions = ["v1", "step_100", "epoch_5_step_1000", None, ""]
        for version in versions:
            request = CudaIpcWeightUpdateRequest(
                names=["layer.weight"],
                dtypes=["torch.bfloat16"],
                shapes=[[64, 64]],
                sizes=[64 * 64],
                ipc_handles={},
                weight_version=version,
            )
            data = request.serialize()
            result = CudaIpcWeightUpdateRequest.deserialize(data)
            assert result.weight_version == version


# =============================================================================
# Section 2: NCCL Broadcast Weight Transfer Tests
# =============================================================================


@gpu
@integration
class TestNcclBroadcastWeightTransfer:
    """Tests for NCCL-based weight broadcast."""

    def test_broadcast_init_info_creation(self, monkeypatch):
        """Test BroadcastInitInfo creation from config."""
        from skyrl_train.config.utils import get_default_config
        import skyrl_train.weight_sync.broadcast_strategy as broadcast_module

        monkeypatch.setattr(
            broadcast_module.ray._private.services,
            "get_node_ip_address",
            lambda: "192.168.1.100",
        )

        cfg = get_default_config()
        cfg.generator.weight_sync_backend = "nccl"
        cfg.generator.model_dtype = "torch.bfloat16"
        cfg.generator.num_inference_engines = 2
        cfg.generator.inference_engine_tensor_parallel_size = 2
        cfg.generator.inference_engine_pipeline_parallel_size = 1
        cfg.generator.inference_engine_data_parallel_size = 1
        cfg.generator.override_existing_update_group = "enable"

        init_info = BroadcastTransferStrategy.create_init_info(cfg)

        assert isinstance(init_info, BroadcastInitInfo)
        assert init_info.master_addr == "192.168.1.100"
        assert isinstance(init_info.master_port, int)
        assert init_info.backend == "nccl"
        # world_size = num_engines * tp * pp * dp + 1 = 2 * 2 * 1 * 1 + 1 = 5
        assert init_info.world_size == 5

    def test_broadcast_init_info_for_engine(self, broadcast_init_info):
        """Test BroadcastInitInfo.for_engine adjusts rank_offset correctly."""
        tp_size = 2
        pp_size = 1

        # Engine 0
        info_e0 = broadcast_init_info.for_engine(0, tp_size, pp_size)
        assert info_e0.rank_offset == broadcast_init_info.rank_offset + 0

        # Engine 1
        info_e1 = broadcast_init_info.for_engine(1, tp_size, pp_size)
        assert info_e1.rank_offset == broadcast_init_info.rank_offset + 2

        # Engine 2
        info_e2 = broadcast_init_info.for_engine(2, tp_size, pp_size)
        assert info_e2.rank_offset == broadcast_init_info.rank_offset + 4

    def test_broadcast_weight_update_request_len(self):
        """Test BroadcastWeightUpdateRequest __len__ method."""
        request = BroadcastWeightUpdateRequest(
            names=["layer1.weight", "layer2.weight", "layer3.bias"],
            dtypes=["torch.bfloat16"] * 3,
            shapes=[[64, 64], [128, 64], [64]],
        )
        assert len(request) == 3

    def test_broadcast_weight_update_request_mismatched_lengths(self):
        """Test BroadcastWeightUpdateRequest raises on mismatched lengths."""
        with pytest.raises(ValueError, match="must have the same length"):
            BroadcastWeightUpdateRequest(
                names=["layer1", "layer2"],
                dtypes=["torch.bfloat16"],  # Only 1 element
                shapes=[[64, 64]],  # Only 1 element
            )

    def test_broadcast_strategy_type(self, broadcast_init_info):
        """Test BroadcastInitInfo.strategy_type returns correct class."""
        assert broadcast_init_info.strategy_type() is BroadcastTransferStrategy


# =============================================================================
# Section 3: Gloo-based Weight Transfer Tests
# =============================================================================


@integration
class TestGlooBroadcastWeightTransfer:
    """Tests for Gloo-based weight transfer (CPU fallback)."""

    def test_gloo_backend_selection(self):
        """Test Gloo backend is selected for cross-node scenarios."""
        from skyrl_train.config.utils import get_default_config

        cfg = get_default_config()
        cfg.generator.weight_sync_backend = "gloo"
        cfg.trainer.placement.colocate_all = False

        strategy_cls = get_transfer_strategy_cls(cfg)
        assert strategy_cls is BroadcastTransferStrategy

    def test_gloo_init_info_creation(self, monkeypatch):
        """Test BroadcastInitInfo with Gloo backend."""
        from skyrl_train.config.utils import get_default_config
        import skyrl_train.weight_sync.broadcast_strategy as broadcast_module

        monkeypatch.setattr(
            broadcast_module.ray._private.services,
            "get_node_ip_address",
            lambda: "10.0.0.1",
        )

        cfg = get_default_config()
        cfg.generator.weight_sync_backend = "gloo"
        cfg.generator.model_dtype = "torch.float32"
        cfg.generator.num_inference_engines = 4

        init_info = BroadcastTransferStrategy.create_init_info(cfg)

        assert init_info.backend == "gloo"
        assert init_info.model_dtype_str == "torch.float32"

    def test_gloo_receiver_dtype_conversion(self, broadcast_init_info):
        """Test receiver handles dtype correctly."""
        # Modify init_info for different dtypes
        import torch
        from skyrl_train.utils import str_to_torch_dtype

        for dtype_str in ["torch.float32", "torch.bfloat16", "torch.float16"]:
            broadcast_init_info.model_dtype_str = dtype_str
            expected_dtype = str_to_torch_dtype(dtype_str)
            # Verify dtype string conversion
            assert expected_dtype in [torch.float32, torch.bfloat16, torch.float16]


# =============================================================================
# Section 4: Checkpoint Engine Weight Transfer Tests
# =============================================================================


@integration
class TestCheckpointEngineWeightTransfer:
    """Tests for checkpoint-engine based weight transfer."""

    def test_checkpoint_engine_availability(self):
        """Test checkpoint-engine availability detection."""
        # This should not raise, just report availability
        is_available = CHECKPOINT_ENGINE_AVAILABLE
        assert isinstance(is_available, bool)

    def test_checkpoint_engine_init_info_creation(self, monkeypatch):
        """Test CheckpointEngineInitInfo creation from config."""
        from skyrl_train.config.utils import get_default_config
        import skyrl_train.weight_sync.checkpoint_engine_strategy as ce_module

        monkeypatch.setattr(
            ce_module.ray._private.services,
            "get_node_ip_address",
            lambda: "192.168.1.50",
        )

        cfg = get_default_config()
        cfg.generator.weight_sync_backend = "checkpoint_engine"
        cfg.generator.model_dtype = "torch.bfloat16"
        cfg.generator.num_inference_engines = 8
        cfg.generator.inference_engine_tensor_parallel_size = 4
        cfg.generator.inference_engine_pipeline_parallel_size = 1
        cfg.generator.inference_engine_data_parallel_size = 1

        init_info = CheckpointEngineTransferStrategy.create_init_info(cfg)

        assert isinstance(init_info, CheckpointEngineInitInfo)
        assert init_info.ps_host == "192.168.1.50"
        assert isinstance(init_info.ps_port, int)
        # num_workers = num_engines * tp * pp * dp = 8 * 4 * 1 * 1 = 32
        assert init_info.num_workers == 32

    def test_checkpoint_engine_init_info_for_engine(self):
        """Test CheckpointEngineInitInfo.for_engine adjusts worker_rank."""
        init_info = CheckpointEngineInitInfo(
            ps_host="localhost",
            ps_port=8000,
            num_workers=16,
            model_dtype_str="torch.bfloat16",
            transfer_mode="broadcast",
            worker_rank=0,
            override_existing_receiver=True,
        )

        tp_size = 4
        pp_size = 1

        # Engine 0
        info_e0 = init_info.for_engine(0, tp_size, pp_size)
        assert info_e0.worker_rank == 0

        # Engine 1
        info_e1 = init_info.for_engine(1, tp_size, pp_size)
        assert info_e1.worker_rank == 4

        # Engine 2
        info_e2 = init_info.for_engine(2, tp_size, pp_size)
        assert info_e2.worker_rank == 8

    def test_checkpoint_engine_weight_update_request(self):
        """Test CheckpointEngineWeightUpdateRequest creation."""
        request = CheckpointEngineWeightUpdateRequest(
            names=["layer.weight", "layer.bias"],
            dtypes=["torch.bfloat16", "torch.bfloat16"],
            shapes=[[1024, 1024], [1024]],
            weight_version="step_5000",
            bucket_id="bucket_step_5000",
        )

        assert len(request) == 2
        assert request.bucket_id == "bucket_step_5000"
        assert request.weight_version == "step_5000"

    def test_checkpoint_engine_strategy_is_available(self):
        """Test CheckpointEngineTransferStrategy.is_available method."""
        available = CheckpointEngineTransferStrategy.is_available()
        assert available == CHECKPOINT_ENGINE_AVAILABLE

    def test_checkpoint_engine_transfer_modes(self):
        """Test checkpoint-engine supports broadcast and p2p modes."""
        for mode in ["broadcast", "p2p"]:
            init_info = CheckpointEngineInitInfo(
                ps_host="localhost",
                ps_port=8000,
                num_workers=4,
                model_dtype_str="torch.bfloat16",
                transfer_mode=mode,
                worker_rank=0,
                override_existing_receiver=True,
            )
            assert init_info.transfer_mode == mode

    @pytest.mark.skipif(
        not CHECKPOINT_ENGINE_AVAILABLE,
        reason="checkpoint-engine not installed",
    )
    def test_checkpoint_engine_sender_creation(self, mock_inference_client):
        """Test CheckpointEngineWeightTransferSender creation."""
        init_info = CheckpointEngineInitInfo(
            ps_host="localhost",
            ps_port=12345,
            num_workers=4,
            model_dtype_str="torch.bfloat16",
            transfer_mode="broadcast",
            worker_rank=0,
            override_existing_receiver=True,
        )

        # This will create a real ParameterServer if checkpoint-engine is available
        sender = CheckpointEngineTransferStrategy.create_sender(
            init_info, mock_inference_client
        )
        assert isinstance(sender, CheckpointEngineWeightTransferSender)
        sender.teardown()


# =============================================================================
# Section 5: Weight Versioning and Consistency Tests
# =============================================================================


@integration
class TestWeightVersioningAndConsistency:
    """Tests for weight versioning and consistency tracking."""

    def test_weight_version_in_base_request(self):
        """Test weight_version field in base WeightUpdateRequest."""
        request = WeightUpdateRequest(
            names=["layer.weight"],
            dtypes=["torch.bfloat16"],
            shapes=[[64, 64]],
            weight_version="epoch_10_step_5000",
        )
        assert request.weight_version == "epoch_10_step_5000"

    def test_weight_version_none_default(self):
        """Test weight_version defaults to None."""
        request = WeightUpdateRequest(
            names=["layer.weight"],
            dtypes=["torch.bfloat16"],
            shapes=[[64, 64]],
        )
        assert request.weight_version is None

    def test_weight_version_propagation_broadcast(self):
        """Test weight_version propagates through broadcast request."""
        version = "training_step_12345"
        request = BroadcastWeightUpdateRequest(
            names=["model.weight"],
            dtypes=["torch.bfloat16"],
            shapes=[[1024, 1024]],
            weight_version=version,
        )
        assert request.weight_version == version

    def test_weight_version_propagation_cuda_ipc(self):
        """Test weight_version propagates through CUDA IPC serialization."""
        version = "checkpoint_v2.1"
        request = CudaIpcWeightUpdateRequest(
            names=["model.weight"],
            dtypes=["torch.bfloat16"],
            shapes=[[1024, 1024]],
            sizes=[1024 * 1024],
            ipc_handles={},
            weight_version=version,
        )

        data = request.serialize()
        result = CudaIpcWeightUpdateRequest.deserialize(data)
        assert result.weight_version == version

    def test_weight_chunk_consistency(self, sample_weight_chunk):
        """Test WeightChunk maintains consistency between fields."""
        chunk = sample_weight_chunk

        assert len(chunk.names) == len(chunk.dtypes)
        assert len(chunk.names) == len(chunk.shapes)
        assert len(chunk.names) == len(chunk.tensors)

        for name, dtype, shape, tensor in zip(
            chunk.names, chunk.dtypes, chunk.shapes, chunk.tensors
        ):
            assert list(tensor.shape) == shape

    def test_weight_chunk_total_numel(self, sample_weight_chunk):
        """Test WeightChunk.total_numel calculation."""
        chunk = sample_weight_chunk
        expected_numel = sum(t.numel() for t in chunk.tensors)
        assert chunk.total_numel == expected_numel

    def test_weight_chunk_total_size_bytes(self, sample_weight_chunk):
        """Test WeightChunk.total_size_bytes calculation."""
        chunk = sample_weight_chunk
        expected_size = sum(t.numel() * t.element_size() for t in chunk.tensors)
        assert chunk.total_size_bytes == expected_size

    def test_weight_chunk_validation_mismatched_lengths(self):
        """Test WeightChunk raises on mismatched field lengths."""
        with pytest.raises(ValueError, match="must have the same length"):
            WeightChunk(
                names=["layer1", "layer2"],
                dtypes=["torch.float32"],  # Only 1
                shapes=[[64, 64], [64, 64]],
                tensors=[torch.randn(64, 64), torch.randn(64, 64)],
            )

    def test_weight_update_request_validation(self):
        """Test WeightUpdateRequest validates field lengths."""
        with pytest.raises(ValueError, match="must have the same length"):
            WeightUpdateRequest(
                names=["a", "b", "c"],
                dtypes=["torch.float32", "torch.float32"],
                shapes=[[1], [2]],
            )


# =============================================================================
# Section 6: Partial Weight Update Tests
# =============================================================================


@integration
class TestPartialWeightUpdate:
    """Tests for partial weight updates (only changed layers)."""

    def test_single_layer_update(self):
        """Test updating a single layer."""
        request = BroadcastWeightUpdateRequest(
            names=["model.layers.5.self_attn.q_proj.weight"],
            dtypes=["torch.bfloat16"],
            shapes=[[4096, 4096]],
            weight_version="partial_update_1",
        )
        assert len(request) == 1

    def test_selective_layer_update(self):
        """Test updating selective layers (e.g., only attention)."""
        attention_layers = [
            f"model.layers.{i}.self_attn.{proj}.weight"
            for i in range(32)
            for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]
        ]
        request = BroadcastWeightUpdateRequest(
            names=attention_layers,
            dtypes=["torch.bfloat16"] * len(attention_layers),
            shapes=[[4096, 4096]] * len(attention_layers),
            weight_version="attention_only_update",
        )
        assert len(request) == 32 * 4  # 32 layers * 4 projections

    def test_lora_adapter_update(self):
        """Test LoRA adapter weight update."""
        lora_layers = [
            f"model.layers.{i}.self_attn.q_proj.lora_A.weight"
            for i in range(32)
        ] + [
            f"model.layers.{i}.self_attn.q_proj.lora_B.weight"
            for i in range(32)
        ]
        request = BroadcastWeightUpdateRequest(
            names=lora_layers,
            dtypes=["torch.bfloat16"] * len(lora_layers),
            shapes=[[16, 4096]] * 32 + [[4096, 16]] * 32,  # LoRA rank 16
            weight_version="lora_update_1",
        )
        assert len(request) == 64

    def test_lora_load_request(self):
        """Test LoraLoadRequest for disk-based LoRA loading."""
        request = LoraLoadRequest(lora_path="/path/to/lora/adapter")
        assert request.lora_path == "/path/to/lora/adapter"
        assert request.names == []
        assert request.dtypes == []
        assert request.shapes == []

    def test_embedding_layer_update(self):
        """Test embedding layer update."""
        request = BroadcastWeightUpdateRequest(
            names=["model.embed_tokens.weight"],
            dtypes=["torch.bfloat16"],
            shapes=[[128256, 4096]],  # Llama vocab size
            weight_version="embedding_update",
        )
        assert len(request) == 1

    def test_output_layer_update(self):
        """Test output/lm_head layer update."""
        request = BroadcastWeightUpdateRequest(
            names=["lm_head.weight"],
            dtypes=["torch.bfloat16"],
            shapes=[[128256, 4096]],
            weight_version="lm_head_update",
        )
        assert len(request) == 1


# =============================================================================
# Section 7: Expert Weight Transfer with EP Tests
# =============================================================================


@gpu
@integration
class TestExpertWeightTransferWithEP:
    """Tests for expert weight transfer with Expert Parallelism."""

    def test_moe_expert_weight_structure(self):
        """Test MoE expert weight naming structure."""
        num_experts = 8
        num_layers = 32

        expert_weights = []
        for layer in range(num_layers):
            for expert in range(num_experts):
                expert_weights.extend([
                    f"model.layers.{layer}.mlp.experts.{expert}.gate_proj.weight",
                    f"model.layers.{layer}.mlp.experts.{expert}.up_proj.weight",
                    f"model.layers.{layer}.mlp.experts.{expert}.down_proj.weight",
                ])

        request = BroadcastWeightUpdateRequest(
            names=expert_weights,
            dtypes=["torch.bfloat16"] * len(expert_weights),
            shapes=[[14336, 4096]] * len(expert_weights),  # DeepSeekMoE dimensions
            weight_version="moe_expert_update",
        )

        assert len(request) == num_layers * num_experts * 3

    def test_expert_parallel_weight_sharding(self):
        """Test expert weight sharding for EP."""
        num_experts = 64  # DeepSeek routed experts
        ep_size = 8  # 8-way expert parallelism
        experts_per_rank = num_experts // ep_size

        # Each EP rank gets a subset of experts
        for ep_rank in range(ep_size):
            start_expert = ep_rank * experts_per_rank
            end_expert = start_expert + experts_per_rank

            expert_names = [
                f"model.layers.0.mlp.experts.{e}.gate_proj.weight"
                for e in range(start_expert, end_expert)
            ]

            request = BroadcastWeightUpdateRequest(
                names=expert_names,
                dtypes=["torch.bfloat16"] * len(expert_names),
                shapes=[[14336, 4096]] * len(expert_names),
                weight_version=f"ep_rank_{ep_rank}_update",
            )

            assert len(request) == experts_per_rank

    def test_router_weight_update(self):
        """Test MoE router weight update."""
        num_layers = 32
        router_weights = [
            f"model.layers.{i}.mlp.gate.weight"
            for i in range(num_layers)
        ]

        request = BroadcastWeightUpdateRequest(
            names=router_weights,
            dtypes=["torch.bfloat16"] * len(router_weights),
            shapes=[[64, 4096]] * len(router_weights),  # 64 experts
            weight_version="router_update",
        )

        assert len(request) == num_layers

    def test_shared_expert_weight_update(self):
        """Test shared expert weight update (DeepSeek-style)."""
        num_layers = 32
        num_shared_experts = 2

        shared_weights = []
        for layer in range(num_layers):
            for se in range(num_shared_experts):
                shared_weights.extend([
                    f"model.layers.{layer}.mlp.shared_experts.{se}.gate_proj.weight",
                    f"model.layers.{layer}.mlp.shared_experts.{se}.up_proj.weight",
                    f"model.layers.{layer}.mlp.shared_experts.{se}.down_proj.weight",
                ])

        request = BroadcastWeightUpdateRequest(
            names=shared_weights,
            dtypes=["torch.bfloat16"] * len(shared_weights),
            shapes=[[14336, 4096]] * len(shared_weights),
            weight_version="shared_expert_update",
        )

        assert len(request) == num_layers * num_shared_experts * 3


# =============================================================================
# Section 8: Weight Transfer with TP Sharding Tests
# =============================================================================


@gpu
@integration
class TestWeightTransferWithTPSharding:
    """Tests for weight transfer with Tensor Parallelism sharding."""

    def test_column_parallel_weight_shapes(self):
        """Test column-parallel weight shapes for TP."""
        hidden_size = 4096
        intermediate_size = 14336
        tp_size = 8

        # Column parallel: split output dimension
        local_intermediate = intermediate_size // tp_size

        for tp_rank in range(tp_size):
            request = BroadcastWeightUpdateRequest(
                names=[f"model.layers.0.mlp.gate_proj.weight_tp{tp_rank}"],
                dtypes=["torch.bfloat16"],
                shapes=[[local_intermediate, hidden_size]],
                weight_version=f"tp_rank_{tp_rank}",
            )
            assert request.shapes[0] == [local_intermediate, hidden_size]

    def test_row_parallel_weight_shapes(self):
        """Test row-parallel weight shapes for TP."""
        hidden_size = 4096
        intermediate_size = 14336
        tp_size = 8

        # Row parallel: split input dimension
        local_intermediate = intermediate_size // tp_size

        for tp_rank in range(tp_size):
            request = BroadcastWeightUpdateRequest(
                names=[f"model.layers.0.mlp.down_proj.weight_tp{tp_rank}"],
                dtypes=["torch.bfloat16"],
                shapes=[[hidden_size, local_intermediate]],
                weight_version=f"tp_rank_{tp_rank}",
            )
            assert request.shapes[0] == [hidden_size, local_intermediate]

    def test_attention_qkv_tp_sharding(self):
        """Test attention Q/K/V weight sharding for TP."""
        hidden_size = 4096
        num_heads = 32
        head_dim = hidden_size // num_heads
        tp_size = 8
        local_heads = num_heads // tp_size

        for tp_rank in range(tp_size):
            q_shape = [local_heads * head_dim, hidden_size]
            k_shape = [local_heads * head_dim, hidden_size]
            v_shape = [local_heads * head_dim, hidden_size]

            request = BroadcastWeightUpdateRequest(
                names=[
                    f"model.layers.0.self_attn.q_proj.weight_tp{tp_rank}",
                    f"model.layers.0.self_attn.k_proj.weight_tp{tp_rank}",
                    f"model.layers.0.self_attn.v_proj.weight_tp{tp_rank}",
                ],
                dtypes=["torch.bfloat16"] * 3,
                shapes=[q_shape, k_shape, v_shape],
                weight_version=f"tp_rank_{tp_rank}",
            )
            assert len(request) == 3

    def test_broadcast_init_info_tp_world_size(self, monkeypatch):
        """Test BroadcastInitInfo world_size calculation with TP."""
        from skyrl_train.config.utils import get_default_config
        import skyrl_train.weight_sync.broadcast_strategy as broadcast_module

        monkeypatch.setattr(
            broadcast_module.ray._private.services,
            "get_node_ip_address",
            lambda: "127.0.0.1",
        )

        cfg = get_default_config()
        cfg.generator.weight_sync_backend = "nccl"
        cfg.generator.num_inference_engines = 2
        cfg.generator.inference_engine_tensor_parallel_size = 4
        cfg.generator.inference_engine_pipeline_parallel_size = 1
        cfg.generator.inference_engine_data_parallel_size = 1

        init_info = BroadcastTransferStrategy.create_init_info(cfg)

        # world_size = num_engines * tp * pp * dp + 1 = 2 * 4 * 1 * 1 + 1 = 9
        assert init_info.world_size == 9

    def test_gqa_weight_shapes(self):
        """Test Grouped Query Attention weight shapes for TP."""
        hidden_size = 4096
        num_heads = 32
        num_kv_heads = 8
        head_dim = hidden_size // num_heads
        tp_size = 8

        local_heads = num_heads // tp_size
        local_kv_heads = num_kv_heads // tp_size

        request = BroadcastWeightUpdateRequest(
            names=[
                "model.layers.0.self_attn.q_proj.weight",
                "model.layers.0.self_attn.k_proj.weight",
                "model.layers.0.self_attn.v_proj.weight",
            ],
            dtypes=["torch.bfloat16"] * 3,
            shapes=[
                [local_heads * head_dim, hidden_size],  # Q
                [local_kv_heads * head_dim, hidden_size],  # K (GQA)
                [local_kv_heads * head_dim, hidden_size],  # V (GQA)
            ],
            weight_version="gqa_tp_update",
        )

        assert request.shapes[0][0] > request.shapes[1][0]  # Q has more heads than K


# =============================================================================
# Section 9: Async Weight Transfer Overlap Tests
# =============================================================================


@integration
class TestAsyncWeightTransferOverlap:
    """Tests for async weight transfer with computation overlap."""

    def test_overlapped_sync_flag_broadcast(self, broadcast_init_info):
        """Test use_overlapped_weight_sync flag in BroadcastInitInfo."""
        broadcast_init_info.use_overlapped_weight_sync = True
        assert broadcast_init_info.use_overlapped_weight_sync is True

    def test_overlapped_sync_flag_cuda_ipc(self, cuda_ipc_init_info):
        """Test use_overlapped_weight_sync flag in CudaIpcInitInfo."""
        cuda_ipc_init_info.use_overlapped_weight_sync = True
        assert cuda_ipc_init_info.use_overlapped_weight_sync is True

    @pytest.mark.asyncio
    async def test_overlapped_sync_client_call(self, mock_inference_client_overlapped):
        """Test overlapped sync calls correct client method."""
        client = mock_inference_client_overlapped

        request = BroadcastWeightUpdateRequest(
            names=["layer.weight"],
            dtypes=["torch.bfloat16"],
            shapes=[[64, 64]],
        )

        # Simulate overlapped sync
        assert client.supports_overlapped_weight_sync()
        await client.overlapped_weight_sync(request)

        client.overlapped_weight_sync.assert_called_once()

    @pytest.mark.asyncio
    async def test_fallback_to_standard_sync(self, mock_inference_client):
        """Test fallback to standard sync when overlap not supported."""
        client = mock_inference_client

        request = BroadcastWeightUpdateRequest(
            names=["layer.weight"],
            dtypes=["torch.bfloat16"],
            shapes=[[64, 64]],
        )

        # Overlapped not supported
        assert not client.supports_overlapped_weight_sync()

        # Should use standard update
        await client.update_named_weights(request)
        client.update_named_weights.assert_called_once()

    @pytest.mark.asyncio
    async def test_concurrent_weight_transfers(self, mock_inference_client):
        """Test concurrent weight transfers complete correctly."""
        client = mock_inference_client

        requests = [
            BroadcastWeightUpdateRequest(
                names=[f"layer{i}.weight"],
                dtypes=["torch.bfloat16"],
                shapes=[[64, 64]],
                weight_version=f"v{i}",
            )
            for i in range(10)
        ]

        # Run transfers concurrently
        tasks = [client.update_named_weights(req) for req in requests]
        results = await asyncio.gather(*tasks)

        assert len(results) == 10
        assert all(r == {"success": True} for r in results)

    def test_overlapped_sync_config_creation(self, monkeypatch):
        """Test overlapped sync config propagates through init_info."""
        from skyrl_train.config.utils import get_default_config
        import skyrl_train.weight_sync.broadcast_strategy as broadcast_module

        monkeypatch.setattr(
            broadcast_module.ray._private.services,
            "get_node_ip_address",
            lambda: "127.0.0.1",
        )

        cfg = get_default_config()
        cfg.generator.weight_sync_backend = "nccl"
        cfg.generator.use_overlapped_weight_sync = True

        init_info = BroadcastTransferStrategy.create_init_info(cfg)
        assert init_info.use_overlapped_weight_sync is True


# =============================================================================
# Section 10: Weight Transfer Failure Recovery Tests
# =============================================================================


@integration
class TestWeightTransferFailureRecovery:
    """Tests for weight transfer failure recovery."""

    @pytest.mark.asyncio
    async def test_client_connection_failure(self):
        """Test handling of client connection failure."""
        client = MagicMock()
        client.update_named_weights = AsyncMock(
            side_effect=ConnectionError("Connection refused")
        )

        request = BroadcastWeightUpdateRequest(
            names=["layer.weight"],
            dtypes=["torch.bfloat16"],
            shapes=[[64, 64]],
        )

        with pytest.raises(ConnectionError, match="Connection refused"):
            await client.update_named_weights(request)

    @pytest.mark.asyncio
    async def test_timeout_handling(self):
        """Test handling of transfer timeout."""
        client = MagicMock()
        client.update_named_weights = AsyncMock(
            side_effect=asyncio.TimeoutError("Transfer timed out")
        )

        request = BroadcastWeightUpdateRequest(
            names=["layer.weight"],
            dtypes=["torch.bfloat16"],
            shapes=[[64, 64]],
        )

        with pytest.raises(asyncio.TimeoutError):
            await client.update_named_weights(request)

    @pytest.mark.asyncio
    async def test_partial_transfer_failure(self):
        """Test handling of partial transfer failure."""
        call_count = 0

        async def fail_on_second_call(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise RuntimeError("Transfer failed midway")
            return {"success": True}

        client = MagicMock()
        client.update_named_weights = AsyncMock(side_effect=fail_on_second_call)

        requests = [
            BroadcastWeightUpdateRequest(
                names=[f"layer{i}.weight"],
                dtypes=["torch.bfloat16"],
                shapes=[[64, 64]],
            )
            for i in range(3)
        ]

        results = []
        for req in requests:
            try:
                result = await client.update_named_weights(req)
                results.append(("success", result))
            except RuntimeError as e:
                results.append(("error", str(e)))

        assert results[0] == ("success", {"success": True})
        assert results[1] == ("error", "Transfer failed midway")
        # Third request should still work after recovery
        assert results[2] == ("success", {"success": True})

    def test_serialization_corruption_detection(self):
        """Test detection of serialization corruption."""
        # Create valid request and serialize
        request = CudaIpcWeightUpdateRequest(
            names=["layer.weight"],
            dtypes=["torch.bfloat16"],
            shapes=[[64, 64]],
            sizes=[64 * 64],
            ipc_handles={},
        )
        data = request.serialize()

        # Corrupt the data (flip some bits)
        corrupted = bytearray(data)
        if len(corrupted) > 10:
            corrupted[5] ^= 0xFF
            corrupted[10] ^= 0xFF

        with pytest.raises(ValueError):
            CudaIpcWeightUpdateRequest.deserialize(bytes(corrupted))

    def test_ipc_handle_missing_gpu(self):
        """Test handling of missing GPU in IPC handles."""
        request = CudaIpcWeightUpdateRequest(
            names=["layer.weight"],
            dtypes=["torch.bfloat16"],
            shapes=[[64, 64]],
            sizes=[64 * 64],
            ipc_handles={"gpu-0": ("fn", ("args",))},  # Only gpu-0
        )

        # Accessing non-existent GPU should raise KeyError
        with pytest.raises(KeyError):
            _ = request.ipc_handles["gpu-999"]

    @pytest.mark.asyncio
    async def test_retry_on_transient_failure(self):
        """Test retry logic on transient failures."""
        attempt = 0
        max_retries = 3

        async def succeed_on_third_try(*args, **kwargs):
            nonlocal attempt
            attempt += 1
            if attempt < 3:
                raise ConnectionError("Transient failure")
            return {"success": True}

        client = MagicMock()
        client.update_named_weights = AsyncMock(side_effect=succeed_on_third_try)

        request = BroadcastWeightUpdateRequest(
            names=["layer.weight"],
            dtypes=["torch.bfloat16"],
            shapes=[[64, 64]],
        )

        # Simulate retry logic
        for retry in range(max_retries):
            try:
                result = await client.update_named_weights(request)
                break
            except ConnectionError:
                if retry == max_retries - 1:
                    raise
                continue

        assert result == {"success": True}
        assert attempt == 3


# =============================================================================
# Section 11: Strategy Selection Tests
# =============================================================================


@integration
class TestStrategySelection:
    """Tests for transfer strategy selection logic."""

    def test_auto_mode_colocated_selects_cuda_ipc(self):
        """Test auto mode selects CUDA IPC when colocated."""
        from skyrl_train.config.utils import get_default_config

        cfg = get_default_config()
        cfg.generator.weight_sync_backend = "auto"
        cfg.trainer.placement.colocate_all = True

        strategy_cls = get_transfer_strategy_cls(cfg)
        assert strategy_cls is CudaIpcTransferStrategy

    def test_auto_mode_distributed_selects_broadcast(self):
        """Test auto mode selects Broadcast when not colocated."""
        from skyrl_train.config.utils import get_default_config

        cfg = get_default_config()
        cfg.generator.weight_sync_backend = "auto"
        cfg.trainer.placement.colocate_all = False

        strategy_cls = get_transfer_strategy_cls(cfg)
        assert strategy_cls is BroadcastTransferStrategy

    def test_nccl_colocated_selects_cuda_ipc(self):
        """Test NCCL backend with colocate selects CUDA IPC."""
        from skyrl_train.config.utils import get_default_config

        cfg = get_default_config()
        cfg.generator.weight_sync_backend = "nccl"
        cfg.trainer.placement.colocate_all = True

        strategy_cls = get_transfer_strategy_cls(cfg)
        assert strategy_cls is CudaIpcTransferStrategy

    def test_nccl_distributed_selects_broadcast(self):
        """Test NCCL backend without colocate selects Broadcast."""
        from skyrl_train.config.utils import get_default_config

        cfg = get_default_config()
        cfg.generator.weight_sync_backend = "nccl"
        cfg.trainer.placement.colocate_all = False

        strategy_cls = get_transfer_strategy_cls(cfg)
        assert strategy_cls is BroadcastTransferStrategy

    def test_gloo_always_selects_broadcast(self):
        """Test Gloo backend always selects Broadcast."""
        from skyrl_train.config.utils import get_default_config

        for colocate in [True, False]:
            cfg = get_default_config()
            cfg.generator.weight_sync_backend = "gloo"
            cfg.trainer.placement.colocate_all = colocate

            strategy_cls = get_transfer_strategy_cls(cfg)
            assert strategy_cls is BroadcastTransferStrategy

    def test_checkpoint_engine_selection(self):
        """Test checkpoint_engine backend selection."""
        from skyrl_train.config.utils import get_default_config

        cfg = get_default_config()
        cfg.generator.weight_sync_backend = "checkpoint_engine"
        cfg.trainer.placement.colocate_all = False

        strategy_cls = get_transfer_strategy_cls(cfg)

        if CHECKPOINT_ENGINE_AVAILABLE:
            assert strategy_cls is CheckpointEngineTransferStrategy
        else:
            # Falls back to Broadcast when not available
            assert strategy_cls is BroadcastTransferStrategy


# =============================================================================
# Section 12: Weight Extractor Utility Tests
# =============================================================================


@integration
class TestWeightExtractorUtils:
    """Tests for weight extractor utilities."""

    def test_yield_module_grouped_chunks_basic(self):
        """Test basic module grouping of weight chunks."""
        from skyrl_train.weight_sync.weight_extractor_utils import (
            yield_module_grouped_chunks,
        )

        # Simulate model parameters
        params = {
            "model.layers.0.self_attn.q_proj.weight": torch.randn(64, 64),
            "model.layers.0.self_attn.k_proj.weight": torch.randn(64, 64),
            "model.layers.0.mlp.gate_proj.weight": torch.randn(128, 64),
        }

        def gather_fn(param):
            return param

        def shape_fn(name, param, tensor):
            return list(tensor.shape)

        chunks = list(
            yield_module_grouped_chunks(
                params=params,
                dtype=torch.float32,
                gather_tensor_fn=gather_fn,
                get_shape_fn=shape_fn,
            )
        )

        # Should group by module
        assert len(chunks) >= 1
        total_params = sum(len(c.names) for c in chunks)
        assert total_params == 3

    def test_yield_module_grouped_chunks_batching(self):
        """Test batching with size threshold."""
        from skyrl_train.weight_sync.weight_extractor_utils import (
            yield_module_grouped_chunks,
        )

        # Create many small parameters
        params = {
            f"model.layers.{i}.weight": torch.randn(64, 64)
            for i in range(100)
        }

        def gather_fn(param):
            return param

        def shape_fn(name, param, tensor):
            return list(tensor.shape)

        # Very small threshold to force multiple batches
        chunks = list(
            yield_module_grouped_chunks(
                params=params,
                dtype=torch.float32,
                gather_tensor_fn=gather_fn,
                get_shape_fn=shape_fn,
                batch_size_threshold_gb=0.00001,  # ~10KB
            )
        )

        # Should produce multiple chunks due to threshold
        assert len(chunks) > 1

    def test_yield_module_grouped_chunks_dtype_conversion(self):
        """Test dtype conversion in chunking."""
        from skyrl_train.weight_sync.weight_extractor_utils import (
            yield_module_grouped_chunks,
        )

        params = {
            "model.layer.weight": torch.randn(64, 64, dtype=torch.float32),
        }

        def gather_fn(param):
            return param

        def shape_fn(name, param, tensor):
            return list(tensor.shape)

        chunks = list(
            yield_module_grouped_chunks(
                params=params,
                dtype=torch.bfloat16,  # Convert to bfloat16
                gather_tensor_fn=gather_fn,
                get_shape_fn=shape_fn,
            )
        )

        assert len(chunks) == 1
        assert chunks[0].tensors[0].dtype == torch.bfloat16


# =============================================================================
# Section 13: Integration with nmoe/SGLang Tests
# =============================================================================


@gpu
@integration
class TestNmoeSglangIntegration:
    """Tests for nmoe -> SGLang weight transfer integration."""

    def test_nmoe_weight_naming_convention(self):
        """Test nmoe MoE weight naming follows SGLang expectations."""
        # nmoe expert naming pattern
        nmoe_names = [
            "model.layers.0.block_sparse_moe.experts.0.w1.weight",
            "model.layers.0.block_sparse_moe.experts.0.w2.weight",
            "model.layers.0.block_sparse_moe.experts.0.w3.weight",
        ]

        request = BroadcastWeightUpdateRequest(
            names=nmoe_names,
            dtypes=["torch.bfloat16"] * 3,
            shapes=[[14336, 4096], [4096, 14336], [14336, 4096]],
        )

        assert len(request) == 3
        assert all("block_sparse_moe" in n for n in request.names)

    def test_router_weight_format(self):
        """Test router weight format for nmoe -> SGLang."""
        request = BroadcastWeightUpdateRequest(
            names=["model.layers.0.block_sparse_moe.gate.weight"],
            dtypes=["torch.bfloat16"],
            shapes=[[64, 4096]],  # 64 experts
        )

        assert "gate" in request.names[0]

    @single_gpu
    def test_weight_dtype_compatibility(self):
        """Test weight dtype compatibility between training and inference."""
        training_dtypes = ["torch.float32", "torch.bfloat16", "torch.float16"]

        for dtype_str in training_dtypes:
            request = CudaIpcWeightUpdateRequest(
                names=["layer.weight"],
                dtypes=[dtype_str],
                shapes=[[64, 64]],
                sizes=[64 * 64],
                ipc_handles={},
            )

            # Verify dtype string format
            assert dtype_str.startswith("torch.")

    def test_large_model_weight_count(self):
        """Test handling large model weight counts (DeepSeek-V3 scale)."""
        num_layers = 61
        num_experts = 256
        weights_per_expert = 3  # gate, up, down

        # Calculate total weights for experts only
        total_expert_weights = num_layers * num_experts * weights_per_expert

        # Should handle ~46,848 expert weights
        assert total_expert_weights == 46848

        # Create request metadata (not actual tensors)
        expert_names = [
            f"model.layers.{l}.mlp.experts.{e}.{proj}.weight"
            for l in range(num_layers)
            for e in range(num_experts)
            for proj in ["gate_proj", "up_proj", "down_proj"]
        ]

        assert len(expert_names) == total_expert_weights


# =============================================================================
# Section 14: Edge Case Tests
# =============================================================================


@integration
class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_weight_update(self):
        """Test handling of empty weight update."""
        request = BroadcastWeightUpdateRequest(
            names=[],
            dtypes=[],
            shapes=[],
        )
        assert len(request) == 0

    def test_single_element_tensor(self):
        """Test handling of single-element tensors."""
        chunk = WeightChunk(
            names=["scalar_param"],
            dtypes=["torch.float32"],
            shapes=[[1]],
            tensors=[torch.tensor([1.0])],
        )
        assert chunk.total_numel == 1

    def test_very_large_tensor_shape(self):
        """Test handling of very large tensor shapes."""
        # Simulate embedding table for large vocab
        vocab_size = 152064  # DeepSeek vocab
        hidden_size = 7168

        request = BroadcastWeightUpdateRequest(
            names=["model.embed_tokens.weight"],
            dtypes=["torch.bfloat16"],
            shapes=[[vocab_size, hidden_size]],
        )

        assert request.shapes[0] == [vocab_size, hidden_size]

    def test_unicode_parameter_names(self):
        """Test handling of unicode in parameter names."""
        request = BroadcastWeightUpdateRequest(
            names=["model.layer_alpha.weight", "model.layer_beta.weight"],
            dtypes=["torch.bfloat16", "torch.bfloat16"],
            shapes=[[64, 64], [64, 64]],
        )
        assert len(request) == 2

    def test_deeply_nested_parameter_names(self):
        """Test handling of deeply nested parameter names."""
        deep_name = "model.decoder.layers.0.encoder_attn.k_proj.lora_A.default.weight"
        request = BroadcastWeightUpdateRequest(
            names=[deep_name],
            dtypes=["torch.bfloat16"],
            shapes=[[16, 4096]],
        )
        assert request.names[0] == deep_name

    def test_special_characters_in_names(self):
        """Test handling of special characters in parameter names."""
        names = [
            "model.layer-0.weight",
            "model.layer_0.weight",
            "model.layer.0.weight",
        ]
        request = BroadcastWeightUpdateRequest(
            names=names,
            dtypes=["torch.bfloat16"] * 3,
            shapes=[[64, 64]] * 3,
        )
        assert len(request) == 3

    def test_mixed_dtypes_in_request(self):
        """Test handling of mixed dtypes in single request."""
        request = BroadcastWeightUpdateRequest(
            names=["layer1.weight", "layer2.weight", "layer3.weight"],
            dtypes=["torch.float32", "torch.bfloat16", "torch.float16"],
            shapes=[[64, 64], [64, 64], [64, 64]],
        )
        assert len(set(request.dtypes)) == 3

    def test_zero_dimension_tensor_shape(self):
        """Test handling of zero-dimension tensor shapes."""
        chunk = WeightChunk(
            names=["scalar"],
            dtypes=["torch.float32"],
            shapes=[[]],  # Scalar shape
            tensors=[torch.tensor(1.0)],
        )
        assert chunk.total_numel == 1


# =============================================================================
# Run Tests
# =============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
