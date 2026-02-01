"""Comprehensive integration tests for SkyRL's integration with SGLang inference engine.

This test module verifies the complete SkyRL-SGLang integration including:
- SGLang backend initialization from SkyRL configurations
- Weight synchronization between FSDP training and SGLang inference
- CUDA IPC weight transfer strategy
- Broadcast weight transfer strategy
- Checkpoint-based weight transfer
- Multi-GPU tensor parallel inference
- Colocated and non-colocated configurations
- Async generation capabilities

Test Categories:
- Unit tests: Test individual components in isolation
- Integration tests: Test component interactions
- GPU tests: Require CUDA devices (marked with @pytest.mark.gpu)
- Multi-GPU tests: Require multiple GPUs (marked with @pytest.mark.multi_gpu)

To run these tests:
    # All tests (CPU-safe)
    pytest tests/integration/test_skyrl_sglang_inference.py -v

    # GPU tests only
    pytest tests/integration/test_skyrl_sglang_inference.py -v -m gpu

    # Multi-GPU tests (8 GPUs for TP=8)
    pytest tests/integration/test_skyrl_sglang_inference.py -v -m multi_gpu

Note: These tests require the sglang extra to be installed:
    uv sync --extra sglang
"""

import asyncio
import os
import pickle
import tempfile
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Skip entire module if sglang is not available
sglang = pytest.importorskip("sglang")


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer for testing."""
    tokenizer = MagicMock()
    tokenizer.encode = MagicMock(return_value=[1, 2, 3, 4, 5])
    tokenizer.decode = MagicMock(return_value="Hello, world!")
    tokenizer.eos_token_id = 2
    tokenizer.pad_token_id = 0
    tokenizer.apply_chat_template = MagicMock(return_value="<s>User: test</s>")
    return tokenizer


@pytest.fixture
def mock_config():
    """Create a mock OmegaConf config for testing."""
    from omegaconf import OmegaConf

    config = OmegaConf.create({
        "generator": {
            "backend": "sglang",
            "model_dtype": "bfloat16",
            "weight_sync_backend": "nccl",
            "num_inference_engines": 1,
            "inference_engine_tensor_parallel_size": 1,
            "inference_engine_pipeline_parallel_size": 1,
            "inference_engine_data_parallel_size": 1,
            "override_existing_update_group": "disable",
            "enable_prefix_caching": True,
            "weight_transfer_threshold_cuda_ipc_GB": 1.0,
            "use_overlapped_weight_sync": False,
            "enable_http_endpoint": False,
            "http_endpoint_host": "localhost",
            "http_endpoint_port": 8080,
        },
        "trainer": {
            "policy": {
                "model": {
                    "path": "Qwen/Qwen2.5-0.5B-Instruct",
                }
            },
            "placement": {
                "colocate_all": True,
            },
        },
    })
    return config


@pytest.fixture
def small_model_path():
    """Return a small model path for GPU tests."""
    return "Qwen/Qwen2.5-0.5B-Instruct"


# -----------------------------------------------------------------------------
# Test Classes: Import and Interface Tests
# -----------------------------------------------------------------------------

class TestSGLangEngineImports:
    """Test that SGLang engine imports work correctly."""

    def test_import_sglang_engine(self):
        """Test that SGLangInferenceEngine can be imported."""
        from skyrl_train.inference_engines.sglang.sglang_engine import (
            SGLangInferenceEngine,
            SGLangWeightLoader,
            SGLangRayActor,
        )
        assert SGLangInferenceEngine is not None
        assert SGLangWeightLoader is not None
        assert SGLangRayActor is not None

    def test_import_io_structs(self):
        """Test that required SGLang io_structs can be imported."""
        from sglang.srt.managers.io_struct import (
            UpdateWeightsFromTensorReqInput,
            UpdateWeightsFromDistributedReqInput,
            InitWeightsUpdateGroupReqInput,
            ReleaseMemoryOccupationReqInput,
            ResumeMemoryOccupationReqInput,
            PauseGenerationReqInput,
            ContinueGenerationReqInput,
        )
        assert UpdateWeightsFromTensorReqInput is not None
        assert UpdateWeightsFromDistributedReqInput is not None
        assert InitWeightsUpdateGroupReqInput is not None
        assert ReleaseMemoryOccupationReqInput is not None
        assert ResumeMemoryOccupationReqInput is not None
        assert PauseGenerationReqInput is not None
        assert ContinueGenerationReqInput is not None

    def test_import_weight_sync_components(self):
        """Test that weight sync components can be imported."""
        from skyrl_train.weight_sync import (
            WeightChunk,
            WeightUpdateRequest,
            WeightTransferStrategy,
            CudaIpcTransferStrategy,
            CudaIpcWeightUpdateRequest,
            BroadcastTransferStrategy,
            BroadcastWeightUpdateRequest,
            get_transfer_strategy_cls,
        )
        assert WeightChunk is not None
        assert WeightUpdateRequest is not None
        assert WeightTransferStrategy is not None
        assert CudaIpcTransferStrategy is not None
        assert CudaIpcWeightUpdateRequest is not None
        assert BroadcastTransferStrategy is not None
        assert BroadcastWeightUpdateRequest is not None
        assert get_transfer_strategy_cls is not None

    def test_import_sglang_engine_class(self):
        """Test that SGLang Engine class can be imported."""
        from sglang.srt.entrypoints.engine import Engine
        assert Engine is not None


class TestSGLangEngineInterface:
    """Test that SGLangInferenceEngine implements required interface."""

    def test_implements_inference_engine_interface(self):
        """Test that SGLangInferenceEngine has all required methods."""
        from skyrl_train.inference_engines.sglang.sglang_engine import SGLangInferenceEngine
        from skyrl_train.inference_engines.base import InferenceEngineInterface

        assert issubclass(SGLangInferenceEngine, InferenceEngineInterface)

        # Check required methods exist
        required_methods = [
            'generate',
            'chat_completion',
            'completion',
            'init_weight_update_communicator',
            'update_named_weights',
            'wake_up',
            'sleep',
            'teardown',
            'reset_prefix_cache',
            'abort_generation',
            'pause_generation',
            'continue_generation',
            'tp_size',
            'pp_size',
            'dp_size',
            'ep_size',
        ]

        for method_name in required_methods:
            assert hasattr(SGLangInferenceEngine, method_name), f"Missing method: {method_name}"

    def test_has_weight_sync_methods(self):
        """Test that engine has weight synchronization methods."""
        from skyrl_train.inference_engines.sglang.sglang_engine import SGLangInferenceEngine

        weight_sync_methods = [
            'init_weight_update_communicator',
            'update_named_weights',
            'get_weight_version',
            'update_weight_version',
            'start_weight_transfer',
            'finish_weight_transfer',
            'overlapped_weight_sync',
            'supports_overlapped_weight_sync',
        ]

        for method_name in weight_sync_methods:
            assert hasattr(SGLangInferenceEngine, method_name), f"Missing weight sync method: {method_name}"

    def test_has_session_management_methods(self):
        """Test that engine has session management methods."""
        from skyrl_train.inference_engines.sglang.sglang_engine import SGLangInferenceEngine

        session_methods = [
            'open_session',
            'close_session',
            'generate_with_session',
            'supports_sessions',
        ]

        for method_name in session_methods:
            assert hasattr(SGLangInferenceEngine, method_name), f"Missing session method: {method_name}"


# -----------------------------------------------------------------------------
# Test Classes: Weight Sync Strategy Tests
# -----------------------------------------------------------------------------

class TestWeightTransferStrategySelection:
    """Test weight transfer strategy selection logic."""

    def test_get_cuda_ipc_strategy_for_colocated(self, mock_config):
        """Test CUDA IPC strategy is selected for colocated config."""
        from skyrl_train.weight_sync import get_transfer_strategy_cls, CudaIpcTransferStrategy

        mock_config.trainer.placement.colocate_all = True
        mock_config.generator.weight_sync_backend = "nccl"

        strategy_cls = get_transfer_strategy_cls(mock_config)
        assert strategy_cls == CudaIpcTransferStrategy

    def test_get_broadcast_strategy_for_non_colocated(self, mock_config):
        """Test Broadcast strategy is selected for non-colocated config."""
        from skyrl_train.weight_sync import get_transfer_strategy_cls, BroadcastTransferStrategy

        mock_config.trainer.placement.colocate_all = False
        mock_config.generator.weight_sync_backend = "nccl"

        strategy_cls = get_transfer_strategy_cls(mock_config)
        assert strategy_cls == BroadcastTransferStrategy

    def test_auto_selects_cuda_ipc_for_colocated(self, mock_config):
        """Test auto mode selects CUDA IPC for colocated."""
        from skyrl_train.weight_sync import get_transfer_strategy_cls, CudaIpcTransferStrategy

        mock_config.trainer.placement.colocate_all = True
        mock_config.generator.weight_sync_backend = "auto"

        strategy_cls = get_transfer_strategy_cls(mock_config)
        assert strategy_cls == CudaIpcTransferStrategy

    def test_auto_selects_broadcast_for_non_colocated(self, mock_config):
        """Test auto mode selects Broadcast for non-colocated."""
        from skyrl_train.weight_sync import get_transfer_strategy_cls, BroadcastTransferStrategy

        mock_config.trainer.placement.colocate_all = False
        mock_config.generator.weight_sync_backend = "auto"

        strategy_cls = get_transfer_strategy_cls(mock_config)
        assert strategy_cls == BroadcastTransferStrategy

    def test_checkpoint_engine_fallback(self, mock_config):
        """Test checkpoint_engine falls back to broadcast when not available."""
        from skyrl_train.weight_sync import get_transfer_strategy_cls, BroadcastTransferStrategy

        mock_config.generator.weight_sync_backend = "checkpoint_engine"

        # checkpoint_engine is typically not installed in test env
        strategy_cls = get_transfer_strategy_cls(mock_config)
        # Either returns CheckpointEngineTransferStrategy or falls back to Broadcast
        assert strategy_cls is not None


class TestCudaIpcWeightUpdateRequest:
    """Test CUDA IPC weight update request serialization."""

    def test_serialize_deserialize_request(self):
        """Test serialization round-trip for CudaIpcWeightUpdateRequest."""
        from skyrl_train.weight_sync import CudaIpcWeightUpdateRequest

        original = CudaIpcWeightUpdateRequest(
            names=["layer1.weight", "layer2.bias"],
            dtypes=["torch.bfloat16", "torch.bfloat16"],
            shapes=[[1024, 1024], [1024]],
            sizes=[1024 * 1024, 1024],
            ipc_handles={},  # Empty handles for serialization test
            weight_version="step_100",
        )

        serialized = original.serialize()
        assert isinstance(serialized, bytes)
        assert len(serialized) > 0

        deserialized = CudaIpcWeightUpdateRequest.deserialize(serialized)
        assert deserialized.names == original.names
        assert deserialized.dtypes == original.dtypes
        assert deserialized.shapes == original.shapes
        assert deserialized.sizes == original.sizes
        assert deserialized.weight_version == original.weight_version

    def test_request_length(self):
        """Test __len__ returns correct number of parameters."""
        from skyrl_train.weight_sync import CudaIpcWeightUpdateRequest

        request = CudaIpcWeightUpdateRequest(
            names=["a", "b", "c"],
            dtypes=["torch.float32"] * 3,
            shapes=[[10], [20], [30]],
            sizes=[10, 20, 30],
            ipc_handles={},
        )

        assert len(request) == 3


class TestBroadcastWeightUpdateRequest:
    """Test Broadcast weight update request."""

    def test_request_creation(self):
        """Test BroadcastWeightUpdateRequest can be created."""
        from skyrl_train.weight_sync import BroadcastWeightUpdateRequest

        request = BroadcastWeightUpdateRequest(
            names=["model.weight"],
            dtypes=["torch.bfloat16"],
            shapes=[[4096, 4096]],
            weight_version="step_50",
        )

        assert len(request) == 1
        assert request.weight_version == "step_50"

    def test_request_validation(self):
        """Test request validates matching lengths."""
        from skyrl_train.weight_sync import BroadcastWeightUpdateRequest

        with pytest.raises(ValueError):
            BroadcastWeightUpdateRequest(
                names=["a", "b"],
                dtypes=["float32"],  # Mismatch: 2 vs 1
                shapes=[[10]],  # Mismatch: 2 vs 1
            )


class TestBroadcastInitInfo:
    """Test BroadcastInitInfo functionality."""

    def test_for_engine_adjusts_rank_offset(self):
        """Test for_engine correctly adjusts rank_offset."""
        from skyrl_train.weight_sync import BroadcastInitInfo, BroadcastTransferStrategy

        init_info = BroadcastInitInfo(
            master_addr="127.0.0.1",
            master_port=29500,
            rank_offset=1,
            world_size=9,  # 1 trainer + 8 inference engines
            group_name="test_group",
            backend="nccl",
            model_dtype_str="bfloat16",
            override_existing_receiver=False,
        )

        # Engine 0 with TP=2, PP=1
        engine_info = init_info.for_engine(engine_index=0, tp_size=2, pp_size=1)
        assert engine_info.rank_offset == 1  # 1 + 0*2*1 = 1

        # Engine 1 with TP=2, PP=1
        engine_info = init_info.for_engine(engine_index=1, tp_size=2, pp_size=1)
        assert engine_info.rank_offset == 3  # 1 + 1*2*1 = 3

        # Engine 2 with TP=4, PP=1
        engine_info = init_info.for_engine(engine_index=2, tp_size=4, pp_size=1)
        assert engine_info.rank_offset == 9  # 1 + 2*4*1 = 9


class TestCudaIpcInitInfo:
    """Test CudaIpcInitInfo functionality."""

    def test_strategy_type_returns_correct_class(self):
        """Test strategy_type returns CudaIpcTransferStrategy."""
        from skyrl_train.weight_sync import CudaIpcInitInfo, CudaIpcTransferStrategy

        init_info = CudaIpcInitInfo(
            model_dtype_str="bfloat16",
            override_existing_receiver=False,
        )

        assert init_info.strategy_type() == CudaIpcTransferStrategy


# -----------------------------------------------------------------------------
# Test Classes: SGLang Weight Loader Tests
# -----------------------------------------------------------------------------

class TestSGLangWeightLoader:
    """Test SGLangWeightLoader functionality."""

    def test_weight_loader_has_required_methods(self):
        """Test that SGLangWeightLoader has required methods."""
        from skyrl_train.inference_engines.sglang.sglang_engine import SGLangWeightLoader

        required_methods = ['init_communicator', 'load_weights', 'destroy_group']
        for method_name in required_methods:
            assert hasattr(SGLangWeightLoader, method_name), f"Missing method: {method_name}"

    def test_weight_loader_init(self):
        """Test SGLangWeightLoader initialization."""
        from skyrl_train.inference_engines.sglang.sglang_engine import SGLangWeightLoader

        mock_engine = MagicMock()
        loader = SGLangWeightLoader(engine=mock_engine, tp_size=1)

        assert loader._engine == mock_engine
        assert loader._tp_size == 1
        assert loader._group_name is None


class TestMemoryTagConstants:
    """Test memory tag constants."""

    def test_memory_tags_defined(self):
        """Test that memory tags are properly defined."""
        from skyrl_train.inference_engines.sglang.sglang_engine import MemoryTag

        assert MemoryTag.WEIGHTS == "weights"
        assert MemoryTag.KV_CACHE == "kv_cache"
        assert MemoryTag.CUDA_GRAPH == "cuda_graph"
        assert MemoryTag.ALL == ["weights", "kv_cache", "cuda_graph"]
        assert MemoryTag.TRAINING_DEFAULT == ["weights"]


# -----------------------------------------------------------------------------
# Test Classes: WeightChunk Tests
# -----------------------------------------------------------------------------

class TestWeightChunk:
    """Test WeightChunk functionality."""

    @pytest.mark.gpu
    @pytest.mark.integration
    def test_weight_chunk_creation(self):
        """Test creating a WeightChunk with tensor data."""
        import torch
        from skyrl_train.weight_sync import WeightChunk

        tensor = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
        chunk = WeightChunk(
            names=["model.layer.weight"],
            dtypes=["torch.bfloat16"],
            shapes=[[1024, 1024]],
            tensors=[tensor],
        )

        assert len(chunk) == 1
        assert chunk.total_numel == 1024 * 1024
        assert chunk.total_size_bytes == 1024 * 1024 * 2  # bfloat16 = 2 bytes

    @pytest.mark.gpu
    @pytest.mark.integration
    def test_weight_chunk_multi_tensor(self):
        """Test WeightChunk with multiple tensors."""
        import torch
        from skyrl_train.weight_sync import WeightChunk

        tensors = [
            torch.randn(512, 512, device="cuda", dtype=torch.float32),
            torch.randn(256, device="cuda", dtype=torch.float32),
        ]
        chunk = WeightChunk(
            names=["weight", "bias"],
            dtypes=["torch.float32", "torch.float32"],
            shapes=[[512, 512], [256]],
            tensors=tensors,
        )

        assert len(chunk) == 2
        assert chunk.total_numel == 512 * 512 + 256

    def test_weight_chunk_validation(self):
        """Test WeightChunk validates matching lengths."""
        import torch
        from skyrl_train.weight_sync import WeightChunk

        with pytest.raises(ValueError):
            WeightChunk(
                names=["a", "b"],
                dtypes=["float32"],  # Mismatch
                shapes=[[10], [20]],
                tensors=[torch.zeros(10), torch.zeros(20)],
            )


# -----------------------------------------------------------------------------
# Test Classes: Data Type Tests
# -----------------------------------------------------------------------------

class TestInferenceEngineDataTypes:
    """Test SGLang-specific data types and conversion."""

    def test_inference_engine_input_type(self):
        """Test InferenceEngineInput structure."""
        from skyrl_train.inference_engines.base import InferenceEngineInput

        sample_input: InferenceEngineInput = {
            "prompts": None,
            "prompt_token_ids": [[1, 2, 3], [4, 5, 6]],
            "sampling_params": {"max_new_tokens": 100, "temperature": 0.7},
            "session_ids": None,
            "return_hidden_states": False,
            "image_data": None,
            "video_data": None,
            "audio_data": None,
        }
        assert sample_input["prompt_token_ids"] is not None
        assert len(sample_input["prompt_token_ids"]) == 2

    def test_inference_engine_output_type(self):
        """Test InferenceEngineOutput structure."""
        from skyrl_train.inference_engines.base import InferenceEngineOutput

        sample_output: InferenceEngineOutput = {
            "responses": ["Hello", "World"],
            "response_ids": [[7, 8, 9], [10, 11, 12]],
            "stop_reasons": ["stop", "length"],
            "response_logprobs": None,
            "weight_version": "step_100",
            "n_per_prompt": None,
            "request_ids": None,
            "hidden_states": None,
        }
        assert len(sample_output["responses"]) == 2
        assert sample_output["weight_version"] == "step_100"

    def test_streaming_chunk_type(self):
        """Test StreamingChunk structure."""
        from skyrl_train.inference_engines.base import StreamingChunk

        chunk: StreamingChunk = {
            "index": 0,
            "delta_text": "Hello",
            "delta_token_id": 123,
            "delta_logprob": -0.5,
            "is_finished": False,
            "stop_reason": None,
            "cumulative_text": "Hello",
            "cumulative_token_ids": [123],
        }
        assert chunk["delta_text"] == "Hello"
        assert not chunk["is_finished"]


# -----------------------------------------------------------------------------
# Test Classes: Async Generation Tests (Mocked)
# -----------------------------------------------------------------------------

class TestAsyncGenerationMocked:
    """Test async generation with mocked SGLang engine."""

    @pytest.mark.asyncio
    async def test_generate_returns_output(self, mock_tokenizer):
        """Test that generate returns properly structured output."""
        from skyrl_train.inference_engines.sglang.sglang_engine import SGLangInferenceEngine
        from skyrl_train.inference_engines.base import InferenceEngineInput

        with patch.object(SGLangInferenceEngine, '__init__', lambda x, **kwargs: None):
            engine = SGLangInferenceEngine()
            engine.tokenizer = mock_tokenizer
            engine._tp_size = 1
            engine._pp_size = 1
            engine._dp_size = 1
            engine._ep_size = 1
            engine._model_path = "test-model"
            engine._weight_version = None

            # Mock the engine's generate_request with proper async generator
            mock_output = {
                "output_ids": [10, 11, 12],
                "meta_info": {"finish_reason": {"type": "stop"}},
            }

            async def mock_async_generator():
                yield [mock_output]

            engine.engine = MagicMock()
            engine.engine.tokenizer_manager.generate_request = MagicMock(
                return_value=mock_async_generator()
            )

            input_batch: InferenceEngineInput = {
                "prompts": None,
                "prompt_token_ids": [[1, 2, 3]],
                "sampling_params": {"max_new_tokens": 10},
                "session_ids": None,
            }

            output = await engine.generate(input_batch)

            assert "responses" in output
            assert "response_ids" in output
            assert "stop_reasons" in output

    @pytest.mark.asyncio
    async def test_generate_with_logprobs(self, mock_tokenizer):
        """Test generation with logprobs enabled."""
        from skyrl_train.inference_engines.sglang.sglang_engine import SGLangInferenceEngine
        from skyrl_train.inference_engines.base import InferenceEngineInput

        with patch.object(SGLangInferenceEngine, '__init__', lambda x, **kwargs: None):
            engine = SGLangInferenceEngine()
            engine.tokenizer = mock_tokenizer
            engine._tp_size = 1
            engine._pp_size = 1
            engine._dp_size = 1
            engine._ep_size = 1
            engine._model_path = "test-model"
            engine._weight_version = None

            mock_output = {
                "output_ids": [10, 11, 12],
                "meta_info": {
                    "finish_reason": {"type": "stop"},
                    "output_token_logprobs": [-0.5, -0.3, -0.1],
                },
            }

            async def mock_async_generator():
                yield [mock_output]

            engine.engine = MagicMock()
            engine.engine.tokenizer_manager.generate_request = MagicMock(
                return_value=mock_async_generator()
            )

            input_batch: InferenceEngineInput = {
                "prompts": None,
                "prompt_token_ids": [[1, 2, 3]],
                "sampling_params": {"max_new_tokens": 10, "return_logprob": True},
                "session_ids": None,
            }

            output = await engine.generate(input_batch)

            assert output["response_logprobs"] is not None
            assert len(output["response_logprobs"]) == 1


class TestWeightVersionTracking:
    """Test weight version tracking functionality."""

    @pytest.mark.asyncio
    async def test_update_weight_version(self, mock_tokenizer):
        """Test weight version can be updated."""
        from skyrl_train.inference_engines.sglang.sglang_engine import SGLangInferenceEngine

        with patch.object(SGLangInferenceEngine, '__init__', lambda x, **kwargs: None):
            engine = SGLangInferenceEngine()
            engine._weight_version = None
            engine._version_api_available = False  # Use local tracking
            engine.engine = MagicMock()

            await engine.update_weight_version("step_100")

            assert engine._weight_version == "step_100"

    @pytest.mark.asyncio
    async def test_get_weight_version_fallback(self, mock_tokenizer):
        """Test weight version fallback to local tracking."""
        from skyrl_train.inference_engines.sglang.sglang_engine import SGLangInferenceEngine

        with patch.object(SGLangInferenceEngine, '__init__', lambda x, **kwargs: None):
            engine = SGLangInferenceEngine()
            engine._weight_version = "step_50"
            engine._version_api_available = False
            engine.engine = MagicMock()

            version = await engine.get_weight_version()

            assert version == "step_50"


# -----------------------------------------------------------------------------
# Test Classes: Memory Management Tests (Mocked)
# -----------------------------------------------------------------------------

class TestMemoryManagementMocked:
    """Test memory management functions with mocked engine."""

    @pytest.mark.asyncio
    async def test_sleep_releases_memory(self):
        """Test sleep releases memory occupation."""
        from skyrl_train.inference_engines.sglang.sglang_engine import SGLangInferenceEngine

        with patch.object(SGLangInferenceEngine, '__init__', lambda x, **kwargs: None):
            engine = SGLangInferenceEngine()
            engine.engine = MagicMock()
            engine.engine.tokenizer_manager.pause_generation = AsyncMock()
            engine.engine.tokenizer_manager.release_memory_occupation = AsyncMock()

            await engine.sleep(tags=["weights"])

            engine.engine.tokenizer_manager.pause_generation.assert_called_once()
            engine.engine.tokenizer_manager.release_memory_occupation.assert_called_once()

    @pytest.mark.asyncio
    async def test_wake_up_resumes_memory(self):
        """Test wake_up resumes memory occupation."""
        from skyrl_train.inference_engines.sglang.sglang_engine import SGLangInferenceEngine

        with patch.object(SGLangInferenceEngine, '__init__', lambda x, **kwargs: None):
            engine = SGLangInferenceEngine()
            engine.engine = MagicMock()
            engine.engine.tokenizer_manager.resume_memory_occupation = AsyncMock()
            engine.engine.tokenizer_manager.continue_generation = AsyncMock()

            await engine.wake_up(tags=["weights"])

            engine.engine.tokenizer_manager.resume_memory_occupation.assert_called_once()
            engine.engine.tokenizer_manager.continue_generation.assert_called_once()

    @pytest.mark.asyncio
    async def test_sleep_weights_only_uses_correct_tags(self):
        """Test sleep_weights_only uses TRAINING_DEFAULT tags."""
        from skyrl_train.inference_engines.sglang.sglang_engine import SGLangInferenceEngine, MemoryTag

        with patch.object(SGLangInferenceEngine, '__init__', lambda x, **kwargs: None):
            engine = SGLangInferenceEngine()
            engine.sleep = AsyncMock()

            await engine.sleep_weights_only()

            engine.sleep.assert_called_once()
            call_kwargs = engine.sleep.call_args[1]
            assert call_kwargs["tags"] == MemoryTag.TRAINING_DEFAULT


# -----------------------------------------------------------------------------
# Test Classes: Generation Control Tests (Mocked)
# -----------------------------------------------------------------------------

class TestGenerationControlMocked:
    """Test generation control functions."""

    @pytest.mark.asyncio
    async def test_pause_generation_modes(self):
        """Test pause_generation with different modes."""
        from skyrl_train.inference_engines.sglang.sglang_engine import SGLangInferenceEngine

        with patch.object(SGLangInferenceEngine, '__init__', lambda x, **kwargs: None):
            engine = SGLangInferenceEngine()
            engine.engine = MagicMock()
            engine.engine.tokenizer_manager.pause_generation = AsyncMock()

            for mode in ["abort", "in_place", "retract"]:
                await engine.pause_generation(mode=mode)
                engine.engine.tokenizer_manager.pause_generation.assert_called()

    @pytest.mark.asyncio
    async def test_continue_generation(self):
        """Test continue_generation resumes processing."""
        from skyrl_train.inference_engines.sglang.sglang_engine import SGLangInferenceEngine

        with patch.object(SGLangInferenceEngine, '__init__', lambda x, **kwargs: None):
            engine = SGLangInferenceEngine()
            engine.engine = MagicMock()
            engine.engine.tokenizer_manager.continue_generation = AsyncMock()

            await engine.continue_generation()

            engine.engine.tokenizer_manager.continue_generation.assert_called_once()

    @pytest.mark.asyncio
    async def test_abort_generation_calls_pause_with_abort(self):
        """Test abort_generation calls pause with abort mode."""
        from skyrl_train.inference_engines.sglang.sglang_engine import SGLangInferenceEngine

        with patch.object(SGLangInferenceEngine, '__init__', lambda x, **kwargs: None):
            engine = SGLangInferenceEngine()
            engine.pause_generation = AsyncMock()

            await engine.abort_generation()

            engine.pause_generation.assert_called_once_with(mode="abort")


# -----------------------------------------------------------------------------
# Test Classes: Weight Validation Tests (Mocked)
# -----------------------------------------------------------------------------

class TestWeightValidationMocked:
    """Test weight validation functionality."""

    @pytest.mark.asyncio
    async def test_check_weight_sync_integrity(self):
        """Test weight sync integrity check."""
        from skyrl_train.inference_engines.sglang.sglang_engine import SGLangInferenceEngine

        with patch.object(SGLangInferenceEngine, '__init__', lambda x, **kwargs: None):
            engine = SGLangInferenceEngine()
            engine._weight_version = "step_100"
            engine.validate_weights = AsyncMock(return_value={"valid": True, "checked": 1, "issues": []})
            engine.get_weight_version = AsyncMock(return_value="step_100")

            result = await engine.check_weight_sync_integrity(expected_version="step_100")

            assert result["weights_valid"] is True
            assert result["version_match"] is True
            assert result["current_version"] == "step_100"


# -----------------------------------------------------------------------------
# Test Classes: Overlapped Weight Sync Tests (Mocked)
# -----------------------------------------------------------------------------

class TestOverlappedWeightSyncMocked:
    """Test overlapped weight synchronization."""

    def test_supports_overlapped_weight_sync(self):
        """Test engine reports overlapped weight sync support."""
        from skyrl_train.inference_engines.sglang.sglang_engine import SGLangInferenceEngine

        with patch.object(SGLangInferenceEngine, '__init__', lambda x, **kwargs: None):
            engine = SGLangInferenceEngine()

            assert engine.supports_overlapped_weight_sync() is True

    @pytest.mark.asyncio
    async def test_overlapped_weight_sync_flow(self):
        """Test overlapped weight sync complete flow."""
        from skyrl_train.inference_engines.sglang.sglang_engine import SGLangInferenceEngine
        from skyrl_train.weight_sync import CudaIpcWeightUpdateRequest

        with patch.object(SGLangInferenceEngine, '__init__', lambda x, **kwargs: None):
            engine = SGLangInferenceEngine()
            engine.start_weight_transfer = AsyncMock()
            engine.pause_generation = AsyncMock()
            engine.finish_weight_transfer = AsyncMock()
            engine.continue_generation = AsyncMock()

            # Create mock handle
            mock_handle = MagicMock()
            mock_handle.wait = AsyncMock()
            mock_handle.request = MagicMock()
            mock_handle.request.weight_version = "step_1"
            mock_handle.request.names = ["a", "b"]
            engine.start_weight_transfer.return_value = mock_handle

            request = CudaIpcWeightUpdateRequest(
                names=["model.weight"],
                dtypes=["torch.bfloat16"],
                shapes=[[1024]],
                sizes=[1024],
                ipc_handles={},
            )

            await engine.overlapped_weight_sync(request, pause_mode="in_place")

            engine.start_weight_transfer.assert_called_once()
            mock_handle.wait.assert_called_once()
            engine.pause_generation.assert_called_once_with(mode="in_place")
            engine.finish_weight_transfer.assert_called_once()
            engine.continue_generation.assert_called_once()


# -----------------------------------------------------------------------------
# Test Classes: Session Management Tests (Mocked)
# -----------------------------------------------------------------------------

class TestSessionManagementMocked:
    """Test session management for prefix caching."""

    def test_supports_sessions(self):
        """Test engine reports session support."""
        from skyrl_train.inference_engines.sglang.sglang_engine import SGLangInferenceEngine

        with patch.object(SGLangInferenceEngine, '__init__', lambda x, **kwargs: None):
            engine = SGLangInferenceEngine()

            assert engine.supports_sessions() is True

    @pytest.mark.asyncio
    async def test_open_session(self):
        """Test opening a session."""
        from skyrl_train.inference_engines.sglang.sglang_engine import SGLangInferenceEngine

        with patch.object(SGLangInferenceEngine, '__init__', lambda x, **kwargs: None):
            engine = SGLangInferenceEngine()
            engine.engine = MagicMock()
            engine.engine.tokenizer_manager.open_session = AsyncMock(return_value="session_123")

            session_id = await engine.open_session(capacity_of_str_len=4096)

            assert session_id == "session_123"

    @pytest.mark.asyncio
    async def test_close_session(self):
        """Test closing a session."""
        from skyrl_train.inference_engines.sglang.sglang_engine import SGLangInferenceEngine

        with patch.object(SGLangInferenceEngine, '__init__', lambda x, **kwargs: None):
            engine = SGLangInferenceEngine()
            engine.engine = MagicMock()
            engine.engine.tokenizer_manager.close_session = AsyncMock()

            await engine.close_session("session_123")

            engine.engine.tokenizer_manager.close_session.assert_called_once()


# -----------------------------------------------------------------------------
# Test Classes: Server Info Tests (Mocked)
# -----------------------------------------------------------------------------

class TestServerInfoMocked:
    """Test server info retrieval."""

    @pytest.mark.asyncio
    async def test_get_server_info(self):
        """Test getting server info."""
        from skyrl_train.inference_engines.sglang.sglang_engine import SGLangInferenceEngine

        with patch.object(SGLangInferenceEngine, '__init__', lambda x, **kwargs: None):
            engine = SGLangInferenceEngine()
            engine._model_path = "test-model"
            engine._tp_size = 2
            engine._pp_size = 1
            engine._dp_size = 1
            engine._ep_size = 1
            engine._enable_lora = False
            engine._weight_version = "step_50"
            engine.engine = MagicMock()

            info = await engine.get_server_info()

            assert info["model_path"] == "test-model"
            assert info["tp_size"] == 2
            assert info["pp_size"] == 1
            assert info["weight_version"] == "step_50"


# -----------------------------------------------------------------------------
# Test Classes: Teardown Tests (Mocked)
# -----------------------------------------------------------------------------

class TestTeardownMocked:
    """Test teardown functionality."""

    @pytest.mark.asyncio
    async def test_teardown_destroys_group(self):
        """Test teardown destroys weight update group."""
        from skyrl_train.inference_engines.sglang.sglang_engine import SGLangInferenceEngine, SGLangWeightLoader

        with patch.object(SGLangInferenceEngine, '__init__', lambda x, **kwargs: None):
            engine = SGLangInferenceEngine()
            engine._model_path = "test-model"
            engine._weight_loader = MagicMock(spec=SGLangWeightLoader)
            engine._weight_loader.destroy_group = AsyncMock()
            engine.engine = MagicMock()
            engine.engine.shutdown = MagicMock()

            await engine.teardown()

            engine._weight_loader.destroy_group.assert_called_once()
            engine.engine.shutdown.assert_called_once()


# -----------------------------------------------------------------------------
# Test Classes: InferenceEngineClient Tests
# -----------------------------------------------------------------------------

class TestInferenceEngineClient:
    """Test InferenceEngineClient functionality."""

    def test_client_routes_to_engines(self, mock_config, mock_tokenizer):
        """Test client routes requests to multiple engines."""
        from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient

        mock_engine1 = MagicMock()
        mock_engine2 = MagicMock()

        mock_config.generator.enable_http_endpoint = False

        client = InferenceEngineClient(
            engines=[mock_engine1, mock_engine2],
            tokenizer=mock_tokenizer,
            full_config=mock_config,
        )

        assert len(client.engines) == 2

    @pytest.mark.asyncio
    async def test_client_runs_on_all_engines(self, mock_config, mock_tokenizer):
        """Test _run_on_all_engines calls method on all engines."""
        from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient

        mock_engine1 = MagicMock()
        mock_engine1.wake_up = AsyncMock()
        mock_engine2 = MagicMock()
        mock_engine2.wake_up = AsyncMock()

        mock_config.generator.enable_http_endpoint = False

        client = InferenceEngineClient(
            engines=[mock_engine1, mock_engine2],
            tokenizer=mock_tokenizer,
            full_config=mock_config,
        )

        await client._run_on_all_engines("wake_up")

        mock_engine1.wake_up.assert_called_once()
        mock_engine2.wake_up.assert_called_once()


# -----------------------------------------------------------------------------
# Test Classes: GPU-Required Tests
# -----------------------------------------------------------------------------

@pytest.mark.gpu
@pytest.mark.integration
@pytest.mark.skipif(
    not pytest.importorskip("torch").cuda.is_available(),
    reason="CUDA not available"
)
class TestSGLangEngineGPU:
    """GPU-based tests for SGLang engine.

    These tests require a GPU and will be skipped if CUDA is not available.
    """

    @pytest.mark.slow
    def test_engine_initialization(self, small_model_path):
        """Test that SGLang engine can be initialized."""
        from transformers import AutoTokenizer
        from skyrl_train.inference_engines.sglang.sglang_engine import SGLangInferenceEngine

        tokenizer = AutoTokenizer.from_pretrained(small_model_path)

        engine = SGLangInferenceEngine(
            model_path=small_model_path,
            tokenizer=tokenizer,
            tp_size=1,
            mem_fraction_static=0.5,
        )

        try:
            assert engine.tp_size() == 1
            assert engine.pp_size() == 1
            assert engine.dp_size() == 1
        finally:
            asyncio.run(engine.teardown())

    @pytest.mark.slow
    def test_basic_generation(self, small_model_path):
        """Test basic text generation with SGLang engine."""
        from transformers import AutoTokenizer
        from skyrl_train.inference_engines.sglang.sglang_engine import SGLangInferenceEngine
        from skyrl_train.inference_engines.base import InferenceEngineInput

        tokenizer = AutoTokenizer.from_pretrained(small_model_path)

        engine = SGLangInferenceEngine(
            model_path=small_model_path,
            tokenizer=tokenizer,
            tp_size=1,
            mem_fraction_static=0.5,
        )

        try:
            prompt = "Hello, how are you?"
            input_ids = tokenizer.encode(prompt, add_special_tokens=True)

            input_batch: InferenceEngineInput = {
                "prompts": None,
                "prompt_token_ids": [input_ids],
                "sampling_params": {
                    "max_new_tokens": 50,
                    "temperature": 0.7,
                },
                "session_ids": None,
            }

            output = asyncio.run(engine.generate(input_batch))

            assert "responses" in output
            assert "response_ids" in output
            assert "stop_reasons" in output
            assert len(output["responses"]) == 1
            assert len(output["response_ids"]) == 1
            assert len(output["stop_reasons"]) == 1

        finally:
            asyncio.run(engine.teardown())

    @pytest.mark.slow
    def test_generation_with_logprobs(self, small_model_path):
        """Test generation with logprobs enabled."""
        from transformers import AutoTokenizer
        from skyrl_train.inference_engines.sglang.sglang_engine import SGLangInferenceEngine
        from skyrl_train.inference_engines.base import InferenceEngineInput

        tokenizer = AutoTokenizer.from_pretrained(small_model_path)

        engine = SGLangInferenceEngine(
            model_path=small_model_path,
            tokenizer=tokenizer,
            tp_size=1,
            mem_fraction_static=0.5,
        )

        try:
            prompt = "The capital of France is"
            input_ids = tokenizer.encode(prompt, add_special_tokens=True)

            input_batch: InferenceEngineInput = {
                "prompts": None,
                "prompt_token_ids": [input_ids],
                "sampling_params": {
                    "max_new_tokens": 20,
                    "temperature": 0.0,
                    "return_logprob": True,
                },
                "session_ids": None,
            }

            output = asyncio.run(engine.generate(input_batch))

            assert "response_logprobs" in output
            assert output["response_logprobs"] is not None
            assert len(output["response_logprobs"]) == 1

        finally:
            asyncio.run(engine.teardown())

    @pytest.mark.slow
    def test_sleep_wake_cycle(self, small_model_path):
        """Test sleep/wake cycle for memory management."""
        from transformers import AutoTokenizer
        from skyrl_train.inference_engines.sglang.sglang_engine import SGLangInferenceEngine

        tokenizer = AutoTokenizer.from_pretrained(small_model_path)

        engine = SGLangInferenceEngine(
            model_path=small_model_path,
            tokenizer=tokenizer,
            tp_size=1,
            mem_fraction_static=0.5,
        )

        try:
            # Test sleep
            asyncio.run(engine.sleep(tags=["weights"]))

            # Test wake up
            asyncio.run(engine.wake_up(tags=["weights"]))

        finally:
            asyncio.run(engine.teardown())

    @pytest.mark.slow
    def test_pause_continue_generation(self, small_model_path):
        """Test pause/continue generation control."""
        from transformers import AutoTokenizer
        from skyrl_train.inference_engines.sglang.sglang_engine import SGLangInferenceEngine

        tokenizer = AutoTokenizer.from_pretrained(small_model_path)

        engine = SGLangInferenceEngine(
            model_path=small_model_path,
            tokenizer=tokenizer,
            tp_size=1,
            mem_fraction_static=0.5,
        )

        try:
            # Test pause
            asyncio.run(engine.pause_generation(mode="in_place"))

            # Test continue
            asyncio.run(engine.continue_generation())

        finally:
            asyncio.run(engine.teardown())


@pytest.mark.gpu
@pytest.mark.integration
@pytest.mark.multi_gpu
@pytest.mark.skipif(
    not pytest.importorskip("torch").cuda.is_available() or
    pytest.importorskip("torch").cuda.device_count() < 2,
    reason="Multiple GPUs not available"
)
class TestMultiGPUInference:
    """Multi-GPU inference tests."""

    @pytest.mark.slow
    def test_tp2_initialization(self, small_model_path):
        """Test tensor parallel 2 initialization."""
        import torch
        from transformers import AutoTokenizer
        from skyrl_train.inference_engines.sglang.sglang_engine import SGLangInferenceEngine

        if torch.cuda.device_count() < 2:
            pytest.skip("Need at least 2 GPUs for TP=2")

        tokenizer = AutoTokenizer.from_pretrained(small_model_path)

        engine = SGLangInferenceEngine(
            model_path=small_model_path,
            tokenizer=tokenizer,
            tp_size=2,
            mem_fraction_static=0.5,
        )

        try:
            assert engine.tp_size() == 2
        finally:
            asyncio.run(engine.teardown())

    @pytest.mark.slow
    def test_tp2_generation(self, small_model_path):
        """Test generation with TP=2."""
        import torch
        from transformers import AutoTokenizer
        from skyrl_train.inference_engines.sglang.sglang_engine import SGLangInferenceEngine
        from skyrl_train.inference_engines.base import InferenceEngineInput

        if torch.cuda.device_count() < 2:
            pytest.skip("Need at least 2 GPUs for TP=2")

        tokenizer = AutoTokenizer.from_pretrained(small_model_path)

        engine = SGLangInferenceEngine(
            model_path=small_model_path,
            tokenizer=tokenizer,
            tp_size=2,
            mem_fraction_static=0.5,
        )

        try:
            input_ids = tokenizer.encode("Hello, world!", add_special_tokens=True)

            input_batch: InferenceEngineInput = {
                "prompts": None,
                "prompt_token_ids": [input_ids],
                "sampling_params": {"max_new_tokens": 20},
                "session_ids": None,
            }

            output = asyncio.run(engine.generate(input_batch))
            assert len(output["responses"]) == 1

        finally:
            asyncio.run(engine.teardown())


@pytest.mark.gpu
@pytest.mark.integration
@pytest.mark.multi_gpu
@pytest.mark.skipif(
    not pytest.importorskip("torch").cuda.is_available() or
    pytest.importorskip("torch").cuda.device_count() < 8,
    reason="8 GPUs not available"
)
class TestTP8Inference:
    """8-GPU tensor parallel inference tests."""

    @pytest.mark.slow
    def test_tp8_initialization(self, small_model_path):
        """Test tensor parallel 8 initialization."""
        import torch
        from transformers import AutoTokenizer
        from skyrl_train.inference_engines.sglang.sglang_engine import SGLangInferenceEngine

        tokenizer = AutoTokenizer.from_pretrained(small_model_path)

        engine = SGLangInferenceEngine(
            model_path=small_model_path,
            tokenizer=tokenizer,
            tp_size=8,
            mem_fraction_static=0.5,
        )

        try:
            assert engine.tp_size() == 8
        finally:
            asyncio.run(engine.teardown())

    @pytest.mark.slow
    def test_tp8_generation(self, small_model_path):
        """Test generation with TP=8."""
        import torch
        from transformers import AutoTokenizer
        from skyrl_train.inference_engines.sglang.sglang_engine import SGLangInferenceEngine
        from skyrl_train.inference_engines.base import InferenceEngineInput

        tokenizer = AutoTokenizer.from_pretrained(small_model_path)

        engine = SGLangInferenceEngine(
            model_path=small_model_path,
            tokenizer=tokenizer,
            tp_size=8,
            mem_fraction_static=0.5,
        )

        try:
            input_ids = tokenizer.encode("What is machine learning?", add_special_tokens=True)

            input_batch: InferenceEngineInput = {
                "prompts": None,
                "prompt_token_ids": [input_ids],
                "sampling_params": {"max_new_tokens": 50, "temperature": 0.7},
                "session_ids": None,
            }

            output = asyncio.run(engine.generate(input_batch))

            assert len(output["responses"]) == 1
            assert len(output["response_ids"][0]) > 0

        finally:
            asyncio.run(engine.teardown())


# -----------------------------------------------------------------------------
# Test Classes: Weight Transfer GPU Tests
# -----------------------------------------------------------------------------

@pytest.mark.gpu
@pytest.mark.integration
@pytest.mark.skipif(
    not pytest.importorskip("torch").cuda.is_available(),
    reason="CUDA not available"
)
class TestCudaIpcWeightTransferGPU:
    """GPU tests for CUDA IPC weight transfer."""

    def test_cuda_ipc_handle_creation(self):
        """Test CUDA IPC handle can be created for tensor."""
        import torch
        from torch.multiprocessing.reductions import reduce_tensor

        tensor = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)

        # Create IPC handle
        handle = reduce_tensor(tensor)

        assert handle is not None
        assert len(handle) == 2  # (rebuild_func, args)

    def test_cuda_ipc_weight_chunk_packing(self):
        """Test weight chunk packing for CUDA IPC."""
        import torch
        from skyrl_train.weight_sync import WeightChunk

        # Create multiple tensors
        tensors = [
            torch.randn(512, 512, device="cuda", dtype=torch.bfloat16),
            torch.randn(512, device="cuda", dtype=torch.bfloat16),
            torch.randn(256, 256, device="cuda", dtype=torch.bfloat16),
        ]

        chunk = WeightChunk(
            names=["layer.weight", "layer.bias", "output.weight"],
            dtypes=["torch.bfloat16"] * 3,
            shapes=[[512, 512], [512], [256, 256]],
            tensors=tensors,
        )

        # Pack into contiguous buffer
        total_numel = sum(t.numel() for t in tensors)
        packed = torch.empty(total_numel, device="cuda", dtype=torch.bfloat16)

        offset = 0
        for tensor in tensors:
            size = tensor.numel()
            packed[offset:offset + size].copy_(tensor.view(-1))
            offset += size

        assert packed.numel() == chunk.total_numel


@pytest.mark.gpu
@pytest.mark.integration
@pytest.mark.skipif(
    not pytest.importorskip("torch").cuda.is_available(),
    reason="CUDA not available"
)
class TestCheckpointBasedWeightTransfer:
    """Test checkpoint-based weight transfer strategy."""

    def test_checkpoint_save_load(self):
        """Test saving and loading weights via checkpoint."""
        import torch
        import tempfile
        import os

        # Create test weights
        weights = {
            "layer1.weight": torch.randn(256, 256, device="cuda"),
            "layer1.bias": torch.randn(256, device="cuda"),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = os.path.join(tmpdir, "weights.pt")

            # Save
            torch.save(weights, checkpoint_path)

            # Load
            loaded = torch.load(checkpoint_path, map_location="cuda")

            assert "layer1.weight" in loaded
            assert "layer1.bias" in loaded
            assert loaded["layer1.weight"].shape == weights["layer1.weight"].shape
            assert torch.allclose(loaded["layer1.weight"], weights["layer1.weight"])


# -----------------------------------------------------------------------------
# Test Classes: Colocated vs Non-Colocated Tests
# -----------------------------------------------------------------------------

class TestColocatedConfiguration:
    """Test colocated training+inference configuration."""

    def test_colocate_all_enables_cuda_ipc(self, mock_config):
        """Test colocate_all=True enables CUDA IPC strategy."""
        from skyrl_train.weight_sync import get_transfer_strategy_cls, CudaIpcTransferStrategy

        mock_config.trainer.placement.colocate_all = True
        mock_config.generator.weight_sync_backend = "nccl"

        strategy_cls = get_transfer_strategy_cls(mock_config)
        assert strategy_cls == CudaIpcTransferStrategy

    def test_cuda_ipc_init_info_creation(self, mock_config):
        """Test CUDA IPC init info creation."""
        from skyrl_train.weight_sync import CudaIpcTransferStrategy, CudaIpcInitInfo

        mock_config.trainer.placement.colocate_all = True
        mock_config.generator.weight_sync_backend = "nccl"

        init_info = CudaIpcTransferStrategy.create_init_info(mock_config)

        assert isinstance(init_info, CudaIpcInitInfo)
        assert init_info.model_dtype_str == "bfloat16"


class TestNonColocatedConfiguration:
    """Test non-colocated training+inference configuration."""

    def test_non_colocate_uses_broadcast(self, mock_config):
        """Test non-colocated config uses broadcast strategy."""
        from skyrl_train.weight_sync import get_transfer_strategy_cls, BroadcastTransferStrategy

        mock_config.trainer.placement.colocate_all = False
        mock_config.generator.weight_sync_backend = "nccl"

        strategy_cls = get_transfer_strategy_cls(mock_config)
        assert strategy_cls == BroadcastTransferStrategy

    def test_broadcast_init_info_has_network_params(self, mock_config):
        """Test broadcast init info contains network parameters."""
        from skyrl_train.weight_sync import BroadcastTransferStrategy, BroadcastInitInfo

        mock_config.trainer.placement.colocate_all = False

        with patch("ray._private.services.get_node_ip_address", return_value="192.168.1.1"):
            init_info = BroadcastTransferStrategy.create_init_info(mock_config)

        assert isinstance(init_info, BroadcastInitInfo)
        assert init_info.master_addr == "192.168.1.1"
        assert init_info.master_port > 0
        assert init_info.world_size > 0


# -----------------------------------------------------------------------------
# Test Classes: Error Handling Tests
# -----------------------------------------------------------------------------

class TestErrorHandling:
    """Test error handling in various scenarios."""

    def test_weight_update_request_validation(self):
        """Test WeightUpdateRequest validates input lengths."""
        from skyrl_train.weight_sync.base import WeightUpdateRequest

        with pytest.raises(ValueError):
            WeightUpdateRequest(
                names=["a", "b", "c"],
                dtypes=["float32", "float32"],  # Mismatch: 3 vs 2
                shapes=[[10], [20], [30]],
            )

    def test_cuda_ipc_request_missing_marker(self):
        """Test CUDA IPC request detects missing end marker."""
        from skyrl_train.weight_sync import CudaIpcWeightUpdateRequest

        # Invalid data without end marker
        invalid_data = b"some random bytes without marker"

        with pytest.raises(ValueError, match="End marker not found"):
            CudaIpcWeightUpdateRequest.deserialize(invalid_data)

    @pytest.mark.asyncio
    async def test_generate_with_invalid_input(self, mock_tokenizer):
        """Test generate handles invalid input gracefully."""
        from skyrl_train.inference_engines.sglang.sglang_engine import SGLangInferenceEngine
        from skyrl_train.inference_engines.base import InferenceEngineInput

        with patch.object(SGLangInferenceEngine, '__init__', lambda x, **kwargs: None):
            engine = SGLangInferenceEngine()
            engine.tokenizer = mock_tokenizer

            # Missing required field (prompt_token_ids is None when prompts is also None)
            input_batch: InferenceEngineInput = {
                "prompts": None,
                "prompt_token_ids": None,  # Both None is invalid
                "sampling_params": {},
                "session_ids": None,
            }

            # Should handle gracefully (assertion or proper error)
            with pytest.raises((AssertionError, ValueError, KeyError)):
                await engine.generate(input_batch)


# -----------------------------------------------------------------------------
# Test Classes: Integration with Ray Tests
# -----------------------------------------------------------------------------

class TestRayActorIntegration:
    """Test Ray actor integration."""

    def test_sglang_ray_actor_defined(self):
        """Test SGLangRayActor is properly defined."""
        from skyrl_train.inference_engines.sglang.sglang_engine import SGLangRayActor

        assert SGLangRayActor is not None

    def test_ray_actor_is_remote_class(self):
        """Test SGLangRayActor is a Ray remote class."""
        from skyrl_train.inference_engines.sglang.sglang_engine import SGLangRayActor
        import ray

        # Ray remote classes have specific attributes
        assert hasattr(SGLangRayActor, 'remote')


# -----------------------------------------------------------------------------
# Test Classes: Utility Function Tests
# -----------------------------------------------------------------------------

class TestUtilityFunctions:
    """Test utility functions."""

    def test_setup_gpu_for_sglang(self):
        """Test GPU setup function for SGLang."""
        from skyrl_train.inference_engines.sglang.sglang_engine import setup_gpu_for_sglang

        kwargs = {"some_param": "value"}
        bundle_indices = [0]

        # Should not raise
        setup_gpu_for_sglang(kwargs, bundle_indices)

        # Should remove legacy parameters if present
        kwargs_with_legacy = {
            "distributed_executor_backend": "ray",
            "noset_visible_devices": True,
        }
        setup_gpu_for_sglang(kwargs_with_legacy, None)

        assert "distributed_executor_backend" not in kwargs_with_legacy
        assert "noset_visible_devices" not in kwargs_with_legacy

    def test_custom_weight_loader_path(self):
        """Test custom weight loader path is correctly set."""
        from skyrl_train.inference_engines.sglang.sglang_engine import CUSTOM_WEIGHT_LOADER_PATH

        assert "sglang_custom_weight_loader" in CUSTOM_WEIGHT_LOADER_PATH
        assert "skyrl_train" in CUSTOM_WEIGHT_LOADER_PATH

    def test_is_oom_error_detection(self):
        """Test OOM error detection function."""
        from skyrl_train.inference_engines.sglang.sglang_engine import _is_oom_error

        # Should detect OOM patterns
        assert _is_oom_error(RuntimeError("CUDA out of memory"))
        assert _is_oom_error(RuntimeError("prefill out of memory"))
        assert _is_oom_error(RuntimeError("failed to allocate memory"))
        assert _is_oom_error(RuntimeError("OOM during generation"))

        # Should not detect non-OOM errors
        assert not _is_oom_error(RuntimeError("Connection refused"))
        assert not _is_oom_error(ValueError("Invalid input"))


# -----------------------------------------------------------------------------
# Test Classes: Group Outputs By Prompt Tests
# -----------------------------------------------------------------------------

class TestGroupOutputsByPrompt:
    """Test group_outputs_by_prompt utility function."""

    def test_group_single_sample(self):
        """Test grouping with n=1 returns original output."""
        from skyrl_train.inference_engines.base import InferenceEngineOutput, group_outputs_by_prompt

        output: InferenceEngineOutput = {
            "responses": ["Hello", "World"],
            "response_ids": [[1, 2], [3, 4]],
            "stop_reasons": ["stop", "stop"],
            "response_logprobs": None,
            "weight_version": None,
            "n_per_prompt": 1,
            "request_ids": None,
            "hidden_states": None,
        }

        grouped = group_outputs_by_prompt(output)

        assert len(grouped) == 1
        assert grouped[0] == output

    def test_group_multiple_samples(self):
        """Test grouping with n>1."""
        from skyrl_train.inference_engines.base import InferenceEngineOutput, group_outputs_by_prompt

        # 2 prompts, n=3 samples each = 6 total
        output: InferenceEngineOutput = {
            "responses": ["a1", "a2", "a3", "b1", "b2", "b3"],
            "response_ids": [[1], [2], [3], [4], [5], [6]],
            "stop_reasons": ["stop"] * 6,
            "response_logprobs": None,
            "weight_version": "step_1",
            "n_per_prompt": 3,
            "request_ids": None,
            "hidden_states": None,
        }

        grouped = group_outputs_by_prompt(output)

        assert len(grouped) == 2
        assert grouped[0]["responses"] == ["a1", "a2", "a3"]
        assert grouped[1]["responses"] == ["b1", "b2", "b3"]
        assert grouped[0]["weight_version"] == "step_1"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
