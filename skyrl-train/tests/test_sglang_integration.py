"""Integration tests for SGLang inference engine with SkyRL.

This test module verifies that the SGLang inference engine works correctly
with the SkyRL training framework.

To run these tests:
    pytest tests/test_sglang_integration.py -v -s

Note: These tests require a GPU and the sglang extra to be installed:
    uv sync --extra sglang
"""

import pytest
import asyncio
from typing import List, Dict, Any

# Skip all tests if sglang is not available
pytest.importorskip("sglang")


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
        )
        assert UpdateWeightsFromTensorReqInput is not None
        assert UpdateWeightsFromDistributedReqInput is not None
        assert InitWeightsUpdateGroupReqInput is not None
        assert ReleaseMemoryOccupationReqInput is not None
        assert ResumeMemoryOccupationReqInput is not None
        assert PauseGenerationReqInput is not None

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

        # Check that it's a subclass of the interface
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
            'tp_size',
            'pp_size',
            'dp_size',
        ]

        for method_name in required_methods:
            assert hasattr(SGLangInferenceEngine, method_name), f"Missing method: {method_name}"


class TestSGLangWeightLoader:
    """Test SGLangWeightLoader functionality."""

    def test_weight_loader_init(self):
        """Test that SGLangWeightLoader can be imported and has required methods."""
        from skyrl_train.inference_engines.sglang.sglang_engine import SGLangWeightLoader

        # Check required methods
        required_methods = ['init_communicator', 'load_weights']
        for method_name in required_methods:
            assert hasattr(SGLangWeightLoader, method_name), f"Missing method: {method_name}"


class TestSGLangDataTypes:
    """Test SGLang-specific data types and conversion."""

    def test_inference_engine_input_type(self):
        """Test InferenceEngineInput structure."""
        from skyrl_train.inference_engines.base import InferenceEngineInput

        # Create a sample input
        sample_input: InferenceEngineInput = {
            "prompts": None,
            "prompt_token_ids": [[1, 2, 3], [4, 5, 6]],
            "sampling_params": {"max_new_tokens": 100, "temperature": 0.7},
            "session_ids": None,
        }
        assert sample_input["prompt_token_ids"] is not None

    def test_inference_engine_output_type(self):
        """Test InferenceEngineOutput structure."""
        from skyrl_train.inference_engines.base import InferenceEngineOutput

        # Create a sample output
        sample_output: InferenceEngineOutput = {
            "responses": ["Hello", "World"],
            "response_ids": [[7, 8, 9], [10, 11, 12]],
            "stop_reasons": ["stop", "length"],
            "response_logprobs": None,
        }
        assert len(sample_output["responses"]) == 2


@pytest.mark.skipif(
    not pytest.importorskip("torch").cuda.is_available(),
    reason="CUDA not available"
)
class TestSGLangEngineGPU:
    """GPU-based tests for SGLang engine.

    These tests require a GPU and will be skipped if CUDA is not available.
    """

    @pytest.fixture
    def small_model_path(self):
        """Return a small model path for testing."""
        # Use a small model for faster tests
        return "Qwen/Qwen2.5-0.5B-Instruct"

    @pytest.mark.slow
    def test_engine_initialization(self, small_model_path):
        """Test that SGLang engine can be initialized."""
        from transformers import AutoTokenizer
        from skyrl_train.inference_engines.sglang.sglang_engine import SGLangInferenceEngine

        tokenizer = AutoTokenizer.from_pretrained(small_model_path)

        # Initialize engine with minimal config
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
            # Prepare input
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

            # Generate
            output = asyncio.run(engine.generate(input_batch))

            # Verify output structure
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
            # Prepare input with logprobs
            prompt = "The capital of France is"
            input_ids = tokenizer.encode(prompt, add_special_tokens=True)

            input_batch: InferenceEngineInput = {
                "prompts": None,
                "prompt_token_ids": [input_ids],
                "sampling_params": {
                    "max_new_tokens": 20,
                    "temperature": 0.0,  # Greedy for deterministic output
                    "return_logprob": True,
                },
                "session_ids": None,
            }

            # Generate
            output = asyncio.run(engine.generate(input_batch))

            # Verify logprobs are returned
            assert "response_logprobs" in output
            assert output["response_logprobs"] is not None
            assert len(output["response_logprobs"]) == 1

        finally:
            asyncio.run(engine.teardown())


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
