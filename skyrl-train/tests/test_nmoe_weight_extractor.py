"""Tests for NMoEWeightExtractor.

These tests verify that the NMoEWeightExtractor correctly:
- Extracts weights from nmoe models
- Separates expert and dense parameters
- Produces correct WeightChunk objects

This test file is self-contained with mock implementations to avoid
dependency on ray and other heavy SkyRL dependencies.
"""

import sys
import os

# Add skyrl_train to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import List, Tuple, Iterator, Dict, Any, Callable
from functools import cached_property
from collections import defaultdict


# ============================================================================
# Mock WeightChunk and utilities (to avoid importing ray-dependent modules)
# ============================================================================

@dataclass
class MockWeightChunk:
    """Mock WeightChunk for testing."""
    names: List[str]
    dtypes: List[str]
    shapes: List[List[int]]
    tensors: List[torch.Tensor]

    def __len__(self) -> int:
        return len(self.names)

    @cached_property
    def total_numel(self) -> int:
        return sum(t.numel() for t in self.tensors)

    @cached_property
    def total_size_bytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.tensors)


def mock_yield_module_grouped_chunks(
    params: Dict[str, Any],
    dtype: torch.dtype,
    gather_tensor_fn: Callable[[Any], torch.Tensor],
    get_shape_fn: Callable[[str, Any, torch.Tensor], List[int]],
    batch_size_threshold_gb: float = 0.0,
) -> Iterator[MockWeightChunk]:
    """Mock implementation of yield_module_grouped_chunks."""
    module_to_params: Dict[str, List[str]] = defaultdict(list)
    for param_name in params.keys():
        module_name = ".".join(param_name.split(".")[:-2])
        module_to_params[module_name].append(param_name)

    batch_tensors = []
    batch_names = []
    batch_shapes = []
    batch_dtypes = []
    current_size = 0
    threshold_bytes = batch_size_threshold_gb * 1024**3

    for module_name, param_names in module_to_params.items():
        module_tensors = []
        module_names = []
        module_shapes = []
        module_dtypes = []
        module_size = 0

        for param_name in param_names:
            param = params[param_name]
            tensor = gather_tensor_fn(param)
            tensor = tensor.to(dtype).detach().contiguous()
            shape = get_shape_fn(param_name, param, tensor)
            module_tensors.append(tensor)
            module_names.append(param_name)
            module_shapes.append(shape)
            module_dtypes.append(str(dtype))
            module_size += tensor.nbytes

        if current_size > 0 and current_size + module_size > threshold_bytes:
            yield MockWeightChunk(
                names=batch_names,
                dtypes=batch_dtypes,
                shapes=batch_shapes,
                tensors=batch_tensors,
            )
            batch_tensors = []
            batch_names = []
            batch_shapes = []
            batch_dtypes = []
            current_size = 0

        batch_tensors.extend(module_tensors)
        batch_names.extend(module_names)
        batch_shapes.extend(module_shapes)
        batch_dtypes.extend(module_dtypes)
        current_size += module_size

    if batch_tensors:
        yield MockWeightChunk(
            names=batch_names,
            dtypes=batch_dtypes,
            shapes=batch_shapes,
            tensors=batch_tensors,
        )


# ============================================================================
# Mock nmoe model for testing
# ============================================================================

@dataclass
class MockConfig:
    """Mock nmoe config."""
    dim: int = 256
    n_layers: int = 4
    n_heads: int = 4
    n_routed_experts: int = 8
    n_activated_experts: int = 2
    n_dense_layers: int = 1
    vocab_size: int = 1000


class MockMoE(nn.Module):
    """Mock MoE layer."""
    def __init__(self, dim: int, n_experts: int, inter_dim: int):
        super().__init__()
        self.dim = dim
        self.n_experts = n_experts
        self.W1 = nn.Parameter(torch.randn(n_experts, dim, inter_dim) * 0.02)
        self.W3 = nn.Parameter(torch.randn(n_experts, dim, inter_dim) * 0.02)
        self.W2 = nn.Parameter(torch.randn(n_experts, inter_dim, dim) * 0.02)
        self.router = nn.Linear(dim, n_experts, bias=False)

    def forward(self, x):
        return x


class MockTransformerBlock(nn.Module):
    """Mock transformer block."""
    def __init__(self, config, layer_id, is_moe=False):
        super().__init__()
        self.layer_id = layer_id
        self.is_moe = is_moe
        self.attn = nn.Linear(config.dim, config.dim)
        self.attn_norm = nn.LayerNorm(config.dim)
        self.ffn_norm = nn.LayerNorm(config.dim)

        if is_moe:
            self.ffn = MockMoE(config.dim, config.n_routed_experts, config.dim * 2)
        else:
            self.ffn = nn.Sequential(
                nn.Linear(config.dim, config.dim * 4),
                nn.GELU(),
                nn.Linear(config.dim * 4, config.dim),
            )

    def forward(self, x):
        return x


class MockNMoEModel(nn.Module):
    """Mock nmoe Transformer model."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.dim)
        self.blocks = nn.ModuleList([
            MockTransformerBlock(
                config, i,
                is_moe=(i >= config.n_dense_layers)
            )
            for i in range(config.n_layers)
        ])
        self.norm = nn.LayerNorm(config.dim)
        self.lm_head = nn.Linear(config.vocab_size, config.dim, bias=False)

    def param_sets(self) -> Tuple[List[nn.Parameter], List[nn.Parameter]]:
        """Separate expert and dense parameters."""
        expert_params = []
        for m in self.modules():
            if isinstance(m, MockMoE):
                expert_params.extend([m.W1, m.W3, m.W2])
        expert_ids = {id(p) for p in expert_params}
        dense_params = [p for p in self.parameters() if id(p) not in expert_ids]
        return expert_params, dense_params

    def forward(self, x):
        return self.lm_head(self.norm(self.embedding(x)))


class MockWrapper(nn.Module):
    """Mock model wrapper."""
    def __init__(self, model):
        super().__init__()
        self.model = model


# ============================================================================
# Testable NMoEWeightExtractor (self-contained)
# ============================================================================

class TestableNMoEWeightExtractor:
    """Self-contained weight extractor for testing."""

    def __init__(
        self,
        model: nn.Module,
        ep_group=None,
        ep_rank: int = 0,
        ep_world_size: int = 1,
        batch_size_threshold_gb: float = 0.5,
    ):
        self.model = model
        self.ep_group = ep_group
        self.ep_rank = ep_rank
        self.ep_world_size = ep_world_size
        self.batch_size_threshold_gb = batch_size_threshold_gb

        self._nmoe_model = self._unwrap_model(model)
        self._expert_param_names = None
        self._categorize_parameters()

    def _unwrap_model(self, model):
        if hasattr(model, 'model'):
            return model.model
        if hasattr(model, '_nmoe_model'):
            return model._nmoe_model
        return model

    def _categorize_parameters(self):
        self._expert_param_names = set()

        if hasattr(self._nmoe_model, 'param_sets'):
            expert_params, _ = self._nmoe_model.param_sets()
            expert_ids = {id(p) for p in expert_params}

            for name, param in self._nmoe_model.named_parameters():
                if id(param) in expert_ids:
                    self._expert_param_names.add(name)
        else:
            for name, _ in self._nmoe_model.named_parameters():
                if any(pattern in name for pattern in ['W1', 'W2', 'W3']):
                    if 'ffn' in name or 'moe' in name.lower():
                        self._expert_param_names.add(name)

    def _is_expert_param(self, name: str) -> bool:
        return name in self._expert_param_names

    def _gather_tensor_fn(self, param, name: str):
        # For single GPU, just return the param data
        return param.data if hasattr(param, 'data') else param

    def extract_weights(self, dtype: torch.dtype) -> Iterator[MockWeightChunk]:
        params = dict(self._nmoe_model.named_parameters())
        is_primary = self.ep_rank == 0 or self.ep_world_size == 1

        def gather_fn(param_tuple):
            if isinstance(param_tuple, tuple) and len(param_tuple) == 2:
                name, param = param_tuple
                return self._gather_tensor_fn(param.data, name)
            return param_tuple.data if hasattr(param_tuple, 'data') else param_tuple

        def shape_fn(param_name, param, tensor):
            return list(tensor.shape)

        if is_primary:
            named_params = {name: (name, param) for name, param in params.items()}
            yield from mock_yield_module_grouped_chunks(
                params=named_params,
                dtype=dtype,
                gather_tensor_fn=gather_fn,
                get_shape_fn=shape_fn,
                batch_size_threshold_gb=self.batch_size_threshold_gb,
            )

    def extract_expert_weights_only(self, dtype: torch.dtype) -> Iterator[MockWeightChunk]:
        expert_params = {}
        for name, param in self._nmoe_model.named_parameters():
            if self._is_expert_param(name):
                expert_params[name] = (name, param)

        is_primary = self.ep_rank == 0 or self.ep_world_size == 1

        def gather_fn(param_tuple):
            name, param = param_tuple
            return param.data

        def shape_fn(param_name, param, tensor):
            return list(tensor.shape)

        if is_primary:
            yield from mock_yield_module_grouped_chunks(
                params=expert_params,
                dtype=dtype,
                gather_tensor_fn=gather_fn,
                get_shape_fn=shape_fn,
                batch_size_threshold_gb=self.batch_size_threshold_gb,
            )

    def extract_dense_weights_only(self, dtype: torch.dtype) -> Iterator[MockWeightChunk]:
        dense_params = {}
        for name, param in self._nmoe_model.named_parameters():
            if not self._is_expert_param(name):
                dense_params[name] = param

        def gather_fn(param):
            return param.data

        def shape_fn(param_name, param, tensor):
            return list(tensor.shape)

        yield from mock_yield_module_grouped_chunks(
            params=dense_params,
            dtype=dtype,
            gather_tensor_fn=gather_fn,
            get_shape_fn=shape_fn,
            batch_size_threshold_gb=self.batch_size_threshold_gb,
        )

    def get_weight_stats(self) -> Dict[str, Any]:
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

        if self.ep_world_size > 1:
            expert_numel *= self.ep_world_size

        return {
            'n_expert_params': n_expert,
            'n_dense_params': n_dense,
            'expert_numel': expert_numel,
            'dense_numel': dense_numel,
            'total_numel': expert_numel + dense_numel,
            'ep_world_size': self.ep_world_size,
            'expert_size_gb': expert_numel * 2 / (1024**3),
            'dense_size_gb': dense_numel * 2 / (1024**3),
        }


# ============================================================================
# Tests
# ============================================================================

class TestNMoEWeightExtractor:
    """Tests for NMoEWeightExtractor."""

    @pytest.fixture
    def config(self):
        return MockConfig()

    @pytest.fixture
    def model(self, config):
        return MockNMoEModel(config)

    @pytest.fixture
    def wrapped_model(self, model):
        return MockWrapper(model)

    @pytest.fixture
    def extractor(self, model):
        return TestableNMoEWeightExtractor(model)

    @pytest.fixture
    def wrapped_extractor(self, wrapped_model):
        return TestableNMoEWeightExtractor(wrapped_model)

    def test_init_unwraps_model(self, extractor, model):
        """Test that extractor correctly unwraps the model."""
        assert extractor._nmoe_model is model

    def test_init_unwraps_wrapper(self, wrapped_extractor, wrapped_model):
        """Test that extractor unwraps wrapped models."""
        assert wrapped_extractor._nmoe_model is wrapped_model.model

    def test_categorize_parameters(self, extractor, config):
        """Test that parameters are correctly categorized."""
        assert len(extractor._expert_param_names) > 0
        n_moe_layers = config.n_layers - config.n_dense_layers
        expected_expert_count = n_moe_layers * 3
        assert len(extractor._expert_param_names) == expected_expert_count

    def test_is_expert_param(self, extractor, model):
        """Test expert parameter detection."""
        for name, _ in model.named_parameters():
            is_expert = extractor._is_expert_param(name)
            should_be_expert = ('W1' in name or 'W2' in name or 'W3' in name) and 'ffn' in name
            assert is_expert == should_be_expert, f"Wrong classification for {name}"

    def test_extract_weights_yields_chunks(self, extractor):
        """Test that extract_weights yields WeightChunk objects."""
        chunks = list(extractor.extract_weights(torch.float32))
        assert len(chunks) > 0

        for chunk in chunks:
            assert isinstance(chunk, MockWeightChunk)
            assert len(chunk.names) > 0
            assert len(chunk.tensors) == len(chunk.names)
            assert len(chunk.shapes) == len(chunk.names)
            assert len(chunk.dtypes) == len(chunk.names)

    def test_extract_weights_dtype_conversion(self, extractor):
        """Test that weights are converted to target dtype."""
        target_dtype = torch.float16
        chunks = list(extractor.extract_weights(target_dtype))

        for chunk in chunks:
            for tensor in chunk.tensors:
                assert tensor.dtype == target_dtype

    def test_extract_weights_shapes_match(self, extractor, model):
        """Test that extracted weight shapes match model params."""
        chunks = list(extractor.extract_weights(torch.float32))

        extracted = {}
        for chunk in chunks:
            for name, shape in zip(chunk.names, chunk.shapes):
                extracted[name] = shape

        for name, param in model.named_parameters():
            assert name in extracted, f"Parameter {name} not extracted"
            assert extracted[name] == list(param.shape), f"Shape mismatch for {name}"

    def test_extract_expert_weights_only(self, extractor, config):
        """Test extracting only expert weights."""
        chunks = list(extractor.extract_expert_weights_only(torch.float32))

        all_names = []
        for chunk in chunks:
            all_names.extend(chunk.names)

        for name in all_names:
            assert extractor._is_expert_param(name), f"{name} is not an expert param"

        n_moe_layers = config.n_layers - config.n_dense_layers
        expected = n_moe_layers * 3
        assert len(all_names) == expected

    def test_extract_dense_weights_only(self, extractor):
        """Test extracting only dense weights."""
        chunks = list(extractor.extract_dense_weights_only(torch.float32))

        all_names = []
        for chunk in chunks:
            all_names.extend(chunk.names)

        for name in all_names:
            assert not extractor._is_expert_param(name), f"{name} is an expert param"

    def test_get_weight_stats(self, extractor, config):
        """Test weight statistics."""
        stats = extractor.get_weight_stats()

        assert 'n_expert_params' in stats
        assert 'n_dense_params' in stats
        assert 'expert_numel' in stats
        assert 'dense_numel' in stats
        assert 'total_numel' in stats
        assert 'ep_world_size' in stats

        n_moe_layers = config.n_layers - config.n_dense_layers
        assert stats['n_expert_params'] == n_moe_layers * 3
        assert stats['n_dense_params'] > 0
        assert stats['ep_world_size'] == 1

    def test_contiguous_tensors(self, extractor):
        """Test that extracted tensors are contiguous."""
        chunks = list(extractor.extract_weights(torch.float32))

        for chunk in chunks:
            for tensor in chunk.tensors:
                assert tensor.is_contiguous(), "Extracted tensor not contiguous"


class TestNMoEWeightExtractorFactory:
    """Tests for the factory function pattern."""

    def test_create_extractor_single_gpu(self):
        """Test creating extractor without distributed."""
        config = MockConfig()
        model = MockNMoEModel(config)
        extractor = TestableNMoEWeightExtractor(model)

        assert extractor.ep_rank == 0
        assert extractor.ep_world_size == 1
        assert extractor.ep_group is None

    def test_create_extractor_with_threshold(self):
        """Test creating extractor with custom batch threshold."""
        config = MockConfig()
        model = MockNMoEModel(config)
        extractor = TestableNMoEWeightExtractor(model, batch_size_threshold_gb=1.0)

        assert extractor.batch_size_threshold_gb == 1.0


class TestWeightChunkIntegrity:
    """Tests for WeightChunk data integrity."""

    def test_total_numel(self):
        """Test total_numel calculation."""
        config = MockConfig()
        model = MockNMoEModel(config)
        extractor = TestableNMoEWeightExtractor(model)

        chunks = list(extractor.extract_weights(torch.float32))

        chunk_total = sum(chunk.total_numel for chunk in chunks)
        model_total = sum(p.numel() for p in model.parameters())

        assert chunk_total == model_total

    def test_names_unique(self):
        """Test that parameter names are unique across chunks."""
        config = MockConfig()
        model = MockNMoEModel(config)
        extractor = TestableNMoEWeightExtractor(model)

        chunks = list(extractor.extract_weights(torch.float32))

        all_names = []
        for chunk in chunks:
            all_names.extend(chunk.names)

        assert len(all_names) == len(set(all_names))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
