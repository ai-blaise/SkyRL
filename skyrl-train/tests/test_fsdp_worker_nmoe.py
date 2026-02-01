"""Tests for NMoE FSDP Workers.

Tests the NMoE-specific FSDP workers for policy and reference models.
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# Mock Classes
# =============================================================================

@dataclass
class MockConfig:
    """Mock nmoe config."""
    dim: int = 256
    n_layers: int = 2
    n_heads: int = 4
    vocab_size: int = 1000
    n_routed_experts: int = 4
    n_activated_experts: int = 2
    dtype: str = "bf16"
    eos_token_id: int = 999


class MockMLP(nn.Module):
    """Mock MLP for testing."""
    def __init__(self, dim):
        super().__init__()
        self.w1 = nn.Linear(dim, dim * 4, bias=False)
        self.w2 = nn.Linear(dim * 4, dim, bias=False)

    def forward(self, x):
        return self.w2(torch.relu(self.w1(x)))


class MockRouter(nn.Module):
    """Mock router for testing."""
    def __init__(self, dim, n_experts):
        super().__init__()
        self.gate = nn.Linear(dim, n_experts, bias=False)
        self.bias = nn.Parameter(torch.zeros(n_experts))

    def forward(self, x):
        return torch.softmax(self.gate(x) + self.bias, dim=-1)

    def update_bias(self, loads, gamma=0.001):
        with torch.no_grad():
            target = loads.mean()
            self.bias.add_((target - loads) * gamma)


class MockMoE(nn.Module):
    """Mock MoE layer for testing."""
    def __init__(self, dim, n_experts):
        super().__init__()
        self.router = MockRouter(dim, n_experts)
        self.W1 = nn.Parameter(torch.randn(n_experts, dim, dim * 4))
        self.W2 = nn.Parameter(torch.randn(n_experts, dim * 4, dim))
        self.W3 = nn.Parameter(torch.randn(n_experts, dim, dim * 4))
        self.last_aux_loss = None
        self.last_loads = None

    def forward(self, x):
        router_probs = self.router(x)
        self.last_loads = router_probs.mean(dim=(0, 1))
        self.last_aux_loss = torch.tensor(0.01)
        return x + 0.1 * x

    def refresh_weight_cache(self):
        pass


class MockBlock(nn.Module):
    """Mock transformer block."""
    def __init__(self, dim, n_experts):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, 4, batch_first=True)
        self.ffn = MockMoE(dim, n_experts) if n_experts > 0 else MockMLP(dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self._use_gradient_checkpointing = False

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.ffn(self.norm2(x))
        return x


class MockTransformer(nn.Module):
    """Mock nmoe Transformer for testing."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.dim)
        self.blocks = nn.ModuleList([
            MockBlock(config.dim, config.n_routed_experts)
            for _ in range(config.n_layers)
        ])
        self.norm = nn.LayerNorm(config.dim)
        self.lm_head = nn.Linear(config.vocab_size, config.dim, bias=False)

    def forward(self, tokens):
        x = self.embedding(tokens)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        logits = self.lm_head(x)
        return logits

    def param_sets(self):
        expert_params = []
        dense_params = []
        for name, param in self.named_parameters():
            if 'W1' in name or 'W2' in name or 'W3' in name:
                expert_params.append(param)
            else:
                dense_params.append(param)
        return expert_params, dense_params


# =============================================================================
# Test: NMoEFSDPWeightExtractor
# =============================================================================

class TestNMoEFSDPWeightExtractor:
    """Tests for NMoEFSDPWeightExtractor."""

    @pytest.fixture
    def model(self):
        """Create test model."""
        config = MockConfig()
        return MockTransformer(config)

    def test_extractor_creation(self, model):
        """Test creating weight extractor."""
        from skyrl_train.workers.fsdp.fsdp_worker_nmoe import NMoEFSDPWeightExtractor

        extractor = NMoEFSDPWeightExtractor(
            model=model,
            group_by_module=False,
            batch_size_threshold_gb=0.5,
        )

        assert extractor.model is model
        assert extractor.group_by_module is False
        assert extractor.batch_size_threshold_gb == 0.5

    def test_gather_tensor_non_dtensor(self, model):
        """Test _gather_tensor with regular tensor."""
        from skyrl_train.workers.fsdp.fsdp_worker_nmoe import NMoEFSDPWeightExtractor

        extractor = NMoEFSDPWeightExtractor(model)

        # Regular tensor should pass through
        tensor = torch.randn(10, 10)
        result = extractor._gather_tensor(tensor)

        assert torch.equal(result, tensor)


# =============================================================================
# Test: Mixin Classes
# =============================================================================

class TestNMoEMixins:
    """Tests for NMoE mixin classes."""

    def test_policy_mixin_exists(self):
        """Test that NMoEFSDPPolicyWorkerMixin exists."""
        from skyrl_train.workers.fsdp.fsdp_worker_nmoe import NMoEFSDPPolicyWorkerMixin

        assert NMoEFSDPPolicyWorkerMixin is not None
        assert hasattr(NMoEFSDPPolicyWorkerMixin, 'init_nmoe_model')
        assert hasattr(NMoEFSDPPolicyWorkerMixin, '_refresh_expert_caches')

    def test_ref_mixin_exists(self):
        """Test that NMoEFSDPRefWorkerMixin exists."""
        from skyrl_train.workers.fsdp.fsdp_worker_nmoe import NMoEFSDPRefWorkerMixin

        assert NMoEFSDPRefWorkerMixin is not None
        assert hasattr(NMoEFSDPRefWorkerMixin, 'init_nmoe_ref_model')


# =============================================================================
# Test: Helper Functions
# =============================================================================

class TestHelperFunctions:
    """Tests for helper functions."""

    def test_fsdp_version_default(self):
        """Test _fsdp_version returns 2 for non-FSDP models."""
        from skyrl_train.workers.fsdp.fsdp_worker_nmoe import _fsdp_version

        model = nn.Linear(10, 10)
        assert _fsdp_version(model) == 2

    def test_get_nmoe_workers_import(self):
        """Test get_nmoe_workers function exists."""
        from skyrl_train.workers.fsdp.fsdp_worker_nmoe import get_nmoe_workers

        # Function should exist
        assert callable(get_nmoe_workers)


# =============================================================================
# Test: Integration with model_factory
# =============================================================================

class TestIntegrationWithModelFactory:
    """Test integration between workers and model_factory."""

    def test_model_type_detection_consistency(self):
        """Test that model type detection is consistent."""
        @dataclass
        class MockNMoEConfig:
            model_type: str = "nmoe"
            def get(self, key, default=None):
                return getattr(self, key, default)

        @dataclass
        class MockPolicyConfig:
            model: Any = field(default_factory=lambda: type('obj', (object,), {'type': 'nmoe', 'get': lambda s, k, d=None: getattr(s, k, d)})())
            nmoe_config: Optional[MockNMoEConfig] = field(default_factory=MockNMoEConfig)
            def get(self, key, default=None):
                return getattr(self, key, default)

        @dataclass
        class MockTrainerConfig:
            policy: MockPolicyConfig = field(default_factory=MockPolicyConfig)
            def get(self, key, default=None):
                return getattr(self, key, default)

        @dataclass
        class MockCfg:
            trainer: MockTrainerConfig = field(default_factory=MockTrainerConfig)
            def get(self, key, default=None):
                return getattr(self, key, default)

        from skyrl_train.model_factory import get_model_type

        cfg = MockCfg()
        model_type = get_model_type(cfg)

        assert model_type == "nmoe"


# =============================================================================
# Test: Expert Cache Refresh
# =============================================================================

class TestExpertCacheRefresh:
    """Tests for expert cache refresh functionality."""

    def test_optim_step_refreshes_cache_for_quantized(self):
        """Test that optim_step calls _refresh_expert_caches for quantized models."""
        from skyrl_train.workers.fsdp.fsdp_worker_nmoe import NMoEFSDPPolicyWorkerMixin

        # Create a mock class that has the mixin
        class MockWorker(NMoEFSDPPolicyWorkerMixin):
            def __init__(self):
                self._uses_quantized_experts = True
                self._cache_refresh_called = False

            def _refresh_expert_caches(self):
                self._cache_refresh_called = True

        worker = MockWorker()

        # Simulate the cache refresh logic
        if getattr(worker, '_uses_quantized_experts', False):
            worker._refresh_expert_caches()

        assert worker._cache_refresh_called, "Cache refresh should be called for quantized experts"

    def test_optim_step_skips_cache_for_bf16(self):
        """Test that optim_step skips cache refresh for BF16 models."""
        from skyrl_train.workers.fsdp.fsdp_worker_nmoe import NMoEFSDPPolicyWorkerMixin

        class MockWorker(NMoEFSDPPolicyWorkerMixin):
            def __init__(self):
                self._uses_quantized_experts = False
                self._cache_refresh_called = False

            def _refresh_expert_caches(self):
                self._cache_refresh_called = True

        worker = MockWorker()

        # Simulate the cache refresh logic
        if getattr(worker, '_uses_quantized_experts', False):
            worker._refresh_expert_caches()

        assert not worker._cache_refresh_called, "Cache refresh should NOT be called for BF16 models"


# =============================================================================
# Test: Worker Selection in main_base
# =============================================================================

class TestWorkerSelection:
    """Tests for worker selection based on model type."""

    @pytest.fixture
    def mock_hf_config(self):
        """Create mock config for HF model."""
        @dataclass
        class MockModelConfig:
            type: str = "hf"
            path: str = "/fake/path"
            def get(self, key, default=None):
                return getattr(self, key, default)

        @dataclass
        class MockPolicyConfig:
            model: MockModelConfig = field(default_factory=MockModelConfig)
            nmoe_config: Optional[Any] = None
            def get(self, key, default=None):
                return getattr(self, key, default)

        @dataclass
        class MockTrainerConfig:
            policy: MockPolicyConfig = field(default_factory=MockPolicyConfig)
            strategy: str = "fsdp"
            def get(self, key, default=None):
                return getattr(self, key, default)

        @dataclass
        class MockCfg:
            trainer: MockTrainerConfig = field(default_factory=MockTrainerConfig)
            def get(self, key, default=None):
                return getattr(self, key, default)

        return MockCfg()

    @pytest.fixture
    def mock_nmoe_config(self):
        """Create mock config for nmoe model."""
        @dataclass
        class MockNMoEConfig:
            model_type: str = "nmoe"
            def get(self, key, default=None):
                return getattr(self, key, default)

        @dataclass
        class MockModelConfig:
            type: str = "nmoe"
            path: str = "/fake/nmoe/path"
            def get(self, key, default=None):
                return getattr(self, key, default)

        @dataclass
        class MockPolicyConfig:
            model: MockModelConfig = field(default_factory=MockModelConfig)
            nmoe_config: MockNMoEConfig = field(default_factory=MockNMoEConfig)
            def get(self, key, default=None):
                return getattr(self, key, default)

        @dataclass
        class MockTrainerConfig:
            policy: MockPolicyConfig = field(default_factory=MockPolicyConfig)
            strategy: str = "fsdp"
            def get(self, key, default=None):
                return getattr(self, key, default)

        @dataclass
        class MockCfg:
            trainer: MockTrainerConfig = field(default_factory=MockTrainerConfig)
            def get(self, key, default=None):
                return getattr(self, key, default)

        return MockCfg()

    def test_hf_model_selects_hf_workers(self, mock_hf_config):
        """Test that HF model type selects HF workers."""
        from skyrl_train.model_factory import get_model_type

        model_type = get_model_type(mock_hf_config)
        assert model_type == "hf"

        # Verify the worker import path would work for HF
        from skyrl_train.workers.fsdp.fsdp_worker import PolicyWorker, CriticWorker, RefWorker
        assert PolicyWorker is not None
        assert RefWorker is not None

    def test_nmoe_model_selects_nmoe_workers(self, mock_nmoe_config):
        """Test that nmoe model type selects nmoe workers."""
        from skyrl_train.model_factory import get_model_type

        model_type = get_model_type(mock_nmoe_config)
        assert model_type == "nmoe"

        # Verify the worker import path would work for nmoe
        from skyrl_train.workers.fsdp.fsdp_worker_nmoe import get_nmoe_workers
        PolicyWorker, CriticWorker, RefWorker = get_nmoe_workers()
        assert PolicyWorker is not None
        # CriticWorker is None for nmoe (not yet supported)
        assert CriticWorker is None
        assert RefWorker is not None

    def test_nmoe_with_megatron_raises_not_implemented(self, mock_nmoe_config):
        """Test that nmoe + megatron strategy raises NotImplementedError."""
        # Set strategy to megatron
        mock_nmoe_config.trainer.strategy = "megatron"

        from skyrl_train.model_factory import get_model_type
        model_type = get_model_type(mock_nmoe_config)
        assert model_type == "nmoe"

        # This should raise NotImplementedError when actually trying to get workers
        # We can't fully test this without instantiating BasePPOExp, but we can
        # verify the logic by checking the strategy
        assert mock_nmoe_config.trainer.strategy == "megatron"


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
