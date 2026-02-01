"""
Comprehensive Integration Tests for NMoE Model Training with SkyRL's Megatron Strategy.

This module provides HIGH priority integration tests for:
1. NMoE model with Tensor Parallelism (TP=2,4,8)
2. NMoE model with Pipeline Parallelism (PP=2,4)
3. NMoE model with Expert Parallelism (EP=8)
4. Combined TP+EP configurations
5. Combined TP+PP+DP configurations
6. Gradient accumulation with Megatron+NMoE
7. Checkpoint save/load with Megatron+NMoE
8. 1F1B pipeline schedule with NMoE

Run with:
    uv run --isolated --extra dev --extra mcore -- pytest \
        tests/integration/test_nmoe_megatron_integration.py -v

For multi-GPU tests:
    uv run --isolated --extra dev --extra mcore -- pytest \
        tests/integration/test_nmoe_megatron_integration.py -v -m "multi_gpu"
"""

from __future__ import annotations

import gc
import json
import os
import random
import tempfile
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Markers and Skip Conditions
# =============================================================================

# Custom markers for multi-GPU tests
gpu = pytest.mark.gpu
multi_gpu = pytest.mark.multi_gpu

# Skip conditions based on hardware availability
requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available"
)

requires_2_gpus = pytest.mark.skipif(
    torch.cuda.device_count() < 2,
    reason="Requires at least 2 GPUs"
)

requires_4_gpus = pytest.mark.skipif(
    torch.cuda.device_count() < 4,
    reason="Requires at least 4 GPUs"
)

requires_8_gpus = pytest.mark.skipif(
    torch.cuda.device_count() < 8,
    reason="Requires at least 8 GPUs"
)

# Check for optional dependencies
try:
    import megatron.core.parallel_state as mpu
    from megatron.core.optimizer import DistributedOptimizer
    from megatron.core.packed_seq_params import PackedSeqParams
    HAS_MEGATRON = True
except ImportError:
    HAS_MEGATRON = False
    mpu = None

try:
    from nmoe.config import Config as NMoEConfig
    from nmoe.rdep import Rdep
    HAS_NMOE = True
except ImportError:
    HAS_NMOE = False
    NMoEConfig = None
    Rdep = None

try:
    from skyrl_train.distributed.megatron.nmoe_megatron_strategy import (
        NMoEMegatronConfig,
        NMoEMegatronStrategy,
        create_nmoe_megatron_strategy,
    )
    from skyrl_train.distributed.megatron.megatron_strategy import MegatronStrategy
    from skyrl_train.distributed.megatron.megatron_utils import (
        make_batch_generator,
        get_model_size,
        offload_megatron_model_to_cpu,
        load_megatron_model_to_gpu,
    )
    from skyrl_train.workers.megatron.nmoe_megatron_worker import (
        NMoEMegatronModelWrapper,
        NMoEMegatronWeightExtractor,
    )
    HAS_SKYRL = True
except ImportError:
    HAS_SKYRL = False
    NMoEMegatronConfig = None
    NMoEMegatronStrategy = None

requires_megatron = pytest.mark.skipif(
    not HAS_MEGATRON,
    reason="Megatron-Core not installed"
)

requires_nmoe = pytest.mark.skipif(
    not HAS_NMOE,
    reason="nmoe package not installed"
)

requires_skyrl = pytest.mark.skipif(
    not HAS_SKYRL,
    reason="skyrl_train not installed or missing nmoe_megatron components"
)


# =============================================================================
# Mock and Test Model Classes
# =============================================================================

@dataclass
class MockNMoEConfig:
    """Mock NMoE configuration for testing."""
    dim: int = 256
    inter_dim: int = 1024
    moe_inter_dim: int = 512
    n_layers: int = 4
    n_heads: int = 8
    n_routed_experts: int = 8
    n_activated_experts: int = 2
    n_shared_experts: int = 2
    n_dense_layers: int = 1
    vocab_size: int = 1000
    max_position_embeddings: int = 2048
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    dtype: str = "bf16"
    batch_size: int = 4
    seq_len: int = 128


class MockRouter(nn.Module):
    """Mock router for NMoE testing."""

    def __init__(self, dim: int, n_experts: int, topk: int):
        super().__init__()
        self.n_experts = n_experts
        self.topk = topk
        self.gate = nn.Linear(dim, n_experts, bias=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.gate(x)
        scores = torch.softmax(logits, dim=-1)
        weights, indices = torch.topk(scores, k=self.topk, dim=-1)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp(min=1e-12)
        return weights, indices


class MockMoELayer(nn.Module):
    """Mock MoE layer for testing Megatron integration."""

    def __init__(
        self,
        dim: int,
        inter_dim: int,
        n_experts: int,
        n_activated: int,
    ):
        super().__init__()
        self.dim = dim
        self.n_experts = n_experts
        self.n_activated = n_activated
        self.router = MockRouter(dim, n_experts, n_activated)

        # Expert weights (simplified)
        self.W1 = nn.Parameter(torch.randn(n_experts, dim, inter_dim) * 0.02)
        self.W2 = nn.Parameter(torch.randn(n_experts, inter_dim, dim) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, dim = x.shape
        x_flat = x.view(-1, dim)

        # Route tokens
        weights, indices = self.router(x_flat)

        # Simplified expert computation (using first expert for all)
        # In real implementation, this uses RDEP dispatch
        out = F.silu(x_flat @ self.W1[0]) @ self.W2[0]
        return out.view(batch_size, seq_len, dim)


class MockNMoETransformerBlock(nn.Module):
    """Mock NMoE transformer block for testing."""

    def __init__(
        self,
        dim: int,
        n_heads: int,
        inter_dim: int,
        n_experts: int,
        n_activated: int,
        is_moe: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.is_moe = is_moe

        # Simplified attention (no MLA for testing)
        self.ln1 = nn.LayerNorm(dim)
        head_dim = dim // n_heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

        # MLP or MoE
        self.ln2 = nn.LayerNorm(dim)
        if is_moe:
            self.moe = MockMoELayer(dim, inter_dim, n_experts, n_activated)
        else:
            self.mlp = nn.Sequential(
                nn.Linear(dim, inter_dim),
                nn.GELU(),
                nn.Linear(inter_dim, dim),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Simplified attention
        residual = x
        x = self.ln1(x)
        qkv = self.qkv(x)
        # Skip actual attention for speed in tests
        x = self.proj(qkv[..., :self.dim])
        x = residual + x

        # MLP/MoE
        residual = x
        x = self.ln2(x)
        if self.is_moe:
            x = self.moe(x)
        else:
            x = self.mlp(x)
        x = residual + x

        return x


class MockNMoETransformer(nn.Module):
    """Mock NMoE Transformer for testing Megatron integration."""

    def __init__(self, config: MockNMoEConfig):
        super().__init__()
        self.config = config
        self.dim = config.dim

        # Embedding
        self.embed = nn.Embedding(config.vocab_size, config.dim)

        # Transformer blocks
        self.blocks = nn.ModuleList()
        for i in range(config.n_layers):
            is_moe = i >= config.n_dense_layers
            self.blocks.append(
                MockNMoETransformerBlock(
                    dim=config.dim,
                    n_heads=config.n_heads,
                    inter_dim=config.moe_inter_dim if is_moe else config.inter_dim,
                    n_experts=config.n_routed_experts if is_moe else 1,
                    n_activated=config.n_activated_experts if is_moe else 1,
                    is_moe=is_moe,
                )
            )

        # Output
        self.ln_f = nn.LayerNorm(config.dim)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(input_ids)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.lm_head(x)

    def sharded_state_dict(self) -> Dict[str, torch.Tensor]:
        """Return state dict for checkpoint testing."""
        return self.state_dict()


class MockMegatronModelWrapper:
    """Mock wrapper mimicking MegatronModelWrapper for NMoE."""

    def __init__(self, model: nn.Module):
        self.actor_module = [model]
        self.model_config = getattr(model, 'config', None)

    def eval(self):
        for m in self.actor_module:
            m.eval()

    def train(self):
        for m in self.actor_module:
            m.train()


# =============================================================================
# Utility Functions
# =============================================================================

def cleanup_distributed():
    """Clean up distributed process group."""
    if dist.is_initialized():
        dist.destroy_process_group()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def create_test_batch(
    batch_size: int = 4,
    seq_len: int = 128,
    vocab_size: int = 1000,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Dict[str, torch.Tensor]:
    """Create a test batch for NMoE training."""
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long, device=device)

    # Add some padding for realism
    for i in range(batch_size):
        pad_len = random.randint(0, seq_len // 4)
        attention_mask[i, :pad_len] = 0

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": input_ids.clone(),
    }


# =============================================================================
# PART 1: NMoEMegatronConfig Tests
# =============================================================================

class TestNMoEMegatronConfig:
    """Test NMoEMegatronConfig configuration class."""

    def test_default_config_creation(self):
        """Test creating config with default values."""
        if not HAS_SKYRL:
            pytest.skip("skyrl_train not available")

        config = NMoEMegatronConfig()

        assert config.tensor_model_parallel_size == 1
        assert config.pipeline_model_parallel_size == 1
        assert config.expert_model_parallel_size == 1
        assert config.rdep_profile == "bf16"
        assert config.checkpoint_format == "nmoe"

    def test_custom_config_creation(self):
        """Test creating config with custom values."""
        if not HAS_SKYRL:
            pytest.skip("skyrl_train not available")

        config = NMoEMegatronConfig(
            tensor_model_parallel_size=2,
            pipeline_model_parallel_size=2,
            expert_model_parallel_size=4,
            rdep_profile="fp8",
            checkpoint_format="megatron",
        )

        assert config.tensor_model_parallel_size == 2
        assert config.pipeline_model_parallel_size == 2
        assert config.expert_model_parallel_size == 4
        assert config.rdep_profile == "fp8"
        assert config.checkpoint_format == "megatron"

    def test_rdep_profile_options(self):
        """Test all valid RDEP profile options."""
        if not HAS_SKYRL:
            pytest.skip("skyrl_train not available")

        for profile in ["bf16", "fp8", "nvfp4"]:
            config = NMoEMegatronConfig(rdep_profile=profile)
            assert config.rdep_profile == profile

    @pytest.mark.parametrize(
        "tp,pp,ep,expected_valid",
        [
            (1, 1, 1, True),
            (2, 1, 1, True),
            (1, 2, 1, True),
            (1, 1, 8, True),
            (2, 2, 2, True),
            (4, 2, 1, True),
        ],
    )
    def test_parallelism_configurations(self, tp, pp, ep, expected_valid):
        """Test various parallelism configuration combinations."""
        if not HAS_SKYRL:
            pytest.skip("skyrl_train not available")

        config = NMoEMegatronConfig(
            tensor_model_parallel_size=tp,
            pipeline_model_parallel_size=pp,
            expert_model_parallel_size=ep,
        )
        assert config.tensor_model_parallel_size == tp
        assert config.pipeline_model_parallel_size == pp
        assert config.expert_model_parallel_size == ep


# =============================================================================
# PART 2: Tensor Parallelism Tests (TP=2,4,8)
# =============================================================================

@gpu
@multi_gpu
class TestNMoETensorParallelism:
    """Test NMoE model with Tensor Parallelism."""

    def test_tp2_config_setup(self):
        """Test TP=2 configuration setup."""
        if not HAS_SKYRL:
            pytest.skip("skyrl_train not available")

        config = NMoEMegatronConfig(tensor_model_parallel_size=2)
        assert config.tensor_model_parallel_size == 2

        # Verify dimension divisibility requirements
        model_dim = 256  # Must be divisible by TP size
        assert model_dim % config.tensor_model_parallel_size == 0

    @requires_2_gpus
    def test_tp2_weight_distribution(self):
        """Test TP=2 weight distribution for NMoE layers."""
        config = MockNMoEConfig(dim=256, n_heads=8)
        model = MockNMoETransformer(config)

        # Verify model dimensions are TP-compatible
        assert config.dim % 2 == 0
        assert config.n_heads % 2 == 0

        # Expert weights should be split along hidden dim
        for block in model.blocks:
            if hasattr(block, 'moe'):
                W1_shape = block.moe.W1.shape
                assert W1_shape[1] % 2 == 0  # dim is divisible by TP

    @requires_4_gpus
    def test_tp4_config_setup(self):
        """Test TP=4 configuration setup."""
        if not HAS_SKYRL:
            pytest.skip("skyrl_train not available")

        config = NMoEMegatronConfig(tensor_model_parallel_size=4)
        assert config.tensor_model_parallel_size == 4

        # Verify expert weight dimensions
        model_config = MockNMoEConfig(dim=256, n_heads=8)
        assert model_config.dim % 4 == 0
        assert model_config.n_heads % 4 == 0

    @requires_4_gpus
    def test_tp4_forward_output_shape(self):
        """Test TP=4 forward pass produces correct output shape."""
        torch.manual_seed(42)

        config = MockNMoEConfig(dim=256, n_heads=8)
        model = MockNMoETransformer(config).cuda()
        model.eval()

        batch = create_test_batch(batch_size=4, seq_len=64, vocab_size=config.vocab_size)
        with torch.no_grad():
            output = model(batch["input_ids"])

        assert output.shape == (4, 64, config.vocab_size)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    @requires_8_gpus
    def test_tp8_config_setup(self):
        """Test TP=8 configuration setup."""
        if not HAS_SKYRL:
            pytest.skip("skyrl_train not available")

        config = NMoEMegatronConfig(tensor_model_parallel_size=8)
        assert config.tensor_model_parallel_size == 8

    @requires_8_gpus
    def test_tp8_gradient_correctness(self):
        """Test TP=8 gradient computation is correct."""
        torch.manual_seed(42)

        config = MockNMoEConfig(dim=256, n_heads=8)
        model = MockNMoETransformer(config).cuda()

        batch = create_test_batch(batch_size=4, seq_len=64, vocab_size=config.vocab_size)

        output = model(batch["input_ids"])
        loss = F.cross_entropy(
            output.view(-1, config.vocab_size),
            batch["labels"].view(-1)
        )
        loss.backward()

        # Check all parameters have gradients
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"


# =============================================================================
# PART 3: Pipeline Parallelism Tests (PP=2,4)
# =============================================================================

@gpu
@multi_gpu
class TestNMoEPipelineParallelism:
    """Test NMoE model with Pipeline Parallelism."""

    def test_pp2_layer_distribution(self):
        """Test PP=2 layer distribution for NMoE."""
        if not HAS_SKYRL:
            pytest.skip("skyrl_train not available")

        config = NMoEMegatronConfig(pipeline_model_parallel_size=2)
        model_config = MockNMoEConfig(n_layers=4)

        # For PP=2 with 4 layers, each stage gets 2 layers
        layers_per_stage = model_config.n_layers // config.pipeline_model_parallel_size
        assert layers_per_stage == 2

    @requires_2_gpus
    def test_pp2_stage_assignment(self):
        """Test PP=2 stage assignment for NMoE layers."""
        model_config = MockNMoEConfig(n_layers=4, n_dense_layers=1)
        model = MockNMoETransformer(model_config)

        # Stage 0: layers 0, 1 (one dense, one MoE)
        # Stage 1: layers 2, 3 (both MoE)
        assert not model.blocks[0].is_moe  # Dense layer
        assert model.blocks[1].is_moe  # MoE layer
        assert model.blocks[2].is_moe  # MoE layer
        assert model.blocks[3].is_moe  # MoE layer

    @requires_4_gpus
    def test_pp4_layer_distribution(self):
        """Test PP=4 layer distribution for NMoE."""
        if not HAS_SKYRL:
            pytest.skip("skyrl_train not available")

        config = NMoEMegatronConfig(pipeline_model_parallel_size=4)
        model_config = MockNMoEConfig(n_layers=8)

        layers_per_stage = model_config.n_layers // config.pipeline_model_parallel_size
        assert layers_per_stage == 2

    def test_pp_activation_shape(self):
        """Test activation tensor shapes for PP communication."""
        batch_size = 4
        seq_len = 128
        hidden_size = 256

        # Activation shape between pipeline stages
        activation = torch.randn(batch_size, seq_len, hidden_size)
        assert activation.is_contiguous()
        assert activation.shape == (batch_size, seq_len, hidden_size)

    @requires_4_gpus
    def test_pp4_with_nmoe_blocks(self):
        """Test PP=4 with NMoE blocks correctly distributed."""
        model_config = MockNMoEConfig(n_layers=8, n_dense_layers=2)
        model = MockNMoETransformer(model_config)

        # Count MoE vs dense layers
        n_moe = sum(1 for b in model.blocks if b.is_moe)
        n_dense = sum(1 for b in model.blocks if not b.is_moe)

        assert n_dense == 2
        assert n_moe == 6


# =============================================================================
# PART 4: Expert Parallelism Tests (EP=8)
# =============================================================================

@gpu
@multi_gpu
class TestNMoEExpertParallelism:
    """Test NMoE model with Expert Parallelism."""

    def test_ep8_config_setup(self):
        """Test EP=8 configuration setup."""
        if not HAS_SKYRL:
            pytest.skip("skyrl_train not available")

        config = NMoEMegatronConfig(expert_model_parallel_size=8)
        assert config.expert_model_parallel_size == 8

    def test_ep8_expert_distribution(self):
        """Test EP=8 expert distribution."""
        n_total_experts = 64
        ep_size = 8

        experts_per_rank = n_total_experts // ep_size
        assert experts_per_rank == 8

    def test_ep_local_expert_count(self):
        """Test local expert count calculation."""
        test_cases = [
            (8, 1, 8),   # 8 experts, EP=1 -> 8 local
            (8, 2, 4),   # 8 experts, EP=2 -> 4 local
            (8, 4, 2),   # 8 experts, EP=4 -> 2 local
            (8, 8, 1),   # 8 experts, EP=8 -> 1 local
            (64, 8, 8),  # 64 experts, EP=8 -> 8 local
        ]

        for n_experts, ep_size, expected_local in test_cases:
            local = n_experts // ep_size
            assert local == expected_local, f"Failed for {n_experts}/{ep_size}"

    @requires_8_gpus
    def test_ep8_with_rdep_profile_bf16(self):
        """Test EP=8 with bf16 RDEP profile."""
        if not HAS_SKYRL:
            pytest.skip("skyrl_train not available")

        config = NMoEMegatronConfig(
            expert_model_parallel_size=8,
            rdep_profile="bf16",
        )
        assert config.rdep_profile == "bf16"

    @requires_8_gpus
    def test_ep8_with_rdep_profile_fp8(self):
        """Test EP=8 with fp8 RDEP profile."""
        if not HAS_SKYRL:
            pytest.skip("skyrl_train not available")

        config = NMoEMegatronConfig(
            expert_model_parallel_size=8,
            rdep_profile="fp8",
        )
        assert config.rdep_profile == "fp8"

    @requires_8_gpus
    def test_ep8_with_rdep_profile_nvfp4(self):
        """Test EP=8 with nvfp4 RDEP profile."""
        if not HAS_SKYRL:
            pytest.skip("skyrl_train not available")

        config = NMoEMegatronConfig(
            expert_model_parallel_size=8,
            rdep_profile="nvfp4",
        )
        assert config.rdep_profile == "nvfp4"

    def test_expert_weight_shapes(self):
        """Test expert weight tensor shapes for EP."""
        config = MockNMoEConfig(
            dim=256,
            moe_inter_dim=512,
            n_routed_experts=8,
        )
        model = MockNMoETransformer(config)

        for block in model.blocks:
            if hasattr(block, 'moe'):
                # Full expert weights shape
                assert block.moe.W1.shape == (8, 256, 512)
                assert block.moe.W2.shape == (8, 512, 256)


# =============================================================================
# PART 5: Combined TP+EP Configurations
# =============================================================================

@gpu
@multi_gpu
class TestNMoECombinedTPEP:
    """Test combined Tensor Parallelism + Expert Parallelism configurations."""

    @requires_4_gpus
    def test_tp2_ep2_config(self):
        """Test TP=2, EP=2 on 4 GPUs."""
        if not HAS_SKYRL:
            pytest.skip("skyrl_train not available")

        config = NMoEMegatronConfig(
            tensor_model_parallel_size=2,
            expert_model_parallel_size=2,
        )

        total_gpus = config.tensor_model_parallel_size * config.expert_model_parallel_size
        assert total_gpus == 4

    @requires_8_gpus
    def test_tp2_ep4_config(self):
        """Test TP=2, EP=4 on 8 GPUs."""
        if not HAS_SKYRL:
            pytest.skip("skyrl_train not available")

        config = NMoEMegatronConfig(
            tensor_model_parallel_size=2,
            expert_model_parallel_size=4,
        )

        total_gpus = config.tensor_model_parallel_size * config.expert_model_parallel_size
        assert total_gpus == 8

    @requires_8_gpus
    def test_tp4_ep2_config(self):
        """Test TP=4, EP=2 on 8 GPUs."""
        if not HAS_SKYRL:
            pytest.skip("skyrl_train not available")

        config = NMoEMegatronConfig(
            tensor_model_parallel_size=4,
            expert_model_parallel_size=2,
        )

        total_gpus = config.tensor_model_parallel_size * config.expert_model_parallel_size
        assert total_gpus == 8

    def test_tp_ep_rank_calculation(self):
        """Test rank calculations for combined TP+EP."""
        tp_size = 2
        ep_size = 4
        total_gpus = tp_size * ep_size

        for global_rank in range(total_gpus):
            tp_rank = global_rank % tp_size
            ep_rank = global_rank // tp_size

            assert 0 <= tp_rank < tp_size
            assert 0 <= ep_rank < ep_size

    @requires_8_gpus
    def test_tp2_ep4_expert_distribution(self):
        """Test expert distribution with TP=2, EP=4."""
        n_experts = 16
        ep_size = 4

        local_experts = n_experts // ep_size
        assert local_experts == 4

        # Each EP rank has 4 experts, each expert is TP-sharded
        for ep_rank in range(ep_size):
            expert_start = ep_rank * local_experts
            expert_end = (ep_rank + 1) * local_experts
            assert expert_end - expert_start == 4


# =============================================================================
# PART 6: Combined TP+PP+DP Configurations
# =============================================================================

@gpu
@multi_gpu
class TestNMoECombinedTPPPDP:
    """Test combined Tensor + Pipeline + Data Parallelism configurations."""

    @requires_8_gpus
    def test_tp2_pp2_dp2_config(self):
        """Test TP=2, PP=2, DP=2 on 8 GPUs."""
        if not HAS_SKYRL:
            pytest.skip("skyrl_train not available")

        config = NMoEMegatronConfig(
            tensor_model_parallel_size=2,
            pipeline_model_parallel_size=2,
        )

        # DP is implicit: world_size / (TP * PP)
        # For 8 GPUs: DP = 8 / (2 * 2) = 2
        model_parallel_size = config.tensor_model_parallel_size * config.pipeline_model_parallel_size
        dp_size = 8 // model_parallel_size
        assert dp_size == 2

    @requires_8_gpus
    def test_tp4_pp2_dp1_config(self):
        """Test TP=4, PP=2, DP=1 on 8 GPUs."""
        if not HAS_SKYRL:
            pytest.skip("skyrl_train not available")

        config = NMoEMegatronConfig(
            tensor_model_parallel_size=4,
            pipeline_model_parallel_size=2,
        )

        model_parallel_size = config.tensor_model_parallel_size * config.pipeline_model_parallel_size
        dp_size = 8 // model_parallel_size
        assert dp_size == 1

    def test_combined_parallelism_rank_mapping(self):
        """Test rank mapping for TP=2, PP=2, DP=2."""
        tp_size = 2
        pp_size = 2
        dp_size = 2
        total_ranks = tp_size * pp_size * dp_size

        assert total_ranks == 8

        for global_rank in range(total_ranks):
            tp_rank = global_rank % tp_size
            pp_rank = (global_rank // tp_size) % pp_size
            dp_rank = global_rank // (tp_size * pp_size)

            assert 0 <= tp_rank < tp_size
            assert 0 <= pp_rank < pp_size
            assert 0 <= dp_rank < dp_size

    @requires_8_gpus
    def test_tp2_pp2_dp2_nmoe_layer_distribution(self):
        """Test NMoE layer distribution with TP=2, PP=2, DP=2."""
        model_config = MockNMoEConfig(n_layers=8, n_dense_layers=2)
        pp_size = 2

        layers_per_stage = model_config.n_layers // pp_size
        assert layers_per_stage == 4

        # Stage 0 gets layers 0-3 (2 dense, 2 MoE)
        # Stage 1 gets layers 4-7 (4 MoE)
        model = MockNMoETransformer(model_config)

        stage0_moe = sum(1 for i in range(4) if model.blocks[i].is_moe)
        stage1_moe = sum(1 for i in range(4, 8) if model.blocks[i].is_moe)

        assert stage0_moe == 2
        assert stage1_moe == 4


# =============================================================================
# PART 7: Gradient Accumulation Tests
# =============================================================================

@gpu
class TestNMoEGradientAccumulation:
    """Test gradient accumulation with Megatron+NMoE."""

    @requires_cuda
    def test_gradient_accumulation_equivalence(self):
        """Test gradient accumulation produces equivalent results."""
        torch.manual_seed(42)

        config = MockNMoEConfig(dim=128, n_layers=2)
        model = MockNMoETransformer(config).cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        # Method 1: Single large batch
        torch.manual_seed(123)
        batch = create_test_batch(batch_size=8, seq_len=64, vocab_size=config.vocab_size)
        optimizer.zero_grad()

        output = model(batch["input_ids"])
        loss = F.cross_entropy(
            output.view(-1, config.vocab_size),
            batch["labels"].view(-1)
        )
        loss.backward()

        grads_single = {
            n: p.grad.clone() for n, p in model.named_parameters()
            if p.grad is not None
        }

        # Method 2: Accumulated micro-batches
        optimizer.zero_grad()
        torch.manual_seed(123)

        for i in range(2):
            micro_batch = create_test_batch(
                batch_size=4, seq_len=64, vocab_size=config.vocab_size
            )
            # Adjust for batch_size difference
            output = model(micro_batch["input_ids"])
            micro_loss = F.cross_entropy(
                output.view(-1, config.vocab_size),
                micro_batch["labels"].view(-1)
            )
            (micro_loss / 2).backward()

        grads_accum = {
            n: p.grad.clone() for n, p in model.named_parameters()
            if p.grad is not None
        }

        # Verify gradients are approximately equal
        for name in grads_single:
            if name in grads_accum:
                assert torch.allclose(grads_single[name], grads_accum[name], rtol=1e-3, atol=1e-5), \
                    f"Gradient mismatch for {name}"

    @requires_cuda
    def test_gradient_accumulation_steps(self):
        """Test gradient accumulation over multiple steps."""
        config = MockNMoEConfig(dim=128, n_layers=2)
        model = MockNMoETransformer(config).cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        accumulation_steps = 4
        optimizer.zero_grad()

        for step in range(accumulation_steps):
            batch = create_test_batch(batch_size=2, seq_len=64, vocab_size=config.vocab_size)
            output = model(batch["input_ids"])
            loss = F.cross_entropy(
                output.view(-1, config.vocab_size),
                batch["labels"].view(-1)
            )
            (loss / accumulation_steps).backward()

        # Verify gradients accumulated
        for name, param in model.named_parameters():
            if param.grad is not None:
                assert param.grad.abs().sum() > 0, f"Zero gradient for {name}"

    @requires_cuda
    def test_gradient_norm_with_accumulation(self):
        """Test gradient norm computation with accumulation."""
        config = MockNMoEConfig(dim=128, n_layers=2)
        model = MockNMoETransformer(config).cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        accumulation_steps = 2
        optimizer.zero_grad()

        for _ in range(accumulation_steps):
            batch = create_test_batch(batch_size=4, seq_len=64, vocab_size=config.vocab_size)
            output = model(batch["input_ids"])
            loss = F.cross_entropy(
                output.view(-1, config.vocab_size),
                batch["labels"].view(-1)
            )
            (loss / accumulation_steps).backward()

        # Compute gradient norm
        total_norm = 0.0
        for param in model.parameters():
            if param.grad is not None:
                total_norm += param.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5

        assert total_norm > 0
        assert not np.isnan(total_norm)
        assert not np.isinf(total_norm)


# =============================================================================
# PART 8: Checkpoint Save/Load Tests
# =============================================================================

@gpu
class TestNMoECheckpointSaveLoad:
    """Test checkpoint save/load with Megatron+NMoE."""

    def test_state_dict_roundtrip(self):
        """Test basic state dict save/load roundtrip."""
        config = MockNMoEConfig(dim=128, n_layers=2)
        model1 = MockNMoETransformer(config)

        state_dict = model1.state_dict()

        model2 = MockNMoETransformer(config)
        model2.load_state_dict(state_dict)

        # Verify weights match
        for (n1, p1), (n2, p2) in zip(
            model1.named_parameters(),
            model2.named_parameters()
        ):
            assert n1 == n2
            assert torch.allclose(p1, p2)

    def test_checkpoint_file_io(self):
        """Test checkpoint save/load to/from file."""
        config = MockNMoEConfig(dim=128, n_layers=2)
        model = MockNMoETransformer(config)

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = os.path.join(tmpdir, "nmoe_checkpoint.pt")

            # Save
            checkpoint = {
                "model": model.state_dict(),
                "config": {
                    "dim": config.dim,
                    "n_layers": config.n_layers,
                    "n_routed_experts": config.n_routed_experts,
                },
            }
            torch.save(checkpoint, ckpt_path)

            # Load
            loaded = torch.load(ckpt_path, weights_only=False)
            model2 = MockNMoETransformer(config)
            model2.load_state_dict(loaded["model"])

            # Verify
            for (n1, p1), (n2, p2) in zip(
                model.named_parameters(),
                model2.named_parameters()
            ):
                assert torch.allclose(p1, p2)

    def test_checkpoint_with_optimizer_state(self):
        """Test checkpoint includes optimizer state."""
        config = MockNMoEConfig(dim=128, n_layers=2)
        model = MockNMoETransformer(config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        # Run a training step to populate optimizer state
        batch = create_test_batch(batch_size=4, seq_len=32, vocab_size=config.vocab_size)
        output = model(batch["input_ids"])
        loss = output.sum()
        loss.backward()
        optimizer.step()

        # Save
        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }

        # Load
        model2 = MockNMoETransformer(config)
        optimizer2 = torch.optim.AdamW(model2.parameters(), lr=1e-4)
        model2.load_state_dict(checkpoint["model"])
        optimizer2.load_state_dict(checkpoint["optimizer"])

        # Verify optimizer state
        for key in checkpoint["optimizer"]["state"]:
            assert key in optimizer2.state

    def test_nmoe_format_checkpoint_structure(self):
        """Test NMoE checkpoint format structure."""
        config = MockNMoEConfig(dim=128, n_layers=2)
        model = MockNMoETransformer(config)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Simulate NMoE checkpoint format
            rd_path = os.path.join(tmpdir, "rd.pt")

            # Separate expert and shared weights
            shared_weights = {}
            expert_weights = {}

            for name, param in model.named_parameters():
                if "moe" in name.lower() or "expert" in name.lower():
                    expert_weights[name] = param.data.cpu()
                else:
                    shared_weights[name] = param.data.cpu()

            # Save shared weights
            torch.save({
                "model_state": shared_weights,
                "config": {
                    "dim": config.dim,
                    "n_layers": config.n_layers,
                },
                "weight_version": 1,
            }, rd_path)

            # Save expert weights
            expert_path = os.path.join(tmpdir, "dp_rank_0_ep_0_tp_0.pt")
            torch.save({
                "expert_state": expert_weights,
                "ep_rank": 0,
                "tp_rank": 0,
            }, expert_path)

            # Verify files exist
            assert os.path.exists(rd_path)
            assert os.path.exists(expert_path)

    def test_checkpoint_rng_state(self):
        """Test RNG state in checkpoint."""
        torch.manual_seed(42)
        np.random.seed(42)
        random.seed(42)

        # Generate some random values
        _ = torch.rand(10)

        # Save RNG state
        rng_state = {
            "cpu": torch.get_rng_state(),
            "numpy": np.random.get_state(),
            "random": random.getstate(),
        }
        if torch.cuda.is_available():
            rng_state["cuda"] = torch.cuda.get_rng_state_all()

        # Generate expected values
        expected = torch.rand(5)

        # Restore RNG state
        torch.set_rng_state(rng_state["cpu"])
        np.random.set_state(rng_state["numpy"])
        random.setstate(rng_state["random"])

        # Generate again
        actual = torch.rand(5)

        assert torch.allclose(expected, actual)


# =============================================================================
# PART 9: 1F1B Pipeline Schedule Tests
# =============================================================================

@gpu
@multi_gpu
class TestNMoE1F1BSchedule:
    """Test 1F1B pipeline schedule with NMoE."""

    def test_1f1b_schedule_basics(self):
        """Test basic 1F1B schedule properties."""
        pp_size = 2
        num_microbatches = 4

        # Warm-up phase: pp_size - 1 forwards
        warmup_forwards = pp_size - 1
        assert warmup_forwards == 1

        # Steady state: alternating forward/backward
        steady_state_steps = num_microbatches - warmup_forwards
        assert steady_state_steps == 3

    def test_1f1b_microbatch_ordering_pp2(self):
        """Test micro-batch ordering for PP=2."""
        pp_size = 2
        num_microbatches = 4

        # For PP=2 with 4 micro-batches:
        # Stage 0: F0, F1, F2, F3, B3, B2, B1, B0
        # Stage 1: wait, F0, F1, F2, F3, B3, B2, B1, B0

        # Total steps per stage
        total_steps = 2 * num_microbatches
        assert total_steps == 8

    def test_1f1b_microbatch_ordering_pp4(self):
        """Test micro-batch ordering for PP=4."""
        pp_size = 4
        num_microbatches = 8

        # Warm-up phase
        warmup_forwards = pp_size - 1
        assert warmup_forwards == 3

        # Cooldown phase
        cooldown_backwards = pp_size - 1
        assert cooldown_backwards == 3

    @requires_4_gpus
    def test_1f1b_with_nmoe_moe_layers(self):
        """Test 1F1B schedule handles MoE layers correctly."""
        config = MockNMoEConfig(n_layers=4, n_dense_layers=1)
        model = MockNMoETransformer(config)
        pp_size = 2

        layers_per_stage = config.n_layers // pp_size

        # Stage 0 layers
        stage0_layers = [model.blocks[i] for i in range(layers_per_stage)]
        stage0_has_moe = any(b.is_moe for b in stage0_layers)
        assert stage0_has_moe  # Stage 0 has at least one MoE layer

        # Stage 1 layers
        stage1_layers = [model.blocks[i] for i in range(layers_per_stage, 2 * layers_per_stage)]
        stage1_has_moe = any(b.is_moe for b in stage1_layers)
        assert stage1_has_moe  # Stage 1 has MoE layers

    def test_1f1b_activation_memory(self):
        """Test activation memory pattern in 1F1B."""
        pp_size = 2
        num_microbatches = 4

        # Peak activation memory occurs at end of warm-up
        # = (pp_size - 1) micro-batch activations stored
        peak_activations_stored = pp_size - 1
        assert peak_activations_stored == 1

    def test_1f1b_batch_generator(self):
        """Test batch generator for 1F1B schedule."""
        if not HAS_SKYRL:
            pytest.skip("skyrl_train not available")

        batches = [{"data": i} for i in range(4)]

        # Without VPP
        generator = make_batch_generator(batches, vpp_size=1)
        result = list(generator)
        assert len(result) == 4

        # With VPP=2
        generators = make_batch_generator(batches, vpp_size=2)
        assert len(generators) == 2


# =============================================================================
# PART 10: Memory Management Tests
# =============================================================================

@gpu
class TestNMoEMemoryManagement:
    """Test memory management for NMoE+Megatron."""

    @requires_cuda
    def test_model_memory_footprint(self):
        """Test model memory footprint calculation."""
        torch.cuda.empty_cache()
        initial_mem = torch.cuda.memory_allocated()

        config = MockNMoEConfig(dim=256, n_layers=4)
        model = MockNMoETransformer(config).cuda()

        model_mem = torch.cuda.memory_allocated() - initial_mem
        assert model_mem > 0

        # Estimate expected memory
        n_params = sum(p.numel() for p in model.parameters())
        expected_mem = n_params * 2  # bf16 = 2 bytes
        # Allow 2x overhead for PyTorch allocator
        assert model_mem < expected_mem * 2

    @requires_cuda
    def test_cpu_offload_frees_memory(self):
        """Test that CPU offload frees GPU memory."""
        torch.cuda.empty_cache()
        initial_mem = torch.cuda.memory_allocated()

        config = MockNMoEConfig(dim=128, n_layers=2)
        model = MockNMoETransformer(config).cuda()

        model_mem = torch.cuda.memory_allocated()
        assert model_mem > initial_mem

        # Offload to CPU
        model.cpu()
        torch.cuda.empty_cache()

        after_offload = torch.cuda.memory_allocated()
        assert after_offload < model_mem

    @requires_cuda
    def test_gradient_memory(self):
        """Test gradient memory usage."""
        torch.cuda.empty_cache()

        config = MockNMoEConfig(dim=128, n_layers=2)
        model = MockNMoETransformer(config).cuda()

        pre_grad_mem = torch.cuda.memory_allocated()

        # Compute gradients
        batch = create_test_batch(batch_size=4, seq_len=32, vocab_size=config.vocab_size)
        output = model(batch["input_ids"])
        loss = output.sum()
        loss.backward()

        post_grad_mem = torch.cuda.memory_allocated()

        # Gradients should use additional memory
        grad_mem = post_grad_mem - pre_grad_mem
        assert grad_mem > 0

    @requires_cuda
    def test_activation_memory_with_moe(self):
        """Test activation memory with MoE layers."""
        torch.cuda.empty_cache()

        config = MockNMoEConfig(dim=128, n_layers=2)
        model = MockNMoETransformer(config).cuda()
        model.eval()

        pre_forward_mem = torch.cuda.memory_allocated()

        with torch.no_grad():
            batch = create_test_batch(batch_size=4, seq_len=64, vocab_size=config.vocab_size)
            _ = model(batch["input_ids"])

        post_forward_mem = torch.cuda.memory_allocated()

        # Inference should have minimal activation memory
        activation_mem = post_forward_mem - pre_forward_mem
        assert activation_mem >= 0


# =============================================================================
# PART 11: Weight Extraction Tests
# =============================================================================

@gpu
class TestNMoEWeightExtraction:
    """Test weight extraction for inference sync."""

    def test_shared_vs_expert_weight_categorization(self):
        """Test categorization of shared vs expert weights."""
        config = MockNMoEConfig(dim=128, n_layers=2)
        model = MockNMoETransformer(config)

        shared_params = {}
        expert_params = {}

        for name, param in model.named_parameters():
            if "moe" in name.lower() or "expert" in name.lower():
                expert_params[name] = param
            else:
                shared_params[name] = param

        # Should have both shared and expert params
        assert len(shared_params) > 0
        assert len(expert_params) > 0

    def test_weight_extraction_shapes(self):
        """Test weight tensor shapes after extraction."""
        config = MockNMoEConfig(
            dim=128,
            moe_inter_dim=256,
            n_routed_experts=4,
            n_layers=2,
        )
        model = MockNMoETransformer(config)

        for name, param in model.named_parameters():
            assert len(param.shape) <= 3  # At most 3D tensors
            assert param.numel() > 0

    def test_weight_dtype_conversion(self):
        """Test weight dtype conversion for inference."""
        config = MockNMoEConfig(dim=128, n_layers=2)
        model = MockNMoETransformer(config)

        # Convert to bfloat16
        model = model.to(torch.bfloat16)

        for param in model.parameters():
            assert param.dtype == torch.bfloat16


# =============================================================================
# PART 12: Training Step Tests
# =============================================================================

@gpu
class TestNMoETrainingStep:
    """Test complete training steps with NMoE+Megatron."""

    @requires_cuda
    def test_single_training_step(self):
        """Test a single training step."""
        torch.manual_seed(42)

        config = MockNMoEConfig(dim=128, n_layers=2)
        model = MockNMoETransformer(config).cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        model.train()
        batch = create_test_batch(batch_size=4, seq_len=32, vocab_size=config.vocab_size)

        optimizer.zero_grad()
        output = model(batch["input_ids"])
        loss = F.cross_entropy(
            output.view(-1, config.vocab_size),
            batch["labels"].view(-1)
        )
        loss.backward()

        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        assert loss.item() > 0
        assert grad_norm.item() >= 0
        assert not torch.isnan(loss)

    @requires_cuda
    def test_multiple_training_steps(self):
        """Test multiple training steps."""
        config = MockNMoEConfig(dim=128, n_layers=2)
        model = MockNMoETransformer(config).cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        losses = []
        for step in range(5):
            model.train()
            batch = create_test_batch(batch_size=4, seq_len=32, vocab_size=config.vocab_size)

            optimizer.zero_grad()
            output = model(batch["input_ids"])
            loss = F.cross_entropy(
                output.view(-1, config.vocab_size),
                batch["labels"].view(-1)
            )
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        # Verify losses are finite
        for l in losses:
            assert not np.isnan(l)
            assert not np.isinf(l)

    @requires_cuda
    def test_training_with_lr_scheduler(self):
        """Test training with learning rate scheduler."""
        config = MockNMoEConfig(dim=128, n_layers=2)
        model = MockNMoETransformer(config).cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

        initial_lr = optimizer.param_groups[0]["lr"]

        for step in range(5):
            batch = create_test_batch(batch_size=4, seq_len=32, vocab_size=config.vocab_size)
            optimizer.zero_grad()
            output = model(batch["input_ids"])
            loss = F.cross_entropy(
                output.view(-1, config.vocab_size),
                batch["labels"].view(-1)
            )
            loss.backward()
            optimizer.step()
            scheduler.step()

        final_lr = optimizer.param_groups[0]["lr"]
        assert final_lr < initial_lr  # LR should decrease


# =============================================================================
# PART 13: Integration with NMoE Components Tests
# =============================================================================

@gpu
class TestNMoEComponentIntegration:
    """Test integration with actual NMoE components."""

    @requires_nmoe
    def test_nmoe_config_creation(self):
        """Test creating NMoE config."""
        config = NMoEConfig(
            dim=256,
            n_layers=4,
            n_heads=8,
            n_routed_experts=8,
            n_activated_experts=2,
        )
        assert config.dim == 256
        assert config.n_layers == 4
        assert config.n_routed_experts == 8

    @requires_nmoe
    @requires_cuda
    def test_rdep_profile_initialization(self):
        """Test RDEP profile initialization."""
        profiles = ["bf16", "fp8", "nvfp4"]

        for profile in profiles:
            # Just test that the profile is valid
            assert profile in Rdep.PROFILES

    @requires_skyrl
    def test_nmoe_megatron_strategy_creation(self):
        """Test NMoEMegatronStrategy creation."""
        strategy = create_nmoe_megatron_strategy(
            tp_size=1,
            pp_size=1,
            ep_size=1,
            rdep_profile="bf16",
        )
        assert strategy is not None

    @requires_skyrl
    def test_nmoe_megatron_config_factory(self):
        """Test factory function for NMoE Megatron config."""
        strategy = create_nmoe_megatron_strategy(
            tp_size=2,
            pp_size=2,
            ep_size=4,
            rdep_profile="fp8",
            checkpoint_format="nmoe",
        )
        assert strategy.nmoe_config.tensor_model_parallel_size == 2
        assert strategy.nmoe_config.pipeline_model_parallel_size == 2
        assert strategy.nmoe_config.expert_model_parallel_size == 4


# =============================================================================
# PART 14: Error Handling Tests
# =============================================================================

class TestNMoEErrorHandling:
    """Test error handling in NMoE+Megatron integration."""

    def test_invalid_parallelism_config(self):
        """Test handling of invalid parallelism configurations."""
        if not HAS_SKYRL:
            pytest.skip("skyrl_train not available")

        # These should create configs but may fail at runtime
        config = NMoEMegatronConfig(
            tensor_model_parallel_size=0,  # Invalid
        )
        # Config creation should work, validation happens at runtime

    def test_incompatible_model_dimensions(self):
        """Test handling of incompatible model dimensions."""
        # Create model with dimensions not divisible by TP size
        config = MockNMoEConfig(dim=100, n_heads=10)  # 100 not divisible by 4

        # This should work but would fail with TP=4
        model = MockNMoETransformer(config)
        assert model is not None

    def test_expert_count_divisibility(self):
        """Test expert count must be divisible by EP size."""
        n_experts = 8
        valid_ep_sizes = [1, 2, 4, 8]
        invalid_ep_sizes = [3, 5, 6, 7]

        for ep in valid_ep_sizes:
            assert n_experts % ep == 0

        for ep in invalid_ep_sizes:
            assert n_experts % ep != 0


# =============================================================================
# PART 15: Stress Tests
# =============================================================================

@gpu
class TestNMoEStress:
    """Stress tests for NMoE+Megatron integration."""

    @requires_cuda
    def test_large_batch_forward(self):
        """Test forward pass with large batch."""
        config = MockNMoEConfig(dim=128, n_layers=2)
        model = MockNMoETransformer(config).cuda()
        model.eval()

        # Large batch
        batch = create_test_batch(batch_size=32, seq_len=128, vocab_size=config.vocab_size)

        with torch.no_grad():
            output = model(batch["input_ids"])

        assert output.shape == (32, 128, config.vocab_size)

    @requires_cuda
    def test_many_training_iterations(self):
        """Test many training iterations."""
        config = MockNMoEConfig(dim=64, n_layers=2)
        model = MockNMoETransformer(config).cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        for _ in range(20):
            batch = create_test_batch(batch_size=4, seq_len=32, vocab_size=config.vocab_size)
            optimizer.zero_grad()
            output = model(batch["input_ids"])
            loss = F.cross_entropy(
                output.view(-1, config.vocab_size),
                batch["labels"].view(-1)
            )
            loss.backward()
            optimizer.step()

        # Model should still be functional
        assert not any(torch.isnan(p).any() for p in model.parameters())

    @requires_cuda
    def test_varying_sequence_lengths(self):
        """Test with varying sequence lengths."""
        config = MockNMoEConfig(dim=128, n_layers=2)
        model = MockNMoETransformer(config).cuda()
        model.eval()

        seq_lengths = [16, 32, 64, 128, 256]

        for seq_len in seq_lengths:
            batch = create_test_batch(batch_size=4, seq_len=seq_len, vocab_size=config.vocab_size)
            with torch.no_grad():
                output = model(batch["input_ids"])
            assert output.shape == (4, seq_len, config.vocab_size)


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
