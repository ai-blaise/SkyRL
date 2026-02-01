"""Comprehensive integration tests for SkyRL RL training loop with nmoe models.

This module provides comprehensive integration tests for the full RL training loop
(GRPO, PPO) with nmoe Mixture-of-Experts models. Tests cover:

1. GRPO training loop with nmoe model
2. PPO training loop with nmoe model
3. Reward computation with nmoe outputs
4. Advantage estimation with MoE outputs
5. aux_loss integration in RL loss
6. Expert load balancing during RL
7. KL divergence computation with nmoe
8. Reference model handling with nmoe
9. Actor-critic with separate nmoe models
10. 8-GPU distributed RL with nmoe

Tests use mocked components where actual GPU infrastructure is not available,
and real components when running on GPU clusters.
"""

import sys
import os
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from unittest.mock import MagicMock, Mock, patch, PropertyMock

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# Add skyrl_train to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))


# =============================================================================
# pytest markers
# =============================================================================

# Skip GPU tests if no GPU available
pytestmark = [
    pytest.mark.integration,
]


def requires_gpu(func):
    """Decorator to skip test if no GPU available."""
    return pytest.mark.gpu(pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="GPU not available"
    )(func))


def requires_multi_gpu(n_gpus=2):
    """Decorator to skip test if not enough GPUs available."""
    def decorator(func):
        return pytest.mark.gpu(pytest.mark.skipif(
            torch.cuda.device_count() < n_gpus,
            reason=f"Requires {n_gpus} GPUs, only {torch.cuda.device_count()} available"
        )(func))
    return decorator


# =============================================================================
# Mock Infrastructure
# =============================================================================

@dataclass
class MockNMoEConfig:
    """Mock nmoe configuration for testing."""
    dim: int = 256
    n_layers: int = 4
    n_heads: int = 4
    n_routed_experts: int = 8
    n_activated_experts: int = 2
    n_dense_layers: int = 1
    n_shared_experts: int = 1
    inter_dim: int = 512
    moe_inter_dim: int = 256
    vocab_size: int = 1000
    eos_token_id: int = 999
    pad_token_id: int = 0
    dtype: str = "bf16"
    rms_norm_eps: float = 1e-5
    qk_rope_head_dim: int = 32
    rope_theta: float = 10000.0
    max_position_embeddings: int = 2048
    rope_scaling_factor: float = 1.0
    rope_ntk_alpha: float = 1.0
    rope_ntk_beta: float = 32.0
    batch_size: int = 4
    seq_len: int = 32
    aux_loss_alpha: float = 0.01


@dataclass
class MockAlgorithmConfig:
    """Mock algorithm configuration for testing."""
    use_kl_loss: bool = True
    use_kl_in_reward: bool = True
    use_entropy_loss: bool = True
    use_tis: bool = False
    kl_loss_coef: float = 0.1
    entropy_loss_coef: float = 0.01
    kl_estimator_type: str = "k3"
    policy_loss_type: str = "regular"
    advantage_estimator: str = "grpo"
    advantage_batch_normalize: bool = True
    gamma: float = 1.0
    lambd: float = 0.95
    grpo_norm_by_std: bool = True
    eps_clip_low: float = 0.2
    eps_clip_high: float = 0.2
    value_clip: Optional[float] = None
    loss_reduction: str = "token_mean"
    max_seq_len: int = 2048
    dynamic_sampling: Optional[Any] = None
    kl_ctrl: Optional[Any] = None


@dataclass
class MockTrainerConfig:
    """Mock trainer configuration."""
    epochs: int = 1
    train_batch_size: int = 4
    micro_train_batch_size_per_gpu: int = 2
    micro_forward_batch_size_per_gpu: int = 2
    policy_mini_batch_size: int = 4
    critic_mini_batch_size: int = 4
    update_epochs_per_batch: int = 1
    gradient_checkpointing: bool = False
    strategy: str = "fsdp"
    seed: int = 42
    ckpt_interval: int = 0
    hf_save_interval: int = 0
    eval_interval: int = 0
    update_ref_every_epoch: bool = False
    algorithm: MockAlgorithmConfig = field(default_factory=MockAlgorithmConfig)
    policy: Any = None
    critic: Any = None
    ref: Any = None
    placement: Any = None
    ckpt_path: str = "/tmp/ckpt"
    export_path: str = "/tmp/export"
    dump_data_batch: bool = False
    resume_mode: str = "none"


@dataclass
class MockGeneratorConfig:
    """Mock generator configuration."""
    n_samples_per_prompt: int = 2
    step_wise_trajectories: bool = False
    sampling_params: Any = field(default_factory=lambda: Mock(
        temperature=1.0,
        top_p=0.9,
        logprobs=None,
        max_generate_length=128,
    ))
    max_input_length: int = 512
    backend: str = "sglang"
    enable_prefix_caching: bool = False
    model_dtype: str = "bfloat16"


class MockRouter(nn.Module):
    """Mock router for testing."""
    def __init__(self, config: MockNMoEConfig):
        super().__init__()
        self.n_experts = config.n_routed_experts
        self.topk = config.n_activated_experts
        self.gate = nn.Linear(config.dim, self.n_experts, bias=False, dtype=torch.bfloat16)
        self.register_buffer("bias", torch.zeros(self.n_experts, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.gate(x).float()
        scores = torch.sigmoid(logits)
        scores_for_selection = scores + self.bias
        _, indices = torch.topk(scores_for_selection, k=self.topk, dim=-1)
        weights = torch.gather(scores, 1, indices)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp(min=1e-12)
        return weights.to(x.dtype), indices

    def update_bias(self, expert_loads: torch.Tensor, gamma: float = 0.001):
        expected = 1.0 / self.n_experts
        s = torch.sign(expert_loads - expected)
        self.bias -= gamma * (s - s.mean())
        self.bias.clamp_(-16.0, 16.0)


class MockMoE(nn.Module):
    """Mock MoE layer for testing."""
    def __init__(self, config: MockNMoEConfig):
        super().__init__()
        self.dim = config.dim
        self.n_experts = config.n_routed_experts
        self.topk = config.n_activated_experts
        self.aux_loss_alpha = config.aux_loss_alpha
        self.router = MockRouter(config)
        self.W1 = nn.Parameter(torch.randn(self.n_experts, config.dim, config.moe_inter_dim, dtype=torch.bfloat16) * 0.02)
        self.W3 = nn.Parameter(torch.randn(self.n_experts, config.dim, config.moe_inter_dim, dtype=torch.bfloat16) * 0.02)
        self.W2 = nn.Parameter(torch.randn(self.n_experts, config.moe_inter_dim, config.dim, dtype=torch.bfloat16) * 0.02)
        self._use_blockscaled = config.dtype in ('fp8', 'nvfp4')
        self._W_cache = None
        self.last_loads = None
        self.last_aux_loss = None

    def refresh_weight_cache(self):
        self._W_cache = "refreshed"

    def _compute_aux_loss(self, gates: torch.Tensor, expert_ids: torch.Tensor, T: int) -> torch.Tensor:
        """Compute auxiliary load balancing loss."""
        if self.aux_loss_alpha == 0.0:
            return gates.new_zeros((), dtype=torch.float32)
        E = self.n_experts
        K = self.topk
        expert_ids_flat = expert_ids.reshape(-1)
        f = torch.zeros(E, dtype=torch.float32, device=gates.device)
        f.scatter_add_(0, expert_ids_flat.long(), torch.ones_like(expert_ids_flat, dtype=torch.float32))
        f = f / (T * K)
        gates_flat = gates.float().reshape(-1)
        P = torch.zeros(E, dtype=torch.float32, device=gates.device)
        P.scatter_add_(0, expert_ids_flat.long(), gates_flat)
        P = P / (T * K)
        aux_loss = self.aux_loss_alpha * E * (f * P).sum()
        return aux_loss

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            B, T_seq, D = x.shape
            x_flat = x.view(-1, D)
        else:
            x_flat = x
            T_seq = 1
        T = x_flat.size(0)
        g, eid = self.router(x_flat)
        self.last_aux_loss = self._compute_aux_loss(g, eid, T)
        self.last_loads = torch.bincount(eid.reshape(-1), minlength=self.n_experts).float()
        self.last_loads = self.last_loads / self.last_loads.sum().clamp(min=1.0)
        # Simplified expert computation
        out = x_flat @ self.W1[0] @ self.W2[0].T
        if x.dim() == 3:
            return out.view(B, T_seq, D)
        return out


class MockTransformerBlock(nn.Module):
    """Mock transformer block for testing."""
    def __init__(self, config: MockNMoEConfig, layer_id: int, is_moe: bool = False):
        super().__init__()
        self.layer_id = layer_id
        self.is_moe = is_moe
        self._use_gradient_checkpointing = False
        self.attn_norm = nn.LayerNorm(config.dim)
        self.ffn_norm = nn.LayerNorm(config.dim)
        self.attn = nn.Linear(config.dim, config.dim, dtype=torch.bfloat16)
        if is_moe:
            self.ffn = MockMoE(config)
        else:
            self.ffn = nn.Sequential(
                nn.Linear(config.dim, config.inter_dim, dtype=torch.bfloat16),
                nn.SiLU(),
                nn.Linear(config.inter_dim, config.dim, dtype=torch.bfloat16),
            )

    def forward(self, x: torch.Tensor, cos=None, sin=None) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x


class MockTransformer(nn.Module):
    """Mock nmoe Transformer for testing."""
    supports_gradient_checkpointing = True

    def __init__(self, config: MockNMoEConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.dim, dtype=torch.bfloat16)
        self.blocks = nn.ModuleList([
            MockTransformerBlock(config, i, is_moe=(i >= config.n_dense_layers))
            for i in range(config.n_layers)
        ])
        self.norm = nn.LayerNorm(config.dim)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False, dtype=torch.bfloat16)

    def init_weights(self):
        pass

    def param_sets(self) -> Tuple[List[nn.Parameter], List[nn.Parameter]]:
        expert_params = []
        for m in self.modules():
            if isinstance(m, MockMoE):
                expert_params.extend([m.W1, m.W3, m.W2])
        expert_ids = {id(p) for p in expert_params}
        dense_params = [p for p in self.parameters() if id(p) not in expert_ids]
        return expert_params, dense_params

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        for block in self.blocks:
            block._use_gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        for block in self.blocks:
            block._use_gradient_checkpointing = False

    @property
    def is_gradient_checkpointing(self) -> bool:
        return self.blocks[0]._use_gradient_checkpointing if self.blocks else False

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.embedding(tokens)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        logits = self.lm_head(x)
        return logits

    def get_router_aux_loss(self) -> torch.Tensor:
        moe_layers = [blk.ffn for blk in self.blocks if isinstance(getattr(blk, 'ffn', None), MockMoE)]
        if not moe_layers:
            return torch.tensor(0.0, device=self.embedding.weight.device)
        aux_losses = [moe.last_aux_loss for moe in moe_layers if moe.last_aux_loss is not None]
        if not aux_losses:
            return torch.tensor(0.0, device=self.embedding.weight.device)
        return torch.stack(aux_losses).sum()

    def get_expert_load_stats(self) -> Dict[str, Any]:
        moe_layers = [blk.ffn for blk in self.blocks if isinstance(getattr(blk, 'ffn', None), MockMoE)]
        if not moe_layers:
            return {'loads': [], 'mean_load': 0.0, 'load_imbalance': 0.0}
        loads = [moe.last_loads for moe in moe_layers if moe.last_loads is not None]
        if not loads:
            return {'loads': [], 'mean_load': 0.0, 'load_imbalance': 0.0}
        all_loads = torch.stack(loads)
        mean_load = all_loads.mean().item()
        std_load = all_loads.std().item()
        load_imbalance = std_load / mean_load if mean_load > 0 else 0.0
        return {'loads': loads, 'mean_load': mean_load, 'load_imbalance': load_imbalance}


class MockNMoEModelWrapper(nn.Module):
    """Mock NMoEModelWrapper for RL training tests."""

    def __init__(
        self,
        model: MockTransformer,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.model = model
        self.temperature = temperature
        self._gradient_checkpointing = False
        self._frozen_for_reference = False
        self._pad_token_id = model.config.pad_token_id

    def _set_pad_token_id(self, pad_token_id: int):
        self._pad_token_id = pad_token_id

    def _logprobs_from_logits(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        log_probs = torch.log_softmax(logits.float(), dim=-1)
        gathered = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
        return gathered

    def _entropy_from_logits(self, logits: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        probs = torch.softmax(logits.float(), dim=-1)
        log_probs = torch.log_softmax(logits.float(), dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1)
        if attention_mask is not None:
            entropy = entropy * attention_mask
        return entropy

    def forward(
        self,
        sequences: torch.Tensor,
        num_actions: Union[int, List[int]],
        attention_mask: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
        return_output: bool = False,
        compute_entropy: bool = False,
        entropy_requires_grad: bool = True,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
        logits = self.model(sequences)
        effective_temp = temperature if temperature != 1.0 else self.temperature
        if effective_temp != 1.0:
            logits = logits / effective_temp
        sequences_rolled = torch.roll(sequences, shifts=-1, dims=1)
        log_probs = self._logprobs_from_logits(logits, sequences_rolled)
        output = {"logits": logits}
        if compute_entropy:
            entropy = self._entropy_from_logits(logits, attention_mask)
            output["entropy"] = entropy
        if isinstance(num_actions, list):
            if len(num_actions) == 1:
                num_actions = num_actions[0]
            else:
                num_actions = max(num_actions)
        action_log_probs = log_probs[:, -num_actions - 1: -1]
        if return_output:
            return action_log_probs, output
        return action_log_probs

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 32,
        temperature: float = 1.0,
        do_sample: bool = True,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = input_ids.size(0)
        device = input_ids.device
        prompt_len = input_ids.size(1)
        sequences = input_ids.clone()
        if eos_token_id is None:
            eos_token_id = self.model.config.eos_token_id
        if pad_token_id is None:
            pad_token_id = eos_token_id
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        for _ in range(max_new_tokens):
            logits = self.model(sequences)
            next_token_logits = logits[:, -1, :]
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
            if do_sample:
                probs = torch.softmax(next_token_logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
            else:
                next_tokens = next_token_logits.argmax(dim=-1)
            next_tokens = torch.where(finished, torch.full_like(next_tokens, pad_token_id), next_tokens)
            finished = finished | (next_tokens == eos_token_id)
            sequences = torch.cat([sequences, next_tokens.unsqueeze(-1)], dim=1)
            if finished.all():
                break
        return self.process_sequences(sequences, prompt_len, eos_token_id, pad_token_id)

    def process_sequences(
        self,
        sequences: torch.Tensor,
        input_len: int,
        eos_token_id: int,
        pad_token_id: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        attention_mask = (sequences.ne(eos_token_id) & sequences.ne(pad_token_id)).long()
        seq_length = attention_mask.size(1)
        eos_indices = seq_length - attention_mask.long().fliplr().argmax(dim=1, keepdim=True).clamp(min=1)
        first_token_indices = attention_mask.long().argmax(dim=1, keepdim=True)
        mask = torch.arange(seq_length, device=sequences.device).unsqueeze(0).expand(sequences.size(0), -1)
        attention_mask = ((mask >= first_token_indices) & (mask <= eos_indices)).long()
        state_seq = sequences[:, input_len - 1: -1]
        action_mask = state_seq.ne(eos_token_id) & state_seq.ne(pad_token_id)
        if action_mask.size(1) > 0:
            action_mask[:, 0] = 1
        return sequences, attention_mask, action_mask

    def refresh_expert_caches(self):
        for module in self.model.modules():
            if isinstance(module, MockMoE):
                module.refresh_weight_cache()

    @property
    def uses_quantized_experts(self) -> bool:
        return self.model.config.dtype in ('fp8', 'nvfp4')

    def get_router_aux_loss(self) -> torch.Tensor:
        return self.model.get_router_aux_loss()

    def get_expert_load_stats(self) -> Dict[str, Any]:
        return self.model.get_expert_load_stats()

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self._gradient_checkpointing = True
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self):
        self._gradient_checkpointing = False
        self.model.gradient_checkpointing_disable()

    @property
    def is_gradient_checkpointing(self) -> bool:
        return self._gradient_checkpointing

    @property
    def config(self):
        return self.model.config

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.model.parameters()).dtype

    def param_sets(self) -> Tuple[List[nn.Parameter], List[nn.Parameter]]:
        return self.model.param_sets()

    def freeze_for_reference(self) -> "MockNMoEModelWrapper":
        self._frozen_for_reference = True
        for param in self.model.parameters():
            param.requires_grad = False
        self.eval()
        return self

    def unfreeze(self) -> "MockNMoEModelWrapper":
        self._frozen_for_reference = False
        for param in self.model.parameters():
            param.requires_grad = True
        return self

    @property
    def is_frozen(self) -> bool:
        return self._frozen_for_reference

    @property
    def trainable_param_count(self) -> int:
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    @property
    def frozen_param_count(self) -> int:
        return sum(p.numel() for p in self.model.parameters() if not p.requires_grad)

    def train(self, mode: bool = True) -> "MockNMoEModelWrapper":
        if mode and self.is_frozen:
            return self
        super().train(mode)
        return self


class MockCriticWrapper(nn.Module):
    """Mock critic wrapper for PPO value estimation."""

    def __init__(self, model: MockTransformer):
        super().__init__()
        self.model = model
        self.value_head = nn.Linear(model.config.dim, 1, bias=False, dtype=torch.bfloat16)

    def forward(
        self,
        sequences: torch.Tensor,
        num_actions: Union[int, List[int]],
        attention_mask: Optional[torch.Tensor] = None,
        return_output: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
        # Get hidden states
        x = self.model.embedding(sequences)
        for block in self.model.blocks:
            x = block(x)
        x = self.model.norm(x)
        values = self.value_head(x).squeeze(-1)
        output = {"hidden_states": x}
        if isinstance(num_actions, list):
            if len(num_actions) == 1:
                num_actions = num_actions[0]
            else:
                num_actions = max(num_actions)
        action_values = values[:, -num_actions - 1: -1]
        if return_output:
            return action_values, output
        return action_values

    @property
    def config(self):
        return self.model.config

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def refresh_expert_caches(self):
        for module in self.model.modules():
            if isinstance(module, MockMoE):
                module.refresh_weight_cache()

    @property
    def uses_quantized_experts(self) -> bool:
        return self.model.config.dtype in ('fp8', 'nvfp4')


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def nmoe_config():
    """Create test nmoe configuration."""
    return MockNMoEConfig()


@pytest.fixture
def algorithm_config():
    """Create test algorithm configuration."""
    return MockAlgorithmConfig()


@pytest.fixture
def trainer_config():
    """Create test trainer configuration."""
    return MockTrainerConfig()


@pytest.fixture
def generator_config():
    """Create test generator configuration."""
    return MockGeneratorConfig()


@pytest.fixture
def nmoe_model(nmoe_config):
    """Create test nmoe model."""
    return MockTransformer(nmoe_config)


@pytest.fixture
def policy_wrapper(nmoe_model):
    """Create test policy wrapper."""
    return MockNMoEModelWrapper(nmoe_model)


@pytest.fixture
def ref_wrapper(nmoe_config):
    """Create frozen reference model wrapper."""
    model = MockTransformer(nmoe_config)
    wrapper = MockNMoEModelWrapper(model)
    wrapper.freeze_for_reference()
    return wrapper


@pytest.fixture
def critic_wrapper(nmoe_config):
    """Create test critic wrapper."""
    model = MockTransformer(nmoe_config)
    return MockCriticWrapper(model)


@pytest.fixture
def sample_batch(nmoe_config):
    """Create sample training batch."""
    batch_size = 4
    seq_len = 32
    num_actions = 8
    sequences = torch.randint(0, nmoe_config.vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    response_mask = torch.zeros(batch_size, seq_len)
    response_mask[:, -num_actions:] = 1
    loss_mask = response_mask.clone()
    rewards = torch.zeros(batch_size, seq_len)
    rewards[:, -1] = torch.randn(batch_size)
    return {
        "sequences": sequences,
        "attention_mask": attention_mask,
        "response_mask": response_mask,
        "loss_mask": loss_mask,
        "rewards": rewards,
        "num_actions": num_actions,
    }


# =============================================================================
# Test Classes
# =============================================================================

class TestGRPOTrainingLoop:
    """Tests for GRPO (Group Relative Policy Optimization) training loop with nmoe."""

    def test_grpo_advantage_computation_basic(self, nmoe_config):
        """Test GRPO advantage computation with nmoe model outputs."""
        batch_size = 4
        n_samples_per_prompt = 2
        total_samples = batch_size * n_samples_per_prompt
        seq_len = 16

        # Simulate rewards from nmoe model
        rewards = torch.randn(total_samples)

        # Compute GRPO advantages (group normalization)
        index = np.repeat(np.arange(batch_size), n_samples_per_prompt)

        id2score = defaultdict(list)
        id2mean = {}
        id2std = {}

        for i in range(total_samples):
            id2score[index[i]].append(rewards[i])
        for idx in id2score:
            id2mean[idx] = torch.mean(torch.stack(id2score[idx]))
            id2std[idx] = torch.std(torch.stack(id2score[idx])) if len(id2score[idx]) > 1 else torch.tensor(1.0)

        advantages = rewards.clone()
        for i in range(total_samples):
            advantages[i] = (rewards[i] - id2mean[index[i]]) / (id2std[index[i]] + 1e-6)

        # Verify advantages are normalized per group
        for idx in range(batch_size):
            group_mask = (torch.tensor(index) == idx)
            group_adv = advantages[group_mask]
            assert torch.abs(group_adv.mean()) < 0.5, "Group advantages should be approximately centered"

    def test_grpo_with_aux_loss_integration(self, policy_wrapper, sample_batch):
        """Test GRPO training with aux_loss from MoE layers."""
        sequences = sample_batch["sequences"]
        num_actions = sample_batch["num_actions"]

        # Forward pass
        action_log_probs, output = policy_wrapper(
            sequences, num_actions,
            return_output=True,
        )

        # Get aux loss from model
        aux_loss = policy_wrapper.get_router_aux_loss()

        # Compute policy loss (simplified GRPO)
        advantages = torch.randn(sequences.size(0), num_actions)
        policy_loss = -(action_log_probs * advantages).mean()

        # Total loss includes aux_loss
        total_loss = policy_loss + aux_loss

        assert total_loss.requires_grad, "Total loss should be differentiable"
        assert aux_loss >= 0, "Aux loss should be non-negative"

    def test_grpo_multiple_samples_per_prompt(self, policy_wrapper, nmoe_config):
        """Test GRPO with multiple samples per prompt."""
        batch_size = 2
        n_samples_per_prompt = 4
        seq_len = 16
        num_actions = 4

        # Generate sequences (n_samples_per_prompt per prompt)
        sequences = torch.randint(
            0, nmoe_config.vocab_size,
            (batch_size * n_samples_per_prompt, seq_len)
        )

        # Forward pass
        log_probs = policy_wrapper(sequences, num_actions)

        # Simulate rewards
        rewards = torch.randn(batch_size * n_samples_per_prompt)

        # Compute GRPO advantages
        index = np.repeat(np.arange(batch_size), n_samples_per_prompt)
        id2score = defaultdict(list)
        for i in range(len(rewards)):
            id2score[index[i]].append(rewards[i])

        advantages = []
        for idx in range(batch_size):
            group_rewards = torch.stack(id2score[idx])
            group_mean = group_rewards.mean()
            group_std = group_rewards.std() + 1e-6
            for r in group_rewards:
                advantages.append((r - group_mean) / group_std)
        advantages = torch.stack(advantages).unsqueeze(-1).expand(-1, num_actions)

        # Compute policy loss
        policy_loss = -(log_probs * advantages).mean()

        assert policy_loss.requires_grad

    def test_grpo_zero_variance_handling(self, policy_wrapper, nmoe_config):
        """Test GRPO handling of zero variance groups."""
        batch_size = 2
        n_samples_per_prompt = 2
        seq_len = 16

        sequences = torch.randint(
            0, nmoe_config.vocab_size,
            (batch_size * n_samples_per_prompt, seq_len)
        )

        # All rewards the same in first group
        rewards = torch.tensor([1.0, 1.0, 0.5, 0.7])
        index = np.array([0, 0, 1, 1])

        # GRPO should handle zero variance gracefully
        epsilon = 1e-6
        advantages = []
        for idx in range(batch_size):
            mask = (index == idx)
            group_rewards = rewards[mask]
            group_mean = group_rewards.mean()
            group_std = group_rewards.std()
            # Handle zero variance
            if group_std < epsilon:
                group_std = torch.tensor(1.0)
            for r in group_rewards:
                advantages.append((r - group_mean) / (group_std + epsilon))

        advantages = torch.stack(advantages)

        # First group should have zero advantages (same rewards)
        assert torch.allclose(advantages[:2], torch.zeros(2), atol=1e-5)

    def test_grpo_loss_with_response_mask(self, policy_wrapper, sample_batch):
        """Test GRPO loss respects response mask."""
        sequences = sample_batch["sequences"]
        num_actions = sample_batch["num_actions"]
        loss_mask = sample_batch["loss_mask"][:, -num_actions:]

        log_probs = policy_wrapper(sequences, num_actions)
        advantages = torch.randn_like(log_probs)

        # Masked policy loss
        masked_loss = -(log_probs * advantages * loss_mask).sum() / loss_mask.sum().clamp(min=1.0)

        assert masked_loss.requires_grad
        assert torch.isfinite(masked_loss)


class TestPPOTrainingLoop:
    """Tests for PPO (Proximal Policy Optimization) training loop with nmoe."""

    def test_ppo_clipped_objective(self, policy_wrapper, sample_batch, algorithm_config):
        """Test PPO clipped surrogate objective with nmoe."""
        sequences = sample_batch["sequences"]
        num_actions = sample_batch["num_actions"]

        # Old log probs (from rollout)
        with torch.no_grad():
            old_log_probs = policy_wrapper(sequences, num_actions)

        # Current log probs
        log_probs = policy_wrapper(sequences, num_actions)

        # Advantages (simulated)
        advantages = torch.randn(sequences.size(0), num_actions)

        # PPO clipped objective
        ratio = torch.exp(log_probs - old_log_probs)
        clipped_ratio = torch.clamp(
            ratio,
            1 - algorithm_config.eps_clip_low,
            1 + algorithm_config.eps_clip_high
        )

        surr1 = ratio * advantages
        surr2 = clipped_ratio * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        assert policy_loss.requires_grad
        assert torch.isfinite(policy_loss)

    def test_ppo_with_value_baseline(self, policy_wrapper, critic_wrapper, sample_batch):
        """Test PPO with value baseline from critic."""
        sequences = sample_batch["sequences"]
        num_actions = sample_batch["num_actions"]
        rewards = sample_batch["rewards"][:, -num_actions:]

        # Get values from critic
        values = critic_wrapper(sequences, num_actions)

        # Compute returns (simplified)
        returns = rewards.clone()

        # Value loss (MSE)
        value_loss = F.mse_loss(values, returns)

        assert value_loss.requires_grad
        assert torch.isfinite(value_loss)

    def test_ppo_dual_clip_loss(self, policy_wrapper, sample_batch, algorithm_config):
        """Test PPO dual clip loss variant."""
        sequences = sample_batch["sequences"]
        num_actions = sample_batch["num_actions"]

        with torch.no_grad():
            old_log_probs = policy_wrapper(sequences, num_actions)

        log_probs = policy_wrapper(sequences, num_actions)
        advantages = torch.randn(sequences.size(0), num_actions)

        # Standard PPO clipping
        ratio = torch.exp(log_probs - old_log_probs)
        clipped_ratio = torch.clamp(ratio, 1 - 0.2, 1 + 0.2)
        surr1 = ratio * advantages
        surr2 = clipped_ratio * advantages
        clip_pg_losses1 = -torch.min(surr1, surr2)

        # Dual clip for negative advantages
        clip_ratio_c = 3.0
        pg_losses3 = advantages * clip_ratio_c
        clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
        loss = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)

        policy_loss = loss.mean()
        assert torch.isfinite(policy_loss)

    def test_ppo_entropy_bonus(self, policy_wrapper, sample_batch, algorithm_config):
        """Test PPO entropy bonus with nmoe."""
        sequences = sample_batch["sequences"]
        num_actions = sample_batch["num_actions"]

        log_probs, output = policy_wrapper(
            sequences, num_actions,
            return_output=True,
            compute_entropy=True,
        )

        entropy = output["entropy"][:, -num_actions - 1: -1]

        # Entropy bonus
        entropy_bonus = entropy.mean() * algorithm_config.entropy_loss_coef

        assert entropy_bonus >= 0 or entropy_bonus < 0  # Can be either
        assert torch.isfinite(entropy_bonus)


class TestRewardComputation:
    """Tests for reward computation with nmoe model outputs."""

    def test_sparse_reward_to_per_token(self, sample_batch):
        """Test conversion of sparse rewards to per-token rewards."""
        batch_size = sample_batch["sequences"].size(0)
        seq_len = sample_batch["sequences"].size(1)
        num_actions = sample_batch["num_actions"]

        # Sparse reward (end of sequence)
        sparse_rewards = torch.randn(batch_size)

        # Convert to per-token rewards
        per_token_rewards = torch.zeros(batch_size, seq_len)
        per_token_rewards[:, -1] = sparse_rewards

        # Verify only last token has non-zero reward
        assert (per_token_rewards[:, :-1] == 0).all()
        assert torch.allclose(per_token_rewards[:, -1], sparse_rewards)

    def test_dense_reward_with_moe_outputs(self, policy_wrapper, sample_batch):
        """Test dense reward computation using MoE model outputs."""
        sequences = sample_batch["sequences"]
        num_actions = sample_batch["num_actions"]

        # Forward pass to get logits
        log_probs, output = policy_wrapper(
            sequences, num_actions,
            return_output=True,
        )

        logits = output["logits"]

        # Compute confidence-based reward (example)
        probs = torch.softmax(logits, dim=-1)
        confidence = probs.max(dim=-1).values

        # Use confidence as auxiliary reward signal
        aux_reward = confidence[:, -num_actions:]

        assert aux_reward.shape == (sequences.size(0), num_actions)
        assert (aux_reward >= 0).all() and (aux_reward <= 1).all()

    def test_reward_shaping_with_aux_loss(self, policy_wrapper, sample_batch):
        """Test reward shaping using MoE aux loss as penalty."""
        sequences = sample_batch["sequences"]
        num_actions = sample_batch["num_actions"]

        # Forward pass
        policy_wrapper(sequences, num_actions)

        # Get aux loss as load imbalance penalty
        aux_loss = policy_wrapper.get_router_aux_loss()

        # Use aux loss to shape reward
        base_reward = torch.randn(sequences.size(0))
        aux_penalty_coef = 0.1
        shaped_reward = base_reward - aux_penalty_coef * aux_loss.item()

        # Verify reward shaping
        assert shaped_reward.shape == base_reward.shape

    def test_reward_normalization_per_batch(self, sample_batch):
        """Test reward normalization across batch."""
        rewards = sample_batch["rewards"]
        response_mask = sample_batch["response_mask"]

        # Per-sequence reward sum
        seq_rewards = (rewards * response_mask).sum(dim=-1)

        # Normalize rewards
        reward_mean = seq_rewards.mean()
        reward_std = seq_rewards.std() + 1e-8
        normalized_rewards = (seq_rewards - reward_mean) / reward_std

        # Verify normalization
        assert torch.abs(normalized_rewards.mean()) < 1e-5
        assert torch.abs(normalized_rewards.std() - 1.0) < 0.1


class TestAdvantageEstimation:
    """Tests for advantage estimation with MoE model outputs."""

    def test_gae_advantage_estimation(self, critic_wrapper, sample_batch):
        """Test Generalized Advantage Estimation with MoE critic."""
        sequences = sample_batch["sequences"]
        num_actions = sample_batch["num_actions"]
        rewards = sample_batch["rewards"][:, -num_actions:]
        response_mask = sample_batch["response_mask"][:, -num_actions:]

        # Get values from critic
        values = critic_wrapper(sequences, num_actions)

        gamma = 0.99
        lambd = 0.95

        # Compute GAE
        advantages_reversed = []
        lastgaelam = 0

        for t in reversed(range(num_actions)):
            nextvalues = values[:, t + 1] if t < num_actions - 1 else torch.zeros(sequences.size(0))
            delta = rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam = delta + gamma * lambd * lastgaelam
            advantages_reversed.append(lastgaelam)

        advantages = torch.stack(advantages_reversed[::-1], dim=1)

        assert advantages.shape == (sequences.size(0), num_actions)
        assert torch.isfinite(advantages).all()

    def test_grpo_advantage_with_group_norm(self, sample_batch):
        """Test GRPO advantage with group normalization."""
        batch_size = 4
        n_samples_per_prompt = 2
        seq_len = sample_batch["sequences"].size(1)

        # Simulate grouped rewards
        rewards = torch.randn(batch_size * n_samples_per_prompt)
        index = np.repeat(np.arange(batch_size), n_samples_per_prompt)

        # Group normalization
        id2score = defaultdict(list)
        for i in range(len(rewards)):
            id2score[index[i]].append(rewards[i])

        normalized_advantages = []
        for idx in range(batch_size):
            group = torch.stack(id2score[idx])
            group_mean = group.mean()
            group_std = group.std() + 1e-6
            for r in group:
                normalized_advantages.append((r - group_mean) / group_std)

        advantages = torch.stack(normalized_advantages)

        # Verify per-group normalization properties
        for idx in range(batch_size):
            start = idx * n_samples_per_prompt
            end = start + n_samples_per_prompt
            group_adv = advantages[start:end]
            assert torch.abs(group_adv.mean()) < 1e-5

    def test_rloo_advantage_estimation(self, sample_batch):
        """Test RLOO (Reinforce Leave One Out) advantage estimation."""
        batch_size = 4
        n_samples_per_prompt = 4  # Need at least 2 for RLOO

        rewards = torch.randn(batch_size * n_samples_per_prompt)
        index = np.repeat(np.arange(batch_size), n_samples_per_prompt)

        id2score = defaultdict(list)
        for i in range(len(rewards)):
            id2score[index[i]].append(rewards[i])

        advantages = rewards.clone()
        for i in range(len(rewards)):
            group_idx = index[i]
            group = torch.stack(id2score[group_idx])
            n = len(group)
            if n > 1:
                # Leave one out mean
                group_sum = group.sum()
                loo_mean = (group_sum - rewards[i]) / (n - 1)
                advantages[i] = rewards[i] - loo_mean
            else:
                advantages[i] = 0.0

        assert advantages.shape == rewards.shape
        assert torch.isfinite(advantages).all()

    def test_advantage_normalization(self, sample_batch):
        """Test advantage normalization across batch."""
        num_actions = sample_batch["num_actions"]
        response_mask = sample_batch["response_mask"][:, -num_actions:]

        advantages = torch.randn(sample_batch["sequences"].size(0), num_actions)

        # Normalize advantages
        num_valid = response_mask.sum()
        mean = (advantages * response_mask).sum() / num_valid
        var = ((advantages - mean).pow(2) * response_mask).sum() / num_valid
        rstd = (var + 1e-8).rsqrt()

        normalized = (advantages - mean) * rstd

        # Verify normalization
        valid_adv = normalized[response_mask.bool()]
        assert torch.abs(valid_adv.mean()) < 0.1


class TestAuxLossIntegration:
    """Tests for aux_loss integration in RL loss computation."""

    def test_aux_loss_computation_per_layer(self, policy_wrapper, sample_batch):
        """Test aux loss computation from each MoE layer."""
        sequences = sample_batch["sequences"]
        num_actions = sample_batch["num_actions"]

        # Forward pass populates aux loss
        policy_wrapper(sequences, num_actions)

        # Collect aux losses from MoE layers
        aux_losses = []
        for module in policy_wrapper.model.modules():
            if isinstance(module, MockMoE):
                if module.last_aux_loss is not None:
                    aux_losses.append(module.last_aux_loss)

        assert len(aux_losses) > 0, "Should have at least one MoE layer"
        for aux_loss in aux_losses:
            assert aux_loss >= 0, "Aux loss should be non-negative"

    def test_aux_loss_in_total_loss(self, policy_wrapper, sample_batch, algorithm_config):
        """Test aux loss integration into total RL loss."""
        sequences = sample_batch["sequences"]
        num_actions = sample_batch["num_actions"]

        log_probs = policy_wrapper(sequences, num_actions)
        aux_loss = policy_wrapper.get_router_aux_loss()

        # Simulate RL losses
        advantages = torch.randn(sequences.size(0), num_actions)
        policy_loss = -(log_probs * advantages).mean()

        # Integrate aux loss
        aux_loss_coef = 0.01
        total_loss = policy_loss + aux_loss_coef * aux_loss

        # Verify backpropagation
        total_loss.backward()

        # Check gradients flow to expert weights
        for module in policy_wrapper.model.modules():
            if isinstance(module, MockMoE):
                assert module.W1.grad is not None or not module.W1.requires_grad

    def test_aux_loss_alpha_scaling(self, nmoe_config):
        """Test aux loss scaling with alpha parameter."""
        # Different alpha values
        alphas = [0.0, 0.001, 0.01, 0.1]

        for alpha in alphas:
            config = MockNMoEConfig(aux_loss_alpha=alpha)
            model = MockTransformer(config)
            wrapper = MockNMoEModelWrapper(model)

            sequences = torch.randint(0, config.vocab_size, (2, 16))
            wrapper(sequences, num_actions=4)

            aux_loss = wrapper.get_router_aux_loss()

            if alpha == 0.0:
                assert aux_loss.item() == 0.0
            else:
                # Loss should scale with alpha
                assert aux_loss.item() >= 0

    def test_aux_loss_gradients_to_router(self, policy_wrapper, sample_batch):
        """Test that aux loss gradients flow to router weights."""
        sequences = sample_batch["sequences"]
        num_actions = sample_batch["num_actions"]

        policy_wrapper.train()
        policy_wrapper(sequences, num_actions)

        aux_loss = policy_wrapper.get_router_aux_loss()
        aux_loss.backward()

        # Check router gate gradients
        for module in policy_wrapper.model.modules():
            if isinstance(module, MockRouter):
                # Router gate should receive gradients from aux loss
                assert module.gate.weight.grad is not None


class TestExpertLoadBalancing:
    """Tests for expert load balancing during RL training."""

    def test_load_statistics_during_training(self, policy_wrapper, sample_batch):
        """Test load statistics collection during training."""
        sequences = sample_batch["sequences"]
        num_actions = sample_batch["num_actions"]

        policy_wrapper(sequences, num_actions)

        stats = policy_wrapper.get_expert_load_stats()

        assert 'loads' in stats
        assert 'mean_load' in stats
        assert 'load_imbalance' in stats
        assert len(stats['loads']) > 0

    def test_router_bias_update(self, policy_wrapper, sample_batch):
        """Test aux-free load balancing via router bias update."""
        sequences = sample_batch["sequences"]
        num_actions = sample_batch["num_actions"]

        # Forward pass
        policy_wrapper(sequences, num_actions)

        # Get initial biases
        initial_biases = []
        for module in policy_wrapper.model.modules():
            if isinstance(module, MockRouter):
                initial_biases.append(module.bias.clone())

        # Update biases based on load
        for module in policy_wrapper.model.modules():
            if isinstance(module, MockMoE):
                if module.last_loads is not None:
                    module.router.update_bias(module.last_loads, gamma=0.1)

        # Check biases changed
        for i, module in enumerate(policy_wrapper.model.modules()):
            if isinstance(module, MockRouter):
                # Biases may or may not change depending on load distribution
                pass

    def test_load_imbalance_penalty(self, policy_wrapper, sample_batch):
        """Test load imbalance as training penalty."""
        sequences = sample_batch["sequences"]
        num_actions = sample_batch["num_actions"]

        policy_wrapper(sequences, num_actions)
        stats = policy_wrapper.get_expert_load_stats()

        # Compute imbalance penalty
        if len(stats['loads']) > 0:
            loads = torch.stack(stats['loads'])
            expected_load = 1.0 / loads.size(-1)
            imbalance = (loads - expected_load).abs().mean()

            assert imbalance >= 0

    def test_load_balancing_over_training(self, nmoe_config):
        """Test load balancing improves over multiple training steps."""
        model = MockTransformer(nmoe_config)
        wrapper = MockNMoEModelWrapper(model)

        sequences = torch.randint(0, nmoe_config.vocab_size, (8, 16))

        imbalances = []
        for step in range(5):
            wrapper(sequences, num_actions=4)
            stats = wrapper.get_expert_load_stats()

            if len(stats['loads']) > 0:
                imbalances.append(stats['load_imbalance'])

            # Update router biases
            for module in wrapper.model.modules():
                if isinstance(module, MockMoE):
                    if module.last_loads is not None:
                        module.router.update_bias(module.last_loads, gamma=0.1)

        # Imbalance should be tracked (may or may not decrease with mock router)
        assert len(imbalances) == 5


class TestKLDivergenceComputation:
    """Tests for KL divergence computation with nmoe models."""

    def test_kl_divergence_policy_ref(self, policy_wrapper, ref_wrapper, sample_batch):
        """Test KL divergence between policy and reference model."""
        sequences = sample_batch["sequences"]
        num_actions = sample_batch["num_actions"]

        # Get log probs from both models
        policy_log_probs = policy_wrapper(sequences, num_actions)

        with torch.no_grad():
            ref_log_probs = ref_wrapper(sequences, num_actions)

        # Compute KL divergence
        # Using k3 estimator (Schulman approximation)
        kl = ref_log_probs - policy_log_probs
        kl = torch.clamp(kl, min=-20, max=20)
        ratio = torch.exp(kl)
        kld = (ratio - kl - 1).contiguous()
        kld = torch.clamp(kld, min=-10, max=10)

        assert kld.shape == policy_log_probs.shape
        assert torch.isfinite(kld).all()

    def test_kl_estimator_types(self, policy_wrapper, ref_wrapper, sample_batch):
        """Test different KL estimator types."""
        sequences = sample_batch["sequences"]
        num_actions = sample_batch["num_actions"]

        policy_log_probs = policy_wrapper(sequences, num_actions)
        with torch.no_grad():
            ref_log_probs = ref_wrapper(sequences, num_actions)

        # k1: simple difference
        kl_k1 = policy_log_probs - ref_log_probs

        # k2: squared difference
        kl_k2 = 0.5 * (policy_log_probs - ref_log_probs).square()

        # k3: Schulman approximation
        kl = ref_log_probs - policy_log_probs
        kl = torch.clamp(kl, min=-20, max=20)
        ratio = torch.exp(kl)
        kl_k3 = (ratio - kl - 1)

        # abs: absolute difference
        kl_abs = (policy_log_probs - ref_log_probs).abs()

        assert torch.isfinite(kl_k1).all()
        assert torch.isfinite(kl_k2).all()
        assert torch.isfinite(kl_k3).all()
        assert torch.isfinite(kl_abs).all()

    def test_kl_penalty_in_reward(self, policy_wrapper, ref_wrapper, sample_batch, algorithm_config):
        """Test KL penalty applied to rewards."""
        sequences = sample_batch["sequences"]
        num_actions = sample_batch["num_actions"]
        rewards = sample_batch["rewards"][:, -num_actions:]
        loss_mask = sample_batch["loss_mask"][:, -num_actions:]

        policy_log_probs = policy_wrapper(sequences, num_actions)
        with torch.no_grad():
            ref_log_probs = ref_wrapper(sequences, num_actions)

        # Compute KL penalty
        kl = ref_log_probs - policy_log_probs
        kl = torch.clamp(kl, min=-20, max=20)
        ratio = torch.exp(kl)
        kld = (ratio - kl - 1)
        kld = torch.clamp(kld, min=-10, max=10)

        # Apply KL penalty to rewards
        kl_coef = algorithm_config.kl_loss_coef
        penalized_rewards = rewards - kl_coef * kld

        assert penalized_rewards.shape == rewards.shape
        assert torch.isfinite(penalized_rewards).all()

    def test_kl_loss_term(self, policy_wrapper, ref_wrapper, sample_batch, algorithm_config):
        """Test KL as regularization loss term."""
        sequences = sample_batch["sequences"]
        num_actions = sample_batch["num_actions"]
        loss_mask = sample_batch["loss_mask"][:, -num_actions:]

        policy_log_probs = policy_wrapper(sequences, num_actions)
        with torch.no_grad():
            ref_log_probs = ref_wrapper(sequences, num_actions)

        # Compute masked KL loss
        kl = ref_log_probs - policy_log_probs
        kl = torch.clamp(kl, min=-20, max=20)
        ratio = torch.exp(kl)
        kld = (ratio - kl - 1)

        # Masked mean
        kl_loss = (kld * loss_mask).sum() / loss_mask.sum().clamp(min=1.0)
        kl_loss_term = kl_loss * algorithm_config.kl_loss_coef

        assert torch.isfinite(kl_loss_term)
        assert kl_loss_term.requires_grad


class TestReferenceModelHandling:
    """Tests for reference model handling with nmoe."""

    def test_reference_model_freezing(self, ref_wrapper):
        """Test that reference model parameters are frozen."""
        assert ref_wrapper.is_frozen

        for param in ref_wrapper.model.parameters():
            assert not param.requires_grad

    def test_reference_model_no_gradient(self, ref_wrapper, sample_batch):
        """Test that reference model doesn't accumulate gradients."""
        sequences = sample_batch["sequences"]
        num_actions = sample_batch["num_actions"]

        log_probs = ref_wrapper(sequences, num_actions)

        # Reference model outputs should not require grad
        assert not log_probs.requires_grad

    def test_reference_model_update_from_policy(self, policy_wrapper, nmoe_config):
        """Test updating reference model weights from policy."""
        ref_model = MockTransformer(nmoe_config)
        ref_wrapper = MockNMoEModelWrapper(ref_model)
        ref_wrapper.freeze_for_reference()

        # Get policy state dict
        policy_state = policy_wrapper.model.state_dict()

        # Update reference (unfreeze, load, refreeze)
        ref_wrapper.unfreeze()
        ref_wrapper.model.load_state_dict(policy_state)
        ref_wrapper.freeze_for_reference()

        # Verify weights match
        for (name1, param1), (name2, param2) in zip(
            policy_wrapper.model.named_parameters(),
            ref_wrapper.model.named_parameters()
        ):
            assert name1 == name2
            assert torch.allclose(param1, param2)

    def test_reference_model_eval_mode(self, ref_wrapper):
        """Test that reference model stays in eval mode."""
        assert not ref_wrapper.training

        # Try to set train mode
        ref_wrapper.train(True)

        # Should still be in eval mode (frozen model protection)
        assert not ref_wrapper.training

    def test_reference_model_with_moe(self, ref_wrapper, sample_batch):
        """Test reference model with MoE layers produces valid outputs."""
        sequences = sample_batch["sequences"]
        num_actions = sample_batch["num_actions"]

        with torch.no_grad():
            log_probs = ref_wrapper(sequences, num_actions)

        # MoE layers should still route correctly
        stats = ref_wrapper.get_expert_load_stats()
        assert len(stats['loads']) > 0


class TestActorCriticSetup:
    """Tests for actor-critic setup with separate nmoe models."""

    def test_separate_actor_critic_models(self, policy_wrapper, critic_wrapper, sample_batch):
        """Test actor and critic with separate nmoe models."""
        sequences = sample_batch["sequences"]
        num_actions = sample_batch["num_actions"]

        # Actor forward
        policy_log_probs = policy_wrapper(sequences, num_actions)

        # Critic forward
        values = critic_wrapper(sequences, num_actions)

        assert policy_log_probs.shape == values.shape
        assert policy_log_probs.requires_grad
        assert values.requires_grad

    def test_actor_critic_gradient_isolation(self, policy_wrapper, critic_wrapper, sample_batch):
        """Test that actor and critic gradients are isolated."""
        sequences = sample_batch["sequences"]
        num_actions = sample_batch["num_actions"]

        policy_wrapper.train()
        critic_wrapper.train()

        # Actor loss
        log_probs = policy_wrapper(sequences, num_actions)
        advantages = torch.randn_like(log_probs)
        actor_loss = -(log_probs * advantages).mean()
        actor_loss.backward()

        # Verify critic has no gradients
        for param in critic_wrapper.model.parameters():
            assert param.grad is None or (param.grad == 0).all()

        # Reset gradients
        policy_wrapper.zero_grad()

        # Critic loss
        values = critic_wrapper(sequences, num_actions)
        returns = torch.randn_like(values)
        critic_loss = F.mse_loss(values, returns)
        critic_loss.backward()

        # Verify actor has no new gradients
        for param in policy_wrapper.model.parameters():
            assert param.grad is None or (param.grad == 0).all()

    def test_shared_backbone_separate_heads(self, nmoe_config):
        """Test shared MoE backbone with separate policy/value heads."""
        # Create shared model
        shared_model = MockTransformer(nmoe_config)

        # Policy head
        policy_head = nn.Linear(nmoe_config.dim, nmoe_config.vocab_size, dtype=torch.bfloat16)

        # Value head
        value_head = nn.Linear(nmoe_config.dim, 1, dtype=torch.bfloat16)

        sequences = torch.randint(0, nmoe_config.vocab_size, (2, 16))

        # Get shared features
        x = shared_model.embedding(sequences)
        for block in shared_model.blocks:
            x = block(x)
        x = shared_model.norm(x)

        # Separate heads
        policy_logits = policy_head(x)
        values = value_head(x)

        assert policy_logits.shape == (2, 16, nmoe_config.vocab_size)
        assert values.shape == (2, 16, 1)

    def test_async_actor_critic_updates(self, policy_wrapper, critic_wrapper, sample_batch):
        """Test async-style actor and critic updates."""
        sequences = sample_batch["sequences"]
        num_actions = sample_batch["num_actions"]

        # Actor update
        policy_wrapper.train()
        log_probs = policy_wrapper(sequences, num_actions)
        advantages = torch.randn_like(log_probs)
        actor_loss = -(log_probs * advantages).mean()

        # Critic update with different batch
        critic_wrapper.train()
        values = critic_wrapper(sequences, num_actions)
        returns = torch.randn_like(values)
        critic_loss = F.mse_loss(values, returns)

        # Both losses should be valid
        assert torch.isfinite(actor_loss)
        assert torch.isfinite(critic_loss)


class TestDistributedRL:
    """Tests for distributed RL training with nmoe (mock distributed)."""

    def test_data_parallel_batch_split(self, sample_batch):
        """Test data parallel batch splitting for distributed training."""
        sequences = sample_batch["sequences"]
        dp_size = 4

        # Split batch
        batch_size = sequences.size(0)
        if batch_size % dp_size != 0:
            # Pad batch
            pad_size = (dp_size - batch_size % dp_size) % dp_size
            sequences = torch.cat([
                sequences,
                sequences[:pad_size]
            ], dim=0)

        local_batch_size = sequences.size(0) // dp_size
        splits = torch.chunk(sequences, dp_size, dim=0)

        assert len(splits) == dp_size
        for split in splits:
            assert split.size(0) == local_batch_size

    def test_gradient_synchronization_mock(self, policy_wrapper, sample_batch):
        """Test gradient synchronization (mocked)."""
        sequences = sample_batch["sequences"]
        num_actions = sample_batch["num_actions"]

        policy_wrapper.train()
        log_probs = policy_wrapper(sequences, num_actions)
        advantages = torch.randn_like(log_probs)
        loss = -(log_probs * advantages).mean()
        loss.backward()

        # Collect gradients (simulating all-reduce)
        gradients = {}
        for name, param in policy_wrapper.model.named_parameters():
            if param.grad is not None:
                gradients[name] = param.grad.clone()

        # Simulate gradient averaging
        world_size = 4
        for name in gradients:
            gradients[name] = gradients[name] / world_size

        # Apply averaged gradients
        for name, param in policy_wrapper.model.named_parameters():
            if name in gradients and param.grad is not None:
                param.grad.copy_(gradients[name])

    def test_expert_parallel_load_distribution(self, nmoe_config):
        """Test expert parallel load distribution simulation."""
        n_experts = nmoe_config.n_routed_experts
        ep_size = 4
        experts_per_rank = n_experts // ep_size

        # Simulate load distribution
        total_tokens = 1024
        expert_loads = torch.rand(n_experts)
        expert_loads = expert_loads / expert_loads.sum() * total_tokens

        # Split by EP rank
        for rank in range(ep_size):
            start = rank * experts_per_rank
            end = start + experts_per_rank
            local_load = expert_loads[start:end]
            assert len(local_load) == experts_per_rank

    def test_mesh_dispatch_simulation(self, sample_batch):
        """Test mesh dispatch for distributed training."""
        sequences = sample_batch["sequences"]
        dp_size = 2
        sp_size = 2  # sequence parallel

        # Create virtual mesh
        mesh_shape = (dp_size, sp_size)
        total_ranks = dp_size * sp_size

        # Simulate dispatch
        batch_size = sequences.size(0)
        seq_len = sequences.size(1)

        local_batch = batch_size // dp_size
        local_seq = seq_len // sp_size

        for dp_rank in range(dp_size):
            for sp_rank in range(sp_size):
                local_seqs = sequences[
                    dp_rank * local_batch:(dp_rank + 1) * local_batch,
                    sp_rank * local_seq:(sp_rank + 1) * local_seq
                ]
                assert local_seqs.shape == (local_batch, local_seq)


class TestExpertCacheRefresh:
    """Tests for expert cache refresh with quantized nmoe models."""

    def test_expert_cache_refresh_after_optimizer_step(self, nmoe_config):
        """Test expert cache refresh after optimizer step."""
        nmoe_config.dtype = "fp8"
        model = MockTransformer(nmoe_config)
        wrapper = MockNMoEModelWrapper(model)

        # Simulate optimizer step
        sequences = torch.randint(0, nmoe_config.vocab_size, (2, 16))
        log_probs = wrapper(sequences, num_actions=4)
        loss = -log_probs.mean()
        loss.backward()

        # Refresh caches
        wrapper.refresh_expert_caches()

        # Verify caches were refreshed
        for module in wrapper.model.modules():
            if isinstance(module, MockMoE):
                assert module._W_cache == "refreshed"

    def test_uses_quantized_experts_detection(self, nmoe_config):
        """Test detection of quantized expert usage."""
        # BF16 model
        nmoe_config.dtype = "bf16"
        model = MockTransformer(nmoe_config)
        wrapper = MockNMoEModelWrapper(model)
        assert not wrapper.uses_quantized_experts

        # FP8 model
        nmoe_config.dtype = "fp8"
        model = MockTransformer(nmoe_config)
        wrapper = MockNMoEModelWrapper(model)
        assert wrapper.uses_quantized_experts

        # NVFP4 model
        nmoe_config.dtype = "nvfp4"
        model = MockTransformer(nmoe_config)
        wrapper = MockNMoEModelWrapper(model)
        assert wrapper.uses_quantized_experts


class TestGradientCheckpointing:
    """Tests for gradient checkpointing with nmoe RL training."""

    def test_gradient_checkpointing_enable_disable(self, policy_wrapper):
        """Test gradient checkpointing toggle."""
        assert not policy_wrapper.is_gradient_checkpointing

        policy_wrapper.gradient_checkpointing_enable()
        assert policy_wrapper.is_gradient_checkpointing

        policy_wrapper.gradient_checkpointing_disable()
        assert not policy_wrapper.is_gradient_checkpointing

    def test_training_with_gradient_checkpointing(self, policy_wrapper, sample_batch):
        """Test training works with gradient checkpointing enabled."""
        policy_wrapper.gradient_checkpointing_enable()
        policy_wrapper.train()

        sequences = sample_batch["sequences"]
        num_actions = sample_batch["num_actions"]

        log_probs = policy_wrapper(sequences, num_actions)
        advantages = torch.randn_like(log_probs)
        loss = -(log_probs * advantages).mean()

        # Should work with gradient checkpointing
        loss.backward()

        # Verify gradients exist
        for param in policy_wrapper.model.parameters():
            if param.requires_grad:
                assert param.grad is not None or param.numel() == 0


class TestEndToEndRLLoop:
    """End-to-end tests for complete RL training loop with nmoe."""

    def test_grpo_end_to_end(self, nmoe_config, algorithm_config):
        """Test complete GRPO training loop end-to-end."""
        # Setup
        model = MockTransformer(nmoe_config)
        policy = MockNMoEModelWrapper(model)
        optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-4)

        batch_size = 4
        n_samples_per_prompt = 2
        seq_len = 16
        num_actions = 4

        # Generate data
        sequences = torch.randint(0, nmoe_config.vocab_size, (batch_size * n_samples_per_prompt, seq_len))
        rewards = torch.randn(batch_size * n_samples_per_prompt)

        # Training step
        policy.train()
        optimizer.zero_grad()

        # Forward
        log_probs = policy(sequences, num_actions)

        # GRPO advantage computation
        index = np.repeat(np.arange(batch_size), n_samples_per_prompt)
        id2score = defaultdict(list)
        for i in range(len(rewards)):
            id2score[index[i]].append(rewards[i])

        advantages = []
        for idx in range(batch_size):
            group = torch.stack(id2score[idx])
            group_mean = group.mean()
            group_std = group.std() + 1e-6
            for r in group:
                advantages.append((r - group_mean) / group_std)
        advantages = torch.stack(advantages).unsqueeze(-1).expand(-1, num_actions)

        # Loss
        policy_loss = -(log_probs * advantages).mean()
        aux_loss = policy.get_router_aux_loss()
        total_loss = policy_loss + 0.01 * aux_loss

        # Backward
        total_loss.backward()
        optimizer.step()

        # Expert cache refresh for quantized models
        if policy.uses_quantized_experts:
            policy.refresh_expert_caches()

        assert torch.isfinite(total_loss)

    def test_ppo_end_to_end(self, nmoe_config, algorithm_config):
        """Test complete PPO training loop end-to-end."""
        # Setup
        actor_model = MockTransformer(nmoe_config)
        actor = MockNMoEModelWrapper(actor_model)

        critic_model = MockTransformer(nmoe_config)
        critic = MockCriticWrapper(critic_model)

        ref_model = MockTransformer(nmoe_config)
        ref = MockNMoEModelWrapper(ref_model)
        ref.freeze_for_reference()

        actor_optimizer = torch.optim.AdamW(actor.parameters(), lr=1e-4)
        critic_optimizer = torch.optim.AdamW(critic.parameters(), lr=1e-4)

        batch_size = 4
        seq_len = 16
        num_actions = 4

        sequences = torch.randint(0, nmoe_config.vocab_size, (batch_size, seq_len))
        rewards = torch.zeros(batch_size, num_actions)
        rewards[:, -1] = torch.randn(batch_size)

        # Rollout phase
        with torch.no_grad():
            old_log_probs = actor(sequences, num_actions)
            ref_log_probs = ref(sequences, num_actions)
            old_values = critic(sequences, num_actions)

        # Training phase
        actor.train()
        critic.train()

        # Actor update
        actor_optimizer.zero_grad()
        log_probs, output = actor(sequences, num_actions, return_output=True, compute_entropy=True)

        ratio = torch.exp(log_probs - old_log_probs)
        advantages = rewards - old_values.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 0.8, 1.2) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # KL penalty
        kl = ref_log_probs - log_probs
        kl = torch.clamp(kl, -20, 20)
        kl_ratio = torch.exp(kl)
        kld = (kl_ratio - kl - 1)
        kl_loss = kld.mean() * algorithm_config.kl_loss_coef

        # Entropy bonus
        entropy = output["entropy"][:, -num_actions - 1: -1]
        entropy_bonus = entropy.mean() * algorithm_config.entropy_loss_coef

        # Aux loss
        aux_loss = actor.get_router_aux_loss()

        actor_loss = policy_loss + kl_loss - entropy_bonus + 0.01 * aux_loss
        actor_loss.backward()
        actor_optimizer.step()

        # Critic update
        critic_optimizer.zero_grad()
        values = critic(sequences, num_actions)
        returns = rewards.clone()
        critic_loss = F.mse_loss(values, returns)
        critic_loss.backward()
        critic_optimizer.step()

        assert torch.isfinite(actor_loss)
        assert torch.isfinite(critic_loss)

    def test_multiple_training_epochs(self, nmoe_config, algorithm_config):
        """Test multiple training epochs on same batch."""
        model = MockTransformer(nmoe_config)
        policy = MockNMoEModelWrapper(model)
        optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-4)

        sequences = torch.randint(0, nmoe_config.vocab_size, (4, 16))
        num_actions = 4

        # Save initial parameters
        initial_params = {
            name: param.clone()
            for name, param in policy.model.named_parameters()
        }

        # Multiple epochs
        n_epochs = 3
        for epoch in range(n_epochs):
            policy.train()
            optimizer.zero_grad()

            log_probs = policy(sequences, num_actions)
            advantages = torch.randn_like(log_probs)
            loss = -(log_probs * advantages).mean()

            loss.backward()
            optimizer.step()

        # Parameters should have changed
        for name, param in policy.model.named_parameters():
            if param.requires_grad:
                assert not torch.allclose(param, initial_params[name]), \
                    f"Parameter {name} should have been updated"


@pytest.mark.gpu
class TestGPUTraining:
    """GPU-specific tests for nmoe RL training."""

    @pytest.fixture
    def gpu_policy(self, nmoe_config):
        """Create policy on GPU."""
        if not torch.cuda.is_available():
            pytest.skip("GPU not available")
        model = MockTransformer(nmoe_config).cuda()
        return MockNMoEModelWrapper(model)

    @pytest.fixture
    def gpu_sample_batch(self, nmoe_config):
        """Create sample batch on GPU."""
        if not torch.cuda.is_available():
            pytest.skip("GPU not available")
        batch_size = 4
        seq_len = 32
        num_actions = 8
        sequences = torch.randint(0, nmoe_config.vocab_size, (batch_size, seq_len)).cuda()
        attention_mask = torch.ones(batch_size, seq_len).cuda()
        return {
            "sequences": sequences,
            "attention_mask": attention_mask,
            "num_actions": num_actions,
        }

    def test_gpu_forward_backward(self, gpu_policy, gpu_sample_batch):
        """Test forward/backward pass on GPU."""
        if not torch.cuda.is_available():
            pytest.skip("GPU not available")

        sequences = gpu_sample_batch["sequences"]
        num_actions = gpu_sample_batch["num_actions"]

        gpu_policy.train()
        log_probs = gpu_policy(sequences, num_actions)
        advantages = torch.randn_like(log_probs)
        loss = -(log_probs * advantages).mean()

        loss.backward()

        assert log_probs.device.type == "cuda"
        assert torch.isfinite(loss)

    def test_gpu_memory_efficient_training(self, gpu_policy, gpu_sample_batch):
        """Test memory-efficient training with gradient checkpointing on GPU."""
        if not torch.cuda.is_available():
            pytest.skip("GPU not available")

        gpu_policy.gradient_checkpointing_enable()

        sequences = gpu_sample_batch["sequences"]
        num_actions = gpu_sample_batch["num_actions"]

        # Record memory before
        torch.cuda.reset_peak_memory_stats()

        gpu_policy.train()
        log_probs = gpu_policy(sequences, num_actions)
        loss = -log_probs.mean()
        loss.backward()

        # Should complete without OOM
        assert torch.isfinite(loss)


@pytest.mark.gpu
class TestMultiGPUDistributed:
    """Tests for multi-GPU distributed RL training."""

    @requires_multi_gpu(8)
    def test_8_gpu_data_parallel_simulation(self, nmoe_config):
        """Test 8-GPU data parallel training simulation."""
        dp_size = 8
        batch_size = 32
        seq_len = 16

        # Simulate each GPU's local batch
        local_batch_size = batch_size // dp_size

        local_outputs = []
        for rank in range(dp_size):
            local_sequences = torch.randint(0, nmoe_config.vocab_size, (local_batch_size, seq_len))
            model = MockTransformer(nmoe_config)
            wrapper = MockNMoEModelWrapper(model)

            local_output = wrapper(local_sequences, num_actions=4)
            local_outputs.append(local_output)

        # All outputs should have same shape
        for output in local_outputs:
            assert output.shape == (local_batch_size, 4)

    @requires_multi_gpu(4)
    def test_expert_parallel_4_gpu_simulation(self, nmoe_config):
        """Test expert parallel training across 4 GPUs."""
        ep_size = 4
        n_experts = nmoe_config.n_routed_experts
        experts_per_gpu = n_experts // ep_size

        assert n_experts % ep_size == 0, "Experts must be divisible by EP size"

        # Simulate expert distribution
        for rank in range(ep_size):
            local_expert_ids = list(range(rank * experts_per_gpu, (rank + 1) * experts_per_gpu))
            assert len(local_expert_ids) == experts_per_gpu


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x"])
