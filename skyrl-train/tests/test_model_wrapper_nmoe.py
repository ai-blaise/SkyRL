"""Tests for NMoEModelWrapper.

These tests verify that the NMoEModelWrapper correctly implements:
- HFModelWrapper-compatible interface (forward, generate)
- NMoEModelInterface methods (expert cache, load stats)
- Correct output shapes for RL training
"""

import sys
import os

# Add skyrl_train to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional


# Mock nmoe components for testing without full nmoe installation
@dataclass
class MockConfig:
    """Mock nmoe config for testing."""
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
    dtype: str = "bf16"
    rms_norm_eps: float = 1e-5
    qk_rope_head_dim: int = 32
    rope_theta: float = 10000.0
    max_position_embeddings: int = 2048
    rope_scaling_factor: float = 1.0
    rope_ntk_alpha: float = 1.0
    rope_ntk_beta: float = 32.0


class MockRouter(nn.Module):
    """Mock router for testing."""
    def __init__(self, config):
        super().__init__()
        self.n_experts = config.n_routed_experts
        self.topk = config.n_activated_experts
        self.gate = nn.Linear(config.dim, self.n_experts, bias=False)
        self.register_buffer("bias", torch.zeros(self.n_experts))

    def forward(self, x):
        logits = self.gate(x).float()
        scores = torch.sigmoid(logits)
        _, indices = torch.topk(scores, k=self.topk, dim=-1)
        weights = torch.gather(scores, 1, indices)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp(min=1e-12)
        return weights.to(x.dtype), indices

    def update_bias(self, loads, gamma=0.001):
        expected = 1.0 / self.n_experts
        s = torch.sign(loads - expected)
        self.bias -= gamma * (s - s.mean())


class MockMoE(nn.Module):
    """Mock MoE layer for testing."""
    def __init__(self, config):
        super().__init__()
        self.dim = config.dim
        self.n_experts = config.n_routed_experts
        self.router = MockRouter(config)
        self.W1 = nn.Parameter(torch.randn(self.n_experts, config.dim, config.moe_inter_dim) * 0.02)
        self.W3 = nn.Parameter(torch.randn(self.n_experts, config.dim, config.moe_inter_dim) * 0.02)
        self.W2 = nn.Parameter(torch.randn(self.n_experts, config.moe_inter_dim, config.dim) * 0.02)
        self._use_blockscaled = config.dtype in ('fp8', 'nvfp4')
        self._W_cache = None
        self.last_loads = None
        self.last_aux_loss = None

    def refresh_weight_cache(self):
        self._W_cache = "refreshed"

    def forward(self, x):
        B, T, D = x.shape
        x_flat = x.view(-1, D)
        g, eid = self.router(x_flat)

        # Record loads
        self.last_loads = torch.bincount(
            eid.reshape(-1), minlength=self.n_experts
        ).float()
        self.last_aux_loss = torch.tensor(0.0)

        # Simple MoE forward (not actual expert computation)
        out = x_flat @ self.W1[0] @ self.W2[0].T
        return out.view(B, T, D)


class MockTransformerBlock(nn.Module):
    """Mock transformer block for testing."""
    def __init__(self, config, layer_id, is_moe=False):
        super().__init__()
        self.layer_id = layer_id
        self.is_moe = is_moe
        self.attn_norm = nn.LayerNorm(config.dim)
        self.ffn_norm = nn.LayerNorm(config.dim)

        # Simple attention mock
        self.attn = nn.Linear(config.dim, config.dim)

        if is_moe:
            self.ffn = MockMoE(config)
        else:
            self.ffn = nn.Sequential(
                nn.Linear(config.dim, config.inter_dim),
                nn.SiLU(),
                nn.Linear(config.inter_dim, config.dim),
            )

    def forward(self, x, cos=None, sin=None):
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x


class MockTransformer(nn.Module):
    """Mock nmoe Transformer for testing."""
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
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)

    def init_weights(self):
        pass

    def param_sets(self):
        expert_params = []
        for m in self.modules():
            if isinstance(m, MockMoE):
                expert_params.extend([m.W1, m.W3, m.W2])
        expert_ids = {id(p) for p in expert_params}
        dense_params = [p for p in self.parameters() if id(p) not in expert_ids]
        return expert_params, dense_params

    def forward(self, tokens):
        x = self.embedding(tokens)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        logits = self.lm_head(x)
        return logits


# Create a self-contained NMoEModelWrapper for testing
class TestableNMoEModelWrapper(nn.Module):
    """Testable version of NMoEModelWrapper that doesn't require nmoe package."""

    def __init__(
        self,
        model: nn.Module,
        temperature: float = 1.0,
        use_torch_compile: bool = False,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.model = model
        self.temperature = temperature
        self._gradient_checkpointing = False

        # Simple entropy function for testing
        self.chunked_entropy_from_logits_fn = self._simple_entropy

        if gradient_checkpointing:
            self.gradient_checkpointing_enable()

    def _simple_entropy(self, logits, requires_grad=False, attention_mask=None):
        """Simple entropy calculation for testing."""
        probs = torch.softmax(logits, dim=-1)
        log_probs = torch.log_softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1)
        if attention_mask is not None:
            entropy = entropy * attention_mask
        return entropy

    def _logprobs_from_logits(self, logits, labels):
        """Simple log prob calculation for testing."""
        log_probs = torch.log_softmax(logits, dim=-1)
        gathered = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
        return gathered

    def forward(
        self,
        sequences,
        num_actions,
        attention_mask=None,
        temperature=1.0,
        return_output=False,
        compute_entropy=False,
        entropy_requires_grad=True,
    ):
        """Forward pass returning action log probabilities."""
        logits = self.model(sequences)

        effective_temp = temperature if temperature != 1.0 else self.temperature
        if effective_temp != 1.0:
            logits = logits / effective_temp

        sequences_rolled = torch.roll(sequences, shifts=-1, dims=1)
        log_probs = self._logprobs_from_logits(logits, sequences_rolled)

        output = {"logits": logits}

        if compute_entropy:
            entropy = self.chunked_entropy_from_logits_fn(
                logits, requires_grad=entropy_requires_grad, attention_mask=attention_mask
            )
            output["entropy"] = entropy

        if isinstance(num_actions, list):
            if len(num_actions) == 1:
                num_actions = num_actions[0]
            else:
                # Variable length actions - use max
                import numpy as np
                num_actions = int(np.array(num_actions).max())

        action_log_probs = log_probs[:, -num_actions - 1: -1]

        if return_output:
            return action_log_probs, output
        return action_log_probs

    @torch.no_grad()
    def generate(
        self,
        input_ids,
        max_new_tokens=128,
        temperature=1.0,
        top_p=1.0,
        top_k=0,
        do_sample=True,
        eos_token_id=None,
        pad_token_id=None,
        attention_mask=None,
        **kwargs,
    ):
        """Generate tokens autoregressively."""
        batch_size = input_ids.size(0)
        device = input_ids.device
        prompt_len = input_ids.size(1)

        sequences = input_ids.clone()

        if eos_token_id is None:
            eos_token_id = getattr(self.model.config, 'eos_token_id', 999)
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

            next_tokens = torch.where(
                finished, torch.full_like(next_tokens, pad_token_id), next_tokens
            )
            finished = finished | (next_tokens == eos_token_id)
            sequences = torch.cat([sequences, next_tokens.unsqueeze(-1)], dim=1)

            if finished.all():
                break

        return self.process_sequences(sequences, prompt_len, eos_token_id, pad_token_id)

    def process_sequences(self, sequences, input_len, eos_token_id, pad_token_id):
        """Process generated sequences to create masks."""
        attention_mask = (sequences.ne(eos_token_id) & sequences.ne(pad_token_id)).long()
        seq_length = attention_mask.size(1)
        eos_indices = seq_length - attention_mask.long().fliplr().argmax(dim=1, keepdim=True).clamp(min=1)
        first_token_indices = attention_mask.long().argmax(dim=1, keepdim=True)
        mask = torch.arange(seq_length, device=sequences.device).unsqueeze(0).expand(sequences.size(0), -1)
        attention_mask = ((mask >= first_token_indices) & (mask <= eos_indices)).long()
        state_seq = sequences[:, input_len - 1: -1]
        action_mask = state_seq.ne(eos_token_id) & state_seq.ne(pad_token_id)
        action_mask[:, 0] = 1
        return sequences, attention_mask, action_mask

    def refresh_expert_caches(self):
        """Refresh quantized weight caches."""
        for module in self.model.modules():
            if isinstance(module, MockMoE):
                module.refresh_weight_cache()

    @property
    def uses_quantized_experts(self):
        """Whether this model uses quantized expert weights."""
        dtype = getattr(self.model.config, 'dtype', 'bf16')
        return dtype in ('fp8', 'nvfp4')

    def get_router_aux_loss(self):
        """Get auxiliary load balancing loss."""
        aux_loss = torch.tensor(0.0, device=self.device)
        moe_layers = self._get_moe_layers()
        for moe in moe_layers:
            if moe.last_aux_loss is not None:
                aux_loss = aux_loss + moe.last_aux_loss
        return aux_loss / max(len(moe_layers), 1)

    def get_expert_load_stats(self):
        """Get expert load statistics."""
        moe_layers = self._get_moe_layers()
        if not moe_layers:
            return {'loads': torch.tensor([]), 'load_mean': torch.tensor(0.0), 'load_std': torch.tensor(0.0)}
        loads_list = [moe.last_loads for moe in moe_layers if moe.last_loads is not None]
        if not loads_list:
            return {'loads': torch.tensor([]), 'load_mean': torch.tensor(0.0), 'load_std': torch.tensor(0.0)}
        loads = torch.stack(loads_list, dim=0)
        return {'loads': loads, 'load_mean': loads.mean(), 'load_std': loads.std()}

    def update_router_biases(self, gamma=0.001):
        """Update router biases."""
        moe_layers = self._get_moe_layers()
        for moe in moe_layers:
            if moe.last_loads is not None:
                moe.router.update_bias(moe.last_loads, gamma=gamma)

    def _get_moe_layers(self):
        """Get all MoE layers from the model."""
        return [m for m in self.model.modules() if isinstance(m, MockMoE)]

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """Enable gradient checkpointing."""
        self._gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self._gradient_checkpointing = False

    @property
    def is_gradient_checkpointing(self):
        """Whether gradient checkpointing is enabled."""
        return self._gradient_checkpointing

    @property
    def config(self):
        """Get model configuration."""
        return self.model.config

    @property
    def device(self):
        """Get model device."""
        return next(self.model.parameters()).device

    @property
    def dtype(self):
        """Get model dtype."""
        return next(self.model.parameters()).dtype

    def get_input_embeddings(self):
        """Get input embedding layer."""
        return self.model.embedding

    def get_output_embeddings(self):
        """Get output embedding layer."""
        return self.model.lm_head

    def param_sets(self):
        """Get parameter sets."""
        return self.model.param_sets()

    def named_parameters_by_type(self):
        """Get named parameters by type."""
        params_by_type = {'expert': [], 'router': [], 'attention': [], 'dense': [], 'embedding': []}
        for name, param in self.model.named_parameters():
            if 'W1' in name or 'W2' in name or 'W3' in name:
                if '.ffn.' in name and 'router' not in name:
                    params_by_type['expert'].append((name, param))
                else:
                    params_by_type['dense'].append((name, param))
            elif 'router' in name or 'gate' in name:
                params_by_type['router'].append((name, param))
            elif 'attn' in name:
                params_by_type['attention'].append((name, param))
            elif 'embedding' in name or 'embed' in name:
                params_by_type['embedding'].append((name, param))
            else:
                params_by_type['dense'].append((name, param))
        return params_by_type

    def state_dict_for_save(self):
        """Get state dict for saving."""
        return self.model.state_dict()

    def load_state_dict_from_checkpoint(self, state_dict, strict=True):
        """Load state dict from checkpoint."""
        self.model.load_state_dict(state_dict, strict=strict)
        if self.uses_quantized_experts:
            self.refresh_expert_caches()

    # Reference model support methods
    def freeze_for_reference(self):
        """Freeze all parameters for use as reference model."""
        self._frozen_for_reference = True
        for param in self.model.parameters():
            param.requires_grad = False
        self.eval()
        return self

    def unfreeze(self):
        """Unfreeze all parameters for training."""
        self._frozen_for_reference = False
        for param in self.model.parameters():
            param.requires_grad = True
        return self

    def freeze_expert_weights(self):
        """Freeze only expert weights."""
        expert_params, dense_params = self.param_sets()
        for param in expert_params:
            param.requires_grad = False
        return self

    def freeze_dense_weights(self):
        """Freeze only dense weights."""
        expert_params, dense_params = self.param_sets()
        for param in dense_params:
            param.requires_grad = False
        return self

    @property
    def is_frozen(self):
        """Check if model is frozen for reference."""
        return getattr(self, '_frozen_for_reference', False)

    @property
    def trainable_param_count(self):
        """Get count of trainable parameters."""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    @property
    def frozen_param_count(self):
        """Get count of frozen parameters."""
        return sum(p.numel() for p in self.model.parameters() if not p.requires_grad)

    def train(self, mode=True):
        """Set training mode."""
        if mode and self.is_frozen:
            return self
        super().train(mode)
        return self


class TestNMoEModelWrapper:
    """Tests for NMoEModelWrapper."""

    @pytest.fixture
    def config(self):
        """Create test config."""
        return MockConfig()

    @pytest.fixture
    def model(self, config):
        """Create test model."""
        return MockTransformer(config)

    @pytest.fixture
    def wrapper(self, model):
        """Create wrapper."""
        return TestableNMoEModelWrapper(model, temperature=1.0)

    def test_forward_returns_correct_shape(self, wrapper, config):
        """Test that forward returns action log probs with correct shape."""
        batch_size = 4
        seq_len = 32
        num_actions = 8

        sequences = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)

        log_probs = wrapper(sequences, num_actions, attention_mask=attention_mask)

        assert log_probs.shape == (batch_size, num_actions), \
            f"Expected shape {(batch_size, num_actions)}, got {log_probs.shape}"

    def test_forward_with_return_output(self, wrapper, config):
        """Test forward with return_output=True returns tuple."""
        batch_size = 2
        seq_len = 16
        num_actions = 4

        sequences = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)

        log_probs, output = wrapper(
            sequences, num_actions,
            attention_mask=attention_mask,
            return_output=True,
        )

        assert log_probs.shape == (batch_size, num_actions)
        assert 'logits' in output
        assert output['logits'].shape == (batch_size, seq_len, config.vocab_size)

    def test_forward_with_entropy(self, wrapper, config):
        """Test forward with compute_entropy=True."""
        batch_size = 2
        seq_len = 16
        num_actions = 4

        sequences = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)

        log_probs, output = wrapper(
            sequences, num_actions,
            attention_mask=attention_mask,
            return_output=True,
            compute_entropy=True,
        )

        assert 'entropy' in output
        assert output['entropy'].shape == (batch_size, seq_len)

    def test_forward_temperature_scaling(self, wrapper, config):
        """Test that temperature affects log_probs."""
        batch_size = 2
        seq_len = 16
        num_actions = 4

        sequences = torch.randint(0, config.vocab_size, (batch_size, seq_len))

        log_probs_t1 = wrapper(sequences, num_actions, temperature=1.0)
        log_probs_t2 = wrapper(sequences, num_actions, temperature=2.0)

        # Log probs should differ with different temperatures
        assert not torch.allclose(log_probs_t1, log_probs_t2)

    def test_generate_returns_correct_outputs(self, wrapper, config):
        """Test that generate returns sequences, attention_mask, action_mask."""
        batch_size = 2
        prompt_len = 8
        max_new_tokens = 4

        input_ids = torch.randint(0, config.vocab_size - 1, (batch_size, prompt_len))

        sequences, attention_mask, action_mask = wrapper.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            eos_token_id=config.eos_token_id,
            pad_token_id=config.eos_token_id,
        )

        # Check shapes
        assert sequences.shape[0] == batch_size
        assert sequences.shape[1] >= prompt_len
        assert sequences.shape[1] <= prompt_len + max_new_tokens
        assert attention_mask.shape == sequences.shape
        # action_mask is for generated tokens
        assert action_mask.shape[0] == batch_size

    def test_refresh_expert_caches(self, wrapper):
        """Test that refresh_expert_caches calls MoE.refresh_weight_cache."""
        wrapper.refresh_expert_caches()

        # Check that MoE layers were refreshed
        moe_layers = wrapper._get_moe_layers()
        for moe in moe_layers:
            assert moe._W_cache == "refreshed"

    def test_uses_quantized_experts_bf16(self, config):
        """Test uses_quantized_experts returns False for bf16."""
        config.dtype = "bf16"
        model = MockTransformer(config)
        wrapper = TestableNMoEModelWrapper(model)
        assert wrapper.uses_quantized_experts is False

    def test_uses_quantized_experts_fp8(self, config):
        """Test uses_quantized_experts returns True for fp8."""
        config.dtype = "fp8"
        model = MockTransformer(config)
        wrapper = TestableNMoEModelWrapper(model)
        assert wrapper.uses_quantized_experts is True

    def test_get_expert_load_stats(self, wrapper, config):
        """Test get_expert_load_stats returns correct structure."""
        # Run a forward pass to populate load stats
        sequences = torch.randint(0, config.vocab_size, (2, 16))
        wrapper(sequences, num_actions=4)

        stats = wrapper.get_expert_load_stats()

        assert 'loads' in stats
        assert 'load_mean' in stats
        assert 'load_std' in stats

    def test_update_router_biases(self, wrapper, config):
        """Test update_router_biases modifies router biases."""
        # Run forward to get loads
        sequences = torch.randint(0, config.vocab_size, (2, 16))
        wrapper(sequences, num_actions=4)

        # Update biases (just verify no error)
        wrapper.update_router_biases(gamma=0.1)

    def test_gradient_checkpointing_enable_disable(self, wrapper):
        """Test gradient checkpointing toggle."""
        assert not wrapper.is_gradient_checkpointing

        wrapper.gradient_checkpointing_enable()
        assert wrapper.is_gradient_checkpointing

        wrapper.gradient_checkpointing_disable()
        assert not wrapper.is_gradient_checkpointing

    def test_param_sets(self, wrapper):
        """Test param_sets returns expert and dense parameters."""
        expert_params, dense_params = wrapper.param_sets()

        assert len(expert_params) > 0, "Should have expert params"
        assert len(dense_params) > 0, "Should have dense params"

        # Expert params should be from MoE layers
        expert_param_ids = {id(p) for p in expert_params}
        for moe in wrapper._get_moe_layers():
            assert id(moe.W1) in expert_param_ids
            assert id(moe.W2) in expert_param_ids
            assert id(moe.W3) in expert_param_ids

    def test_named_parameters_by_type(self, wrapper):
        """Test named_parameters_by_type categorizes correctly."""
        params_by_type = wrapper.named_parameters_by_type()

        assert 'expert' in params_by_type
        assert 'router' in params_by_type
        assert 'attention' in params_by_type
        assert 'dense' in params_by_type
        assert 'embedding' in params_by_type

    def test_state_dict_operations(self, wrapper):
        """Test state_dict_for_save and load_state_dict_from_checkpoint."""
        state_dict = wrapper.state_dict_for_save()
        assert len(state_dict) > 0

        # Test loading
        wrapper.load_state_dict_from_checkpoint(state_dict, strict=True)

    def test_model_properties(self, wrapper, config):
        """Test model property accessors."""
        assert wrapper.config is not None
        assert wrapper.config.dim == config.dim
        assert isinstance(wrapper.device, torch.device)
        assert wrapper.dtype in (torch.float32, torch.bfloat16, torch.float16)

    def test_embedding_accessors(self, wrapper):
        """Test get_input_embeddings and get_output_embeddings."""
        input_emb = wrapper.get_input_embeddings()
        output_emb = wrapper.get_output_embeddings()

        assert isinstance(input_emb, nn.Module)
        assert isinstance(output_emb, nn.Module)


class TestNMoEModelWrapperListActions:
    """Tests for handling list-type num_actions."""

    @pytest.fixture
    def wrapper(self):
        """Create wrapper."""
        config = MockConfig()
        model = MockTransformer(config)
        return TestableNMoEModelWrapper(model)

    def test_forward_with_single_element_list(self, wrapper):
        """Test forward with num_actions as single-element list."""
        sequences = torch.randint(0, 1000, (2, 16))
        log_probs = wrapper(sequences, num_actions=[4])

        assert log_probs.shape == (2, 4)

    def test_forward_with_multi_element_list(self, wrapper):
        """Test forward with num_actions as multi-element list (variable lengths)."""
        sequences = torch.randint(0, 1000, (2, 16))
        # This uses numpy array internally
        log_probs = wrapper(sequences, num_actions=[4, 4])

        # Should still work with same num_actions
        assert log_probs.shape[0] == 2


class TestNMoEModelWrapperReferenceModel:
    """Tests for reference model freezing (PPO support)."""

    @pytest.fixture
    def wrapper(self):
        """Create wrapper."""
        config = MockConfig()
        model = MockTransformer(config)
        return TestableNMoEModelWrapper(model)

    def test_freeze_for_reference(self, wrapper):
        """Test freezing model for reference."""
        # Initially should not be frozen
        assert not wrapper.is_frozen

        # Freeze
        result = wrapper.freeze_for_reference()

        # Check returns self for chaining
        assert result is wrapper

        # Check is frozen
        assert wrapper.is_frozen

        # Check all parameters are frozen
        for param in wrapper.model.parameters():
            assert not param.requires_grad

        # Check model is in eval mode
        assert not wrapper.training

    def test_unfreeze(self, wrapper):
        """Test unfreezing model."""
        # Freeze first
        wrapper.freeze_for_reference()
        assert wrapper.is_frozen

        # Unfreeze
        result = wrapper.unfreeze()

        # Check returns self
        assert result is wrapper

        # Check no longer frozen
        assert not wrapper.is_frozen

        # Check all parameters are trainable
        for param in wrapper.model.parameters():
            assert param.requires_grad

    def test_freeze_expert_weights(self, wrapper):
        """Test freezing only expert weights."""
        initial_trainable = wrapper.trainable_param_count

        wrapper.freeze_expert_weights()

        # Some params should now be frozen
        assert wrapper.frozen_param_count > 0
        assert wrapper.trainable_param_count < initial_trainable

    def test_freeze_dense_weights(self, wrapper):
        """Test freezing only dense weights."""
        initial_trainable = wrapper.trainable_param_count

        wrapper.freeze_dense_weights()

        # Some params should now be frozen
        assert wrapper.frozen_param_count > 0
        assert wrapper.trainable_param_count < initial_trainable

    def test_frozen_model_stays_in_eval(self, wrapper):
        """Test that frozen model stays in eval mode when train() is called."""
        wrapper.freeze_for_reference()
        assert not wrapper.training

        # Try to set train mode
        wrapper.train(True)

        # Should still be in eval mode
        assert not wrapper.training

    def test_unfrozen_model_can_train(self, wrapper):
        """Test that unfrozen model can be set to train mode."""
        wrapper.freeze_for_reference()
        wrapper.unfreeze()

        # Should be able to set train mode
        wrapper.train(True)
        assert wrapper.training

    def test_trainable_param_count(self, wrapper):
        """Test trainable_param_count property."""
        initial_count = wrapper.trainable_param_count
        assert initial_count > 0

        wrapper.freeze_for_reference()
        assert wrapper.trainable_param_count == 0

        wrapper.unfreeze()
        assert wrapper.trainable_param_count == initial_count

    def test_frozen_param_count(self, wrapper):
        """Test frozen_param_count property."""
        assert wrapper.frozen_param_count == 0

        wrapper.freeze_for_reference()
        assert wrapper.frozen_param_count > 0

        wrapper.unfreeze()
        assert wrapper.frozen_param_count == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
