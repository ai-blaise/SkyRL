"""Tests for NMoE configuration integration with SkyRL.

Tests the nmoe config schema, model factory, and config parsing.
"""

import os
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# Mock Config Classes (to avoid Hydra dependency in tests)
# =============================================================================

@dataclass
class MockLoraConfig:
    """Mock LoRA config."""
    rank: int = 0
    alpha: int = 16
    dropout: float = 0
    lora_sync_path: str = "/tmp/lora"
    target_modules: str = "all-linear"
    exclude_modules: Optional[str] = None
    init_method: str = "kaiming"

    def get(self, key, default=None):
        return getattr(self, key, default)


@dataclass
class MockModelConfig:
    """Mock model config."""
    path: str = "/path/to/model"
    type: str = "hf"
    lora: MockLoraConfig = field(default_factory=MockLoraConfig)

    def get(self, key, default=None):
        return getattr(self, key, default)


@dataclass
class MockNMoEConfig:
    """Mock nmoe config section."""
    model_type: str = "nmoe"
    hidden_size: int = 4096
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    intermediate_size: int = 11008
    moe_intermediate_size: int = 1408
    num_experts: int = 64
    num_experts_per_tok: int = 6
    n_shared_experts: int = 2
    first_k_dense_replace: int = 1
    router_aux_loss_coef: float = 0.001
    router_bias_update_rate: float = 0.0001
    attention_type: str = "mla"
    q_lora_rank: int = 1536
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    max_position_embeddings: int = 8192
    rope_theta: float = 50000.0
    rms_norm_eps: float = 1e-5
    vocab_size: int = 201088
    torch_dtype: str = "bfloat16"
    quantization: Optional[str] = None
    training: Dict[str, Any] = field(default_factory=lambda: {
        "gradient_checkpointing": True,
        "use_torch_compile": False,
    })
    rdep: Dict[str, Any] = field(default_factory=lambda: {
        "mode": "auto",
        "profile": "bf16",
        "capacity": 65536,
    })

    def get(self, key, default=None):
        return getattr(self, key, default)


@dataclass
class MockPolicyConfig:
    """Mock policy config."""
    model: MockModelConfig = field(default_factory=MockModelConfig)
    nmoe_config: Optional[MockNMoEConfig] = None
    sequence_parallel_size: int = 1
    use_torch_compile: bool = False
    model_config_kwargs: Dict[str, Any] = field(default_factory=dict)
    fsdp_config: Dict[str, Any] = field(default_factory=dict)
    optimizer_config: Dict[str, Any] = field(default_factory=dict)

    def get(self, key, default=None):
        return getattr(self, key, default)


@dataclass
class MockGeneratorConfig:
    """Mock generator config."""
    weight_transfer_threshold_cuda_ipc_GB: float = 0.5

    def get(self, key, default=None):
        return getattr(self, key, default)


@dataclass
class MockTrainerConfig:
    """Mock trainer config."""
    policy: MockPolicyConfig = field(default_factory=MockPolicyConfig)
    flash_attn: bool = True
    gradient_checkpointing: bool = True
    gradient_checkpointing_use_reentrant: bool = False
    use_sample_packing: bool = False
    bf16: bool = True

    def get(self, key, default=None):
        return getattr(self, key, default)


@dataclass
class MockConfig:
    """Mock Hydra config."""
    trainer: MockTrainerConfig = field(default_factory=MockTrainerConfig)
    generator: MockGeneratorConfig = field(default_factory=MockGeneratorConfig)

    def get(self, key, default=None):
        return getattr(self, key, default)


# =============================================================================
# Test: get_model_type
# =============================================================================

class TestGetModelType:
    """Tests for get_model_type function."""

    def test_default_is_hf(self):
        """Default model type should be hf."""
        from skyrl_train.model_factory import get_model_type

        cfg = MockConfig()
        assert get_model_type(cfg) == "hf"

    def test_explicit_hf_type(self):
        """Explicit hf type in model config."""
        from skyrl_train.model_factory import get_model_type

        cfg = MockConfig()
        cfg.trainer.policy.model.type = "hf"
        assert get_model_type(cfg) == "hf"

    def test_nmoe_type_from_model(self):
        """nmoe type from model.type field."""
        from skyrl_train.model_factory import get_model_type

        cfg = MockConfig()
        cfg.trainer.policy.model.type = "nmoe"
        assert get_model_type(cfg) == "nmoe"

    def test_nmoe_type_from_nmoe_config(self):
        """nmoe type inferred from nmoe_config section."""
        from skyrl_train.model_factory import get_model_type

        cfg = MockConfig()
        cfg.trainer.policy.model.type = "hf"  # This would normally be default
        cfg.trainer.policy.nmoe_config = MockNMoEConfig()
        assert get_model_type(cfg) == "nmoe"

    def test_case_insensitive(self):
        """Model type should be case insensitive."""
        from skyrl_train.model_factory import get_model_type

        cfg = MockConfig()
        cfg.trainer.policy.model.type = "NMoE"
        assert get_model_type(cfg) == "nmoe"


# =============================================================================
# Test: NMoE Config Schema
# =============================================================================

class TestNMoEConfigSchema:
    """Tests for nmoe config schema validation."""

    def test_nmoe_config_fields(self):
        """Test that all required nmoe config fields are present."""
        cfg = MockNMoEConfig()

        # Core fields
        assert cfg.model_type == "nmoe"
        assert cfg.hidden_size == 4096
        assert cfg.num_hidden_layers == 32
        assert cfg.num_attention_heads == 32

        # MoE fields
        assert cfg.num_experts == 64
        assert cfg.num_experts_per_tok == 6
        assert cfg.n_shared_experts == 2
        assert cfg.first_k_dense_replace == 1

        # Router fields
        assert cfg.router_aux_loss_coef == 0.001
        assert cfg.router_bias_update_rate == 0.0001

        # Attention fields
        assert cfg.attention_type == "mla"
        assert cfg.q_lora_rank == 1536
        assert cfg.kv_lora_rank == 512

    def test_nmoe_config_to_nmoe_model_config(self):
        """Test conversion from SkyRL nmoe config to NMoEModelConfig."""
        # Add nmoe path to sys.path
        import sys
        nmoe_path = str(Path(__file__).parent.parent.parent.parent / "nmoe")
        if nmoe_path not in sys.path:
            sys.path.insert(0, nmoe_path)

        try:
            from nmoe.unified.config import NMoEModelConfig
        except ImportError:
            pytest.skip("nmoe.unified.config not available")

        cfg = MockNMoEConfig()

        nmoe_model_config = NMoEModelConfig(
            hidden_size=cfg.hidden_size,
            num_hidden_layers=cfg.num_hidden_layers,
            num_attention_heads=cfg.num_attention_heads,
            intermediate_size=cfg.intermediate_size,
            moe_intermediate_size=cfg.moe_intermediate_size,
            num_experts=cfg.num_experts,
            num_experts_per_tok=cfg.num_experts_per_tok,
            n_shared_experts=cfg.n_shared_experts,
            first_k_dense_replace=cfg.first_k_dense_replace,
            router_aux_loss_coef=cfg.router_aux_loss_coef,
            router_bias_update_rate=cfg.router_bias_update_rate,
            attention_type=cfg.attention_type,
            q_lora_rank=cfg.q_lora_rank,
            kv_lora_rank=cfg.kv_lora_rank,
            qk_nope_head_dim=cfg.qk_nope_head_dim,
            qk_rope_head_dim=cfg.qk_rope_head_dim,
            v_head_dim=cfg.v_head_dim,
            vocab_size=cfg.vocab_size,
        )

        assert nmoe_model_config.hidden_size == cfg.hidden_size
        assert nmoe_model_config.num_experts == cfg.num_experts
        assert nmoe_model_config.is_moe is True


# =============================================================================
# Test: RDEP Config
# =============================================================================

class TestRDEPConfig:
    """Tests for RDEP (Expert Parallelism) configuration."""

    def test_rdep_defaults(self):
        """Test default RDEP configuration."""
        cfg = MockNMoEConfig()

        assert cfg.rdep["mode"] == "auto"
        assert cfg.rdep["profile"] == "bf16"
        assert cfg.rdep["capacity"] == 65536

    def test_rdep_mode_detection(self):
        """Test RDEP mode auto-detection logic."""
        # Add nmoe path to sys.path
        import sys
        nmoe_path = str(Path(__file__).parent.parent.parent.parent / "nmoe")
        if nmoe_path not in sys.path:
            sys.path.insert(0, nmoe_path)

        try:
            from nmoe.unified.config import NMoERDEPConfig
        except ImportError:
            pytest.skip("nmoe.unified.config not available")

        rdep_cfg = NMoERDEPConfig(mode="auto")

        # Single GPU
        assert rdep_cfg.detect_mode(world_size=1, local_world_size=1) == "single"

        # Multi-GPU single node
        assert rdep_cfg.detect_mode(world_size=8, local_world_size=8) == "ipc"

        # Multi-node
        assert rdep_cfg.detect_mode(world_size=16, local_world_size=8) == "hybrid"

    def test_rdep_profile_id(self):
        """Test RDEP profile ID mapping."""
        # Add nmoe path to sys.path
        import sys
        nmoe_path = str(Path(__file__).parent.parent.parent.parent / "nmoe")
        if nmoe_path not in sys.path:
            sys.path.insert(0, nmoe_path)

        try:
            from nmoe.unified.config import NMoERDEPConfig
        except ImportError:
            pytest.skip("nmoe.unified.config not available")

        assert NMoERDEPConfig(profile="bf16").get_profile_id() == -1
        assert NMoERDEPConfig(profile="fp8").get_profile_id() == 0
        assert NMoERDEPConfig(profile="nvfp4").get_profile_id() == 1


# =============================================================================
# Test: Model Factory
# =============================================================================

class TestModelFactory:
    """Tests for model factory functions."""

    def test_create_hf_wrapper(self):
        """Test creating HFModelWrapper."""
        from skyrl_train.model_factory import create_model_wrapper

        cfg = MockConfig()
        cfg.trainer.policy.model.type = "hf"

        # Mock HFModelWrapper to avoid loading actual model
        with patch("skyrl_train.model_factory._create_hf_wrapper") as mock_create:
            mock_wrapper = MagicMock()
            mock_create.return_value = mock_wrapper

            result = create_model_wrapper("/fake/path", cfg, for_training=True)

            mock_create.assert_called_once()
            assert result == mock_wrapper

    def test_create_nmoe_wrapper_from_config(self):
        """Test creating NMoEModelWrapper from config."""
        from skyrl_train.model_factory import create_model_wrapper

        cfg = MockConfig()
        cfg.trainer.policy.model.type = "nmoe"
        cfg.trainer.policy.nmoe_config = MockNMoEConfig()

        # Mock NMoEModelWrapper to avoid loading actual model
        with patch("skyrl_train.model_factory._create_nmoe_wrapper") as mock_create:
            mock_wrapper = MagicMock()
            mock_create.return_value = mock_wrapper

            result = create_model_wrapper("/fake/path", cfg, for_training=True)

            mock_create.assert_called_once()
            assert result == mock_wrapper

    def test_unknown_model_type_raises(self):
        """Test that unknown model type raises ValueError."""
        from skyrl_train.model_factory import create_model_wrapper

        cfg = MockConfig()
        cfg.trainer.policy.model.type = "unknown_type"

        with pytest.raises(ValueError, match="Unknown model_type"):
            create_model_wrapper("/fake/path", cfg)


# =============================================================================
# Test: Weight Extractor Factory
# =============================================================================

class TestWeightExtractorFactory:
    """Tests for weight extractor factory."""

    def test_create_nmoe_weight_extractor(self):
        """Test creating NMoE weight extractor selects correct type."""
        from skyrl_train.model_factory import get_model_type

        cfg = MockConfig()
        cfg.trainer.policy.model.type = "nmoe"
        cfg.trainer.policy.nmoe_config = MockNMoEConfig()

        # Verify that get_model_type returns "nmoe" for this config
        # The actual weight extractor creation is tested in test_nmoe_weight_extractor.py
        assert get_model_type(cfg) == "nmoe"

    def test_create_hf_weight_extractor(self):
        """Test that HF model type doesn't use nmoe extractor."""
        from skyrl_train.model_factory import get_model_type

        cfg = MockConfig()
        cfg.trainer.policy.model.type = "hf"

        assert get_model_type(cfg) == "hf"


# =============================================================================
# Test: Config File Loading
# =============================================================================

class TestConfigFileLoading:
    """Tests for loading config files."""

    def test_nmoe_model_yaml_exists(self):
        """Test that nmoe_model.yaml config file exists."""
        config_dir = Path(__file__).parent.parent / "skyrl_train" / "config" / "nmoe_config"
        nmoe_yaml = config_dir / "nmoe_model.yaml"

        assert nmoe_yaml.exists(), f"nmoe_model.yaml not found at {nmoe_yaml}"

    def test_nmoe_grpo_config_yaml_exists(self):
        """Test that nmoe_grpo_config.yaml exists."""
        config_dir = Path(__file__).parent.parent / "skyrl_train" / "config"
        grpo_yaml = config_dir / "nmoe_grpo_config.yaml"

        assert grpo_yaml.exists(), f"nmoe_grpo_config.yaml not found at {grpo_yaml}"

    def test_nmoe_model_yaml_has_required_fields(self):
        """Test that nmoe_model.yaml has required fields."""
        import yaml

        config_dir = Path(__file__).parent.parent / "skyrl_train" / "config" / "nmoe_config"
        nmoe_yaml = config_dir / "nmoe_model.yaml"

        with open(nmoe_yaml) as f:
            config = yaml.safe_load(f)

        # Check required fields
        assert "model_type" in config
        assert config["model_type"] == "nmoe"

        # MoE fields
        assert "num_experts" in config
        assert "num_experts_per_tok" in config
        assert "n_shared_experts" in config

        # Router fields
        assert "router_aux_loss_coef" in config
        assert "router_bias_update_rate" in config

        # RDEP section
        assert "rdep" in config
        assert "mode" in config["rdep"]
        assert "profile" in config["rdep"]

        # Training section
        assert "training" in config
        assert "gradient_checkpointing" in config["training"]


# =============================================================================
# Test: is_nmoe_model Detection
# =============================================================================

class TestIsNMoEModelDetection:
    """Tests for detecting nmoe models from checkpoints."""

    def test_is_nmoe_model_with_nmoe_checkpoint(self):
        """Test detection of nmoe model from config.json."""
        import json
        from skyrl_train.model_factory import is_nmoe_model

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a mock nmoe checkpoint
            config = {"model_type": "nmoe", "hidden_size": 4096}
            config_path = Path(tmpdir) / "config.json"
            with open(config_path, "w") as f:
                json.dump(config, f)

            assert is_nmoe_model(tmpdir) is True

    def test_is_nmoe_model_with_hf_checkpoint(self):
        """Test detection returns False for HF model."""
        import json
        from skyrl_train.model_factory import is_nmoe_model

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a mock HF checkpoint
            config = {"model_type": "llama", "hidden_size": 4096}
            config_path = Path(tmpdir) / "config.json"
            with open(config_path, "w") as f:
                json.dump(config, f)

            assert is_nmoe_model(tmpdir) is False

    def test_is_nmoe_model_with_no_config(self):
        """Test detection returns False when no config.json."""
        from skyrl_train.model_factory import is_nmoe_model

        with tempfile.TemporaryDirectory() as tmpdir:
            assert is_nmoe_model(tmpdir) is False


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
