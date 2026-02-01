"""Tests for SGLang nmoe model loader integration.

Tests the nmoe model loader and model wrapper for SGLang serving.

Note: These tests avoid importing the full sglang package since it requires
many dependencies. Instead, we test by parsing files directly or using
importlib to load specific modules.
"""

import ast
import importlib.util
import sys
from pathlib import Path

import pytest
import torch

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

SGLANG_PATH = Path(__file__).parent.parent.parent.parent / "sglang" / "python"


def load_module_without_sglang_init(module_path: Path, module_name: str):
    """Load a Python module without triggering sglang/__init__.py imports."""
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    return module, spec


def parse_python_file(file_path: Path) -> ast.Module:
    """Parse a Python file and return its AST."""
    with open(file_path) as f:
        return ast.parse(f.read())


# =============================================================================
# Test: LoadFormat Enum
# =============================================================================

class TestLoadFormatEnum:
    """Tests for LoadFormat enum with NMOE value."""

    def test_nmoe_load_format_exists(self):
        """Test that NMOE LoadFormat value exists in load_config.py."""
        load_config_path = SGLANG_PATH / "sglang" / "srt" / "configs" / "load_config.py"
        content = load_config_path.read_text()

        # Check that NMOE is defined in LoadFormat enum
        assert 'NMOE = "nmoe"' in content

    def test_nmoe_load_format_in_choices(self):
        """Test that nmoe is in server_args choices."""
        server_args_path = SGLANG_PATH / "sglang" / "srt" / "server_args.py"
        content = server_args_path.read_text()
        assert '"nmoe"' in content or "'nmoe'" in content


# =============================================================================
# Test: NMoEModelLoader
# =============================================================================

class TestNMoEModelLoader:
    """Tests for NMoEModelLoader class."""

    def test_loader_file_exists(self):
        """Test that nmoe_loader.py exists."""
        loader_path = SGLANG_PATH / "sglang" / "srt" / "model_loader" / "nmoe_loader.py"
        assert loader_path.exists()

    def test_loader_parses(self):
        """Test that nmoe_loader.py parses correctly."""
        loader_path = SGLANG_PATH / "sglang" / "srt" / "model_loader" / "nmoe_loader.py"
        tree = parse_python_file(loader_path)
        assert tree is not None

    def test_loader_defines_required_classes(self):
        """Test that loader defines required classes and functions."""
        loader_path = SGLANG_PATH / "sglang" / "srt" / "model_loader" / "nmoe_loader.py"
        tree = parse_python_file(loader_path)

        # Find all class and function definitions
        class_names = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        func_names = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]

        # Check required classes
        assert "NMoEModelLoader" in class_names

        # Check required functions
        assert "_find_checkpoint_files" in func_names
        assert "_load_nmoe_checkpoint" in func_names
        assert "_expand_expert_weights" in func_names
        assert "_map_block_name" in func_names

    def test_loader_registered_in_get_model_loader(self):
        """Test that NMoEModelLoader is registered in get_model_loader."""
        loader_main_path = SGLANG_PATH / "sglang" / "srt" / "model_loader" / "loader.py"
        content = loader_main_path.read_text()

        assert "LoadFormat.NMOE" in content
        assert "NMoEModelLoader" in content


# =============================================================================
# Test: Weight Name Mapping
# =============================================================================

class TestWeightNameMapping:
    """Tests for nmoe to HF weight name mapping."""

    def test_weight_mapping_defined(self):
        """Test that weight mapping dict is defined."""
        loader_path = SGLANG_PATH / "sglang" / "srt" / "model_loader" / "nmoe_loader.py"
        content = loader_path.read_text()

        # Check for NMOE_TO_HF_MAPPING definition
        assert "NMOE_TO_HF_MAPPING" in content
        assert '"embedding.weight": "model.embed_tokens.weight"' in content
        assert '"lm_head.weight": "lm_head.weight"' in content

    def test_block_name_mapping_logic(self):
        """Test block name mapping by checking code patterns."""
        loader_path = SGLANG_PATH / "sglang" / "srt" / "model_loader" / "nmoe_loader.py"
        content = loader_path.read_text()

        # Check that mapping function handles key patterns
        assert "attn_norm" in content
        assert "ffn_norm" in content
        assert "input_layernorm" in content
        assert "post_attention_layernorm" in content
        assert "gate_proj" in content
        assert "up_proj" in content
        assert "down_proj" in content


# =============================================================================
# Test: Expert Weight Expansion
# =============================================================================

class TestExpertWeightExpansion:
    """Tests for expert weight expansion logic."""

    def test_expand_expert_weights_function_exists(self):
        """Test that _expand_expert_weights function is defined."""
        loader_path = SGLANG_PATH / "sglang" / "srt" / "model_loader" / "nmoe_loader.py"
        tree = parse_python_file(loader_path)

        func_names = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        assert "_expand_expert_weights" in func_names

    def test_expand_logic_handles_batched_experts(self):
        """Test that expansion logic handles batched expert weights."""
        loader_path = SGLANG_PATH / "sglang" / "srt" / "model_loader" / "nmoe_loader.py"
        content = loader_path.read_text()

        # Check that the function handles the W1, W3, W2 -> gate_proj, up_proj, down_proj mapping
        assert ".mlp.experts.W" in content
        assert "gate_proj" in content
        assert "up_proj" in content
        assert "down_proj" in content
        # Check for tensor operations
        assert ".t()" in content or ".transpose" in content  # Transpose for W1/W3
        assert "expert_idx" in content or "expert_weight" in content


# =============================================================================
# Test: NMoEForCausalLM Model
# =============================================================================

class TestNMoEForCausalLM:
    """Tests for NMoEForCausalLM model wrapper."""

    def test_model_file_exists(self):
        """Test that nmoe.py model file exists."""
        model_path = SGLANG_PATH / "sglang" / "srt" / "models" / "nmoe.py"
        assert model_path.exists()

    def test_model_parses(self):
        """Test that nmoe.py parses correctly."""
        model_path = SGLANG_PATH / "sglang" / "srt" / "models" / "nmoe.py"
        tree = parse_python_file(model_path)
        assert tree is not None

    def test_model_defines_required_classes(self):
        """Test that model defines required classes."""
        model_path = SGLANG_PATH / "sglang" / "srt" / "models" / "nmoe.py"
        tree = parse_python_file(model_path)

        class_names = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]

        assert "NMoEConfig" in class_names
        assert "NMoEForCausalLM" in class_names

    def test_entry_class_defined(self):
        """Test that EntryClass is defined."""
        model_path = SGLANG_PATH / "sglang" / "srt" / "models" / "nmoe.py"
        content = model_path.read_text()

        assert "EntryClass = [NMoEForCausalLM]" in content

    def test_model_has_load_weights_method(self):
        """Test that model defines load_weights method."""
        model_path = SGLANG_PATH / "sglang" / "srt" / "models" / "nmoe.py"
        content = model_path.read_text()

        assert "def load_weights" in content

    def test_model_has_forward_method(self):
        """Test that model defines forward method."""
        model_path = SGLANG_PATH / "sglang" / "srt" / "models" / "nmoe.py"
        content = model_path.read_text()

        assert "def forward" in content


# =============================================================================
# Test: get_model_loader Integration
# =============================================================================

class TestGetModelLoaderIntegration:
    """Tests for get_model_loader integration with nmoe."""

    def test_nmoe_loader_registered(self):
        """Test that NMoEModelLoader is registered in get_model_loader."""
        loader_path = SGLANG_PATH / "sglang" / "srt" / "model_loader" / "loader.py"
        content = loader_path.read_text()

        # Check that NMOE format is handled
        assert "LoadFormat.NMOE" in content
        assert "NMoEModelLoader" in content
        assert "from sglang.srt.model_loader.nmoe_loader import NMoEModelLoader" in content


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
