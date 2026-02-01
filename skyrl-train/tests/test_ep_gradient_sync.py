"""Tests for EP-aware gradient synchronization utilities.

Tests the integration between SkyRL's FSDP utilities and nmoe's expert parallelism.
"""

import pytest
import sys
sys.path.insert(0, '/home/nourdine/sglang_nmoe/nether-soup/SkyRL/skyrl-train')

import torch
import torch.nn as nn


class TestGetEPGroupInfo:
    """Test EP group info utilities."""

    def test_get_ep_group_info_not_initialized(self):
        """Test EP group info when dist is not initialized."""
        from skyrl_train.distributed.fsdp_utils import get_ep_group_info

        ep_rank, ep_world_size = get_ep_group_info(None)
        assert ep_rank == 0
        assert ep_world_size == 1

    def test_get_ep_group_info_returns_tuple(self):
        """Test that get_ep_group_info returns a tuple."""
        from skyrl_train.distributed.fsdp_utils import get_ep_group_info

        result = get_ep_group_info()
        assert isinstance(result, tuple)
        assert len(result) == 2


class TestSyncExpertGradients:
    """Test expert gradient synchronization."""

    def test_sync_expert_gradients_not_initialized(self):
        """Test sync_expert_gradients when dist is not initialized."""
        from skyrl_train.distributed.fsdp_utils import sync_expert_gradients

        # Create dummy parameters
        param1 = nn.Parameter(torch.randn(10, 10))
        param1.grad = torch.randn(10, 10)
        param2 = nn.Parameter(torch.randn(5, 5))
        param2.grad = torch.randn(5, 5)

        # Should return None when dist is not initialized
        result = sync_expert_gradients([param1, param2])
        assert result is None

    def test_sync_expert_gradients_handles_none_grad(self):
        """Test sync_expert_gradients handles parameters without gradients."""
        from skyrl_train.distributed.fsdp_utils import sync_expert_gradients

        param1 = nn.Parameter(torch.randn(10, 10))
        param1.grad = None  # No gradient

        # Should not raise
        result = sync_expert_gradients([param1])
        assert result is None

    def test_sync_expert_gradients_signature(self):
        """Test sync_expert_gradients function signature."""
        from skyrl_train.distributed.fsdp_utils import sync_expert_gradients
        import inspect

        sig = inspect.signature(sync_expert_gradients)
        params = list(sig.parameters.keys())

        assert 'expert_params' in params
        assert 'ep_group' in params
        assert 'async_op' in params


class TestAverageExpertGradientsAfterSync:
    """Test gradient averaging after async sync."""

    def test_average_expert_gradients_after_sync(self):
        """Test that gradients are correctly divided."""
        from skyrl_train.distributed.fsdp_utils import average_expert_gradients_after_sync

        param = nn.Parameter(torch.ones(4, 4))
        param.grad = torch.ones(4, 4) * 8.0

        average_expert_gradients_after_sync([param], ep_world_size=4)

        # Gradient should be divided by 4
        assert torch.allclose(param.grad, torch.ones(4, 4) * 2.0)

    def test_average_expert_gradients_handles_none(self):
        """Test that None gradients are handled."""
        from skyrl_train.distributed.fsdp_utils import average_expert_gradients_after_sync

        param = nn.Parameter(torch.ones(4, 4))
        param.grad = None

        # Should not raise
        average_expert_gradients_after_sync([param], ep_world_size=4)


class TestReduceScatterExpertGradients:
    """Test reduce-scatter gradient operation."""

    def test_reduce_scatter_not_initialized(self):
        """Test reduce_scatter when dist is not initialized."""
        from skyrl_train.distributed.fsdp_utils import reduce_scatter_expert_gradients

        param = nn.Parameter(torch.randn(10, 10))
        param.grad = torch.randn(10, 10)

        result = reduce_scatter_expert_gradients([param])
        assert result is None

    def test_reduce_scatter_signature(self):
        """Test reduce_scatter function signature."""
        from skyrl_train.distributed.fsdp_utils import reduce_scatter_expert_gradients
        import inspect

        sig = inspect.signature(reduce_scatter_expert_gradients)
        params = list(sig.parameters.keys())

        assert 'expert_params' in params
        assert 'ep_group' in params
        assert 'async_op' in params


class TestEPGradientSynchronizer:
    """Test EPGradientSynchronizer class."""

    def test_init(self):
        """Test EPGradientSynchronizer initialization."""
        from skyrl_train.distributed.fsdp_utils import EPGradientSynchronizer

        model = nn.Linear(10, 10)
        synchronizer = EPGradientSynchronizer(model)

        assert synchronizer.model is model
        assert synchronizer.ep_group is None
        assert 'expert' in synchronizer.expert_param_names

    def test_init_with_custom_patterns(self):
        """Test EPGradientSynchronizer with custom expert patterns."""
        from skyrl_train.distributed.fsdp_utils import EPGradientSynchronizer

        model = nn.Linear(10, 10)
        patterns = ['custom_expert', 'my_moe']
        synchronizer = EPGradientSynchronizer(
            model,
            expert_param_names=patterns,
        )

        assert synchronizer.expert_param_names == patterns

    def test_is_expert_param(self):
        """Test _is_expert_param method."""
        from skyrl_train.distributed.fsdp_utils import EPGradientSynchronizer

        model = nn.Linear(10, 10)
        synchronizer = EPGradientSynchronizer(model)

        assert synchronizer._is_expert_param('layer.expert.weight')
        assert synchronizer._is_expert_param('moe.ffn.bias')
        assert synchronizer._is_expert_param('EXPERT_LAYER')  # Case insensitive
        assert not synchronizer._is_expert_param('layer.dense.weight')
        assert not synchronizer._is_expert_param('embedding.weight')

    def test_categorize_params_simple_model(self):
        """Test parameter categorization with a simple model."""
        from skyrl_train.distributed.fsdp_utils import EPGradientSynchronizer

        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.expert_layer = nn.Linear(10, 10)
                self.dense_layer = nn.Linear(10, 10)

        model = SimpleModel()
        synchronizer = EPGradientSynchronizer(model)

        expert_params = synchronizer.expert_params
        non_expert_params = synchronizer.non_expert_params

        # expert_layer params should be in expert_params
        assert len(expert_params) == 2  # weight and bias
        assert len(non_expert_params) == 2  # weight and bias

    def test_categorize_params_with_frozen(self):
        """Test that frozen params are excluded."""
        from skyrl_train.distributed.fsdp_utils import EPGradientSynchronizer

        class ModelWithFrozen(nn.Module):
            def __init__(self):
                super().__init__()
                self.expert_layer = nn.Linear(10, 10)
                self.frozen_layer = nn.Linear(10, 10)

        model = ModelWithFrozen()
        # Freeze one layer
        for param in model.frozen_layer.parameters():
            param.requires_grad = False

        synchronizer = EPGradientSynchronizer(model)

        expert_params = synchronizer.expert_params
        non_expert_params = synchronizer.non_expert_params

        # Frozen params should be excluded
        total_trainable = len(expert_params) + len(non_expert_params)
        assert total_trainable == 2  # Only expert_layer params

    def test_sync_expert_grads_not_initialized(self):
        """Test sync_expert_grads when dist is not initialized."""
        from skyrl_train.distributed.fsdp_utils import EPGradientSynchronizer

        class ExpertModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.expert = nn.Linear(10, 10)

        model = ExpertModel()
        # Set up gradients
        model.expert.weight.grad = torch.randn(10, 10)

        synchronizer = EPGradientSynchronizer(model)
        result = synchronizer.sync_expert_grads()

        # Should return None when dist is not initialized
        assert result is None

    def test_repr(self):
        """Test __repr__ output."""
        from skyrl_train.distributed.fsdp_utils import EPGradientSynchronizer

        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.expert = nn.Linear(10, 10)
                self.dense = nn.Linear(10, 10)

        model = SimpleModel()
        synchronizer = EPGradientSynchronizer(model)

        repr_str = repr(synchronizer)
        assert 'EPGradientSynchronizer' in repr_str
        assert 'ep=' in repr_str
        assert 'expert_params=' in repr_str
        assert 'non_expert_params=' in repr_str


class TestEPGradientSynchronizerMoEModel:
    """Test EPGradientSynchronizer with MoE-like model structure."""

    def test_moe_model_categorization(self):
        """Test categorization with a model mimicking MoE structure."""
        from skyrl_train.distributed.fsdp_utils import EPGradientSynchronizer

        class MoEBlock(nn.Module):
            def __init__(self, dim, n_experts):
                super().__init__()
                self.router = nn.Linear(dim, n_experts)
                self.experts = nn.ModuleList([
                    nn.Sequential(
                        nn.Linear(dim, dim * 4),
                        nn.ReLU(),
                        nn.Linear(dim * 4, dim),
                    ) for _ in range(n_experts)
                ])
                self.norm = nn.LayerNorm(dim)

            def forward(self, x):
                return x

        class MoEModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(1000, 64)
                self.moe_block = MoEBlock(64, 4)
                self.output = nn.Linear(64, 1000)

        model = MoEModel()
        synchronizer = EPGradientSynchronizer(
            model,
            expert_param_names=['expert', 'moe'],
        )

        expert_params = synchronizer.expert_params
        non_expert_params = synchronizer.non_expert_params

        # MoE block experts should be in expert_params
        # embed, output, router, norm should be in non_expert_params
        assert len(expert_params) > 0
        assert len(non_expert_params) > 0

        # Verify expert params are from moe_block
        total_expert_elements = sum(p.numel() for p in expert_params)
        assert total_expert_elements > 0


class TestIntegrationWithBridge:
    """Test integration between EPGradientSynchronizer and SkyRLRdepBridge."""

    def test_synchronizer_accepts_bridge_ep_group(self):
        """Test that synchronizer can accept ep_group from bridge."""
        from skyrl_train.distributed.fsdp_utils import EPGradientSynchronizer

        # Simulate what would happen with a bridge
        # bridge.ep_group would be a ProcessGroup, but we use None for unit test
        ep_group = None

        model = nn.Linear(10, 10)
        synchronizer = EPGradientSynchronizer(model, ep_group=ep_group)

        assert synchronizer.ep_group is ep_group

    def test_full_workflow_simulation(self):
        """Simulate full workflow: forward, backward, sync."""
        from skyrl_train.distributed.fsdp_utils import EPGradientSynchronizer

        class SimpleExpertModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.expert_ffn = nn.Linear(10, 10)
                self.output = nn.Linear(10, 5)

            def forward(self, x):
                x = self.expert_ffn(x)
                return self.output(x)

        model = SimpleExpertModel()
        synchronizer = EPGradientSynchronizer(model)

        # Forward pass
        x = torch.randn(4, 10)
        out = model(x)

        # Backward pass
        loss = out.sum()
        loss.backward()

        # Verify gradients exist
        assert model.expert_ffn.weight.grad is not None
        assert model.output.weight.grad is not None

        # Sync (no-op when dist not initialized)
        result = synchronizer.sync_expert_grads()
        assert result is None

        # Gradients should still be valid
        assert model.expert_ffn.weight.grad is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
