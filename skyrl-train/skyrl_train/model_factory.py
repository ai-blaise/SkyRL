"""Model Factory for SkyRL.

This module provides a unified factory for creating model wrappers based on
config settings. It supports:
- HuggingFace models via HFModelWrapper (default)
- NMoE models via NMoEModelWrapper

Usage:
    from skyrl_train.model_factory import create_model_wrapper

    # Creates HFModelWrapper or NMoEModelWrapper based on config
    wrapper = create_model_wrapper(model_path, cfg, for_training=True)
"""

from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


def get_model_type(cfg: Any) -> str:
    """Extract model type from config.

    Checks multiple config locations for model_type (in priority order):
    1. cfg.trainer.policy.nmoe_config.model_type == "nmoe" (presence of nmoe_config)
    2. cfg.trainer.policy.model.type (explicit type field)
    3. Falls back to "hf" (HuggingFace)

    Args:
        cfg: Hydra config object

    Returns:
        Model type string: "hf" or "nmoe"
    """
    # Check nmoe_config first - its presence indicates nmoe model
    try:
        nmoe_cfg = cfg.trainer.policy.get("nmoe_config", None)
        if nmoe_cfg and nmoe_cfg.get("model_type") == "nmoe":
            return "nmoe"
    except (AttributeError, KeyError):
        pass

    # Check direct model.type field
    try:
        model_type = cfg.trainer.policy.model.get("type", None)
        if model_type:
            return model_type.lower()
    except (AttributeError, KeyError):
        pass

    # Default to HuggingFace
    return "hf"


def create_model_wrapper(
    model_path: str,
    cfg: Any,
    for_training: bool = True,
    **kwargs,
) -> nn.Module:
    """Create a model wrapper based on config settings.

    This factory function instantiates the appropriate model wrapper:
    - HFModelWrapper for standard HuggingFace models
    - NMoEModelWrapper for nmoe MoE models

    Args:
        model_path: Path to model checkpoint or HF model ID
        cfg: Hydra config with trainer.policy settings
        for_training: If True, prepare for training (e.g., enable grad checkpointing)
        **kwargs: Additional kwargs passed to the wrapper

    Returns:
        Model wrapper instance (HFModelWrapper or NMoEModelWrapper)

    Raises:
        ValueError: If model_type is unknown
        ImportError: If nmoe is not installed but model_type is "nmoe"
    """
    model_type = get_model_type(cfg)

    if model_type == "hf":
        return _create_hf_wrapper(model_path, cfg, for_training, **kwargs)
    elif model_type == "nmoe":
        return _create_nmoe_wrapper(model_path, cfg, for_training, **kwargs)
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Expected 'hf' or 'nmoe'.")


def _create_hf_wrapper(
    model_path: str,
    cfg: Any,
    for_training: bool,
    **kwargs,
) -> nn.Module:
    """Create HFModelWrapper for HuggingFace models."""
    from skyrl_train.model_wrapper import HFModelWrapper

    policy_cfg = cfg.trainer.policy
    model_cfg = policy_cfg.model
    lora_cfg = model_cfg.get("lora", {})

    wrapper = HFModelWrapper(
        model_path,
        use_flash_attention_2=cfg.trainer.get("flash_attn", True),
        bf16=not for_training,  # fp32 during training, bf16 for inference
        lora_rank=lora_cfg.get("rank", 0),
        lora_alpha=lora_cfg.get("alpha", 16),
        lora_dropout=lora_cfg.get("dropout", 0),
        lora_init_method=lora_cfg.get("init_method", "kaiming"),
        target_modules=lora_cfg.get("target_modules", "all-linear"),
        exclude_modules=lora_cfg.get("exclude_modules", None),
        sequence_parallel_size=policy_cfg.get("sequence_parallel_size", 1),
        use_sample_packing=cfg.trainer.get("use_sample_packing", False),
        use_torch_compile=policy_cfg.get("use_torch_compile", False),
        model_config_kwargs=policy_cfg.get("model_config_kwargs", {}),
        **kwargs,
    )

    if for_training and cfg.trainer.get("gradient_checkpointing", False):
        wrapper.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={
                "use_reentrant": cfg.trainer.get("gradient_checkpointing_use_reentrant", False)
            }
        )

    logger.info(f"[ModelFactory] Created HFModelWrapper for {model_path}")
    return wrapper


def _create_nmoe_wrapper(
    model_path: str,
    cfg: Any,
    for_training: bool,
    **kwargs,
) -> nn.Module:
    """Create NMoEModelWrapper for nmoe models."""
    try:
        from skyrl_train.model_wrapper_nmoe import NMoEModelWrapper
    except ImportError as e:
        raise ImportError(
            "NMoEModelWrapper not available. Make sure nmoe is installed and "
            "skyrl_train.model_wrapper_nmoe exists."
        ) from e

    policy_cfg = cfg.trainer.policy
    nmoe_cfg = policy_cfg.get("nmoe_config", {})
    training_cfg = nmoe_cfg.get("training", {})

    # Build config for NMoEModelWrapper
    # First, try to load from checkpoint config.json
    wrapper_kwargs = {
        "temperature": 1.0,
        "use_torch_compile": training_cfg.get("use_torch_compile", False),
        "gradient_checkpointing": (
            for_training and training_cfg.get("gradient_checkpointing", True)
        ),
    }

    # If model_path points to a checkpoint, load the model
    # Otherwise, create from config
    import os
    if os.path.exists(model_path):
        # Load from checkpoint directory
        wrapper = NMoEModelWrapper(model_path, **wrapper_kwargs, **kwargs)
        logger.info(f"[ModelFactory] Created NMoEModelWrapper from checkpoint: {model_path}")
    else:
        # model_path might be a config dict or we need to build from nmoe_config
        from nmoe.unified.config import NMoEModelConfig

        # Build config from nmoe_config section
        nmoe_model_config = NMoEModelConfig(
            hidden_size=nmoe_cfg.get("hidden_size"),
            num_hidden_layers=nmoe_cfg.get("num_hidden_layers"),
            num_attention_heads=nmoe_cfg.get("num_attention_heads"),
            intermediate_size=nmoe_cfg.get("intermediate_size"),
            moe_intermediate_size=nmoe_cfg.get("moe_intermediate_size"),
            num_experts=nmoe_cfg.get("num_experts"),
            num_experts_per_tok=nmoe_cfg.get("num_experts_per_tok"),
            n_shared_experts=nmoe_cfg.get("n_shared_experts", 2),
            first_k_dense_replace=nmoe_cfg.get("first_k_dense_replace", 1),
            router_aux_loss_coef=nmoe_cfg.get("router_aux_loss_coef", 0.0),
            router_bias_update_rate=nmoe_cfg.get("router_bias_update_rate", 1e-4),
            attention_type=nmoe_cfg.get("attention_type", "mla"),
            q_lora_rank=nmoe_cfg.get("q_lora_rank", 1536),
            kv_lora_rank=nmoe_cfg.get("kv_lora_rank", 512),
            qk_nope_head_dim=nmoe_cfg.get("qk_nope_head_dim", 128),
            qk_rope_head_dim=nmoe_cfg.get("qk_rope_head_dim", 64),
            v_head_dim=nmoe_cfg.get("v_head_dim", 128),
            max_position_embeddings=nmoe_cfg.get("max_position_embeddings", 8192),
            rope_theta=nmoe_cfg.get("rope_theta", 50000.0),
            rms_norm_eps=nmoe_cfg.get("rms_norm_eps", 1e-5),
            vocab_size=nmoe_cfg.get("vocab_size", 201088),
            torch_dtype=nmoe_cfg.get("torch_dtype", "bfloat16"),
            quantization=nmoe_cfg.get("quantization"),
        )

        wrapper = NMoEModelWrapper(nmoe_model_config, **wrapper_kwargs, **kwargs)
        logger.info(f"[ModelFactory] Created NMoEModelWrapper from config")

    return wrapper


def create_weight_extractor(
    model: nn.Module,
    cfg: Any,
    ep_group: Optional["torch.distributed.ProcessGroup"] = None,
) -> "WeightExtractor":
    """Create appropriate weight extractor for the model.

    Args:
        model: The model wrapper (HFModelWrapper or NMoEModelWrapper)
        cfg: Hydra config
        ep_group: Expert parallel process group (for NMoE models)

    Returns:
        WeightExtractor instance
    """
    model_type = get_model_type(cfg)

    if model_type == "nmoe":
        from skyrl_train.distributed.nmoe_weight_extractor import create_nmoe_weight_extractor

        nmoe_cfg = cfg.trainer.policy.get("nmoe_config", {})
        batch_threshold = cfg.generator.get("weight_transfer_threshold_cuda_ipc_GB", 0.5)

        return create_nmoe_weight_extractor(
            model=model,
            ep_group=ep_group,
            batch_size_threshold_gb=batch_threshold,
        )
    else:
        # Default FSDP weight extractor
        from skyrl_train.workers.fsdp.fsdp_worker import FSDPWeightExtractor
        from skyrl_train.weight_sync import CudaIpcTransferStrategy

        # Check if CUDA IPC is being used
        transfer_strategy_cls = cfg.get("_transfer_strategy_cls", None)
        group_by_module = transfer_strategy_cls is CudaIpcTransferStrategy

        return FSDPWeightExtractor(
            model.model if hasattr(model, "model") else model,
            group_by_module=group_by_module,
            batch_size_threshold_gb=(
                cfg.generator.get("weight_transfer_threshold_cuda_ipc_GB", 0.5)
                if group_by_module else 0.0
            ),
        )


def get_nmoe_config_from_checkpoint(checkpoint_path: str) -> Optional[Dict[str, Any]]:
    """Load nmoe config from checkpoint directory.

    Args:
        checkpoint_path: Path to checkpoint directory

    Returns:
        Dict with nmoe config, or None if not an nmoe checkpoint
    """
    import json
    import os

    config_path = os.path.join(checkpoint_path, "config.json")
    if not os.path.exists(config_path):
        return None

    with open(config_path) as f:
        config = json.load(f)

    # Check if this is an nmoe model
    if config.get("model_type") == "nmoe":
        return config

    return None


def is_nmoe_model(model_path: str) -> bool:
    """Check if a model path points to an nmoe model.

    Args:
        model_path: Path to checkpoint or model ID

    Returns:
        True if this is an nmoe model
    """
    config = get_nmoe_config_from_checkpoint(model_path)
    return config is not None
