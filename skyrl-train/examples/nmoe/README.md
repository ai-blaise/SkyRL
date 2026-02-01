# NMoE Training with SkyRL

This example demonstrates training nmoe (B200-optimized MoE) models with SkyRL using GRPO (Group Relative Policy Optimization).

## Overview

nmoe is a high-performance MoE (Mixture of Experts) training library optimized for NVIDIA B200 GPUs. This integration allows you to:

- Train nmoe models with reinforcement learning algorithms (GRPO, PPO)
- Use SGLang as the inference backend for efficient rollout generation
- Leverage Expert Parallelism (EP) via RDEP dispatcher
- Support FP8/NVFP4 quantized expert weights

## Requirements

- NVIDIA B200 GPUs (8 GPUs recommended)
- nmoe library installed
- SGLang with nmoe backend support
- SkyRL training framework

## Quick Start

### 1. Prepare Your Data

Training data should be in parquet format with at least a `prompt` column:

```python
import pandas as pd

data = [
    {"prompt": "Solve: What is 2+2?"},
    {"prompt": "Solve: What is the derivative of x^2?"},
    # ... more prompts
]
df = pd.DataFrame(data)
df.to_parquet("train_data.parquet")
```

### 2. Run Training

```bash
cd SkyRL/skyrl-train

python -m skyrl_train.cli \
    --config-path examples/nmoe \
    --config-name config \
    trainer.policy.model.path=/path/to/nmoe/checkpoint \
    data.train_data=["path/to/train_data.parquet"]
```

### 3. Monitor Training

Training logs are sent to Weights & Biases by default. You can view:
- Policy loss
- KL divergence
- Reward statistics
- Expert load balancing metrics

## Configuration

### Key Settings

| Setting | Description | Default |
|---------|-------------|---------|
| `trainer.strategy` | Training strategy | `fsdp2` |
| `trainer.policy.model.type` | Model type | `nmoe` |
| `trainer.algorithm.advantage_estimator` | RL algorithm | `grpo` |
| `generator.backend` | Inference backend | `sglang` |
| `trainer.policy.nmoe_config.rdep.mode` | EP mode | `auto` |

### Expert Parallelism Modes

The RDEP dispatcher supports three modes:

- **single**: Single GPU, no expert parallelism
- **ipc**: Multi-GPU on single node using CUDA IPC
- **hybrid**: Multi-node using NVSHMEM

Set `trainer.policy.nmoe_config.rdep.mode: auto` to automatically detect the best mode.

### Quantization Profiles

Expert weights can be quantized for memory efficiency:

- **bf16**: Full BFloat16 precision (default)
- **fp8**: FP8 E4M3 quantized experts
- **nvfp4**: NVIDIA FP4 quantized experts (requires SM100)

Set via `trainer.policy.nmoe_config.rdep.profile`.

## Files

- `config.yaml` - Main training configuration
- `run_training.sh` - Convenience script for launching training (TODO)

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     SkyRL Training Loop                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐    │
│  │   Policy     │     │  Reference   │     │  Generator   │    │
│  │   Model      │     │    Model     │     │  (SGLang)    │    │
│  │ (NMoEWrapper)│     │(NMoEWrapper) │     │              │    │
│  └──────┬───────┘     └──────┬───────┘     └──────┬───────┘    │
│         │                    │                    │             │
│         │ FSDP Sharded      │ Frozen            │ EP-aware    │
│         │                    │                    │ inference   │
│         ▼                    ▼                    ▼             │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              RDEP Expert Parallelism                      │  │
│  │  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ │  │
│  │  │GPU0 │ │GPU1 │ │GPU2 │ │GPU3 │ │GPU4 │ │GPU5 │ │GPU6 │ │  │
│  │  └─────┘ └─────┘ └─────┘ └─────┘ └─────┘ └─────┘ └─────┘ │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Troubleshooting

### CUDA Out of Memory

Try reducing batch sizes:
```bash
trainer.train_batch_size=64 \
trainer.mini_batch_size=8 \
trainer.micro_train_batch_size_per_gpu=1
```

Or enable CPU offload:
```bash
trainer.policy.fsdp_config.cpu_offload=true
```

### Weight Sync Failures

If weight sync between training and inference fails, try:
```bash
generator.weight_sync_backend=nccl  # Force NCCL backend
```

### Expert Load Imbalance

If experts have imbalanced load, increase the auxiliary loss:
```bash
trainer.policy.nmoe_config.router_aux_loss_coef=0.01
```

## References

- [nmoe Repository](https://github.com/your-org/nmoe)
- [SkyRL Documentation](https://skyrl.readthedocs.io)
- [SGLang Documentation](https://sgl-project.github.io)
