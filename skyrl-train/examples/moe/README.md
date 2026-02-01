# MoE (Mixture of Experts) Training

Train large-scale Mixture-of-Experts models using SkyRL with expert parallelism.

> **Status:** Experimental - under active development. For tracking, see [training issue #203](https://github.com/NovaSky-AI/SkyRL/issues/203) and [inference issue #202](https://github.com/NovaSky-AI/SkyRL/issues/202).

---

## Overview

MoE (Mixture of Experts) models like Qwen1.5-MoE, Mixtral, and DeepSeek-MoE use sparse expert layers that activate only a subset of parameters per token. This enables training larger models with manageable compute costs.

### Key Concepts

| Concept | Description |
|---------|-------------|
| **Expert Parallelism (EP)** | Distributes experts across GPUs |
| **Tensor Parallelism (TP)** | Shards individual experts across GPUs |
| **Active Parameters** | Subset of experts activated per token (e.g., 2 of 8) |
| **Top-K Routing** | Selects K experts per token based on router scores |

---

## Prerequisites

### Hardware Requirements

- **Minimum:** 4× NVIDIA GPUs with 24GB+ VRAM each (A100, H100, L40S, RTX 4090)
- **Recommended:** 8× GPUs for larger MoE models
- **Network:** NVLink or high-bandwidth interconnect for multi-GPU

### Software Requirements

```bash
# Python 3.12 required
python --version  # Should be 3.12.x

# Install SkyRL with vLLM backend (MoE inference)
uv sync --extra vllm
source .venv/bin/activate
```

---

## Quick Start

### 1. Prepare Dataset

```bash
python examples/gsm8k/gsm8k_dataset.py
```

### 2. Run MoE Training

```bash
# Set logging (optional)
export WANDB_API_KEY=<your_key>  # or use LOGGER="console"

# Run training
bash examples/moe/run_qwen1_5_MoE_A2_7B.sh
```

---

## Configuration

### Key Parameters

The MoE example (`run_qwen1_5_MoE_A2_7B.sh`) configures:

```yaml
# Model
trainer.policy.model.path: "Qwen/Qwen1.5-MoE-A2.7B-Chat"

# Parallelism
generator.inference_engine_tensor_parallel_size: 4   # TP size
generator.inference_engine_expert_parallel_size: 4   # EP size
generator.inference_engine_data_parallel_size: 1     # DP size

# Training
trainer.strategy: fsdp2
trainer.placement.colocate_all: true
trainer.placement.policy_num_gpus_per_node: 4
```

### Parallelism Strategy

For MoE models, choose parallelism based on your hardware:

| Configuration | GPUs | Use Case |
|---------------|------|----------|
| TP=4, EP=1 | 4 | Small MoE, single node |
| TP=4, EP=4 | 4 | Medium MoE, experts on same GPUs |
| TP=2, EP=4 | 8 | Large MoE, separate expert groups |
| TP=8, EP=8 | 8 | Maximum parallelism |

**Rule of thumb:** `TP × EP ≤ num_gpus_per_node`

---

## Supported Models

| Model | Active Params | Total Params | Recommended GPUs |
|-------|---------------|--------------|------------------|
| Qwen1.5-MoE-A2.7B | 2.7B | 14B | 4× A100/H100 |
| Mixtral-8x7B | 12B | 46B | 8× A100/H100 |
| DeepSeek-MoE-16B | 2.4B | 16B | 4× A100/H100 |

---

## Memory Optimization

MoE models can be memory-intensive. Tips:

```bash
# Reduce memory usage
generator.gpu_memory_utilization=0.7 \
trainer.micro_train_batch_size_per_gpu=4 \
trainer.micro_forward_batch_size_per_gpu=4

# Use gradient checkpointing
trainer.policy.gradient_checkpointing=true
```

---

## Troubleshooting

### Out of Memory

```bash
# Reduce batch sizes
trainer.train_batch_size=512 \
trainer.policy_mini_batch_size=128 \
trainer.micro_train_batch_size_per_gpu=4
```

### Expert Parallelism Errors

```bash
# Ensure EP divides number of experts evenly
# For 8 experts: EP must be 1, 2, 4, or 8
generator.inference_engine_expert_parallel_size=4
```

### Slow Training

```bash
# Enable NCCL weight sync
generator.weight_sync_backend=nccl

# Use async generation
generator.async_engine=true
```

---

## Current Limitations

1. **Multi-node training:** Not fully supported yet (see [#203](https://github.com/NovaSky-AI/SkyRL/issues/203))
2. **SGLang backend:** Limited EP support; use vLLM for now
3. **Large MoE models:** May require custom sharding strategies

---

## Related Documentation

- [Training Strategies](../../docs/TRAINING_STRATEGIES.md)
- [Distributed Training](../../docs/SGLANG_INTEGRATION_GUIDE.md#6-parallelism-options)
- [FSDP Configuration](../training_backends/README.md)