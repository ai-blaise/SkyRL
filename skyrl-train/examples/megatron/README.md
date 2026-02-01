# Megatron Backend Examples

Large-scale model training using the Megatron backend with 5D parallelism support.

## When to Use Megatron

Use Megatron backend when:
- Training models 7B+ parameters
- Using multi-node setups
- Needing tensor, pipeline, data, sequence, and expert parallelism
- Memory-constrained with large models

Use FSDP/FSDP2 instead when:
- Single-node training
- Models <7B
- Simpler setup preferred

## Prerequisites

- Multi-GPU setup (typically 8+ GPUs)
- Megatron-LM installed
- NCCL properly configured

## Available Scripts

| Script | Model | Description |
|--------|-------|-------------|
| `run_megatron.sh` | Default | Basic Megatron training |
| `run_megatron_lora_*.sh` | Various | LoRA fine-tuning with Megatron |
| `run_megatron_dapo_*.sh` | Qwen2.5 | DAPO algorithm with Megatron |
| `run_megatron_qwen3-*.sh` | Qwen3 | Qwen3 model training |
| `run_fsdp_baseline.sh` | - | FSDP comparison baseline |
| `run_search_megatron.sh` | - | Search agent with Megatron |

## Quick Start

```bash
# Basic Megatron training
bash examples/megatron/run_megatron.sh

# With LoRA
bash examples/megatron/run_megatron_lora_qwen2.5_0.5b.sh

# DAPO algorithm
bash examples/megatron/run_megatron_dapo_qwen2.5_0.5b.sh
```

## Configuration

Key settings for Megatron backend:

```yaml
trainer:
  strategy: megatron  # Enable Megatron backend

  placement:
    policy_num_nodes: 2
    policy_num_gpus_per_node: 8
```

## Related Documentation

- [Training Strategies Guide](../../docs/TRAINING_STRATEGIES.md)
- [Backend Selection Guide](../../docs/BACKEND_SELECTION.md)
