# Training Backend Configuration

Examples demonstrating different training backend options and their trade-offs.

## Overview

SkyRL supports multiple training backends:
- **FSDP** (v1): Original Fully Sharded Data Parallel
- **FSDP2** (v2): Improved FSDP with better memory efficiency (recommended)
- **Megatron**: For very large models with tensor/pipeline parallelism

This directory contains examples showing specific backend configurations.

## Available Scripts

| Script | Description |
|--------|-------------|
| `run_no_seq_pack.sh` | FSDP2 training without sequence packing |

## Sequence Packing

Sequence packing combines multiple shorter sequences into a single batch element to improve GPU utilization. However, some scenarios require it to be disabled:

```yaml
# Disable sequence packing
trainer.use_sample_packing: false

# Adjust batch sizes accordingly
trainer.micro_train_batch_size_per_gpu: 8
trainer.micro_forward_batch_size_per_gpu: 32
```

### When to Disable Sequence Packing

- **Variable-length sequences**: When sequences have very different lengths
- **Debugging**: Easier to debug without packing
- **Certain models**: Some models don't support packed attention masks
- **Memory constraints**: Unpacked batches can be more predictable

## FSDP vs FSDP2

| Feature | FSDP (v1) | FSDP2 (v2) |
|---------|-----------|------------|
| Memory efficiency | Good | Better |
| Gradient checkpointing | Supported | Improved |
| CPU offloading | Supported | Improved |
| Recommended for | Legacy | New projects |

### Configuration

```yaml
# Use FSDP2 (recommended)
trainer.strategy: fsdp2

# Or use original FSDP
trainer.strategy: fsdp
```

## Quick Start

```bash
# Prepare dataset
python examples/gsm8k/gsm8k_dataset.py

# Run without sequence packing
export WANDB_API_KEY=<your_key_here>
bash examples/training_backends/run_no_seq_pack.sh
```

## Memory Configuration

```yaml
# CPU offloading for memory savings
trainer.policy.fsdp_config.cpu_offload: false  # Policy stays on GPU
trainer.ref.fsdp_config.cpu_offload: true      # Reference model to CPU

# Gradient checkpointing
trainer.policy.gradient_checkpointing: true
```

## Related Documentation

- [Training Strategies Guide](../../docs/TRAINING_STRATEGIES.md)
- [Megatron Example](../megatron/README.md)
- [Batch Sizes Guide](../../docs/BATCH_SIZES.md)
