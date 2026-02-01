# PPO Example

Train using Proximal Policy Optimization with a learned value function (critic).

## Overview

This example demonstrates PPO with GAE (Generalized Advantage Estimation):
- Separate critic model for value estimation
- GAE for computing advantages
- PPO clipping for stable updates

## When to Use PPO vs GRPO

| Algorithm | Critic Required | Best For |
|-----------|-----------------|----------|
| **GRPO** | No | Simple rewards, faster setup |
| **PPO+GAE** | Yes | Complex rewards, dense feedback |

## Files

| File | Description |
|------|-------------|
| `run_ppo.sh` | Training script with PPO configuration |

## Quick Start

```bash
bash examples/ppo/run_ppo.sh
```

## Key Configuration

```yaml
trainer:
  algorithm:
    advantage_estimator: gae  # Use GAE (requires critic)
    gamma: 0.99               # Discount factor
    lambda_: 0.95             # GAE lambda

  # Critic model (learns value function)
  critic:
    model:
      path: "Qwen/Qwen2.5-0.5B-Instruct"
    learning_rate: 5e-6

  # PPO clipping
  algorithm:
    eps_clip_low: 0.2
    eps_clip_high: 0.2
```

## How It Works

1. **Generate**: Create responses using policy model
2. **Score**: Compute rewards using environment
3. **Value Estimate**: Critic predicts expected returns
4. **GAE**: Compute advantages using rewards and value estimates
5. **Update Critic**: Train critic to predict returns
6. **Update Policy**: PPO gradient update with clipping

## Memory Requirements

PPO requires more GPU memory than GRPO due to the critic model:
- GRPO: 1 model (policy)
- PPO: 2 models (policy + critic)

Consider using:
- `gradient_checkpointing: true`
- Smaller `micro_train_batch_size_per_gpu`
- CPU offloading for critic

## Documentation

- [Algorithms Guide](../../docs/ALGORITHMS.md) - Detailed GAE/PPO explanation
- [Batch Sizes Guide](../../docs/BATCH_SIZES.md) - Memory optimization
