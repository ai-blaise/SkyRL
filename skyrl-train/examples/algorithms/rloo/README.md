# RLOO (REINFORCE Leave-One-Out)

RLOO uses a leave-one-out baseline for variance reduction in policy gradient estimation.

## Overview

Instead of using a learned value function (like PPO) or group mean (like GRPO), RLOO computes the baseline for each sample by averaging the rewards of all OTHER samples in the same batch. This provides a low-variance, unbiased baseline without requiring a separate critic network.

## Key Concept

For a batch of N samples from the same prompt:
```
baseline_i = (sum of all rewards except reward_i) / (N - 1)
advantage_i = reward_i - baseline_i
```

This "leave-one-out" approach ensures each sample has an unbiased baseline estimate.

## Quick Start

```bash
# 1. Prepare dataset
python examples/gsm8k/gsm8k_dataset.py

# 2. Run RLOO training
export WANDB_API_KEY=<your_key_here>
bash examples/algorithms/rloo/run_rloo.sh
```

## Configuration

```yaml
# RLOO-specific settings
trainer.algorithm.advantage_estimator: "rloo"
trainer.algorithm.use_kl_loss: false
trainer.algorithm.use_kl_in_reward: true  # Add KL penalty to reward

# Multiple samples required for leave-one-out
generator.n_samples_per_prompt: 5  # Minimum 2, recommended 4-8
```

## When to Use RLOO

| Scenario | RLOO vs Alternatives |
|----------|---------------------|
| Limited compute | RLOO (no critic needed) |
| High reward variance | RLOO (better variance reduction) |
| Need simplicity | RLOO (simpler than PPO) |
| Very large batches | GRPO may be comparable |

## Comparison with Other Algorithms

| Algorithm | Baseline | Pros | Cons |
|-----------|----------|------|------|
| RLOO | Leave-one-out mean | Unbiased, no critic | Requires multiple samples |
| GRPO | Group mean | Simple | Biased for small groups |
| PPO | Learned critic | Low variance | Requires critic training |

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `advantage_estimator` | - | Set to `"rloo"` |
| `use_kl_in_reward` | true | Add KL divergence to reward signal |
| `n_samples_per_prompt` | 5 | Samples per prompt (min 2 for RLOO) |

## Related Documentation

- [Algorithms Guide](../../../docs/ALGORITHMS.md)
- [GRPO Example](../../gsm8k/README.md)
- [PPO Example](../../ppo/README.md)
