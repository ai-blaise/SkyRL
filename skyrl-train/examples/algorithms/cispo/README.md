# CISPO (Clipped Importance Sampling Policy Optimization)

Policy optimization with clipped importance sampling weights for improved efficiency.

## Overview

CISPO clips the importance sampling ratio to prevent extreme updates, similar to PPO's clipping but applied differently. This allows for more aggressive policy updates while maintaining stability.

## Key Concept

Standard policy gradient can have high variance due to large importance ratios:
```
ratio = π_new(a|s) / π_old(a|s)
```

CISPO clips this ratio:
```
clipped_ratio = clip(ratio, 1 - eps_low, 1 + eps_high)
```

The asymmetric clipping (`eps_low` vs `eps_high`) allows fine-grained control over how much the policy can change in each direction.

## Quick Start

```bash
# 1. Prepare dataset
python examples/gsm8k/gsm8k_dataset.py

# 2. Run CISPO training
export WANDB_API_KEY=<your_key_here>
bash examples/algorithms/cispo/run_cispo_gsm8k.sh
```

## Configuration

```yaml
# CISPO-specific settings
trainer.algorithm.policy_loss_type: "cispo"
trainer.algorithm.cispo.cispo_eps_clip_low: 0    # Lower bound (0 = no lower clip)
trainer.algorithm.cispo.cispo_eps_clip_high: 5   # Upper bound
trainer.algorithm.use_kl_loss: false
```

## Clipping Parameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| `cispo_eps_clip_low` | 0 | How much ratio can decrease (0 = unlimited) |
| `cispo_eps_clip_high` | 5 | How much ratio can increase (5 = 6x max) |

### Tuning Guidelines

- **Conservative**: `eps_low=0.2, eps_high=0.2` (similar to PPO)
- **Aggressive exploration**: `eps_low=0, eps_high=5` (default)
- **Stable refinement**: `eps_low=0.1, eps_high=0.5`

## When to Use CISPO

| Scenario | Recommendation |
|----------|---------------|
| Exploration needed | CISPO with high eps_high |
| Risk of collapse | Lower eps_high |
| Fine-tuning | Conservative clipping |
| From scratch | Aggressive clipping |

## Comparison with PPO

| Feature | CISPO | PPO |
|---------|-------|-----|
| Clipping | Asymmetric | Symmetric |
| Critic | Not required | Required |
| Flexibility | More tunable | Standard |
| Sample efficiency | Medium | High |

## Related Documentation

- [PPO Example](../../ppo/README.md)
- [DAPO Example](../dapo/README.md) (uses dual-clip)
- [Algorithms Guide](../../../docs/ALGORITHMS.md)
