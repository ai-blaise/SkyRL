# REINFORCE++

Enhanced REINFORCE algorithm with improved KL estimation using the K2 estimator.

## Overview

REINFORCE++ improves upon standard REINFORCE by using a more accurate KL divergence estimator (K2) that provides better gradient estimates, especially when the policy has drifted significantly from the reference.

## Key Concept

Standard KL estimation can be noisy. REINFORCE++ uses the K2 estimator:

```
K2_KL = 0.5 * (ratio - 1)^2
where ratio = exp(log_prob_policy - log_prob_ref)
```

This provides a smoother, more stable gradient signal compared to the standard KL estimator.

## Quick Start

```bash
# 1. Prepare dataset
python examples/gsm8k/gsm8k_dataset.py

# 2. Run REINFORCE++ training
export WANDB_API_KEY=<your_key_here>
bash examples/algorithms/reinforce++/run_reinforce++.sh
```

## Configuration

```yaml
# REINFORCE++ specific settings
trainer.algorithm.advantage_estimator: "reinforce++"
trainer.algorithm.use_kl_loss: false
trainer.algorithm.use_kl_in_reward: true
trainer.algorithm.kl_estimator_type: "k2"  # Key difference
```

## KL Estimator Types

| Type | Formula | When to Use |
|------|---------|-------------|
| `k1` | `ratio * log(ratio)` | Standard, can be unstable |
| `k2` | `0.5 * (ratio - 1)^2` | More stable, REINFORCE++ default |
| `k3` | `(ratio - 1) - log(ratio)` | Alternative, less common |
| `abs` | `abs(log(ratio))` | Simple, robust |

## When to Use REINFORCE++

- **Policy drift**: When policy may drift far from reference
- **Stability**: When training is unstable with standard KL
- **Long training**: For extended training runs

## Comparison with RLOO

| Feature | REINFORCE++ | RLOO |
|---------|------------|------|
| Baseline | None (reward whitening) | Leave-one-out |
| KL handling | K2 estimator | Standard |
| Stability | High | Medium |
| Sample efficiency | Lower | Higher |

## Related Documentation

- [RLOO Example](../rloo/README.md)
- [Algorithms Guide](../../../docs/ALGORITHMS.md)
