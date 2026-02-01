# TIS (Trajectory Importance Sampling) Correction

Stabilize off-policy RL training with importance sampling correction.

## Overview

When using asynchronous RL or when there's a delay between trajectory collection and training, the policy used for generation differs from the current policy. **TIS correction** addresses this by reweighting gradients based on the importance ratio between old and current policies.

This is essential for:
- **Async RL**: When generation runs ahead of training
- **Large batch sizes**: When batches span multiple policy updates
- **Off-policy stability**: Preventing gradient explosion from stale trajectories

## Prerequisites

- 4+ GPUs
- GSM8K dataset prepared
- Understanding of importance sampling concepts

## Quick Start

```bash
# 1. Prepare dataset
python examples/gsm8k/gsm8k_dataset.py

# 2. Run TIS-corrected DAPO
export WANDB_API_KEY=<your_key_here>
bash examples/tis_correction/run_dapo_tis.sh
```

## Key Configuration

```yaml
# Enable TIS correction
trainer.algorithm.use_tis: true
trainer.algorithm.tis_imp_ratio_cap: 2.0  # Cap importance ratios

# Required: return rollout logprobs for importance ratio computation
generator.sampling_params.logprobs: 0
```

## Available Scripts

| Script | Description |
|--------|-------------|
| `run_dapo_tis.sh` | DAPO with TIS correction on GSM8K |
| `main_tis_dapo.py` | Custom entrypoint with TIS implementation |

## How TIS Works

1. **During generation**: Store log probabilities of generated tokens under the rollout policy
2. **During training**: Compute current policy log probabilities for same tokens
3. **Importance ratio**: `r = exp(log_prob_current - log_prob_rollout)`
4. **Gradient reweighting**: Scale gradients by capped importance ratio

```
Importance Ratio = P_current(action) / P_rollout(action)

Capped Ratio = min(ratio, tis_imp_ratio_cap)
```

## When to Use TIS

| Scenario | Use TIS? |
|----------|----------|
| Synchronous RL | Optional |
| Async RL (1-step off) | Recommended |
| Fully async RL | Required |
| Large batch delays | Recommended |

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `use_tis` | false | Enable TIS correction |
| `tis_imp_ratio_cap` | 2.0 | Maximum importance ratio (prevents instability) |

## Combining with DAPO

This example uses DAPO (Decoupled Alignment from Preference Optimization) with TIS:

```yaml
# DAPO settings
trainer.algorithm.policy_loss_type: "dual_clip"
trainer.algorithm.eps_clip_low: 0.2
trainer.algorithm.eps_clip_high: 0.28
trainer.algorithm.dynamic_sampling.type: filter
trainer.algorithm.loss_reduction: "token_mean"
```

## Related Documentation

- [DAPO Algorithm](../algorithms/dapo/README.md)
- [Algorithms Guide](../../docs/ALGORITHMS.md)
- [Async RL Documentation](https://skyrl.readthedocs.io/en/latest/tutorials/fully_async.html)
