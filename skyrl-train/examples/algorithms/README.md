# RL Algorithms

SkyRL supports multiple reinforcement learning algorithms for LLM post-training. This directory contains examples for each algorithm.

## Quick Reference

| Algorithm | Directory | Use Case | Key Feature |
|-----------|-----------|----------|-------------|
| **GRPO** | `../gsm8k/` | General purpose | Baseline algorithm |
| **DAPO** | `dapo/` | Stable training | Dual-clip, overlong filtering |
| **RLOO** | `rloo/` | Variance reduction | Leave-one-out baseline |
| **REINFORCE++** | `reinforce++/` | Improved REINFORCE | K2 KL estimator |
| **CISPO** | `cispo/` | Efficient updates | Clipped importance sampling |
| **Dr. GRPO** | `drgrpo/` | Token-level training | Sequence-mean token-sum normalization |
| **GSPO** | `gspo/` | Sequence-level | Sequence-mean loss |
| **SAPO** | `sapo/` | Self-adaptive | Dynamic parameter adjustment |

## Algorithm Details

### GRPO (Group Relative Policy Optimization)
The default algorithm. Uses group-relative advantage estimation.

```yaml
trainer.algorithm.advantage_estimator: "grpo"
```

### DAPO (Decoupled Alignment from Preference Optimization)
Enhanced stability with dual-clip and dynamic sampling.

```yaml
trainer.algorithm.policy_loss_type: "dual_clip"
trainer.algorithm.eps_clip_low: 0.2
trainer.algorithm.eps_clip_high: 0.28
trainer.algorithm.dynamic_sampling.type: filter
```

**See:** [dapo/README.md](dapo/README.md)

### RLOO (REINFORCE Leave-One-Out)
Uses leave-one-out baseline for variance reduction.

```yaml
trainer.algorithm.advantage_estimator: "rloo"
trainer.algorithm.use_kl_in_reward: true
```

### REINFORCE++
Improved REINFORCE with K2 KL estimator.

```yaml
trainer.algorithm.advantage_estimator: "reinforce++"
trainer.algorithm.use_kl_in_reward: true
trainer.algorithm.kl_estimator_type: "k2"
```

### CISPO (Clipped Importance Sampling Policy Optimization)
Efficient policy updates with importance sampling clipping.

```yaml
trainer.algorithm.policy_loss_type: "cispo"
trainer.algorithm.cispo.cispo_eps_clip_low: 0
trainer.algorithm.cispo.cispo_eps_clip_high: 5
```

### Dr. GRPO
Token-level normalized GRPO.

```yaml
trainer.algorithm.advantage_estimator: "grpo"
trainer.algorithm.loss_reduction: "seq_mean_token_sum_norm"
trainer.algorithm.grpo_norm_by_std: false
```

### GSPO (Group Sequence Policy Optimization)
Sequence-level optimization.

```yaml
trainer.algorithm.policy_loss_type: "gspo"
trainer.algorithm.loss_reduction: "sequence_mean"
```

### SAPO (Self-Adaptive Policy Optimization)
Dynamic adjustment during training.

**See:** [sapo/README.md](sapo/README.md)

## Custom Algorithms

SkyRL supports custom algorithm implementations:

### Custom Advantage Estimator
```python
# See: custom_advantage_estimator/main_custom_adv_est.py
```

### Custom Policy Loss
```python
# See: custom_policy_loss/main_custom_policy_loss.py
```

## Choosing an Algorithm

| Scenario | Recommended Algorithm |
|----------|----------------------|
| Starting out | GRPO |
| Training instability | DAPO |
| High variance | RLOO |
| Long sequences | DAPO with overlong filtering |
| Token-level rewards | Dr. GRPO |
| Custom needs | Custom advantage/loss |

## Quick Start

All algorithms use GSM8K as the default dataset:

```bash
# 1. Prepare dataset
python examples/gsm8k/gsm8k_dataset.py

# 2. Set up logging
export WANDB_API_KEY=<your_key_here>

# 3. Run your chosen algorithm
bash examples/algorithms/<algorithm>/run_<algorithm>_gsm8k.sh
```

## Directory Contents

| Directory | Has README | Scripts |
|-----------|------------|---------|
| `cispo/` | No | `run_cispo_gsm8k.sh` |
| `clip_cov_kl_cov/` | Yes | `run_clip_cov.sh`, `run_kl_cov.sh` |
| `custom_advantage_estimator/` | No | `run_custom_adv_est.sh` |
| `custom_policy_loss/` | No | `run_custom_policy_loss.sh` |
| `dapo/` | Yes | Multiple AIME/GSM8K scripts |
| `drgrpo/` | No | `run_drgrpo_gsm8k.sh` |
| `gspo/` | No | `run_gspo_gsm8k.sh` |
| `reinforce++/` | No | `run_reinforce++.sh` |
| `rloo/` | No | `run_rloo.sh` |
| `sapo/` | Yes | `run_sapo_gsm8k.sh`, `run_sapo_qwen3_4b_aime.sh` |

## Related Documentation

- [Algorithms Guide](../../docs/ALGORITHMS.md)
- [DAPO Example](dapo/README.md)
- [SAPO Example](sapo/README.md)
- [TIS Correction](../tis_correction/README.md) - Works with any algorithm
