# SAPO (Soft Adaptive Policy Optimization)

Train models using SAPO - a smooth alternative to PPO clipping that uses sigmoid gating functions.

**Paper:** [SAPO: Soft Adaptive Policy Optimization](https://arxiv.org/abs/2511.20347)

---

## Overview

SAPO replaces PPO's hard clipping with smooth gating functions:

```
gate(x, tau) = sigmoid(tau * (x - 1)) * (4/tau)
loss = -gate(ratio, tau) * advantage
```

**Benefits:**
- Smoother optimization landscape
- No sudden clipping transitions
- Separate temperature control for positive/negative advantages

---

## Quick Start

### Prerequisites

1. Prepare DAPO training data:
```bash
bash examples/algorithms/dapo/prepare_dapo_data.sh
```

2. Set up Weights & Biases (optional):
```bash
export WANDB_API_KEY=<your_key_here>
```

### Running SAPO

```bash
bash examples/algorithms/sapo/run_sapo_qwen3_4b_aime.sh
```

This trains Qwen3-4B-Base on DAPO math data and validates on AIME 2024.

---

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `tau_pos` | 1.0 | Temperature for positive advantages |
| `tau_neg` | 1.05 | Temperature for negative advantages |
| `loss_reduction` | `sequence_mean` | Recommended for SAPO |

**Higher tau values:**
- Sharper gating function
- Behavior closer to standard PPO clipping

**Lower tau values:**
- Smoother transitions
- More gradual policy updates

---

## Configuration

### Minimal SAPO Config

```yaml
trainer:
  algorithm:
    advantage_estimator: "grpo"
    policy_loss_type: "sapo"
    loss_reduction: "sequence_mean"
    sapo:
      tau_pos: 1.0
      tau_neg: 1.05
```

### Full Configuration Example

```yaml
trainer:
  algorithm:
    advantage_estimator: "grpo"
    policy_loss_type: "sapo"
    loss_reduction: "sequence_mean"  # Recommended for SAPO
    use_kl_loss: false
    sapo:
      tau_pos: 1.0   # Temperature for positive advantages
      tau_neg: 1.05  # Temperature for negative advantages

  policy:
    optimizer_config:
      lr: 1e-6
      weight_decay: 0.1
      max_grad_norm: 1.0
      num_warmup_steps: 80

generator:
  n_samples_per_prompt: 16
  sampling_params:
    temperature: 1.0
    top_p: 1.0
```

---

## When to Use SAPO

**Use SAPO when:**
- PPO training is unstable due to hard clipping transitions
- You want smoother gradient behavior
- Training on tasks with high variance advantages

**Compared to PPO:**
- Smoother loss landscape
- Less sensitivity to clip ratio selection
- May converge more stably on some tasks

**Compared to DAPO (dual_clip):**
- Different approach to stability
- SAPO uses smooth gating, DAPO uses dual clipping

---

## Directory Structure

| File | Purpose |
|------|---------|
| `run_sapo_qwen3_4b_aime.sh` | Launch script for Qwen3-4B training |
| Uses `main_dapo.py` | Shares entry point with DAPO example |

---

## Reference

- Paper: https://arxiv.org/abs/2511.20347
- See also: [ALGORITHMS.md](../../../docs/ALGORITHMS.md) for detailed algorithm documentation
