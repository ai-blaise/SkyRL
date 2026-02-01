# RL Algorithms in SkyRL

**Complete guide to reinforcement learning algorithms available in SkyRL.**

---

## Overview

SkyRL supports multiple policy optimization algorithms through a modular design:

1. **Advantage Estimators**: Compute advantage signals from rewards
2. **Policy Loss Functions**: Optimize the policy based on advantages

Both components are customizable via registries, allowing you to extend SkyRL with your own algorithms.

---

## Quick Reference

| Algorithm | Advantage Estimator | Policy Loss | Critic Required | Best For |
|-----------|---------------------|-------------|-----------------|----------|
| **GRPO** | `grpo` | `regular` | No | General RL fine-tuning |
| **RLOO** | `rloo` | `regular` | No | Sample-efficient training |
| **PPO** | `gae` | `regular` | Yes | Value function learning |
| **REINFORCE++** | `reinforce++` | `regular` | No | Outcome-based RL |
| **GSPO** | `grpo` | `gspo` | No | Sequence-level optimization |
| **DAPO** | `grpo` | `dual_clip` | No | Stability with dual clipping |
| **SAPO** | `grpo` | `sapo` | No | Smooth adaptive updates |
| **CISPO** | `grpo` | `cispo` | No | Clipped IS-weight optimization |

---

## 1. Advantage Estimators

### 1.1 GRPO (Group Relative Policy Optimization)

**Paper**: [DeepSeekMath: Pushing the Limits of Mathematical Reasoning](https://arxiv.org/abs/2402.03300)

GRPO computes advantages by comparing responses within the same prompt group:

```
advantage = (reward - mean(group_rewards)) / std(group_rewards)
```

**Configuration:**
```yaml
trainer:
  algorithm:
    advantage_estimator: "grpo"
    grpo_norm_by_std: true  # Normalize by standard deviation
```

**When to use:**
- Default choice for most RL fine-tuning tasks
- Works well with outcome-based rewards (single scalar per response)
- No critic model required

---

### 1.2 RLOO (REINFORCE Leave-One-Out)

**Paper**: [Back to Basics: Revisiting REINFORCE Style Optimization](https://arxiv.org/abs/2402.14740)

RLOO uses leave-one-out baseline estimation:

```
baseline = mean(other_responses_in_group)
advantage = reward - baseline
```

**Configuration:**
```yaml
trainer:
  algorithm:
    advantage_estimator: "rloo"
```

**When to use:**
- More sample-efficient than GRPO
- Requires multiple samples per prompt (`n_samples_per_prompt > 1`)
- Good for small batch sizes

---

### 1.3 GAE (Generalized Advantage Estimation)

**Paper**: [High-Dimensional Continuous Control Using GAE](https://arxiv.org/abs/1506.02438)

GAE combines temporal difference errors with discounting:

```
A_t = delta_t + (gamma * lambda) * delta_{t+1} + ...
delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
```

**Configuration:**
```yaml
trainer:
  algorithm:
    advantage_estimator: "gae"
    gamma: 0.99  # Discount factor
    lambd: 0.95  # GAE lambda parameter
  critic:
    model:
      path: "Qwen/Qwen2.5-1.5B-Instruct"  # Requires critic model
```

**When to use:**
- When using a critic/value function
- For token-level credit assignment
- Traditional PPO setup

---

### 1.4 REINFORCE++

**Paper**: [REINFORCE++: A Simple and Efficient Approach for Aligning Large Language Models](https://arxiv.org/abs/2501.03262)

REINFORCE++ computes discounted returns with whitening:

```
returns[t] = reward[t] + gamma * returns[t+1]
advantages = whiten(returns)
```

**Configuration:**
```yaml
trainer:
  algorithm:
    advantage_estimator: "reinforce++"
    gamma: 1.0  # Discount factor (1.0 for no discounting)
```

**When to use:**
- Simple baseline without critic
- Token-level rewards with temporal structure

---

## 2. Policy Loss Functions

### 2.1 Regular PPO Loss

Standard PPO clipped surrogate objective:

```
ratio = exp(log_prob - old_log_prob)
surr1 = ratio * advantage
surr2 = clip(ratio, 1-eps, 1+eps) * advantage
loss = -min(surr1, surr2)
```

**Configuration:**
```yaml
trainer:
  algorithm:
    policy_loss_type: "regular"
    eps_clip_low: 0.2
    eps_clip_high: 0.2
```

---

### 2.2 GSPO (Group Sequence Policy Optimization)

**Paper**: [GSPO: Group Sequence Policy Optimization](https://arxiv.org/abs/2507.18071)

GSPO uses sequence-level importance sampling instead of token-level:

```
log_importance = mean(log_ratio) across sequence
ratio = exp(log_importance)  # Applied uniformly to all tokens
```

**Configuration:**
```yaml
trainer:
  algorithm:
    policy_loss_type: "gspo"
    loss_reduction: "sequence_mean"  # Recommended for GSPO
    eps_clip_low: 0.2
    eps_clip_high: 0.2
```

**When to use:**
- More stable training with sequence-level clipping
- Reduces variance in clipping behavior within sequences

---

### 2.3 Dual Clip (DAPO)

**Paper**: [DAPO: An Open-Source LLM Reinforcement Learning System at Scale](https://arxiv.org/abs/2503.14476)

Dual clipping adds a lower bound on negative advantages:

```
if advantage < 0:
    loss = max(loss, -advantage * clip_ratio_c)
```

**Configuration:**
```yaml
trainer:
  algorithm:
    policy_loss_type: "dual_clip"
    eps_clip_low: 0.2
    eps_clip_high: 0.2
    clip_ratio_c: 3.0  # Dual clip ratio
```

**When to use:**
- When training is unstable with negative advantages
- Prevents excessive punishment of bad responses

---

### 2.4 SAPO (Soft Adaptive Policy Optimization)

**Paper**: [SAPO: Soft Adaptive Policy Optimization](https://arxiv.org/abs/2511.20347)

SAPO uses smooth gating functions instead of hard clipping:

```
gate(x, tau) = sigmoid(tau * (x - 1)) * (4/tau)
loss = -gate(ratio, tau) * advantage
```

**Configuration:**
```yaml
trainer:
  algorithm:
    policy_loss_type: "sapo"
    loss_reduction: "sequence_mean"  # Recommended for SAPO
    sapo:
      tau_pos: 1.0   # Temperature for positive advantages
      tau_neg: 1.05  # Temperature for negative advantages
```

**When to use:**
- Smoother optimization landscape
- Avoids sudden clipping transitions

---

### 2.5 CISPO (Clipped IS-weight Policy Optimization)

**Paper**: [CISPO: Clipped IS-weight Policy Optimization](https://arxiv.org/abs/2506.13585)

CISPO clips the importance ratio in the gradient rather than the loss:

```
clamped_ratio = clamp(ratio, 1-eps_low, 1+eps_high)
loss = -advantage * clamped_ratio.detach() * log_prob
```

**Configuration:**
```yaml
trainer:
  algorithm:
    policy_loss_type: "cispo"
    cispo:
      cispo_eps_clip_low: 0    # Lower bound offset
      cispo_eps_clip_high: 5   # Upper bound offset
```

**When to use:**
- Model can still learn from clipped samples (non-zero gradients)
- More exploration compared to PPO

---

### 2.6 Clip-Cov (Covariance-based Clipping)

Uses covariance between advantages and log-probs to select tokens for clipping:

**Configuration:**
```yaml
trainer:
  algorithm:
    policy_loss_type: "clip_cov"
    clip_cov:
      clip_ratio: 0.0002   # Fraction of tokens to clip
      clip_cov_lb: 1.0     # Lower bound for covariance
      clip_cov_ub: 5.0     # Upper bound for covariance
```

---

### 2.7 KL-Cov (Covariance-based KL Regularization)

Applies KL regularization to high-covariance tokens:

**Configuration:**
```yaml
trainer:
  algorithm:
    policy_loss_type: "kl_cov"
    kl_cov:
      kl_cov_frac: 0.2     # Fraction of tokens to regularize (20%)
      ppo_kl_coef: 1.0     # KL coefficient
```

---

## 3. KL Divergence Control

### 3.1 KL Loss

Apply KL divergence as an additional loss term:

```yaml
trainer:
  algorithm:
    use_kl_loss: true
    kl_loss_coef: 0.001
    kl_estimator_type: "k3"  # k1, k2, k3, abs
```

**KL Estimator Types:**
| Type | Formula | Notes |
|------|---------|-------|
| `k1` | `log_prob - log_prob_ref` | Simple difference |
| `k2` | `0.5 * (log_prob - log_prob_ref)^2` | Squared difference |
| `k3` | `ratio - log(ratio) - 1` | Schulman's approximation |
| `abs` | `\|log_prob - log_prob_ref\|` | Absolute difference |

### 3.2 KL in Reward

Alternatively, add KL penalty to rewards:

```yaml
trainer:
  algorithm:
    use_kl_in_reward: true
    use_kl_loss: false
    kl_loss_coef: 0.01
```

### 3.3 Adaptive KL Controller

Dynamically adjust KL coefficient:

```yaml
trainer:
  algorithm:
    kl_ctrl:
      type: "adaptive"  # or "fixed"
      kl_target: 0.1    # Target KL divergence
      horizon: 10000    # Update rate
```

---

## 4. Entropy Regularization

Encourage exploration with entropy loss:

```yaml
trainer:
  algorithm:
    use_entropy_loss: true
    entropy_loss_coef: 0.01
```

---

## 5. Loss Reduction Strategies

| Reduction | Formula | Best For |
|-----------|---------|----------|
| `token_mean` | `mean(loss * mask) / sum(mask)` | Default, token-level |
| `sequence_mean` | `mean(mean(loss, dim=-1))` | GSPO, SAPO |
| `seq_mean_token_sum_norm` | `mean(sum(loss * mask, dim=-1) / max_len)` | Dr. GRPO, length-bias free |

**Configuration:**
```yaml
trainer:
  algorithm:
    loss_reduction: "token_mean"  # or "sequence_mean", "seq_mean_token_sum_norm"
```

---

## 6. Dynamic Sampling

Filter or replace low-quality samples during training:

### 6.1 Filter Strategy

Remove prompts with zero reward variance:

```yaml
trainer:
  algorithm:
    dynamic_sampling:
      type: "filter"
      max_sample_batches: 30
```

### 6.2 Replace Strategy

Replace bad samples with new generations:

```yaml
trainer:
  algorithm:
    dynamic_sampling:
      type: "replace"
      max_sample_batches: 30
      min_replace_ratio: 0.3  # Minimum good samples to replace
```

### 6.3 Zero Variance Filter

Mask out prompts where all responses have the same reward:

```yaml
trainer:
  algorithm:
    zero_variance_filter: true
```

---

## 7. Truncated Importance Sampling

For off-policy training, cap importance ratios:

```yaml
trainer:
  algorithm:
    use_tis: true
    tis_imp_ratio_cap: 5.0
```

---

## 8. Algorithm Configurations

### 8.1 GRPO (Recommended Default)

```yaml
trainer:
  algorithm:
    advantage_estimator: "grpo"
    policy_loss_type: "regular"
    grpo_norm_by_std: true
    loss_reduction: "token_mean"
    use_kl_loss: true
    kl_loss_coef: 0.001
    eps_clip_low: 0.2
    eps_clip_high: 0.2
generator:
  n_samples_per_prompt: 5
```

### 8.2 RLOO

```yaml
trainer:
  algorithm:
    advantage_estimator: "rloo"
    policy_loss_type: "regular"
    loss_reduction: "token_mean"
    use_kl_loss: true
    kl_loss_coef: 0.001
    eps_clip_low: 0.2
    eps_clip_high: 0.2
generator:
  n_samples_per_prompt: 8  # More samples help RLOO
```

### 8.3 DAPO

```yaml
trainer:
  algorithm:
    advantage_estimator: "grpo"
    policy_loss_type: "dual_clip"
    loss_reduction: "token_mean"
    use_kl_loss: false  # DAPO typically doesn't use KL loss
    eps_clip_low: 0.2
    eps_clip_high: 0.2
    clip_ratio_c: 3.0
generator:
  apply_overlong_filtering: true  # DAPO overlong filtering
```

### 8.4 PPO with GAE

```yaml
trainer:
  algorithm:
    advantage_estimator: "gae"
    policy_loss_type: "regular"
    loss_reduction: "token_mean"
    gamma: 0.99
    lambd: 0.95
    eps_clip_low: 0.2
    eps_clip_high: 0.2
  critic:
    model:
      path: "Qwen/Qwen2.5-1.5B-Instruct"
```

### 8.5 GSPO

```yaml
trainer:
  algorithm:
    advantage_estimator: "grpo"
    policy_loss_type: "gspo"
    loss_reduction: "sequence_mean"
    eps_clip_low: 0.2
    eps_clip_high: 0.2
```

---

## 9. Custom Algorithms

### 9.1 Custom Advantage Estimator

```python
from skyrl_train.utils.ppo_utils import (
    register_advantage_estimator,
    AdvantageEstimatorRegistry
)

@register_advantage_estimator("my_advantage")
def my_advantage_estimator(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    **kwargs
) -> tuple[torch.Tensor, torch.Tensor]:
    """Custom advantage computation."""
    # Your logic here
    advantages = ...
    returns = ...
    return advantages, returns
```

**Usage:**
```yaml
trainer:
  algorithm:
    advantage_estimator: "my_advantage"
```

### 9.2 Custom Policy Loss

```python
from skyrl_train.utils.ppo_utils import (
    register_policy_loss,
    PolicyLossRegistry
)

@register_policy_loss("my_loss")
def my_policy_loss(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    config: DictConfig,
    loss_mask: Optional[torch.Tensor] = None,
    rollout_logprobs: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, float]:
    """Custom policy loss computation."""
    # Your logic here
    loss = ...
    clip_ratio = ...
    return loss, clip_ratio
```

**Usage:**
```yaml
trainer:
  algorithm:
    policy_loss_type: "my_loss"
```

### 9.3 Registration in Entry Point

```python
import ray
from skyrl_train.utils.ppo_utils import sync_registries

@ray.remote
def skyrl_entrypoint(cfg):
    # Register custom algorithms BEFORE sync
    from my_module import my_advantage_estimator, my_policy_loss

    # Sync registries to Ray actor
    sync_registries()

    # Run training
    from skyrl_train.entrypoints import BasePPOExp
    exp = BasePPOExp(cfg)
    exp.run()
```

---

## 10. Algorithm Selection Guide

```
Is your task...

Math/Reasoning problems?
├── Yes → Use GRPO (default)
│         └── Unstable? → Try DAPO (dual_clip)
│
Code generation?
├── Yes → Use GRPO with zero_variance_filter: true
│
Multi-turn conversations?
├── Yes → Use GRPO with sequence_mean reduction
│
Have a trained value function?
├── Yes → Use GAE + PPO
│
Training unstable?
├── Yes → Try:
│         ├── Lower learning rate
│         ├── DAPO (dual_clip)
│         ├── SAPO (smooth gating)
│         └── Stronger KL penalty
│
Want more exploration?
├── Yes → Try CISPO or increase entropy_loss_coef
│
Sample-limited?
└── Yes → Use RLOO with more n_samples_per_prompt
```

---

## References

- [GRPO - DeepSeekMath](https://arxiv.org/abs/2402.03300)
- [RLOO - Back to Basics](https://arxiv.org/abs/2402.14740)
- [GAE - High-Dimensional Control](https://arxiv.org/abs/1506.02438)
- [PPO - Proximal Policy Optimization](https://arxiv.org/abs/1707.06347)
- [REINFORCE++ - Simple RL Alignment](https://arxiv.org/abs/2501.03262)
- [GSPO - Group Sequence Policy Optimization](https://arxiv.org/abs/2507.18071)
- [DAPO - Open-Source LLM RL](https://arxiv.org/abs/2503.14476)
- [SAPO - Soft Adaptive Policy](https://arxiv.org/abs/2511.20347)
- [CISPO - Clipped IS-weight](https://arxiv.org/abs/2506.13585)
- [SkyRL Examples](../examples/algorithms/)
