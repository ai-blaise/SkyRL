# Custom Policy Loss

Implement custom policy loss functions for specialized training objectives.

## Overview

When the built-in policy losses (standard, dual_clip, CISPO, GSPO) don't meet your needs, you can implement a custom policy loss function. This is useful for research or specialized training scenarios.

## Quick Start

```bash
# 1. Prepare dataset
python examples/gsm8k/gsm8k_dataset.py

# 2. Run with custom policy loss
export WANDB_API_KEY=<your_key_here>
bash examples/algorithms/custom_policy_loss/run_custom_policy_loss.sh
```

## Implementation Guide

### Step 1: Create Custom Loss Function

See `main_custom_policy_loss.py` for the full implementation:

```python
from skyrl_train.algorithms.policy_losses import PolicyLoss, register_policy_loss

class MyCustomPolicyLoss(PolicyLoss):
    """Custom policy loss example."""

    def __init__(self, config):
        super().__init__(config)
        # Custom parameters from config
        self.my_param = config.get("my_param", 1.0)

    def compute_loss(
        self,
        log_probs: torch.Tensor,      # Current policy log probs
        old_log_probs: torch.Tensor,  # Reference log probs
        advantages: torch.Tensor,      # Computed advantages
        masks: torch.Tensor,           # Valid token mask
        **kwargs
    ) -> torch.Tensor:
        """
        Compute policy gradient loss.

        Args:
            log_probs: [batch, seq_len] - log π(a|s) under current policy
            old_log_probs: [batch, seq_len] - log π(a|s) under old policy
            advantages: [batch, seq_len] - advantage estimates
            masks: [batch, seq_len] - 1 for valid, 0 for padding

        Returns:
            loss: Scalar tensor
        """
        # Compute importance ratio
        ratio = torch.exp(log_probs - old_log_probs)

        # Your custom loss logic
        # Example: simple policy gradient with ratio clipping
        clipped_ratio = torch.clamp(ratio, 0.8, 1.2)

        loss1 = -advantages * ratio
        loss2 = -advantages * clipped_ratio

        # Take max (pessimistic bound)
        loss = torch.max(loss1, loss2)

        # Apply mask and reduce
        loss = (loss * masks).sum() / masks.sum()

        return loss
```

### Step 2: Register and Use

```python
from skyrl_train.algorithms.policy_losses import register_policy_loss

# Register your loss
register_policy_loss("my_custom_loss", MyCustomPolicyLoss)

# Use in config
# trainer.algorithm.policy_loss_type: "my_custom_loss"
```

### Step 3: Custom Entry Point

```python
# main_custom_policy_loss.py
from my_loss import MyCustomPolicyLoss
from skyrl_train.algorithms.policy_losses import register_policy_loss
from skyrl_train.entrypoints.main_base import main

# Register before training starts
register_policy_loss("my_custom_loss", MyCustomPolicyLoss)

if __name__ == "__main__":
    main()
```

## Built-in Policy Losses for Reference

| Loss Type | Description | Key Feature |
|-----------|-------------|-------------|
| `standard` | Basic policy gradient | Simple, no clipping |
| `dual_clip` | DAPO-style dual clipping | Asymmetric bounds |
| `cispo` | Importance sampling clipping | Aggressive exploration |
| `gspo` | Sequence-level optimization | Equal sequence weight |

## Common Customizations

1. **Custom clipping**: Different clipping strategies
2. **Regularization**: Add entropy bonus, KL penalties
3. **Multi-objective**: Combine multiple loss terms
4. **Curriculum**: Loss that changes during training

## Configuration

```yaml
# Use custom entry point
# python -m examples.algorithms.custom_policy_loss.main_custom_policy_loss

trainer.algorithm.policy_loss_type: "my_custom_loss"
# Custom parameters accessible via config
+trainer.algorithm.my_custom_loss.my_param: 1.5
```

## Example: Entropy-Regularized Loss

```python
class EntropyRegularizedLoss(PolicyLoss):
    def __init__(self, config):
        super().__init__(config)
        self.entropy_coef = config.get("entropy_coef", 0.01)

    def compute_loss(self, log_probs, old_log_probs, advantages, masks, **kwargs):
        # Standard policy gradient
        ratio = torch.exp(log_probs - old_log_probs)
        pg_loss = -(advantages * ratio * masks).sum() / masks.sum()

        # Entropy bonus (encourage exploration)
        entropy = -(log_probs.exp() * log_probs * masks).sum() / masks.sum()

        # Combined loss
        return pg_loss - self.entropy_coef * entropy
```

## Related Documentation

- [Custom Advantage Estimator](../custom_advantage_estimator/README.md)
- [DAPO (dual_clip)](../dapo/README.md)
- [CISPO](../cispo/README.md)
- [Algorithms Guide](../../../docs/ALGORITHMS.md)
