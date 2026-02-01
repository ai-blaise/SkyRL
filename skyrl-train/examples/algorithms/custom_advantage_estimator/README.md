# Custom Advantage Estimator

Create your own advantage estimation method for specialized RL training scenarios.

## Overview

SkyRL allows you to implement custom advantage estimators when the built-in options (GRPO, RLOO, REINFORCE++, GAE) don't fit your needs. This example shows how to create and register a custom estimator.

## Quick Start

```bash
# 1. Prepare dataset
python examples/gsm8k/gsm8k_dataset.py

# 2. Run with custom advantage estimator
export WANDB_API_KEY=<your_key_here>
bash examples/algorithms/custom_advantage_estimator/run_custom_adv_est.sh
```

## Implementation Guide

### Step 1: Create Custom Estimator

See `main_custom_adv_est.py` for the full implementation:

```python
from skyrl_train.algorithms.advantage_estimators import AdvantageEstimator

class MyCustomAdvantageEstimator(AdvantageEstimator):
    """Custom advantage estimator example."""

    def compute_advantages(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,  # Optional, may be None
        masks: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute advantages for policy gradient.

        Args:
            rewards: Shape [batch, seq_len] - reward at each token
            values: Shape [batch, seq_len] - value estimates (if using critic)
            masks: Shape [batch, seq_len] - 1 for valid tokens, 0 for padding

        Returns:
            advantages: Shape [batch, seq_len] - advantage estimates
        """
        # Your custom logic here
        # Example: simple reward-to-go
        advantages = torch.zeros_like(rewards)
        running_sum = torch.zeros(rewards.shape[0], device=rewards.device)

        for t in reversed(range(rewards.shape[1])):
            running_sum = rewards[:, t] + 0.99 * running_sum * masks[:, t]
            advantages[:, t] = running_sum

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages
```

### Step 2: Register and Use

```python
from skyrl_train.algorithms.advantage_estimators import register_advantage_estimator

# Register your estimator
register_advantage_estimator("my_custom", MyCustomAdvantageEstimator)

# Use in config
# trainer.algorithm.advantage_estimator: "my_custom"
```

### Step 3: Custom Entry Point

The example uses a custom entry point (`main_custom_adv_est.py`) that:
1. Imports and registers the custom estimator
2. Calls the standard training loop

```python
# main_custom_adv_est.py
from my_estimator import MyCustomAdvantageEstimator
from skyrl_train.algorithms.advantage_estimators import register_advantage_estimator
from skyrl_train.entrypoints.main_base import main

# Register before training starts
register_advantage_estimator("my_custom", MyCustomAdvantageEstimator)

if __name__ == "__main__":
    main()
```

## Built-in Estimators for Reference

| Estimator | Description | When to Use |
|-----------|-------------|-------------|
| `grpo` | Group relative policy optimization | Default, general purpose |
| `rloo` | Leave-one-out baseline | Low variance, no critic |
| `reinforce++` | Enhanced REINFORCE with K2 KL | Stable training |
| `gae` | Generalized Advantage Estimation | With critic (PPO) |

## Common Customizations

1. **Custom baseline**: Use different reference for advantage computation
2. **Reward shaping**: Transform rewards before advantage computation
3. **Credit assignment**: Custom temporal credit assignment
4. **Multi-objective**: Combine multiple reward signals

## Configuration

```yaml
# Use custom entry point
# python -m examples.algorithms.custom_advantage_estimator.main_custom_adv_est

trainer.algorithm.advantage_estimator: "my_custom"
# ... rest of config
```

## Related Documentation

- [Custom Policy Loss](../custom_policy_loss/README.md)
- [Algorithms Guide](../../../docs/ALGORITHMS.md)
- [Custom Environments](../../../docs/CUSTOM_ENVIRONMENTS.md)
