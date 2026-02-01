# Multiply Example

A minimal "Hello World" example for SkyRL - train a model to multiply two numbers.

## Overview

This is the simplest possible SkyRL example:
- **Task**: Multiply two single-digit numbers
- **Reward**: 1.0 if correct, 0.0 if wrong
- **Purpose**: Verify your setup works before tackling complex tasks

## Files

| File | Description |
|------|-------------|
| `env.py` | Custom environment that scores multiplication answers |
| `multiply_dataset.py` | Generates training data (multiplication problems) |
| `main_multiply.py` | Training entry point |
| `run_multiply.sh` | Shell script to run training |

## Quick Start

### 1. Generate Dataset

```bash
python examples/multiply/multiply_dataset.py
```

Creates problems like:
- Prompt: "What is 3 times 7?"
- Expected: "21"

### 2. Run Training

```bash
bash examples/multiply/run_multiply.sh
```

Or directly:

```bash
python -m skyrl_train.entrypoints.main_base \
  --config-path examples/multiply \
  --config-name main_multiply
```

### 3. Expected Results

The model should quickly learn to multiply single-digit numbers:
- Initial accuracy: ~10-20%
- After training: ~90%+

## Custom Environment

The `env.py` shows how to create a simple reward function:

```python
class MultiplyEnv(BaseEnv):
    def compute_rewards(self, prompts, responses, reward_specs):
        rewards = []
        for response, spec in zip(responses, reward_specs):
            expected = spec["ground_truth"]
            # Check if response contains the correct answer
            if expected in response:
                rewards.append(1.0)
            else:
                rewards.append(0.0)
        return rewards
```

## Use This As Template

This example is ideal for:
1. **Verifying installation** - Quick sanity check
2. **Learning SkyRL** - Minimal code to understand
3. **Starting custom tasks** - Copy and modify for your use case

## Documentation

- [Custom Environments Guide](../../docs/CUSTOM_ENVIRONMENTS.md) - Create your own
- [Data Format](../../docs/DATA_FORMAT.md) - Dataset structure
