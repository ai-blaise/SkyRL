# Step-Wise Multi-Turn Training

Train models with step-wise trajectory generation for fine-grained, token-level rewards in multi-turn scenarios.

## Overview

Step-wise training breaks down multi-turn conversations into individual steps, enabling:
- **Token-level rewards**: Assign rewards at each turn rather than only at the end
- **Better credit assignment**: Model learns which specific responses led to success/failure
- **Iterative refinement**: Model can learn from intermediate feedback

This is particularly useful for **Text-to-SQL** and other tool-use tasks where intermediate steps matter.

## Prerequisites

- 8 GPUs (H100 recommended)
- SkyRL-SQL dataset (~50GB with databases)
- ~29GB context length support per sequence

## Quick Start

### 1. Download Dataset

```bash
# Download SkyRL-SQL dataset
huggingface-cli download NovaSky-AI/SkyRL-SQL-653-data-newfmt \
  --local-dir ~/data/sql --repo-type dataset

# Download OmniSQL database files (~22GB)
huggingface-cli download seeklhy/OmniSQL-datasets data.zip \
  --repo-type dataset --local-dir ~/data/sql/db_files/

# Extract databases
unzip ~/data/sql/db_files/data.zip -d ~/data/sql/db_files/
```

### 2. Run Training

```bash
export WANDB_API_KEY=<your_key_here>

# Qwen2.5-Coder-7B
bash examples/step_wise/run_skyrl_sql_step_wise.sh

# Qwen3 variant
bash examples/step_wise/run_skyrl_sql_step_wise_qwen3.sh
```

## Key Configuration

```yaml
# Enable step-wise training
generator.step_wise_trajectories: true

# Multi-turn settings
generator.use_conversation_multi_turn: true
generator.max_turns: 6

# Long context support
generator.max_input_length: 29000
generator.sampling_params.max_generate_length: 3000

# DAPO dual-clip for stability
trainer.algorithm.policy_loss_type: "dual_clip"
```

## Available Scripts

| Script | Model | Description |
|--------|-------|-------------|
| `run_skyrl_sql_step_wise.sh` | Qwen2.5-Coder-7B | Standard step-wise SQL training |
| `run_skyrl_sql_step_wise_qwen3.sh` | Qwen3 | Step-wise with Qwen3 model |

## How Step-Wise Training Works

1. **Initial prompt**: Natural language question + database schema
2. **Step 1**: Model generates SQL query
3. **Execution**: Query runs against actual database
4. **Feedback**: Results or error message returned
5. **Step 2-N**: Model refines based on feedback
6. **Rewards**: Each step receives a reward signal

Unlike end-to-end training where only the final answer is rewarded, step-wise training provides learning signal at each turn.

## Comparison with Standard Multi-Turn

| Feature | Standard Multi-Turn | Step-Wise |
|---------|---------------------|-----------|
| Reward timing | End of trajectory | Each step |
| Credit assignment | Coarse | Fine-grained |
| Training efficiency | Lower | Higher |
| Implementation | Simpler | More complex |

## Related Documentation

- [Multi-Turn Training Guide](../../docs/MULTI_TURN_TRAINING.md)
- [Text-to-SQL Example](../text_to_sql/README.md)
- [Custom Environments](../../docs/CUSTOM_ENVIRONMENTS.md)
