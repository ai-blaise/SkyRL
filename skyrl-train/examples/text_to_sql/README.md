# Text-to-SQL Training

Train models to convert natural language questions into SQL queries using multi-turn RL with tool feedback.

## Overview

This example demonstrates **multi-turn agent training** where the model:
1. Receives a natural language question
2. Generates a SQL query
3. Executes the query against a database
4. Gets feedback (results or error)
5. Iteratively refines the query

This is a key example of **tool-use RL training** with SGLang.

## Prerequisites

- SkyRL installed with SGLang: `uv sync --extra sglang`
- ~50GB disk space for databases
- 4+ GPUs recommended

## Quick Start

### 1. Download Training Data

```bash
# Download SkyRL-SQL dataset
huggingface-cli download NovaSky-AI/SkyRL-SQL-653-data-newfmt \
  --local-dir ~/data/sql --repo-type dataset
```

### 2. Download Database Files

```bash
# Download OmniSQL database files (~22GB)
huggingface-cli download seeklhy/OmniSQL-datasets data.zip \
  --repo-type dataset --local-dir ~/data/sql/db_files/

# Extract
unzip ~/data/sql/db_files/data.zip -d ~/data/sql/db_files/
```

### 3. Run Training

```bash
# Basic FSDP training
bash examples/text_to_sql/run_sql_fsdp.sh

# With conversation format (multi-turn)
bash examples/text_to_sql/run_skyrl_sql_conversation_format.sh

# With Megatron + LoRA
bash examples/text_to_sql/run_skyrl_sql_megatron_lora.sh
```

## Available Scripts

| Script | Description |
|--------|-------------|
| `run_sql_fsdp.sh` | Basic FSDP training |
| `run_skyrl_sql.sh` | Standard SkyRL training |
| `run_skyrl_sql_conversation_format.sh` | Multi-turn with conversation format |
| `run_skyrl_sql_megatron_lora.sh` | Large-scale with Megatron + LoRA |
| `run_sql_fsdp_2node.sh` | Multi-node distributed training |

## Configuration Notes

For SGLang backend, ensure:
```yaml
generator:
  backend: sglang
  use_conversation_multi_turn: true  # Required for multi-turn
  max_turns: 5                        # Allow query refinement
```

## How It Works

1. **Prompt**: Natural language question + database schema
2. **Generation**: Model outputs SQL query
3. **Execution**: Query runs against actual database
4. **Feedback**: Results or error message returned
5. **Iteration**: Model refines query based on feedback
6. **Reward**: Correct results = 1.0, otherwise 0.0

## Related Documentation

- [Multi-Turn Training Guide](../../docs/MULTI_TURN_TRAINING.md)
- [Custom Environments](../../docs/CUSTOM_ENVIRONMENTS.md)
- [SGLang Integration](../../docs/SGLANG_INTEGRATION_GUIDE.md)

## Dataset Details

See [sql.md](./sql.md) for detailed dataset download instructions.
