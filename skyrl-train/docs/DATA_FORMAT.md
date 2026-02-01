# SkyRL Data Format Specification

**Complete specification for datasets used in SkyRL RL training.**

---

## Overview

SkyRL datasets consist of prompts with associated environment configurations and reward specifications. This document covers:

1. Required and optional fields
2. Data format examples
3. File format support
4. Dataset preprocessing

---

## 1. Dataset Schema

### Required Fields

Every dataset item must contain these fields:

```python
{
    "prompt": List[Dict[str, str]],  # Chat messages (OpenAI format)
    "env_class": str,                 # Environment identifier
    "reward_spec": Dict[str, Any],    # Reward configuration
}
```

### Optional Fields

```python
{
    "data_source": str,               # Dataset source identifier
    "extra_info": Dict[str, Any],     # Additional metadata
    "ability": str,                   # Task category (e.g., "math", "code")
}
```

---

## 2. Field Specifications

### 2.1 prompt

The `prompt` field contains chat messages in OpenAI format:

```python
[
    {
        "role": "system",    # Optional
        "content": "You are a helpful assistant..."
    },
    {
        "role": "user",      # Required: at least one user message
        "content": "Your question or task here"
    }
]
```

**Supported Roles:**
- `system` - System instructions (optional, usually first)
- `user` - User messages (required)
- `assistant` - Pre-filled assistant responses (optional)

**Example:**
```python
"prompt": [
    {"role": "system", "content": "You are a math tutor."},
    {"role": "user", "content": "What is 2 + 2?"}
]
```

**Processing:**
- Applied through tokenizer's `chat_template`
- Adds generation prompt for assistant response
- Left-padded to `max_prompt_length`

---

### 2.2 env_class

Environment identifier matching a registered environment:

```python
"env_class": "gsm8k"
```

**Built-in Environments:**
| ID | Description |
|----|-------------|
| `gsm8k` | Grade school math (single-turn) |
| `gsm8k_multi_turn` | Math with multiple attempts |
| `aime` | Advanced math problems |
| `text2sql` | SQL generation |
| `search` | Information retrieval |
| `lcb` | Live code bench |
| `searchcode` | Code search |

**Custom Environments:**
Register at runtime before training:
```python
from skyrl_gym.envs import register
register(id="my_env", entry_point="my_module.env:MyEnvClass")
```

---

### 2.3 reward_spec

Configuration for reward computation:

```python
"reward_spec": {
    "method": "rule",           # Reward method (usually "rule")
    "ground_truth": <value>,    # Expected answer/reference
}
```

**Ground Truth Types:**

1. **String (exact match):**
   ```python
   "ground_truth": "42"
   ```

2. **Number:**
   ```python
   "ground_truth": 42.0
   ```

3. **List of acceptable answers:**
   ```python
   "ground_truth": ["Paris", "paris", "PARIS"]
   ```

4. **Test cases (for code):**
   ```python
   "ground_truth": [
       {"input": "5", "output": "25"},
       {"input": "10", "output": "100"}
   ]
   ```

---

### 2.4 extra_info

Additional metadata passed to the environment:

```python
"extra_info": {
    "split": "train",           # Dataset split
    "index": 0,                 # Original index
    "max_turns": 5,             # For multi-turn environments
    "question": "original q",   # Original question text
    "answer": "full answer",    # Complete reference answer
    "difficulty": "easy",       # Task difficulty
    "topic": "algebra",         # Topic category
}
```

---

## 3. Complete Examples

### 3.1 Math Problem (GSM8K)

```python
{
    "data_source": "openai/gsm8k",
    "prompt": [
        {
            "role": "user",
            "content": "Janet has 3 apples. She buys 2 more apples. How many apples does Janet have now? Let's think step by step and output the final answer after \"####\"."
        }
    ],
    "env_class": "gsm8k",
    "reward_spec": {
        "method": "rule",
        "ground_truth": "5"
    },
    "extra_info": {
        "split": "train",
        "index": 0,
        "answer": "Janet starts with 3 apples. She buys 2 more. 3 + 2 = 5.\n#### 5",
        "question": "Janet has 3 apples. She buys 2 more apples. How many apples does Janet have now?"
    }
}
```

---

### 3.2 Multiplication (Custom Environment)

```python
{
    "data_source": "synthetic_multiply",
    "prompt": [
        {
            "role": "system",
            "content": "You are a helpful assistant that solves multiplication problems. Put your final answer in \\boxed{answer} format."
        },
        {
            "role": "user",
            "content": "What is 42 * 73?"
        }
    ],
    "env_class": "multiply",
    "reward_spec": {
        "method": "rule",
        "ground_truth": "3066"
    },
    "extra_info": {
        "num_digits": 2,
        "split": "train"
    }
}
```

---

### 3.3 Code Generation (LiveCodeBench)

```python
{
    "data_source": "livecodebench",
    "prompt": [
        {
            "role": "system",
            "content": "You are an expert Python programmer. Write clean, efficient code."
        },
        {
            "role": "user",
            "content": "Write a function that returns the square of a number.\n\n```python\ndef square(n):\n    # YOUR CODE HERE\n```"
        }
    ],
    "env_class": "lcb",
    "reward_spec": {
        "method": "rule",
        "ground_truth": [
            {"input": "5", "output": "25"},
            {"input": "-3", "output": "9"},
            {"input": "0", "output": "0"}
        ]
    },
    "extra_info": {
        "split": "train",
        "difficulty": "easy"
    }
}
```

---

### 3.4 Multi-turn with Tool Use (SQL)

```python
{
    "data_source": "spider",
    "prompt": [
        {
            "role": "system",
            "content": "You are a SQL expert. Use <sql>...</sql> to execute queries. When ready, provide your final answer in <solution>...</solution>."
        },
        {
            "role": "user",
            "content": "Database: employees\nTables: employees (id, name, salary)\n\nQuestion: What is the average salary?"
        }
    ],
    "env_class": "text2sql",
    "reward_spec": {
        "method": "rule",
        "ground_truth": "SELECT AVG(salary) FROM employees"
    },
    "extra_info": {
        "db_id": "employees",
        "max_turns": 5,
        "split": "train"
    }
}
```

---

### 3.5 Search/Retrieval

```python
{
    "data_source": "searchR1_nq",
    "prompt": [
        {
            "role": "system",
            "content": "You are a helpful assistant. Use <search>query</search> to search for information."
        },
        {
            "role": "user",
            "content": "Who won the 2024 Super Bowl?"
        }
    ],
    "env_class": "search",
    "reward_spec": {
        "ground_truth": ["Kansas City Chiefs", "Chiefs"],
        "question": "Who won the 2024 Super Bowl?",
        "data_source": "searchR1_nq"
    },
    "extra_info": {
        "need_tools_kwargs": true,
        "tools_kwargs": {
            "search": {
                "create_kwargs": {
                    "ground_truth": ["Kansas City Chiefs"],
                    "question": "Who won the 2024 Super Bowl?"
                }
            }
        }
    }
}
```

---

## 4. File Formats

### 4.1 Parquet (Recommended)

```python
from datasets import Dataset
import pandas as pd

# From list of dicts
data = [
    {"prompt": [...], "env_class": "gsm8k", "reward_spec": {...}},
    {"prompt": [...], "env_class": "gsm8k", "reward_spec": {...}},
]
dataset = Dataset.from_list(data)
dataset.to_parquet("train.parquet")

# From pandas
df = pd.DataFrame(data)
df.to_parquet("train.parquet")
```

**Configuration:**
```yaml
data:
  train_data:
    - "~/data/my_dataset/train.parquet"
  val_data:
    - "~/data/my_dataset/validation.parquet"
```

---

### 4.2 JSON/JSONL

```json
// data.json (JSON array)
[
    {"prompt": [...], "env_class": "gsm8k", "reward_spec": {...}},
    {"prompt": [...], "env_class": "gsm8k", "reward_spec": {...}}
]
```

```jsonl
// data.jsonl (one JSON object per line)
{"prompt": [...], "env_class": "gsm8k", "reward_spec": {...}}
{"prompt": [...], "env_class": "gsm8k", "reward_spec": {...}}
```

---

### 4.3 HuggingFace Datasets

```yaml
data:
  train_data:
    - "openai/gsm8k"        # Uses "train" split
    - "openai/gsm8k:test"   # Specific split
```

**Note:** HuggingFace datasets may need preprocessing to match the required schema.

---

## 5. Dataset Creation Script

Template for creating custom datasets:

```python
#!/usr/bin/env python
"""Create dataset for SkyRL training."""

import argparse
import os
from datasets import Dataset

def create_example(question: str, answer: str, idx: int) -> dict:
    """Create a single dataset example."""
    return {
        "data_source": "my_custom_dataset",
        "prompt": [
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": question
            }
        ],
        "env_class": "my_env",  # Must match registered environment
        "reward_spec": {
            "method": "rule",
            "ground_truth": answer
        },
        "extra_info": {
            "split": "train",
            "index": idx
        }
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="~/data/my_dataset")
    parser.add_argument("--train_size", type=int, default=1000)
    parser.add_argument("--val_size", type=int, default=100)
    args = parser.parse_args()

    output_dir = os.path.expanduser(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Create training data
    train_examples = []
    for i in range(args.train_size):
        # Your logic to generate questions and answers
        question = f"Question {i}"
        answer = f"Answer {i}"
        train_examples.append(create_example(question, answer, i))

    # Create validation data
    val_examples = []
    for i in range(args.val_size):
        question = f"Val Question {i}"
        answer = f"Val Answer {i}"
        val_examples.append(create_example(question, answer, i))

    # Save as parquet
    Dataset.from_list(train_examples).to_parquet(
        os.path.join(output_dir, "train.parquet")
    )
    Dataset.from_list(val_examples).to_parquet(
        os.path.join(output_dir, "validation.parquet")
    )

    print(f"Created {len(train_examples)} training examples")
    print(f"Created {len(val_examples)} validation examples")
    print(f"Saved to {output_dir}")

if __name__ == "__main__":
    main()
```

---

## 6. Preprocessing Pipeline

### 6.1 Loading

```python
from skyrl_train.dataset.dataset import PromptDataset

dataset = PromptDataset(
    datasets=["~/data/my_dataset/train.parquet"],
    tokenizer=tokenizer,
    max_prompt_length=512,
    num_workers=8,
    prompt_key="prompt",
    env_class_key="env_class",
)
```

### 6.2 Filtering

Prompts exceeding `max_prompt_length` are filtered out:

```python
# Prompt tokenization
tokens = tokenizer.apply_chat_template(
    prompt,
    add_generation_prompt=True,
    tokenize=True
)

# Keep only if within limit
if len(tokens) <= max_prompt_length:
    keep_example()
```

### 6.3 Batching

Dataset items are batched for training:

```python
batch = {
    "prompt": [msg1, msg2, ...],      # Chat messages
    "env_class": "gsm8k",
    "env_extras": {
        "reward_spec": {...},
        "max_turns": 5,
        ...
    },
    "uid": "item_0"
}
```

---

## 7. Validation Checklist

Before training, verify your dataset:

- [ ] All items have `prompt` field (list of message dicts)
- [ ] All items have `env_class` field (string)
- [ ] All items have `reward_spec` field with `ground_truth`
- [ ] Prompts use correct role names (`system`, `user`, `assistant`)
- [ ] At least one `user` message in each prompt
- [ ] `env_class` matches a registered environment
- [ ] `ground_truth` type matches environment expectations
- [ ] Tokenized prompts fit within `max_prompt_length`

### Validation Script

```python
from datasets import load_dataset

def validate_dataset(path: str):
    """Validate dataset format."""
    ds = load_dataset("parquet", data_files=path)["train"]

    errors = []
    for i, item in enumerate(ds):
        # Check required fields
        if "prompt" not in item:
            errors.append(f"Row {i}: Missing 'prompt'")
        if "env_class" not in item:
            errors.append(f"Row {i}: Missing 'env_class'")
        if "reward_spec" not in item:
            errors.append(f"Row {i}: Missing 'reward_spec'")

        # Check prompt format
        if "prompt" in item:
            prompt = item["prompt"]
            if not isinstance(prompt, list):
                errors.append(f"Row {i}: 'prompt' must be a list")
            else:
                has_user = any(m.get("role") == "user" for m in prompt)
                if not has_user:
                    errors.append(f"Row {i}: No 'user' message in prompt")

        # Check reward_spec
        if "reward_spec" in item:
            spec = item["reward_spec"]
            if "ground_truth" not in spec:
                errors.append(f"Row {i}: Missing 'ground_truth' in reward_spec")

    if errors:
        print(f"Found {len(errors)} errors:")
        for e in errors[:10]:
            print(f"  - {e}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")
    else:
        print(f"Dataset valid! {len(ds)} examples.")

# Usage
validate_dataset("~/data/my_dataset/train.parquet")
```

---

## 8. Common Issues

### Issue: Dataset loads but training fails

**Cause:** Schema mismatch with environment expectations.

**Solution:** Check environment's `__init__` for required fields in `extras`.

### Issue: All rewards are 0

**Cause:** Answer parsing doesn't match output format.

**Solution:** Check environment's `_parse_answer` method against model outputs.

### Issue: Tokenization errors

**Cause:** Invalid message format or special characters.

**Solution:** Validate message structure and escape special tokens.

---

## References

- [Custom Environments Guide](./CUSTOM_ENVIRONMENTS.md)
- [GSM8K Dataset Example](../examples/gsm8k/gsm8k_dataset.py)
- [Multiply Dataset Example](../examples/multiply/multiply_dataset.py)
