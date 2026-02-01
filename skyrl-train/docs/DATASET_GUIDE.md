# Dataset Preparation Guide

**Complete guide for preparing datasets for SkyRL training.**

---

## Table of Contents

1. [Dataset Format Overview](#1-dataset-format-overview)
2. [Required Fields](#2-required-fields)
3. [Optional Fields](#3-optional-fields)
4. [Field Placement and env_extras](#4-field-placement-and-env_extras)
5. [Supported File Formats](#5-supported-file-formats)
6. [Silent Filtering Behavior](#6-silent-filtering-behavior)
7. [Environment-Specific Datasets](#7-environment-specific-datasets)
8. [Dataset Preparation Scripts](#8-dataset-preparation-scripts)
9. [Validation and Debugging](#9-validation-and-debugging)
10. [Best Practices](#10-best-practices)

---

## 1. Dataset Format Overview

SkyRL datasets follow a structured format where each sample contains:

```json
{
    "prompt": [{"role": "user", "content": "Your question here"}],
    "env_class": "gsm8k",
    "reward_spec": {
        "method": "rule",
        "ground_truth": "42"
    },
    "extra_field_1": "...",
    "extra_field_2": "..."
}
```

### Key Principles

1. **Conversation format**: `prompt` uses chat message format
2. **Environment routing**: `env_class` determines which environment evaluates the response
3. **Reward specification**: `reward_spec` tells the environment how to compute rewards
4. **Extra fields**: All other fields are passed as `env_extras` to the environment

---

## 2. Required Fields

### 2.1 `prompt` (Required)

The input prompt in chat message format:

```json
{
    "prompt": [
        {"role": "system", "content": "You are a math expert."},
        {"role": "user", "content": "What is 2+2?"}
    ]
}
```

**Single-turn format:**
```json
{"prompt": [{"role": "user", "content": "Question here"}]}
```

**Multi-turn format:**
```json
{
    "prompt": [
        {"role": "user", "content": "Initial question"},
        {"role": "assistant", "content": "Initial response"},
        {"role": "user", "content": "Follow-up question"}
    ]
}
```

**Important:**
- Must be a list of message objects
- Each message must have `role` and `content` keys
- Valid roles: `"system"`, `"user"`, `"assistant"`
- Cannot mix message-list and string formats in same dataset

### 2.2 `env_class` (Required)

The registered environment class name:

```json
{"env_class": "gsm8k"}
```

**Built-in environments:**
| env_class | Task Type |
|-----------|-----------|
| `gsm8k` | Math (single-turn) |
| `gsm8k_multi_turn` | Math (multi-turn) |
| `aime` | Advanced math |
| `search` | Search/retrieval |
| `text2sql` | SQL generation |
| `lcb` | LiveCodeBench |
| `searchcode` | Code search |

### 2.3 `reward_spec` (Required for most environments)

Specifies how to compute rewards:

```json
{
    "reward_spec": {
        "method": "rule",
        "ground_truth": "42"
    }
}
```

**Common reward_spec formats:**

**Rule-based (exact match):**
```json
{
    "method": "rule",
    "ground_truth": "expected_answer"
}
```

**Model-based (LLM judge):**
```json
{
    "method": "model",
    "judge_model": "gpt-4",
    "criteria": "correctness"
}
```

**Custom:**
```json
{
    "method": "custom",
    "reward_fn": "my_module.reward_function"
}
```

---

## 3. Optional Fields

### 3.1 `extra_info` or `metadata`

Additional context passed to the environment:

```json
{
    "extra_info": {
        "difficulty": "hard",
        "topic": "algebra",
        "source": "competition"
    }
}
```

### 3.2 `trajectory_id`

**Required for step-wise training:**

```json
{"trajectory_id": "sample_12345"}
```

This enables tracking rewards across multi-turn conversations.

### 3.3 `data_source`

Track data provenance:

```json
{"data_source": "openai/gsm8k"}
```

### 3.4 Custom Fields

Any additional fields are passed to the environment as `env_extras`:

```json
{
    "prompt": [...],
    "env_class": "text2sql",
    "reward_spec": {...},
    "database_path": "/path/to/db.sqlite",
    "schema": "CREATE TABLE users (...)",
    "expected_rows": 5
}
```

These become available in the environment as:
```python
def step(self, action, env_extras):
    db_path = env_extras["database_path"]
    schema = env_extras["schema"]
```

---

## 4. Field Placement and env_extras

### How Fields Flow Through the System

```
Dataset Sample
      │
      ├── prompt ──────────────────────► Generator (tokenized)
      ├── env_class ───────────────────► Environment routing
      └── [all other fields] ──────────► env_extras dict
                                               │
                                               ▼
                                        Environment.step()
```

### Field Extraction (from `dataset.py`)

```python
def __getitem__(self, item):
    row_dict = self.dataframe[item]

    # Extract required fields
    messages = row_dict.pop(self.prompt_key)  # "prompt"
    env_class = row_dict.pop(self.env_class_key, None)  # "env_class"

    # Everything else becomes env_extras
    extra = {key: value for key, value in row_dict.items()
             if key != self.prompt_key and key != self.env_class_key}

    return messages, env_class, extra, uid
```

### Accessing Fields in Your Environment

```python
class MyEnv:
    def __init__(self, env_extras: dict):
        # Access any custom fields from dataset
        self.ground_truth = env_extras.get("reward_spec", {}).get("ground_truth")
        self.database = env_extras.get("database_path")
        self.metadata = env_extras.get("extra_info", {})

    def step(self, action):
        # Use the fields
        is_correct = self.evaluate(action, self.ground_truth)
        return {"reward": float(is_correct), "done": True}
```

---

## 5. Supported File Formats

### 5.1 Parquet (Recommended)

Best for large datasets:

```python
# Create
import pandas as pd
df = pd.DataFrame(data)
df.to_parquet("train.parquet")

# Load in config
data:
  train_data: "['path/to/train.parquet']"
```

### 5.2 JSON/JSONL

For smaller datasets:

```json
// train.json (array format)
[
    {"prompt": [...], "env_class": "gsm8k", ...},
    {"prompt": [...], "env_class": "gsm8k", ...}
]
```

```json
// train.jsonl (line-delimited)
{"prompt": [...], "env_class": "gsm8k", ...}
{"prompt": [...], "env_class": "gsm8k", ...}
```

### 5.3 HuggingFace Datasets

Direct loading from HF Hub:

```yaml
data:
  train_data: "openai/gsm8k:train"  # dataset:split
  val_data: "openai/gsm8k:test"
```

**Note:** HF datasets must have compatible field names or be preprocessed.

---

## 6. Silent Filtering Behavior

### 6.1 Length-Based Filtering

Prompts exceeding `max_prompt_length` are **silently filtered**:

```python
# From dataset.py
self.dataframe = self.dataframe.filter(
    lambda doc: len(tokenizer.apply_chat_template(doc[prompt_key], add_generation_prompt=True))
    <= self.max_prompt_length,
    num_proc=self.num_workers,
    desc=f"Filtering prompts longer than {self.max_prompt_length} tokens",
)
```

**What you'll see in logs:**
```
Total dataset size: 10000
Filtering prompts longer than 4096 tokens
Filtered dataset size: 9234  ← 766 samples silently removed
```

### 6.2 Implications

- **No error**: Long prompts are dropped without warning
- **Dataset shrinkage**: Final dataset may be smaller than expected
- **Training impact**: If many samples filtered, training may differ from expectations

### 6.3 Preventing Unexpected Filtering

1. **Pre-check your data:**
   ```python
   from transformers import AutoTokenizer

   tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")

   for sample in dataset:
       length = len(tokenizer.apply_chat_template(
           sample["prompt"],
           add_generation_prompt=True
       ))
       if length > 4096:
           print(f"Sample {sample['id']} exceeds limit: {length} tokens")
   ```

2. **Increase max_prompt_length:**
   ```yaml
   data:
     max_prompt_length: 8192  # Increase from default
   ```

3. **Truncate in preprocessing:**
   ```python
   def truncate_prompt(sample, max_chars=15000):
       content = sample["prompt"][0]["content"]
       if len(content) > max_chars:
           sample["prompt"][0]["content"] = content[:max_chars] + "..."
       return sample
   ```

### 6.4 Other Silent Behaviors

| Behavior | Description | Detection |
|----------|-------------|-----------|
| Length filtering | Prompts > max_prompt_length dropped | Compare dataset sizes in logs |
| Missing env_class | Samples without env_class may error | Check for KeyError |
| Invalid reward_spec | May cause runtime errors | Test with small batch first |
| Unicode issues | Invalid characters may cause tokenizer errors | Pre-validate text |

---

## 7. Environment-Specific Datasets

### 7.1 GSM8K (Math)

```json
{
    "prompt": [{"role": "user", "content": "What is 15 * 7? Let's think step by step."}],
    "env_class": "gsm8k",
    "reward_spec": {
        "method": "rule",
        "ground_truth": "105"
    },
    "extra_info": {
        "difficulty": "easy"
    }
}
```

### 7.2 Text-to-SQL

```json
{
    "prompt": [{"role": "user", "content": "List all users who joined in 2024.\n\nSchema:\nCREATE TABLE users (id INT, name TEXT, joined DATE);"}],
    "env_class": "text2sql",
    "reward_spec": {
        "method": "execution",
        "expected_result": [["Alice"], ["Bob"]]
    },
    "database_path": "/path/to/db.sqlite"
}
```

### 7.3 Search/Retrieval

```json
{
    "prompt": [{"role": "user", "content": "Find information about climate change impacts."}],
    "env_class": "search",
    "reward_spec": {
        "method": "retrieval_relevance",
        "relevant_docs": ["doc_123", "doc_456"]
    }
}
```

### 7.4 Code Generation (LiveCodeBench)

```json
{
    "prompt": [{"role": "user", "content": "def fibonacci(n):\n    '''Return the nth Fibonacci number.'''\n    # Your code here"}],
    "env_class": "lcb",
    "reward_spec": {
        "method": "execution",
        "test_cases": [
            {"input": [0], "expected": 0},
            {"input": [5], "expected": 5},
            {"input": [10], "expected": 55}
        ]
    }
}
```

### 7.5 Multi-Turn Math

```json
{
    "prompt": [{"role": "user", "content": "Solve: 3x + 5 = 20"}],
    "env_class": "gsm8k_multi_turn",
    "reward_spec": {
        "method": "rule",
        "ground_truth": "5"
    },
    "max_turns": 3
}
```

---

## 8. Dataset Preparation Scripts

### 8.1 Basic Script Template

```python
#!/usr/bin/env python
"""Prepare dataset for SkyRL training."""

import argparse
import datasets
import os

def process_sample(example, idx):
    """Convert raw sample to SkyRL format."""
    return {
        "prompt": [{"role": "user", "content": example["question"]}],
        "env_class": "my_env",
        "reward_spec": {
            "method": "rule",
            "ground_truth": example["answer"]
        },
        "extra_info": {
            "index": idx,
            "source": "my_dataset"
        }
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="~/data/my_dataset")
    args = parser.parse_args()

    # Load raw data
    raw_data = datasets.load_dataset("my_hf_dataset")

    # Process
    train_data = raw_data["train"].map(process_sample, with_indices=True)
    val_data = raw_data["test"].map(process_sample, with_indices=True)

    # Save
    output_dir = os.path.expanduser(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    train_data.to_parquet(f"{output_dir}/train.parquet")
    val_data.to_parquet(f"{output_dir}/validation.parquet")

    print(f"Saved {len(train_data)} train, {len(val_data)} val samples")
```

### 8.2 GSM8K Preparation

```bash
# Use provided script
cd SkyRL/skyrl-train
python examples/gsm8k/gsm8k_dataset.py --output_dir ~/data/gsm8k
```

### 8.3 Custom Dataset from CSV

```python
import pandas as pd

df = pd.read_csv("my_data.csv")
samples = []

for _, row in df.iterrows():
    samples.append({
        "prompt": [{"role": "user", "content": row["question"]}],
        "env_class": "gsm8k",
        "reward_spec": {"method": "rule", "ground_truth": str(row["answer"])}
    })

pd.DataFrame(samples).to_parquet("train.parquet")
```

---

## 9. Validation and Debugging

### 9.1 Validate Dataset Format

```python
import pandas as pd
from transformers import AutoTokenizer

def validate_dataset(path: str, tokenizer_name: str, max_length: int = 4096):
    """Validate dataset before training."""

    df = pd.read_parquet(path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    issues = []

    for idx, row in df.iterrows():
        # Check required fields
        if "prompt" not in row:
            issues.append(f"Row {idx}: Missing 'prompt'")
            continue

        if "env_class" not in row:
            issues.append(f"Row {idx}: Missing 'env_class'")

        # Check prompt format
        prompt = row["prompt"]
        if not isinstance(prompt, list):
            issues.append(f"Row {idx}: 'prompt' must be a list")
            continue

        for msg in prompt:
            if "role" not in msg or "content" not in msg:
                issues.append(f"Row {idx}: Invalid message format")

        # Check length
        try:
            tokens = tokenizer.apply_chat_template(prompt, add_generation_prompt=True)
            if len(tokens) > max_length:
                issues.append(f"Row {idx}: Length {len(tokens)} > {max_length}")
        except Exception as e:
            issues.append(f"Row {idx}: Tokenization error: {e}")

    if issues:
        print(f"Found {len(issues)} issues:")
        for issue in issues[:10]:
            print(f"  - {issue}")
        if len(issues) > 10:
            print(f"  ... and {len(issues) - 10} more")
    else:
        print(f"Dataset valid: {len(df)} samples, all checks passed")

    return len(issues) == 0

# Usage
validate_dataset("train.parquet", "Qwen/Qwen2.5-7B")
```

### 9.2 Debug Sample Loading

```python
from skyrl_train.dataset.dataset import PromptDataset
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")
dataset = PromptDataset(
    datasets=["train.parquet"],
    tokenizer=tokenizer,
    max_prompt_length=4096
)

# Check first sample
messages, env_class, extras, uid = dataset[0]
print(f"Messages: {messages}")
print(f"Env class: {env_class}")
print(f"Extras: {extras}")
```

### 9.3 Common Issues and Solutions

| Issue | Symptom | Solution |
|-------|---------|----------|
| Missing prompt | `KeyError: 'prompt'` | Add prompt field or configure `prompt_key` |
| Wrong prompt format | `TypeError` during tokenization | Use message list format |
| Long prompts | Dataset smaller than expected | Increase `max_prompt_length` or truncate |
| Missing env_class | `KeyError: 'env_class'` | Add env_class or configure `env_class_key` |
| Invalid reward_spec | Runtime error in environment | Validate reward_spec format |

---

## 10. Best Practices

### 10.1 Dataset Size Guidelines

| Task Type | Recommended Size | Notes |
|-----------|------------------|-------|
| Math (GSM8K-like) | 5K-50K | Diverse problems |
| Code generation | 10K-100K | Multiple difficulty levels |
| Text-to-SQL | 5K-20K | Various query types |
| Search/QA | 10K-50K | Wide topic coverage |

### 10.2 Data Quality Tips

1. **Validate answers**: Ensure ground_truth values are correct
2. **Consistent formatting**: Same prompt structure across samples
3. **Balanced difficulty**: Mix easy/medium/hard samples
4. **Diverse topics**: Cover the task domain comprehensively

### 10.3 Performance Optimization

```yaml
# Config for efficient data loading
data:
  num_workers: 8           # Parallel data loading
  prefetch_factor: 2       # Prefetch batches
  max_prompt_length: 4096  # Set based on actual data
```

### 10.4 Checklist Before Training

- [ ] Dataset saved in parquet format
- [ ] All samples have `prompt`, `env_class`, `reward_spec`
- [ ] `prompt` uses message list format
- [ ] `env_class` matches registered environment
- [ ] Checked for length filtering (compare before/after sizes)
- [ ] Validated with small test run
- [ ] Verified reward_spec format for chosen environment

---

## Related Documentation

- [Custom Environments](./CUSTOM_ENVIRONMENTS.md) - How to create environments
- [Configuration Reference](./CONFIG_REFERENCE.md) - Full config options
- [Troubleshooting](./TROUBLESHOOTING.md) - Common errors and fixes
