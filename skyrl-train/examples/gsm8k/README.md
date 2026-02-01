# GSM8K Example

Train a language model to solve grade school math problems using GRPO.

## Overview

[GSM8K](https://github.com/openai/grade-school-math) is a dataset of 8.5K grade school math word problems. This example demonstrates:

- GRPO (Group Relative Policy Optimization) training
- SGLang or vLLM as inference backend
- Reward based on answer correctness

## Files

| File | Description |
|------|-------------|
| `gsm8k_dataset.py` | Script to download and prepare GSM8K dataset |
| `gsm8k-grpo-sglang-skypilot.yaml` | SGLang backend configuration (SkyPilot) |
| `gsm8k-grpo-skypilot.yaml` | vLLM backend configuration (SkyPilot) |
| `run_gsm8k.sh` | Local training script |
| `run_32b_gsm8k.sh` | Multi-GPU training for larger models |
| `run_gsm8k_modal.sh` | Modal cloud training script |
| `run_generation_gsm8k.sh` | Evaluation/generation only |

## Quick Start

### 1. Prepare Dataset

```bash
python examples/gsm8k/gsm8k_dataset.py --output_dir ~/data/gsm8k
```

Creates:
- `~/data/gsm8k/train.parquet` (~7,500 problems)
- `~/data/gsm8k/validation.parquet` (~1,300 problems)

### 2. Run Training (SGLang)

```bash
python -m skyrl_train.entrypoints.main_base \
  +experiment=grpo_qwen2.5-0.5b_math500 \
  generator.backend=sglang \
  +generator.engine_init_kwargs.attention_backend=flashinfer
```

Or use the shell script:

```bash
bash examples/gsm8k/run_gsm8k.sh
```

### 3. Expected Results

| Model | Initial Accuracy | After Training |
|-------|-----------------|----------------|
| Qwen2.5-0.5B-Instruct | ~24% | ~45% |

## Configuration

Key settings in `gsm8k-grpo-sglang-skypilot.yaml`:

```yaml
trainer:
  algorithm:
    advantage_estimator: grpo  # Group-relative advantages
  train_batch_size: 256
  policy_mini_batch_size: 64

generator:
  backend: sglang
  n_samples_per_prompt: 4  # Generate 4 responses per problem
  sampling_params:
    temperature: 1.0  # Exploration during training
```

## Reward Function

The GSM8K environment (`env_class: gsm8k`):
1. Parses the model's final answer (looks for `\boxed{...}` or final number)
2. Compares to ground truth
3. Returns reward: 1.0 (correct) or 0.0 (incorrect)

## Documentation

- [Quickstart Guide](../../docs/QUICKSTART_SGLANG.md)
- [Full Tutorial](../../docs/TUTORIAL_SGLANG.md)
- [Algorithms Guide](../../docs/ALGORITHMS.md)
