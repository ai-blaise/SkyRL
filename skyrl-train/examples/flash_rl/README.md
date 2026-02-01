# FlashRL: Quantized Inference for RL Training

Use INT8 or FP8 quantized rollouts for memory-efficient RL training.

## Overview

FlashRL enables quantized inference during trajectory generation while maintaining full-precision training. This dramatically reduces GPU memory requirements for the inference engine, allowing:
- **Larger models** on the same hardware
- **More inference engines** for higher throughput
- **Lower memory utilization** for colocated setups

## Prerequisites

- 4+ H100 GPUs (INT8/FP8 support required)
- CUDA 12.8+
- GSM8K dataset prepared

## Quick Start

```bash
# 1. Prepare dataset
python examples/gsm8k/gsm8k_dataset.py

# 2. Run with INT8 quantization
export WANDB_API_KEY=<your_key_here>
bash examples/flash_rl/run_dapo_gsm8k_flashrl_0.5b_int8.sh

# Or with FP8 quantization
bash examples/flash_rl/run_dapo_gsm8k_flashrl_0.5b_fp8.sh
```

## Available Scripts

| Script | Model | Quantization |
|--------|-------|--------------|
| `run_dapo_gsm8k_flashrl_0.5b_int8.sh` | Qwen2.5-0.5B | INT8 |
| `run_dapo_gsm8k_flashrl_0.5b_fp8.sh` | Qwen2.5-0.5B | FP8 |
| `run_dapo_gsm8k_flashrl_32b_int8.sh` | Qwen2.5-32B | INT8 |
| `run_dapo_repro_flashrl_0.5b_int8.sh` | Qwen2.5-0.5B | INT8 (DAPO repro) |
| `run_dapo_repro_flashrl_32b_int8.sh` | Qwen2.5-32B | INT8 (DAPO repro) |

## Environment Files

FlashRL uses environment files to configure quantization:

| File | Description |
|------|-------------|
| `.env.int8` | INT8 quantization settings |
| `.env.fp8` | FP8 quantization settings |
| `.env.0.5b_int8` | 0.5B model INT8 settings |

## How to Use

FlashRL uses a custom entrypoint with specific vLLM wheel:

```bash
uv run --isolated --extra flashrl \
  --env-file examples/flash_rl/.env.int8 \
  --with vllm@<flashrl-vllm-wheel-url> \
  --with transformers==4.53.3 \
  -- python -m examples.flash_rl.main_dapo_flashrl \
  ...
```

## Memory Comparison

| Configuration | Model | Inference Memory | Training Memory |
|---------------|-------|------------------|-----------------|
| FP16 | 32B | ~64GB | ~128GB |
| INT8 | 32B | ~32GB | ~128GB |
| FP8 | 32B | ~32GB | ~128GB |

## Configuration Notes

```yaml
# Lower GPU utilization due to quantization overhead
generator.gpu_memory_utilization: 0.6

# Disable eager mode for best performance
generator.enforce_eager: false

# TIS is enabled by default in FlashRL scripts
trainer.algorithm.use_tis: true
trainer.algorithm.tis_imp_ratio_cap: 2.0
```

## Precision Trade-offs

| Quantization | Memory Savings | Speed | Accuracy Impact |
|--------------|----------------|-------|-----------------|
| FP16 (baseline) | 0% | Baseline | None |
| INT8 | ~50% | +10-20% | Minimal |
| FP8 | ~50% | +20-30% | Minimal |

## Related Documentation

- [TIS Correction](../tis_correction/README.md) - Used with FlashRL
- [DAPO Algorithm](../algorithms/dapo/README.md)
- [Training Strategies](../../docs/TRAINING_STRATEGIES.md)
