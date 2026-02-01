# GPT-OSS Training

Train OpenAI's open-source GPT-OSS-20B model with GRPO on GSM8K.

## Overview

GPT-OSS (gpt-oss-20b) is OpenAI's open-source model that uses flex attention with attention sinks. This example shows how to train it with SkyRL.

**Important Limitations:**
- Flash Attention must be disabled (attention sinks not supported)
- Sequence packing must be disabled
- Only BF16 precision is supported
- Multi-turn training not yet supported

## Prerequisites

- 8 GPUs (H100 recommended)
- ~160GB total GPU memory
- GSM8K dataset prepared

## Quick Start

```bash
# 1. Prepare dataset
python examples/gsm8k/gsm8k_dataset.py

# 2. Set up logging
export WANDB_API_KEY=<your_key_here>

# 3. Run training
bash examples/gptoss/run_gsm8k_gptoss.sh
```

## Configuration Highlights

```yaml
# Required for GPT-OSS
trainer.flash_attn: false
trainer.use_sample_packing: false

# Model
trainer.policy.model.path: "unsloth/gpt-oss-20b-BF16"

# Inference
generator.inference_engine_tensor_parallel_size: 4
generator.num_inference_engines: 2
generator.enforce_eager: true

# Reasoning effort (via chat template)
+generator.chat_template_kwargs: {reasoning_effort: 'low'}
```

## Available Scripts

| Script | Description |
|--------|-------------|
| `run_gsm8k_gptoss.sh` | GRPO training on GSM8K |
| `bench_flex_attn.py` | Flex attention benchmarking |
| `test_gptoss.py` | Model testing utilities |

## Backend Support

Both vLLM and SGLang are supported. Change the backend in the script:

```bash
INFERENCE_BACKEND="vllm"    # Default
# or
INFERENCE_BACKEND="sglang"  # Alternative
```

## Reasoning Effort

Control reasoning depth via `chat_template_kwargs`:

```yaml
+generator.chat_template_kwargs: {reasoning_effort: 'low'}    # Fast, less reasoning
+generator.chat_template_kwargs: {reasoning_effort: 'medium'} # Balanced
+generator.chat_template_kwargs: {reasoning_effort: 'high'}   # Deep reasoning
```

## Related Documentation

- [SGLang Integration Guide](../../docs/SGLANG_INTEGRATION_GUIDE.md)
- [GSM8K Example](../gsm8k/README.md)
