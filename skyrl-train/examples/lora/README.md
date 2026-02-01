# LoRA Fine-Tuning

Parameter-efficient RL training using Low-Rank Adaptation (LoRA).

## When to Use LoRA

Use LoRA when:
- Limited GPU memory
- Want to preserve base model capabilities
- Training adapter for specific task
- Faster iteration cycles needed

Use full fine-tuning when:
- Maximum performance required
- Sufficient GPU memory available
- Training from scratch or major adaptation

## Memory Savings

| Model Size | Full Fine-Tune | LoRA (r=32) | Savings |
|------------|----------------|-------------|---------|
| 0.5B | ~4GB | ~2GB | 50% |
| 7B | ~28GB | ~8GB | 70% |
| 72B | ~288GB | ~40GB | 85% |

## Available Scripts

| Script | Algorithm | Model |
|--------|-----------|-------|
| `run_qwen2_5_0.5b_gsm8k_grpo_lora.sh` | GRPO | Qwen2.5-0.5B |
| `run_qwen2_5_0.5b_gsm8k_ppo_lora.sh` | PPO | Qwen2.5-0.5B |

## Quick Start

```bash
# GRPO with LoRA
bash examples/lora/run_qwen2_5_0.5b_gsm8k_grpo_lora.sh

# PPO with LoRA
bash examples/lora/run_qwen2_5_0.5b_gsm8k_ppo_lora.sh
```

## Configuration

```yaml
trainer:
  policy:
    model:
      path: "Qwen/Qwen2.5-0.5B-Instruct"
      lora:
        rank: 32
        alpha: 32
        target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]
    optimizer_config:
      lr: 3.0e-5  # Higher LR for LoRA

generator:
  enable_lora: true
  engine_init_kwargs:
    max_lora_rank: 64
```

## LoRA Parameters

| Parameter | Typical Value | Effect |
|-----------|---------------|--------|
| `rank` | 16-64 | Higher = more capacity, more memory |
| `alpha` | rank | Scaling factor (usually = rank) |
| `target_modules` | q,k,v,o_proj | Which layers to adapt |

## Related Documentation

- [SGLang LoRA Support](../../docs/SGLANG_INTEGRATION_GUIDE.md#advanced-features)
- [SGLang Limitations](../../docs/SGLANG_LIMITATIONS.md#5-lora-adapters)
