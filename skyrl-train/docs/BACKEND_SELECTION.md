# Backend Selection Guide: SGLang vs vLLM

**How to choose between SGLang and vLLM for your RL training.**

---

## Quick Decision

```
What's your priority?

Need multi-turn/agent training?
├── Yes → SGLang (better prefix caching)
└── No → Continue...

Using RTX 3090/4090 or other consumer GPU?
├── Yes → SGLang (FlashInfer works everywhere)
└── No → Continue...

Already familiar with vLLM?
├── Yes → vLLM (familiar API)
└── No → SGLang (recommended default)
```

---

## Comparison Matrix

| Feature | SGLang | vLLM |
|---------|--------|------|
| **Prefix Caching** | RadixAttention (tree-based) | PagedAttention |
| **Multi-turn** | Excellent | Good |
| **Weight Sync** | Native distributed API | Worker extension |
| **Sleep/Wake** | All-or-nothing | Configurable levels |
| **Attention Backends** | FlashInfer, FA3 | FlashAttention, FlashInfer |
| **GPU Compatibility** | Broad (FlashInfer) | Broad |
| **PyTorch Version** | 2.9.1 | 2.8.0 |
| **Documentation** | Newer | More mature |

---

## 1. SGLang (Recommended for New Users)

### When to Choose SGLang

- **Multi-turn agent training** - RadixAttention efficiently caches conversation prefixes
- **Consumer GPUs** (RTX 3090, 4090) - FlashInfer works on SM86+
- **Starting fresh** - Native weight sync integration
- **Memory-constrained** - Sleep releases all GPU memory

### Configuration

```yaml
generator:
  backend: sglang
  num_inference_engines: 4
  gpu_memory_utilization: 0.8
  enable_prefix_caching: true
  use_conversation_multi_turn: true  # Required for SGLang
  weight_sync_backend: nccl

  engine_init_kwargs:
    attention_backend: flashinfer  # Works on most GPUs
```

### Installation

```bash
cd SkyRL/skyrl-train
uv sync --extra sglang
source .venv/bin/activate
```

### Key Differences

1. **Multi-turn Required**: SGLang requires `use_conversation_multi_turn=true`
2. **Sleep Behavior**: Releases ALL GPU memory (vs vLLM's configurable levels)
3. **Attention Backend**: Specify via `engine_init_kwargs.attention_backend`

---

## 2. vLLM

### When to Choose vLLM

- **Already using vLLM** - Familiar API and configuration
- **Need fine-grained sleep** - Configure what to release during training
- **Specific vLLM features** - Using vLLM-specific extensions

### Configuration

```yaml
generator:
  backend: vllm
  num_inference_engines: 4
  gpu_memory_utilization: 0.8
  enable_prefix_caching: true
  weight_sync_backend: nccl

  # vLLM-specific
  vllm_v1_disable_multiproc: false
  enforce_eager: false
```

### Installation

```bash
cd SkyRL/skyrl-train
uv sync --extra vllm
source .venv/bin/activate
```

### Key Differences

1. **Sleep Levels**: Can configure what resources to release
2. **Multi-turn**: Optional, not required
3. **v1 API**: Has multiprocessing toggle

---

## 3. Performance Comparison

### Generation Throughput

| Scenario | SGLang | vLLM |
|----------|--------|------|
| Single-turn prompts | Similar | Similar |
| Multi-turn (shared prefix) | **Faster** | Good |
| Long sequences | Similar | Similar |

### Weight Sync Time

Both backends support CUDA IPC with `colocate_all=true`:

| Model Size | SGLang | vLLM |
|------------|--------|------|
| 0.5B | ~0.5s | ~0.5s |
| 7B | ~1.9s | ~2.0s |
| 70B | ~8s | ~8s |

### Memory Efficiency

| Aspect | SGLang | vLLM |
|--------|--------|------|
| KV Cache | PagedAttention | PagedAttention |
| Sleep/Wake | All-or-nothing | Configurable levels |
| Prefix Sharing | RadixAttention (tree) | Block-based |

---

## 4. GPU Compatibility

### SGLang

| GPU | FlashInfer | FA3 |
|-----|------------|-----|
| RTX 3090 (SM86) | Yes | No |
| RTX 4090 (SM89) | Yes | No |
| A100 (SM80) | Yes | Yes |
| H100 (SM90) | Yes | Yes |
| L4 (SM89) | Yes | No |

**Use `attention_backend=flashinfer` for consumer GPUs.**

### vLLM

| GPU | FlashAttention | FlashInfer |
|-----|----------------|------------|
| RTX 3090 | Yes (v2) | Yes |
| RTX 4090 | Yes (v2) | Yes |
| A100 | Yes (v2/v3) | Yes |
| H100 | Yes (v2/v3) | Yes |

---

## 5. Migration Guide

### vLLM to SGLang

```yaml
# Before (vLLM)
generator:
  backend: vllm
  vllm_v1_disable_multiproc: true
  enforce_eager: true

# After (SGLang)
generator:
  backend: sglang
  use_conversation_multi_turn: true  # Add this
  engine_init_kwargs:
    attention_backend: flashinfer  # Add for GPU compatibility
```

**Key changes:**
1. Add `use_conversation_multi_turn: true`
2. Add `engine_init_kwargs.attention_backend: flashinfer`
3. Remove vLLM-specific settings

### SGLang to vLLM

```yaml
# Before (SGLang)
generator:
  backend: sglang
  use_conversation_multi_turn: true
  engine_init_kwargs:
    attention_backend: flashinfer

# After (vLLM)
generator:
  backend: vllm
  # Remove use_conversation_multi_turn (optional for vLLM)
  # Remove engine_init_kwargs (different format)
```

**Note:** Must reinstall with different extra:
```bash
# Create new venv or reinstall
uv sync --extra vllm  # Was: --extra sglang
```

---

## 6. Common Scenarios

### Scenario 1: Math Problem Solving (GSM8K)

**Recommendation:** Either works well, SGLang slightly preferred for consistency.

```yaml
generator:
  backend: sglang
  n_samples_per_prompt: 4
  sampling_params:
    temperature: 1.0
```

### Scenario 2: Multi-turn Agent Training

**Recommendation:** SGLang for better prefix caching.

```yaml
generator:
  backend: sglang
  use_conversation_multi_turn: true
  enable_prefix_caching: true
  # RadixAttention will cache system prompts efficiently
```

### Scenario 3: Code Generation

**Recommendation:** Either works, depends on GPU availability.

```yaml
generator:
  backend: sglang
  sampling_params:
    max_generate_length: 2048
    temperature: 0.7
```

### Scenario 4: Memory-Constrained (Single GPU)

**Recommendation:** SGLang with aggressive memory management.

```yaml
generator:
  backend: sglang
  gpu_memory_utilization: 0.5  # Reserve for training

trainer:
  placement:
    colocate_all: true  # Share GPU between train/inference
```

---

## 7. Troubleshooting

### SGLang-Specific Issues

**"use_conversation_multi_turn=False is not supported"**
```yaml
# Fix: Always keep multi-turn enabled for SGLang
generator:
  use_conversation_multi_turn: true
```

**"AssertionError: FlashAttention v3 Backend requires SM>=80"**
```yaml
# Fix: Use FlashInfer instead
generator:
  engine_init_kwargs:
    attention_backend: flashinfer
```

### vLLM-Specific Issues

**Worker hangs during initialization**
```yaml
# Fix: Disable multiprocessing
generator:
  vllm_v1_disable_multiproc: true
```

**OOM during generation**
```yaml
# Fix: Reduce memory usage
generator:
  gpu_memory_utilization: 0.6
  enforce_eager: true  # Disable CUDA graphs
```

---

## 8. Feature Support Matrix

| Feature | SGLang | vLLM |
|---------|--------|------|
| GRPO | Yes | Yes |
| PPO with GAE | Yes | Yes |
| RLOO | Yes | Yes |
| DAPO | Yes | Yes |
| LoRA | Yes | Yes |
| Tensor Parallel | Yes | Yes |
| Pipeline Parallel | Yes | Yes |
| FSDP2 Training | Yes | Yes |
| Megatron Training | Yes | Yes |
| Multi-node | Yes | Yes |
| Prefix Caching | Yes (RadixAttention) | Yes (PagedAttention) |
| Speculative Decoding | Yes | Yes |

---

## 9. Installation Notes

### Cannot Install Both

Due to PyTorch version conflicts, you cannot have both backends in the same environment:

```bash
# SGLang needs PyTorch 2.9.1
uv sync --extra sglang

# vLLM needs PyTorch 2.8.0
uv sync --extra vllm

# This will NOT work:
# uv sync --extra sglang --extra vllm
```

### Switching Backends

```bash
# Option 1: Reinstall
uv sync --extra sglang  # Switch to SGLang
uv sync --extra vllm    # Switch to vLLM

# Option 2: Separate venvs
python -m venv .venv-sglang
python -m venv .venv-vllm
```

---

## Summary

| Choose SGLang If... | Choose vLLM If... |
|---------------------|-------------------|
| Multi-turn agent training | Already familiar with vLLM |
| Consumer GPU (RTX 3090/4090) | Need fine-grained sleep levels |
| Starting fresh | Using vLLM-specific features |
| Want native weight sync | Have existing vLLM setup |

**When in doubt, choose SGLang** - it's the recommended default for SkyRL.

---

## References

- [SGLang Quickstart](./QUICKSTART_SGLANG.md)
- [Full Configuration Reference](./SGLANG_INTEGRATION_GUIDE.md)
- [SGLang Documentation](https://docs.sglang.io/)
- [vLLM Documentation](https://docs.vllm.ai/)
