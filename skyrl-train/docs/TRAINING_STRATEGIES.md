# Training Strategies: FSDP2 vs Megatron

**Guide for choosing between FSDP2 and Megatron training backends.**

---

## Quick Decision Guide

```
What model are you training?

MoE (Mixture of Experts)?
├── Yes → Use Megatron (Expert Parallelism support)
└── No → Continue...

Model size > 30B?
├── Yes → Use Megatron (better scaling)
└── No → Continue...

Need maximum flexibility?
├── Yes → Use FSDP2 (easier HuggingFace integration)
└── No → Continue...

Production/Performance critical?
├── Yes → Benchmark both, choose faster
└── No → Use FSDP2 (simpler setup)
```

---

## Comparison Table

| Feature | FSDP2 | Megatron |
|---------|-------|----------|
| **Setup Complexity** | Simple | Complex |
| **HuggingFace Integration** | Native | Requires conversion |
| **MoE Support** | Limited | Full (5D parallelism) |
| **Small Models (<7B)** | Recommended | Overkill |
| **Large Models (>30B)** | Good | Better |
| **Custom Architectures** | Easy | Harder |
| **Checkpointing** | Standard PyTorch | Custom format |
| **Memory Efficiency** | Good | Better |
| **Throughput (Dense)** | Good | Similar |
| **Throughput (MoE)** | N/A | Excellent |

---

## 1. FSDP2 (Recommended Default)

### When to Use

- Models up to ~30B parameters
- Standard dense architectures (Llama, Qwen, Mistral)
- Quick experimentation
- HuggingFace model compatibility required
- Simpler deployment pipeline

### Configuration

```yaml
trainer:
  strategy: fsdp2
  policy:
    fsdp_config:
      cpu_offload: false           # Offload to CPU for memory
      reshard_after_forward: true  # Reshard weights after forward
      fsdp_size: -1                # Auto-detect sharding size
    sequence_parallel_size: 1      # Sequence parallelism (Ulysses)
```

### Key Parameters

| Parameter | Values | Description |
|-----------|--------|-------------|
| `cpu_offload` | true/false | Offload params and optimizer to CPU |
| `reshard_after_forward` | true/false/int | When to reshard weights |
| `fsdp_size` | -1 or int | Number of GPUs for sharding (-1 = all) |
| `sequence_parallel_size` | 1+ | Ulysses sequence parallel degree |

### Memory Optimization

**For memory-constrained setups:**
```yaml
trainer:
  policy:
    fsdp_config:
      cpu_offload: true            # Offload to CPU
      reshard_after_forward: true  # Aggressive resharding
  gradient_checkpointing: true     # Activation checkpointing
```

### Example: 7B Model on 4 GPUs

```yaml
trainer:
  strategy: fsdp2
  placement:
    policy_num_gpus_per_node: 4
  policy:
    fsdp_config:
      cpu_offload: false
      reshard_after_forward: true
      fsdp_size: 4
    sequence_parallel_size: 1
  gradient_checkpointing: true
  micro_train_batch_size_per_gpu: 1
```

---

## 2. Megatron

### When to Use

- MoE models (DeepSeek, Mixtral)
- Models >30B parameters
- Need expert parallelism
- Maximum throughput required
- Multi-node training

### Configuration

```yaml
trainer:
  strategy: megatron
  policy:
    megatron_config:
      tensor_model_parallel_size: 2
      pipeline_model_parallel_size: 1
      expert_model_parallel_size: 1  # For MoE
      sequence_parallel: true
```

### 5D Parallelism

Megatron supports combining:
1. **Data Parallel (DP)** - Replicate model across GPUs
2. **Tensor Parallel (TP)** - Split layers across GPUs
3. **Pipeline Parallel (PP)** - Split model stages across GPUs
4. **Expert Parallel (EP)** - Split MoE experts across GPUs
5. **Sequence Parallel (SP)** - Split sequences across GPUs

```yaml
trainer:
  policy:
    megatron_config:
      tensor_model_parallel_size: 4   # TP
      pipeline_model_parallel_size: 2  # PP
      expert_model_parallel_size: 8    # EP (MoE only)
      sequence_parallel: true          # SP
      # DP is implicit: total_gpus / (TP * PP * EP)
```

### Key Parameters

| Parameter | Description |
|-----------|-------------|
| `tensor_model_parallel_size` | Split attention/FFN across GPUs |
| `pipeline_model_parallel_size` | Split layers into pipeline stages |
| `expert_model_parallel_size` | Split MoE experts across GPUs |
| `sequence_parallel` | Enable sequence parallelism |
| `recompute_granularity` | Activation checkpointing level |

### Example: 30B MoE Model on 64 GPUs

```yaml
trainer:
  strategy: megatron
  placement:
    policy_num_nodes: 8
    policy_num_gpus_per_node: 8
  policy:
    megatron_config:
      tensor_model_parallel_size: 4
      pipeline_model_parallel_size: 2
      expert_model_parallel_size: 8
      sequence_parallel: true
      recompute_granularity: "selective"
      transformer_config_kwargs:
        recompute_modules: ["ffn"]
```

---

## 3. Performance Benchmarks

### Dense Models (Search-R1 Task)

| Model | Strategy | GPUs | Time/Step | Throughput |
|-------|----------|------|-----------|------------|
| 3B | FSDP2 | 8×H100 | 45s | 1.0x |
| 3B | Megatron | 8×H100 | 43s | 1.05x |
| 7B | FSDP2 | 8×H100 | 82s | 1.0x |
| 7B | Megatron | 8×H100 | 78s | 1.05x |
| 30B | FSDP2 | 32×H100 | 156s | 1.0x |
| 30B | Megatron | 32×H100 | 132s | 1.18x |

### MoE Models

| Model | Strategy | Config | Time/Step |
|-------|----------|--------|-----------|
| DeepSeek-V3 | Megatron | TP4×EP8 | 89s |
| Mixtral-8x7B | Megatron | TP2×EP4 | 67s |

**Note:** MoE models require Megatron for efficient training.

---

## 4. Migration Guide

### FSDP2 to Megatron

1. **Update strategy:**
   ```yaml
   trainer:
     strategy: megatron  # was: fsdp2
   ```

2. **Add Megatron config:**
   ```yaml
   trainer:
     policy:
       megatron_config:
         tensor_model_parallel_size: 2
         pipeline_model_parallel_size: 1
   ```

3. **Update inference backend:**
   ```yaml
   generator:
     backend: vllm  # Megatron works with vLLM or SGLang
   ```

4. **Checkpoint conversion:**
   - Megatron uses different checkpoint format
   - Cannot directly resume FSDP2 checkpoints in Megatron
   - Convert HuggingFace → Megatron format if needed

### Megatron to FSDP2

1. **Export to HuggingFace format:**
   ```yaml
   trainer:
     hf_save_interval: 10  # Save HF format periodically
   ```

2. **Update strategy:**
   ```yaml
   trainer:
     strategy: fsdp2
     policy:
       fsdp_config:
         cpu_offload: false
         reshard_after_forward: true
   ```

3. **Remove Megatron config**

---

## 5. Troubleshooting

### FSDP2 Out of Memory

```yaml
# Enable all memory optimizations
trainer:
  strategy: fsdp2
  gradient_checkpointing: true
  policy:
    fsdp_config:
      cpu_offload: true
      reshard_after_forward: true
  micro_train_batch_size_per_gpu: 1
```

### Megatron Initialization Slow

```yaml
# Pre-compile kernels
trainer:
  policy:
    megatron_config:
      empty_cuda_cache: false  # Don't clear cache between steps
```

### Megatron NCCL Timeout

```bash
# Increase timeout for large models
export NCCL_TIMEOUT=3600
export NCCL_DEBUG=INFO
```

### FSDP2 Gradient NaN

```yaml
# Stricter gradient clipping
trainer:
  policy:
    optimizer_config:
      max_grad_norm: 0.5  # Reduce from 1.0
```

---

## 6. Feature Support Matrix

| Feature | FSDP2 | Megatron |
|---------|-------|----------|
| LoRA Fine-tuning | Yes | Yes |
| Full Fine-tuning | Yes | Yes |
| Gradient Checkpointing | Yes | Yes (selective) |
| CPU Offloading | Yes | Yes |
| Sequence Parallelism | Ulysses | Native |
| Expert Parallelism | No | Yes |
| Pipeline Parallelism | No | Yes |
| Tensor Parallelism | No | Yes |
| HF Checkpoint Format | Native | Requires conversion |
| Custom Models | Easy | Harder |

---

## 7. Recommended Configurations

### Quick Start (Any Model <14B)

```yaml
trainer:
  strategy: fsdp2
  gradient_checkpointing: true
  policy:
    fsdp_config:
      cpu_offload: false
      reshard_after_forward: true
```

### Production (Dense 14B-70B)

```yaml
trainer:
  strategy: megatron
  policy:
    megatron_config:
      tensor_model_parallel_size: 4
      pipeline_model_parallel_size: 2
      sequence_parallel: true
      recompute_granularity: "selective"
```

### MoE Models

```yaml
trainer:
  strategy: megatron
  policy:
    megatron_config:
      tensor_model_parallel_size: 4
      expert_model_parallel_size: 8
      sequence_parallel: true
```

---

## References

- [Megatron Example](../examples/megatron/)
- [FSDP Configuration](./SGLANG_INTEGRATION_GUIDE.md#4-configuration-reference)
- [Batch Sizes Guide](./BATCH_SIZES.md)
- [Troubleshooting](./TROUBLESHOOTING.md)
