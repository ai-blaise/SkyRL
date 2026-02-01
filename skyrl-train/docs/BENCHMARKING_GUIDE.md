# Benchmarking and Performance Tuning Guide

**Comprehensive guide for measuring and optimizing SkyRL + SGLang training performance.**

---

## Table of Contents

1. [Performance Metrics Overview](#1-performance-metrics-overview)
2. [Profiling Your Training](#2-profiling-your-training)
3. [Generation Performance](#3-generation-performance)
4. [Training Performance](#4-training-performance)
5. [Weight Synchronization](#5-weight-synchronization)
6. [Memory Optimization](#6-memory-optimization)
7. [Multi-GPU Scaling](#7-multi-gpu-scaling)
8. [Common Bottlenecks](#8-common-bottlenecks)
9. [Benchmark Scripts](#9-benchmark-scripts)
10. [Reference Performance Numbers](#10-reference-performance-numbers)

---

## 1. Performance Metrics Overview

### Key Metrics to Track

| Metric | Target | Impact |
|--------|--------|--------|
| **Tokens/second (generation)** | Model-dependent | Higher = faster rollouts |
| **Training throughput** | Samples/hour | Higher = faster convergence |
| **Weight sync time** | < 2s (colocated) | Lower = less idle time |
| **GPU utilization** | > 80% | Higher = better efficiency |
| **Memory usage** | < 90% peak | Lower = larger batches possible |
| **Time to first token** | < 100ms | Lower = better latency |

### Performance Breakdown (Typical)

```
Training Step Timeline
├── Generation (rollouts)     40-60%
├── Training (gradient)        20-30%
├── Weight sync                 5-15%
├── Data loading                2-5%
└── Other overhead              5-10%
```

---

## 2. Profiling Your Training

### 2.1 Enable Logging

```bash
# Comprehensive logging
export SGLANG_LOG_LEVEL=debug
export SKYRL_LOG_LEVEL=debug
export NCCL_DEBUG=INFO

python -m skyrl_train.entrypoints.main_base \
  trainer.policy.record_memory=true \
  trainer.dump_data_batch=true \
  ...
```

### 2.2 Built-in Timing

```python
import time
from skyrl_train.utils.timing import Timer

# Use context manager
with Timer("generation"):
    outputs = await generator.generate(batch)

# Or manual
timer = Timer("training_step")
timer.start()
# ... training code ...
timer.stop()
print(timer.summary())
```

### 2.3 SGLang Profiling

```bash
# Enable SGLang profiler
export SGLANG_ENABLE_PROFILER=1

# Profiling output goes to /tmp/sglang_profile/
python -m skyrl_train.entrypoints.main_base ...

# Analyze results
python -c "
import json
with open('/tmp/sglang_profile/profile.json') as f:
    data = json.load(f)
    for event in data['traceEvents'][:10]:
        print(event)
"
```

### 2.4 PyTorch Profiler

```python
import torch.profiler as profiler

with profiler.profile(
    activities=[
        profiler.ProfilerActivity.CPU,
        profiler.ProfilerActivity.CUDA,
    ],
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
) as prof:
    # Run one training step
    trainer.step(batch)

# Export
prof.export_chrome_trace("trace.json")
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```

### 2.5 CUDA Memory Tracking

```python
import torch

# Enable memory tracking
torch.cuda.memory._record_memory_history()

# Run your code
...

# Dump memory snapshot
torch.cuda.memory._dump_snapshot("memory_snapshot.pickle")

# Analyze
snapshot = torch.cuda.memory._load_snapshot("memory_snapshot.pickle")
```

---

## 3. Generation Performance

### 3.1 Throughput Measurement

```python
import time
import asyncio

async def benchmark_generation(engine, prompts, sampling_params, warmup=5, iterations=20):
    """Benchmark generation throughput."""

    # Warmup
    for _ in range(warmup):
        await engine.generate(prompts[:10], sampling_params)

    # Measure
    total_tokens = 0
    start = time.time()

    for _ in range(iterations):
        outputs = await engine.generate(prompts, sampling_params)
        total_tokens += sum(len(o.output_ids) for o in outputs)

    elapsed = time.time() - start

    return {
        "tokens_per_second": total_tokens / elapsed,
        "prompts_per_second": len(prompts) * iterations / elapsed,
        "avg_latency_ms": elapsed * 1000 / iterations,
    }

# Usage
results = await benchmark_generation(engine, prompts, {"max_new_tokens": 512})
print(f"Throughput: {results['tokens_per_second']:.1f} tokens/sec")
```

### 3.2 Generation Optimization Strategies

#### Increase GPU Memory for KV Cache

```yaml
generator:
  gpu_memory_utilization: 0.85  # Default is 0.9
```

**Impact:** More KV cache = more concurrent sequences

#### Enable Prefix Caching

```yaml
generator:
  enable_prefix_caching: true
```

**Impact:** Reuse cached prefixes, especially for similar prompts

#### Optimize Attention Backend

```yaml
# For H100
generator:
  engine_init_kwargs:
    attention_backend: "fa3"

# For other GPUs
generator:
  engine_init_kwargs:
    attention_backend: "flashinfer"
```

**Impact:** 10-30% throughput improvement

#### Batch Size Tuning

```yaml
generator:
  max_num_seqs: 512        # Concurrent sequences
  max_num_batched_tokens: 32768  # Tokens per batch
```

**Trade-off:** Larger = higher throughput, but more memory

### 3.3 Generation Performance Reference

| Model | GPU | Backend | Tokens/sec | Config |
|-------|-----|---------|------------|--------|
| Qwen2.5-0.5B | H100 | fa3 | ~15,000 | `max_num_seqs=512` |
| Qwen2.5-7B | H100 | fa3 | ~4,000 | `max_num_seqs=256` |
| Qwen2.5-32B | 4xH100 | fa3 | ~2,000 | TP=4, `max_num_seqs=128` |
| Llama-3-8B | A100 | flashinfer | ~3,500 | `max_num_seqs=256` |

---

## 4. Training Performance

### 4.1 Training Throughput Measurement

```python
import time

def measure_training_throughput(trainer, dataloader, steps=100):
    """Measure training samples per second."""

    total_samples = 0
    start = time.time()

    for i, batch in enumerate(dataloader):
        if i >= steps:
            break
        trainer.step(batch)
        total_samples += len(batch)

    elapsed = time.time() - start

    return {
        "samples_per_second": total_samples / elapsed,
        "seconds_per_step": elapsed / steps,
        "samples_per_hour": total_samples / elapsed * 3600,
    }
```

### 4.2 Training Optimization Strategies

#### Gradient Checkpointing

```yaml
trainer:
  gradient_checkpointing: true
```

**Impact:** 30-50% memory reduction, 10-20% slower

#### Mixed Precision

```yaml
trainer:
  mixed_precision: "bf16"  # or "fp16"
```

**Impact:** ~2x memory reduction, faster compute

#### Batch Size Scaling

```yaml
trainer:
  micro_train_batch_size_per_gpu: 4
  gradient_accumulation_steps: 8
```

**Effective batch size:** `4 * 8 * num_gpus`

#### Optimizer CPU Offload

```yaml
trainer:
  policy:
    fsdp_config:
      cpu_offload: true
```

**Impact:** Reduces GPU memory, slower updates

### 4.3 FSDP Tuning

```yaml
trainer:
  policy:
    fsdp_config:
      sharding_strategy: "FULL_SHARD"  # or "SHARD_GRAD_OP", "NO_SHARD"
      backward_prefetch: "BACKWARD_PRE"  # or "BACKWARD_POST"
      forward_prefetch: true
      limit_all_gathers: true
```

| Strategy | Memory | Speed | Use Case |
|----------|--------|-------|----------|
| `FULL_SHARD` | Lowest | Moderate | Large models |
| `SHARD_GRAD_OP` | Medium | Fast | Medium models |
| `NO_SHARD` | Highest | Fastest | Small models |

---

## 5. Weight Synchronization

### 5.1 Sync Time Measurement

```python
import time

def measure_weight_sync(trainer, engine, iterations=10):
    """Measure weight synchronization time."""

    times = []
    for _ in range(iterations):
        start = time.time()
        trainer.sync_weights_to_engine(engine)
        elapsed = time.time() - start
        times.append(elapsed)

    return {
        "mean_ms": sum(times) / len(times) * 1000,
        "min_ms": min(times) * 1000,
        "max_ms": max(times) * 1000,
    }
```

### 5.2 Sync Optimization Strategies

#### Use CUDA IPC (Same Node)

```yaml
trainer:
  placement:
    colocate_all: true  # Enable CUDA IPC
generator:
  weight_sync_backend: nccl
```

**Impact:** 10-100x faster than network sync

#### Sync Only Changed Parameters

```yaml
trainer:
  weight_sync:
    sync_full_model: false  # Sync only LoRA
```

**Impact:** Faster for LoRA fine-tuning

#### Increase Timeout for Large Models

```yaml
generator:
  weight_sync_timeout: 120  # seconds
```

### 5.3 Weight Sync Benchmark Script

Use the benchmark script to measure weight sync latency for your setup:

```bash
# Benchmark broadcast strategy for 7B model
python scripts/benchmark_weight_sync.py --model-size 7B --strategy broadcast

# Compare all strategies
python scripts/benchmark_weight_sync.py --model-size 7B --compare-all --iterations 10

# Benchmark different model sizes
python scripts/benchmark_weight_sync.py --model-size 0.5B --strategy cuda_ipc
python scripts/benchmark_weight_sync.py --model-size 70B --strategy broadcast
```

**Available Strategies:**
- `cuda_ipc`: CUDA IPC handles, fastest for same-node (colocated)
- `broadcast`: NCCL/Gloo broadcast, works cross-node
- `checkpoint_engine`: ParameterServer-based, optimized for large scale

### 5.4 Weight Sync Reference Times

| Model Size | Method | Time | Config |
|------------|--------|------|--------|
| 0.5B | CUDA IPC | ~100ms | Colocated |
| 7B | CUDA IPC | ~500ms | Colocated |
| 7B | NCCL broadcast | ~2s | Distributed |
| 32B | CUDA IPC | ~2s | Colocated, TP=4 |
| 70B | NCCL broadcast | ~10s | Distributed, TP=8 |

---

## 6. Memory Optimization

### 6.1 Memory Budget Planning

```
Total GPU Memory = Model + KV Cache + Gradients + Optimizer + Activations

Example for 7B model on 80GB H100:
- Model weights (bf16):     14 GB
- KV cache:                 20 GB
- Gradients:                14 GB
- Optimizer states:         28 GB
- Activations:              10 GB
- Buffer/fragmentation:      5 GB
Total:                      91 GB (OOM!)
```

### 6.2 Memory Reduction Strategies

| Strategy | Memory Savings | Performance Impact |
|----------|----------------|-------------------|
| Gradient checkpointing | 30-50% | 10-20% slower |
| CPU offload | 20-30% | 30-50% slower |
| Lower precision (fp16/bf16) | 50% | Minimal |
| LoRA instead of full FT | 90%+ | Similar quality |
| Reduce batch size | Linear | Lower throughput |
| Reduce KV cache | Variable | Fewer concurrent seqs |

### 6.3 Memory Monitoring

```python
import torch

def log_memory_usage():
    """Log current GPU memory usage."""
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / 1e9
        reserved = torch.cuda.memory_reserved(i) / 1e9
        total = torch.cuda.get_device_properties(i).total_memory / 1e9
        print(f"GPU {i}: {allocated:.1f}GB allocated, {reserved:.1f}GB reserved, {total:.1f}GB total")

# Call during training
log_memory_usage()
```

### 6.4 Sleep/Wake for Memory Efficiency

```yaml
# Enable sleep/wake to share memory between training and inference
trainer:
  use_sleep_wake: true
generator:
  enable_sleep: true
```

**How it works:**
1. During training: Inference engine "sleeps" (releases memory)
2. Trainer uses freed memory for gradients
3. After training: Engine "wakes" (reclaims memory)
4. Generation proceeds with updated weights

---

## 7. Multi-GPU Scaling

### 7.1 Scaling Efficiency

```
Ideal: N GPUs = N× throughput
Reality: N GPUs ≈ 0.7-0.9N× throughput (due to communication)
```

### 7.2 Scaling Strategies

#### Data Parallelism

```yaml
trainer:
  strategy: fsdp
  num_gpus: 8
```

- Each GPU processes different data
- Best for: Most training scenarios

#### Tensor Parallelism (Inference)

```yaml
generator:
  inference_engine_tensor_parallel_size: 4
```

- Model split across GPUs
- Best for: Large models that don't fit on single GPU

#### Pipeline Parallelism

```yaml
trainer:
  strategy: megatron
  pipeline_parallel_size: 2
```

- Model layers split across GPUs
- Best for: Very large models

### 7.3 Multi-Node Configuration

```yaml
# Node 0 (head)
ray start --head --port=6379

# Node 1+
ray start --address='<head-ip>:6379'

# Training config
trainer:
  num_nodes: 2
  num_gpus_per_node: 8
```

### 7.4 Scaling Reference

| GPUs | Model | Config | Throughput | Scaling |
|------|-------|--------|------------|---------|
| 1 | 7B | FSDP | 100 samples/hr | 1.0× |
| 2 | 7B | FSDP | 180 samples/hr | 0.9× |
| 4 | 7B | FSDP | 340 samples/hr | 0.85× |
| 8 | 7B | FSDP | 640 samples/hr | 0.8× |
| 8 | 70B | FSDP+TP=8 | 50 samples/hr | - |

---

## 8. Common Bottlenecks

### 8.1 Identifying Bottlenecks

```
Symptom                    → Likely Bottleneck
─────────────────────────────────────────────
GPU util < 50%             → I/O or CPU bound
GPU util fluctuating       → Weight sync or data loading
Memory spikes then OOM     → Gradient accumulation issue
Slow first step only       → Model loading/compilation
All steps equally slow     → Compute or memory bound
```

### 8.2 Bottleneck Solutions

| Bottleneck | Diagnosis | Solution |
|------------|-----------|----------|
| Data loading | CPU util high, GPU idle | Increase `num_workers`, `prefetch_factor` |
| Generation | Long time in generate() | Increase `gpu_memory_utilization`, add engines |
| Weight sync | Long sync times in logs | Use CUDA IPC, colocate training/inference |
| Training | High GPU util, slow | Reduce batch size, enable checkpointing |
| Memory | OOM errors | See Memory Optimization section |
| Network | NCCL timeouts | Check bandwidth, increase timeout |

### 8.3 Quick Diagnostics

```bash
# Check GPU utilization
watch -n 1 nvidia-smi

# Check CPU utilization
htop

# Check network bandwidth
iperf3 -c <other-node>

# Check disk I/O
iostat -x 1
```

---

## 9. Benchmark Scripts

### 9.1 End-to-End Benchmark

```python
#!/usr/bin/env python
"""Benchmark SkyRL training end-to-end."""

import asyncio
import time
from dataclasses import dataclass
from typing import Dict

@dataclass
class BenchmarkResults:
    generation_tokens_per_sec: float
    training_samples_per_sec: float
    weight_sync_ms: float
    gpu_memory_peak_gb: float
    total_time_sec: float

async def run_benchmark(config_path: str, num_steps: int = 100) -> BenchmarkResults:
    """Run benchmark with given config."""

    from skyrl_train.entrypoints.main_base import create_experiment
    import torch

    exp = create_experiment(config_path)

    # Warmup
    for _ in range(5):
        await exp.run_step()

    # Measure
    start = time.time()
    total_tokens = 0
    total_samples = 0
    sync_times = []

    for step in range(num_steps):
        step_start = time.time()

        # Generation
        gen_start = time.time()
        outputs = await exp.generator.generate(exp.get_batch())
        total_tokens += sum(len(o.output_ids) for o in outputs)
        total_samples += len(outputs)
        gen_time = time.time() - gen_start

        # Training
        exp.trainer.step(outputs)

        # Weight sync
        sync_start = time.time()
        exp.sync_weights()
        sync_times.append((time.time() - sync_start) * 1000)

    total_time = time.time() - start
    peak_memory = torch.cuda.max_memory_allocated() / 1e9

    return BenchmarkResults(
        generation_tokens_per_sec=total_tokens / total_time,
        training_samples_per_sec=total_samples / total_time,
        weight_sync_ms=sum(sync_times) / len(sync_times),
        gpu_memory_peak_gb=peak_memory,
        total_time_sec=total_time,
    )

if __name__ == "__main__":
    results = asyncio.run(run_benchmark("config.yaml"))
    print(f"Generation: {results.generation_tokens_per_sec:.0f} tokens/sec")
    print(f"Training: {results.training_samples_per_sec:.1f} samples/sec")
    print(f"Weight sync: {results.weight_sync_ms:.1f} ms")
    print(f"Peak memory: {results.gpu_memory_peak_gb:.1f} GB")
```

### 9.2 Quick Generation Benchmark

```bash
# Using SGLang directly
python -m sglang.bench_serving \
  --model Qwen/Qwen2.5-7B \
  --num-prompts 100 \
  --request-rate 10
```

### 9.3 Memory Benchmark

```python
import torch

def memory_benchmark(model_path, batch_sizes=[1, 2, 4, 8, 16]):
    """Find maximum batch size before OOM."""
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    model.cuda()

    for bs in batch_sizes:
        torch.cuda.reset_peak_memory_stats()
        try:
            x = torch.randint(0, 32000, (bs, 512)).cuda()
            with torch.no_grad():
                _ = model(x)

            peak = torch.cuda.max_memory_allocated() / 1e9
            print(f"Batch size {bs}: {peak:.1f} GB")
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"Batch size {bs}: OOM")
                break
            raise
```

### 9.4 n>1 Sampling Benchmark

Compare n>1 parallel sampling (shared prefill) vs N separate requests:

```bash
# Basic benchmark
python scripts/benchmark_n_sampling.py --model Qwen/Qwen2.5-0.5B-Instruct --n 4

# Compare multiple n values
python scripts/benchmark_n_sampling.py --model Qwen/Qwen2.5-0.5B-Instruct \
    --n 2 4 8 16 --num-prompts 10 --max-tokens 128

# With tensor parallelism
python scripts/benchmark_n_sampling.py --model meta-llama/Llama-3.2-1B-Instruct \
    --n 8 --tp-size 2
```

**Expected Results:**

| n | n>1 Sampling | N Separate | Speedup |
|---|--------------|------------|---------|
| 2 | ~100ms | ~180ms | 1.8x |
| 4 | ~150ms | ~350ms | 2.3x |
| 8 | ~250ms | ~700ms | 2.8x |
| 16 | ~400ms | ~1400ms | 3.5x |

The speedup comes from shared prefill computation - with n>1 sampling, the prompt is
processed once and sampling is done in parallel, while separate requests re-compute
the prefill for each sample.

**Key metrics:**
- `Speedup`: How much faster n>1 is vs separate requests
- `Tokens/sec`: Generation throughput
- `Time saved %`: Percentage of time saved by using n>1

---

## 10. Reference Performance Numbers

### 10.1 Baseline Configurations

All numbers measured on 8x H100-80GB, using SGLang with FlashAttention 3.

| Model | Algorithm | Samples/hr | Config |
|-------|-----------|------------|--------|
| Qwen2.5-0.5B | GRPO | 10,000 | Default |
| Qwen2.5-7B | GRPO | 1,500 | 4 inference engines |
| Qwen2.5-32B | DAPO | 400 | TP=4, 2 engines |
| Llama-3-8B | PPO | 1,200 | 4 inference engines |
| Llama-3-70B | GRPO | 150 | TP=8, 1 engine |

### 10.2 Optimization Impact

| Optimization | Speedup | Memory Reduction |
|--------------|---------|-----------------|
| Enable prefix caching | 1.2-1.5× | - |
| Increase engines 1→4 | 2-3× | 4× more memory |
| CUDA IPC vs broadcast | 10-100× sync | - |
| bf16 vs fp32 | 1.5-2× | 50% |
| Gradient checkpointing | 0.8-0.9× | 30-50% |
| LoRA (rank 32) | - | 80-90% |

### 10.3 Cost Efficiency

```
Cost per 1M training samples (approximate, cloud pricing):

Model Size    | GPU Hours | ~Cost (H100)
───────────────────────────────────────
0.5B          | 100       | $200
7B            | 700       | $1,400
32B           | 2,500     | $5,000
70B           | 7,000     | $14,000

Note: Highly dependent on task complexity,
sequence length, and number of turns.
```

---

## Related Documentation

- [SGLANG_INTEGRATION_GUIDE.md](./SGLANG_INTEGRATION_GUIDE.md) - Full integration guide
- [CONFIG_REFERENCE.md](./CONFIG_REFERENCE.md) - All configuration options
- [TROUBLESHOOTING.md](./TROUBLESHOOTING.md) - Common issues
