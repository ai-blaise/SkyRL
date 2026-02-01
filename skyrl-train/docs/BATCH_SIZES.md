# Understanding Batch Sizes in SkyRL

**Complete guide to batch size configuration and how they interact.**

---

## Quick Reference

```
train_batch_size (prompts per step)
        │
        │ × n_samples_per_prompt
        ▼
Total Samples = train_batch_size × n_samples_per_prompt
        │
        │ ÷ policy_mini_batch_size
        ▼
Mini-batch iterations per step
        │
        │ ÷ (num_gpus × dp_size)
        ▼
micro_train_batch_size_per_gpu (gradient accumulation)
```

---

## 1. The Three Batch Size Parameters

### 1.1 train_batch_size

**What it is:** Number of unique prompts loaded per training step.

```yaml
trainer:
  train_batch_size: 1024  # Load 1024 unique prompts per step
```

**Actual samples per step:**
```
actual_samples = train_batch_size × n_samples_per_prompt
```

**Example:**
- `train_batch_size: 1024`, `n_samples_per_prompt: 5`
- Actual samples = 1024 × 5 = 5120 samples per step

---

### 1.2 policy_mini_batch_size

**What it is:** Number of samples per optimizer update.

```yaml
trainer:
  policy_mini_batch_size: 256  # 256 samples per gradient update
```

**Mini-batch iterations per step:**
```
iterations = (train_batch_size × n_samples_per_prompt) / policy_mini_batch_size
```

**Example:**
- Total samples: 5120
- `policy_mini_batch_size: 256`
- Iterations = 5120 / 256 = 20 optimizer steps per training step

---

### 1.3 micro_train_batch_size_per_gpu

**What it is:** Samples per GPU per forward pass (for gradient accumulation).

```yaml
trainer:
  micro_train_batch_size_per_gpu: 1  # Process 1 sample at a time per GPU
```

**Gradient accumulation steps per mini-batch:**
```
accumulation_steps = policy_mini_batch_size / (micro_train_batch_size_per_gpu × num_gpus)
```

**Example:**
- `policy_mini_batch_size: 256`
- `micro_train_batch_size_per_gpu: 1`
- 4 GPUs in data parallel
- Accumulation steps = 256 / (1 × 4) = 64 forward passes before optimizer step

---

## 2. Complete Example

### Configuration
```yaml
trainer:
  train_batch_size: 1024
  policy_mini_batch_size: 256
  micro_train_batch_size_per_gpu: 1
  update_epochs_per_batch: 1
  placement:
    policy_num_gpus_per_node: 4

generator:
  n_samples_per_prompt: 5
```

### Training Flow

```
Step 1: Load Prompts
├── Load 1024 prompts from dataset
└── Each prompt repeated 5 times → 5120 total samples

Step 2: Generation
├── Generate responses for all 5120 samples
└── Compute rewards via environment

Step 3: Advantage Computation
└── GRPO/GAE advantage estimation on 5120 samples

Step 4: Policy Updates (20 mini-batch iterations)
├── Mini-batch 1: samples 0-255
│   ├── 64 forward passes (gradient accumulation)
│   │   ├── GPU 0: sample 0, 4, 8, ... (micro-batch)
│   │   ├── GPU 1: sample 1, 5, 9, ...
│   │   ├── GPU 2: sample 2, 6, 10, ...
│   │   └── GPU 3: sample 3, 7, 11, ...
│   └── Optimizer step (after 64 accumulations)
├── Mini-batch 2: samples 256-511
│   └── ... (same pattern)
└── ... (18 more mini-batches)

Step 5: Weight Sync
└── Broadcast new weights to inference engines

→ Repeat for next training step
```

---

## 3. Memory vs Speed Tradeoffs

### More GPU Memory Available

Increase `micro_train_batch_size_per_gpu`:
```yaml
trainer:
  micro_train_batch_size_per_gpu: 4  # Process 4 samples per GPU
```

**Effect:**
- Fewer gradient accumulation steps
- Faster training (fewer forward passes)
- More GPU memory usage

### Less GPU Memory Available

Decrease `micro_train_batch_size_per_gpu`:
```yaml
trainer:
  micro_train_batch_size_per_gpu: 1  # Minimum setting
```

**Effect:**
- More gradient accumulation steps
- Slower training (more forward passes)
- Less GPU memory usage

---

## 4. Constraints and Validation

### Divisibility Requirements

```python
# These must be evenly divisible:
assert train_batch_size % policy_mini_batch_size == 0
assert (train_batch_size * n_samples_per_prompt) % policy_mini_batch_size == 0
assert policy_mini_batch_size % (micro_train_batch_size_per_gpu * num_dp_gpus) == 0
```

### Valid Configurations

| train_batch_size | n_samples | mini_batch | micro_batch | GPUs | Valid? |
|------------------|-----------|------------|-------------|------|--------|
| 1024 | 5 | 256 | 1 | 4 | Yes |
| 1024 | 5 | 256 | 2 | 4 | Yes |
| 1024 | 5 | 256 | 8 | 4 | No (32 not divisible by 256) |
| 512 | 8 | 256 | 1 | 8 | Yes |
| 100 | 5 | 256 | 1 | 4 | No (500 not divisible by 256) |

### Invalid Configuration Example

```yaml
# BAD: 1024 * 5 = 5120, not divisible by 300
trainer:
  train_batch_size: 1024
  policy_mini_batch_size: 300  # ERROR!
```

**Error:**
```
ValueError: train_batch_size * n_samples_per_prompt (5120) must be
divisible by policy_mini_batch_size (300)
```

---

## 5. Relationship with Critic Training

The critic has its own mini-batch size:

```yaml
trainer:
  policy_mini_batch_size: 256   # For policy updates
  critic_mini_batch_size: 256   # For critic updates (if using GAE)
```

**When using GAE (with critic):**
- Critic trains first (computes values)
- Policy trains second (uses critic values for advantages)
- Both use the same `train_batch_size` but can have different `mini_batch_size`

---

## 6. update_epochs_per_batch

**What it is:** Number of passes over each batch before loading new data.

```yaml
trainer:
  update_epochs_per_batch: 1  # Single pass (standard)
  # update_epochs_per_batch: 4  # Four passes (PPO-style)
```

**Effect:**
```
update_epochs_per_batch: 1
├── Load batch, train once, load next batch

update_epochs_per_batch: 4
├── Load batch
├── Train pass 1 (20 mini-batches)
├── Train pass 2 (20 mini-batches, same data)
├── Train pass 3 (20 mini-batches, same data)
├── Train pass 4 (20 mini-batches, same data)
└── Load next batch
```

**When to increase:**
- More sample efficiency (reuse expensive generations)
- Risk of overfitting to generated samples
- Traditional PPO uses 4 update epochs

---

## 7. Evaluation Batch Size

Separate from training:

```yaml
trainer:
  eval_batch_size: 1024  # Prompts per evaluation

generator:
  eval_n_samples_per_prompt: 1  # Typically 1 for deterministic eval
```

---

## 8. Common Configurations

### Small Model (0.5B-3B) on Single Node (4 GPUs)

```yaml
trainer:
  train_batch_size: 512
  policy_mini_batch_size: 128
  micro_train_batch_size_per_gpu: 2
  placement:
    policy_num_gpus_per_node: 4

generator:
  n_samples_per_prompt: 8
```

**Samples per step:** 512 × 8 = 4096
**Mini-batches:** 4096 / 128 = 32
**Accumulation steps:** 128 / (2 × 4) = 16

### Medium Model (7B-14B) on Single Node (8 GPUs)

```yaml
trainer:
  train_batch_size: 256
  policy_mini_batch_size: 64
  micro_train_batch_size_per_gpu: 1
  placement:
    policy_num_gpus_per_node: 8

generator:
  n_samples_per_prompt: 8
```

**Samples per step:** 256 × 8 = 2048
**Mini-batches:** 2048 / 64 = 32
**Accumulation steps:** 64 / (1 × 8) = 8

### Large Model (70B) Multi-Node (16 GPUs)

```yaml
trainer:
  train_batch_size: 128
  policy_mini_batch_size: 32
  micro_train_batch_size_per_gpu: 1
  placement:
    policy_num_nodes: 2
    policy_num_gpus_per_node: 8

generator:
  n_samples_per_prompt: 4
```

**Samples per step:** 128 × 4 = 512
**Mini-batches:** 512 / 32 = 16
**Accumulation steps:** 32 / (1 × 16) = 2

---

## 9. Troubleshooting

### Out of Memory During Training

**Solutions:**
1. Reduce `micro_train_batch_size_per_gpu`:
   ```yaml
   micro_train_batch_size_per_gpu: 1  # Minimum
   ```

2. Enable gradient checkpointing:
   ```yaml
   trainer:
     gradient_checkpointing: true
   ```

3. Reduce `policy_mini_batch_size`:
   ```yaml
   policy_mini_batch_size: 64  # Smaller mini-batches
   ```

### Training Too Slow

**Solutions:**
1. Increase `micro_train_batch_size_per_gpu`:
   ```yaml
   micro_train_batch_size_per_gpu: 4  # If memory allows
   ```

2. Reduce `n_samples_per_prompt`:
   ```yaml
   generator:
     n_samples_per_prompt: 4  # Fewer generations per prompt
   ```

3. Use larger `policy_mini_batch_size`:
   ```yaml
   policy_mini_batch_size: 512  # Fewer optimizer steps
   ```

### Validation Error on Batch Sizes

**Fix:** Ensure all divisibility constraints are met:
```python
# Check your config:
train_batch_size = 1024
n_samples = 5
mini_batch = 256

total_samples = train_batch_size * n_samples  # 5120
assert total_samples % mini_batch == 0, f"{total_samples} not divisible by {mini_batch}"
```

---

## 10. Summary Table

| Parameter | Default | Purpose | Memory Impact |
|-----------|---------|---------|---------------|
| `train_batch_size` | 1024 | Prompts per training step | Low |
| `n_samples_per_prompt` | 5 | Responses per prompt | Medium (generation) |
| `policy_mini_batch_size` | 256 | Samples per optimizer step | None |
| `micro_train_batch_size_per_gpu` | 1 | Samples per forward pass | High |
| `update_epochs_per_batch` | 1 | Passes over each batch | None |

---

## References

- [Configuration Reference](./SGLANG_INTEGRATION_GUIDE.md)
- [Algorithms Guide](./ALGORITHMS.md) - How batch sizes affect different algorithms
- [Troubleshooting](./TROUBLESHOOTING.md) - Memory issues
