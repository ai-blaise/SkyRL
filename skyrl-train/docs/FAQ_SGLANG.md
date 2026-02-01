# SGLang + SkyRL FAQ

**Frequently asked questions about using SGLang with SkyRL for RL training.**

---

## General Questions

### Q: What's the difference between SGLang and vLLM in SkyRL?

Both are inference backends for generating trajectories during RL training:

| Aspect | SGLang | vLLM |
|--------|--------|------|
| **Prefix Caching** | RadixAttention (tree-based) | PagedAttention |
| **Multi-turn** | Required (`use_conversation_multi_turn=true`) | Optional |
| **Weight Sync** | Native distributed API | Worker extension |
| **Sleep/Wake** | All-or-nothing | Configurable levels |
| **Attention** | FlashInfer, FA3 | FlashAttention, FlashInfer |

**When to use SGLang:**
- Multi-turn agent training (better prefix caching)
- Need FlashInfer for GPU compatibility
- Want native distributed weight sync

**When to use vLLM:**
- Already have vLLM experience
- Need fine-grained sleep levels
- Using vLLM-specific features

---

### Q: Can I use a remote SGLang server?

Yes, but with limitations. Use `generator.run_engines_locally=false` and configure:

```yaml
generator:
  run_engines_locally: false
  remote_inference_engine_urls:
    - "http://10.0.0.1:8000"
    - "http://10.0.0.2:8000"
```

**Limitation:** Weight synchronization won't work with remote servers. Use for inference-only or with manual weight loading.

---

### Q: How much GPU memory does SGLang need?

Rule of thumb:
- **0.5B model:** ~4GB
- **1.5B model:** ~8GB
- **7B model:** ~16GB (or TP=2 with 2x 12GB GPUs)
- **70B model:** TP=8 across 8x 80GB GPUs

The `gpu_memory_utilization` setting (default 0.8) controls KV cache allocation. Lower it if OOM:

```yaml
generator:
  gpu_memory_utilization: 0.6
```

---

## Installation Issues

### Q: "ModuleNotFoundError: No module named 'sglang'"

**Cause:** SGLang not installed or wrong environment.

**Fix:**
```bash
# Reinstall with sglang extra
cd SkyRL/skyrl-train
uv sync --extra sglang
source .venv/bin/activate

# Verify
python -c "import sglang; print(sglang.__version__)"
```

---

### Q: "Failed to generate package metadata for sglang @ editable+..."

**Cause:** Ray's uv hook tries to replicate editable install paths that don't exist in Ray workers.

**Fix:**
```bash
unset RAY_RUNTIME_ENV_HOOK
python -m skyrl_train.entrypoints.main_base ...
```

---

### Q: "FlashAttention v3 Backend requires SM>=80 and SM<=90"

**Cause:** Your GPU doesn't support FlashAttention v3 (needs A100/H100).

**Fix:** Use FlashInfer backend:
```bash
python -m skyrl_train.entrypoints.main_base \
  +generator.engine_init_kwargs.attention_backend=flashinfer \
  +generator.engine_init_kwargs.mm_attention_backend=flashinfer
```

---

### Q: "ImportError: cannot import name 'Engine' from 'sglang'"

**Cause:** Old SGLang version or incorrect import path.

**Fix:**
```bash
# Update SGLang
pip install --upgrade sglang

# Or if using editable install
cd /path/to/sglang/python
pip install -e .
```

---

## Configuration Issues

### Q: "use_conversation_multi_turn=False is not supported"

**Cause:** SGLang backend requires multi-turn mode.

**Fix:** Don't set `use_conversation_multi_turn=false`. The default is `true` which is correct.

---

### Q: "tokenizer is required for SGLangInferenceEngine"

**Cause:** Tokenizer not passed to engine.

**Fix:** This should be automatic. If you're using custom code, ensure tokenizer is passed:
```python
engine = SGLangInferenceEngine(
    tokenizer=tokenizer,  # Required!
    model_path="...",
)
```

---

### Q: How do I enable LoRA with SGLang?

```yaml
generator:
  backend: sglang
  enable_lora: true
  engine_init_kwargs:
    max_lora_rank: 64
    max_loras_per_batch: 8
    lora_backend: "csgmv"
```

Then load adapters at runtime:
```python
await engine.load_lora_adapter(
    lora_name="my_adapter",
    lora_path="/path/to/lora"
)
```

---

## Weight Synchronization Issues

### Q: "Weight sync timed out"

**Cause:** Large model or slow network.

**Fix:**
```yaml
generator:
  weight_sync_timeout: 120  # seconds
```

---

### Q: "Group weight_update_group not in ['skyrl']"

**Cause:** Group name mismatch in weight sync (this was a bug, now fixed).

**Fix:** Update to latest SkyRL code with the fix:
```bash
git pull origin main
```

---

### Q: Weight sync takes too long (>10s)

**Causes & Fixes:**

1. **Not using CUDA IPC:** Set `colocate_all=true`
   ```yaml
   trainer:
     placement:
       colocate_all: true
   generator:
     weight_sync_backend: nccl
   ```

2. **Large model:** Expected for 70B+ models. Consider:
   - Reducing sync frequency (train multiple steps per sync)
   - Using LoRA (smaller weight updates)

3. **Broadcast strategy:** Falls back to sequential updates. Use IPC when possible.

---

### Q: "NCCL error" during weight sync

**Causes:**
- Firewall blocking NCCL ports
- Network timeout
- Mismatched NCCL versions

**Fixes:**
```bash
# Check NCCL version
python -c "import torch; print(torch.cuda.nccl.version())"

# Set longer timeout
export NCCL_TIMEOUT=1800

# Debug NCCL
export NCCL_DEBUG=INFO
```

---

## Training Issues

### Q: Training is slow (many seconds per step)

**Debugging steps:**

1. **Check generation time** - Is SGLang slow?
   - Increase `gpu_memory_utilization`
   - Enable prefix caching
   - Use more inference engines

2. **Check weight sync time** - Is sync slow?
   - Use `colocate_all=true` for CUDA IPC
   - Consider LoRA for smaller updates

3. **Check training time** - Is backprop slow?
   - Enable gradient checkpointing
   - Use smaller batch size

---

### Q: Loss is NaN or training diverges

**Fixes:**
```yaml
trainer:
  policy:
    optimizer_config:
      lr: 1.0e-7  # Lower learning rate
      max_grad_norm: 0.5  # Stricter gradient clipping

  algorithm:
    kl_loss_coef: 0.01  # Stronger KL penalty
    eps_clip_low: 0.1   # Tighter PPO clipping
    eps_clip_high: 0.1
```

---

### Q: Model outputs are repetitive or degenerate

**Fixes:**
```yaml
generator:
  sampling_params:
    temperature: 1.0        # Increase exploration
    repetition_penalty: 1.1 # Penalize repetition
    top_p: 0.95             # Nucleus sampling
```

---

## Memory Issues

### Q: "CUDA out of memory" during training

**Fixes:**
```yaml
trainer:
  micro_train_batch_size_per_gpu: 1  # Smaller micro-batch
  gradient_checkpointing: true        # Trade compute for memory

  policy:
    fsdp_config:
      cpu_offload: true  # Offload optimizer to CPU

generator:
  gpu_memory_utilization: 0.6  # Less memory for KV cache
```

---

### Q: "CUDA out of memory" during generation

**Fixes:**
```yaml
generator:
  gpu_memory_utilization: 0.5  # Reduce KV cache
  max_num_seqs: 256            # Fewer concurrent sequences
  max_num_batched_tokens: 4096 # Smaller batches
```

---

### Q: How does sleep/wake work with SGLang?

SGLang releases **all GPU memory** on sleep (unlike vLLM which has levels):

```python
# Sleep - releases everything
await engine.sleep()

# Wake - must re-sync weights!
await engine.wake_up()
await engine.update_named_weights(weights)  # Required!
```

**When to use:** With `colocate_all=true` to share GPUs between training and inference.

---

## Multi-GPU / Distributed Issues

### Q: How do I use tensor parallelism?

```yaml
generator:
  inference_engine_tensor_parallel_size: 2  # Split model across 2 GPUs
  num_inference_engines: 2  # 2 engines × 2 GPUs = 4 GPUs total
```

---

### Q: How do I use pipeline parallelism?

```yaml
generator:
  inference_engine_tensor_parallel_size: 2
  inference_engine_pipeline_parallel_size: 2  # 2 TP × 2 PP = 4 GPUs/engine
```

---

### Q: Multi-node training setup?

1. Start Ray cluster:
```bash
# Head node
ray start --head --port=6379

# Worker nodes
ray start --address=<head-ip>:6379
```

2. Configure SkyRL:
```yaml
trainer:
  placement:
    colocate_all: false  # Must be false for multi-node
    policy_num_nodes: 2
```

3. Use broadcast weight sync (IPC doesn't work across nodes):
```yaml
generator:
  weight_sync_backend: nccl
```

---

## Debugging

### Q: How do I enable debug logging?

```bash
export SGLANG_LOG_LEVEL=debug
export SKYRL_LOG_LEVEL=debug
export NCCL_DEBUG=INFO

python -m skyrl_train.entrypoints.main_base ...
```

---

### Q: Where are the logs?

- **Ray logs:** `/tmp/ray/session_latest/logs/`
- **Worker logs:** `/tmp/ray/session_latest/logs/worker-*.out`
- **Hydra logs:** `outputs/<date>/<time>/`

---

### Q: How do I profile performance?

```python
# In training code
engine.start_profile()
# ... run inference ...
engine.stop_profile()  # Saves to /tmp/sglang_profile/
```

Or use Ray dashboard at `http://localhost:8265`

---

## Migration from vLLM

### Q: How do I switch from vLLM to SGLang?

1. **Change backend:**
```yaml
generator:
  backend: sglang  # Was: vllm
```

2. **Update attention backend** (if needed):
```yaml
generator:
  engine_init_kwargs:
    attention_backend: flashinfer  # SGLang-specific
```

3. **Remove vLLM-specific settings:**
```yaml
# Remove these (vLLM-only):
# vllm_v1_disable_multiproc: true
# enforce_eager: true
```

4. **Add SGLang requirements:**
```yaml
generator:
  use_conversation_multi_turn: true  # Required for SGLang
```

---

## Links

- [Quickstart Guide](./QUICKSTART_SGLANG.md) - Get started fast
- [Full Integration Guide](./SGLANG_INTEGRATION_GUIDE.md) - Complete reference
- [Limitations](./SGLANG_LIMITATIONS.md) - Known constraints
- [SGLang Docs](https://docs.sglang.io/) - Official SGLang documentation
