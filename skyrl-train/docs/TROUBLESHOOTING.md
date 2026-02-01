# Troubleshooting Guide: SGLang + SkyRL

**Complete guide to diagnosing and fixing common issues.**

---

## Quick Diagnosis

| Symptom | Likely Cause | Jump To |
|---------|--------------|---------|
| `ModuleNotFoundError: sglang` | Installation issue | [Installation Issues](#1-installation-issues) |
| `FlashAttention v3 Backend requires SM>=80` | GPU incompatibility | [GPU Compatibility](#2-gpu-compatibility) |
| `Failed to generate package metadata` | Ray + editable install | [Ray Issues](#3-ray-issues) |
| `Weight sync timed out` | Network/NCCL issue | [Weight Sync Issues](#4-weight-sync-issues) |
| `CUDA out of memory` | Memory management | [Memory Issues](#5-memory-issues) |
| `use_conversation_multi_turn=False not supported` | Config error | [Configuration Issues](#6-configuration-issues) |
| Training loss is NaN | Training instability | [Training Issues](#7-training-issues) |
| Generation is slow | Performance tuning | [Performance Issues](#8-performance-issues) |
| `Dataset not found on Hugging Face` | Data path error | [Data and Template Errors](#11-data-and-template-errors) |
| `Template file not found` | Chat template config | [Data and Template Errors](#11-data-and-template-errors) |
| `dynamic sampling limit` | DAPO sampling exhausted | [Training State Errors](#12-training-state-errors) |
| `step_wise_trajectories doesn't support` | Feature incompatibility | [Training State Errors](#12-training-state-errors) |
| `Could not find transformer layer class` | FSDP wrapping failed | [Distributed Training Errors](#13-distributed-training-errors) |
| `LoRA is not enabled` | LoRA config missing | [LoRA Errors](#14-lora-errors) |
| `wake_up timed out` / `sleep timed out` | Engine memory timeout | [Engine Lifecycle Errors](#15-engine-lifecycle-errors) |
| `Dataset has no 'prompt' field` | Missing required field | [Dataset Validation Errors](#16-dataset-validation-errors) |
| `Tokenizer has no pad_token` | Tokenizer config | [Tokenization Errors](#17-tokenization-errors) |
| `Expected all tensors on same device` | Device mismatch | [Device Placement Errors](#18-device-placement-errors) |
| `Checkpoint architecture mismatch` | Wrong checkpoint | [Checkpoint and Resume Errors](#19-checkpoint-and-resume-errors) |
| `DataLoader worker exited` | Worker OOM | [Data Loader Errors](#20-data-loader-errors) |
| `RadixCache: prefix not found` | Cache eviction | [SGLang Backend Errors](#21-sglang-backend-specific-errors) |
| `Cannot pause: engine not running` | Engine state issue | [Pause/Continue Errors](#22-pausecontinue-generation-errors) |
| `Conversation state not found` | Multi-turn state loss | [Multi-Turn Errors](#23-multi-turn-generation-errors) |
| `Reward tensor shape mismatch` | Reward format error | [Reward Computation Errors](#24-reward-computation-errors) |
| `Connection refused on port 8000` | HTTP endpoint issue | [HTTP Endpoint Errors](#25-http-endpoint-errors) |
| `DAPO requires n_samples >= 2` | Algorithm config | [Algorithm-Specific Errors](#26-algorithm-specific-errors) |
| `Tool not found` | Tool group missing | [Environment-Specific Errors](#27-environment-specific-errors) |
| `No validation data configured` | Missing val data | [Validation Errors](#28-validation-and-evaluation-errors) |

---

## 1. Installation Issues

### Error: `ModuleNotFoundError: No module named 'sglang'`

**Cause:** SGLang not installed or wrong environment.

**Solution:**
```bash
cd SkyRL/skyrl-train
uv sync --extra sglang
source .venv/bin/activate

# Verify
python -c "import sglang; print(sglang.__version__)"
```

---

### Error: `ImportError: cannot import name 'Engine' from 'sglang'`

**Cause:** Old SGLang version or incorrect import path.

**Solution:**
```bash
# Update SGLang
pip install --upgrade sglang

# Or if using editable install
cd /path/to/sglang/python
pip install -e .
```

---

### Error: `No module named 'flashinfer'`

**Cause:** FlashInfer not installed.

**Solution:**
```bash
pip install flashinfer-python
```

---

### Error: `torch.cuda.is_available() returns False`

**Cause:** PyTorch not built with CUDA or driver issue.

**Solution:**
```bash
# Check CUDA
nvidia-smi

# Reinstall PyTorch with CUDA
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

---

## 2. GPU Compatibility

### Error: `AssertionError: FlashAttention v3 Backend requires SM>=80 and SM<=90`

**Cause:** Your GPU doesn't support FlashAttention v3 (requires A100, H100, or similar).

**Solution:** Use FlashInfer backend:
```bash
python -m skyrl_train.entrypoints.main_base \
  generator.backend=sglang \
  +generator.engine_init_kwargs.attention_backend=flashinfer \
  +generator.engine_init_kwargs.mm_attention_backend=flashinfer
```

**GPU Compute Capability Reference:**
| GPU | SM | Recommended Backend |
|-----|----|--------------------|
| RTX 3090 | 86 | `flashinfer` |
| RTX 4090 | 89 | `flashinfer` |
| A100 | 80 | `fa3` or `flashinfer` |
| H100 | 90 | `fa3` |
| L4 | 89 | `flashinfer` |

---

### Error: `CUDA error: no kernel image is available for execution on the device`

**Cause:** CUDA kernels not compiled for your GPU architecture.

**Solution:**
```bash
# Rebuild FlashInfer for your GPU
pip uninstall flashinfer-python
pip install flashinfer-python --no-cache-dir
```

---

## 3. Ray Issues

### Error: `Failed to generate package metadata for 'sglang @ editable+../../sglang/python'`

**Cause:** Ray's UV runtime env hook tries to replicate editable install paths that don't exist in Ray workers.

**Solution:**
```bash
# Unset the UV hook
unset RAY_RUNTIME_ENV_HOOK

# Then run training
source .venv/bin/activate
python -m skyrl_train.entrypoints.main_base ...
```

---

### Error: `ray.exceptions.RayActorError: The actor died`

**Cause:** Actor crashed, often due to OOM or initialization failure.

**Solution:**
1. Check Ray logs:
   ```bash
   cat /tmp/ray/session_latest/logs/worker-*.out
   ```

2. Reduce memory usage:
   ```yaml
   generator:
     gpu_memory_utilization: 0.6
     max_num_seqs: 256
   ```

3. Check actor initialization:
   ```bash
   export RAY_DEDUP_LOGS=0
   python -m skyrl_train.entrypoints.main_base ...
   ```

---

### Error: `RuntimeError: Cannot connect to Ray cluster`

**Cause:** Ray not started or connection issue.

**Solution:**
```bash
# Start Ray if not running
ray start --head

# Or let SkyRL start Ray automatically
ray stop
python -m skyrl_train.entrypoints.main_base ...
```

---

## 4. Weight Sync Issues

### Error: `Weight sync timed out`

**Cause:** Large model or slow network.

**Solution:**
```yaml
generator:
  weight_sync_timeout: 120  # Increase timeout (seconds)
```

Or set environment variable:
```bash
export NCCL_TIMEOUT=1800  # 30 minutes
```

---

### Error: `AssertionError: Group weight_update_group not in ['skyrl']`

**Cause:** Group name mismatch between training and inference.

**Solution:** This was a bug that has been fixed. Update to latest SkyRL:
```bash
git pull origin main
```

---

### Error: `NCCL error: unhandled system error`

**Causes:**
- Firewall blocking NCCL ports
- Network timeout
- Mismatched NCCL versions

**Solutions:**

1. Check NCCL version:
   ```bash
   python -c "import torch; print(torch.cuda.nccl.version())"
   ```

2. Enable NCCL debug:
   ```bash
   export NCCL_DEBUG=INFO
   export NCCL_DEBUG_SUBSYS=ALL
   ```

3. Set longer timeout:
   ```bash
   export NCCL_TIMEOUT=1800
   ```

4. Check firewall:
   ```bash
   # NCCL uses ports 29400-29500 by default
   sudo ufw allow 29400:29500/tcp
   ```

---

### Error: `IPC weight update failed`

**Cause:** CUDA IPC handle sharing failed.

**Solutions:**
1. Ensure processes are on same node
2. Check CUDA driver compatibility
3. Fall back to broadcast strategy:
   ```yaml
   trainer:
     placement:
       colocate_all: false  # Disable CUDA IPC
   ```

---

### Weight sync takes >10s

**Causes & Solutions:**

1. **Not using CUDA IPC:** Enable colocated mode:
   ```yaml
   trainer:
     placement:
       colocate_all: true
   generator:
     weight_sync_backend: nccl
   ```

2. **Large model:** Expected for 70B+ models. Consider:
   - LoRA fine-tuning (smaller updates)
   - Sync less frequently

3. **Network bottleneck:** Use IPC when possible.

---

## 5. Memory Issues

### Error: `CUDA out of memory` during training

**Solutions:**

1. Reduce micro-batch size:
   ```yaml
   trainer:
     micro_train_batch_size_per_gpu: 1
   ```

2. Enable gradient checkpointing:
   ```yaml
   trainer:
     gradient_checkpointing: true
   ```

3. Offload optimizer to CPU:
   ```yaml
   trainer:
     policy:
       fsdp_config:
         cpu_offload: true
   ```

4. Reduce SGLang memory:
   ```yaml
   generator:
     gpu_memory_utilization: 0.5
   ```

---

### Error: `CUDA out of memory` during generation

**Solutions:**

1. Reduce KV cache:
   ```yaml
   generator:
     gpu_memory_utilization: 0.5
   ```

2. Limit concurrent sequences:
   ```yaml
   generator:
     max_num_seqs: 256
   ```

3. Reduce batch size:
   ```yaml
   generator:
     max_num_batched_tokens: 4096
   ```

---

### Error: `RuntimeError: CUDA error: out of memory` on startup

**Cause:** Model too large for GPU.

**Solutions:**

1. Use tensor parallelism:
   ```yaml
   generator:
     inference_engine_tensor_parallel_size: 2
   ```

2. Use quantization:
   ```yaml
   generator:
     engine_init_kwargs:
       quantization: "fp8"
   ```

3. Reduce model size or use smaller model.

---

## 6. Configuration Issues

### Error: `NotImplementedError: use_conversation_multi_turn=False is not supported for SGLang backend`

**Cause:** SGLang backend requires multi-turn mode.

**Solution:** Don't set `use_conversation_multi_turn=false`. The default is `true`.

---

### Error: `ValueError: tokenizer is required for SGLangInferenceEngine`

**Cause:** Tokenizer not passed to engine.

**Solution:** This should be automatic. If using custom code:
```python
engine = SGLangInferenceEngine(
    tokenizer=tokenizer,  # Required!
    model_path="...",
)
```

---

### Error: `ValueError: generator.async_engine must be True when generator.enable_http_endpoint==True`

**Cause:** HTTP endpoints require async engine.

**Solution:**
```yaml
generator:
  async_engine: true
  enable_http_endpoint: true
```

---

### Error: `ValueError: only vllm and sglang are supported with megatron`

**Cause:** Invalid backend for Megatron strategy.

**Solution:** Use `vllm` or `sglang` backend:
```yaml
generator:
  backend: sglang  # or vllm
trainer:
  strategy: megatron
```

---

## 7. Training Issues

### Loss is NaN or training diverges

**Solutions:**

1. Lower learning rate:
   ```yaml
   trainer:
     policy:
       optimizer_config:
         lr: 1.0e-7
   ```

2. Stricter gradient clipping:
   ```yaml
   trainer:
     policy:
       optimizer_config:
         max_grad_norm: 0.5
   ```

3. Stronger KL penalty:
   ```yaml
   trainer:
     algorithm:
       kl_loss_coef: 0.01
   ```

4. Tighter PPO clipping:
   ```yaml
   trainer:
     algorithm:
       eps_clip_low: 0.1
       eps_clip_high: 0.1
   ```

---

### Model outputs are repetitive

**Solutions:**
```yaml
generator:
  sampling_params:
    temperature: 1.0         # Increase exploration
    repetition_penalty: 1.1  # Penalize repetition
    top_p: 0.95              # Nucleus sampling
```

---

### Rewards are always 0

**Causes:**
1. Parsing not extracting answers
2. Ground truth format mismatch

**Debug:**
```python
# In your environment
def step(self, action):
    answer = self._parse_answer(action)
    print(f"Parsed: {answer}, Expected: {self.ground_truth}")
    ...
```

---

### Error: `grad_norm is not finite`

**Cause:** Gradient explosion.

**Solutions:**
1. Reduce learning rate
2. Enable gradient clipping
3. Check input data for anomalies
4. Use mixed precision carefully

---

## 8. Performance Issues

### Generation is slow

**Diagnosis:**
```bash
# Profile generation time
export SGLANG_LOG_LEVEL=debug
python -m skyrl_train.entrypoints.main_base ...
```

**Solutions:**

1. Increase GPU memory for KV cache:
   ```yaml
   generator:
     gpu_memory_utilization: 0.9
   ```

2. Enable prefix caching:
   ```yaml
   generator:
     enable_prefix_caching: true
   ```

3. Use more inference engines:
   ```yaml
   generator:
     num_inference_engines: 8
   ```

4. Use better attention backend:
   ```yaml
   generator:
     engine_init_kwargs:
       attention_backend: "flashinfer"  # or "fa3" for H100
   ```

---

### Training step takes too long

**Diagnosis:**
- Check generation time vs training time vs weight sync time

**Solutions:**

1. **Generation slow:** See above
2. **Weight sync slow:** Use CUDA IPC (colocate_all=true)
3. **Training slow:** Enable gradient checkpointing, reduce batch size

---

## 9. Environment Issues

### Error: `KeyError: 'my_env'`

**Cause:** Environment not registered.

**Solution:** Register before training:
```python
from skyrl_gym.envs import register

@ray.remote
def skyrl_entrypoint(cfg):
    register(id="my_env", entry_point="my_module.env:MyEnv")
    exp = BasePPOExp(cfg)
    exp.run()
```

---

### Error: `reward_spec field is required`

**Cause:** Dataset missing reward_spec.

**Solution:** Ensure dataset has correct format:
```python
{
    "prompt": [...],
    "env_class": "my_env",
    "reward_spec": {
        "method": "rule",
        "ground_truth": "expected"
    }
}
```

---

## 10. Debugging Commands

### Check GPU Status
```bash
nvidia-smi
watch -n 1 nvidia-smi
```

### Check Ray Status
```bash
ray status
ray dashboard  # Opens web UI at localhost:8265
```

### View Ray Logs
```bash
cat /tmp/ray/session_latest/logs/worker-*.out
tail -f /tmp/ray/session_latest/logs/worker-*.err
```

### Enable Debug Logging
```bash
export SGLANG_LOG_LEVEL=debug
export SKYRL_LOG_LEVEL=debug
export NCCL_DEBUG=INFO
```

### Check Port Availability
```bash
lsof -i :8000  # HTTP endpoint
lsof -i :29500  # NCCL
```

### Kill Stuck Processes
```bash
pkill -f sglang
pkill -f ray
ray stop --force
```

### Clear GPU Memory
```bash
nvidia-smi | grep python | awk '{print $5}' | xargs -I {} kill -9 {}
```

---

## 11. Data and Template Errors

### Error: `ValueError: Dataset <name> not found on Hugging Face`

**Cause:** Invalid dataset path or no internet access.

**Solutions:**
1. Verify dataset name:
   ```bash
   huggingface-cli list-datasets | grep <name>
   ```

2. Use local path:
   ```yaml
   data:
     train_data: "['~/data/train.parquet']"
     val_data: "['~/data/val.parquet']"
   ```

3. Check internet:
   ```bash
   curl -I https://huggingface.co
   ```

---

### Error: `ValueError: Template file not found` or `Template name not found`

**Cause:** Chat template configuration error.

**Solutions:**
1. Verify template source (`name` or `file`):
   ```yaml
   generator:
     chat_template:
       source: "name"           # Use built-in template
       name_or_path: "chatml"   # Template name
   ```

2. For file-based templates, use absolute path:
   ```yaml
   generator:
     chat_template:
       source: "file"
       name_or_path: "/absolute/path/to/template.jinja"
   ```

3. Available built-in templates: `chatml`, `zephyr`, `llama2`, `mistral`

---

### Error: `ValueError: Expected message role to be 'user' or 'assistant', got <role>`

**Cause:** Chat message format error.

**Solution:** Ensure messages use standard roles:
```python
messages = [
    {"role": "user", "content": "question"},      # OK
    {"role": "assistant", "content": "answer"},   # OK
    {"role": "system", "content": "prompt"},      # OK
    # {"role": "bot", ...}  # NOT OK
]
```

---

## 12. Training State Errors

### Error: `RuntimeError: Exiting training due to hitting dynamic sampling limit`

**Cause:** Dynamic sampling (DAPO feature) couldn't find enough good samples within the limit.

**Solutions:**
1. Increase max sample batches:
   ```yaml
   trainer:
     algorithm:
       dynamic_sampling:
         max_sample_batches: 50  # Increase from default 30
   ```

2. Disable dynamic sampling:
   ```yaml
   trainer:
     algorithm:
       dynamic_sampling:
         type: "none"
   ```

3. Check your data - may be too difficult

---

### Error: `ValueError: step_wise_trajectories doesn't support batched=True`

**Cause:** Feature incompatibility with step-wise training.

**Solution:** Use async generation:
```yaml
generator:
  step_wise_trajectories: true
  batched: false  # Required for step-wise
  async_engine: true
```

---

### Error: `ValueError: step_wise_trajectories doesn't support custom chat template`

**Cause:** Step-wise training requires standard chat template handling.

**Solution:** Remove custom chat template:
```yaml
generator:
  step_wise_trajectories: true
  chat_template: null  # Use default
```

---

## 13. Distributed Training Errors

### Error: `Exception: Could not find the transformer layer class to wrap`

**Cause:** FSDP auto-wrapping couldn't detect model architecture.

**Solutions:**
1. Use a supported model architecture (transformers-based)

2. Manually specify wrapping policy:
   ```yaml
   trainer:
     policy:
       fsdp_config:
         transformer_layer_cls_to_wrap: "QwenDecoderLayer"  # Model-specific
   ```

3. Check model is HuggingFace compatible

---

### Error: `RuntimeError: mesh_rank must be initialized before calling _normalize_mini_batch_size`

**Cause:** Worker initialization order issue.

**Solution:** This is an internal error. Update SkyRL or file issue:
```bash
git pull origin main
pip install -e .
```

---

### Error: `ValueError: Tensor must be on CUDA device to use CUDA IPC`

**Cause:** Attempting CUDA IPC with CPU tensors.

**Solutions:**
1. Ensure colocated mode:
   ```yaml
   trainer:
     placement:
       colocate_all: true
   ```

2. Check tensor device placement in custom code

3. Disable CUDA IPC:
   ```yaml
   generator:
     weight_sync_backend: gloo  # Use gloo instead
   ```

---

## 14. LoRA Errors

### Error: `RuntimeError: LoRA is not enabled. Set enable_lora=True`

**Cause:** Attempting LoRA operations without enabling LoRA.

**Solution:**
```yaml
generator:
  enable_lora: true
  engine_init_kwargs:
    max_lora_rank: 64

trainer:
  policy:
    model:
      lora:
        rank: 32
        alpha: 32
```

---

### Error: `ValueError: Unexpected keys in LoRA adapter state dict`

**Cause:** LoRA checkpoint doesn't match model architecture.

**Solutions:**
1. Verify LoRA checkpoint is for correct base model
2. Check rank configuration matches
3. Ensure checkpoint isn't corrupted

---

## 15. Engine Lifecycle Errors

### Error: `RuntimeError: wake_up timed out after Xs`

**Cause:** Engine memory resume took too long.

**Solutions:**
1. Increase timeout:
   ```yaml
   generator:
     engine_init_kwargs:
       wake_timeout: 120  # Increase from default
   ```

2. Reduce model size or use quantization

3. Check GPU memory isn't fragmented:
   ```bash
   nvidia-smi --query-gpu=memory.used --format=csv
   ```

---

### Error: `RuntimeError: sleep timed out after Xs`

**Cause:** Engine couldn't release memory in time.

**Solution:** Increase sleep timeout or check for stuck requests:
```yaml
generator:
  engine_init_kwargs:
    sleep_timeout: 60
```

---

## 16. Dataset Validation Errors

### Error: `ValueError: Dataset has no 'prompt' field`

**Cause:** Dataset missing required prompt field.

**Solution:** Ensure dataset has correct structure:
```python
# Required fields
{
    "prompt": [{"role": "user", "content": "..."}],  # Required
    "env_class": "gsm8k",                            # Required
    "reward_spec": {...}                             # Required
}
```

---

### Error: `ValueError: Field 'reward_spec' must be a dictionary`

**Cause:** reward_spec is wrong type (string, list, etc.).

**Solution:**
```python
# Correct
"reward_spec": {"method": "rule", "ground_truth": "42"}

# Wrong
"reward_spec": "rule:42"
```

---

### Error: `KeyError: 'env_class'`

**Cause:** Dataset missing environment class specification.

**Solution:** Add env_class to each sample:
```python
{
    "prompt": [...],
    "env_class": "gsm8k",  # Must match registered environment
    "reward_spec": {...}
}
```

---

### Warning: `Filtered X samples with empty prompts`

**Cause:** Dataset contains samples with empty or whitespace-only prompts.

**Note:** This is a warning, not error. Samples are silently filtered.

**Debug:** Check your data:
```python
import pandas as pd
df = pd.read_parquet("train.parquet")
empty_prompts = df[df['prompt'].apply(lambda x: len(str(x).strip()) == 0)]
print(f"Empty prompts: {len(empty_prompts)}")
```

---

### Error: `ValueError: Cannot mix single-turn and multi-turn samples`

**Cause:** Dataset contains both message-list and string prompts.

**Solution:** Ensure consistent format:
```python
# All multi-turn (recommended)
{"prompt": [{"role": "user", "content": "..."}]}

# Or all single-turn
{"prompt": "..."}
```

---

## 17. Tokenization Errors

### Error: `ValueError: Tokenizer has no pad_token`

**Cause:** Tokenizer missing padding token.

**Solution:**
```python
# In your setup
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id
```

Or in config:
```yaml
model:
  tokenizer_config:
    pad_token: "<|endoftext|>"
```

---

### Error: `RuntimeError: Token index out of range`

**Cause:** Token ID exceeds vocabulary size, often from corrupted data.

**Debug:**
```python
max_token = max(max(ids) for ids in input_ids)
vocab_size = tokenizer.vocab_size
if max_token >= vocab_size:
    print(f"Token {max_token} exceeds vocab {vocab_size}")
```

---

### Error: `ValueError: Input length exceeds model's max position embeddings`

**Cause:** Sequence too long for model.

**Solutions:**
1. Truncate input:
   ```yaml
   data:
     max_prompt_length: 4096
   generator:
     max_model_len: 8192
   ```

2. Use model with longer context:
   ```yaml
   model:
     model_path: "Qwen/Qwen2.5-7B-Instruct"  # 32K context
   ```

---

### Error: `UnicodeDecodeError` during tokenization

**Cause:** Invalid UTF-8 characters in data.

**Solution:**
```python
# Clean data before training
text = text.encode('utf-8', errors='ignore').decode('utf-8')
```

---

### Warning: `Truncating sequence from X to Y tokens`

**Cause:** Input exceeds configured max length.

**Note:** Sequences are silently truncated, which may affect training.

**Solution:** Increase max length or filter long sequences:
```yaml
data:
  max_prompt_length: 8192  # Increase
  filter_by_length: true   # Or filter long samples
```

---

## 18. Device Placement Errors

### Error: `RuntimeError: Expected all tensors to be on the same device`

**Cause:** Tensor device mismatch between operations.

**Debug:**
```python
print(f"tensor1: {tensor1.device}, tensor2: {tensor2.device}")
```

**Solution:** Ensure tensors on same device:
```python
tensor2 = tensor2.to(tensor1.device)
```

---

### Error: `RuntimeError: CUDA error: invalid device ordinal`

**Cause:** Requesting GPU that doesn't exist.

**Debug:**
```bash
echo "Available GPUs: $(nvidia-smi -L | wc -l)"
```

**Solution:** Check GPU indices in config:
```yaml
# If you have 4 GPUs, use indices 0-3
trainer:
  placement:
    devices: [0, 1, 2, 3]
```

---

### Error: `ValueError: Tensor must be on CPU for pickling`

**Cause:** Trying to serialize CUDA tensor through Ray.

**Solution:** Move to CPU before sending:
```python
# In custom code
tensor_cpu = tensor.cpu()
ray.put(tensor_cpu)
```

---

### Error: `RuntimeError: Cannot re-initialize CUDA in forked subprocess`

**Cause:** CUDA initialized before fork().

**Solutions:**
1. Use spawn instead of fork:
   ```python
   import multiprocessing
   multiprocessing.set_start_method('spawn')
   ```

2. Initialize CUDA only in child processes

3. Set environment:
   ```bash
   export CUDA_VISIBLE_DEVICES=0,1,2,3
   ```

---

## 19. Checkpoint and Resume Errors

### Error: `ValueError: Checkpoint architecture mismatch`

**Cause:** Checkpoint from different model architecture.

**Debug:**
```python
import torch
ckpt = torch.load("checkpoint.pt")
print(f"Checkpoint keys: {ckpt.keys()}")
```

**Solution:** Ensure checkpoint matches model:
```yaml
model:
  model_path: "Qwen/Qwen2.5-7B"  # Must match checkpoint's base model
```

---

### Error: `RuntimeError: Error(s) in loading state_dict: size mismatch`

**Cause:** Model dimension mismatch with checkpoint.

**Solutions:**
1. Use correct base model
2. Check LoRA rank matches:
   ```yaml
   trainer:
     policy:
       model:
         lora:
           rank: 32  # Must match checkpoint
   ```

---

### Error: `FileNotFoundError: Checkpoint step_X not found`

**Cause:** Checkpoint directory missing or corrupted.

**Debug:**
```bash
ls -la checkpoints/
# Should show step_X directories
```

**Solution:** Resume from valid checkpoint:
```yaml
trainer:
  resume_checkpoint: "checkpoints/step_1000"  # Use existing
```

---

### Warning: `Optimizer state not found, starting fresh`

**Cause:** Checkpoint missing optimizer state.

**Note:** Training continues but learning rate schedule resets.

**Solution:** Ensure full checkpoint:
```yaml
trainer:
  save_optimizer: true  # Save optimizer state
```

---

### Error: `ValueError: Config incompatible with checkpoint`

**Cause:** Training config differs from checkpoint config.

**Common mismatches:**
- Different batch size (can cause optimizer issues)
- Different learning rate schedule
- Different algorithm parameters

**Solution:** Match config or acknowledge reset:
```yaml
trainer:
  resume_checkpoint: "checkpoints/step_1000"
  reset_optimizer: true  # Accept optimizer reset
```

---

## 20. Data Loader Errors

### Error: `RuntimeError: DataLoader worker exited unexpectedly`

**Cause:** Worker process crashed (often OOM).

**Solutions:**
1. Reduce workers:
   ```yaml
   data:
     num_workers: 2  # Reduce from default
   ```

2. Reduce prefetch:
   ```yaml
   data:
     prefetch_factor: 1
   ```

3. Check worker logs:
   ```bash
   tail /tmp/ray/session_latest/logs/worker-*.err
   ```

---

### Error: `StopIteration` during training

**Cause:** Dataset exhausted unexpectedly.

**Solutions:**
1. Check dataset size:
   ```python
   print(f"Dataset size: {len(dataset)}")
   ```

2. Enable infinite iteration:
   ```yaml
   data:
     infinite: true  # Loop dataset
   ```

---

### Error: `RuntimeError: Cannot pickle generator object`

**Cause:** Dataset contains unpicklable objects.

**Solution:** Ensure dataset items are serializable:
```python
# Wrong
{"generator": (x for x in range(10))}

# Correct
{"data": list(range(10))}
```

---

## 21. SGLang Backend Specific Errors

### Error: `sglang.srt.server_args.ServerArgs: error: unrecognized arguments`

**Cause:** Passing unknown argument to SGLang server.

**Solution:** Check argument compatibility with SGLang version:
```bash
python -m sglang.launch_server --help | grep <arg_name>
```

---

### Error: `RuntimeError: RadixCache: prefix not found`

**Cause:** Prefix cache miss, possibly due to eviction.

**Note:** Usually not fatal, just reduces efficiency.

**Solution:** Increase cache size:
```yaml
generator:
  gpu_memory_utilization: 0.8  # More memory for cache
```

---

### Error: `ValueError: Generation cancelled: maximum tokens reached`

**Cause:** Output hit max_tokens limit.

**Solution:** Increase output limit:
```yaml
generator:
  sampling_params:
    max_new_tokens: 2048  # Increase from default
```

---

### Error: `RuntimeError: Scheduler queue full`

**Cause:** Too many concurrent requests.

**Solution:** Reduce batch size or increase scheduler capacity:
```yaml
generator:
  max_num_seqs: 128  # Reduce concurrent sequences
```

---

### Error: `sglang.srt.managers.schedule_batch.ABORT`

**Cause:** Request was aborted (timeout, error, or explicit cancel).

**Debug:** Check why requests abort:
```bash
export SGLANG_LOG_LEVEL=debug
```

---

## 22. Pause/Continue Generation Errors

### Error: `RuntimeError: Cannot pause: engine not in running state`

**Cause:** Attempting to pause already-paused engine.

**Debug:**
```python
# Check engine state before pause
print(f"Engine state: {engine.get_state()}")
```

---

### Error: `RuntimeError: Retract mode failed: requests still in progress`

**Cause:** Active requests during retract-mode pause.

**Solution:** Use abort mode instead:
```yaml
generator:
  pause_mode: "abort"  # Instead of "retract"
```

---

### Error: `RuntimeError: KV cache corrupted after wake_up`

**Cause:** Memory corruption during sleep/wake cycle.

**Solutions:**
1. Clear cache on wake:
   ```yaml
   generator:
     clear_cache_on_wake: true
   ```

2. Increase sleep timeout:
   ```yaml
   generator:
     engine_init_kwargs:
       sleep_timeout: 120
   ```

---

## 23. Multi-Turn Generation Errors

### Error: `ValueError: Conversation state not found for request_id`

**Cause:** State lost between turns, possibly due to eviction.

**Solutions:**
1. Increase state cache:
   ```yaml
   generator:
     multi_turn_state_cache_size: 10000
   ```

2. Reduce time between turns

---

### Error: `RuntimeError: Turn index out of range`

**Cause:** Requesting turn that doesn't exist.

**Debug:**
```python
print(f"Num turns: {len(conversation.turns)}")
print(f"Requested turn: {turn_index}")
```

---

### Error: `ValueError: Cannot add turn: conversation ended`

**Cause:** Attempting to add turn after terminal state.

**Solution:** Check conversation state:
```python
if not conversation.is_ended:
    conversation.add_turn(...)
```

---

## 24. Reward Computation Errors

### Error: `ValueError: Reward tensor shape mismatch`

**Cause:** Reward shape doesn't match sequence shape.

**Debug:**
```python
print(f"Rewards shape: {rewards.shape}")
print(f"Sequences shape: {sequences.shape}")
```

**Solution:** Ensure reward matches sequence length:
```python
# Per-sequence reward
rewards = torch.tensor([1.0] * batch_size)

# Per-token reward (step-wise)
rewards = torch.zeros(batch_size, seq_len)
rewards[:, -1] = final_reward  # Reward at end
```

---

### Error: `RuntimeError: Cannot compute advantage: reference log probs missing`

**Cause:** Reference model outputs not available.

**Solution:** Enable reference model:
```yaml
trainer:
  algorithm:
    use_reference_model: true
```

---

### Error: `ValueError: KL divergence is NaN`

**Cause:** Numerical instability in KL computation.

**Solutions:**
1. Clamp log probs:
   ```yaml
   trainer:
     algorithm:
       log_prob_clamp: 1e-8
   ```

2. Reduce KL coefficient:
   ```yaml
   trainer:
     algorithm:
       kl_loss_coef: 0.001
   ```

---

## 25. HTTP Endpoint Errors

### Error: `ConnectionRefusedError: Connection refused on port 8000`

**Cause:** HTTP endpoint not started or wrong port.

**Debug:**
```bash
curl http://localhost:8000/health
```

**Solution:**
```yaml
generator:
  enable_http_endpoint: true
  http_port: 8000
```

---

### Error: `HTTP 503: Service Unavailable`

**Cause:** Engine paused or overloaded.

**Debug:**
```bash
curl http://localhost:8000/get_server_info
```

---

### Error: `HTTP 400: Invalid request format`

**Cause:** Malformed request body.

**Example correct request:**
```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello", "sampling_params": {"max_new_tokens": 100}}'
```

---

### Error: `HTTP 408: Request Timeout`

**Cause:** Generation took too long.

**Solution:**
```yaml
generator:
  request_timeout: 300  # Increase timeout (seconds)
```

---

## 26. Algorithm-Specific Errors

### Error: `ValueError: DAPO requires n_samples_per_prompt >= 2`

**Cause:** DAPO needs multiple samples for comparison.

**Solution:**
```yaml
generator:
  n_samples_per_prompt: 16  # Increase from 1
```

---

### Error: `ValueError: GRPO advantage estimator requires group size > 1`

**Cause:** GRPO needs sample groups.

**Solution:**
```yaml
generator:
  n_samples_per_prompt: 8
trainer:
  algorithm:
    advantage_estimator: "grpo"
```

---

### Error: `ValueError: Clip ratio must be positive`

**Cause:** Invalid clipping configuration.

**Solution:**
```yaml
trainer:
  algorithm:
    eps_clip_low: 0.2   # Must be > 0
    eps_clip_high: 0.28  # Must be > eps_clip_low
```

---

### Error: `ValueError: CISPO requires variance tracking`

**Cause:** CISPO algorithm missing required config.

**Solution:**
```yaml
trainer:
  algorithm:
    policy_loss_type: "cispo"
    cispo:
      enable_variance_tracking: true
```

---

## 27. Environment-Specific Errors

### Error: `ValueError: Tool 'search' not found`

**Cause:** Tool group not registered.

**Solution:**
```yaml
generator:
  tool_groups:
    - name: "SearchToolGroup"
      config:
        retriever_url: "http://localhost:8081"
```

---

### Error: `RuntimeError: Python code execution timed out`

**Cause:** Code execution exceeded timeout.

**Solution:**
```yaml
generator:
  tool_groups:
    - name: "PythonCodeExecutorToolGroup"
      config:
        timeout: 30  # Increase from default
```

---

### Error: `ValueError: SQL query failed: table not found`

**Cause:** SQL environment database issue.

**Solution:** Ensure database setup:
```yaml
generator:
  tool_groups:
    - name: "SQLCodeExecutorToolGroup"
      config:
        database_path: "/path/to/database.db"
```

---

### Error: `RuntimeError: Environment step returned invalid observation`

**Cause:** Custom environment returning wrong format.

**Solution:** Ensure step() returns valid format:
```python
def step(self, action):
    # ...
    return {
        "observation": "text",  # Required
        "reward": 1.0,          # Required
        "done": False,          # Required
        "info": {}              # Optional
    }
```

---

## 28. Validation and Evaluation Errors

### Error: `ValueError: No validation data configured`

**Cause:** Validation requested but no data provided.

**Solution:**
```yaml
data:
  val_data: "['validation.parquet']"
trainer:
  validation:
    run_validation: true
```

---

### Error: `RuntimeError: Evaluation metric undefined for task`

**Cause:** Metric not implemented for environment.

**Solution:** Implement in environment:
```python
class MyEnv:
    def get_metrics(self):
        return {
            "accuracy": self.correct / self.total,
            "avg_reward": sum(self.rewards) / len(self.rewards)
        }
```

---

## 29. Getting Help

1. **Check existing docs:**
   - [FAQ](./FAQ_SGLANG.md)
   - [Full Integration Guide](./SGLANG_INTEGRATION_GUIDE.md)

2. **Search issues:**
   - [SkyRL Issues](https://github.com/NovaSky-AI/SkyRL/issues)
   - [SGLang Issues](https://github.com/sgl-project/sglang/issues)

3. **Join community:**
   - [SkyRL Discord](https://discord.gg/RBAjeWSA)
   - [SGLang Discord](https://discord.gg/sglang)

4. **File a bug:**
   Include:
   - Full error message and stack trace
   - Configuration (yaml)
   - Environment (GPU, CUDA version, Python version)
   - Steps to reproduce
