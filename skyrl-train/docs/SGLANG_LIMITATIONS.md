# SGLang Backend Feature Support in SkyRL

This document provides a comprehensive list of SGLang backend features in SkyRL, with direct code references.

**Last Updated:** 2026-01-09
**Status:** All major features now supported

---

## Summary Table

| Feature | Status | Notes |
|---------|--------|-------|
| Tensor Parallelism (TP > 1) | **Supported** | Native SGLang support |
| Pipeline Parallelism (PP > 1) | **Supported** | Native SGLang support |
| Data Parallelism (DP > 1) | **Supported** | Native SGLang support |
| Expert Parallelism (EP > 1) | **Supported** | Native SGLang support for MoE |
| LoRA Adapters | **Supported** | Runtime load/unload APIs |
| Stop Sequences | **Supported** | Converted to stop_token_ids |
| HTTP Endpoints | **Supported** | Uses external tokenizer |
| Megatron Training | **Supported** | Experimental, with warning |
| min_new_tokens | **Supported** | Uses eos_token_id from external tokenizer |
| Multi-turn Required | Yes | use_conversation_multi_turn=true required |

---

## Supported Features

### 1. Tensor Parallelism (TP > 1)

**Status:** Fully supported

**File:** `skyrl_train/utils/utils.py`
**Lines:** 393-394

```python
# SGLang supports TP > 1 natively - no restriction needed
# The tp_size param is passed directly to sglang.Engine()
```

**Usage:**
```yaml
generator:
  backend: sglang
  inference_engine_tensor_parallel_size: 4
```

---

### 2. Pipeline Parallelism (PP > 1)

**Status:** Fully supported

**File:** `skyrl_train/inference_engines/ray_wrapped_inference_engine.py`
**Lines:** 246-247

```python
if pipeline_parallel_size > 1:
    logger.info(f"SGLang backend: Using pipeline parallelism with pp_size={pipeline_parallel_size}")
```

**Usage:**
```yaml
generator:
  backend: sglang
  inference_engine_pipeline_parallel_size: 2
```

---

### 3. Data Parallelism (DP > 1)

**Status:** Fully supported

**File:** `skyrl_train/inference_engines/ray_wrapped_inference_engine.py`
**Lines:** 248-249

```python
if data_parallel_size > 1:
    logger.info(f"SGLang backend: Using data parallelism with dp_size={data_parallel_size}")
```

**Usage:**
```yaml
generator:
  backend: sglang
  inference_engine_data_parallel_size: 2
```

---

### 4. Expert Parallelism (EP > 1)

**Status:** Fully supported for MoE models

**File:** `skyrl_train/inference_engines/ray_wrapped_inference_engine.py`
**Lines:** 244-245, 314

```python
if expert_parallel_size > 1:
    logger.info(f"SGLang backend: Using expert parallelism with ep_size={expert_parallel_size}")
```

**Usage:**
```yaml
generator:
  backend: sglang
  inference_engine_expert_parallel_size: 4
  engine_init_kwargs:
    moe_a2a_backend: "deepep"
```

---

### 5. LoRA Adapters

**Status:** Fully supported with runtime load/unload

**File:** `skyrl_train/inference_engines/sglang/sglang_engine.py`
**Lines:** 357-387

```python
async def load_lora_adapter(self, lora_name: str, lora_path: str, pinned: bool = False):
    """Load a LoRA adapter at runtime."""
    ...

async def unload_lora_adapter(self, lora_name: str):
    """Unload a LoRA adapter at runtime."""
    ...
```

**Usage:**
```yaml
generator:
  backend: sglang
  enable_lora: true
  engine_init_kwargs:
    max_lora_rank: 64
    max_loras_per_batch: 8
    lora_backend: "csgmv"
```

---

### 6. Stop Sequences

**Status:** Supported via stop_token_ids conversion

**File:** `skyrl_train/inference_engines/sglang/sglang_engine.py`
**Lines:** 402-418

```python
# Convert stop strings to stop_token_ids using our external tokenizer
# This works even with skip_tokenizer_init=True since we use the external tokenizer
stop_strings = sampling_params.pop("stop", None)
if stop_strings is not None:
    stop_token_ids = sampling_params.get("stop_token_ids", []) or []
    for stop_str in stop_strings:
        token_ids = self.tokenizer.encode(stop_str, add_special_tokens=False)
        if token_ids:
            stop_token_ids.append(token_ids[-1] if len(token_ids) > 1 else token_ids[0])
    if stop_token_ids:
        sampling_params["stop_token_ids"] = list(set(stop_token_ids))
```

**Usage:**
```yaml
generator:
  backend: sglang
  sampling_params:
    stop: ["</s>", "\n\n"]
```

---

### 7. min_new_tokens

**Status:** Fully supported via eos_token_id injection

**Files:**
- SGLang: `sglang/python/sglang/srt/sampling/sampling_params.py` - Added `eos_token_id` parameter
- SGLang: `sglang/python/sglang/srt/sampling/penaltylib/min_new_tokens.py` - Uses `sampling_params.eos_token_id` if set
- SkyRL: `skyrl_train/inference_engines/sglang/sglang_engine.py:420-424` - Passes eos_token_id from external tokenizer

```python
# SkyRL automatically passes eos_token_id when min_new_tokens is set
if sampling_params.get("min_new_tokens", 0) > 0:
    sampling_params["eos_token_id"] = self.tokenizer.eos_token_id
```

**Usage:**
```yaml
generator:
  backend: sglang
  sampling_params:
    min_new_tokens: 10
    max_generate_length: 1024
```

---

### 8. HTTP Endpoints (OpenAI-compatible API)

**Status:** Supported via external tokenizer

**File:** `skyrl_train/inference_engines/sglang/sglang_engine.py`
**Lines:** 508-664

```python
async def chat_completion(self, request_payload: Dict[str, Any]) -> Dict[str, Any]:
    """Handle OpenAI-compatible chat completion request.
    Uses external tokenizer to convert text<->tokens since we run with skip_tokenizer_init=True.
    """
    ...

async def completion(self, request_payload: Dict[str, Any]) -> Dict[str, Any]:
    """Handle OpenAI-compatible text completion request.
    Uses external tokenizer to convert text<->tokens since we run with skip_tokenizer_init=True.
    """
    ...
```

**Usage:**
```yaml
generator:
  backend: sglang
  enable_http_endpoint: true
  http_endpoint_host: "0.0.0.0"
  http_endpoint_port: 8000
  async_engine: true
```

---

### 9. Megatron Distributed Training

**Status:** Supported (experimental)

**File:** `skyrl_train/utils/utils.py`
**Lines:** 185-190

```python
assert cfg.generator.backend in ["vllm", "sglang"], "only vllm and sglang are supported with megatron"
if cfg.generator.backend == "sglang":
    logger.warning(
        "SGLang backend with Megatron training is experimental. "
        "If you encounter issues, try switching to vLLM backend."
    )
```

**Usage:**
```yaml
trainer:
  strategy: megatron
generator:
  backend: sglang
  weight_sync_backend: nccl
```

---

## Remaining Limitations

### 1. use_conversation_multi_turn Required

**File:** `skyrl_train/utils/utils.py`
**Lines:** 396-397

```python
if cfg.generator.backend == "sglang" and not cfg.generator.use_conversation_multi_turn:
    raise NotImplementedError("`use_conversation_multi_turn=False` is not supported for SGLang backend")
```

**Note:** This is the default, so no action needed.

---

## Configuration Examples

### Full-Featured SGLang Configuration

```yaml
generator:
  backend: sglang
  run_engines_locally: true
  num_inference_engines: 8
  inference_engine_tensor_parallel_size: 2
  inference_engine_pipeline_parallel_size: 2
  inference_engine_data_parallel_size: 1
  inference_engine_expert_parallel_size: 1
  gpu_memory_utilization: 0.8
  enable_prefix_caching: true
  weight_sync_backend: nccl
  async_engine: true
  batched: true
  enable_lora: true
  enable_http_endpoint: true
  http_endpoint_port: 8000
  sampling_params:
    max_generate_length: 1024
    temperature: 1.0
    stop: ["</s>"]
  engine_init_kwargs:
    attention_backend: "fa3"
    max_lora_rank: 64
    max_loras_per_batch: 8

trainer:
  strategy: fsdp2
  placement:
    colocate_all: true
```

### MoE Model with Expert Parallelism

```yaml
generator:
  backend: sglang
  inference_engine_tensor_parallel_size: 4
  inference_engine_expert_parallel_size: 4
  engine_init_kwargs:
    moe_a2a_backend: "deepep"
    moe_runner_backend: "auto"
    enable_eplb: true
```

---

## References

- SGLang Documentation: https://docs.sglang.io/
- SkyRL Documentation: `docs/examples/sglang_backend.rst`
