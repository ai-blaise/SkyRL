# SkyRL + SGLang Integration Guide

**Comprehensive guide for using SGLang as the inference backend for SkyRL reinforcement learning training.**

**Last Updated:** 2026-01-10
**Status:** Production Ready - Verified with end-to-end training

## Verified Training Results

The SGLang integration has been tested with full RL training:

| Metric | Value |
|--------|-------|
| **Model** | Qwen/Qwen2.5-0.5B-Instruct |
| **Dataset** | GSM8K (grade school math) |
| **Algorithm** | GRPO |
| **Initial Accuracy** | 24.41% |
| **Peak Accuracy** | 44.66% |
| **Improvement** | +20pp (82% relative) |
| **Weight Sync Time** | ~1.9s per step |

---

## Table of Contents

1. [Overview](#1-overview)
2. [Architecture](#2-architecture)
3. [Quick Start](#3-quick-start)
4. [Configuration Reference](#4-configuration-reference)
5. [Weight Synchronization](#5-weight-synchronization)
6. [Parallelism Options](#6-parallelism-options)
7. [Advanced Features](#7-advanced-features)
8. [Training Loop Integration](#8-training-loop-integration)
9. [Performance Tuning](#9-performance-tuning)
10. [Troubleshooting](#10-troubleshooting)
11. [SGLang Inference Engine Details](#11-sglang-inference-engine-details)
12. [Internal RL Training APIs](#12-internal-rl-training-apis)
13. [Multi-Node Distributed Training](#13-multi-node-distributed-training)
14. [Performance Benchmarking](#14-performance-benchmarking)
15. [Generator Modes and Step-Wise Trajectories](#15-generator-modes-and-step-wise-trajectories)
16. [Dynamic Sampling Strategies](#16-dynamic-sampling-strategies)

---

## 1. Overview

### What is the SkyRL + SGLang Integration?

SkyRL is a reinforcement learning framework for training large language models using algorithms like PPO, GRPO, and RLOO. SGLang is a high-performance inference engine with advanced features like RadixAttention for prefix caching.

The integration allows SkyRL to use SGLang as the inference backend during RL training, enabling:
- **Fast rollout generation** using SGLang's optimized inference
- **Efficient weight synchronization** between training and inference
- **Memory-efficient training** via sleep/wake cycles
- **Multi-turn conversation support** for agentic training

### Key Features

| Feature | Support Level | Notes |
|---------|--------------|-------|
| Token-in-token-out mode | **Required** | Always uses `skip_tokenizer_init=True` |
| Tensor Parallelism (TP) | **Native** | Direct SGLang support |
| Pipeline Parallelism (PP) | **Native** | Direct SGLang support |
| Data Parallelism (DP) | **Native** | Direct SGLang support |
| Expert Parallelism (EP) | **Native** | For MoE models |
| LoRA Adapters | **Supported** | Runtime load/unload |
| Prefix Caching | **Supported** | RadixAttention |
| Weight Sync (NCCL) | **Supported** | Zero-copy IPC for colocated |
| Weight Sync (Broadcast) | **Supported** | For distributed setups |
| HTTP Endpoints | **Supported** | OpenAI-compatible API |
| min_new_tokens | **Supported** | Via eos_token_id injection |
| Multi-turn Training | **Required** | `use_conversation_multi_turn=true` |
| Structured Output | **Supported** | JSON schema, regex, EBNF constraints |

---

## 2. Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         SkyRL Trainer                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │ Policy Model│  │ Critic Model│  │ Reference Model         │  │
│  │   (FSDP2)   │  │   (FSDP2)   │  │   (FSDP2)               │  │
│  └──────┬──────┘  └─────────────┘  └─────────────────────────┘  │
│         │                                                        │
│         │ Weight Sync (NCCL IPC or Broadcast)                   │
│         ▼                                                        │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │              InferenceEngineClient                          ││
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐           ││
│  │  │SGLang   │ │SGLang   │ │SGLang   │ │SGLang   │           ││
│  │  │Engine 0 │ │Engine 1 │ │Engine 2 │ │Engine N │  (Ray)    ││
│  │  └─────────┘ └─────────┘ └─────────┘ └─────────┘           ││
│  └─────────────────────────────────────────────────────────────┘│
│         │                                                        │
│         ▼                                                        │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                    Generator                                 ││
│  │  SkyRLGymGenerator (multi-turn agent loop)                  ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

### Component Breakdown

#### 1. SGLangInferenceEngine (Ray Actor)
**File:** `skyrl_train/inference_engines/sglang/sglang_engine.py`

- Wraps SGLang's native `Engine` class
- Always runs with `skip_tokenizer_init=True` (token-in-token-out mode)
- Manages weight loading via `SGLangWeightLoader`
- Provides OpenAI-compatible HTTP endpoints

```python
# Key initialization (lines 307-343)
class SGLangInferenceEngine(InferenceEngineInterface):
    def __init__(self, tokenizer, **kwargs):
        kwargs["skip_tokenizer_init"] = True  # Always token-in-token-out
        kwargs["custom_weight_loader"] = CUSTOM_WEIGHT_LOADER_PATH
        self.engine = Engine(**kwargs)
        self._weight_loader = SGLangWeightLoader(self.engine, ...)
```

#### 2. InferenceEngineClient
**File:** `skyrl_train/inference_engines/inference_engine_client.py`

- Manages multiple SGLang engine instances
- Routes prompts to engines (session-based or load-balanced)
- Handles weight synchronization coordination

```python
# Prompt routing (lines 92-96)
engine_idx_to_prompt_ids = route_prompts_to_engines(
    num_prompts=num_prompts,
    num_inference_engines=len(self.engines),
    session_ids=session_ids,  # For consistent routing in multi-turn
)
```

#### 3. Weight Synchronization
**File:** `skyrl_train/weight_sync/`

Two strategies based on configuration:

| Strategy | When Used | Mechanism |
|----------|-----------|-----------|
| **CUDA IPC** | `weight_sync_backend=nccl` + `colocate_all=true` | Zero-copy GPU memory sharing |
| **Broadcast** | Other configurations | `torch.distributed.broadcast()` |

---

## 3. Quick Start

### Installation

```bash
# Install SkyRL with SGLang support
cd /path/to/SkyRL/skyrl-train
uv sync --extra sglang

# Or with pip
pip install -e ".[sglang]"
```

**Note:** The sglang extra installs specific versions: torch 2.9.1, flashinfer 0.5.3, flash-attn >= 2.8.3.

### Complete Minimal Configuration

```yaml
# config.yaml - Complete working example
trainer:
  strategy: fsdp2
  placement:
    colocate_all: true  # Recommended for efficient weight sync
  policy:
    model:
      path: "Qwen/Qwen2.5-1.5B-Instruct"

generator:
  backend: sglang
  num_inference_engines: 4
  inference_engine_tensor_parallel_size: 1
  gpu_memory_utilization: 0.8
  enable_prefix_caching: true
  weight_sync_backend: nccl
  async_engine: true
  model_dtype: "bfloat16"
  use_conversation_multi_turn: true  # Required for SGLang
  sampling_params:
    max_generate_length: 1024
    temperature: 1.0

# Data configuration (required)
data:
  train_data: "path/to/train.jsonl"
  val_data: "path/to/val.jsonl"

# Environment configuration
env:
  name: "gsm8k"  # or your environment
  max_turns: 1
```

### Running Training

```bash
# Option 1: Using activated venv (recommended for editable SGLang installs)
source .venv/bin/activate
unset RAY_RUNTIME_ENV_HOOK  # Required if SGLang is installed as editable

python -m skyrl_train.entrypoints.main_base \
  +experiment=grpo_qwen2.5-0.5b_math500 \
  generator.backend=sglang \
  +generator.engine_init_kwargs.attention_backend=flashinfer

# Option 2: Using uv run (for non-editable installs)
uv run --isolated --extra sglang -m skyrl_train.entrypoints.main_base \
  +experiment=grpo_qwen2.5-0.5b_math500 \
  generator.backend=sglang
```

**Important Notes:**
- Use `flashinfer` attention backend if your GPU doesn't support FlashAttention v3 (requires SM80-90)
- Unset `RAY_RUNTIME_ENV_HOOK` when using editable SGLang installs to avoid Ray worker failures

### Example Configurations

Pre-built example configurations are available:

| Example | Path | Description |
|---------|------|-------------|
| Full SGLang Config | `examples/sglang/sglang_full_config_example.yaml` | All options documented |
| GSM8K + SGLang | `examples/gsm8k/gsm8k-grpo-sglang-skypilot.yaml` | Working GSM8K training |

---

## 4. Configuration Reference

### Generator Configuration (Complete)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `backend` | str | "vllm" | Set to `"sglang"` for SGLang backend |
| `model_dtype` | str | "bfloat16" | Model data type ("bfloat16", "float16", "float32") |
| `run_engines_locally` | bool | true | Run engines in same Ray cluster |
| `num_inference_engines` | int | 1 | Number of parallel inference engine instances |
| `inference_engine_tensor_parallel_size` | int | 1 | TP size per engine |
| `inference_engine_pipeline_parallel_size` | int | 1 | PP size per engine |
| `inference_engine_data_parallel_size` | int | 1 | DP size per engine |
| `inference_engine_expert_parallel_size` | int | 1 | EP size for MoE models |
| `gpu_memory_utilization` | float | 0.8 | Fraction of GPU memory (maps to `mem_fraction_static`) |
| `max_num_batched_tokens` | int | 8192 | Max tokens in prefill (maps to `max_prefill_tokens`) |
| `max_num_seqs` | int | 1024 | Max concurrent sequences (maps to `max_running_requests`) |
| `enable_prefix_caching` | bool | true | RadixAttention (inverse of `disable_radix_cache`) |
| `weight_sync_backend` | str | "nccl" | Weight sync: "nccl" or "gloo" |
| `weight_transfer_threshold_cuda_ipc_GB` | float | 1.0 | Batch size for CUDA IPC transfers |
| `async_engine` | bool | true | Use async generation (required for HTTP endpoints) |
| `batched` | bool | true | Enable batched generation |
| `enable_lora` | bool | false | Enable LoRA support |
| `enable_http_endpoint` | bool | false | Enable OpenAI-compatible HTTP API |
| `http_endpoint_host` | str | "127.0.0.1" | HTTP endpoint bind host |
| `http_endpoint_port` | int | 8000 | HTTP endpoint port |
| `use_conversation_multi_turn` | bool | true | **Required for SGLang** |
| `max_turns` | int | 1 | Maximum conversation turns |
| `override_existing_update_group` | str | "auto" | Weight update group behavior |

### Advanced Generator Options

These options are for specialized use cases:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_samples_per_prompt` | int | 5 | Responses per prompt during training (critical for GRPO/RLOO) |
| `eval_n_samples_per_prompt` | int | 1 | Responses per prompt during evaluation (for pass@n metrics) |
| `step_wise_trajectories` | bool | false | Generate step-by-step for token-level rewards. **Constraints:** Incompatible with `batched=true` and custom `chat_template` |
| `zero_reward_on_non_stop` | bool | false | Set reward=0 if generation doesn't end with stop sequence |
| `apply_overlong_filtering` | bool | false | Apply DAPO overlong filtering to loss masks (for truncated sequences) |
| `append_eos_token_after_stop_str_in_multi_turn` | bool | true | Append EOS after stop string in multi-turn mode |
| `chat_template` | dict | null | Custom chat template: `{source: "name"\|"file", name_or_path: str}` |
| `chat_template_kwargs` | dict | {} | Kwargs for `tokenizer.apply_chat_template()`. **Incompatible with `batched=true`** |
| `remote_inference_engine_urls` | list | ["127.0.0.1:8001"] | URLs for remote inference servers. **Note:** Weight sync disabled with remote servers |
| `vllm_v1_disable_multiproc` | bool | true | Disable multiprocessing in vLLM V1 (vLLM-only, not applicable to SGLang) |

### Sampling Parameters (Complete)

```yaml
generator:
  sampling_params:
    # Generation length
    max_generate_length: 1024    # Maximum tokens to generate
    min_new_tokens: 0            # Minimum tokens (suppresses EOS until reached)

    # Temperature and sampling
    temperature: 1.0             # Sampling temperature (>0, <1e-6 = greedy)
    top_p: 1.0                   # Nucleus sampling threshold (0, 1]
    top_k: -1                    # Top-k sampling (-1 = disabled, uses all vocab)
    min_p: 0.0                   # Minimum probability threshold [0, 1]

    # Penalty parameters
    frequency_penalty: 0.0       # Cumulative frequency penalty [-2, 2]
    presence_penalty: 0.0        # Binary presence penalty [-2, 2]
    repetition_penalty: 1.0      # Multiplicative repetition penalty [0, 2]

    # Stop conditions
    stop: ["</s>"]               # Stop strings (converted to token IDs)
    stop_token_ids: []           # Direct stop token IDs
    stop_regex: null             # Regex-based stop patterns
    ignore_eos: false            # Force generation past EOS token

    # Structured output (mutually exclusive)
    json_schema: null            # JSON schema constraint
    regex: null                  # Regex pattern constraint
    ebnf: null                   # EBNF grammar constraint
    structural_tag: null         # Structural tag for output

    # Advanced options
    n: 1                         # Number of sequences to generate
    logprobs: 0                  # Return logprobs (0 = return, N = top N)
    logit_bias: null             # Token ID → logit bias mapping
    sampling_seed: null          # Deterministic sampling seed
    no_stop_trim: true           # Don't trim stop strings (always true)
    skip_special_tokens: true    # Skip special tokens in decode
    spaces_between_special_tokens: true  # Add spaces between special tokens
```

### Engine Init Kwargs (Complete)

Pass SGLang-specific parameters via `engine_init_kwargs`:

```yaml
generator:
  engine_init_kwargs:
    # Attention backends - choose based on GPU compute capability
    # - fa3: FlashAttention 3 (requires SM80-90, i.e. A100, H100)
    # - flashinfer: Works on broader GPU range (recommended for compatibility)
    attention_backend: "flashinfer"     # "fa3" or "flashinfer"
    mm_attention_backend: "flashinfer"  # Multi-modal attention backend

    # LoRA configuration (when enable_lora=true)
    max_lora_rank: 64               # Maximum LoRA rank
    max_loras_per_batch: 8          # Max adapters per batch
    max_loaded_loras: null          # Max loaded adapters (memory limit)
    lora_backend: "csgmv"           # "csgmv", "triton", "ascend", "torch_native"
    lora_eviction_policy: "lru"     # "lru" or "lfu" eviction policy
    lora_paths: null                # Pre-load adapters on startup
    max_lora_chunk_size: 16         # Chunk size (power of 2, 16-128)

    # MoE configuration
    moe_a2a_backend: "deepep"       # All-to-all backend for MoE
    moe_runner_backend: "auto"      # MoE runner backend
    enable_eplb: true               # Expert load balancing

    # Memory and performance
    enable_memory_saver: true       # Enable sleep/wake functionality
    enable_symm_mem: false          # Symmetric memory for NCCL
    enable_nccl_nvls: false         # NCCL NVLS communication
```

### Environment Variables

These are automatically set during SGLang initialization:

| Variable | Default | Purpose |
|----------|---------|---------|
| `NCCL_CUMEM_ENABLE` | 0 | NCCL cumulative memory |
| `CUDA_DEVICE_MAX_CONNECTIONS` | 8 | Max CUDA connections |
| `SGLANG_RUN_ID` | (generated) | Unique run identifier |
| `SGLANG_LOG_LEVEL` | (user) | Logging level |

---

## 5. Weight Synchronization

### Strategy Selection

The weight sync strategy is automatically selected based on configuration:

```python
# From weight_sync/__init__.py (lines 33-50)
def get_transfer_strategy_cls(cfg):
    if cfg.generator.weight_sync_backend == "nccl" and cfg.trainer.placement.colocate_all:
        return CudaIpcTransferStrategy  # Zero-copy
    return BroadcastTransferStrategy    # torch.distributed
```

### Prerequisites

- **torch.distributed** must be initialized (automatic in SkyRL)
- **CUDA GPUs** required for both strategies
- **model_dtype** must match between trainer and inference engine

### CUDA IPC Strategy (Recommended for Colocated)

**When:** `weight_sync_backend=nccl` AND `colocate_all=true`

**How it works:**
1. Training workers pack tensors into contiguous buffers
2. Create CUDA IPC handles for zero-copy sharing
3. SGLang engines receive handles and reconstruct tensors
4. No data copying between processes on same node

```yaml
trainer:
  placement:
    colocate_all: true

generator:
  weight_sync_backend: nccl
  weight_transfer_threshold_cuda_ipc_GB: 1.0  # Batch weights up to 1GB
```

### Broadcast Strategy (For Distributed)

**When:** Any other configuration

**How it works:**
1. Training rank 0 initiates process group with SGLang engines
2. Uses `torch.distributed.broadcast()` for tensor transfer
3. Works across nodes

**Note:** Only NCCL backend is supported for Megatron training.

```yaml
trainer:
  placement:
    colocate_all: false

generator:
  weight_sync_backend: nccl  # or gloo
```

### Weight Sync Flow

```
Training Step Complete
        │
        ▼
sync_policy_weights_to_inference_engines()
        │
        ▼
PolicyWorker.broadcast_to_inference_engines()
        │
        ├─ FSDP: Gather sharded tensors
        │
        ▼
WeightTransferSender.send_chunks()
        │
        ├─ IPC: Pack → Create handles → Send request
        │  └─ SGLang: Custom weight loader → model.load_weights()
        │
        └─ Broadcast: Send RPC → torch.distributed.broadcast()
           └─ SGLang: update_weights_from_distributed()
```

---

## 6. Parallelism Options

### Tensor Parallelism (TP)

Split model layers across multiple GPUs:

```yaml
generator:
  num_inference_engines: 2
  inference_engine_tensor_parallel_size: 4  # 4 GPUs per engine
```

**Total GPUs:** `num_engines × tp_size = 2 × 4 = 8`

### Pipeline Parallelism (PP)

Split model stages across GPUs:

```yaml
generator:
  inference_engine_tensor_parallel_size: 2
  inference_engine_pipeline_parallel_size: 2  # 2 pipeline stages
```

**Total GPUs per engine:** `tp_size × pp_size = 2 × 2 = 4`

### Data Parallelism (DP)

Run multiple model replicas:

```yaml
generator:
  inference_engine_data_parallel_size: 2  # 2 DP replicas
```

### Expert Parallelism (EP) for MoE

Distribute experts across GPUs:

```yaml
generator:
  inference_engine_tensor_parallel_size: 4
  inference_engine_expert_parallel_size: 4
  engine_init_kwargs:
    moe_a2a_backend: "deepep"
    enable_eplb: true  # Expert load balancing
```

**Constraint:** `dp_size * tp_size == ep_size` when EP > 1

### Combined Configuration Example

```yaml
# 8 engines × (TP=2 × PP=2) = 32 GPUs
generator:
  backend: sglang
  num_inference_engines: 8
  inference_engine_tensor_parallel_size: 2
  inference_engine_pipeline_parallel_size: 2
  inference_engine_data_parallel_size: 1
```

---

## 7. Advanced Features

### 7.1 LoRA Support

Enable runtime LoRA loading/unloading:

```yaml
generator:
  backend: sglang
  enable_lora: true
  engine_init_kwargs:
    max_lora_rank: 64           # Maximum rank supported
    max_loras_per_batch: 8      # Max adapters per batch
    max_loaded_loras: 16        # Max loaded in memory (evicts LRU)
    lora_backend: "csgmv"       # Backend: csgmv, triton, ascend, torch_native
    lora_eviction_policy: "lru" # Eviction: lru or lfu
    lora_paths:                 # Pre-load adapters
      adapter1: "/path/to/lora1"
      adapter2: "/path/to/lora2"
```

**Supported LoRA Target Modules:**
- Attention: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `qkv_proj`
- FFN: `gate_proj`, `up_proj`, `down_proj`, `gate_up_proj`
- Embedding: `embed_tokens`, `lm_head`
- Special: `"all"` (applies to all supported modules)

**API:**
```python
# Load adapter at runtime
await engine.load_lora_adapter(
    lora_name="adapter1",
    lora_path="/path/to/lora",
    pinned=False  # If True, won't be evicted
)

# Unload adapter
await engine.unload_lora_adapter(lora_name="adapter1")
```

**Constraints:**
- `max_loaded_loras >= max_loras_per_batch`
- `len(lora_paths) <= max_loaded_loras`
- `max_lora_chunk_size` must be power of 2 (16-128)

### 7.2 HTTP Endpoints

Enable OpenAI-compatible API:

```yaml
generator:
  backend: sglang
  enable_http_endpoint: true
  http_endpoint_host: "0.0.0.0"
  http_endpoint_port: 8000
  async_engine: true  # Required
```

**Endpoints:**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | Chat completion |
| `/v1/completions` | POST | Text completion |
| `/health` | GET | Health check |

**Chat Completion Request:**
```json
{
  "model": "model-name",
  "messages": [
    {"role": "user", "content": "Hello"}
  ],
  "max_tokens": 1024,
  "temperature": 1.0,
  "top_p": 1.0,
  "top_k": -1,
  "stop": ["</s>"]
}
```

**Chat Completion Response:**
```json
{
  "id": "chatcmpl-1234567890",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "model-name",
  "choices": [{
    "index": 0,
    "message": {"role": "assistant", "content": "Hi there!"},
    "token_ids": [1, 2, 3],
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 5,
    "total_tokens": 15
  }
}
```

**Text Completion Request:**
```json
{
  "model": "model-name",
  "prompt": "Once upon a time",
  "max_tokens": 100
}
```

**Error Response Format:**
```json
{
  "error": {
    "message": "Error description",
    "type": "Bad Request",
    "code": 400,
    "param": null
  }
}
```

**Limitations:**
- Streaming (`stream: true`) not supported
- `n > 1` not supported for completions
- Session-based routing requires `session_id` for multi-turn

### 7.3 Stop Sequences

Stop sequences are automatically converted to token IDs:

```yaml
generator:
  sampling_params:
    stop: ["</s>", "\n\n", "```"]      # String-based
    stop_token_ids: [151643, 151645]   # Direct token IDs
    stop_regex: "\\d{4}-\\d{2}-\\d{2}" # Regex pattern
```

**Implementation (sglang_engine.py lines 402-418):**
```python
# Stop strings converted using external tokenizer
for stop_str in stop_strings:
    token_ids = self.tokenizer.encode(stop_str, add_special_tokens=False)
    stop_token_ids.append(token_ids[-1] if len(token_ids) > 1 else token_ids[0])
```

### 7.4 min_new_tokens Support

Force minimum generation length:

```yaml
generator:
  sampling_params:
    min_new_tokens: 10
    max_generate_length: 1024
```

**Note:** This works by suppressing EOS tokens until min_new_tokens is reached. The `eos_token_id` is automatically injected from the external tokenizer.

### 7.5 Structured Output

Constrain generation to specific formats:

```yaml
generator:
  sampling_params:
    # JSON Schema (mutually exclusive with regex/ebnf)
    json_schema: '{"type": "object", "properties": {"answer": {"type": "string"}}}'

    # Or regex pattern
    regex: "[A-Z][a-z]+ [0-9]+"

    # Or EBNF grammar
    ebnf: 'root ::= "yes" | "no"'
```

**Note:** Only one of `json_schema`, `regex`, or `ebnf` can be specified.

### 7.6 Sleep/Wake for Memory Efficiency

When `colocate_all=true`, inference engines can release memory during training:

```python
# Sleep (release memory)
await engine.sleep(
    tags=["weights", "kv_cache"],  # Memory tags to release
    timeout=60.0,                   # Timeout in seconds
    abort_first=True,               # Abort in-flight requests first
    drain_timeout=5.0               # Wait for requests to drain
)

# Wake (resume)
await engine.wake_up(
    tags=["weights", "kv_cache"],
    timeout=60.0
)
# Must sync weights after wake
await engine.update_named_weights(request)
```

**Note:** SGLang always releases all memory on sleep (no partial levels like vLLM). Weights must be re-synced after wake.

---

## 8. Training Loop Integration

### Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. GENERATION PHASE                                             │
│                                                                 │
│ Trainer.train()                                                 │
│   ├─ prepare_generator_input()                                  │
│   └─ Generator.generate()                                       │
│       └─ For each prompt: agent_loop()                         │
│           ├─ env.init(chat_history)                            │
│           └─ While not done:                                   │
│               ├─ InferenceEngineClient.generate()   ← SGLang   │
│               │   Returns: tokens, logprobs, stop_reason       │
│               ├─ env.step(output)                              │
│               └─ Update state                                  │
│                                                                 │
│ Output: response_ids, rewards, rollout_logprobs                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 2. FORWARD PASSES                                               │
│                                                                 │
│ fwd_logprobs_values_reward()                                   │
│   ├─ Policy model → action_log_probs                           │
│   ├─ Reference model → base_action_log_probs                   │
│   └─ Critic model → values                                     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 3. ADVANTAGE COMPUTATION                                        │
│                                                                 │
│ compute_advantages_and_returns()                               │
│   ├─ GAE: Generalized Advantage Estimation                     │
│   ├─ GRPO: Group Relative Policy Optimization                  │
│   └─ RLOO: REINFORCE Leave-One-Out                             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 4. TRAINING                                                     │
│                                                                 │
│ train_critic_and_policy()                                      │
│   └─ PPO loss with clipping                                    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 5. WEIGHT SYNC                                                  │
│                                                                 │
│ sync_policy_weights_to_inference_engines()                     │
│   └─ Broadcast updated weights to all SGLang engines           │
└─────────────────────────────────────────────────────────────────┘
```

### Supported Algorithms

| Algorithm | Config Value | Description |
|-----------|--------------|-------------|
| **GAE** | `adv_estimator: gae` | Generalized Advantage Estimation (requires critic) |
| **GRPO** | `adv_estimator: grpo` | Group-normalized advantages (no critic) |
| **RLOO** | `adv_estimator: rloo` | REINFORCE with leave-one-out baseline |
| **REINFORCE++** | `adv_estimator: reinforce_plus_plus` | Enhanced REINFORCE |

### Policy Loss Functions

| Loss | Config Value | Description |
|------|--------------|-------------|
| **PPO** | `policy_loss: regular` | Standard PPO clipping |
| **GSPO** | `policy_loss: gspo` | Sequence-level importance sampling |
| **SAPO** | `policy_loss: sapo` | Self-play advantage |

---

## 9. Performance Tuning

### Memory Optimization

```yaml
generator:
  gpu_memory_utilization: 0.85     # Increase if not OOM
  enable_prefix_caching: true      # RadixAttention for repeated prefixes

trainer:
  placement:
    colocate_all: true             # Enable sleep/wake cycles
```

### Throughput Optimization

```yaml
generator:
  num_inference_engines: 8         # More engines = more parallelism
  batched: true                    # Batch multiple prompts
  async_engine: true               # Async generation

  engine_init_kwargs:
    attention_backend: "fa3"       # FlashAttention 3
```

### Multi-Turn Optimization

```yaml
generator:
  enable_prefix_caching: true      # Reuse KV cache for conversation history

  # Session-based routing keeps conversations on same engine
  # (automatic when session_ids provided)
```

### Weight Sync Optimization

```yaml
# For colocated setups (fastest)
trainer:
  placement:
    colocate_all: true
generator:
  weight_sync_backend: nccl        # CUDA IPC zero-copy
  weight_transfer_threshold_cuda_ipc_GB: 0.5  # Batch weights up to 0.5GB
```

**Note:** For TP > 1 with broadcast strategy, CUDA IPC (`colocate_all=true`) is recommended for better performance.

---

## 10. Troubleshooting

### Common Issues

#### 1. "use_conversation_multi_turn=False is not supported"

**Error:**
```
NotImplementedError: `use_conversation_multi_turn=False` is not supported for SGLang backend
```

**Solution:** SGLang requires multi-turn mode. This is the default, so don't explicitly set it to false.

#### 2. Signal Handler Error in Ray

**Error:**
```
ValueError: signal only works in main thread
```

**Cause:** SGLang tries to register signal handlers, but Ray actors run in worker threads.

**Solution:** This is automatically patched (sglang_engine.py lines 46-111). If you see this error, ensure you're using the SkyRL SGLang engine wrapper.

#### 3. Weight Sync Timeout

**Error:**
```
TimeoutError: Weight sync timed out
```

**Solutions:**
- Increase timeout: `generator.weight_sync_timeout: 120`
- Check network connectivity between training and inference nodes
- Ensure NCCL ports are open

#### 4. Ray + uv Editable Install Issue

**Error:**
```
error: Failed to generate package metadata for `sglang @ editable+../../sglang/python`
  Caused by: Distribution not found at: file:///tmp/ray/session_.../runtime_resources/sglang/python
```

**Cause:** Ray's UV runtime env hook (`RAY_RUNTIME_ENV_HOOK`) tries to replicate the uv environment on Ray workers. When SGLang is installed as an editable package with a relative path, the path doesn't exist in Ray's runtime directory.

**Solution:** Unset the hook and use direct venv activation:
```bash
source .venv/bin/activate
unset RAY_RUNTIME_ENV_HOOK
python -m skyrl_train.entrypoints.main_base ...
```

#### 4b. FlashAttention v3 Not Supported

**Error:**
```
AssertionError: FlashAttention v3 Backend requires SM>=80 and SM<=90
```

**Cause:** Your GPU doesn't support FlashAttention v3 (requires A100, H100, or similar).

**Solution:** Use FlashInfer backend instead:
```bash
python -m skyrl_train.entrypoints.main_base \
  generator.backend=sglang \
  +generator.engine_init_kwargs.attention_backend=flashinfer \
  +generator.engine_init_kwargs.mm_attention_backend=flashinfer
```

#### 5. Multiple Stop Tokens Not Working

**Issue:** Generation stops early despite min_new_tokens.

**Cause:** Model may have multiple stop tokens (e.g., Qwen2.5 has both `<|im_end|>` and `<|endoftext|>`).

**Solution:** Specify all stop tokens:
```yaml
generator:
  sampling_params:
    min_new_tokens: 30
    stop_token_ids: [151643]  # Additional stop tokens to suppress
```

#### 6. Out of Memory

**Solutions:**
1. Reduce `gpu_memory_utilization`
2. Reduce `max_num_seqs`
3. Enable `colocate_all` for sleep/wake
4. Reduce batch size

#### 7. Missing Tokenizer Error

**Error:**
```
ValueError: tokenizer is required for SGLangInferenceEngine
```

**Cause:** Tokenizer not passed during engine initialization.

**Solution:** Ensure tokenizer is properly configured in your training config.

#### 8. LoRA Not Enabled Error

**Error:**
```
RuntimeError: LoRA is not enabled. Set enable_lora=True when creating the engine.
```

**Solution:** Add `enable_lora: true` to your generator config before calling LoRA methods.

#### 9. Weight Update Group Initialization Failed

**Error:**
```
RuntimeError: Failed to initialize weight update group: <message>
```

**Cause:** Distributed process group setup failed.

**Solutions:**
- Check NCCL/Gloo installation
- Verify network connectivity
- Check firewall rules for NCCL ports

#### 10. Sleep/Wake Timeout

**Error:**
```
RuntimeError: sleep timed out after 60s
RuntimeError: wake_up timed out after 60s
```

**Cause:** Memory operations taking too long.

**Solutions:**
- Increase timeout parameter
- Check for in-flight requests blocking sleep
- Reduce memory pressure

#### 11. IPC Weight Update Failed

**Error:**
```
RuntimeError: IPC weight update failed: <message>
```

**Cause:** CUDA IPC handle sharing failed.

**Solutions:**
- Ensure processes are on same node
- Check CUDA driver compatibility
- Try broadcast strategy instead

#### 12. SGLang with Megatron Experimental Warning

**Warning:**
```
SGLang backend with Megatron training is experimental.
```

**Note:** SGLang + Megatron is experimental. If issues occur, switch to vLLM backend.

#### 13. HTTP Endpoint Requires async_engine

**Error:**
```
ValueError: generator.async_engine must be True when generator.enable_http_endpoint==True
```

**Solution:** Set `async_engine: true` in your config.

### Debug Logging

Enable verbose logging:

```bash
export SGLANG_LOG_LEVEL=debug
export SKYRL_LOG_LEVEL=debug
```

### Checking Engine Status

```python
# From inference engine client
info = await engine.get_server_info()
print(info)  # Shows model, scheduler, version info
```

---

## 11. Advanced Configuration Reference

This section documents less commonly used configuration options.

### Fully Async Training Options

For fully asynchronous training mode:

```yaml
trainer:
  fully_async:
    enabled: false                          # Enable fully async mode
    max_staleness_steps: 4                  # Maximum weight staleness
    num_parallel_generation_workers: 768    # Parallel generation workers
    buffer_size: 1024                       # Sample buffer size
```

### Environment Configuration (skyrl_gym)

When using `environment.env_class: "skyrl_gym"`:

```yaml
environment:
  skyrl_gym:
    gym_name: "custom_env"              # Registered gym environment name
    max_episode_steps: 100              # Maximum steps per episode
    action_space_type: "discrete"       # discrete or continuous
    reward_scale: 1.0                   # Reward scaling factor
```

### LoRA Training Options

Extended LoRA configuration:

```yaml
trainer:
  lora:
    enabled: false                      # Enable LoRA training
    rank: 8                             # LoRA rank
    alpha: 16                           # LoRA alpha
    dropout: 0.0                        # LoRA dropout
    target_modules: ["q_proj", "v_proj"] # Target modules
    bias: "none"                        # "none", "all", or "lora_only"
```

### FSDP Options

Extended FSDP configuration:

```yaml
trainer:
  fsdp:
    sharding_strategy: "FULL_SHARD"     # FULL_SHARD, SHARD_GRAD_OP, NO_SHARD
    mixed_precision: "bf16"             # bf16, fp16, or fp32
    cpu_offload: false                  # Offload to CPU
    backward_prefetch: "BACKWARD_PRE"   # Prefetch strategy
    limit_all_gathers: true             # Limit concurrent all-gathers
```

### RoPE Configuration

For models using rotary position embeddings:

```yaml
trainer:
  policy:
    model:
      rope_scaling:
        type: "linear"                  # linear, dynamic, yarn
        factor: 2.0                     # Scaling factor
```

### Algorithm Variants

Extended algorithm configuration:

```yaml
trainer:
  algorithm:
    # GRPO specific
    grpo_norm_by_std: true              # Normalize by std in GRPO
    grpo_clip_ratio: 0.2                # GRPO clipping ratio

    # DAPO specific
    clip_lower: 0.8                     # Lower clip bound (DAPO)
    clip_upper: 1.28                    # Upper clip bound (DAPO)

    # Loss reduction
    loss_reduction: "token_mean"        # token_mean, sequence_mean, seq_mean_token_sum_norm

    # KL estimator
    kl_estimator: "k2"                  # k1, k2, k3, or abs
```

### Data Loading Options

```yaml
data:
  num_workers: 4                        # DataLoader workers
  prefetch_factor: 2                    # Prefetch batches
  pin_memory: true                      # Pin memory for GPU transfer
  drop_last: true                       # Drop incomplete batches
```

### Checkpointing Options

```yaml
trainer:
  ckpt:
    save_optimizer: true                # Save optimizer state
    save_scheduler: true                # Save scheduler state
    async_save: false                   # Async checkpoint saving
    max_checkpoints: 5                  # Maximum checkpoints to keep
```

### Logger Options

```yaml
trainer:
  logger: "wandb"                       # wandb, tensorboard, mlflow, swanlab, console
  logger_kwargs:
    tags: ["experiment"]                # W&B tags
    notes: "Training run"               # W&B notes
    group: "grpo-experiments"           # W&B group
```

---

## Appendix: File Reference

| Component | File | Key Lines |
|-----------|------|-----------|
| SGLang Engine | `inference_engines/sglang/sglang_engine.py` | 304-798 |
| Weight Loader | `inference_engines/sglang/sglang_engine.py` | 172-302 |
| Ray Wrapper | `inference_engines/ray_wrapped_inference_engine.py` | 233-343 |
| Engine Client | `inference_engines/inference_engine_client.py` | 31-150 |
| HTTP Endpoint | `inference_engines/inference_engine_client_http_endpoint.py` | 1-310 |
| Weight Sync Base | `weight_sync/base.py` | 1-90 |
| CUDA IPC Strategy | `weight_sync/cuda_ipc_strategy.py` | 90-234 |
| Broadcast Strategy | `weight_sync/broadcast_strategy.py` | 72-194 |
| Config Validation | `utils/utils.py` | 185-450 |
| Trainer | `trainer.py` | 152-348 |
| PPO Utils | `utils/ppo_utils.py` | 557-1111 |

---

## 12. Internal RL Training APIs

These HTTP endpoints are used internally by SkyRL for RL training. They're documented here for debugging and advanced use cases.

### 12.1 Weight Update Group Management

#### POST `/init_weights_update_group`

Initialize distributed weight synchronization group.

**Request:**
```json
{
  "master_address": "192.168.1.1",
  "master_port": 29500,
  "rank_offset": 0,
  "world_size": 8,
  "group_name": "weight_update_group",
  "backend": "nccl"
}
```

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `master_address` | str | required | Master node IP address |
| `master_port` | int | required | Master node port (typically 29500-29599) |
| `rank_offset` | int | 0 | Starting rank offset for this engine |
| `world_size` | int | required | Total processes in the group |
| `group_name` | str | "weight_update_group" | Unique group identifier |
| `backend` | str | "nccl" | Backend: "nccl" or "gloo" |

**Response:**
```json
{"success": true}
```

#### POST `/destroy_weights_update_group`

Destroy a weight update group.

**Request:**
```json
{
  "group_name": "weight_update_group"
}
```

### 12.2 Weight Synchronization Endpoints

#### POST `/update_weights_from_distributed`

Broadcast weights from training rank to inference engine.

**Request:**
```json
{
  "name": "weight_update_group",
  "dtype": "bfloat16",
  "flush_cache": true,
  "abort_all_requests": true,
  "weight_version": "step_100"
}
```

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | str | required | Process group name |
| `dtype` | str | required | Weight dtype ("bfloat16", "float16", "float32") |
| `flush_cache` | bool | true | Flush KV cache after update |
| `abort_all_requests` | bool | true | Abort in-flight requests |
| `weight_version` | str | null | Version tag for tracking |

#### POST `/update_weights_from_tensor`

Update weights directly from tensor data.

**Request:**
```json
{
  "tensor": "<base64 encoded tensor>",
  "load_format": "flattened_bucket",
  "dtype": "bfloat16",
  "flush_cache": true,
  "abort_all_requests": true
}
```

**Load Formats:**
| Format | Description |
|--------|-------------|
| `flattened_bucket` | Flattened tensor bucket (SkyRL default) |
| `direct` | Direct tensor assignment |
| `<custom>` | Custom weight loader name |

#### POST `/update_weights_from_ipc`

Update weights via CUDA IPC (zero-copy, same node only).

**Request:**
```json
{
  "zmq_handles": ["<ipc_handle_1>", "<ipc_handle_2>"],
  "flush_cache": true,
  "weight_version": "step_100"
}
```

**Note:** This is the fastest method when trainer and inference engine are colocated.

#### POST `/update_weight_version`

Update weight version tag without reloading weights.

**Request:**
```json
{
  "version": "step_100",
  "abort_all_requests": false
}
```

### 12.3 Memory Management Endpoints

#### POST `/release_memory_occupation`

Release GPU memory for training.

**Request:**
```json
{
  "tags": ["weights", "kv_cache"]
}
```

**Tags:**
| Tag | Description |
|-----|-------------|
| `weights` | Release model weights from GPU |
| `kv_cache` | Release KV cache |
| (null) | Release all memory |

**Note:** SGLang always releases all memory regardless of tags.

#### POST `/resume_memory_occupation`

Resume GPU memory occupation after release.

**Request:**
```json
{
  "tags": ["weights", "kv_cache"]
}
```

#### POST `/flush_cache`

Flush the RadixAttention prefix cache.

**Request:**
```json
{}
```

**When to use:**
- After weight updates if prefix cache is stale
- When memory pressure is high
- Before evaluation for consistent results

### 12.4 Pause/Continue Generation

#### POST `/pause_generation`

Pause inference for weight updates.

**Request:**
```json
{
  "mode": "abort"
}
```

**Modes:**
| Mode | Behavior | Use Case |
|------|----------|----------|
| `abort` | Abort all running requests immediately | Quick weight updates |
| `in_place` | Pause inference, keep KV cache, resume later | Preserve cache state |
| `retract` | Pause inference, retract requests to queue | Memory reclamation |

#### POST `/continue_generation`

Resume inference after pause.

**Request:**
```json
{}
```

### 12.5 Weight Version Tracking

#### GET `/get_weight_version`

Get current weight version.

**Response:**
```json
{
  "version": "step_100"
}
```

---

## 13. Multi-Node Distributed Training

### 13.1 Prerequisites

For multi-node training with SGLang:

1. **Network Configuration:**
   - All nodes must have network connectivity
   - NCCL ports (29400-29599) must be open
   - Shared filesystem or S3 for checkpoints

2. **Environment Variables:**
   ```bash
   export NCCL_TIMEOUT=1800        # 30 minutes for large models
   export NCCL_DEBUG=INFO          # Enable debugging
   export NCCL_IB_DISABLE=0        # Enable InfiniBand (if available)
   export NCCL_SOCKET_IFNAME=eth0  # Network interface
   ```

### 13.2 Multi-Node Configuration

```yaml
# multi_node_config.yaml
trainer:
  strategy: fsdp2
  placement:
    colocate_all: false  # Distributed across nodes
    policy_num_nodes: 2
    policy_num_gpus_per_node: 8

generator:
  backend: sglang
  num_inference_engines: 16  # Spread across nodes
  weight_sync_backend: nccl
  run_engines_locally: true
```

### 13.3 Launch Commands

**Node 0 (Master):**
```bash
ray start --head --port=6379 --num-gpus=8

python -m skyrl_train.entrypoints.main_base \
  +experiment=multi_node_grpo \
  trainer.placement.policy_num_nodes=2 \
  trainer.placement.policy_num_gpus_per_node=8
```

**Node 1 (Worker):**
```bash
ray start --address=<master_ip>:6379 --num-gpus=8
```

### 13.4 NCCL Troubleshooting

| Issue | Environment Variable | Fix |
|-------|---------------------|-----|
| Timeout | `NCCL_TIMEOUT=3600` | Increase timeout |
| InfiniBand issues | `NCCL_IB_DISABLE=1` | Disable IB |
| Socket interface | `NCCL_SOCKET_IFNAME=eth0` | Set interface |
| Debug | `NCCL_DEBUG=INFO` | Enable logging |

### 13.5 Multi-Node NCCL Test

Before training, verify NCCL connectivity:

```bash
python scripts/multi_node_nccl_test.py \
  --master-addr=<master_ip> \
  --master-port=29500 \
  --world-size=16 \
  --rank=<current_rank>
```

---

## 14. Performance Benchmarking

### 14.1 Key Metrics to Track

| Metric | Target | Description |
|--------|--------|-------------|
| Weight sync time | < 2s | Time to sync weights to all engines |
| Tokens/second | Model-dependent | Generation throughput |
| GPU utilization | > 80% | Training + inference combined |
| Memory usage | < 90% peak | Avoid OOM |

### 14.2 Profiling Commands

```bash
# Enable SGLang profiling
export SGLANG_ENABLE_PROFILER=1

# Enable memory tracking
python -m skyrl_train.entrypoints.main_base \
  trainer.policy.record_memory=true \
  trainer.dump_data_batch=true
```

### 14.3 Benchmark Script

```python
import time
from skyrl_train.inference_engines.sglang import SGLangInferenceEngine

# Measure generation throughput
start = time.time()
outputs = await engine.generate(prompts, sampling_params)
elapsed = time.time() - start

total_tokens = sum(len(o.output_ids) for o in outputs)
print(f"Throughput: {total_tokens / elapsed:.1f} tokens/sec")
```

---

## 15. Generator Modes and Step-Wise Trajectories

### 15.1 Generator Mode Overview

SkyRL supports two fundamental generation modes:

| Mode | Flag | Use Case | Performance |
|------|------|----------|-------------|
| **Batched** | `batched: true` | Single-turn, high throughput | Faster |
| **Non-Batched** | `batched: false` | Multi-turn, agent loops | More flexible |

```yaml
# Batched mode (default for single-turn)
generator:
  batched: true
  use_conversation_multi_turn: false  # Single-turn only

# Non-batched mode (required for multi-turn)
generator:
  batched: false
  use_conversation_multi_turn: true
  max_turns: 6
```

### 15.2 Batched vs Non-Batched Generation

#### Batched Generation

Uses SGLang's offline synchronous engine for maximum throughput:

```python
# Internal: Uses offline batch generation
engine = sgl.Engine(model_path, ...)
outputs = engine.generate(prompts, sampling_params)  # Synchronous batch
```

**Advantages:**
- Higher throughput (batch parallelism)
- Lower latency per token
- Simpler scheduling

**Limitations:**
- Single-turn only
- No environment interaction during generation
- Cannot use custom chat templates at runtime

#### Non-Batched Generation

Uses async generation with environment stepping:

```python
# Internal: Uses async generation per prompt
async def agent_loop(prompt, env):
    for turn in range(max_turns):
        response = await engine.async_generate(...)
        observation = await env.step(response)
        if done:
            break
```

**Advantages:**
- Multi-turn conversation support
- Environment interaction between turns
- Tool/action execution
- Step-wise reward assignment

**Limitations:**
- Lower throughput
- More complex scheduling
- Requires async engine

### 15.3 Step-Wise Trajectories

Step-wise training enables **token-level rewards** at each turn:

```yaml
generator:
  step_wise_trajectories: true  # Enable step-wise
  batched: false                 # Required
  use_conversation_multi_turn: true
  async_engine: true            # Required
  max_turns: 6
```

#### How Step-Wise Training Works

```
Turn 1: [Prompt] → Model generates → Environment executes → Reward₁
Turn 2: [Context + Result₁] → Model generates → Environment executes → Reward₂
Turn 3: [Context + Result₂] → Model generates → Final result → Reward₃

Total training signal: Reward₁ + Reward₂ + Reward₃ (per-turn)
```

**vs Standard Multi-Turn:**
```
[All turns generated] → Final evaluation → Single Reward

Total training signal: Reward_final (end-only)
```

#### Step-Wise Output Structure

```python
# Step-wise generates StepWiseOutput
@dataclass
class StepWiseOutput:
    step_outputs: List[StepOutput]  # One per turn

@dataclass
class StepOutput:
    response_ids: List[int]         # Tokens for this turn
    prompt_ids: List[int]           # Input context
    loss_mask: List[int]            # Which tokens to train on
    rollout_logprobs: List[float]   # Log probs for each token
    per_token_rewards: List[float]  # Reward per token (usually 0 except end)
```

#### Configuration Requirements

Step-wise training has specific requirements:

```yaml
generator:
  step_wise_trajectories: true
  batched: false                    # REQUIRED
  use_conversation_multi_turn: true # REQUIRED
  async_engine: true                # REQUIRED
  chat_template: null               # Custom templates not supported
```

**Validation rules (from `_validate_cfg`):**
1. `step_wise_trajectories=true` requires `batched=false`
2. `step_wise_trajectories=true` requires `use_conversation_multi_turn=true`
3. `step_wise_trajectories=true` cannot use custom chat templates

### 15.4 Multi-Turn Generation Flow

For non-batched multi-turn generation:

```
┌─────────────────────────────────────────────────────────┐
│                    Agent Loop                            │
├─────────────────────────────────────────────────────────┤
│  1. Initialize: input_ids = tokenize(prompt)            │
│                                                          │
│  2. For each turn:                                       │
│     ├─ Generate: response = engine.generate(input_ids)   │
│     ├─ Parse: action = parse_action(response)            │
│     ├─ Execute: obs, reward, done = env.step(action)     │
│     ├─ Update: input_ids += response + observation       │
│     └─ If done: break                                    │
│                                                          │
│  3. Return: trajectory with all turns + rewards          │
└─────────────────────────────────────────────────────────┘
```

### 15.5 When to Use Each Mode

| Scenario | Recommended Mode | Reason |
|----------|------------------|--------|
| Math problems (GSM8K) | Batched | Single answer, high throughput |
| Code debugging | Non-batched + step-wise | Multi-turn refinement |
| Text-to-SQL | Non-batched + step-wise | Execute queries, iterate |
| Search/retrieval | Non-batched | Tool calls between turns |
| Simple QA | Batched | Single-turn, fast |
| Multi-hop reasoning | Non-batched | Multiple tool calls |

### 15.6 Performance Comparison

| Metric | Batched | Non-Batched | Step-Wise |
|--------|---------|-------------|-----------|
| Throughput | High | Medium | Lower |
| Memory | Lower | Medium | Higher |
| Training signal | End-only | End-only | Per-turn |
| Use case | Single-turn | Multi-turn | Fine-grained |

### 15.7 Trajectory ID Requirement

Step-wise training requires `trajectory_ids` for proper reward assignment:

```python
# Dataset must include trajectory_ids for step-wise
{
    "prompt": [...],
    "env_class": "text2sql",
    "reward_spec": {...},
    "trajectory_id": "unique_id_123"  # Required for step-wise
}
```

This enables tracking rewards across turns within the same trajectory.

---

## 16. Dynamic Sampling Strategies

### 16.1 Overview

Dynamic sampling adjusts batch composition during training:

| Strategy | Description | Use Case |
|----------|-------------|----------|
| `none` | No dynamic sampling | Standard training |
| `filter` | Remove low-quality samples | Quality-focused |
| `replace` | Replace samples dynamically | Curriculum learning |

```yaml
trainer:
  algorithm:
    dynamic_sampling:
      type: "filter"  # or "none", "replace"
      max_sample_batches: 30
```

### 16.2 Filter Strategy

Removes samples that don't meet quality criteria:

```yaml
trainer:
  algorithm:
    dynamic_sampling:
      type: "filter"
      min_advantage_threshold: -0.5  # Remove very negative advantages
      max_attempts: 30
```

**How it works:**
1. Generate batch of samples
2. Compute advantages for each
3. Filter out samples below threshold
4. If not enough samples, generate more
5. Error if max_sample_batches reached

### 16.3 Replace Strategy

Dynamically replaces samples during training:

```yaml
trainer:
  algorithm:
    dynamic_sampling:
      type: "replace"
      replacement_ratio: 0.2  # Replace 20% of samples
```

### 16.4 DAPO Dynamic Sampling

DAPO algorithm uses specialized dynamic sampling:

```yaml
trainer:
  algorithm:
    advantage_estimator: "dapo"
    dynamic_sampling:
      type: "filter"
      max_sample_batches: 50  # Higher for difficult tasks
```

**Error handling:**
```
RuntimeError: Exiting training due to hitting dynamic sampling limit
```

This occurs when filter strategy cannot find enough good samples. Solutions:
1. Increase `max_sample_batches`
2. Lower quality threshold
3. Disable dynamic sampling (`type: "none"`)
4. Check if task is too difficult for current model

---

## References

- [SGLang Documentation](https://docs.sglang.io/)
- [SkyRL Documentation](https://github.com/NovaSky-AI/SkyRL)
- [Ray Documentation](https://docs.ray.io/)
