# SGLang API Reference for SkyRL Integration

**Complete API documentation for SGLang methods used in SkyRL RL training.**

---

## Overview

SkyRL uses SGLang's Engine class for inference during RL training. This document covers:
1. Weight synchronization APIs
2. Memory management (sleep/wake)
3. Generation APIs
4. Process group management

---

## 1. Weight Synchronization APIs

### 1.1 update_weights_from_tensor

Update model weights directly from PyTorch tensors. **Fastest method for colocated training.**

```python
engine.update_weights_from_tensor(
    named_tensors: List[Tuple[str, torch.Tensor]],
    load_format: Optional[str] = None,
    flush_cache: bool = True
)
```

**Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `named_tensors` | `List[Tuple[str, Tensor]]` | List of (parameter_name, tensor) pairs |
| `load_format` | `Optional[str]` | `None` (standard) or `"flattened_bucket"` |
| `flush_cache` | `bool` | Clear KV cache after update (default: True) |

**Example:**
```python
# After training step, update weights
weights = [
    ("model.layers.0.self_attn.q_proj.weight", q_proj_tensor),
    ("model.layers.0.self_attn.k_proj.weight", k_proj_tensor),
    ("model.layers.0.self_attn.v_proj.weight", v_proj_tensor),
]
engine.update_weights_from_tensor(weights, flush_cache=True)
```

**SkyRL Usage:** Used by `CudaIpcTransferStrategy` for zero-copy weight updates.

---

### 1.2 update_weights_from_distributed

Update weights via distributed process group (NCCL/Gloo broadcast).

```python
engine.update_weights_from_distributed(
    names: List[str],
    dtypes: List[str],
    shapes: List[List[int]],
    group_name: str = "weight_update_group",
    flush_cache: bool = True,
    load_format: Optional[str] = None
)
```

**Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `names` | `List[str]` | Parameter names to update |
| `dtypes` | `List[str]` | Data types (e.g., `["bfloat16", "bfloat16"]`) |
| `shapes` | `List[List[int]]` | Tensor shapes (e.g., `[[4096, 4096], [4096, 4096]]`) |
| `group_name` | `str` | Process group name (default: `"weight_update_group"`) |
| `flush_cache` | `bool` | Clear KV cache after update |
| `load_format` | `Optional[str]` | Weight format |

**Example:**
```python
# Initialize process group first
engine.init_weights_update_group(
    master_address="localhost",
    master_port=29500,
    rank_offset=0,
    world_size=9,  # 8 training ranks + 1 SGLang
    group_name="skyrl",
    backend="nccl"
)

# Broadcast weights from training rank 0
engine.update_weights_from_distributed(
    names=["model.layers.0.self_attn.q_proj.weight"],
    dtypes=["bfloat16"],
    shapes=[[4096, 4096]],
    group_name="skyrl",
    flush_cache=True
)
```

**SkyRL Usage:** Used by `BroadcastTransferStrategy` for multi-node setups.

---

### 1.3 init_weights_update_group

Initialize a distributed process group for weight synchronization.

```python
engine.init_weights_update_group(
    master_address: str,
    master_port: int,
    rank_offset: int,
    world_size: int,
    group_name: str,
    backend: str = "nccl"
)
```

**Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `master_address` | `str` | Master node IP/hostname |
| `master_port` | `int` | Master node port |
| `rank_offset` | `int` | Rank offset for this group |
| `world_size` | `int` | Total processes (training + inference) |
| `group_name` | `str` | Unique group identifier |
| `backend` | `str` | `"nccl"` (GPU) or `"gloo"` (CPU) |

**SkyRL Configuration:**
```yaml
generator:
  weight_sync_backend: nccl  # or gloo
```

---

### 1.4 destroy_weights_update_group

Clean up process group after training.

```python
engine.destroy_weights_update_group(group_name: str)
```

---

### 1.5 update_weights_from_disk

Load weights from a checkpoint file/directory.

```python
engine.update_weights_from_disk(
    model_path: str,
    load_format: Optional[str] = None
)
```

**Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `model_path` | `str` | Path to checkpoint |
| `load_format` | `Optional[str]` | `"auto"`, `"pt"`, `"safetensors"`, etc. |

---

## 2. Memory Management APIs

### 2.1 release_memory_occupation (Sleep)

Release GPU memory for other operations (e.g., training).

```python
engine.release_memory_occupation(
    tags: Optional[List[str]] = None
)
```

**Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `tags` | `Optional[List[str]]` | Memory regions to release |

**Available Tags:**
- `"weights"` - Model weights
- `"kv_cache"` - KV cache
- `"cuda_graph"` - CUDA graphs

**Example:**
```python
# Release all memory for training
engine.release_memory_occupation(tags=["weights", "kv_cache", "cuda_graph"])

# ... perform training ...

# Resume and re-sync weights
engine.resume_memory_occupation(tags=["weights", "kv_cache", "cuda_graph"])
engine.update_weights_from_tensor(new_weights)
```

**SkyRL Configuration:**
```yaml
trainer:
  placement:
    colocate_all: true  # Enables sleep/wake cycles
```

---

### 2.2 resume_memory_occupation (Wake)

Restore GPU memory occupation after release.

```python
engine.resume_memory_occupation(
    tags: Optional[List[str]] = None
)
```

**Important:** After waking, you MUST re-sync weights since SGLang discards them during sleep.

---

### 2.3 flush_cache

Clear KV cache to free memory.

```python
engine.flush_cache()
```

**Usage:** Called automatically after weight updates when `flush_cache=True`.

---

## 3. Generation APIs

### 3.1 generate

Generate text completions.

```python
result = engine.generate(
    prompt: Optional[Union[List[str], str]] = None,
    sampling_params: Optional[Union[List[Dict], Dict]] = None,
    input_ids: Optional[Union[List[List[int]], List[int]]] = None,
    return_logprob: Optional[Union[List[bool], bool]] = None,
    logprob_start_len: Optional[Union[List[int], int]] = None,
    top_logprobs_num: Optional[Union[List[int], int]] = None,
    return_hidden_states: bool = False,
    stream: bool = False,
    lora_path: Optional[List[Optional[str]]] = None,
)
```

**Key Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `prompt` | `str` or `List[str]` | Text prompt(s) |
| `input_ids` | `List[int]` | Token IDs (alternative to prompt) |
| `sampling_params` | `Dict` | Sampling configuration |
| `return_logprob` | `bool` | Return token log probabilities |
| `return_hidden_states` | `bool` | Return hidden states |
| `stream` | `bool` | Stream output tokens |
| `lora_path` | `str` | LoRA adapter to use |

**SkyRL Usage:**
```python
# SkyRL always uses input_ids (token-in-token-out mode)
result = engine.generate(
    input_ids=prompt_token_ids,
    sampling_params={
        "max_new_tokens": 1024,
        "temperature": 1.0,
        "return_logprob": True,
    }
)
```

**Return Structure:**
```python
{
    "text": "Generated text...",
    "meta_info": {
        "prompt_tokens": 50,
        "completion_tokens": 100,
        "cached_tokens": 20,
    },
    "finish_reason": "stop",  # or "length", "stop_str"
    "output_token_logprobs": [...],  # If return_logprob=True
}
```

---

### 3.2 Sampling Parameters

Complete sampling configuration:

```python
sampling_params = {
    # Length
    "max_new_tokens": 1024,
    "min_new_tokens": 0,

    # Temperature
    "temperature": 1.0,        # 0 = greedy, higher = more random
    "top_p": 1.0,              # Nucleus sampling [0, 1]
    "top_k": -1,               # Top-k (-1 = disabled)
    "min_p": 0.0,              # Minimum probability [0, 1]

    # Penalties
    "frequency_penalty": 0.0,  # [-2, 2]
    "presence_penalty": 0.0,   # [-2, 2]
    "repetition_penalty": 1.0, # [0, 2]

    # Stop conditions
    "stop": ["</s>"],          # Stop strings
    "stop_token_ids": [2],     # Stop token IDs
    "ignore_eos": False,

    # Structured output (mutually exclusive)
    "json_schema": None,
    "regex": None,
    "ebnf": None,

    # Logprobs
    "return_logprob": True,
    "top_logprobs_num": 5,

    # Reproducibility
    "sampling_seed": 42,
}
```

---

## 4. LoRA Management

### 4.1 load_lora_adapter

Load a LoRA adapter at runtime.

```python
engine.load_lora_adapter(
    lora_name: str,
    lora_path: str,
    pinned: bool = False
)
```

**Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `lora_name` | `str` | Unique adapter name |
| `lora_path` | `str` | Path to LoRA weights |
| `pinned` | `bool` | Keep in memory (won't evict) |

**SkyRL Configuration:**
```yaml
generator:
  enable_lora: true
  engine_init_kwargs:
    max_lora_rank: 64
    max_loras_per_batch: 8
    lora_backend: "csgmv"
```

---

### 4.2 unload_lora_adapter

Unload a LoRA adapter.

```python
engine.unload_lora_adapter(lora_name: str)
```

---

## 5. Engine Initialization

### 5.1 Constructor Parameters

Key parameters for RL training:

```python
from sglang import Engine

engine = Engine(
    # Model
    model_path="meta-llama/Llama-3.1-8B-Instruct",
    dtype="bfloat16",

    # Memory
    mem_fraction_static=0.8,
    max_running_requests=1024,

    # Parallelism
    tp_size=2,
    pp_size=1,
    dp_size=1,
    ep_size=1,

    # RL-specific
    skip_tokenizer_init=True,  # Token-in-token-out
    enable_memory_saver=True,  # Sleep/wake support
    custom_weight_loader=["path/to/loader.py"],

    # Performance
    attention_backend="flashinfer",
    disable_radix_cache=False,  # Enable prefix caching

    # LoRA
    enable_lora=True,
    max_lora_rank=64,
)
```

**SkyRL Configuration Mapping:**

| SkyRL Config | SGLang Parameter |
|--------------|------------------|
| `gpu_memory_utilization` | `mem_fraction_static` |
| `max_num_seqs` | `max_running_requests` |
| `max_num_batched_tokens` | `max_prefill_tokens` |
| `enable_prefix_caching` | `disable_radix_cache` (inverted) |
| `inference_engine_tensor_parallel_size` | `tp_size` |
| `inference_engine_pipeline_parallel_size` | `pp_size` |

---

## 6. Server Information

### 6.1 get_server_info

Get runtime engine information.

```python
info = engine.get_server_info()
```

**Return Structure:**
```python
{
    "model_path": "meta-llama/Llama-3.1-8B-Instruct",
    "tp_size": 2,
    "pp_size": 1,
    "max_running_requests": 1024,
    "version": "0.4.0",
    ...
}
```

---

## 7. Lifecycle

### 7.1 shutdown

Gracefully shutdown the engine.

```python
engine.shutdown()
```

### 7.2 Context Manager

```python
with Engine(model_path="...") as engine:
    result = engine.generate("Hello")
# shutdown() called automatically
```

---

## 8. SkyRL Integration Example

Complete example of SGLang usage in SkyRL training:

```python
from sglang import Engine
import torch

class RLTrainer:
    def __init__(self, model_path: str):
        # Initialize SGLang engine
        self.engine = Engine(
            model_path=model_path,
            skip_tokenizer_init=True,
            mem_fraction_static=0.7,
            enable_memory_saver=True,
        )

        # Initialize weight sync group
        self.engine.init_weights_update_group(
            master_address="localhost",
            master_port=29500,
            rank_offset=0,
            world_size=5,
            group_name="skyrl",
            backend="nccl"
        )

    def generate_rollouts(self, prompts: list):
        """Generate responses for RL training."""
        results = self.engine.generate(
            input_ids=prompts,
            sampling_params={
                "max_new_tokens": 1024,
                "temperature": 1.0,
                "return_logprob": True,
            }
        )
        return results

    def update_weights(self, weights: dict):
        """Sync weights from training to inference."""
        named_tensors = [(k, v) for k, v in weights.items()]
        self.engine.update_weights_from_tensor(
            named_tensors=named_tensors,
            flush_cache=True
        )

    def train_step(self, batch):
        # 1. Generate rollouts
        outputs = self.generate_rollouts(batch["prompts"])

        # 2. Compute rewards
        rewards = self.compute_rewards(outputs, batch["ground_truth"])

        # 3. Policy gradient update (your training code)
        new_weights = self.policy_gradient_step(outputs, rewards)

        # 4. Sync weights to SGLang
        self.update_weights(new_weights)

        return rewards

# Usage
trainer = RLTrainer("Qwen/Qwen2.5-0.5B-Instruct")
for batch in dataloader:
    rewards = trainer.train_step(batch)
```

---

## 9. Parallelism Query APIs

Query the engine's parallelism configuration at runtime.

### 9.1 tp_size

Get the tensor parallel size.

```python
size = engine.tp_size()
# Returns: int (e.g., 2 for TP=2)
```

### 9.2 pp_size

Get the pipeline parallel size.

```python
size = engine.pp_size()
# Returns: int (e.g., 1 for no PP)
```

### 9.3 dp_size

Get the data parallel size.

```python
size = engine.dp_size()
# Returns: int (e.g., 1 for single replica)
```

### 9.4 ep_size

Get the expert parallel size (for MoE models).

```python
size = engine.ep_size()
# Returns: int (e.g., 4 for EP=4)
```

**Example Usage:**
```python
# Verify parallelism configuration
info = f"TP={engine.tp_size()}, PP={engine.pp_size()}, DP={engine.dp_size()}"
if hasattr(engine, 'ep_size'):
    info += f", EP={engine.ep_size()}"
print(f"Engine parallelism: {info}")
```

---

## 10. Cache Management APIs

### 10.1 reset_prefix_cache

Clear the RadixAttention prefix cache.

```python
engine.reset_prefix_cache()
```

**When to Use:**
- After significant prompt distribution changes
- To reclaim memory during training
- Before evaluation with different prompt patterns

**Note:** Prefix cache is automatically managed in most cases. Only call explicitly when needed.

### 10.2 get_cache_stats

Get cache utilization statistics (if available).

```python
stats = engine.get_server_info()
# Check cache-related fields in server info
```

---

## 11. Request Management APIs

### 11.1 abort_generation

Abort ongoing generation requests.

```python
await engine.abort_generation(request_id: str)
```

**Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `request_id` | `str` | ID of request to abort |

**When to Use:**
- Cancel long-running requests
- Implement timeout handling
- Clean up after errors

**Example:**
```python
# With timeout handling
import asyncio

async def generate_with_timeout(engine, prompts, timeout=30):
    request_id = await engine.generate_async(prompts)
    try:
        result = await asyncio.wait_for(
            engine.get_result(request_id),
            timeout=timeout
        )
        return result
    except asyncio.TimeoutError:
        await engine.abort_generation(request_id)
        raise
```

---

## 12. Weight Management Extended APIs

### 12.1 get_weights_by_name

Retrieve specific weight tensors from the model.

```python
weights = engine.get_weights_by_name(
    names: List[str],
    dtype: Optional[str] = None
)
```

**Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `names` | `List[str]` | Parameter names to retrieve |
| `dtype` | `Optional[str]` | Convert to dtype (e.g., "float32") |

**Returns:** Dictionary mapping names to tensors.

**Example:**
```python
# Get specific layer weights
weights = engine.get_weights_by_name([
    "model.layers.0.self_attn.q_proj.weight",
    "model.layers.0.self_attn.k_proj.weight"
])
for name, tensor in weights.items():
    print(f"{name}: shape={tensor.shape}, dtype={tensor.dtype}")
```

### 12.2 get_weight_names

List all model weight parameter names.

```python
names = engine.get_weight_names()
# Returns: List[str] of all parameter names
```

**Example:**
```python
# Inspect model structure
all_names = engine.get_weight_names()
layer_names = [n for n in all_names if "layers.0" in n]
print(f"Layer 0 parameters: {layer_names}")
```

---

## 13. Engine State APIs

### 13.1 is_sleeping

Check if engine is in sleep state (memory released).

```python
is_asleep = engine.is_sleeping()
# Returns: bool
```

### 13.2 get_num_unfinished_requests

Get count of pending requests.

```python
count = engine.get_num_unfinished_requests()
# Returns: int
```

**Example:**
```python
# Wait for all requests to complete
while engine.get_num_unfinished_requests() > 0:
    time.sleep(0.1)
print("All requests completed")
```

---

## 14. Async Generation APIs

### 14.1 generate_async

Non-blocking generation (returns immediately).

```python
request_id = await engine.generate_async(
    input_ids: List[List[int]],
    sampling_params: Dict,
    **kwargs
)
# Returns: request_id (str)
```

### 14.2 get_result

Retrieve result for an async request.

```python
result = await engine.get_result(request_id: str)
```

**Complete Async Example:**
```python
async def batch_generate_async(engine, batches):
    """Generate multiple batches concurrently."""
    request_ids = []

    # Submit all batches
    for batch in batches:
        req_id = await engine.generate_async(
            input_ids=batch["input_ids"],
            sampling_params={"max_new_tokens": 512}
        )
        request_ids.append(req_id)

    # Collect results
    results = []
    for req_id in request_ids:
        result = await engine.get_result(req_id)
        results.append(result)

    return results
```

---

## 15. Generation Control APIs

### 15.1 pause_generation

Pause ongoing generation for weight updates or maintenance.

```python
await engine.pause_generation(mode: str = "abort")
```

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mode` | `str` | `"abort"` | Pause mode (see below) |

**Pause Modes:**

| Mode | Behavior | Use Case |
|------|----------|----------|
| `"abort"` | Abort all running requests immediately | Quick weight updates |
| `"in_place"` | Pause inference, keep KV cache, resume later | Preserve cache state |
| `"retract"` | Pause inference, retract requests to queue, can flush cache | Memory reclamation |

**Example:**
```python
# Pause for weight update
await engine.pause_generation(mode="abort")

# Update weights
await engine.update_weights_from_tensor(new_weights)

# Resume generation
await engine.continue_generation()
```

### 15.2 continue_generation

Resume generation after pause.

```python
await engine.continue_generation()
```

**Example:**
```python
# Pause, update, resume pattern
await engine.pause_generation()
await engine.update_weights_from_tensor(weights)
await engine.continue_generation()
```

---

## 16. Weight Versioning APIs

Track weight versions for debugging and coordination.

### 16.1 get_weight_version

Get current weight version identifier.

```python
version = await engine.get_weight_version()
# Returns: str (e.g., "v1", "step_100", etc.)
```

### 16.2 update_weight_version

Update the weight version identifier.

```python
await engine.update_weight_version(
    new_version: str,
    abort_all_requests: bool = True
)
```

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `new_version` | `str` | Required | New version identifier |
| `abort_all_requests` | `bool` | `True` | Abort requests during update |

**Example:**
```python
# Track versions during training
for step in range(num_steps):
    # Training step...
    await engine.update_weight_version(f"step_{step}")
    await engine.update_weights_from_tensor(new_weights)
```

### 16.3 Weight Update with Version

All weight update methods support optional versioning:

```python
await engine.update_weights_from_tensor(
    named_tensors=weights,
    flush_cache=True,
    weight_version="step_100"  # Optional version tracking
)
```

---

## 17. Custom Weight Loaders

For advanced weight loading scenarios.

### 17.1 Registering a Custom Loader

```python
# In your custom loader module
def my_custom_weight_loader(model, named_tensors):
    """
    Custom weight loader function.

    Args:
        model: The model to update
        named_tensors: Serialized weight data
    """
    # Deserialize and load weights
    for name, tensor_data in named_tensors:
        param = get_param_by_name(model, name)
        param.data.copy_(deserialize(tensor_data))
```

### 17.2 Using Custom Loader

```python
engine = Engine(
    model_path="...",
    custom_weight_loader=["my_module.my_custom_weight_loader"]
)
```

### 17.3 Built-in Load Formats

| Format | Description |
|--------|-------------|
| `None` (default) | Standard tensor update |
| `"flattened_bucket"` | Batched update with FlattenedTensorBucket |
| `"direct"` | Direct tensor assignment |

---

## 18. OpenAI-Compatible API

SkyRL's inference engines support OpenAI-compatible APIs for chat completions and text completions.

### 18.1 chat_completion

Handle OpenAI-compatible chat completion requests with tool calling support.

```python
response = await engine.chat_completion({
    "json": {
        "model": "model-name",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What's the weather in Tokyo?"}
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the current weather in a location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string", "description": "City name"},
                            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                        },
                        "required": ["location"]
                    }
                }
            }
        ],
        "tool_choice": "auto",
        "max_tokens": 1024,
        "temperature": 0.7
    }
})
```

**Request Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `messages` | `List[Dict]` | Required | Conversation messages |
| `model` | `str` | Model path | Model identifier |
| `tools` | `List[Dict]` | `None` | List of tool/function definitions |
| `tool_choice` | `str/Dict` | `"auto"` | Tool selection mode |
| `max_tokens` | `int` | `1024` | Max completion tokens |
| `temperature` | `float` | `1.0` | Sampling temperature |
| `top_p` | `float` | `1.0` | Nucleus sampling parameter |
| `n` | `int` | `1` | Number of completions |

**Tool Choice Options:**
- `"auto"`: Model decides whether to call tools
- `"none"`: Never call tools
- `"required"`: Must call at least one tool
- `{"type": "function", "function": {"name": "func_name"}}`: Force specific function

**Response Structure:**
```python
{
    "id": "chatcmpl-123456",
    "object": "chat.completion",
    "created": 1234567890,
    "model": "model-name",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": null,  # May be null when tool_calls present
                "tool_calls": [
                    {
                        "id": "call_abc123",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": "{\"location\": \"Tokyo\", \"unit\": \"celsius\"}"
                        }
                    }
                ]
            },
            "finish_reason": "tool_calls"  # "stop" for regular completions
        }
    ],
    "usage": {
        "prompt_tokens": 50,
        "completion_tokens": 30,
        "total_tokens": 80
    }
}
```

### 18.2 Tool Calling Workflow

Complete example of multi-turn tool calling:

```python
# 1. Initial request with tools
messages = [{"role": "user", "content": "What's the weather in Tokyo?"}]
tools = [{"type": "function", "function": {"name": "get_weather", ...}}]

response = await engine.chat_completion({
    "json": {"messages": messages, "tools": tools}
})

choice = response["choices"][0]
if choice.get("finish_reason") == "tool_calls":
    # 2. Execute the tool(s)
    for tool_call in choice["message"]["tool_calls"]:
        func_name = tool_call["function"]["name"]
        func_args = json.loads(tool_call["function"]["arguments"])

        # Call your actual function
        result = get_weather(**func_args)

        # 3. Add tool result to messages
        messages.append(choice["message"])  # Assistant's tool call
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call["id"],
            "content": json.dumps(result)
        })

    # 4. Get final response
    final_response = await engine.chat_completion({
        "json": {"messages": messages, "tools": tools}
    })
```

### 18.3 Tool Parsing (SGLang)

SGLang's implementation includes automatic tool call parsing that supports multiple formats:

1. **JSON Format**: `{"name": "func", "arguments": {...}}`
2. **XML Format**: `<tool_call>{"name": "func", "arguments": {...}}</tool_call>`
3. **Function Call Syntax**: `func_name({"arg": "value"})`

### 18.4 Tool Calling (vLLM)

vLLM uses its native OpenAI serving layer with `enable_auto_tools=True`:

```python
# vLLM automatically handles tool calling via its OpenAI-compatible layer
response = await engine.chat_completion({
    "json": {
        "messages": messages,
        "tools": tools,
        "tool_choice": "auto"
    }
})
```

### 18.5 Backend Comparison

| Feature | SGLang | vLLM |
|---------|--------|------|
| Tool calling | ✅ Custom parser | ✅ Native OpenAI layer |
| Multiple formats | ✅ JSON/XML/Function | ✅ Model-dependent |
| Auto tool detection | ✅ | ✅ (enable_auto_tools) |
| Parallel tool calls | ✅ | ✅ |

---

## 19. Multimodal (Vision) Support

SkyRL supports vision-language models (VLMs) for multimodal RL training.

### 19.1 Multimodal via generate()

Pass image data directly to the generate method:

```python
# Using generate() with image_data
output = await engine.generate({
    "prompt_token_ids": [token_ids],
    "sampling_params": {"max_new_tokens": 256},
    "image_data": [
        "path/to/image.jpg",  # File path
        # or: "https://example.com/image.png"  # URL
        # or: base64_encoded_bytes  # Base64 bytes
        # or: pil_image  # PIL Image object
    ]
})
```

**Supported Image Formats:**

| Format | Description |
|--------|-------------|
| File path | Local path: `"/path/to/image.jpg"` |
| URL | HTTP(S) URL: `"https://example.com/image.png"` |
| Base64 | Raw bytes from `base64.b64decode()` |
| PIL Image | `PIL.Image.Image` object |

### 19.2 Multimodal via chat_completion()

Use OpenAI-compatible multimodal message format:

```python
response = await engine.chat_completion({
    "json": {
        "model": "llava-v1.6-mistral-7b",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's in this image?"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/jpeg;base64,/9j/4AAQSkZJRg..."
                        }
                    }
                ]
            }
        ],
        "max_tokens": 256
    }
})
```

### 19.3 Multiple Images Per Message

```python
# Multiple images in a single message
messages = [{
    "role": "user",
    "content": [
        {"type": "text", "text": "Compare these two images:"},
        {"type": "image_url", "image_url": {"url": "path/to/image1.jpg"}},
        {"type": "image_url", "image_url": {"url": "path/to/image2.jpg"}},
        {"type": "text", "text": "What are the differences?"}
    ]
}]
```

### 19.4 Supported Vision Models

| Model Family | Examples | Backend Support |
|--------------|----------|-----------------|
| LLaVA | llava-v1.6-mistral-7b, llava-v1.5-13b | SGLang, vLLM |
| Qwen-VL | Qwen2-VL-7B, Qwen-VL-Chat | SGLang, vLLM |
| InternVL | InternVL-Chat-V1.5, InternVL2 | SGLang, vLLM |
| Phi-3 Vision | Phi-3-vision-128k-instruct | SGLang, vLLM |
| Pixtral | pixtral-12b | SGLang, vLLM |

### 19.5 Vision RL Training Example

```python
# Example: Vision-based reward model training
class VisionRLEnv:
    async def step(self, image_path: str, question: str) -> dict:
        # Encode image for the model
        with open(image_path, "rb") as f:
            image_bytes = base64.b64encode(f.read()).decode()

        # Generate response about the image
        response = await self.engine.chat_completion({
            "json": {
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {
                            "url": f"data:image/jpeg;base64,{image_bytes}"
                        }},
                        {"type": "text", "text": question}
                    ]
                }],
                "max_tokens": 256
            }
        })

        answer = response["choices"][0]["message"]["content"]
        reward = self.compute_reward(answer, image_path)
        return {"response": answer, "reward": reward}
```

### 19.6 Backend Comparison

| Feature | SGLang | vLLM |
|---------|--------|------|
| Image in generate() | ✅ image_data param | ❌ Use chat_completion |
| Image in chat_completion | ✅ OpenAI format | ✅ OpenAI format |
| Base64 images | ✅ | ✅ |
| URL images | ✅ | ✅ |
| File path images | ✅ | ✅ |
| Multiple images | ✅ | ✅ |
| Video support | Model-dependent | Model-dependent |

---

## 20. Skip Tokenizer Init Limitations

SkyRL uses `skip_tokenizer_init=True` with SGLang for token-in-token-out mode. This provides memory efficiency and eliminates tokenization overhead, but has some limitations.

### 20.1 Why skip_tokenizer_init=True?

Token-in-token-out mode is optimal for RL training because:
- **Single tokenizer instance**: Training process owns the tokenizer, avoiding duplicate loading
- **Token-level precision**: Direct access to token IDs for policy gradients
- **Zero tokenization overhead**: No encode/decode in inference engine
- **Memory efficiency**: No redundant tokenizer weights in GPU memory

### 20.2 Parameter Limitations

#### Not Supported: `stop_regex`

```python
# ❌ This will raise ValueError
sampling_params = {
    "stop_regex": r"\\n\\n",  # Not supported with skip_tokenizer_init=True
}
```

**Reason**: SGLang's regex matching requires `tokenizer.decode()` internally to match regex patterns against generated text. With `skip_tokenizer_init=True`, the tokenizer is `None`, causing a crash.

**Workaround**: Use `stop_token_ids` instead:
```python
# ✅ Use stop_token_ids directly
sampling_params = {
    "stop_token_ids": [tokenizer.encode("\n\n", add_special_tokens=False)[-1]],
}
```

**Reference**: [SGLang Issue #9039](https://github.com/sgl-project/sglang/issues/9039)

#### Partial Support: `stop` (string stop sequences)

```python
# ⚠️ Works but with limitations
sampling_params = {
    "stop": ["</answer>", "\n\n"],  # Converted to stop_token_ids
}
```

**How it works**: SkyRL converts stop strings to `stop_token_ids` using the external tokenizer:
- Single-token stops: Direct mapping (works perfectly)
- Multi-token stops: Uses last token only (may cause premature stopping)

**Limitation**: For multi-token stop sequences like `"</answer>"` (which may tokenize to multiple tokens), only the last token is used as the stop token. This can cause false positives if that token appears in other contexts.

**Best practice**: Use `stop_token_ids` directly for precise control:
```python
# ✅ Precise multi-token stop (check each token individually)
eos_ids = tokenizer.encode("</answer>", add_special_tokens=False)
sampling_params = {
    "stop_token_ids": eos_ids,  # Stop on any of these tokens
}
```

#### Fully Supported: `min_new_tokens`

```python
# ✅ Fully supported with automatic eos_token_id injection
sampling_params = {
    "min_new_tokens": 10,  # Minimum tokens before allowing stop
}
```

SkyRL automatically passes `eos_token_id` to SGLang when `min_new_tokens > 0`, enabling the penalizer to suppress premature stopping.

### 20.3 Feature Compatibility Matrix

| Feature | Support | Notes |
|---------|---------|-------|
| `stop_token_ids` | ✅ Full | Direct token-based stopping |
| `stop` (strings) | ⚠️ Partial | Converted to last-token stops |
| `stop_regex` | ❌ None | Requires tokenizer.decode() |
| `min_new_tokens` | ✅ Full | Auto eos_token_id injection |
| `json_schema` | ✅ Full | Grammar-based decoding |
| `regex` (structured) | ✅ Full | Grammar-based decoding |
| `ebnf` | ✅ Full | Grammar-based decoding |
| `logit_bias` | ✅ Full | Token-level bias |
| `temperature`, `top_p`, `top_k` | ✅ Full | Standard sampling |

### 20.4 When to Use What

| Use Case | Recommended Approach |
|----------|---------------------|
| Stop on EOS token | Use default behavior (automatic) |
| Stop on specific token | `stop_token_ids: [token_id]` |
| Stop on specific string | `stop: ["string"]` (if single-token) or `stop_token_ids` (if multi-token) |
| Stop on regex pattern | Not supported - use grammar-based decoding (`regex` param) instead |
| Minimum generation length | `min_new_tokens: N` |

---

## 21. Speculative Decoding

Speculative decoding enables 2-3x inference speedup by using a draft model to predict multiple tokens ahead, then verifying them with the target model.

### 21.1 Supported Algorithms

| Algorithm | Description | Draft Model Required | Best For |
|-----------|-------------|---------------------|----------|
| `eagle` | Tree-based speculative decoding | Yes (EAGLE-trained) | Highest speedup, production |
| `eagle3` | EAGLE v3 with improved accuracy | Yes (EAGLE-trained) | Better acceptance rate |
| `standalone` | Separate small draft model | Yes (same family) | Simple setup |
| `ngram` | Pattern matching in history | No | Quick testing, LoRA compatible |

### 21.2 Configuration

Enable speculative decoding in your config:

```yaml
generator:
  backend: "sglang"  # Required
  speculative_decoding:
    enabled: true
    algorithm: "eagle"  # or "eagle3", "standalone", "ngram"

    # For EAGLE/standalone: provide draft model path
    draft_model_path: "yuhuili/EAGLE-LLaMA3-Instruct-8B"

    # Optional: tune parameters (auto-chosen if null)
    num_steps: 5        # Speculative steps per iteration
    eagle_topk: 4       # Candidates per step (1 for simpler models)
    num_draft_tokens: 8 # Total draft tokens

    # Acceptance thresholds (1.0 = strict, lower = more aggressive)
    accept_threshold_single: 1.0
    accept_threshold_acc: 1.0
```

### 21.3 Algorithm-Specific Examples

**EAGLE (Fastest, Recommended):**
```yaml
speculative_decoding:
  enabled: true
  algorithm: "eagle"
  draft_model_path: "yuhuili/EAGLE-LLaMA3-Instruct-8B"
  # For LLaMA models: auto-chooses (5, 4, 8)
```

**Standalone (Simpler Setup):**
```yaml
speculative_decoding:
  enabled: true
  algorithm: "standalone"
  draft_model_path: "meta-llama/Llama-3.2-1B"  # Small model from same family
  # Auto-chooses: (3, 1, 4)
```

**N-gram (No Draft Model, LoRA Compatible):**
```yaml
speculative_decoding:
  enabled: true
  algorithm: "ngram"
  ngram:
    max_match_window_size: 12
    max_bfs_breadth: 10
    match_type: "BFS"  # or "PROB"
```

**DeepSeek V3 (Built-in MTP):**
```yaml
speculative_decoding:
  enabled: true
  algorithm: "eagle"
  # No draft_model_path needed - uses built-in MTP weights
```

### 21.4 Performance Tuning

| Parameter | Effect | Recommendation |
|-----------|--------|----------------|
| `num_steps` | More steps = more tokens drafted | 3-5 for most models |
| `eagle_topk` | Higher = better acceptance, slower | 1 for simple, 4 for LLaMA |
| `num_draft_tokens` | Total tokens to draft | Auto-chosen based on topk |
| `accept_threshold_*` | Lower = accept more drafts | Keep at 1.0 unless debugging |

### 21.5 Requirements and Limitations

**Requirements:**
- SGLang backend only (not vLLM)
- `topk > 1` requires `flashinfer` or `fa3` attention backend
- CUDA device required for N-gram

**Limitations:**
- LoRA only compatible with N-gram algorithm
- EAGLE/standalone incompatible with DP attention
- Mixed chunked prefill disabled automatically

### 21.6 Expected Speedups

| Model Size | Algorithm | Typical Speedup |
|------------|-----------|-----------------|
| 7B-8B | EAGLE | 2-3x |
| 13B | EAGLE | 1.8-2.5x |
| 70B | EAGLE | 1.5-2x |
| Any | N-gram | 1.2-1.5x |

### 21.7 Debugging

Enable verbose logging to see acceptance rates:

```yaml
generator:
  engine_init_kwargs:
    log_level: "info"
```

Watch for metrics like:
- `spec_accept_rate`: Ratio of accepted draft tokens
- `spec_accept_length`: Average accepted tokens per step

---

## 22. FP8 KV Cache

FP8 KV cache reduces memory usage by ~50% with minimal accuracy impact, enabling larger batch sizes or longer sequences. **SGLang backend only.**

### 22.1 Basic Configuration

```yaml
generator:
  backend: "sglang"
  kv_cache:
    # FP8 format (recommended for most cases)
    dtype: "fp8_e4m3"
```

### 22.2 Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dtype` | string | `"auto"` | KV cache data type |
| `quantization_param_path` | string | `null` | Path to scaling factors JSON |
| `fp8_gemm_backend` | string | `"auto"` | FP8 GEMM computation backend |

**Supported `dtype` values:**

| Value | Description | CUDA Requirement |
|-------|-------------|------------------|
| `"auto"` | Uses model data type (default) | - |
| `"fp8_e4m3"` | FP8 with 4-bit exponent, 3-bit mantissa | CUDA 11.8+ |
| `"fp8_e5m2"` | FP8 with 5-bit exponent, 2-bit mantissa | CUDA 11.8+ |
| `"bf16"` / `"bfloat16"` | BFloat16 (no compression) | - |

**Supported `fp8_gemm_backend` values:**

| Value | Description | Best For |
|-------|-------------|----------|
| `"auto"` | Auto-select based on hardware | Default |
| `"deep_gemm"` | JIT-compiled (DeepGEMM) | Hopper (SM90), Blackwell (SM100) |
| `"flashinfer_trtllm"` | TensorRT-LLM integration | Blackwell, low-latency |
| `"cutlass"` | NVIDIA CUTLASS | Hopper/Blackwell, high-throughput |
| `"triton"` | OpenAI Triton | Fallback, widely compatible |
| `"aiter"` | AMD ROCm | ROCm only |

### 22.3 With Scaling Factors (Recommended)

For best accuracy, provide calibrated scaling factors:

```yaml
generator:
  backend: "sglang"
  kv_cache:
    dtype: "fp8_e4m3"
    quantization_param_path: "/path/to/kv_scales.json"
```

**Scaling factors JSON format:**
```json
{
  "kv_cache": {
    "dtype": "float8_e4m3fn",
    "scaling_factor": {
      "0": {"0": 1.0, "1": 1.0, "...": "..."},
      "1": {"0": 1.0, "1": 1.0, "...": "..."}
    }
  }
}
```

Where outer keys are TP rank indices and inner keys are layer indices.

### 22.4 Hardware Recommendations

| GPU Architecture | Recommended dtype | Notes |
|------------------|-------------------|-------|
| Blackwell (SM100) | `fp8_e4m3` | Native FP8 support, best performance |
| Hopper (SM90) | `fp8_e4m3` | Full FP8 support |
| Ampere (SM80) | `bf16` | Limited FP8 support |
| Ada Lovelace | `fp8_e4m3` | Good FP8 support |

### 22.5 Complete Example

```yaml
generator:
  backend: "sglang"

  # Enable FP8 KV cache
  kv_cache:
    dtype: "fp8_e4m3"
    quantization_param_path: "/data/models/llama3-8b/kv_scales.json"
    fp8_gemm_backend: "cutlass"  # For Hopper/Blackwell

  # Increase batch size due to memory savings
  max_num_seqs: 2048
  gpu_memory_utilization: 0.9
```

### 22.6 Memory Savings

| Model Size | bf16 KV Cache | fp8 KV Cache | Savings |
|------------|---------------|--------------|---------|
| 7B-8B | ~2GB | ~1GB | ~50% |
| 13B | ~4GB | ~2GB | ~50% |
| 70B | ~20GB | ~10GB | ~50% |

*Note: Actual savings depend on sequence length and batch size.*

### 22.7 Compatibility Notes

**Attention Backend Compatibility:**
- `fa3` (FlashAttention3): Only supports `fp8_e4m3` (auto-fallback for `fp8_e5m2`)
- `flashinfer`: Full FP8 support
- `triton`: Works with both FP8 formats

**LoRA Compatibility:**
- FP8 KV cache works with LoRA
- No restrictions on LoRA + FP8 combination

**Speculative Decoding:**
- Can combine FP8 KV cache with speculative decoding
- Both features reduce memory and improve throughput

### 22.8 Troubleshooting

**Accuracy degradation:**
```yaml
# Provide scaling factors for better accuracy
kv_cache:
  dtype: "fp8_e4m3"
  quantization_param_path: "/path/to/scales.json"  # Required for best accuracy
```

**CUDA version errors:**
- Ensure CUDA 11.8+ for FP8 support
- Check with: `python -c "import torch; print(torch.version.cuda)"`

**Hardware capability errors:**
- FP8 requires SM80+ (Ampere or newer)
- Check with: `python -c "import torch; print(torch.cuda.get_device_capability())"`

---

## 23. Session-Based Generation

Session-based generation enables efficient multi-turn RL by maintaining KV cache state across conversation turns. This avoids redundant prefix recomputation, significantly improving throughput for multi-turn environments. **SGLang backend only.**

### 23.1 Overview

In multi-turn RL environments (e.g., agentic tasks, tool use, conversations), each turn builds on the previous context. Without sessions, the full conversation history must be reprocessed for every turn. Sessions maintain the KV cache from previous turns, enabling:

- **~2-10x speedup** for multi-turn generation (depending on context length)
- **Reduced memory pressure** through prefix sharing
- **Branching support** for tree search / MCTS algorithms

### 23.2 Configuration

```yaml
generator:
  backend: "sglang"

  sessions:
    # Enable session-based generation
    enabled: true

    # KV cache capacity per session (in tokens)
    default_capacity: 8192

    # Reuse sessions across batches
    pool_sessions: false
    max_pool_size: 64
```

### 23.3 API Usage

**Basic multi-turn conversation:**

```python
# Open a session
session_id = await engine.open_session(capacity_of_str_len=8192)

try:
    # First turn
    output1 = await engine.generate_with_session(
        session_id=session_id,
        input_batch={"prompt_token_ids": [first_prompt_ids], "sampling_params": params},
    )
    rid1 = output1["request_ids"][0]  # Save for next turn

    # Second turn - appends to first turn's context
    output2 = await engine.generate_with_session(
        session_id=session_id,
        input_batch={"prompt_token_ids": [second_prompt_ids], "sampling_params": params},
        rid=rid1,  # Continue from previous turn
    )
    rid2 = output2["request_ids"][0]

    # Third turn
    output3 = await engine.generate_with_session(
        session_id=session_id,
        input_batch={"prompt_token_ids": [third_prompt_ids], "sampling_params": params},
        rid=rid2,
    )
finally:
    await engine.close_session(session_id)
```

### 23.4 Session Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `session_id` | str | required | Session ID from `open_session()` |
| `rid` | str | None | Request ID to continue from (use output's `request_ids[i]`) |
| `offset` | int | -1 | Token position to continue from (-1 = append at end) |
| `replace` | bool | False | If True, clears child branches when branching |
| `drop_previous_output` | bool | False | If True, drops previous generated output, keeps only prompt |

### 23.5 Branching for Tree Search

Sessions support tree-based branching for algorithms like MCTS:

```python
session_id = await engine.open_session(capacity_of_str_len=16384)

# Root generation
root_output = await engine.generate_with_session(
    session_id=session_id,
    input_batch={"prompt_token_ids": [root_prompt], "sampling_params": params},
)
root_rid = root_output["request_ids"][0]

# Branch A - continues from root
branch_a = await engine.generate_with_session(
    session_id=session_id,
    input_batch={"prompt_token_ids": [branch_a_prompt], "sampling_params": params},
    rid=root_rid,
    offset=0,  # Branch at token 0 of root's output
    replace=False,  # Keep other branches
)

# Branch B - also continues from root (parallel branch)
branch_b = await engine.generate_with_session(
    session_id=session_id,
    input_batch={"prompt_token_ids": [branch_b_prompt], "sampling_params": params},
    rid=root_rid,
    offset=0,
)

await engine.close_session(session_id)
```

### 23.6 Configuration Examples

**Agentic tool-use environment:**
```yaml
generator:
  backend: "sglang"
  sessions:
    enabled: true
    default_capacity: 16384  # Larger for tool call/response pairs
    pool_sessions: true
    max_pool_size: 128  # One per concurrent agent
```

**MCTS/tree search:**
```yaml
generator:
  backend: "sglang"
  sessions:
    enabled: true
    default_capacity: 32768  # Large for deep trees
    pool_sessions: false  # Each search gets fresh session
```

### 23.7 Memory Considerations

- Each session reserves `capacity_of_str_len` tokens of KV cache
- With `pool_sessions=true`, memory = `max_pool_size * capacity * kv_memory_per_token`
- Use FP8 KV cache (`kv_cache.dtype: fp8_e4m3`) to halve session memory

**Memory calculation example:**
```
# LLaMA 70B, 8K capacity per session, 64 sessions
# BF16: ~64 * 8192 * 8KB = 4GB for sessions
# FP8:  ~64 * 8192 * 4KB = 2GB for sessions
```

### 23.8 Compatibility

| Feature | Session Compatible |
|---------|-------------------|
| FP8 KV Cache | Yes |
| Speculative Decoding | Yes |
| LoRA | Yes |
| Tensor Parallelism | Yes |
| Prefix Caching | Yes (enhanced) |
| Streaming | No (use standard generate) |

### 23.9 Troubleshooting

**"Session not found" errors:**
- Ensure session was opened before generation
- Sessions are engine-local; don't share IDs across engines

**Memory exhaustion:**
- Reduce `default_capacity` or `max_pool_size`
- Enable FP8 KV cache for 50% memory reduction
- Close unused sessions promptly

**Performance not improved:**
- Ensure `rid` is passed correctly between turns
- First turn is always full compute; savings appear on turn 2+
- Very short contexts may not benefit significantly

---

## 24. Model Quantization

SkyRL exposes SGLang's comprehensive quantization support with 23+ methods for 2-4x memory reduction and faster inference. **SGLang backend only.**

### 24.1 Quick Start

```yaml
generator:
  backend: "sglang"
  quantization:
    method: "fp8"  # Or awq, gptq, etc.
```

### 24.2 Supported Quantization Methods

| Category | Method | Bits | GPU Requirement | Notes |
|----------|--------|------|-----------------|-------|
| **Weight-only** | `awq` | 4 | SM75+ (Turing) | Activation-aware |
| | `awq_marlin` | 4/8 | SM75+ | Faster Marlin kernels |
| | `gptq` | 2/3/4/8 | SM60+ | General-purpose |
| | `gptq_marlin` | 4/8 | SM60+ | GPTQ + Marlin |
| | `gguf` | Various | SM60+ | Requires `load_format: gguf` |
| | `bitsandbytes` | 4 | Any | NF4/FP4 formats |
| | `auto-round` | 2/3/4/8 | SM60+ | Intel Auto-Round |
| **FP8** | `fp8` | 8 | SM80+ (Ampere) | Dynamic FP8 |
| | `w8a8_fp8` | 8 | SM89+ (Hopper) | Weight & Activation FP8 |
| | `modelopt_fp8` | 8 | SM89+ | NVIDIA ModelOpt |
| **FP4/INT4** | `modelopt_fp4` | 4 | SM89+ | NVIDIA NVFP4 |
| | `w4afp8` | 4/8 | SM90+ | Weight 4-bit, Act FP8 |
| | `mxfp4` | 4 | SM90+, CUDA 12.8+ | Microscaling FP4 |
| | `petit_nvfp4` | 4 | ROCm gfx90a/942 | AMD FP4 |
| **INT8** | `w8a8_int8` | 8 | SM80+ or CPU AMX | Weight & Act INT8 |
| **MoE** | `moe_wna16` | Various | Varies | MoE-specific |
| **Other** | `qoq` | 4 | SM80+ | Quantization-of-Quantization |
| | `compressed-tensors` | Various | Varies | External library |
| | `modelopt` | Auto | Varies | Auto-detect ModelOpt |

### 24.3 Configuration Options

```yaml
generator:
  backend: "sglang"
  quantization:
    # Main quantization method (see table above)
    method: "fp8"

    # Weight loading format
    # - "auto": Auto-detect (default)
    # - "gguf": Required for gguf quantization
    # - "bitsandbytes": BitsAndBytes format
    # - "flash_rl": Profile-free FP8 for RL (recommended for online training)
    load_format: "auto"

    # Keep language model head in FP32 for accuracy
    enable_fp32_lm_head: false

    # ModelOpt-specific settings
    modelopt:
      quant_type: null  # fp8, int4_awq, w4a8_awq, nvfp4, nvfp4_awq
      checkpoint_restore_path: null
      checkpoint_save_path: null
      export_path: null
      quantize_and_serve: false

    # Draft model quantization (for speculative decoding)
    draft_model_quantization: null  # "unquant" or any method
```

### 24.4 Common Configurations

**FP8 for Hopper/Blackwell GPUs:**
```yaml
generator:
  backend: "sglang"
  quantization:
    method: "fp8"
    enable_fp32_lm_head: true  # Better accuracy
  kv_cache:
    dtype: "fp8_e4m3"  # Combine with FP8 KV cache
```

**AWQ for consumer GPUs:**
```yaml
generator:
  backend: "sglang"
  quantization:
    method: "awq_marlin"  # Fast Marlin kernels
```

**GGUF models:**
```yaml
generator:
  backend: "sglang"
  quantization:
    method: "gguf"
    load_format: "gguf"  # Required for GGUF
```

**Profile-free FP8 for RL:**
```yaml
generator:
  backend: "sglang"
  quantization:
    method: "fp8"
    load_format: "flash_rl"  # No calibration needed
```

**ModelOpt on-the-fly quantization:**
```yaml
generator:
  backend: "sglang"
  quantization:
    method: "modelopt_fp8"
    modelopt:
      quant_type: "fp8"
      quantize_and_serve: true  # Slower startup, no pre-quantization
```

**Speculative decoding with quantized draft:**
```yaml
generator:
  backend: "sglang"
  speculative_decoding:
    enabled: true
    algorithm: "eagle"
    draft_model_path: "yuhuili/EAGLE-LLaMA3-8B"
  quantization:
    method: "fp8"
    draft_model_quantization: "fp8"  # Also quantize draft model
```

### 24.5 Memory Savings by Method

| Method | Memory Reduction | Speed Impact | Accuracy Impact |
|--------|------------------|--------------|-----------------|
| FP8 | ~50% | +10-20% | Minimal |
| AWQ/GPTQ 4-bit | ~75% | +0-10% | Low-Moderate |
| ModelOpt FP4 | ~75% | +20-30% | Moderate |
| INT8 | ~50% | +5-15% | Minimal |
| BitsAndBytes | ~75% | -10-20% | Low-Moderate |

### 24.6 Hardware Compatibility Matrix

| GPU Generation | FP8 | FP4/NVFP4 | AWQ/GPTQ | INT8 | GGUF |
|----------------|-----|-----------|----------|------|------|
| Blackwell (SM100) | ✓ | ✓ | ✓ | ✓ | ✓ |
| Hopper (SM90) | ✓ | ✓ | ✓ | ✓ | ✓ |
| Ada Lovelace (SM89) | ✓ | ✓ | ✓ | ✓ | ✓ |
| Ampere (SM80) | ✓ | - | ✓ | ✓ | ✓ |
| Turing (SM75) | - | - | ✓ | - | ✓ |
| Volta (SM70) | - | - | ✓ | - | ✓ |
| Pascal (SM60) | - | - | ✓ | - | ✓ |

### 24.7 Combining with Other Features

Quantization works with most SkyRL features:

| Feature | Compatible | Notes |
|---------|------------|-------|
| FP8 KV Cache | ✓ | Recommended combination |
| Speculative Decoding | ✓ | Can quantize draft separately |
| Sessions | ✓ | Full support |
| LoRA | ✓ | Some methods only |
| Tensor Parallelism | ✓ | Full support |
| Weight Sync | ✓ | Full support |

### 24.8 Troubleshooting

**"CUDA capability too low" errors:**
- Check GPU compatibility in the table above
- Use a compatible method for your GPU

**Accuracy degradation:**
```yaml
quantization:
  method: "fp8"
  enable_fp32_lm_head: true  # Helps with accuracy
```

**GGUF loading failures:**
```yaml
quantization:
  method: "gguf"
  load_format: "gguf"  # Must be set explicitly
```

**ModelOpt library not found:**
- Install: `pip install nvidia-modelopt`
- Only needed for `modelopt*` methods

**Slow startup with quantize_and_serve:**
- Pre-quantize model and save checkpoint
- Use `checkpoint_restore_path` for faster loads

---

## 25. Custom Logit Processors

Custom logit processors enable advanced sampling control by modifying logits before temperature scaling and sampling. **SGLang backend only.**

### 25.1 Enabling Custom Logit Processors

```yaml
generator:
  backend: "sglang"
  custom_logit_processor:
    enabled: true  # Required for security (disabled by default)
```

### 25.2 Built-in Processors

SGLang provides several built-in logit processors:

| Processor | Description | Parameters |
|-----------|-------------|------------|
| `DisallowedTokensLogitsProcessor` | Mask specific token IDs | `token_ids`: list of IDs to block |
| `ThinkingBudgetLogitProcessor` | Control reasoning length | `thinking_budget`: max thinking tokens |
| `DeepseekOCRNoRepeatNGramLogitProcessor` | Prevent n-gram repetition | `ngram_size`, `window_size` |

**Model-specific thinking processors:**
- `Glm4MoeThinkingBudgetLogitProcessor` (GLM-4)
- `Qwen3ThinkingBudgetLogitProcessor` (Qwen-3)
- `DeepSeekR1ThinkingBudgetLogitProcessor` (DeepSeek-R1)

### 25.3 Using Built-in Processors

**Disallowing specific tokens:**
```python
from sglang.srt.sampling.custom_logit_processor import DisallowedTokensLogitsProcessor

# Create and serialize processor
processor = DisallowedTokensLogitsProcessor(token_ids=[1234, 5678])
processor_str = processor.to_str()

# Pass in sampling params
sampling_params = {
    "temperature": 1.0,
    "custom_logit_processor": processor_str,
    "custom_params": {"token_ids": [1234, 5678]}
}
```

**Controlling thinking budget (for reasoning models):**
```python
from sglang.srt.sampling.custom_logit_processor import DeepSeekR1ThinkingBudgetLogitProcessor

processor = DeepSeekR1ThinkingBudgetLogitProcessor()
processor_str = processor.to_str()

sampling_params = {
    "temperature": 1.0,
    "custom_logit_processor": processor_str,
    "custom_params": {"thinking_budget": 500}  # Max 500 thinking tokens
}
```

### 25.4 Creating Custom Processors

```python
from sglang.srt.sampling.custom_logit_processor import CustomLogitProcessor
import torch

class MyLogitProcessor(CustomLogitProcessor):
    """Custom processor that boosts certain tokens."""

    def __call__(self, logits: torch.Tensor, custom_param_list) -> torch.Tensor:
        for i, params in enumerate(custom_param_list):
            if params and "boost_tokens" in params:
                for token_id in params["boost_tokens"]:
                    logits[i, token_id] += params.get("boost_amount", 5.0)
        return logits

# Serialize for transmission
processor_str = MyLogitProcessor.to_str()
```

### 25.5 Configuration in SkyRL

```yaml
generator:
  backend: "sglang"

  # Enable custom processor support
  custom_logit_processor:
    enabled: true

  sampling_params:
    temperature: 1.0
    top_p: 0.95

    # Serialized processor string (from processor.to_str())
    custom_logit_processor: null  # Set programmatically

    # Parameters passed to processor's __call__
    custom_params:
      token_ids: [1234, 5678]
```

### 25.6 Programmatic Usage

```python
from sglang.srt.sampling.custom_logit_processor import DisallowedTokensLogitsProcessor

# Create processor
processor = DisallowedTokensLogitsProcessor(token_ids=[])
processor_str = processor.to_str()

# Modify sampling params before generation
input_batch = {
    "prompt_token_ids": [[1, 2, 3]],
    "sampling_params": {
        "temperature": 1.0,
        "custom_logit_processor": processor_str,
        "custom_params": {
            "token_ids": [50256]  # Block EOS token
        }
    }
}

output = await engine.generate(input_batch)
```

### 25.7 Application Order

Custom logit processors are applied **before** built-in sampling transformations:

1. **Custom logit processor** ← Your modifications
2. Temperature scaling
3. Top-k filtering
4. Top-p (nucleus) sampling
5. Min-p filtering
6. Final sampling

### 25.8 Use Cases for RL

**Preventing reward hacking:**
```python
# Block tokens that exploit reward function
bad_tokens = [token_id for word in ["cheat", "hack"]
              for token_id in tokenizer.encode(word)]
processor = DisallowedTokensLogitsProcessor(token_ids=bad_tokens)
```

**Enforcing output format:**
```python
class FormatEnforcingProcessor(CustomLogitProcessor):
    def __call__(self, logits, custom_param_list):
        # Force specific tokens at certain positions
        for i, params in enumerate(custom_param_list):
            if params.get("force_json_start"):
                logits[i, :] = -float("inf")
                logits[i, self.open_brace_id] = 0
        return logits
```

**Diversity promotion:**
```python
class DiversityProcessor(CustomLogitProcessor):
    def __call__(self, logits, custom_param_list):
        for i, params in enumerate(custom_param_list):
            # Penalize recently used tokens
            for token_id in params.get("recent_tokens", []):
                logits[i, token_id] -= 2.0
        return logits
```

### 25.9 Security Considerations

Custom logit processors are **disabled by default** because:
- Processors execute arbitrary Python code
- Serialized processors use `dill` (can execute code on deserialization)
- Only enable in trusted environments

```yaml
# Only enable if you trust the processor source
custom_logit_processor:
  enabled: true  # Be cautious!
```

### 25.10 Limitations

- Cannot use list of processors with `parallel_sample_num > 1`
- Processors must be serializable with `dill`
- State is not preserved between calls (stateless design)
- Applied per-batch, not per-token

---

## 26. RoPE Scaling (Context Length Extension)

RoPE (Rotary Position Embedding) scaling enables extending model context length beyond the original training length. **Works with both SGLang and vLLM backends.**

### 26.1 Configuration

```yaml
trainer:
  # Base frequency for rotary embeddings
  rope_theta: null  # Uses model default (10000 for LLaMA, 1000000 for Qwen2)

  # RoPE scaling configuration
  rope_scaling:
    rope_type: linear
    factor: 2.0

generator:
  # Inherits from trainer by default
  rope_scaling: ${trainer.rope_scaling}
  rope_theta: ${trainer.rope_theta}
```

### 26.2 Supported Scaling Types

| Type | Description | Required Params | Use Case |
|------|-------------|-----------------|----------|
| `linear` | Simple position interpolation | `factor` | Quick 2-4x extension |
| `dynamic` | Dynamic NTK scaling | `factor` | Adaptive extension |
| `yarn` | YaRN interpolation + extrapolation | `factor`, `original_max_position_embeddings` | Best quality |
| `deepseek_yarn` | DeepSeek's YaRN variant | `factor`, `original_max_position_embeddings` | DeepSeek models |
| `llama3` | LLaMA 3 frequency-selective | `factor`, `low_freq_factor`, `high_freq_factor`, `original_max_position_embeddings` | LLaMA 3 models |
| `longrope` | Phi3 dual-factor scaling | `short_factor`, `long_factor`, `original_max_position_embeddings` | Phi3 models |
| `default` | Standard/mRoPE | (none) | No scaling |

### 26.3 Common Configurations

**Linear scaling (2x context):**
```yaml
trainer:
  rope_scaling:
    rope_type: linear
    factor: 2.0
```

**YaRN scaling (recommended for quality):**
```yaml
trainer:
  rope_scaling:
    rope_type: yarn
    factor: 4.0
    original_max_position_embeddings: 8192
    attn_factor: 1.0
    beta_fast: 32
    beta_slow: 1
```

**Dynamic NTK scaling:**
```yaml
trainer:
  rope_scaling:
    rope_type: dynamic
    factor: 2.0
```

**LLaMA 3.1 (128K context):**
```yaml
trainer:
  rope_theta: 500000
  rope_scaling:
    rope_type: llama3
    factor: 8.0
    low_freq_factor: 1.0
    high_freq_factor: 4.0
    original_max_position_embeddings: 8192
```

**DeepSeek with YaRN:**
```yaml
trainer:
  rope_scaling:
    rope_type: deepseek_yarn
    factor: 4.0
    original_max_position_embeddings: 4096
    mscale: 1.0
    mscale_all_dim: false
```

### 26.4 Parameter Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `rope_type` | str | - | Scaling algorithm (see table above) |
| `factor` | float | 1.0 | Scaling factor (2.0 = 2x context) |
| `original_max_position_embeddings` | int | - | Original model context length |
| `attn_factor` | float | 1.0 | Attention scaling (YaRN) |
| `beta_fast` | int | 32 | Fast decay frequency (YaRN) |
| `beta_slow` | int | 1 | Slow decay frequency (YaRN) |
| `low_freq_factor` | float | - | Low frequency scaling (LLaMA 3) |
| `high_freq_factor` | float | - | High frequency scaling (LLaMA 3) |
| `mscale` | float | - | Magnitude scaling (DeepSeek) |
| `short_factor` | List[float] | - | Short wavelength factors (LongRoPE) |
| `long_factor` | List[float] | - | Long wavelength factors (LongRoPE) |

### 26.5 Common rope_theta Values

| Model | Default rope_theta | Max Position |
|-------|-------------------|--------------|
| LLaMA 2 | 10000 | 4096 |
| LLaMA 3 | 500000 | 8192 |
| LLaMA 3.1 | 500000 | 128K |
| Qwen 2 | 1000000 | 32768 |
| Mistral | 10000 | 32768 |
| DeepSeek | 10000 | Varies |

### 26.6 Quality vs Extension Tradeoff

| Scaling Type | Quality | Max Extension | Notes |
|--------------|---------|---------------|-------|
| Linear | Good | 2-4x | Fast, simple |
| Dynamic | Better | 4-8x | Adapts to length |
| YaRN | Best | 8-16x | Recommended |
| LLaMA 3 | Excellent | 16x | Model-specific |

### 26.7 Troubleshooting

**Quality degradation at long contexts:**
- Use YaRN instead of linear scaling
- Reduce scaling factor
- Use model fine-tuned for extended context

**OOM with extended context:**
- Combine with FP8 KV cache for memory savings
- Reduce batch size
- Use tensor parallelism

**Incorrect outputs:**
- Verify `original_max_position_embeddings` matches model
- Check `rope_theta` matches model default

---

## 27. CUDA Graph Capture (SGLang Only)

CUDA graphs capture GPU operations and replay them with minimal CPU overhead, significantly reducing per-token latency during decode operations. SGLang provides comprehensive CUDA graph support with fine-grained control.

### 27.1 Overview

CUDA graphs work by:
1. **Capturing** a sequence of GPU operations during warmup
2. **Replaying** the captured operations with minimal CPU overhead
3. **Reducing** kernel launch latency by 50-90% for small batches

Key benefits:
- Reduced decode latency (especially important for interactive applications)
- Lower CPU overhead during inference
- Better GPU utilization for small batch sizes

### 27.2 Configuration

**Basic CUDA Graph Configuration:**
```yaml
generator:
  backend: sglang
  # Note: enforce_eager: true disables CUDA graphs entirely
  enforce_eager: false

  cuda_graph:
    # Disable CUDA graphs (for debugging or compatibility)
    disable: false

    # Maximum batch size for CUDA graph capture
    # Auto-configured based on GPU if null
    max_bs: 256

    # Explicit batch sizes to capture (overrides auto-padding)
    batch_sizes: null  # e.g., [1, 2, 4, 8, 16, 32, 64]

    # Disable padding optimization (capture ALL batch sizes)
    disable_padding: false

    # Enable profiling during capture
    enable_profiling: false

    # Enable garbage collection during capture
    enable_gc: false
```

**Piecewise CUDA Graphs (Prefill Optimization):**
```yaml
generator:
  piecewise_cuda_graph:
    # Enable piecewise CUDA graphs for prefill phase
    enabled: true

    # Maximum token count for capture
    max_tokens: 4096

    # Explicit token counts to capture
    token_counts: null

    # Compiler: "eager" or "inductor"
    compiler: "eager"
```

**torch.compile Integration:**
```yaml
generator:
  torch_compile:
    # Enable torch.compile optimization
    enabled: true

    # Debug mode for diagnosing issues
    debug_mode: false

    # Maximum batch size for torch.compile
    max_bs: 32
```

### 27.3 Auto-Configuration by GPU

When `cuda_graph.max_bs` is null, SGLang auto-configures based on GPU:

| GPU | Memory | Default max_bs (TP<4) | Default max_bs (TP≥4) |
|-----|--------|----------------------|----------------------|
| T4 / RTX 4080 | <20GB | 8 | 8 |
| A10 / RTX 4090 / RTX 5090 | 20-35GB | 24 | 80 |
| A100 (40GB) / L40 | 35-60GB | 32 | 160 |
| H100 / A100 (80GB) | 60-90GB | 256 | 512 |
| H200 / B200 / MI300 | >90GB | 256-512 | 512 |

### 27.4 Batch Size Padding Strategy

By default, SGLang uses optimized padding to reduce memory usage:

**Default captured batch sizes:**
```
[1, 2, 4, 8, 12] + [16, 24, 32, ...256 step 8] + [272, 288, ...512 step 16] + [512, 544, ...max_bs step 32]
```

**With speculative decoding (more granular):**
```
[1-8] + [10, 12, ...32 step 2] + [40, 44, ...64 step 4] + [72, 80, ...256 step 8] + [272, ...max_bs step 16]
```

**Custom batch sizes:**
```yaml
generator:
  cuda_graph:
    # Capture only specific batch sizes
    batch_sizes: [1, 2, 4, 8, 16, 32, 64, 128]
```

### 27.5 Memory Considerations

CUDA graphs consume additional memory. Reserved memory formula:
```
reserved = chunked_prefill_size * 1.5 + cuda_graph_max_bs * 2
```

**Additional memory for:**
- DP Attention: `cuda_graph_max_bs * dp_size * 3`
- Piecewise CUDA Graphs: `piecewise_max_tokens // 8`
- Speculative Decoding (EAGLE): `2 GB`
- Speculative Decoding (STANDALONE): `6 GB`

**Memory-constrained configuration:**
```yaml
generator:
  cuda_graph:
    max_bs: 32  # Lower max batch size
  gpu_memory_utilization: 0.85
  max_num_batched_tokens: 4096
```

### 27.6 Use Cases

**High Throughput (Large Batches):**
```yaml
generator:
  cuda_graph:
    max_bs: 512
  max_num_batched_tokens: 8192
  gpu_memory_utilization: 0.85
```

**Low Latency (Small Batches):**
```yaml
generator:
  cuda_graph:
    max_bs: 32
  max_num_batched_tokens: 4096
  gpu_memory_utilization: 0.90
```

**With Piecewise CUDA Graphs (Full Optimization):**
```yaml
generator:
  cuda_graph:
    max_bs: 128
  piecewise_cuda_graph:
    enabled: true
    max_tokens: 4096
    compiler: "eager"
```

**With torch.compile (Maximum Performance):**
```yaml
generator:
  cuda_graph:
    max_bs: 64
  piecewise_cuda_graph:
    enabled: true
    compiler: "inductor"
  torch_compile:
    enabled: true
    max_bs: 32
```

### 27.7 When CUDA Graphs are Automatically Disabled

SGLang automatically disables CUDA graphs in these scenarios:

1. **Incompatible attention backends:**
   - `torch_native` backend
   - `flex_attention` backend

2. **Certain model configurations:**
   - DeepEP MoE with `deepep_mode="normal"`
   - Diffusion LLM on AMD/HIP

3. **Disaggregation mode:**
   - Prefill server without `enable_piecewise_cuda_graph`

4. **Debugging modes:**
   - Tensor dump output enabled

### 27.8 Troubleshooting

**Out of Memory during graph capture:**
```yaml
generator:
  cuda_graph:
    max_bs: 64  # Reduce from default
  gpu_memory_utilization: 0.80
```

**Slow decode with eager mode:**
```yaml
generator:
  enforce_eager: false  # Enable CUDA graphs
  cuda_graph:
    max_bs: 128
```

**Variable prefill performance:**
```yaml
generator:
  piecewise_cuda_graph:
    enabled: true
    max_tokens: 4096
```

**Multi-node TP deadlock:**
```yaml
generator:
  cuda_graph:
    disable: true  # Disable for multi-node
```

**Batch size not captured (fallback to eager):**
```yaml
generator:
  cuda_graph:
    # Either increase max_bs or capture specific sizes
    max_bs: 512
    # Or use disable_padding to capture all sizes (more memory)
    disable_padding: true
```

### 27.9 Parameter Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `cuda_graph.disable` | bool | false | Disable CUDA graphs entirely |
| `cuda_graph.max_bs` | int | auto | Maximum batch size for capture |
| `cuda_graph.batch_sizes` | List[int] | auto | Explicit batch sizes to capture |
| `cuda_graph.disable_padding` | bool | false | Capture ALL batch sizes (more memory) |
| `cuda_graph.enable_profiling` | bool | false | Profile during capture |
| `cuda_graph.enable_gc` | bool | false | Enable GC during capture |
| `piecewise_cuda_graph.enabled` | bool | false | Enable piecewise CUDA graphs |
| `piecewise_cuda_graph.max_tokens` | int | 4096 | Max tokens for piecewise capture |
| `piecewise_cuda_graph.token_counts` | List[int] | auto | Explicit token counts |
| `piecewise_cuda_graph.compiler` | str | "eager" | Compiler: "eager" or "inductor" |
| `torch_compile.enabled` | bool | false | Enable torch.compile |
| `torch_compile.debug_mode` | bool | false | Debug mode for torch.compile |
| `torch_compile.max_bs` | int | 32 | Max batch size for torch.compile |

---

## 28. Attention Backends (SGLang Only)

SGLang supports 17+ attention backends optimized for different hardware platforms and use cases. By default, SGLang auto-selects the optimal backend based on GPU architecture.

### 28.1 Overview

Attention backends control how self-attention and cross-attention computations are executed. Different backends offer trade-offs between:
- **Performance**: Throughput and latency
- **Memory efficiency**: KV cache and activation memory usage
- **Hardware compatibility**: GPU architecture requirements
- **Feature support**: MLA, cross-attention, sparse attention

### 28.2 Supported Backends

#### Common Backends (All Platforms)

| Backend | Description | Hardware | Performance |
|---------|-------------|----------|-------------|
| `flashinfer` | FlashInfer kernels | NVIDIA SM80+ | Fast, versatile (default) |
| `fa3` | FlashAttention v3 | NVIDIA SM90+ | Fastest on H100/B200 |
| `fa4` | FlashAttention v4 | NVIDIA SM90+ | Newer FA variant |
| `triton` | Triton kernels | Universal | Customizable, double sparsity |
| `torch_native` | PyTorch SDPA | Universal | Slowest, most compatible |
| `flex_attention` | PyTorch 2.5+ block masks | SM90+ | Good, requires PyTorch 2.5+ |

#### MLA-Specific Backends (DeepSeek-V2.5, QwQ, etc.)

| Backend | Page Size | Hardware | Notes |
|---------|-----------|----------|-------|
| `flashmla` | 64 (required) | SM90+ | Optimized for MLA |
| `cutlass_mla` | 128 (required) | SM90+ | CUTLASS MLA kernels |
| `trtllm_mla` | 32/64 | SM100 only | TensorRT-LLM MLA |

#### Sparse Attention

| Backend | Description | Best For |
|---------|-------------|----------|
| `nsa` | Native Sparse Attention | DeepSeek-V3, sparse models |
| `double_sparsity` | Heavy token + sparse (triton) | Token pruning patterns |

#### Platform-Specific

| Backend | Platform | Notes |
|---------|----------|-------|
| `aiter` | AMD RDNA 3+ | Primary AMD backend |
| `wave` | AMD WAVE | AMD architecture specific |
| `intel_amx` | Intel Xeon 4th gen+ | CPU with AMX |
| `intel_xpu` | Intel Flex 170 | Data Center GPU |
| `ascend` | Huawei Ascend | NPU backend |
| `trtllm_mha` | NVIDIA SM100 | Blackwell MHA |
| `dual_chunk_flash_attn` | SM100 | Specialized |

### 28.3 Configuration

**Basic Configuration:**
```yaml
generator:
  backend: sglang
  attention:
    # Main attention backend (auto-selected if null)
    backend: "flashinfer"

    # Override for prefill phase only
    prefill_backend: null

    # Override for decode phase only
    decode_backend: null

    # Multimodal attention (for VLMs)
    mm_backend: null

    # Enable double sparsity (requires backend="triton")
    enable_double_sparsity: false
```

**Separate Prefill/Decode Backends:**
```yaml
generator:
  attention:
    backend: "flashinfer"
    # Use different backend for prefill (longer sequences)
    prefill_backend: "fa3"
    # Use different backend for decode (smaller batches)
    decode_backend: "flashinfer"
```

**MLA Model Configuration:**
```yaml
generator:
  attention:
    backend: "flashmla"
  # Note: flashmla requires page_size=64
  engine_init_kwargs:
    page_size: 64
```

**Sparse Attention (DeepSeek-V3):**
```yaml
generator:
  attention:
    backend: "nsa"
    nsa:
      prefill_backend: "flashmla_sparse"
      decode_backend: "fa3"
```

**Double Sparsity:**
```yaml
generator:
  attention:
    backend: "triton"
    enable_double_sparsity: true
```

### 28.4 Auto-Selection Defaults

When `attention.backend` is null, SGLang auto-selects based on hardware:

| Hardware | Auto-Selected Backend |
|----------|----------------------|
| NVIDIA SM100 (B200/GB200) | `trtllm_mha` |
| NVIDIA SM90 (H100) | `fa3` |
| NVIDIA SM80 (A100) | `flashinfer` |
| NVIDIA SM70-75 | `flashinfer` |
| AMD RDNA 3+ | `aiter` |
| Intel XPU | `intel_xpu` |
| Huawei Ascend | `ascend` |
| CPU/Fallback | `torch_native` |

### 28.5 Hardware Compatibility Matrix

| Backend | SM80 | SM90 | SM100 | AMD | Intel | Ascend |
|---------|------|------|-------|-----|-------|--------|
| `flashinfer` | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ |
| `fa3` | ✗ | ✓ | ✓ | ✗ | ✗ | ✗ |
| `fa4` | ~ | ✓ | ✓ | ✗ | ✗ | ✗ |
| `triton` | ✓ | ✓ | ✓ | ✓ | ~ | ~ |
| `torch_native` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `flex_attention` | ✗ | ✓ | ✓ | ✗ | ✗ | ✗ |
| `flashmla` | ✗ | ✓ | ✓ | ✗ | ✗ | ✗ |
| `cutlass_mla` | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ |
| `trtllm_mha` | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ |
| `trtllm_mla` | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ |
| `aiter` | ~ | ~ | ✗ | ✓ | ✗ | ✗ |
| `wave` | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ |
| `intel_amx` | ✗ | ✗ | ✗ | ✗ | ✓ | ✗ |
| `intel_xpu` | ✗ | ✗ | ✗ | ✗ | ✓ | ✗ |
| `ascend` | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ |

### 28.6 Feature Support Matrix

| Backend | Cross-Attention | MLA | Speculative | CUDA Graphs | Sliding Window |
|---------|----------------|-----|-------------|-------------|----------------|
| `flashinfer` | ✓ | ✓ | ✓ | ✓ | ✓ |
| `fa3` | ✗ | ✓ | ✓ | ✓ | ✓ |
| `triton` | ✓ | ✗ | ✓ | ✓ | ✓ |
| `torch_native` | ✓ | ✗ | ✓ | ✓ | ✓ |
| `flex_attention` | ✗ | ✗ | ~ | ~ | ✓ |
| `flashmla` | ✗ | ✓ | ✗ | ✓ | ✗ |
| `cutlass_mla` | ✗ | ✓ | ✗ | ✓ | ✗ |
| `nsa` | ✗ | ✓ | ✗ | ✓ | ✗ |

### 28.7 Performance Recommendations

**For Maximum Performance (NVIDIA H100/B200):**
```yaml
generator:
  attention:
    backend: "fa3"
```

**For Versatility (NVIDIA A100/RTX):**
```yaml
generator:
  attention:
    backend: "flashinfer"
```

**For MLA Models (DeepSeek-V2.5, QwQ):**
```yaml
generator:
  attention:
    backend: "flashmla"
  engine_init_kwargs:
    page_size: 64
```

**For AMD GPUs:**
```yaml
generator:
  attention:
    backend: "aiter"
```

**For Debugging/Compatibility:**
```yaml
generator:
  attention:
    backend: "torch_native"
```

**For Encoder-Decoder Models:**
```yaml
generator:
  attention:
    # fa3 doesn't support cross-attention
    backend: "flashinfer"  # or "triton"
```

### 28.8 NSA (Native Sparse Attention) Configuration

NSA is optimized for models with native sparse attention patterns like DeepSeek-V3.

```yaml
generator:
  attention:
    backend: "nsa"
    nsa:
      # Prefill backend options:
      # "flashmla_sparse", "flashmla_kv", "flashmla_auto", "fa3", "tilelang", "aiter"
      prefill_backend: "flashmla_sparse"

      # Decode backend options (same as prefill)
      decode_backend: "fa3"
```

### 28.9 Multimodal Attention

For vision-language models, configure the multimodal attention backend:

```yaml
generator:
  attention:
    backend: "flashinfer"
    # Multimodal backend options: "sdpa", "fa3", "triton_attn", "ascend_attn", "aiter_attn"
    mm_backend: "fa3"
```

### 28.10 Troubleshooting

**"Backend X not supported on this hardware":**
```yaml
generator:
  attention:
    # Use auto-selection or choose compatible backend
    backend: null  # Let SGLang choose
```

**Cross-attention not working:**
```yaml
generator:
  attention:
    # fa3 doesn't support cross-attention, use flashinfer
    backend: "flashinfer"
```

**MLA model errors:**
```yaml
generator:
  attention:
    backend: "flashmla"
  engine_init_kwargs:
    page_size: 64  # Must match MLA requirements
```

**Slow performance on H100:**
```yaml
generator:
  attention:
    # fa3 is optimized for H100
    backend: "fa3"
```

**Deterministic inference required:**
```yaml
generator:
  attention:
    # Only flashinfer, fa3, triton support deterministic mode
    backend: "flashinfer"
```

### 28.11 Parameter Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `attention.backend` | str | auto | Main attention backend |
| `attention.prefill_backend` | str | null | Override for prefill phase |
| `attention.decode_backend` | str | null | Override for decode phase |
| `attention.mm_backend` | str | null | Multimodal attention backend |
| `attention.enable_double_sparsity` | bool | false | Enable double sparsity (triton) |
| `attention.nsa.prefill_backend` | str | null | NSA prefill backend |
| `attention.nsa.decode_backend` | str | null | NSA decode backend |

---

## 29. LoRA Hot-Swapping (SGLang Only)

SGLang supports serving multiple LoRA adapters simultaneously with runtime loading/unloading, using S-LoRA architecture and Punica's SGMV kernels for efficient multi-adapter serving.

### 29.1 Overview

LoRA hot-swapping enables:
- **Multi-adapter serving**: Serve multiple task-specific adapters from a single model
- **Runtime loading**: Add adapters without server restart
- **Runtime unloading**: Remove adapters to free memory
- **Efficient batching**: Mix requests with different adapters in the same batch
- **Memory management**: LRU/FIFO eviction for adapter memory pools

### 29.2 Configuration

**Basic Multi-Adapter Configuration:**
```yaml
generator:
  backend: sglang
  lora:
    # Pre-load adapters at startup
    paths:
      qa: "path/to/qa_adapter"
      sql: "path/to/sql_adapter"
      code: "path/to/code_adapter"

    # Maximum rank (auto-inferred if null)
    max_rank: 32

    # Max adapters in one batch
    max_loras_per_batch: 8

    # Backend: csgmv (default), triton, ascend, torch_native
    backend: "csgmv"
```

**Alternative Path Formats:**
```yaml
generator:
  lora:
    # List format
    paths: ["path1", "path2", "path3"]

    # Named list format
    paths: ["qa=path/to/qa", "sql=path/to/sql"]

    # With pinning (keep in memory)
    paths:
      - lora_name: "critical_adapter"
        lora_path: "path/to/adapter"
        pinned: true
      - lora_name: "optional_adapter"
        lora_path: "path/to/adapter2"
        pinned: false
```

**Memory Management:**
```yaml
generator:
  lora:
    paths: {"qa": "path1", "sql": "path2"}
    max_rank: 64

    # Max adapters in GPU memory per batch
    max_loras_per_batch: 8

    # Max adapters in CPU memory (for swapping)
    max_loaded_loras: 32

    # Eviction policy when pool is full
    eviction_policy: "lru"  # or "fifo"
```

### 29.3 LoRA Backends

| Backend | Description | embed_tokens/lm_head | Performance |
|---------|-------------|---------------------|-------------|
| `csgmv` | Chunked SGMV (Punica) | ✗ Not supported | Fastest |
| `triton` | Full Triton kernels | ✓ Supported | Fast |
| `ascend` | Ascend NPU | ✗ | Hardware-specific |
| `torch_native` | Pure PyTorch | ✓ | Slowest |

**Backend Selection:**
```yaml
generator:
  lora:
    # Default: fastest for multi-adapter
    backend: "csgmv"

    # Use triton for embed_tokens/lm_head support
    backend: "triton"
    target_modules: ["q_proj", "v_proj", "embed_tokens", "lm_head"]
```

### 29.4 Target Modules

Supported modules for LoRA:

| Module | Description |
|--------|-------------|
| `q_proj` | Query projection in attention |
| `k_proj` | Key projection in attention |
| `v_proj` | Value projection in attention |
| `o_proj` | Output projection in attention |
| `gate_proj` | Gate projection in MLP |
| `up_proj` | Up projection in MLP |
| `down_proj` | Down projection in MLP |
| `qkv_proj` | Fused Q/K/V projection |
| `gate_up_proj` | Fused gate/up projection |
| `embed_tokens` | Embedding layer (triton only) |
| `lm_head` | Output layer (triton only) |
| `all` | All supported modules |

```yaml
generator:
  lora:
    # Specific modules
    target_modules: ["q_proj", "v_proj", "down_proj"]

    # All modules (csgmv will auto-exclude embed_tokens/lm_head)
    target_modules: ["all"]
```

### 29.5 Chunk Size Tuning

The CSGMV backend uses chunked operations for efficiency:

```yaml
generator:
  lora:
    backend: "csgmv"
    # Power of 2 between 16-128
    # Smaller = better for small batches
    # Larger = better throughput for large batches
    max_chunk_size: 16  # default
```

| Chunk Size | Best For |
|------------|----------|
| 16 | Small batches, low latency |
| 32 | Balanced workloads |
| 64 | Large batches |
| 128 | Maximum throughput |

### 29.6 Use Cases

**Multi-Task Serving:**
```yaml
generator:
  lora:
    paths:
      summarization: "adapters/summarization"
      translation: "adapters/translation"
      qa: "adapters/qa"
      code: "adapters/code"
    max_rank: 32
    max_loras_per_batch: 8
```

**A/B Testing:**
```yaml
generator:
  lora:
    paths:
      model_v1: "adapters/v1"
      model_v2: "adapters/v2"
      model_v3: "adapters/v3"
    max_rank: 16
```

**High-Memory Multi-Adapter:**
```yaml
generator:
  lora:
    paths: {...}  # Many adapters
    max_rank: 64
    max_loras_per_batch: 16
    max_loaded_loras: 100
    eviction_policy: "lru"
```

**With Pinning (Keep Critical Adapters):**
```yaml
generator:
  lora:
    paths:
      - lora_name: "production"
        lora_path: "adapters/prod"
        pinned: true  # Never evicted
      - lora_name: "experimental"
        lora_path: "adapters/exp"
        pinned: false  # Can be evicted
```

### 29.7 Memory Considerations

**Memory Usage Formula:**
```
adapter_memory = num_layers × Σ(max_loras_per_batch × rank × (input_dim + output_dim))
                 for each target_module
```

**Example (7B model, rank=32, max_loras_per_batch=8):**
- ~100-500 MB per adapter slot in GPU memory
- Scale with rank and number of target modules

**Memory Optimization:**
```yaml
generator:
  lora:
    # Reduce memory by limiting adapters per batch
    max_loras_per_batch: 4

    # Use smaller rank
    max_rank: 16

    # Limit target modules
    target_modules: ["q_proj", "v_proj"]
```

### 29.8 Constraints and Limitations

1. **Speculative Decoding**: Only NGRAM algorithm compatible with LoRA
2. **Pinning Limit**: Cannot pin all slots (one must remain for eviction)
3. **embed_tokens/lm_head**: Only supported with `backend: "triton"`
4. **Adapter Compatibility**: All adapters must have same target modules

**Validation Rules:**
```python
# max_loaded_loras must be >= max_loras_per_batch
max_loaded_loras >= max_loras_per_batch

# Initial adapter count must fit in max_loaded_loras
len(paths) <= max_loaded_loras

# max_chunk_size must be power of 2
16 <= max_chunk_size <= 128
```

### 29.9 Runtime Loading/Unloading

SGLang provides HTTP endpoints for runtime adapter management:

**Load Adapter:**
```bash
curl -X POST http://localhost:8000/load_lora_adapter \
  -H "Content-Type: application/json" \
  -d '{"lora_name": "new_adapter", "lora_path": "path/to/adapter", "pinned": false}'
```

**Unload Adapter:**
```bash
curl -X POST http://localhost:8000/unload_lora_adapter \
  -H "Content-Type: application/json" \
  -d '{"lora_name": "old_adapter"}'
```

### 29.10 Integration with Training

For RL training with LoRA, you typically use the trainer's LoRA config for weight updates while the generator's LoRA config is for inference-only multi-adapter serving:

```yaml
trainer:
  policy:
    model:
      lora:
        rank: 32
        alpha: 64
        target_modules: "all-linear"

generator:
  backend: sglang
  # For inference-only multi-adapter serving
  lora:
    paths: null  # Use trainer's LoRA for training
    max_rank: 32
    max_loras_per_batch: 1  # Single adapter during training
```

### 29.11 Troubleshooting

**"LoRA adapter not compatible":**
```yaml
generator:
  lora:
    # Ensure max_rank >= adapter rank
    max_rank: 64
    # Ensure target_modules include adapter's modules
    target_modules: ["all"]
```

**Out of memory:**
```yaml
generator:
  lora:
    max_loras_per_batch: 4  # Reduce
    max_rank: 16  # Lower rank
```

**embed_tokens/lm_head not working:**
```yaml
generator:
  lora:
    backend: "triton"  # Required for these modules
    target_modules: ["q_proj", "v_proj", "embed_tokens"]
```

**Slow adapter switching:**
```yaml
generator:
  lora:
    eviction_policy: "lru"  # Better cache efficiency
    max_loaded_loras: 32  # More CPU cache
```

### 29.12 Parameter Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `lora.paths` | list/dict | null | Pre-loaded adapter paths |
| `lora.max_rank` | int | auto | Maximum LoRA rank |
| `lora.target_modules` | list | auto | Modules for LoRA |
| `lora.max_loras_per_batch` | int | 8 | Max adapters per batch |
| `lora.max_loaded_loras` | int | null | Max adapters in CPU memory |
| `lora.eviction_policy` | str | "lru" | "lru" or "fifo" |
| `lora.backend` | str | "csgmv" | LoRA kernel backend |
| `lora.max_chunk_size` | int | 16 | CSGMV chunk size (16-128) |

---

## 30. Priority Scheduling and Request Preemption (SGLang Only)

SGLang provides advanced request scheduling and priority-based preemption capabilities for optimizing throughput, latency, and resource utilization. SkyRL exposes these controls through the `generator.scheduling` configuration.

### 30.1 Overview

Priority scheduling enables:
- **Request prioritization**: Process important requests first
- **Preemption**: High-priority requests can interrupt lower-priority ones
- **Scheduling policies**: Different strategies for request ordering
- **Capacity management**: Control concurrent request limits
- **Chunked prefill**: Smoother latency through incremental prefilling

### 30.2 Basic Configuration

```yaml
generator:
  backend: sglang
  scheduling:
    # Scheduling policy
    policy: "fcfs"  # First-Come-First-Served (default)

    # Enable priority scheduling
    enable_priority: true

    # Preemption threshold
    preemption_threshold: 10
```

### 30.3 Scheduling Policies

SGLang supports multiple scheduling policies:

| Policy | Description | Best For |
|--------|-------------|----------|
| `fcfs` | First-Come-First-Served | General use, fairness |
| `lpm` | Longest Prefix Match | High prefix cache hit rate |
| `dfs-weight` | Weighted DFS tree traversal | Complex prefix patterns |
| `lof` | Longest Output First | Prioritize near-completion |
| `random` | Random selection | Load balancing |

**Example: Optimize for prefix caching:**
```yaml
generator:
  scheduling:
    policy: "lpm"  # Maximize prefix cache reuse
```

**Note**: Priority scheduling only works with `fcfs` or `lof` policies.

### 30.4 Priority Scheduling

When enabled, requests can specify priority values. Lower numeric values indicate higher priority (by default).

```yaml
generator:
  scheduling:
    enable_priority: true

    # Priority value interpretation
    # false (default): higher values = higher priority
    # true: lower values = higher priority
    low_priority_values_first: false

    # Abort requests that specify priority when disabled
    abort_on_priority_when_disabled: false
```

**Setting request priority:**
In your application code, include priority in the sampling params:
```python
sampling_params = {
    "max_tokens": 1024,
    "temperature": 1.0,
    "priority": 0,  # Highest priority (when low_priority_values_first=true)
}
```

### 30.5 Preemption Configuration

Preemption allows high-priority requests to interrupt running lower-priority requests:

```yaml
generator:
  scheduling:
    enable_priority: true

    # Preemption triggers when priority difference > threshold
    # Higher values = less aggressive preemption
    preemption_threshold: 10  # Default
```

**How preemption works:**
1. Request A (priority=0) is running
2. Request B (priority=15) arrives in queue
3. Priority difference = 15 - 0 = 15 > threshold (10)
4. Request A is preempted (paused/aborted) to process B first

**Tuning preemption:**
```yaml
# Aggressive preemption (any priority difference triggers)
generator:
  scheduling:
    preemption_threshold: 0

# Conservative preemption (only large differences)
generator:
  scheduling:
    preemption_threshold: 50
```

### 30.6 Schedule Conservativeness

Control how aggressively the scheduler fills GPU memory:

```yaml
generator:
  scheduling:
    # 0.0 to 1.0+
    # Lower = more aggressive (may cause memory pressure)
    # Higher = more conservative (better stability)
    conservativeness: 1.0  # Default (balanced)
```

| Value | Behavior |
|-------|----------|
| 0.5 | Aggressive - higher throughput, risk of OOM |
| 1.0 | Balanced (default) |
| 2.0 | Very conservative - stable, potentially lower throughput |

### 30.7 Chunked Prefill

Chunked prefill breaks long prompts into smaller pieces for smoother latency:

```yaml
generator:
  scheduling:
    # Maximum tokens per prefill chunk
    chunked_prefill_size: 4096  # null = SGLang default (~8192)

    # Auto-adjust chunk size based on load
    enable_dynamic_chunking: false
```

**Trade-offs:**
- Smaller chunks → Smoother latency, more overhead
- Larger chunks → Better throughput, latency spikes for long prompts

### 30.8 Capacity Limits

Control maximum concurrent requests and tokens:

```yaml
generator:
  scheduling:
    # Maximum concurrent running requests
    max_running_requests: 256  # null = SGLang default

    # Maximum queued requests waiting
    max_queued_requests: 1000  # null = unlimited

    # Maximum tokens in prefill queue
    max_prefill_tokens: 16384  # null = SGLang default

    # Maximum total tokens (all requests)
    max_total_tokens: 131072  # null = memory-based
```

### 30.9 Use Cases

#### High-Priority Interactive Requests

```yaml
generator:
  scheduling:
    policy: "fcfs"
    enable_priority: true
    preemption_threshold: 5  # Aggressive preemption
    low_priority_values_first: true
```

#### Maximum Throughput (Batch Processing)

```yaml
generator:
  scheduling:
    policy: "lof"  # Prioritize near-completion
    conservativeness: 0.8  # Slightly aggressive
    chunked_prefill_size: 8192  # Larger chunks
    max_running_requests: 512
```

#### Prefix Cache Optimization

```yaml
generator:
  scheduling:
    policy: "lpm"  # Longest Prefix Match
    conservativeness: 1.0
```

#### Stable Production (Conservative)

```yaml
generator:
  scheduling:
    policy: "fcfs"
    conservativeness: 1.5
    max_running_requests: 128
    max_queued_requests: 500
    chunked_prefill_size: 4096
```

#### Mixed Priority Workloads

```yaml
generator:
  scheduling:
    policy: "fcfs"
    enable_priority: true
    preemption_threshold: 10  # Moderate preemption
    low_priority_values_first: true
    max_running_requests: 256
    max_queued_requests: 2000  # Large queue for background tasks
```

### 30.10 Scheduling with Other Features

**With CUDA graphs:**
```yaml
generator:
  scheduling:
    policy: "fcfs"
    max_running_requests: 128  # Match CUDA graph batch sizes
  cuda_graph:
    disable: false
    max_bs: 128
```

**With attention backends:**
```yaml
generator:
  scheduling:
    policy: "lpm"
    conservativeness: 1.0
  attention:
    backend: "flashinfer"
```

**With LoRA hot-swapping:**
```yaml
generator:
  scheduling:
    policy: "fcfs"
    max_running_requests: 64  # Lower for multi-adapter overhead
  lora:
    max_loras_per_batch: 4
```

### 30.11 Troubleshooting

**High latency spikes:**
```yaml
generator:
  scheduling:
    chunked_prefill_size: 2048  # Smaller chunks
    enable_dynamic_chunking: true
    conservativeness: 1.2
```

**Out of memory errors:**
```yaml
generator:
  scheduling:
    conservativeness: 1.5  # More conservative
    max_running_requests: 64  # Limit concurrent
    max_total_tokens: 65536  # Cap total tokens
```

**Priority not working:**
```yaml
generator:
  scheduling:
    policy: "fcfs"  # or "lof" - required for priority
    enable_priority: true
    abort_on_priority_when_disabled: true  # Catch errors
```

**Low throughput:**
```yaml
generator:
  scheduling:
    conservativeness: 0.8  # More aggressive
    chunked_prefill_size: 8192  # Larger chunks
    max_running_requests: 256  # Allow more concurrent
```

### 30.12 Parameter Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `scheduling.policy` | str | "fcfs" | Scheduling policy |
| `scheduling.enable_priority` | bool | false | Enable priority scheduling |
| `scheduling.abort_on_priority_when_disabled` | bool | false | Abort priority requests when disabled |
| `scheduling.low_priority_values_first` | bool | false | Lower values = higher priority |
| `scheduling.preemption_threshold` | int | 10 | Priority difference for preemption |
| `scheduling.conservativeness` | float | 1.0 | Schedule aggressiveness (0.0-2.0+) |
| `scheduling.chunked_prefill_size` | int | null | Max tokens per prefill chunk |
| `scheduling.enable_dynamic_chunking` | bool | false | Auto-adjust chunk size |
| `scheduling.max_running_requests` | int | null | Max concurrent running requests |
| `scheduling.max_queued_requests` | int | null | Max queued requests |
| `scheduling.max_prefill_tokens` | int | null | Max tokens in prefill queue |
| `scheduling.max_total_tokens` | int | null | Max total tokens (all requests) |

---

## 31. Disaggregated Prefill/Decode (SGLang Only)

Disaggregated prefill/decode separates the two phases of LLM inference across different GPU workers. Prefill workers process input prompts and generate KV cache, while decode workers perform autoregressive token generation. This architecture can significantly improve throughput and latency characteristics.

### 31.1 Overview

Benefits of disaggregation:
- **Improved throughput**: Prefill and decode can run in parallel on different GPUs
- **Better resource utilization**: Each phase uses optimized configurations
- **Reduced latency variance**: Decode workers aren't interrupted by long prefills
- **Scalability**: Scale prefill and decode workers independently

### 31.2 Architecture

```
┌─────────────────┐     KV Cache      ┌─────────────────┐
│  Prefill Worker │ ───────────────→  │  Decode Worker  │
│  (mode=prefill) │   (via transfer   │  (mode=decode)  │
│                 │    backend)       │                 │
└─────────────────┘                   └─────────────────┘
        ↑                                     │
        │                                     │
   Input Prompt                         Output Tokens
```

### 31.3 Basic Configuration

**Prefill-only instance:**
```yaml
generator:
  backend: sglang
  disaggregation:
    mode: "prefill"
    transfer_backend: "mooncake"
    bootstrap_port: 8998
```

**Decode-only instance:**
```yaml
generator:
  backend: sglang
  disaggregation:
    mode: "decode"
    transfer_backend: "mooncake"
    bootstrap_port: 8998
```

### 31.4 Disaggregation Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| `"null"` | Disabled (default) | Standard unified execution |
| `"prefill"` | Prefill-only worker | Processes prompts, sends KV cache to decode workers |
| `"decode"` | Decode-only worker | Receives KV cache, generates tokens |

### 31.5 Transfer Backends

The transfer backend handles KV cache communication between workers:

| Backend | Description | Requirements |
|---------|-------------|--------------|
| `"mooncake"` | High-performance RDMA transfer (default) | RDMA-capable network |
| `"nixl"` | NixL architecture backend | NixL infrastructure |
| `"ascend"` | Huawei Ascend NPU backend | Ascend hardware |
| `"fake"` | Testing/development backend | None (no actual transfer) |

```yaml
generator:
  disaggregation:
    mode: "prefill"
    transfer_backend: "mooncake"
    ib_device: "mlx5_0"  # Specific InfiniBand device
```

### 31.6 Worker Configuration

#### Decode Worker Settings (for prefill instances)

Configure how decode workers receive and handle KV cache:

```yaml
generator:
  disaggregation:
    mode: "prefill"
    decode:
      # Tensor parallel size for decode workers
      tp_size: 4

      # Data parallel size for decode workers
      dp_size: 2

      # Enable KV cache offloading to CPU
      enable_offload_kvcache: true

      # Polling interval for KV cache checks (ms)
      polling_interval: 1

      # Enable fake auto mode for testing
      enable_fake_auto: false
```

#### Prefill Worker Settings (for decode instances)

Configure prefill worker pipeline parallelism:

```yaml
generator:
  disaggregation:
    mode: "decode"
    prefill:
      # Pipeline parallel size for prefill workers
      pp_size: 2
```

### 31.7 Data Parallel Attention (DP-Attention)

DP-Attention enables attention computation across data parallel workers, useful with or without disaggregation:

```yaml
generator:
  disaggregation:
    enable_dp_attention: true
    enable_dp_lm_head: true
```

This is particularly useful for:
- Very large batch sizes
- Memory-limited scenarios
- Combined with disaggregation for maximum throughput

### 31.8 Bootstrap and Coordination

Workers coordinate through a bootstrap handshake:

```yaml
generator:
  disaggregation:
    mode: "prefill"
    bootstrap_port: 8998  # Port for coordination
    num_reserved_decode_tokens: 512  # Reserved KV cache capacity
```

### 31.9 Use Cases

#### High-Throughput Batch Processing

```yaml
# Prefill instance - optimized for large batches
generator:
  disaggregation:
    mode: "prefill"
    transfer_backend: "mooncake"
    decode:
      tp_size: 4
      dp_size: 2
  scheduling:
    max_prefill_tokens: 32768
    chunked_prefill_size: 8192
```

```yaml
# Decode instance - optimized for token generation
generator:
  disaggregation:
    mode: "decode"
    transfer_backend: "mooncake"
    num_reserved_decode_tokens: 1024
  scheduling:
    max_running_requests: 512
```

#### Low-Latency Interactive

```yaml
# Prefill instance - minimize prefill latency
generator:
  disaggregation:
    mode: "prefill"
    decode:
      tp_size: 2
  piecewise_cuda_graph:
    enabled: true
```

```yaml
# Decode instance - maximize decode throughput
generator:
  disaggregation:
    mode: "decode"
  cuda_graph:
    disable: false
    max_bs: 64
```

#### Memory-Constrained Deployment

```yaml
generator:
  disaggregation:
    mode: "decode"
    decode:
      enable_offload_kvcache: true  # Offload to CPU when GPU full
    num_reserved_decode_tokens: 256  # Smaller reservation
```

#### Testing and Development

```yaml
generator:
  disaggregation:
    mode: "prefill"
    transfer_backend: "fake"  # No actual transfer
    decode:
      enable_fake_auto: true  # Skip bootstrap_host
```

### 31.10 Multi-Node Deployment

For production deployments across multiple nodes:

```yaml
# Node 1: Prefill workers
generator:
  disaggregation:
    mode: "prefill"
    transfer_backend: "mooncake"
    ib_device: "mlx5_0"
    bootstrap_port: 8998
    decode:
      tp_size: 8
      dp_size: 4

# Node 2+: Decode workers
generator:
  disaggregation:
    mode: "decode"
    transfer_backend: "mooncake"
    ib_device: "mlx5_0"
    bootstrap_port: 8998
    prefill:
      pp_size: 2
```

### 31.11 Combining with Other Features

**With attention backends:**
```yaml
generator:
  disaggregation:
    mode: "prefill"
  attention:
    prefill_backend: "flashinfer"
    decode_backend: "fa3"
```

**With priority scheduling:**
```yaml
generator:
  disaggregation:
    mode: "decode"
  scheduling:
    enable_priority: true
    policy: "fcfs"
```

**With LoRA hot-swapping:**
```yaml
generator:
  disaggregation:
    mode: "decode"
  lora:
    max_loras_per_batch: 4
    eviction_policy: "lru"
```

### 31.12 Troubleshooting

**KV cache transfer failures:**
```yaml
generator:
  disaggregation:
    transfer_backend: "mooncake"
    ib_device: "mlx5_0"  # Specify exact device
    decode:
      polling_interval: 5  # Increase polling interval
```

**Out of memory on decode workers:**
```yaml
generator:
  disaggregation:
    mode: "decode"
    decode:
      enable_offload_kvcache: true
    num_reserved_decode_tokens: 256  # Reduce reservation
```

**Bootstrap handshake timeout:**
```yaml
generator:
  disaggregation:
    bootstrap_port: 9000  # Try different port
    # Ensure port is open in firewall
```

**Testing without RDMA:**
```yaml
generator:
  disaggregation:
    mode: "prefill"
    transfer_backend: "fake"  # Use fake backend for testing
```

### 31.13 Parameter Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `disaggregation.mode` | str | "null" | Disaggregation mode: "null", "prefill", "decode" |
| `disaggregation.transfer_backend` | str | "mooncake" | KV cache transfer backend |
| `disaggregation.bootstrap_port` | int | 8998 | Bootstrap handshake port |
| `disaggregation.ib_device` | str | null | InfiniBand device name |
| `disaggregation.num_reserved_decode_tokens` | int | 512 | Reserved tokens for decode KV cache |
| `disaggregation.enable_dp_attention` | bool | false | Enable DP-Attention |
| `disaggregation.enable_dp_lm_head` | bool | false | Enable DP for LM head |
| `disaggregation.decode.tp_size` | int | null | Decode worker TP size |
| `disaggregation.decode.dp_size` | int | null | Decode worker DP size |
| `disaggregation.decode.enable_offload_kvcache` | bool | false | Enable KV cache offloading |
| `disaggregation.decode.polling_interval` | int | 1 | KV cache polling interval (ms) |
| `disaggregation.decode.enable_fake_auto` | bool | false | Enable fake auto mode |
| `disaggregation.prefill.pp_size` | int | 1 | Prefill worker PP size |

---

## 32. Multi-Node Inference (SGLang Only)

SGLang supports distributed inference across multiple nodes with optimized NCCL communication. SkyRL exposes these capabilities through the `generator.multi_node` configuration, enabling large-scale model deployment.

### 32.1 Overview

Multi-node inference enables:
- **Larger models**: Deploy models that don't fit on a single node
- **Higher throughput**: Scale horizontally across multiple machines
- **Tensor parallelism**: Split model layers across nodes
- **Pipeline parallelism**: Distribute pipeline stages across nodes
- **Optimized communication**: NCCL tuning for InfiniBand/NVLink

### 32.2 Architecture

```
┌──────────────────┐     NCCL      ┌──────────────────┐
│     Node 0       │ ←──────────→  │     Node 1       │
│  (rank 0, GPUs)  │               │  (rank 1, GPUs)  │
│                  │               │                  │
│  TP workers 0-3  │               │  TP workers 4-7  │
└──────────────────┘               └──────────────────┘
         ↑                                  ↑
         │          Ray Cluster             │
         └──────────────────────────────────┘
```

### 32.3 Basic Configuration

**Two-node tensor parallelism:**
```yaml
generator:
  backend: sglang
  inference_engine_tensor_parallel_size: 8  # 4 GPUs per node × 2 nodes
  multi_node:
    nnodes: 2
    nccl:
      enable_symm_mem: true
```

**Manual node configuration:**
```yaml
# Node 0 (master)
generator:
  multi_node:
    nnodes: 2
    node_rank: 0
    dist_init_addr: "10.0.0.1:29500"

# Node 1
generator:
  multi_node:
    nnodes: 2
    node_rank: 1
    dist_init_addr: "10.0.0.1:29500"
```

### 32.4 NCCL Configuration

NCCL (NVIDIA Collective Communications Library) optimizations for multi-node communication:

#### Symmetric Memory

Enable NVIDIA symmetric memory for faster all-reduce operations:

```yaml
generator:
  multi_node:
    nccl:
      enable_symm_mem: true  # Requires CUDA 12.4+
```

This enables:
- `NCCL_CUMEM_ENABLE=1` - Symmetric memory allocation
- `NCCL_NVLS_ENABLE=1` - NVLink Switch (auto-enabled)

#### NVLink Switch (NVLS)

For systems with NVLink connectivity:

```yaml
generator:
  multi_node:
    nccl:
      enable_nvls: true  # Enable without symm_mem
```

#### NCCL Timeout

Increase timeout for large models or slow networks:

```yaml
generator:
  multi_node:
    nccl:
      timeout: 1200  # 20 minutes (default: 600 seconds)
```

#### NCCL Debugging

For troubleshooting multi-node communication:

```yaml
generator:
  multi_node:
    nccl:
      debug_level: "INFO"  # WARN, INFO, DEBUG, TRACE
```

### 32.5 InfiniBand Optimization

For InfiniBand/RoCE networks:

```yaml
generator:
  multi_node:
    enable_ib_optimization: true
```

This sets:
- `NCCL_IB_DISABLE=0` - Enable InfiniBand
- `NCCL_NET_GDR_LEVEL=5` - GPU Direct RDMA level

### 32.6 CUDA Configuration

Control GPU connection parallelism:

```yaml
generator:
  multi_node:
    cuda_device_max_connections: 16  # Default: 8
```

Higher values improve parallelism but consume more resources.

### 32.7 Parallelism Strategies

#### Tensor Parallelism (TP)

Split model weights across GPUs/nodes:

```yaml
generator:
  # 8-way TP across 2 nodes (4 GPUs each)
  inference_engine_tensor_parallel_size: 8
  multi_node:
    nnodes: 2
    nccl:
      enable_symm_mem: true
```

#### Pipeline Parallelism (PP)

Distribute layers across pipeline stages:

```yaml
generator:
  # 2 pipeline stages
  inference_engine_pipeline_parallel_size: 2
  multi_node:
    nnodes: 2
```

#### Combined TP + PP

For very large models:

```yaml
generator:
  # 4-way TP × 2 PP = 8 GPUs
  inference_engine_tensor_parallel_size: 4
  inference_engine_pipeline_parallel_size: 2
  multi_node:
    nnodes: 2
    nccl:
      enable_symm_mem: true
```

#### Expert Parallelism (EP) for MoE

Distribute MoE experts across nodes:

```yaml
generator:
  inference_engine_expert_parallel_size: 8
  multi_node:
    nnodes: 2
```

### 32.8 Use Cases

#### Large Model Deployment (70B+)

```yaml
generator:
  inference_engine_tensor_parallel_size: 8
  multi_node:
    nnodes: 2
    nccl:
      enable_symm_mem: true
      timeout: 1200
    enable_ib_optimization: true
```

#### High-Throughput Production

```yaml
generator:
  inference_engine_tensor_parallel_size: 4
  inference_engine_data_parallel_size: 2
  multi_node:
    nnodes: 2
    nccl:
      enable_nvls: true
    cuda_device_max_connections: 16
```

#### Development and Testing

```yaml
generator:
  multi_node:
    nnodes: 2
    nccl:
      debug_level: "INFO"
      timeout: 300  # Lower timeout for faster failure detection
```

#### MoE Model Deployment

```yaml
generator:
  inference_engine_tensor_parallel_size: 4
  inference_engine_expert_parallel_size: 8
  multi_node:
    nnodes: 4
    nccl:
      enable_symm_mem: true
```

### 32.9 Ray Cluster Integration

SkyRL uses Ray for multi-node orchestration. With Ray:

```yaml
generator:
  multi_node:
    # nnodes and node_rank auto-detected from Ray
    nccl:
      enable_symm_mem: true
```

**Starting Ray cluster:**
```bash
# On head node
ray start --head --port=6379

# On worker nodes
ray start --address='head-node-ip:6379'
```

### 32.10 Combining with Other Features

**With disaggregation:**
```yaml
generator:
  multi_node:
    nnodes: 4
    nccl:
      enable_symm_mem: true
  disaggregation:
    mode: "prefill"
```

**With attention backends:**
```yaml
generator:
  multi_node:
    nnodes: 2
  attention:
    backend: "flashinfer"
```

**With scheduling:**
```yaml
generator:
  multi_node:
    nnodes: 2
  scheduling:
    policy: "lpm"
    max_running_requests: 512
```

### 32.11 Troubleshooting

**NCCL timeout errors:**
```yaml
generator:
  multi_node:
    nccl:
      timeout: 1800  # Increase to 30 minutes
      debug_level: "INFO"
```

**Slow communication:**
```yaml
generator:
  multi_node:
    enable_ib_optimization: true
    nccl:
      enable_symm_mem: true
    cuda_device_max_connections: 16
```

**Connection failures:**
```yaml
generator:
  multi_node:
    dist_init_addr: "master-hostname:29500"
    nccl:
      debug_level: "DEBUG"
```

**Out of memory:**
```yaml
generator:
  # Reduce TP, increase PP
  inference_engine_tensor_parallel_size: 4
  inference_engine_pipeline_parallel_size: 2
```

### 32.12 Environment Variables

Multi-node configuration sets these environment variables:

| Variable | Description |
|----------|-------------|
| `NCCL_CUMEM_ENABLE` | Enable symmetric memory |
| `NCCL_NVLS_ENABLE` | Enable NVLink Switch |
| `NCCL_TIMEOUT` | NCCL operation timeout |
| `NCCL_DEBUG` | Debug verbosity level |
| `NCCL_IB_DISABLE` | InfiniBand disable flag |
| `NCCL_NET_GDR_LEVEL` | GPU Direct RDMA level |
| `CUDA_DEVICE_MAX_CONNECTIONS` | Max GPU connections |
| `SKYRL_WORKER_NCCL_TIMEOUT_IN_S` | SkyRL NCCL timeout |

### 32.13 Parameter Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `multi_node.nnodes` | int | null | Number of nodes in cluster |
| `multi_node.node_rank` | int | null | Current node rank (0-indexed) |
| `multi_node.dist_init_addr` | str | null | Master node address (hostname:port) |
| `multi_node.nccl.enable_symm_mem` | bool | false | Enable NVIDIA symmetric memory |
| `multi_node.nccl.enable_nvls` | bool | false | Enable NVLS (NVLink Switch) |
| `multi_node.nccl.timeout` | int | null | NCCL timeout in seconds |
| `multi_node.nccl.debug_level` | str | null | NCCL debug level (WARN/INFO/DEBUG/TRACE) |
| `multi_node.enable_ib_optimization` | bool | false | Enable InfiniBand optimization |
| `multi_node.cuda_device_max_connections` | int | null | Max CUDA connections per GPU |

---

## 33. Prometheus Metrics and Observability (SGLang Only)

SGLang provides comprehensive observability through Prometheus metrics, OpenTelemetry tracing, and structured request logging. SkyRL exposes these capabilities through the `generator.metrics` configuration.

### 33.1 Overview

Observability features include:
- **Prometheus metrics**: 50+ metrics covering latency, throughput, cache hits, GPU usage
- **OpenTelemetry tracing**: Distributed request tracing for debugging
- **Request logging**: Structured logging of inputs, outputs, and metadata
- **Per-request export**: Detailed metrics export to files for analysis

### 33.2 Basic Configuration

**Enable Prometheus metrics:**
```yaml
generator:
  backend: sglang
  metrics:
    enabled: true
```

**Complete observability setup:**
```yaml
generator:
  metrics:
    enabled: true
    tracing:
      enabled: true
      otlp_endpoint: "localhost:4317"
    logging:
      enabled: true
      format: "json"
```

### 33.3 Prometheus Metrics

When enabled, metrics are exposed on the `/metrics` endpoint (same port as server).

#### Enabling Metrics

```yaml
generator:
  metrics:
    enabled: true

    # Enable metrics on all TP ranks (for dp_attention setups)
    enable_for_all_schedulers: false
```

#### Key Metrics Collected

| Metric | Type | Description |
|--------|------|-------------|
| `sglang:num_running_reqs` | Gauge | Currently running requests |
| `sglang:gen_throughput` | Gauge | Generation throughput (tokens/s) |
| `sglang:cache_hit_rate` | Gauge | Prefix cache hit rate |
| `sglang:time_to_first_token_seconds` | Histogram | Time-to-first-token latency |
| `sglang:inter_token_latency_seconds` | Histogram | Inter-token latency |
| `sglang:queue_time_seconds` | Histogram | Request queueing time |
| `sglang:token_usage` | Gauge | KV cache token usage |
| `sglang:gpu_execution_seconds_total` | Counter | Total GPU execution time |

### 33.4 Latency Histogram Buckets

Customize histogram buckets for precise latency analysis:

```yaml
generator:
  metrics:
    enabled: true
    buckets:
      # Time-to-first-token buckets (seconds)
      time_to_first_token: [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]

      # Inter-token latency buckets (seconds)
      inter_token_latency: [0.01, 0.05, 0.1, 0.2, 0.5]

      # End-to-end request latency buckets (seconds)
      e2e_request_latency: [1.0, 5.0, 10.0, 30.0, 60.0]
```

### 33.5 Token Histograms

Enable token count distribution analysis:

```yaml
generator:
  metrics:
    enabled: true
    collect_tokens_histogram: true

    # Bucket rules for prompt tokens
    # Options: "default", "tse", or custom list
    prompt_tokens_buckets: "default"

    # Bucket rules for generation tokens
    generation_tokens_buckets: "default"
```

### 33.6 Per-Request Metrics Export

Export detailed per-request metrics to files:

```yaml
generator:
  metrics:
    enabled: true
    export_to_file:
      enabled: true
      directory: "/var/log/sglang/metrics"
```

**Exported data includes:**
- Prompt/completion token counts
- Time-to-first-token
- Inter-token latencies
- Queue and forward durations
- Sampling parameters
- Finish reasons

**File format:** Hourly files named `sglang-request-metrics-{hour}.log`

### 33.7 Custom Labels

Add custom labels to metrics for multi-tenant or routing scenarios:

```yaml
generator:
  metrics:
    enabled: true
    custom_labels:
      # HTTP header for custom labels
      header: "x-custom-labels"

      # Allowed label names (null = allow all)
      allowed:
        - "tenant_id"
        - "request_type"
        - "region"
```

**Using custom labels in requests:**
```bash
curl -H "x-custom-labels: tenant_id=abc,request_type=chat" ...
```

### 33.8 OpenTelemetry Tracing

Enable distributed tracing for request debugging:

```yaml
generator:
  metrics:
    tracing:
      enabled: true
      otlp_endpoint: "localhost:4317"
```

**Jaeger/Zipkin setup:**
```yaml
generator:
  metrics:
    tracing:
      enabled: true
      otlp_endpoint: "jaeger-collector:4317"
```

### 33.9 Request Logging

Enable structured request logging:

```yaml
generator:
  metrics:
    logging:
      enabled: true

      # Verbosity level (0-3)
      # 0 = metadata only
      # 1 = + sampling params
      # 2 = + partial I/O (default)
      # 3 = full I/O
      level: 2

      # Format: "text" or "json"
      format: "json"

      # Log targets (stdout and/or file paths)
      targets:
        - "stdout"
        - "/var/log/sglang/requests"
```

### 33.10 Use Cases

#### Production Monitoring

```yaml
generator:
  metrics:
    enabled: true
    enable_for_all_schedulers: true
    buckets:
      time_to_first_token: [0.1, 0.25, 0.5, 1.0, 2.0, 5.0]
      inter_token_latency: [0.01, 0.025, 0.05, 0.1, 0.25]
    collect_tokens_histogram: true
```

#### Debugging and Development

```yaml
generator:
  metrics:
    enabled: true
    tracing:
      enabled: true
      otlp_endpoint: "localhost:4317"
    logging:
      enabled: true
      level: 3  # Full I/O logging
      format: "json"
      targets:
        - "stdout"
```

#### Multi-Tenant Monitoring

```yaml
generator:
  metrics:
    enabled: true
    custom_labels:
      header: "x-tenant-id"
      allowed:
        - "tenant_id"
        - "service_name"
```

#### Performance Analysis

```yaml
generator:
  metrics:
    enabled: true
    export_to_file:
      enabled: true
      directory: "/data/metrics"
    collect_tokens_histogram: true
    buckets:
      time_to_first_token: [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
```

### 33.11 Grafana Dashboard

Example Prometheus queries for Grafana:

**Throughput:**
```promql
rate(sglang:gen_throughput[5m])
```

**P99 Time-to-First-Token:**
```promql
histogram_quantile(0.99, rate(sglang:time_to_first_token_seconds_bucket[5m]))
```

**Cache Hit Rate:**
```promql
sglang:cache_hit_rate
```

**Running Requests:**
```promql
sglang:num_running_reqs
```

**Token Usage:**
```promql
sglang:token_usage
```

### 33.12 Combining with Other Features

**With multi-node:**
```yaml
generator:
  multi_node:
    nnodes: 2
  metrics:
    enabled: true
    enable_for_all_schedulers: true
```

**With disaggregation:**
```yaml
generator:
  disaggregation:
    mode: "prefill"
  metrics:
    enabled: true
    # Disaggregation-specific metrics auto-collected
```

**With scheduling:**
```yaml
generator:
  scheduling:
    policy: "lpm"
  metrics:
    enabled: true
    # Scheduling metrics auto-collected
```

### 33.13 Troubleshooting

**Metrics not appearing:**
```yaml
generator:
  metrics:
    enabled: true
    enable_for_all_schedulers: true  # If using DP attention
```

**High cardinality labels:**
```yaml
generator:
  metrics:
    custom_labels:
      # Restrict allowed labels
      allowed:
        - "service"
        - "region"
```

**Large metric files:**
```yaml
generator:
  metrics:
    export_to_file:
      enabled: true
      directory: "/data/metrics"
      # Files rotate hourly automatically
```

### 33.14 Parameter Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `metrics.enabled` | bool | false | Enable Prometheus metrics |
| `metrics.enable_for_all_schedulers` | bool | false | Enable metrics on all TP ranks |
| `metrics.buckets.time_to_first_token` | list | null | TTFT histogram buckets |
| `metrics.buckets.inter_token_latency` | list | null | ITL histogram buckets |
| `metrics.buckets.e2e_request_latency` | list | null | E2E latency histogram buckets |
| `metrics.collect_tokens_histogram` | bool | false | Enable token count histograms |
| `metrics.prompt_tokens_buckets` | str/list | null | Prompt token histogram buckets |
| `metrics.generation_tokens_buckets` | str/list | null | Generation token histogram buckets |
| `metrics.export_to_file.enabled` | bool | false | Enable per-request metrics export |
| `metrics.export_to_file.directory` | str | null | Directory for metrics files |
| `metrics.custom_labels.header` | str | "x-custom-labels" | HTTP header for custom labels |
| `metrics.custom_labels.allowed` | list | null | Allowed custom label names |
| `metrics.tracing.enabled` | bool | false | Enable OpenTelemetry tracing |
| `metrics.tracing.otlp_endpoint` | str | "localhost:4317" | OTLP Collector endpoint |
| `metrics.logging.enabled` | bool | false | Enable request logging |
| `metrics.logging.level` | int | 2 | Logging verbosity (0-3) |
| `metrics.logging.format` | str | "text" | Log format (text/json) |
| `metrics.logging.targets` | list | null | Log destinations |

---

## 34. Load Balancing and Request Routing (SGLang Only)

SkyRL exposes SGLang's load balancing and request routing capabilities for distributing requests across data parallel workers, managing expert parallelism in MoE models, and optimizing request batching.

### Overview

Load balancing in SGLang operates at multiple levels:
1. **Request Distribution**: How incoming requests are distributed across DP workers
2. **Expert Parallelism (EP)**: How MoE experts are partitioned and load-balanced
3. **Request Batching**: How requests are grouped for efficient processing

### Load Balance Methods

| Method | Description | Use Case |
|--------|-------------|----------|
| `auto` | Automatic selection based on configuration | Default - works for most setups |
| `round_robin` | Distributes requests in round-robin order | Simple, predictable distribution |
| `shortest_queue` | Routes to worker with fewest pending requests | Load-aware, reduces tail latency |
| `minimum_tokens` | Routes based on pending token count (deprecated) | Token-aware distribution |
| `follow_bootstrap_room` | Routes based on bootstrap room assignment | Prefill-Decode disaggregation |

### Configuration

```yaml
generator:
  backend: sglang

  load_balancing:
    # Request distribution method across DP workers
    method: "shortest_queue"  # or "round_robin", "auto"

    # Expert Parallelism (EP) for MoE models
    expert_parallelism:
      ep_size: 4                    # Number of expert groups
      dispatch_algorithm: null      # Rank selection algorithm
      num_redundant_experts: 2      # Redundant experts for load balancing
      init_expert_location: "trivial"  # Initial expert assignment

    # Expert-Parallel Load Balancing (EPLB)
    eplb:
      enabled: true
      algorithm: "auto"
      rebalance_num_iterations: 500    # Rebalance frequency
      rebalance_layers_per_chunk: 4    # Layers per rebalance step
      min_rebalancing_utilization_threshold: 0.8

    # Expert distribution monitoring
    expert_metrics:
      recorder_mode: null
      recorder_buffer_size: null
      enabled: true  # Log expert load statistics

    # Request batching configuration
    batching:
      max_prefill_tokens: 32768    # Max tokens per prefill batch
      max_total_tokens: null       # Max tokens in memory pool
      tokenizer_worker_num: 2      # Parallel tokenizer workers
```

### Expert Parallelism (EP)

Expert Parallelism partitions MoE experts across multiple GPUs. Each GPU handles a subset of experts, enabling serving of larger MoE models.

```yaml
generator:
  # Basic EP setup for Mixtral 8x7B
  inference_engine_expert_parallel_size: 4  # From existing config

  load_balancing:
    expert_parallelism:
      ep_size: 4
      num_redundant_experts: 1  # Trade memory for better balance
```

**How EP Works:**
- With `ep_size=4` and 8 experts, each GPU handles 2 experts
- Redundant experts are copied to multiple GPUs for load balancing
- Dispatch algorithm determines which GPU handles each token's expert routing

### Expert-Parallel Load Balancing (EPLB)

EPLB dynamically rebalances expert assignments based on runtime utilization metrics.

```yaml
generator:
  load_balancing:
    eplb:
      enabled: true
      algorithm: "auto"

      # Rebalance every 500 iterations
      rebalance_num_iterations: 500

      # Spread rebalancing overhead across 4 forward passes
      rebalance_layers_per_chunk: 4

      # Only rebalance when GPU utilization exceeds 80%
      min_rebalancing_utilization_threshold: 0.8
```

**EPLB Benefits:**
- Reduces hot-spot experts that cause throughput bottlenecks
- Adapts to workload distribution changes
- Minimizes expert load imbalance in production

### Request Batching

Configure how requests are grouped for efficient GPU utilization:

```yaml
generator:
  load_balancing:
    batching:
      # Larger batches = higher throughput, higher latency
      max_prefill_tokens: 32768

      # Memory pool size (null = auto-calculated)
      max_total_tokens: 131072

      # More tokenizer workers for high request volume
      tokenizer_worker_num: 4
```

### Use Cases

#### High-Throughput Serving

```yaml
generator:
  inference_engine_data_parallel_size: 4

  load_balancing:
    method: "shortest_queue"  # Load-aware distribution
    batching:
      max_prefill_tokens: 65536
      tokenizer_worker_num: 8
```

#### MoE Model with Load Balancing

```yaml
generator:
  # Mixtral or DeepSeek model
  inference_engine_expert_parallel_size: 8

  load_balancing:
    expert_parallelism:
      ep_size: 8
      num_redundant_experts: 2
    eplb:
      enabled: true
      rebalance_num_iterations: 1000
    expert_metrics:
      enabled: true  # Monitor expert load distribution
```

#### Prefill-Decode Disaggregation

```yaml
generator:
  disaggregation:
    mode: "prefill"  # or "decode"

  load_balancing:
    # Automatic for PD mode, or explicit:
    method: "follow_bootstrap_room"
```

### Monitoring Expert Load Balance

When `expert_metrics.enabled: true`, SGLang logs expert distribution statistics:

```
[INFO] Expert distribution - Layer 0: mean=125.3, std=12.4, max_ratio=1.32
[INFO] Expert distribution - Layer 1: mean=124.8, std=8.2, max_ratio=1.18
```

A `max_ratio` close to 1.0 indicates good load balance. High values (>1.5) suggest hot-spot experts that may benefit from EPLB or more redundant experts.

### Performance Considerations

| Configuration | Throughput | Latency | Memory |
|--------------|------------|---------|--------|
| Round-robin | Medium | Low variance | Baseline |
| Shortest-queue | High | Lower tail | Baseline |
| EPLB enabled | Higher | May vary | +5-10% |
| More redundant experts | Higher | Lower | +per-expert |

### Quick Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `load_balancing.method` | str | null | Load balance method (auto/round_robin/shortest_queue) |
| `load_balancing.expert_parallelism.ep_size` | int | null | Expert parallelism size |
| `load_balancing.expert_parallelism.dispatch_algorithm` | str | null | EP dispatch algorithm |
| `load_balancing.expert_parallelism.num_redundant_experts` | int | null | Redundant experts count |
| `load_balancing.expert_parallelism.init_expert_location` | str | null | Initial expert location |
| `load_balancing.eplb.enabled` | bool | false | Enable EPLB |
| `load_balancing.eplb.algorithm` | str | null | EPLB algorithm |
| `load_balancing.eplb.rebalance_num_iterations` | int | null | Iterations between rebalance |
| `load_balancing.eplb.rebalance_layers_per_chunk` | int | null | Layers per rebalance chunk |
| `load_balancing.eplb.min_rebalancing_utilization_threshold` | float | null | Min utilization to trigger |
| `load_balancing.expert_metrics.recorder_mode` | str | null | Expert distribution recorder mode |
| `load_balancing.expert_metrics.recorder_buffer_size` | int | null | Recorder buffer size |
| `load_balancing.expert_metrics.enabled` | bool | false | Log expert balancedness metrics |
| `load_balancing.batching.max_prefill_tokens` | int | null | Max tokens in prefill batch |
| `load_balancing.batching.max_total_tokens` | int | null | Max tokens in memory pool |
| `load_balancing.batching.tokenizer_worker_num` | int | null | Tokenizer worker processes |

---

## 35. Health Checks and Kubernetes Probes (SGLang Only)

SkyRL exposes SGLang's health check and watchdog configuration for production deployments, enabling proper Kubernetes liveness/readiness probes and automatic recovery from stuck processes.

### Overview

SGLang provides multiple health check mechanisms:
1. **HTTP Endpoints**: `/health`, `/ping` for liveness/readiness probes
2. **gRPC Health Service**: Standard `grpc.health.v1.Health` for gRPC probes
3. **Watchdog**: Monitors scheduler/tokenizer and crashes on timeout to prevent hangs
4. **Startup Probes**: Configurable timeouts for model loading and warmup

### Available Endpoints

| Endpoint | Protocol | Purpose | Response |
|----------|----------|---------|----------|
| `/health` | HTTP GET | Readiness/Liveness probe | 200 (healthy), 503 (unhealthy) |
| `/ping` | HTTP GET | SageMaker-compatible health | 200 (always) |
| `/` | HTTP GET | Ollama-compatible health | "Ollama is running" |
| `grpc.health.v1.Health/Check` | gRPC | Standard gRPC health check | SERVING/NOT_SERVING |

### Configuration

```yaml
generator:
  backend: sglang

  health_checks:
    # Watchdog Configuration
    watchdog:
      timeout: 300          # Hard timeout - crash if exceeded
      soft_timeout: 120     # Soft timeout - dump debug info

    # Distributed initialization timeout
    dist_timeout: 600       # For multi-node setups

    # Health endpoint behavior
    endpoint:
      timeout: 30           # Health check request timeout
      enable_generation: false  # Disable test generation for faster checks

    # Startup timeouts
    startup:
      weights_ready_timeout: 300   # Model loading timeout
      warmup_timeout: 1800         # Kernel JIT compilation time
```

### Watchdog Configuration

The watchdog monitors forward batch execution and takes action if batches hang:

```yaml
generator:
  health_checks:
    watchdog:
      # Hard timeout: Process crashes if exceeded
      # Prevents indefinite hangs in production
      timeout: 300  # 5 minutes

      # Soft timeout: Dumps debug info without crashing
      # Useful for diagnosing performance issues
      soft_timeout: 120  # 2 minutes
```

**Watchdog Behavior:**
- **Hard timeout**: If any forward batch takes longer than `timeout`, the process crashes. This ensures pods get restarted by Kubernetes rather than hanging indefinitely.
- **Soft timeout**: If exceeded, dumps stack traces and memory info for debugging, but continues execution.

### Health Endpoint Configuration

Control how the `/health` endpoint behaves:

```yaml
generator:
  health_checks:
    endpoint:
      # How long to wait for scheduler response
      timeout: 30

      # Whether to perform actual inference during health checks
      # true = sends 1-token generation request (thorough but slower)
      # false = only checks connectivity (faster)
      enable_generation: true
```

**When to Disable Generation:**
- High-traffic deployments where health check latency matters
- When using aggressive probe intervals (< 5 seconds)
- When connection-level health is sufficient

### Kubernetes Deployment Examples

#### Standard Deployment

```yaml
# Kubernetes Deployment
apiVersion: apps/v1
kind: Deployment
spec:
  template:
    spec:
      containers:
      - name: sglang
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 300  # Wait for model loading
          periodSeconds: 30
          timeoutSeconds: 30
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 10
          timeoutSeconds: 20
          failureThreshold: 3
        startupProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 30
          failureThreshold: 30  # 5 minutes total
```

#### SkyRL Config for Kubernetes

```yaml
generator:
  backend: sglang

  health_checks:
    watchdog:
      timeout: 600          # 10 minutes for large batches
      soft_timeout: 300     # Debug after 5 minutes

    endpoint:
      timeout: 25           # Slightly less than probe timeout
      enable_generation: true  # Full health verification

    startup:
      weights_ready_timeout: 300   # Large model loading
      warmup_timeout: 1800         # Allow kernel compilation
```

### Large Model Deployment

For large models (70B+) with slow loading and kernel compilation:

```yaml
generator:
  health_checks:
    watchdog:
      timeout: 900          # 15 minutes for large batches
      soft_timeout: 600

    startup:
      weights_ready_timeout: 600   # 10 minutes for model loading
      warmup_timeout: 3600         # 1 hour for kernel JIT
```

### Multi-Node Deployment

For distributed inference across multiple nodes:

```yaml
generator:
  multi_node:
    nnodes: 4
    # ... other multi-node config

  health_checks:
    # Longer timeout for distributed initialization
    dist_timeout: 1200  # 20 minutes

    watchdog:
      timeout: 600  # Longer for cross-node coordination
```

### Fast Health Checks

For latency-sensitive deployments with minimal probe overhead:

```yaml
generator:
  health_checks:
    endpoint:
      timeout: 5
      enable_generation: false  # Skip inference test

    watchdog:
      timeout: 120  # Faster crash recovery
```

### Environment Variables

These parameters are set as environment variables in the SGLang process:

| Config Path | Environment Variable | Default |
|-------------|---------------------|---------|
| `endpoint.timeout` | `SGLANG_HEALTH_CHECK_TIMEOUT` | 20 |
| `endpoint.enable_generation` | `SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION` | true |
| `startup.weights_ready_timeout` | `SGLANG_WAIT_WEIGHTS_READY_TIMEOUT` | 120 |
| `startup.warmup_timeout` | `SGLANG_WARMUP_TIMEOUT` | -1 (disabled) |

### Debugging Health Issues

When pods fail health checks, check:

1. **Watchdog triggers**: Look for crash logs with "watchdog timeout"
2. **Soft timeout dumps**: Debug info dumped when soft timeout exceeded
3. **Startup timeouts**: Model loading or warmup taking too long

Enable soft watchdog to capture debug info before hard crashes:

```yaml
generator:
  health_checks:
    watchdog:
      timeout: 600
      soft_timeout: 300  # Capture debug info first
```

### Quick Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `health_checks.watchdog.timeout` | float | null (300s) | Hard watchdog timeout - crash on exceed |
| `health_checks.watchdog.soft_timeout` | float | null | Soft watchdog - dump debug info |
| `health_checks.dist_timeout` | int | null | torch.distributed init timeout |
| `health_checks.endpoint.timeout` | int | null (20s) | Health check request timeout |
| `health_checks.endpoint.enable_generation` | bool | null (true) | Enable test generation in health checks |
| `health_checks.startup.weights_ready_timeout` | int | null (120s) | Model loading timeout |
| `health_checks.startup.warmup_timeout` | float | null (-1) | Warmup timeout (-1 = disabled) |

---

## 36. Hierarchical Cache (GPU↔CPU↔NVMe) (SGLang Only)

SkyRL exposes SGLang's hierarchical caching system for multi-tier KV cache management, enabling serving of longer context lengths than GPU memory alone allows.

### Overview

The hierarchical cache system provides three storage tiers:
1. **Tier 1 (GPU)**: Fast GPU HBM for active KV cache
2. **Tier 2 (CPU)**: Host RAM for overflow and prefetching
3. **Tier 3 (Storage)**: NVMe/SSD for very long contexts

### Memory Hierarchy

```
┌─────────────────────────────────────────────────────┐
│                    Tier 1: GPU HBM                   │
│            (Fastest, limited capacity)               │
│                   ~40-80GB per GPU                   │
└─────────────────────────────────────────────────────┘
                          ↕ PCIe/NVLink
┌─────────────────────────────────────────────────────┐
│                   Tier 2: CPU RAM                    │
│          (Fast, larger capacity, pinned)             │
│                   ~256-512GB+                        │
└─────────────────────────────────────────────────────┘
                          ↕ NVMe/SSD
┌─────────────────────────────────────────────────────┐
│                Tier 3: NVMe Storage                  │
│           (Slower, very large capacity)              │
│                    ~1-8TB+                           │
└─────────────────────────────────────────────────────┘
```

### Configuration

```yaml
generator:
  backend: sglang

  hierarchical_cache:
    enabled: true

    # Host memory (Tier 2) configuration
    host_memory:
      ratio: 2.0           # CPU cache = 2x GPU cache size
      # OR explicit size:
      # size_gb: 128       # 128GB CPU cache

    # Write policy
    write_policy: "write_through"  # or "write_back"

    # I/O backend
    io_backend: "kernel"   # or "direct" for lower latency

    # Memory layout
    mem_layout: "layer_first"

    # Storage backend (Tier 3)
    storage:
      backend: "file"      # Local NVMe/SSD
      prefetch_policy: "best_effort"

    # Cache eviction
    eviction_policy: "lru"

    # KV cache dtype (can reduce memory 2-4x)
    kv_cache_dtype: "fp8_e4m3"
```

### Host Memory Configuration

Configure Tier 2 (CPU RAM) cache size:

```yaml
generator:
  hierarchical_cache:
    enabled: true
    host_memory:
      # Option 1: Ratio-based (relative to GPU cache)
      ratio: 3.0  # CPU cache = 3x GPU cache

      # Option 2: Explicit size in GB
      size_gb: 256  # 256GB CPU cache (overrides ratio)
```

**Sizing Guidelines:**
- `ratio: 2.0` - Standard setup, 2x GPU capacity in CPU RAM
- `ratio: 4.0` - Long context workloads
- `size_gb: 512` - Maximum utilization for high-memory nodes

### Write Policies

| Policy | Description | Use Case |
|--------|-------------|----------|
| `write_through` | Write to all tiers immediately | Default, safe, consistent |
| `write_back` | Write on eviction only | Higher throughput, risk of data loss |
| `write_through_selective` | Smart write-through | Balance of safety and performance |

```yaml
generator:
  hierarchical_cache:
    enabled: true
    write_policy: "write_back"  # Higher throughput
```

### Storage Backends (Tier 3)

Enable Tier 3 storage for very long contexts:

```yaml
generator:
  hierarchical_cache:
    enabled: true
    storage:
      # Local file system (NVMe/SSD)
      backend: "file"

      # OR distributed storage:
      # backend: "mooncake"   # Mooncake distributed storage
      # backend: "nixl"       # NIXL storage
      # backend: "hf3fs"      # HuggingFace 3FS

      prefetch_policy: "best_effort"
```

### KV Cache Compression

Reduce memory usage with quantized KV cache:

```yaml
generator:
  hierarchical_cache:
    # FP8 KV cache - 2x memory reduction
    kv_cache_dtype: "fp8_e4m3"  # or "fp8_e5m2"

    # FP4 (experimental) - 4x reduction
    # kv_cache_dtype: "fp4_e2m1"
```

**Dtype Options:**
| Dtype | Memory | Quality | GPU Requirement |
|-------|--------|---------|-----------------|
| `auto` | Baseline | Best | Any |
| `float16` | Baseline | Best | Any |
| `bfloat16` | Baseline | Best | Ampere+ |
| `fp8_e4m3` | 50% | Good | Hopper+ |
| `fp8_e5m2` | 50% | Good | Hopper+ |
| `fp4_e2m1` | 25% | Fair | Hopper+ |

### CPU Weight Offloading

For models larger than GPU memory, offload weights to CPU:

```yaml
generator:
  cpu_offload:
    # Reserve CPU memory for weights
    size_gb: 32

    # Enable CPU backup copy
    enabled: true

    # For speculative decoding
    draft_weights_enabled: true

    # Layer grouping
    group:
      size: 4           # 4 layers per group
      num_offload: 2    # Offload 2 layers per group
      prefetch_step: 2  # Prefetch 2 steps ahead
```

### Use Cases

#### Long Context Serving (128K+ tokens)

```yaml
generator:
  hierarchical_cache:
    enabled: true
    host_memory:
      ratio: 4.0  # Large CPU cache
    write_policy: "write_through"
    kv_cache_dtype: "fp8_e4m3"  # Compression
    eviction_policy: "lru"
```

#### Very Long Context with NVMe (1M+ tokens)

```yaml
generator:
  hierarchical_cache:
    enabled: true
    host_memory:
      size_gb: 512
    storage:
      backend: "file"
      prefetch_policy: "best_effort"
    io_backend: "direct"  # Bypass page cache
    kv_cache_dtype: "fp8_e4m3"
```

#### Large Model on Limited GPU

```yaml
generator:
  # 70B model on single GPU
  cpu_offload:
    size_gb: 64
    enabled: true
    group:
      size: 8
      num_offload: 4
      prefetch_step: 2

  hierarchical_cache:
    enabled: true
    host_memory:
      ratio: 2.0
```

### Performance Considerations

| Configuration | Context Length | Throughput | Latency |
|--------------|----------------|------------|---------|
| GPU only | Up to ~32K | Highest | Lowest |
| GPU + CPU | Up to ~256K | High | Low |
| GPU + CPU + NVMe | 1M+ | Medium | Medium |
| + FP8 KV cache | 2x above | Same | Same |

### Quick Reference - Hierarchical Cache

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `hierarchical_cache.enabled` | bool | false | Enable hierarchical cache |
| `hierarchical_cache.host_memory.ratio` | float | null (2.0) | CPU/GPU cache size ratio |
| `hierarchical_cache.host_memory.size_gb` | int | null | Explicit CPU cache size in GB |
| `hierarchical_cache.write_policy` | str | null | Write policy (write_through/write_back) |
| `hierarchical_cache.io_backend` | str | null | I/O backend (kernel/direct) |
| `hierarchical_cache.mem_layout` | str | null | Memory layout strategy |
| `hierarchical_cache.storage.backend` | str | null | Tier 3 storage backend |
| `hierarchical_cache.storage.prefetch_policy` | str | null | Storage prefetch policy |
| `hierarchical_cache.storage.extra_config` | str | null | Backend-specific JSON config |
| `hierarchical_cache.eviction_policy` | str | null (lru) | Cache eviction policy |
| `hierarchical_cache.kv_cache_dtype` | str | null (auto) | KV cache data type |
| `hierarchical_cache.page_size` | int | null (1) | Page size for allocation |

### Quick Reference - CPU Offload

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `cpu_offload.size_gb` | int | null (0) | CPU memory for weight offloading |
| `cpu_offload.enabled` | bool | false | Enable CPU weight backup |
| `cpu_offload.draft_weights_enabled` | bool | false | CPU backup for draft weights |
| `cpu_offload.mode` | str | null (cpu) | Offload mode |
| `cpu_offload.group.size` | int | null (-1) | Layers per offload group |
| `cpu_offload.group.num_offload` | int | null (1) | Layers to offload per group |
| `cpu_offload.group.prefetch_step` | int | null (1) | Prefetch steps ahead |

---

## References

- [SGLang Engine Source](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/entrypoints/engine.py)
- [SkyRL SGLang Engine Wrapper](../skyrl_train/inference_engines/sglang/sglang_engine.py)
- [Weight Sync Implementation](../skyrl_train/weight_sync/)
- [SGLang Issue #9039 (skip_tokenizer_init)](https://github.com/sgl-project/sglang/issues/9039)
- [EAGLE Paper](https://arxiv.org/abs/2401.15077) - Speculative Sampling for Fast LLM Inference
- [SGLang Speculative Decoding](https://github.com/sgl-project/sglang/tree/main/python/sglang/srt/speculative)
- [FP8 Training and Inference](https://developer.nvidia.com/blog/accelerating-ai-training-with-tf32-tensor-cores/) - NVIDIA FP8 Overview
- [SGLang Session Controller](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/session_controller.py) - Session Management Implementation
- [SGLang Quantization Methods](https://github.com/sgl-project/sglang/tree/main/python/sglang/srt/layers/quantization) - Supported Quantization Implementations
- [SGLang Custom Logit Processors](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/sampling/custom_logit_processor.py) - Custom Sampling Control
- [SGLang CUDA Graph Runner](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/model_executor/cuda_graph_runner.py) - CUDA Graph Implementation
- [SGLang Piecewise CUDA Graph](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py) - Piecewise CUDA Graph Implementation
- [SGLang Attention Registry](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/attention_registry.py) - Attention Backend Registration
- [SGLang Attention Backends](https://github.com/sgl-project/sglang/tree/main/python/sglang/srt/layers/attention) - Attention Backend Implementations
- [SGLang LoRA Manager](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/lora/lora_manager.py) - S-LoRA/Punica Implementation
- [SGLang LoRA Backends](https://github.com/sgl-project/sglang/tree/main/python/sglang/srt/lora/backend) - LoRA Kernel Implementations
- [SGLang Scheduler](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/scheduler.py) - Request Scheduling and Preemption Implementation
- [SGLang Schedule Policy](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/schedule_policy.py) - Scheduling Policy Implementations
- [SGLang Disaggregation](https://github.com/sgl-project/sglang/tree/main/python/sglang/srt/disaggregation) - Disaggregated Prefill/Decode Implementation
- [SGLang ServerArgs](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/server_args.py) - Server Arguments including Disaggregation Config
- [NCCL Documentation](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/) - NVIDIA Collective Communications Library
- [NCCL Environment Variables](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html) - NCCL Configuration Reference
- [SGLang Metrics Collector](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/metrics/collector.py) - Prometheus Metrics Implementation
- [Prometheus Python Client](https://github.com/prometheus/client_python) - Prometheus Client Library
- [OpenTelemetry Python](https://opentelemetry.io/docs/languages/python/) - OpenTelemetry Tracing Documentation
- [SGLang Data Parallel Controller](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/data_parallel_controller.py) - Load Balancing Implementation
- [SGLang Expert Parallelism](https://github.com/sgl-project/sglang/tree/main/python/sglang/srt/layers/moe) - MoE and EP Implementation
- [SGLang HTTP Server Health Endpoints](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/entrypoints/http_server.py) - Health Check Implementation
- [SGLang gRPC Health Servicer](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/grpc/health_servicer.py) - gRPC Health Check Service
- [SGLang Watchdog](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/utils/watchdog.py) - Watchdog Implementation
- [SGLang Hierarchical Cache Storage](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/hicache_storage.py) - Hierarchical Cache Implementation
- [SGLang Memory Pool](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/memory_pool.py) - GPU Memory Pool
- [SGLang Host Memory Pool](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/memory_pool_host.py) - CPU Memory Pool
