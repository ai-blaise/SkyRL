# SGLang LoRA Features Audit: What's Exposed vs What's Missing in SkyRL

**Generated:** 2025-01-12  
**Project:** SkyRL / SGLang Integration  
**Focus:** LoRA adapter management and per-request selection capabilities

---

## Executive Summary

SkyRL exposes **basic LoRA adapter management** through SGLang's engine layer but lacks **per-request adapter selection** and several advanced S-LoRA features. The infrastructure exists in SGLang but is not wired into SkyRL's sampling parameters or request-level API.

Key Gap: **SkyRL supports loading/unloading adapters asynchronously but doesn't expose the per-request `lora_name` parameter that SGLang supports in GenerateReqInput.**

---

## Features Present (VERIFIED)

### 1. Async Adapter Loading/Unloading
**Status:** ✓ IMPLEMENTED  
**Location:** `/skyrl_train/inference_engines/sglang/sglang_engine.py:491-521`

```python
async def load_lora_adapter(self, lora_name: str, lora_path: str, pinned: bool = False):
    """Load a LoRA adapter at runtime.
    - lora_name: Unique identifier for the adapter
    - lora_path: Path to adapter files
    - pinned: If True, adapter won't be evicted from memory
    """
    req = LoadLoRAAdapterReqInput(lora_name=lora_name, lora_path=lora_path, pinned=pinned)
    result = await self.engine.tokenizer_manager.load_lora_adapter(req, None)

async def unload_lora_adapter(self, lora_name: str):
    """Unload a LoRA adapter at runtime."""
    req = UnloadLoRAAdapterReqInput(lora_name=lora_name)
    result = await self.engine.tokenizer_manager.unload_lora_adapter(req, None)
```

### 2. Adapter Pinning (Static)
**Status:** ✓ PARTIALLY EXPOSED  
**Location:** `sglang_engine.py:497` + `ppo_base_config.yaml:749`

- **Config support:** Pinning available in `load_lora_adapter(pinned=True)`
- **What it does:** Prevents memory eviction of specific adapters
- **Limitation:** Only used during `load_lora_adapter()` calls, NOT per-request

Config example (line 749):
```yaml
lora:
  paths:
    - lora_name: "qa"
      lora_path: "path/to/qa"
      pinned: true  # ← Prevents eviction
```

### 3. LoRA Kernel Backends
**Status:** ✓ EXPOSED  
**Location:** `ppo_base_config.yaml:783-793`

Supported backends:
- `csgmv` - Chunked SGMV/Punica kernels (default, fastest for multi-adapter)
- `triton` - Full Triton (supports embed_tokens/lm_head)
- `ascend` - Ascend NPU
- `torch_native` - Pure PyTorch (slowest)

Configuration:
```yaml
lora:
  backend: "csgmv"
  max_chunk_size: 16  # Power of 2 between 16-128
```

### 4. Adapter Pool Management
**Status:** ✓ EXPOSED  
**Location:** `ppo_base_config.yaml:768-781`

Configurable parameters:
- `max_loras_per_batch: 8` - Max adapters per batch (including base model)
- `max_loaded_loras: null` - Max adapters in CPU memory (null = unlimited)
- `eviction_policy: "lru"` - LRU or FIFO eviction when pool full

```yaml
lora:
  max_loras_per_batch: 8      # ← Batch-level pool size
  max_loaded_loras: null       # ← Memory-level pool
  eviction_policy: "lru"       # ← Auto-eviction strategy
```

### 5. Multi-Adapter Batching (Static)
**Status:** ✓ HARDWARE EXPOSED, NOT REQUEST-LEVEL  
**Location:** `ppo_base_config.yaml:771`

- Engine can batch multiple adapters in one forward pass
- S-LoRA with Punica SGMV kernels support efficient multi-adapter inference
- **But:** SkyRL doesn't expose per-request adapter selection, so all requests in a batch use the same adapter

### 6. Training-Time LoRA Configuration
**Status:** ✓ EXPOSED  
**Location:** `ppo_base_config.yaml:27-36`

```yaml
trainer:
  policy:
    model:
      lora:
        rank: 0           # LoRA rank (0 = disabled)
        alpha: 16         # Scaling factor
        dropout: 0        # LoRA dropout
        lora_sync_path: "/tmp/skyrl_lora_sync"
        target_modules: "all-linear"
        exclude_modules: null
        init_method: "kaiming"
```

---

## Features MISSING (NOT EXPOSED TO SKYRL)

### 1. Per-Request LoRA Selection (HIGH PRIORITY GAP)
**Status:** ✗ NOT IMPLEMENTED  
**SGLang Support:** YES - GenerateReqInput accepts `lora_name` parameter

**What's missing:**
- SkyRL doesn't expose `lora_name` in per-request sampling_params
- Cannot dynamically select which adapter to use per prompt/request
- All requests in a batch forced to use same adapter

**Where SGLang supports it:**
`sglang_engine.py:872-883`:
```python
obj = GenerateReqInput(
    input_ids=token_ids_prompts,
    sampling_params=dict(sampling_params),
    priority=priority,  # ← Priority IS exposed
    # lora_name=???     # ← THIS IS MISSING!
)
```

**Impact:** Cannot do multi-adapter inference where requests choose different adapters dynamically

**Example use case (NOT SUPPORTED):**
```python
# Desired but not possible:
sampling_params = [
    {"max_new_tokens": 100, "lora_name": "qa_adapter"},     # ← Not supported
    {"max_new_tokens": 100, "lora_name": "sql_adapter"},    # ← Not supported
]
```

### 2. Adapter Priority / Weighted Selection
**Status:** ✗ NOT IMPLEMENTED  
**SGLang Support:** Unclear (may be in SGMV kernel)

**What's missing:**
- No mechanism to prioritize which adapter gets GPU slots when pool is full
- LRU/FIFO eviction applies uniformly to all adapters
- No "preferred adapter" hints at request level

**Would enable:** Keeping high-demand adapters in memory longer

### 3. Hot-Swap API at Request Time
**Status:** ✗ NOT IMPLEMENTED  
**SGLang Support:** YES - Native load/unload but async-only

**What's missing:**
- Can load/unload adapters globally, but not swapped atomically with requests
- No "load-if-missing-then-use" pattern for single requests
- Must load adapters ahead of time, not on-demand per-request

**Current implementation:** Fully synchronous, requires explicit async calls
```python
await engine.load_lora_adapter(name, path)  # Global, blocks
response = await engine.generate(...)       # Then use it
```

**What's needed:**
```python
# Atomic load-then-generate (not possible now)
response = await engine.generate(..., auto_load_lora="adapter_name")
```

### 4. Per-Request Adapter Priority/Preemption
**Status:** ✗ NOT IMPLEMENTED  
**Related to:** Request priority scheduling (which IS supported - line 859)

**What's missing:**
- Request priority exists: `sampling_params["priority"] = 0`
- But NO adapter-level priority pairing
- Can't say "this high-priority request needs this adapter"

**Would enable:** 
- Ensuring reward model requests get fast adapters
- Preempting low-priority adapter computations for high-priority requests

### 5. Multi-Adapter Batching with Per-Request Selection
**Status:** ✗ NOT IMPLEMENTED  
**SGLang Capability:** S-LoRA with Punica supports true multi-adapter batching

**What's missing:**
- Hardware can batch requests with different adapters
- SkyRL forces all batch members to use same adapter (or base model)
- No way to specify different adapter for each request

**Would enable:**
```
Batch: [req1(adapter_qa), req2(adapter_sql), req3(base_model)]
      ↓
      Single forward pass with 3 different adapters!
```

### 6. Adapter Capacity Forecasting / Preloading Strategy
**Status:** ✗ NOT IMPLEMENTED  
**What it would do:**
- Predictively load frequently-used adapters before requests arrive
- Monitor eviction patterns and retain hot adapters
- Estimate memory needs for multi-adapter batch

### 7. LoRA Adapter Metrics & Observability
**Status:** ✗ NOT IMPLEMENTED  
**What's missing:**
- No per-adapter hit rates / usage statistics
- No eviction event logging
- No memory utilization per adapter
- Cannot monitor which adapters are active/pinned

---

## Configuration Exposure Analysis

### Fully Exposed (Training + Inference)
| Feature | Config Location | Supported |
|---------|-----------------|-----------|
| LoRA rank for training | `trainer.policy.model.lora.rank` | ✓ |
| Target modules | `trainer.policy.model.lora.target_modules` | ✓ |
| Adapter loading/unloading | `SGLangInferenceEngine.load_lora_adapter()` | ✓ |
| Adapter pinning | `load_lora_adapter(pinned=True)` | ✓ |
| Max adapters per batch | `generator.lora.max_loras_per_batch` | ✓ |
| Eviction policy | `generator.lora.eviction_policy` | ✓ |
| Kernel backend | `generator.lora.backend` | ✓ |
| Max chunk size | `generator.lora.max_chunk_size` | ✓ |

### Partially Exposed
| Feature | Where Exposed | What's Missing |
|---------|---------------|-----------------|
| Request priority | `sampling_params["priority"]` | No adapter pairing |
| Multi-adapter batching | Hardware capable | Per-request selection |

### Not Exposed
| Feature | Reason | Impact |
|---------|--------|--------|
| Per-request `lora_name` | Not passed to GenerateReqInput | Can't select adapter per request |
| Hot-swap on demand | No on-demand loading API | Must preload all adapters |
| Adapter priority/hints | Not in SGLang API | Uniform eviction for all adapters |
| Eviction statistics | No telemetry | Can't monitor adapter behavior |

---

## Code Locations Summary

### Where LoRA is Wired In

**Training-side configuration:**
- `ppo_base_config.yaml:27-36` - Policy model LoRA config
- `ppo_base_config.yaml:70-76` - Critic model LoRA config

**Inference initialization:**
- `main_base.py:64-68` - Enable LoRA flag + rank
- `main_base.py:239-261` - generator.lora config parsing

**Inference engine:**
- `sglang_engine.py:437-470` - Init with enable_lora flag
- `sglang_engine.py:491-521` - load/unload_lora_adapter methods
- `sglang_engine.py:502-503` - LoadLoRAAdapterReqInput creation

**Config schema:**
- `ppo_base_config.yaml:740-793` - Full LoRA hot-swapping section (docs are comprehensive)

### Where Per-Request Adapter Selection Would Go

**If implemented:**
- `sglang_engine.py:872-883` - GenerateReqInput construction (would add lora_name here)
- `base.py:16` - InferenceEngineInput TypedDict (would extend sampling_params docs)
- `sglang_engine.py:523-603` - _preprocess_prompts (would validate lora_name)

---

## Recommendations for Missing Features

### Priority 1: Per-Request LoRA Selection
**Complexity:** Medium | **Impact:** High  
**Implementation path:**
1. Add `lora_name` to sampling_params in InferenceEngineInput documentation
2. Extract `lora_name` in `_preprocess_prompts()` like priority handling (line 859)
3. Pass to GenerateReqInput (line 872)
4. Auto-load if pinning enabled + adapter missing

### Priority 2: Adapter Hot-Swap with Auto-Load
**Complexity:** Medium | **Impact:** Medium  
**Implementation path:**
1. Extend `load_lora_adapter()` with retry logic
2. Wrap generate() to auto-load on first-use
3. Cache loaded adapters to avoid re-loading

### Priority 3: Multi-Adapter Batch Awareness
**Complexity:** High | **Impact:** Medium  
**Implementation path:**
1. Track which adapter each request uses
2. If batch size > 1 and mixed adapters, split or warn
3. Use S-LoRA multi-adapter batching if all fit in max_loras_per_batch

### Priority 4: Adapter Telemetry
**Complexity:** Low | **Impact:** Low  
**Implementation path:**
1. Hook SGLang's adapter event callbacks
2. Log load/unload/evict events
3. Export metrics via Prometheus (already supported in config)

---

## SGLang Native APIs Being Used

**Verified in use:**
```python
from sglang.srt.managers.io_struct import (
    LoadLoRAAdapterReqInput,  # ✓ Used for loading
    UnloadLoRAAdapterReqInput,  # ✓ Used for unloading
    GenerateReqInput,  # ✓ Used for generation, but lora_name not passed
)
```

**Available but not used:**
- GenerateReqInput `lora_name` parameter (line 503 shows request object creation)
- S-LoRA multi-adapter batching kernels (backend="csgmv" enabled, but no per-request adapter)

---

## Testing Evidence

**Test file:** `/tests/gpu/gpu_ci/test_lora.py`
- Demonstrates basic LoRA loading for training (enable_lora=True)
- No tests for per-request adapter selection
- No tests for multi-adapter scenarios

---

## Summary Table

| Feature | Exposed | Can Use | SGLang Has | Effort |
|---------|---------|---------|-----------|--------|
| Adapter loading | ✓ | ✓ | ✓ | N/A |
| Adapter pinning | ✓ | ✓ | ✓ | N/A |
| Eviction policy | ✓ | ✓ | ✓ | N/A |
| Kernel backends | ✓ | ✓ | ✓ | N/A |
| **Per-request selection** | **✗** | **✗** | **✓** | **Med** |
| **Adapter priority** | **✗** | **✗** | **~** | **High** |
| **Hot-swap on-demand** | **✗** | **✗** | **✓** | **Med** |
| **Multi-adapter batching** | **✗** | **✗** | **✓** | **High** |
| **Adapter metrics** | **✗** | **✗** | **~** | **Low** |

---

## Key Insight

**The gap is at the request level, not the engine level.**

SGLang's GenerateReqInput fully supports per-request LoRA selection via the `lora_name` parameter. SkyRL initializes the infrastructure (load/unload methods, pinning support, eviction policies) but doesn't expose the request-level API that would allow:

1. Each request in a batch to specify its own adapter
2. Atomic load-then-use patterns
3. True multi-adapter batching with S-LoRA

This is a **straightforward integration gap** - not a missing SGLang feature, but missing SkyRL middleware to wire it through.

