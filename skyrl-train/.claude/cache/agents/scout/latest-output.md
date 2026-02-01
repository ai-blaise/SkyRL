# Codebase Report: Reward Model Scoring Path Optimization Analysis

Generated: 2026-01-12

## Summary

The reward model scoring path in SkyRL uses SGLang's native `score()` API via `ScoreReqInput` for RLHF workflows. However, **the scoring path does NOT currently leverage SGLang's priority scheduling or advanced batching optimizations**. The implementation is straightforward request passthrough without explicit priority scheduling or batch-level optimizations for reward model requests.

## Findings

### Score API Implementation

**Location:** `/home/nourdine/sglang_skyrl/SkyRL/skyrl-train/skyrl_train/inference_engines/sglang/sglang_engine.py:2546-2630`

✓ VERIFIED: The engine provides a native `score()` method that wraps SGLang's reward model scoring:

```python
async def score(
    self,
    input_ids: List[List[int]],
    output_ids: List[List[int]],
    return_hidden_states: bool = False,
) -> Dict[str, Any]:
    """Compute reward scores using a reward model."""
    try:
        from sglang.srt.managers.io_struct import ScoreReqInput

        # Build score request
        obj = ScoreReqInput(
            input_ids=input_ids,
            output_ids=output_ids,
            return_hidden_states=return_hidden_states,
        )

        # Call SGLang's score API
        result = await self.engine.tokenizer_manager.score(obj, None)
        
        # Process and return results
        scores = []
        hidden_states = [] if return_hidden_states else None
        # ... result parsing ...
        return {
            "scores": scores,
            "hidden_states": hidden_states,
        }
```

### Priority Scheduling in Generate Path

✓ VERIFIED: Priority scheduling IS implemented for the `generate()` method at line 893-895:

```python
# Extract request-level priority for scheduling (SGLang priority scheduling feature)
# Lower values = higher priority. Use for reward model requests during RL training.
priority = sampling_params.pop("priority", None)
```

And used when creating the request at line 924:

```python
obj = GenerateReqInput(
    input_ids=token_ids_prompts,
    sampling_params=dict(sampling_params),
    # ... other params ...
    priority=priority,  # ← PRIORITY PASSED HERE
    lora_name=lora_name,
)
```

### Critical Gap: Score API Lacks Priority Support

✗ VERIFIED ABSENCE: The `score()` API does NOT extract or pass priority parameters:

1. **No priority parameter extraction** - Unlike `generate()`, score() has no mechanism to extract priority from input
2. **No priority field in ScoreReqInput** - The score request is built with only 3 fields:
   - `input_ids`
   - `output_ids`
   - `return_hidden_states`

3. **Direct tokenizer_manager call** - Score bypasses GenerateReqInput layer:
   ```python
   result = await self.engine.tokenizer_manager.score(obj, None)
   # vs
   generator = self.engine.tokenizer_manager.generate_request(obj, None)
   ```

### Batching Support

✓ VERIFIED: The score API accepts batched requests (lists of input/output pairs):
- `input_ids: List[List[int]]` - multiple prompts can be scored
- `output_ids: List[List[int]]` - multiple responses can be scored
- Results are returned as lists maintaining batch correspondence

However:
- ? INFERRED: No explicit batch size constraints documented
- ? INFERRED: Batching behavior determined by SGLang backend, not explicitly configured in SkyRL

### Ray Wrapper Layer

✓ VERIFIED: Ray wrapper at `/home/nourdine/sglang_skyrl/SkyRL/skyrl-train/skyrl_train/inference_engines/ray_wrapped_inference_engine.py:317-328` provides distributed access:

```python
async def score(
    self,
    input_ids: List[List[int]],
    output_ids: List[List[int]],
    return_hidden_states: bool = False,
) -> Dict[str, Any]:
    """Compute reward scores using a reward model."""
    return await self.inference_engine_actor.score.remote(
        input_ids=input_ids,
        output_ids=output_ids,
        return_hidden_states=return_hidden_states,
    )
```

This is a simple passthrough with no additional priority or batching optimizations.

### Priority Scheduling Configuration

✓ VERIFIED: Priority scheduling IS configurable for the entire engine at lines 1325-1354 in ray_wrapped_inference_engine.py:

```python
enable_priority = scheduling.get("enable_priority", False)
if enable_priority:
    # ... enable priority scheduling ...
    scheduling_kwargs["enable_priority_scheduling"] = True

abort_on_priority = scheduling.get("abort_on_priority_when_disabled", False)
if abort_on_priority:
    scheduling_kwargs["abort_on_priority_when_disabled"] = True

low_priority_first = scheduling.get("low_priority_values_first", False)
if low_priority_first:
    scheduling_kwargs["schedule_low_priority_values_first"] = True
```

However: These settings apply to the full engine initialization, NOT specifically to score operations.

## Architecture Map

```
User Request
    ↓
RayWrappedInferenceEngine.score()
    ↓
SGLangEngine.score()
    ↓
ScoreReqInput (NO priority, NO batch constraints)
    ↓
self.engine.tokenizer_manager.score()
    ↓
SGLang Backend (native scoring, handles batching)
    ↓
Reward Model Forward Pass
```

vs Generate Path:

```
User Request (with optional priority)
    ↓
RayWrappedInferenceEngine.generate()
    ↓
SGLangEngine.generate()
    ↓
GenerateReqInput (WITH priority) ← OPTIMIZATION APPLIED
    ↓
self.engine.tokenizer_manager.generate_request()
    ↓
SGLang Backend (with priority scheduling)
    ↓
LLM Forward Pass
```

## Key Observations

| Aspect | Status | Details |
|--------|--------|---------|
| Score API Exists | ✓ YES | Native SGLang wrapper at line 2546 |
| Batching Support | ✓ YES | Accepts List[List[int]], results batched |
| Priority Scheduling | ✗ NO | Priority parameter NOT passed to score requests |
| Batch Size Config | ? UNCLEAR | No explicit batch size constraints for scoring |
| Hidden States | ✓ YES | Extraction supported via return_hidden_states param |
| Ray Distribution | ✓ YES | Remote execution via Ray actor |

## Optimization Recommendations

Based on the analysis, here are missing optimizations for the reward model scoring path:

1. **Add Priority Support to Score API** (MISSING)
   - Extract `priority` from input parameters
   - Pass to ScoreReqInput if supported by SGLang version
   - Enable high-priority scoring for critical reward evaluations

2. **Batch Size Configuration** (UNCONFIGURED)
   - Add explicit `max_score_batch_size` configuration
   - Prevent reward model OOM on large batch scoring

3. **Score-Specific Scheduling** (NOT IMPLEMENTED)
   - Separate scheduling config for reward vs generation
   - Allow different priority preemption thresholds for scoring

4. **Parallel Generation + Scoring** (POTENTIAL)
   - Pipeline: generate responses → score in parallel
   - Currently no explicit coordination layer

## File References

| File | Purpose | Key Lines |
|------|---------|-----------|
| `/home/nourdine/sglang_skyrl/SkyRL/skyrl-train/skyrl_train/inference_engines/sglang/sglang_engine.py` | Main SGLang engine | 2546-2630 (score API), 893-895 (priority in generate) |
| `/home/nourdine/sglang_skyrl/SkyRL/skyrl-train/skyrl_train/inference_engines/ray_wrapped_inference_engine.py` | Ray wrapper | 317-328 (score wrapper), 1325-1354 (scheduling config) |
| `/home/nourdine/sglang_skyrl/SkyRL/skyrl-train/skyrl_train/inference_engines/base.py` | Base interface | 289-316 (score method signature) |

## Conclusion

**The reward model scoring path uses SGLang's native score API but does NOT leverage priority scheduling or explicit batching optimizations.** While the generate path has priority scheduling fully integrated (line 893-895), the score API remains a simple passthrough without priority support. This creates a potential bottleneck where reward model requests may be delayed during high-volume generation phases.

The scoring is functional and supports batching at the SGLang backend level, but lacks the proactive scheduling optimizations available in the generation path.

