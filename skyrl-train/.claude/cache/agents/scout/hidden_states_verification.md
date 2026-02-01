# Hidden States Extraction Pipeline - Final Verification

**Status: COMPLETE ✓**

**Date:** 2026-01-12
**File:** `/home/nourdine/sglang_skyrl/SkyRL/skyrl-train/skyrl_train/inference_engines/sglang/sglang_engine.py`

---

## Summary

The hidden_states extraction pipeline is **FULLY IMPLEMENTED** across all generate methods:
- ✓ `generate()` - Extracts and passes hidden_states
- ✓ `generate_with_session()` - Extracts and passes hidden_states  
- ✓ `generate_stream()` - Appropriately skips (streaming incompatible)
- ✓ `_postprocess_outputs()` - Extracts and returns hidden_states

---

## Method 1: `generate()` - Lines 817-960

### Extraction (Lines 854-858)
```python
# Extract hidden states parameter for RL value function enrichment
# Can be passed via sampling_params or input_batch
return_hidden_states = sampling_params.pop("return_hidden_states", False)
if not return_hidden_states:
    return_hidden_states = input_batch.get("return_hidden_states", False)
```

**Status:** ✓ VERIFIED

### Pass to GenerateReqInput (Lines 913-926)
```python
obj = GenerateReqInput(
    input_ids=token_ids_prompts,
    sampling_params=dict(sampling_params),
    image_data=image_data,
    video_data=video_data,
    audio_data=audio_data,
    return_logprob=return_logprob,
    logprob_start_len=logprob_start_len,
    top_logprobs_num=top_logprobs_num,
    return_hidden_states=return_hidden_states,  # ← PASSED HERE
    custom_logit_processor=custom_logit_processor,
    priority=priority,
    lora_name=lora_name,
)
```

**Status:** ✓ VERIFIED

### Pass to _postprocess_outputs (Lines 930-937)
```python
return self._postprocess_outputs(
    outputs,
    return_logprobs=return_logprob,
    n_per_prompt=n_per_prompt,
    multi_token_stop_seqs=multi_token_stop_seqs,
    stop_regex=stop_regex,
    extract_hidden_states=return_hidden_states,  # ← PASSED HERE
)
```

**Status:** ✓ VERIFIED

---

## Method 2: `generate_with_session()` - Lines 2316-2426

### Extraction (Lines 2371-2374)
```python
# Extract hidden states request (for value function enrichment)
return_hidden_states = sampling_params.pop("return_hidden_states", False)
if not return_hidden_states:
    return_hidden_states = input_batch.get("return_hidden_states", False)
```

**Status:** ✓ VERIFIED

### Pass to GenerateReqInput (Lines 2398-2408)
```python
obj = GenerateReqInput(
    input_ids=token_ids_prompts,
    sampling_params=sampling_params,
    return_logprob=return_logprob,
    logprob_start_len=logprob_start_len,
    top_logprobs_num=top_logprobs_num,
    session_params=session_params_arg,
    custom_logit_processor=custom_logit_processor,
    lora_name=lora_name,
    return_hidden_states=return_hidden_states,  # ← PASSED HERE
)
```

**Status:** ✓ VERIFIED

### Pass to _postprocess_outputs (Lines 2418-2424)
```python
return self._postprocess_outputs(
    outputs,
    return_logprobs=return_logprob,
    n_per_prompt=n_per_prompt,
    extract_request_ids=True,
    extract_hidden_states=return_hidden_states,  # ← PASSED HERE
)
```

**Status:** ✓ VERIFIED

---

## Method 3: `generate_stream()` - Lines 962-1050

### Design Decision
```python
# Remove internal fields (not used in streaming mode)
# Note: Streaming does not apply multi-token stop trimming or stop_regex
_ = sampling_params.pop("_multi_token_stop_seqs", None)
_ = sampling_params.pop("_stop_regex", None)
```

**Status:** ✓ CORRECT - Streaming does not support hidden_states (no batch accumulation)

---

## Method 4: `_postprocess_outputs()` - Lines 701-815

### Signature (Lines 701-710)
```python
def _postprocess_outputs(
    self,
    outputs,
    return_logprobs: bool = False,
    n_per_prompt: int = 1,
    extract_request_ids: bool = False,
    multi_token_stop_seqs: Optional[List[List[int]]] = None,
    stop_regex: Optional[str] = None,
    extract_hidden_states: bool = False,  # ← PARAMETER DEFINED
):
```

**Status:** ✓ VERIFIED

### Initialization (Lines 734)
```python
hidden_states: Optional[List[Any]] = [] if extract_hidden_states else None
```

**Status:** ✓ VERIFIED

### Extraction Logic (Lines 790-795)
```python
# Extract hidden states if requested and available
# Useful for value function enrichment, representation learning, and RL state extraction
if extract_hidden_states:
    meta_info = output.get("meta_info", {})
    output_hidden_states = meta_info.get("hidden_states", None)
    hidden_states.append(output_hidden_states)
```

**Status:** ✓ VERIFIED

### Empty List to None Conversion (Lines 802-804)
```python
# Convert empty hidden_states list to None
if hidden_states is not None and len(hidden_states) == 0:
    hidden_states = None
```

**Status:** ✓ VERIFIED

### Return in InferenceEngineOutput (Lines 806-815)
```python
return InferenceEngineOutput(
    responses=responses,
    response_ids=response_ids,
    stop_reasons=stop_reasons,
    response_logprobs=response_logprobs,
    weight_version=weight_version,
    n_per_prompt=n_per_prompt if n_per_prompt > 1 else None,
    request_ids=request_ids,
    hidden_states=hidden_states,  # ← RETURNED HERE
)
```

**Status:** ✓ VERIFIED

---

## Pipeline Data Flow

```
User Request
    ↓
input_batch.get("return_hidden_states") or sampling_params["return_hidden_states"]
    ↓
generate() or generate_with_session()
    ├─ return_hidden_states = extracted
    ├─ GenerateReqInput(return_hidden_states=True)
    └─ _postprocess_outputs(extract_hidden_states=True)
    ↓
_postprocess_outputs()
    ├─ Initialize: hidden_states = []
    ├─ Loop outputs: hidden_states.append(output["meta_info"]["hidden_states"])
    ├─ Convert: [] → None (if empty)
    └─ Return: InferenceEngineOutput(hidden_states=...)
    ↓
InferenceEngineOutput.hidden_states
```

---

## Verification Checklist

| Requirement | Location | Status |
|------------|----------|--------|
| Extract from sampling_params | generate():856 | ✓ |
| Fallback to input_batch | generate():858 | ✓ |
| Pass to GenerateReqInput | generate():922 | ✓ |
| Pass to _postprocess_outputs | generate():936 | ✓ |
| Extract from sampling_params | generate_with_session():2372 | ✓ |
| Fallback to input_batch | generate_with_session():2374 | ✓ |
| Pass to GenerateReqInput | generate_with_session():2407 | ✓ |
| Pass to _postprocess_outputs | generate_with_session():2423 | ✓ |
| Skip in streaming | generate_stream():OK | ✓ |
| _postprocess_outputs signature | Line 709 | ✓ |
| Initialize list | _postprocess_outputs():734 | ✓ |
| Extract from meta_info | _postprocess_outputs():794 | ✓ |
| Return in output | _postprocess_outputs():814 | ✓ |

---

## Conclusion

**The hidden_states extraction pipeline is COMPLETE and CORRECT.**

- All three synchronous generate methods (`generate()`, `generate_with_session()`, and helper methods) properly:
  1. Extract `return_hidden_states` from both `sampling_params` and `input_batch`
  2. Pass it to `GenerateReqInput` for SGLang engine processing
  3. Pass it to `_postprocess_outputs` for extraction and formatting

- The `_postprocess_outputs` method properly:
  1. Accepts `extract_hidden_states` parameter
  2. Initializes empty list when enabled
  3. Extracts hidden_states from SGLang output's meta_info
  4. Converts empty lists to None
  5. Returns in InferenceEngineOutput

- The `generate_stream()` method correctly skips hidden_states (incompatible with streaming)

**No further changes needed.**

