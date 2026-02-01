SGLang Backend
==============

This guide explains how to use SGLang as the inference backend for SkyRL training. SGLang provides high-performance inference with RadixAttention for efficient prefix caching, making it an excellent choice for RL training workloads.

.. contents:: Table of Contents
   :local:
   :depth: 3

Overview
--------

SkyRL supports two inference backends:

- **vLLM** (default): Full-featured backend with mature ecosystem
- **SGLang**: High-throughput backend with RadixAttention and native parallelism support

Choose SGLang when you need:

- Efficient prefix caching for multi-turn conversations (RadixAttention)
- High-throughput inference for large batch sizes
- Native tensor/pipeline/data/expert parallelism support
- Memory-efficient inference with sleep/wake cycles
- FlashAttention 3 (fa3) backend for modern GPUs

Both backends now have feature parity for most use cases.

Architecture Overview
---------------------

SkyRL + SGLang Integration
~~~~~~~~~~~~~~~~~~~~~~~~~~

The integration follows a layered architecture:

.. code-block:: text

   ┌─────────────────────────────────────────────────────────────┐
   │                     RayPPOTrainer                           │
   │  - Orchestrates training loop                               │
   │  - Manages weight sync between training and inference       │
   │  - Coordinates sleep/wake cycles for memory management      │
   └─────────────────────────────────────────────────────────────┘
                               │
                               ▼
   ┌─────────────────────────────────────────────────────────────┐
   │                  InferenceEngineClient                      │
   │  - Routes requests to inference engines                     │
   │  - Handles session-based sticky routing                     │
   │  - Manages HTTP endpoint (if enabled)                       │
   │  - Supports pause/resume for in-flight weight updates       │
   └─────────────────────────────────────────────────────────────┘
                               │
                               ▼
   ┌─────────────────────────────────────────────────────────────┐
   │               RayWrappedInferenceEngine                     │
   │  - Ray actor wrapper for distributed execution              │
   │  - Handles placement group scheduling                       │
   └─────────────────────────────────────────────────────────────┘
                               │
                               ▼
   ┌─────────────────────────────────────────────────────────────┐
   │                 SGLangInferenceEngine                       │
   │  - Token-in-token-out generation (skip_tokenizer_init=True) │
   │  - External tokenizer for text↔token conversion             │
   │  - Custom weight loader for CUDA IPC transfers              │
   │  - LoRA adapter load/unload at runtime                      │
   │  - Stop sequence → stop_token_ids conversion                │
   └─────────────────────────────────────────────────────────────┘
                               │
                               ▼
   ┌─────────────────────────────────────────────────────────────┐
   │                     sglang.Engine()                         │
   │  - Native SGLang inference engine                           │
   │  - RadixAttention for prefix caching                        │
   │  - FlashAttention 3 backend (default)                       │
   └─────────────────────────────────────────────────────────────┘

Token-in-Token-Out Mode
~~~~~~~~~~~~~~~~~~~~~~~

SGLang runs with ``skip_tokenizer_init=True`` for efficient RL training:

- **Input**: Token IDs (not text strings)
- **Output**: Token IDs with optional logprobs
- **External Tokenizer**: Used for:
  - Decoding output tokens to text
  - Converting stop strings to stop_token_ids
  - HTTP endpoint text↔token conversion
  - Chat template application

This mode enables:

- Zero tokenization overhead in the inference engine
- Direct token-level logprob extraction for policy gradients
- Efficient multi-turn conversation handling

Installation
------------

SGLang and vLLM have conflicting dependencies, so you must install them separately:

.. code-block:: bash

   # For SGLang backend
   uv sync --extra sglang

   # For vLLM backend (default)
   uv sync --extra vllm

.. warning::

   Do not install both ``--extra sglang`` and ``--extra vllm`` in the same environment. They have conflicting dependencies.

Quick Start
-----------

To switch from vLLM to SGLang, change the ``backend`` parameter:

.. code-block:: yaml

   generator:
     backend: "sglang"  # Instead of "vllm"

Minimal Configuration
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   uv run --extra sglang -m skyrl_train.entrypoints.main_base \
     generator.backend=sglang \
     generator.run_engines_locally=true \
     # ... other parameters

Full Example (GSM8K with GRPO)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

See ``examples/gsm8k/gsm8k-grpo-sglang-skypilot.yaml`` for a complete SkyPilot configuration:

.. code-block:: bash

   # Run locally with environment variable
   INFERENCE_BACKEND=sglang bash examples/gsm8k/run_gsm8k.sh

   # Or via SkyPilot
   sky launch examples/gsm8k/gsm8k-grpo-sglang-skypilot.yaml

Configuration Reference
-----------------------

Backend Selection
~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   generator:
     backend: "sglang"           # Required: select SGLang backend
     run_engines_locally: true   # Recommended for SGLang

Parallelism Settings
~~~~~~~~~~~~~~~~~~~~

SGLang natively supports all parallelism types:

.. list-table::
   :header-rows: 1
   :widths: 40 15 15 30

   * - Parameter
     - vLLM
     - SGLang
     - Notes
   * - ``inference_engine_tensor_parallel_size``
     - 1-8+
     - 1-8+
     - Native support via ``tp_size``
   * - ``inference_engine_pipeline_parallel_size``
     - 1-4+
     - 1-4+
     - Native support via ``pp_size``
   * - ``inference_engine_data_parallel_size``
     - 1-8+
     - 1-8+
     - Native support via ``dp_size``
   * - ``inference_engine_expert_parallel_size``
     - 1-8+
     - 1-8+
     - For MoE models via ``ep_size``
   * - ``num_inference_engines``
     - 1-N
     - 1-N
     - Additional scaling mechanism

Example with parallelism:

.. code-block:: yaml

   generator:
     backend: sglang
     num_inference_engines: 2
     inference_engine_tensor_parallel_size: 4
     inference_engine_pipeline_parallel_size: 2
     inference_engine_data_parallel_size: 1
     inference_engine_expert_parallel_size: 4  # For MoE models

Memory Settings
~~~~~~~~~~~~~~~

.. code-block:: yaml

   generator:
     gpu_memory_utilization: 0.8  # Maps to SGLang's mem_fraction_static
     max_num_batched_tokens: 8192  # Maps to max_prefill_tokens
     max_num_seqs: 1024            # Maps to max_running_requests

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Model Size
     - Recommended ``gpu_memory_utilization``
   * - < 2B
     - 0.85 - 0.9
   * - 2B - 7B
     - 0.8 - 0.85
   * - 7B - 13B
     - 0.75 - 0.8
   * - > 13B
     - 0.7 - 0.75

Sampling Parameters
~~~~~~~~~~~~~~~~~~~

SGLang uses different parameter names than vLLM. SkyRL handles the translation automatically:

.. list-table::
   :header-rows: 1
   :widths: 35 35 30

   * - Config Parameter
     - vLLM Internal
     - SGLang Internal
   * - ``max_generate_length``
     - ``max_tokens``
     - ``max_new_tokens``
   * - ``logprobs: 0``
     - ``logprobs=0``
     - ``return_logprob=True``
   * - ``logprobs: N``
     - ``logprobs=N``
     - ``return_logprob=True, top_logprobs_num=N``
   * - ``stop``
     - Supported
     - Converted to ``stop_token_ids``

.. code-block:: yaml

   generator:
     sampling_params:
       max_generate_length: 1024
       temperature: 1.0
       top_p: 1.0
       top_k: -1
       min_p: 0.0
       logprobs: 0      # Enable logprobs collection
       stop: ["</s>", "\n\n"]  # Supported! Converted to stop_token_ids

Stop Sequence Support
^^^^^^^^^^^^^^^^^^^^^

Stop sequences are automatically converted to ``stop_token_ids`` using the external tokenizer:

.. code-block:: python

   # Internal conversion (sglang_engine.py:402-418)
   for stop_str in stop_strings:
       token_ids = tokenizer.encode(stop_str, add_special_tokens=False)
       stop_token_ids.append(token_ids[-1] if len(token_ids) > 1 else token_ids[0])
   sampling_params["stop_token_ids"] = list(set(stop_token_ids))

LoRA Support
~~~~~~~~~~~~

SGLang supports runtime LoRA adapter loading and unloading:

.. code-block:: yaml

   generator:
     backend: sglang
     enable_lora: true
     engine_init_kwargs:
       max_lora_rank: 64
       max_loras_per_batch: 8
       lora_backend: "csgmv"  # NVIDIA CSGMV backend

Runtime LoRA APIs:

.. code-block:: python

   # Load adapter at runtime
   await engine.load_lora_adapter(
       lora_name="my_adapter",
       lora_path="/path/to/adapter",
       pinned=False  # If True, won't be evicted
   )

   # Unload adapter
   await engine.unload_lora_adapter(lora_name="my_adapter")

HTTP Endpoints
~~~~~~~~~~~~~~

SGLang supports OpenAI-compatible HTTP endpoints via external tokenizer:

.. code-block:: yaml

   generator:
     backend: sglang
     enable_http_endpoint: true
     http_endpoint_host: "0.0.0.0"
     http_endpoint_port: 8000
     async_engine: true  # Required for HTTP endpoint

Supported endpoints:

- ``POST /v1/chat/completions`` - Chat completion API
- ``POST /v1/completions`` - Text completion API
- ``GET /health`` - Health check

Session-based sticky routing ensures same session routes to same engine for prefix caching benefits.

Weight Synchronization
~~~~~~~~~~~~~~~~~~~~~~

Two strategies available:

**CUDA IPC (Recommended for colocated)**:

.. code-block:: yaml

   generator:
     weight_sync_backend: "nccl"
   trainer:
     placement:
       colocate_all: true

- Zero-copy GPU memory sharing
- Best performance for same-node setups
- Supports TP > 1 via internal rank coordination

**Broadcast (For distributed)**:

.. code-block:: yaml

   generator:
     weight_sync_backend: "broadcast"
   trainer:
     placement:
       colocate_all: false

- Uses torch.distributed broadcast
- Works across machines
- Requires process group initialization

Engine Init Kwargs
~~~~~~~~~~~~~~~~~~

Pass SGLang-specific parameters via ``engine_init_kwargs``:

.. code-block:: yaml

   generator:
     engine_init_kwargs:
       # Attention Backend
       attention_backend: "fa3"           # FlashAttention 3 (default)
       mm_attention_backend: "fa3"        # For multi-modal models

       # LoRA Configuration
       max_lora_rank: 64
       max_loras_per_batch: 8
       lora_backend: "csgmv"

       # MoE Configuration
       moe_a2a_backend: "deepep"
       moe_runner_backend: "auto"
       enable_eplb: true                  # Expert load balancing

Feature Comparison
------------------

.. list-table::
   :header-rows: 1
   :widths: 40 15 15 30

   * - Feature
     - vLLM
     - SGLang
     - Notes
   * - Token-in-token-out
     - Yes
     - Yes
     - Both support efficient token-level I/O
   * - Prefix Caching
     - Yes
     - Yes (RadixAttention)
     - SGLang's radix tree is highly efficient
   * - Tensor Parallelism (TP)
     - Yes
     - **Yes**
     - Native ``tp_size`` parameter
   * - Pipeline Parallelism (PP)
     - Yes
     - **Yes**
     - Native ``pp_size`` parameter
   * - Data Parallelism (DP)
     - Yes
     - **Yes**
     - Native ``dp_size`` parameter
   * - Expert Parallelism (EP)
     - Yes
     - **Yes**
     - Native ``ep_size`` for MoE
   * - LoRA
     - Yes
     - **Yes**
     - Runtime load/unload APIs
   * - Stop Sequences
     - Yes
     - **Yes**
     - Converted to stop_token_ids
   * - OpenAI HTTP Endpoints
     - Yes
     - **Yes**
     - Via external tokenizer
   * - Megatron Training
     - Yes
     - **Yes (experimental)**
     - With warning message
   * - Memory Saver (sleep/wake)
     - Yes (levels 1-2)
     - Yes (level 2 only)
     - SGLang always discards weights on sleep
   * - Multi-turn Training
     - Yes
     - Yes (required)
     - ``use_conversation_multi_turn=true``

Remaining Limitation
~~~~~~~~~~~~~~~~~~~~

**min_new_tokens**: Not supported with SGLang because it requires ``skip_tokenizer_init=False``.

Memory Management
-----------------

Sleep/Wake Cycle
~~~~~~~~~~~~~~~~

When ``colocate_all=true``, training and inference share GPUs. The memory management cycle:

1. **Before Generation**: ``wake_up(tags=["kv_cache"])`` - Load KV cache memory
2. **After Generation**: ``sleep()`` - Release all inference memory
3. **Before Weight Sync**: ``wake_up(tags=["weights"])`` - Prepare for weight transfer
4. **After Weight Sync**: Continue to generation

SGLang-specific behavior:

- Sleep always releases all memory (no sleep levels like vLLM)
- Weight sync **always required** after wake_up (weights are discarded)
- Automatically aborts in-flight requests before sleeping

.. code-block:: python

   # Sleep API (sglang_engine.py:706-757)
   async def sleep(
       tags: Optional[List[str]] = None,     # ["weights"], ["kv_cache"], or None
       timeout: float = 60.0,
       abort_first: bool = True,             # Abort in-flight requests
       drain_timeout: float = 5.0            # Wait for requests to drain
   )

   # Wake API (sglang_engine.py:678-704)
   async def wake_up(
       tags: Optional[List[str]] = None,     # Memory tags to resume
       timeout: float = 60.0
   )

Hybrid Engine Mode
~~~~~~~~~~~~~~~~~~

When training and inference share GPUs:

.. code-block:: yaml

   trainer:
     placement:
       colocate_all: true
       policy_num_gpus_per_node: 4
       policy_num_nodes: 2

   generator:
     num_inference_engines: 8  # Must equal total policy GPUs
     gpu_memory_utilization: 0.8

Weight Sync Architecture
------------------------

Weight Transfer Flow
~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   Training Worker (FSDP/Megatron)
           │
           │ extract_weights() → WeightChunk iterator
           ▼
   ┌─────────────────────────────┐
   │   WeightTransferSender      │
   │   - CUDA IPC: Pack tensors, │
   │     create IPC handles      │
   │   - Broadcast: Send via     │
   │     torch.distributed       │
   └─────────────────────────────┘
           │
           │ WeightUpdateRequest
           ▼
   ┌─────────────────────────────┐
   │   SGLangWeightLoader        │
   │   - IPC: Deserialize,       │
   │     reconstruct tensors     │
   │   - Broadcast: Receive via  │
   │     distributed API         │
   └─────────────────────────────┘
           │
           │ model.load_weights()
           ▼
   ┌─────────────────────────────┐
   │   SGLang Engine             │
   │   - flush_cache=True        │
   │     (invalidate KV cache)   │
   └─────────────────────────────┘

CUDA IPC Strategy
~~~~~~~~~~~~~~~~~

Used when ``weight_sync_backend="nccl"`` and ``colocate_all=true``:

.. code-block:: python

   # Custom weight loader (sglang_engine.py:133-167)
   def sglang_custom_weight_loader(model, named_tensors):
       # Deserialize IPC request from tensor
       request = CudaIpcWeightUpdateRequest.deserialize(tensor.cpu().numpy().tobytes())

       # Create receiver and reconstruct tensors
       receiver = CudaIpcWeightTransferReceiver(model_dtype)
       weights_to_load = list(receiver.receive_weights(request))

       # Load into model
       model.load_weights(weights_to_load)

Broadcast Strategy
~~~~~~~~~~~~~~~~~~

Used for distributed setups:

.. code-block:: python

   # Initialize process group (sglang_engine.py:191-233)
   obj = InitWeightsUpdateGroupReqInput(
       master_address=init_info.master_addr,
       master_port=init_info.master_port,
       rank_offset=init_info.rank_offset,
       world_size=init_info.world_size,
       group_name=init_info.group_name,
       backend=init_info.backend,  # "nccl" or "gloo"
   )
   await engine.tokenizer_manager.init_weights_update_group(obj, None)

   # Update weights (sglang_engine.py:271-301)
   for name, dtype, shape in zip(request.names, request.dtypes, request.shapes):
       obj = UpdateWeightsFromDistributedReqInput(name=name, dtype=dtype, shape=shape)
       await engine.tokenizer_manager.update_weights_from_distributed(obj, None)

Generation Flow
---------------

Multi-turn Agent Loop
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   SkyRLGymGenerator.agent_loop()
           │
           ├─► Environment.init(chat_history) → initial prompt
           │
           └─► WHILE not done:
               │
               ├─► InferenceEngineClient.generate()
               │   └─► SGLangInferenceEngine.generate()
               │       └─► engine.async_generate(input_ids, sampling_params)
               │           └─► Returns: output_ids, finish_reason, logprobs
               │
               ├─► Environment.step(response)
               │   └─► Returns: reward, observation, done
               │
               └─► Update AgentLoopState:
                   - accumulate input_ids
                   - accumulate loss_mask
                   - accumulate rollout_logprobs
                   - update chat_history

Logprobs Extraction
~~~~~~~~~~~~~~~~~~~

SGLang returns logprobs in ``meta_info``:

.. code-block:: python

   # sglang_engine.py:443-459
   output_token_logprobs = output["meta_info"].get("output_token_logprobs")
   if isinstance(output_token_logprobs[0], (list, tuple)):
       # Format: [(logprob, token_id, decoded_token), ...]
       logprobs = [lp[0] for lp in output_token_logprobs]
   else:
       logprobs = output_token_logprobs

Loss Mask Calculation
~~~~~~~~~~~~~~~~~~~~~

Per-turn loss mask distinguishes generated tokens from observations:

.. code-block:: python

   # TurnOutput.get_turn_loss_mask() (skyrl_gym_generator.py:71-85)
   loss_mask = [1] * len(output_ids) + [0] * len(obs_ids)
   # If EOS was manually added, mask it: loss_mask[-len(obs_ids)-1] = 0

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

**"use_conversation_multi_turn=False is not supported"**

SGLang requires multi-turn mode. This is the default, but ensure you haven't disabled it:

.. code-block:: yaml

   generator:
     use_conversation_multi_turn: true  # Required, this is the default

**"tokenizer is required for SGLangInferenceEngine"**

SGLang requires an explicit tokenizer for token-in-token-out mode. This is handled automatically by SkyRL.

**"min_new_tokens not supported"**

This is the only remaining limitation. Use vLLM if you need ``min_new_tokens``.

**OOM during generation**

- Reduce ``gpu_memory_utilization`` (try 0.75)
- Reduce ``max_num_batched_tokens``
- Reduce ``max_num_seqs``

**Hanging on wake_up()**

- Increase timeout: ``timeout: 120.0`` in advanced configs
- Check GPU memory availability
- Ensure proper placement group configuration

Performance Tips
----------------

1. **Enable prefix caching** for multi-turn or shared prompts:

   .. code-block:: yaml

      generator:
        enable_prefix_caching: true  # Enabled by default

2. **Use CUDA IPC** for weight sync when colocated:

   .. code-block:: yaml

      generator:
        weight_sync_backend: nccl
      trainer:
        placement:
          colocate_all: true

3. **Tune attention backend** for your hardware:

   .. code-block:: yaml

      generator:
        engine_init_kwargs:
          attention_backend: "fa3"  # FlashAttention 3 (default, best for modern GPUs)
          # attention_backend: "flashinfer"  # Alternative

4. **Use session_id** for HTTP endpoints to maximize prefix cache hits:

   .. code-block:: json

      {
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "messages": [...],
        "session_id": "rollout-123"
      }

5. **Scale with parallelism** for large models:

   .. code-block:: yaml

      generator:
        backend: sglang
        inference_engine_tensor_parallel_size: 4
        inference_engine_pipeline_parallel_size: 2

Example Configurations
----------------------

Full-Featured Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

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

MoE Model with Expert Parallelism
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   generator:
     backend: sglang
     inference_engine_tensor_parallel_size: 4
     inference_engine_expert_parallel_size: 4
     engine_init_kwargs:
       moe_a2a_backend: "deepep"
       moe_runner_backend: "auto"
       enable_eplb: true

Megatron Training (Experimental)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   trainer:
     strategy: megatron

   generator:
     backend: sglang
     weight_sync_backend: nccl

.. note::

   SGLang with Megatron training is experimental. If you encounter issues, try switching to vLLM backend.

API Reference
-------------

SGLangInferenceEngine Methods
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Parallelism getters
   def tp_size(self) -> int
   def pp_size(self) -> int
   def dp_size(self) -> int
   def ep_size(self) -> int

   # Generation (token-in-token-out)
   async def generate(input_batch: InferenceEngineInput) -> InferenceEngineOutput

   # HTTP endpoints
   async def chat_completion(request_payload: Dict) -> Dict
   async def completion(request_payload: Dict) -> Dict

   # LoRA management
   async def load_lora_adapter(lora_name: str, lora_path: str, pinned: bool = False)
   async def unload_lora_adapter(lora_name: str)

   # Memory management
   async def sleep(tags=None, timeout=60.0, abort_first=True, drain_timeout=5.0)
   async def wake_up(tags=None, timeout=60.0)

   # Weight synchronization
   async def init_weight_update_communicator(init_info: WeightSyncInitInfo)
   async def update_named_weights(request: WeightUpdateRequest)

   # Cache management
   async def reset_prefix_cache()

   # Lifecycle
   async def teardown()

See Also
--------

- :doc:`training_backends` - Training backend options (FSDP2, Megatron)
- :doc:`lora` - LoRA fine-tuning
- :doc:`remote_server` - Remote inference server setup
- :doc:`../configuration/config` - Full configuration reference
- ``docs/SGLANG_LIMITATIONS.md`` - Feature support reference with code locations
