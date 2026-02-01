# Complete Configuration Reference

**Comprehensive reference for all SkyRL configuration options.**

This document covers configuration options that may not be covered in other guides. For common options, see [SGLANG_INTEGRATION_GUIDE.md](./SGLANG_INTEGRATION_GUIDE.md).

---

## Table of Contents

1. [Trainer Configuration](#1-trainer-configuration)
2. [Algorithm Configuration](#2-algorithm-configuration)
3. [Generator Configuration](#3-generator-configuration)
4. [Environment Configuration](#4-environment-configuration)
5. [Placement Configuration](#5-placement-configuration)
6. [Fully Async Training](#6-fully-async-training)
7. [LoRA Configuration](#7-lora-configuration)
8. [FSDP Configuration](#8-fsdp-configuration)
9. [Megatron Configuration](#9-megatron-configuration)
10. [Data Configuration](#10-data-configuration)

---

## 1. Trainer Configuration

### Core Training Options

| Config Path | Type | Default | Description |
|-------------|------|---------|-------------|
| `trainer.strategy` | str | `"fsdp2"` | Training strategy: `"fsdp"`, `"fsdp2"`, or `"megatron"` |
| `trainer.epochs` | int | `20` | Number of training epochs |
| `trainer.train_batch_size` | int | `1024` | Total training batch size |
| `trainer.policy_mini_batch_size` | int | `256` | Policy gradient mini-batch size |
| `trainer.critic_mini_batch_size` | int | `256` | Critic mini-batch size |
| `trainer.micro_train_batch_size_per_gpu` | int | `1` | Micro-batch per GPU for training |
| `trainer.micro_forward_batch_size_per_gpu` | int | `64` | Micro-batch per GPU for forward pass |
| `trainer.update_epochs_per_batch` | int | `1` | PPO update epochs per batch |
| `trainer.max_prompt_length` | int | `512` | Maximum prompt token length |

### Advanced Training Options

| Config Path | Type | Default | Description |
|-------------|------|---------|-------------|
| `trainer.sequence_parallel_backend` | str | `"ulysses"` | Sequence parallelism: `"ulysses"` or `"ring"` |
| `trainer.use_sample_packing` | bool | `true` | Pack multiple sequences per batch |
| `trainer.update_ref_every_epoch` | bool | `false` | Update reference model weights each epoch |
| `trainer.use_torch_compile` | bool | `false` | Enable torch.compile on policy logits |
| `trainer.record_memory` | bool | `false` | Save memory snapshots for debugging |
| `trainer.dump_data_batch` | bool | `false` | Dump training batches to disk |
| `trainer.dump_eval_results` | bool | `true` | Dump evaluation results to disk |
| `trainer.model_config_kwargs` | dict | `{}` | Pass-through kwargs to HuggingFace model config |

### RoPE Configuration

| Config Path | Type | Default | Description |
|-------------|------|---------|-------------|
| `trainer.rope_scaling` | dict | `null` | RoPE scaling config: `{rope_type: "yarn", factor: 2.0}` |
| `trainer.rope_theta` | float | `null` | RoPE theta parameter override |

### Checkpointing

| Config Path | Type | Default | Description |
|-------------|------|---------|-------------|
| `trainer.ckpt_path` | str | `"~/ckpts/"` | Checkpoint directory |
| `trainer.ckpt_interval` | int | `10` | Save checkpoint every N steps |
| `trainer.max_ckpts_to_keep` | int | `-1` | Max checkpoints to keep (-1 = all) |
| `trainer.hf_save_interval` | int | `-1` | HuggingFace export interval (-1 = disabled) |
| `trainer.export_path` | str | `"~/exports/"` | Export directory |
| `trainer.resume_mode` | str | `null` | Resume mode: `null`, `"latest"`, `"from_path"` |
| `trainer.resume_path` | str | `null` | Path for `resume_mode="from_path"` |

### Evaluation

| Config Path | Type | Default | Description |
|-------------|------|---------|-------------|
| `trainer.eval_batch_size` | int | `1024` | Evaluation batch size |
| `trainer.eval_before_train` | bool | `true` | Run evaluation before training starts |
| `trainer.eval_interval` | int | `5` | Evaluate every N steps |

### Logging

| Config Path | Type | Default | Description |
|-------------|------|---------|-------------|
| `trainer.logger` | str | `"wandb"` | Logger: `"wandb"`, `"tensorboard"`, `"mlflow"`, `"swanlab"`, `"console"` |
| `trainer.project_name` | str | `"skyrl"` | Project name for logging |
| `trainer.run_name` | str | `"test"` | Run name for logging |
| `trainer.logger_kwargs` | dict | `{}` | Additional logger kwargs (tags, notes, group) |

---

## 2. Algorithm Configuration

### Core Algorithm Options

| Config Path | Type | Default | Description |
|-------------|------|---------|-------------|
| `trainer.algorithm.advantage_estimator` | str | `"grpo"` | Estimator: `"grpo"`, `"gae"`, `"rloo"`, `"reinforce++"` |
| `trainer.algorithm.policy_loss_type` | str | `"regular"` | Loss type: `"regular"`, `"dual_clip"`, `"gspo"`, `"cispo"`, `"sapo"`, `"clip_cov"`, `"kl_cov"` |
| `trainer.algorithm.loss_reduction` | str | `"token_mean"` | Reduction: `"token_mean"`, `"sequence_mean"`, `"seq_mean_token_sum_norm"` |

### PPO/GRPO Options

| Config Path | Type | Default | Description |
|-------------|------|---------|-------------|
| `trainer.algorithm.eps_clip` | float | `0.2` | PPO clipping epsilon |
| `trainer.algorithm.eps_clip_low` | float | `0.2` | Lower clipping bound (asymmetric) |
| `trainer.algorithm.eps_clip_high` | float | `0.2` | Upper clipping bound (asymmetric) |
| `trainer.algorithm.grpo_norm_by_std` | bool | `true` | Normalize by std in GRPO |

### KL Penalty Options

| Config Path | Type | Default | Description |
|-------------|------|---------|-------------|
| `trainer.algorithm.use_kl_loss` | bool | `false` | Enable KL divergence penalty |
| `trainer.algorithm.kl_loss_coef` | float | `0.001` | KL loss coefficient |
| `trainer.algorithm.kl_estimator` | str | `"k2"` | KL estimator: `"k1"`, `"k2"`, `"k3"`, `"abs"` |

### Entropy Options

| Config Path | Type | Default | Description |
|-------------|------|---------|-------------|
| `trainer.algorithm.use_entropy_loss` | bool | `false` | Enable entropy bonus |
| `trainer.algorithm.entropy_loss_coef` | float | `0.01` | Entropy loss coefficient |

### SAPO Algorithm

| Config Path | Type | Default | Description |
|-------------|------|---------|-------------|
| `trainer.algorithm.sapo.tau_pos` | float | `1.0` | Positive threshold parameter |
| `trainer.algorithm.sapo.tau_neg` | float | `1.05` | Negative threshold parameter |

### CISPO Algorithm

| Config Path | Type | Default | Description |
|-------------|------|---------|-------------|
| `trainer.algorithm.cispo.cispo_eps_clip_low` | float | `0.0` | Lower bound for importance ratio clipping |
| `trainer.algorithm.cispo.cispo_eps_clip_high` | float | `5.0` | Upper bound for importance ratio clipping |

### Clip-Cov Algorithm

| Config Path | Type | Default | Description |
|-------------|------|---------|-------------|
| `trainer.algorithm.clip_cov.clip_ratio` | float | `0.0002` | Fraction of tokens to clip |
| `trainer.algorithm.clip_cov.clip_cov_lb` | float | `1.0` | Lower bound for covariance clipping |
| `trainer.algorithm.clip_cov.clip_cov_ub` | float | `5.0` | Upper bound for covariance clipping |

### KL-Cov Algorithm

| Config Path | Type | Default | Description |
|-------------|------|---------|-------------|
| `trainer.algorithm.kl_cov.kl_cov_frac` | float | `0.2` | Fraction of tokens for KL regularization |
| `trainer.algorithm.kl_cov.ppo_kl_coef` | float | `1.0` | KL penalty coefficient |

### Truncated Importance Sampling (TIS)

| Config Path | Type | Default | Description |
|-------------|------|---------|-------------|
| `trainer.algorithm.use_tis` | bool | `false` | Enable TIS for off-policy training |
| `trainer.algorithm.tis_imp_ratio_cap` | float | `-1.0` | Importance ratio cap (-1 = no cap) |

### Dynamic Sampling

| Config Path | Type | Default | Description |
|-------------|------|---------|-------------|
| `trainer.algorithm.dynamic_sampling.type` | str | `null` | Strategy: `null`, `"filter"`, `"replace"` |
| `trainer.algorithm.dynamic_sampling.max_sample_batches` | int | `30` | Max batches before stopping |
| `trainer.algorithm.dynamic_sampling.min_replace_ratio` | float | `0.3` | Min ratio for replace strategy |

### Other Algorithm Options

| Config Path | Type | Default | Description |
|-------------|------|---------|-------------|
| `trainer.algorithm.zero_variance_filter` | bool | `false` | Filter zero-variance reward prompts |
| `trainer.algorithm.value_head_prefix` | str | `"value_head"` | Critic value head naming prefix |

---

## 3. Generator Configuration

### Backend Selection

| Config Path | Type | Default | Description |
|-------------|------|---------|-------------|
| `generator.backend` | str | `"vllm"` | Inference backend: `"vllm"` or `"sglang"` |
| `generator.model_dtype` | str | `"bfloat16"` | Model dtype: `"bfloat16"`, `"float16"`, `"float32"` |
| `generator.run_engines_locally` | bool | `true` | Run engines in same Ray cluster |

### Parallelism

| Config Path | Type | Default | Description |
|-------------|------|---------|-------------|
| `generator.num_inference_engines` | int | `1` | Number of parallel inference engines |
| `generator.inference_engine_tensor_parallel_size` | int | `1` | TP size per engine |
| `generator.inference_engine_pipeline_parallel_size` | int | `1` | PP size per engine |
| `generator.inference_engine_data_parallel_size` | int | `1` | DP size per engine |
| `generator.inference_engine_expert_parallel_size` | int | `1` | EP size for MoE models |

### Memory Configuration

| Config Path | Type | Default | Description |
|-------------|------|---------|-------------|
| `generator.gpu_memory_utilization` | float | `0.8` | GPU memory fraction for KV cache |
| `generator.max_num_batched_tokens` | int | `8192` | Max tokens in prefill |
| `generator.max_num_seqs` | int | `1024` | Max concurrent sequences |
| `generator.enable_prefix_caching` | bool | `true` | Enable RadixAttention prefix caching |
| `generator.enable_chunked_prefill` | bool | `true` | Enable chunked prefill (vLLM) |

### Weight Synchronization

| Config Path | Type | Default | Description |
|-------------|------|---------|-------------|
| `generator.weight_sync_backend` | str | `"nccl"` | Sync backend: `"nccl"` or `"gloo"` |
| `generator.weight_transfer_threshold_cuda_ipc_GB` | float | `1.0` | Batch size for CUDA IPC transfers |
| `generator.override_existing_update_group` | str | `"auto"` | Override behavior: `"auto"`, `"enable"`, `"disable"` |

### Generation Options

| Config Path | Type | Default | Description |
|-------------|------|---------|-------------|
| `generator.n_samples_per_prompt` | int | `5` | Responses per prompt (training) |
| `generator.eval_n_samples_per_prompt` | int | `1` | Responses per prompt (evaluation) |
| `generator.async_engine` | bool | `true` | Use async generation |
| `generator.batched` | bool | `true` | Enable batched generation |

### Multi-Turn Options

| Config Path | Type | Default | Description |
|-------------|------|---------|-------------|
| `generator.use_conversation_multi_turn` | bool | `true` | Enable multi-turn format |
| `generator.max_turns` | int | `1` | Maximum conversation turns |
| `generator.step_wise_trajectories` | bool | `false` | Step-by-step generation for token rewards |
| `generator.max_input_length` | int | varies | Max input length for multi-turn |
| `generator.append_eos_token_after_stop_str_in_multi_turn` | bool | `true` | Append EOS after stop string |

### Reward Options

| Config Path | Type | Default | Description |
|-------------|------|---------|-------------|
| `generator.zero_reward_on_non_stop` | bool | `false` | Zero reward for truncated generation |
| `generator.apply_overlong_filtering` | bool | `false` | DAPO overlong filtering |

### Chat Template

| Config Path | Type | Default | Description |
|-------------|------|---------|-------------|
| `generator.chat_template.source` | str | `"name"` | Template source: `"name"` or `"file"` |
| `generator.chat_template.name_or_path` | str | `null` | Model name or Jinja2 template path |
| `generator.chat_template_kwargs` | dict | `{}` | Kwargs for `apply_chat_template()` |

### HTTP Endpoints

| Config Path | Type | Default | Description |
|-------------|------|---------|-------------|
| `generator.enable_http_endpoint` | bool | `false` | Enable OpenAI-compatible API |
| `generator.http_endpoint_host` | str | `"127.0.0.1"` | HTTP endpoint host |
| `generator.http_endpoint_port` | int | `8000` | HTTP endpoint port |

### LoRA

| Config Path | Type | Default | Description |
|-------------|------|---------|-------------|
| `generator.enable_lora` | bool | `false` | Enable LoRA adapter support |
| `generator.fully_sharded_loras` | bool | `false` | Fully shard LoRA adapters |

### Sampling Parameters

| Config Path | Type | Default | Description |
|-------------|------|---------|-------------|
| `generator.sampling_params.max_generate_length` | int | `1024` | Maximum tokens to generate |
| `generator.sampling_params.min_new_tokens` | int | `0` | Minimum tokens (suppresses EOS) |
| `generator.sampling_params.temperature` | float | `1.0` | Sampling temperature |
| `generator.sampling_params.top_p` | float | `1.0` | Nucleus sampling threshold |
| `generator.sampling_params.top_k` | int | `-1` | Top-k sampling (-1 = disabled) |
| `generator.sampling_params.min_p` | float | `0.0` | Minimum probability threshold |
| `generator.sampling_params.frequency_penalty` | float | `0.0` | Frequency penalty |
| `generator.sampling_params.presence_penalty` | float | `0.0` | Presence penalty |
| `generator.sampling_params.repetition_penalty` | float | `1.0` | Repetition penalty |
| `generator.sampling_params.stop` | list | `[]` | Stop strings |
| `generator.sampling_params.stop_token_ids` | list | `[]` | Stop token IDs |

### Engine Init Kwargs

| Config Path | Type | Default | Description |
|-------------|------|---------|-------------|
| `generator.engine_init_kwargs.attention_backend` | str | `"flashinfer"` | Attention: `"fa3"` or `"flashinfer"` |
| `generator.engine_init_kwargs.mm_attention_backend` | str | `"flashinfer"` | Multi-modal attention backend |
| `generator.engine_init_kwargs.enable_memory_saver` | bool | `true` | Enable sleep/wake functionality |
| `generator.engine_init_kwargs.max_lora_rank` | int | `64` | Maximum LoRA rank |
| `generator.engine_init_kwargs.max_loras_per_batch` | int | `8` | Max adapters per batch |
| `generator.engine_init_kwargs.lora_backend` | str | `"csgmv"` | LoRA backend |

---

## 4. Environment Configuration

### Core Environment Options

| Config Path | Type | Default | Description |
|-------------|------|---------|-------------|
| `environment.env_class` | str | `"gsm8k"` | Environment: `"gsm8k"`, `"aime"`, `"text2sql"`, `"search"`, `"lcb"`, `"searchcode"` |

### SkyRL-Gym Options

| Config Path | Type | Default | Description |
|-------------|------|---------|-------------|
| `environment.skyrl_gym.max_env_workers` | int | `32` | Background workers for env step calls |

### Text2SQL Options

| Config Path | Type | Default | Description |
|-------------|------|---------|-------------|
| `environment.skyrl_gym.text2sql.db_path` | str | `"/home/ray/default/sql_data"` | Database path |

### LLM-as-a-Judge Options

| Config Path | Type | Default | Description |
|-------------|------|---------|-------------|
| `environment.skyrl_gym.llm_as_a_judge.model` | str | `"gpt-4o-mini"` | Judge LLM model |
| `environment.skyrl_gym.llm_as_a_judge.base_url` | str | `null` | Custom API endpoint |

### Search Options

| Config Path | Type | Default | Description |
|-------------|------|---------|-------------|
| `environment.skyrl_gym.search.search_url` | str | `"http://127.0.0.1:8000/retrieve"` | Search backend URL |
| `environment.skyrl_gym.search.topk` | int | `3` | Number of search results |
| `environment.skyrl_gym.search.timeout` | int | `30` | Request timeout (seconds) |
| `environment.skyrl_gym.search.log_requests` | bool | `false` | Log search requests |

---

## 5. Placement Configuration

| Config Path | Type | Default | Description |
|-------------|------|---------|-------------|
| `trainer.placement.colocate_all` | bool | `true` | Colocate all models + inference |
| `trainer.placement.colocate_policy_ref` | bool | `true` | Colocate policy and reference |
| `trainer.placement.policy_num_nodes` | int | `1` | Nodes for policy model |
| `trainer.placement.policy_num_gpus_per_node` | int | `4` | GPUs per node for policy |
| `trainer.placement.critic_num_nodes` | int | `1` | Nodes for critic model |
| `trainer.placement.critic_num_gpus_per_node` | int | `4` | GPUs per node for critic |
| `trainer.placement.ref_num_nodes` | int | `1` | Nodes for reference model |
| `trainer.placement.ref_num_gpus_per_node` | int | `4` | GPUs per node for reference |

---

## 6. Fully Async Training

| Config Path | Type | Default | Description |
|-------------|------|---------|-------------|
| `trainer.fully_async.enabled` | bool | `false` | Enable fully async mode |
| `trainer.fully_async.max_staleness_steps` | int | `4` | Maximum weight staleness |
| `trainer.fully_async.num_parallel_generation_workers` | int | `768` | Parallel generation workers |

**Constraint:** `num_parallel_generation_workers` must be >= `policy_mini_batch_size` and <= `policy_mini_batch_size * (max_staleness_steps + 1)`

---

## 7. LoRA Configuration

### Policy LoRA

| Config Path | Type | Default | Description |
|-------------|------|---------|-------------|
| `trainer.policy.model.lora.rank` | int | `0` | LoRA rank (0 = disabled) |
| `trainer.policy.model.lora.alpha` | float | `16` | LoRA alpha |
| `trainer.policy.model.lora.dropout` | float | `0.0` | LoRA dropout |
| `trainer.policy.model.lora.target_modules` | list | `["q_proj", "v_proj"]` | Target modules |
| `trainer.policy.model.lora.exclude_modules` | list | `null` | Modules to exclude |
| `trainer.policy.model.lora.init_method` | str | `"kaiming"` | Init: `"kaiming"`, `"xavier"`, `"normal"`, `"zero"` |
| `trainer.policy.model.lora.lora_sync_path` | str | `"/tmp/skyrl_lora_sync"` | Sync path for distributed |

### Critic LoRA

Same options available under `trainer.critic.model.lora.*`

---

## 8. FSDP Configuration

| Config Path | Type | Default | Description |
|-------------|------|---------|-------------|
| `trainer.policy.fsdp_config.fsdp_size` | int | `-1` | FSDP world size (-1 = auto) |
| `trainer.policy.fsdp_config.reshard_after_forward` | bool/int | `true` | Reshard strategy |
| `trainer.policy.fsdp_config.sharding_strategy` | str | `"FULL_SHARD"` | Sharding: `"FULL_SHARD"`, `"SHARD_GRAD_OP"`, `"NO_SHARD"` |
| `trainer.policy.fsdp_config.mixed_precision` | str | `"bf16"` | Mixed precision: `"bf16"`, `"fp16"`, `"fp32"` |
| `trainer.policy.fsdp_config.cpu_offload` | bool | `false` | Offload to CPU |
| `trainer.policy.fsdp_config.backward_prefetch` | str | `"BACKWARD_PRE"` | Prefetch strategy |
| `trainer.policy.fsdp_config.limit_all_gathers` | bool | `true` | Limit concurrent all-gathers |

---

## 9. Megatron Configuration

| Config Path | Type | Default | Description |
|-------------|------|---------|-------------|
| `trainer.policy.megatron_config.tensor_model_parallel_size` | int | `1` | Tensor parallelism |
| `trainer.policy.megatron_config.pipeline_model_parallel_size` | int | `1` | Pipeline parallelism |
| `trainer.policy.megatron_config.context_parallel_size` | int | `1` | Context parallelism |
| `trainer.policy.megatron_config.sequence_parallel` | bool | `false` | Sequence parallelism |
| `trainer.policy.megatron_config.empty_cuda_cache` | bool | `false` | Clear cache between steps |
| `trainer.policy.megatron_config.transformer_config_kwargs` | dict | `{}` | Model config overrides |

---

## 10. Data Configuration

| Config Path | Type | Default | Description |
|-------------|------|---------|-------------|
| `data.train_data` | list | Required | Training data paths |
| `data.val_data` | list | Required | Validation data paths |
| `data.num_workers` | int | `4` | DataLoader workers |
| `data.prefetch_factor` | int | `2` | Prefetch batches |
| `data.pin_memory` | bool | `true` | Pin memory for GPU transfer |
| `data.drop_last` | bool | `true` | Drop incomplete batches |

---

## References

- [SGLANG_INTEGRATION_GUIDE.md](./SGLANG_INTEGRATION_GUIDE.md) - SGLang-specific configuration
- [ALGORITHMS.md](./ALGORITHMS.md) - Algorithm details
- [ENVIRONMENT_VARIABLES.md](./ENVIRONMENT_VARIABLES.md) - Environment variables
- [TRAINING_STRATEGIES.md](./TRAINING_STRATEGIES.md) - FSDP/Megatron strategies
