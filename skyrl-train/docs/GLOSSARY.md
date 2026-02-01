# Glossary

**Key terms and acronyms used in SkyRL and SGLang documentation.**

---

## RL Algorithms

### GRPO (Group Relative Policy Optimization)
A reinforcement learning algorithm that computes advantages relative to the group mean. For each prompt, multiple responses are generated, and advantages are calculated as `(reward - mean) / std` within each group. Does not require a critic model.

### RLOO (REINFORCE Leave-One-Out)
A baseline technique where the advantage for each sample is computed by leaving that sample out and using the mean of remaining samples as the baseline. More stable than vanilla REINFORCE.

### GAE (Generalized Advantage Estimation)
A technique for computing advantages using a learned value function (critic). Combines TD(0) and Monte Carlo returns using a lambda parameter. Requires a separate critic model.

### PPO (Proximal Policy Optimization)
A policy gradient algorithm that uses clipped surrogate objectives to limit policy updates. Often combined with GAE for advantage estimation.

### DAPO (Decoupled Alignment and Policy Optimization)
Separates token-level optimization from sequence-level rewards. Uses overlong reward to handle length bias.

---

## Parallelism

### FSDP (Fully Sharded Data Parallel)
PyTorch's distributed training strategy that shards model parameters, gradients, and optimizer states across GPUs. FSDP2 is the newer version with improved performance.

### TP (Tensor Parallelism)
Splits individual layers (attention, FFN) across multiple GPUs. Each GPU holds a slice of the weight matrices.

### PP (Pipeline Parallelism)
Splits the model into sequential stages, with each GPU handling different layers. Uses micro-batching to overlap computation.

### DP (Data Parallelism)
Replicates the entire model on each GPU and splits the batch across GPUs. Gradients are synchronized after each step.

### EP (Expert Parallelism)
For Mixture-of-Experts (MoE) models, distributes different experts across GPUs. Used with Megatron training strategy.

---

## SGLang Concepts

### RadixAttention
SGLang's tree-based prefix caching system. Efficiently stores and reuses KV cache for shared prefixes across requests. Particularly beneficial for multi-turn conversations where system prompts are repeated.

### Prefix Caching
Technique to reuse computed KV cache for repeated prompt prefixes. In SkyRL config: `enable_prefix_caching: true` (maps to SGLang's `disable_radix_cache: false`).

### FlashInfer
A CUDA library for efficient attention computation. Works on GPUs with compute capability 8.0+ (A100, RTX 3090, RTX 4090, H100). Alternative to FlashAttention.

### FlashAttention (FA3)
NVIDIA's optimized attention implementation. FA3 requires SM>=80 and SM<=90 (A100, H100). Use FlashInfer for other GPUs.

### Token-in-Token-out Mode
SGLang mode where the engine receives and returns tokens (not text). SkyRL handles tokenization/detokenization. Enabled via `skip_tokenizer_init=True`.

---

## Weight Synchronization

### Weight Sync
The process of transferring updated model weights from the training process to inference engines after each training step. Critical for online RL training.

### CUDA IPC (Inter-Process Communication)
Zero-copy weight transfer between processes on the same GPU. Fastest method, enabled with `colocate_all: true`.

### NCCL Broadcast
Network-based weight transfer using NVIDIA's collective communication library. Used for multi-node setups or non-colocated processes.

### Colocate
Running training and inference on the same GPUs. Enables CUDA IPC for fast weight sync but requires careful memory management (sleep/wake).

---

## Memory Management

### Sleep/Wake (Memory Saver)
SGLang's memory release mechanism for sharing GPUs between training and inference:
- **Sleep**: Releases GPU memory (weights, KV cache) for training
- **Wake**: Reallocates memory for inference (requires weight re-sync)

SGLang API names:
- `release_memory_occupation()` = sleep
- `resume_memory_occupation()` = wake

### gpu_memory_utilization
Fraction of GPU memory allocated for SGLang's KV cache. Default: 0.8 (80%). Maps to SGLang's `mem_fraction_static`.

---

## Training Configuration

### train_batch_size
Number of unique prompts loaded per training step. Actual samples = `train_batch_size * n_samples_per_prompt`.

### policy_mini_batch_size
Number of samples per optimizer step. Determines how many gradient accumulation steps occur.

### micro_train_batch_size_per_gpu
Samples processed per GPU per forward pass. Lower values use less memory but require more gradient accumulation.

### n_samples_per_prompt
Number of responses generated per prompt. Used by GRPO/RLOO for computing group-relative advantages.

### Rollout
The process of generating responses (trajectories) from the current policy. In RL, rollouts are scored by the reward function and used to compute policy gradients.

---

## Advantage Estimation

### Advantage
A measure of how much better an action is compared to the expected value. Positive advantage = better than average, negative = worse.

### Baseline
A reference value subtracted from rewards to reduce variance in policy gradient estimates. GRPO uses group mean; GAE uses a learned critic.

### KL Divergence (KL Loss)
Measures how much the policy has diverged from a reference model. Used to prevent reward hacking and maintain generation quality.

---

## Inference Engine

### Inference Engine
An SGLang or vLLM server that generates responses. SkyRL can run multiple engines in parallel (`num_inference_engines`).

### Inference Backend
The framework used for generation (SGLang or vLLM). Set via `generator.backend`.

### Ray Actor
A distributed process managed by Ray. SGLang inference engines run as Ray actors for parallel generation.

---

## Configuration Mapping

| SkyRL Config | SGLang Parameter | Description |
|--------------|------------------|-------------|
| `gpu_memory_utilization` | `mem_fraction_static` | GPU memory fraction for KV cache |
| `enable_prefix_caching` | `disable_radix_cache` (inverse) | Enable RadixAttention |
| `engine_init_kwargs.attention_backend` | `attention_backend` | flashinfer or fa3 |

---

## References

- [Algorithms Guide](./ALGORITHMS.md) - Detailed algorithm explanations
- [Training Strategies](./TRAINING_STRATEGIES.md) - FSDP2 vs Megatron
- [Batch Sizes Guide](./BATCH_SIZES.md) - Batch size relationships
- [SGLang Integration Guide](./SGLANG_INTEGRATION_GUIDE.md) - Full configuration reference
