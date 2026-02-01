# SGLang + SkyRL Quickstart Guide

**Get started with RL training using SGLang as the inference backend in under 10 minutes.**

---

## Quick Environment Check

Before starting, verify your environment meets requirements:

```bash
# 1. Check Python version (MUST be 3.12.x)
python --version
# Should show: Python 3.12.x (if not, see "Install Python 3.12" below)

# 2. Check CUDA availability
nvidia-smi
# Should show NVIDIA driver and GPU info

# 3. Check GPU compute capability (for attention backend selection)
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}, SM: {torch.cuda.get_device_capability()[0]}.{torch.cuda.get_device_capability()[1]}')"
# SM 86-89 → use flashinfer
# SM 80-90 → can use fa3 or flashinfer

# 4. Check disk space (need ~10GB)
df -h ~
```

**STOP HERE** if Python is not 3.12.x - SkyRL requires exactly Python 3.12.

---

## What You'll Build

By the end of this guide, you'll have a working RL training pipeline that:
- Uses **SGLang** for fast trajectory generation (rollouts)
- Uses **SkyRL** for GRPO reinforcement learning
- Trains a model to solve math problems (GSM8K)

**Verified Results:** Qwen2.5-0.5B-Instruct improves from 24% → 45% accuracy on GSM8K.

---

## Prerequisites

- Linux with NVIDIA GPU (CUDA 12.8+)
- Python 3.12 (strict requirement)
- ~16GB GPU memory (for 0.5B model)
- `uv` package manager
- Hugging Face account (for model access)

### Install Python 3.12

If you don't have Python 3.12, install it using one of these methods:

```bash
# Option 1: Using pyenv (recommended)
curl https://pyenv.run | bash
pyenv install 3.12
pyenv local 3.12

# Option 2: Using conda
conda create -n skyrl python=3.12
conda activate skyrl

# Option 3: Using apt (Ubuntu 22.04+)
sudo apt update && sudo apt install python3.12 python3.12-venv

# Verify version
python --version  # Should show Python 3.12.x
```

### Install uv Package Manager

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc  # or restart terminal
```

### Hugging Face Authentication

The Qwen2.5-0.5B-Instruct model requires authentication:

```bash
# Option 1: Login interactively
pip install huggingface_hub
huggingface-cli login

# Option 2: Set token directly
export HF_TOKEN="hf_your_token_here"
```

Get your token at: https://huggingface.co/settings/tokens

---

## Step 1: Clone and Install

```bash
# Clone SkyRL
git clone https://github.com/NovaSky-AI/SkyRL
cd SkyRL/skyrl-train

# Install with SGLang backend
uv sync --extra sglang

# Activate environment
source .venv/bin/activate
```

**What this installs:**
- SkyRL training framework
- SGLang inference engine
- FlashInfer attention kernels
- Ray for distributed execution

**Verify installation:**
```bash
# Run pre-flight check
bash scripts/preflight_check.sh
```

This checks Python version, CUDA availability, and all required packages.

---

## Step 2: Prepare Data

```bash
# Download and prepare GSM8K dataset
python examples/gsm8k/gsm8k_dataset.py
```

This creates:
- `~/data/gsm8k/train.parquet` (~7.5K math problems)
- `~/data/gsm8k/validation.parquet` (~1.3K problems)

---

## Step 3: Run Training

```bash
# IMPORTANT: Unset Ray's UV hook to avoid editable install issues
# WHY: Ray workers try to replicate the package environment. If SGLang is installed
# as an editable package (pip install -e .), the paths don't exist in workers.
# This causes "Failed to generate package metadata for 'sglang @ editable+...'"
unset RAY_RUNTIME_ENV_HOOK

# Verify it's unset
echo $RAY_RUNTIME_ENV_HOOK  # Should print nothing

# Run GRPO training with SGLang backend
python -m skyrl_train.entrypoints.main_base \
  +experiment=grpo_qwen2.5-0.5b_math500 \
  generator.backend=sglang \
  +generator.engine_init_kwargs.attention_backend=flashinfer
```

### Expected First Run Timeline

| Phase | Duration | What You'll See |
|-------|----------|-----------------|
| Ray startup | ~2 min | "Started Ray workers..." |
| Model download | ~5 min | Progress bar (first run only) |
| Model loading | ~3 min | "Loading model weights..." |
| CUDA compilation | ~2 min | "Compiling CUDA kernels..." |
| First step | ~1 min | "Step 0: generating..." |
| **Total to first metrics** | **~15 min** | "eval/all/pass_at_1: 0.24" |

**Don't worry** if it seems stuck - the first run takes longer due to model download and CUDA kernel compilation. Subsequent runs start much faster (~3 min).

**What happens:**
1. SGLang loads Qwen2.5-0.5B-Instruct model
2. For each training step:
   - SGLang generates 2 responses per math problem
   - GSM8K environment scores responses (correct=1, wrong=0)
   - GRPO algorithm computes advantages within each group
   - Policy model is updated via gradient descent
   - New weights are synced to SGLang (~1.9s)
3. Evaluation runs every 5 steps

---

## Step 4: Monitor Training

Training outputs to console:
```
Step 60: {'eval/all/pass_at_1': '0.2441', ...}
Step 65: {'eval/all/pass_at_1': '0.2684', ...}
...
Step 140: {'eval/all/pass_at_1': '0.4466', ...}  # Peak!
```

Checkpoints saved to `~/ckpts/global_step_*/`

---

## Configuration Explained

Here's what the key settings do:

```yaml
generator:
  backend: sglang                    # Use SGLang (not vLLM)
  num_inference_engines: 1           # Number of parallel inference workers
  gpu_memory_utilization: 0.8        # GPU memory for SGLang KV cache
  enable_prefix_caching: true        # Reuse KV cache for repeated prefixes
  weight_sync_backend: nccl          # Fast GPU-to-GPU weight transfer

  sampling_params:
    temperature: 1.0                 # Exploration during training
    max_generate_length: 1024        # Max response tokens

  engine_init_kwargs:
    attention_backend: flashinfer    # FlashInfer (or "fa3" for A100/H100)

trainer:
  strategy: fsdp2                    # Fully Sharded Data Parallel
  placement:
    colocate_all: true               # Training + inference on same GPUs

  algorithm:
    advantage_estimator: grpo        # Group Relative Policy Optimization
    kl_loss_coef: 0.001              # Small KL penalty
```

---

## Common Issues & Solutions

### 1. FlashAttention v3 Error
```
AssertionError: FlashAttention v3 Backend requires SM>=80 and SM<=90
```
**Fix:** Use FlashInfer instead:
```bash
+generator.engine_init_kwargs.attention_backend=flashinfer
```

### 2. Ray Worker Crashes with Editable SGLang
```
error: Failed to generate package metadata for `sglang @ editable+...`
```
**Fix:** Unset the uv hook:
```bash
unset RAY_RUNTIME_ENV_HOOK
```

### 3. Out of Memory
**Fix:** Reduce batch size or GPU memory:
```bash
trainer.train_batch_size=2 \
generator.gpu_memory_utilization=0.7
```

### 4. Weight Sync Timeout
**Fix:** Increase timeout:
```bash
generator.weight_sync_timeout=120
```

---

## Next Steps

### Scale Up
```bash
# Use larger model
trainer.policy.model.path=Qwen/Qwen2.5-7B-Instruct \
generator.inference_engine_tensor_parallel_size=2

# More inference engines
generator.num_inference_engines=4
```

### Different Algorithms
```bash
# PPO with value function
trainer.algorithm.advantage_estimator=gae

# REINFORCE Leave-One-Out
trainer.algorithm.advantage_estimator=rloo
```

### Custom Environment
See `examples/` for:
- `text_to_sql/` - SQL generation
- `livecodebench/` - Code generation
- `mini_swe_agent/` - Software engineering

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     SkyRL Training Loop                      │
│                                                              │
│  1. GENERATE ─────────────────────────────────────────────► │
│     │                                                        │
│     │  SGLang Engine (Ray Actor)                            │
│     │  ├── Receives prompts (math problems)                 │
│     │  ├── Generates 2 responses per prompt                 │
│     │  └── Returns tokens + logprobs                        │
│     │                                                        │
│  2. REWARD ───────────────────────────────────────────────► │
│     │                                                        │
│     │  GSM8K Environment                                    │
│     │  ├── Parses model's final answer                      │
│     │  └── Returns reward (1=correct, 0=wrong)              │
│     │                                                        │
│  3. TRAIN ────────────────────────────────────────────────► │
│     │                                                        │
│     │  GRPO Algorithm                                       │
│     │  ├── Computes group-normalized advantages             │
│     │  ├── Policy gradient update                           │
│     │  └── KL divergence penalty                            │
│     │                                                        │
│  4. SYNC ─────────────────────────────────────────────────► │
│     │                                                        │
│     │  Weight Synchronization                               │
│     │  ├── Extract weights from FSDP model                  │
│     │  ├── Transfer via CUDA IPC (zero-copy)                │
│     │  └── Load into SGLang (~1.9s)                         │
│     │                                                        │
│  5. REPEAT ───────────────────────────────────────────────► │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Why SGLang?

| Feature | Benefit for RL |
|---------|----------------|
| **RadixAttention** | Prefix caching for multi-turn conversations |
| **FlashInfer** | Fast attention on various GPU architectures |
| **Weight Updates** | Native support for distributed weight sync |
| **Memory Saver** | Release GPU memory during training phase |

---

## Links

- [Full SGLang Integration Guide](./SGLANG_INTEGRATION_GUIDE.md) - Complete configuration reference
- [SGLang Documentation](https://docs.sglang.io/) - SGLang official docs
- [SkyRL GitHub](https://github.com/NovaSky-AI/SkyRL) - Main repository
- [Example Configs](../examples/) - More training examples
