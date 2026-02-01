# End-to-End Tutorial: RL Training with SGLang + SkyRL

**A complete walkthrough from installation to trained model.**

---

## What You'll Learn

This tutorial guides you through:
1. Setting up your environment
2. Preparing a dataset
3. Understanding the training configuration
4. Running RL training with SGLang
5. Evaluating and using your trained model

**Time:** ~30 minutes (excluding training time)

---

## Prerequisites

- Linux with NVIDIA GPU (16GB+ VRAM recommended)
- CUDA 12.8+
- Python 3.12
- Basic familiarity with command line

---

## Part 1: Installation

### Step 1.1: Install uv Package Manager

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc  # or restart your terminal
```

### Step 1.2: Clone SkyRL

```bash
git clone https://github.com/NovaSky-AI/SkyRL
cd SkyRL/skyrl-train
```

### Step 1.3: Create Virtual Environment with SGLang

```bash
uv sync --extra sglang
source .venv/bin/activate
```

This installs:
- SkyRL training framework
- SGLang inference engine
- FlashInfer attention kernels
- Ray distributed framework
- PyTorch 2.9.1+

### Step 1.4: Verify Installation

```bash
python -c "import sglang; print(f'SGLang version: {sglang.__version__}')"
python -c "import skyrl_train; print('SkyRL installed successfully')"
```

---

## Part 2: Dataset Preparation

We'll use the GSM8K dataset (grade school math problems) as our example.

### Step 2.1: Download and Prepare Dataset

```bash
python examples/gsm8k/gsm8k_dataset.py --output_dir ~/data/gsm8k
```

This creates:
- `~/data/gsm8k/train.parquet` (~7,500 problems)
- `~/data/gsm8k/validation.parquet` (~1,300 problems)

### Step 2.2: Understand the Data Format

Each row in the dataset contains:

```python
{
    "prompt": [
        {"role": "system", "content": "You are a helpful math tutor..."},
        {"role": "user", "content": "Janet has 3 apples. She buys 2 more..."}
    ],
    "env_class": "gsm8k",
    "reward_spec": {
        "method": "rule",
        "ground_truth": "5"
    },
    "extra_info": {...}
}
```

**Key fields:**
- `prompt`: OpenAI chat format messages
- `env_class`: Environment identifier for reward computation
- `reward_spec.ground_truth`: Expected answer for scoring

---

## Part 3: Understanding the Configuration

Let's examine a complete training configuration:

### Step 3.1: Create Configuration File

Create `my_training_config.yaml`:

```yaml
# my_training_config.yaml
defaults:
  - /ppo_base_config
  - _self_

# === DATA ===
data:
  train_data:
    - "${oc.env:HOME}/data/gsm8k/train.parquet"
  val_data:
    - "${oc.env:HOME}/data/gsm8k/validation.parquet"

# === TRAINER ===
trainer:
  # Training strategy
  strategy: fsdp2                    # Fully Sharded Data Parallel v2

  # GPU placement
  placement:
    colocate_all: true               # Training + inference on same GPUs (enables CUDA IPC)
    policy_num_gpus_per_node: 4
    critic_num_gpus_per_node: 4
    ref_num_gpus_per_node: 4

  # Policy model (the model we're training)
  policy:
    model:
      path: "Qwen/Qwen2.5-0.5B-Instruct"
    optimizer_config:
      lr: 1.0e-6                     # Learning rate
      max_grad_norm: 1.0             # Gradient clipping

  # Reference model (for KL penalty)
  ref:
    model:
      path: "Qwen/Qwen2.5-0.5B-Instruct"

  # RL Algorithm
  algorithm:
    advantage_estimator: grpo        # Group Relative Policy Optimization
    use_kl_loss: true
    kl_loss_coef: 0.001              # Small KL penalty

  # Training hyperparameters
  epochs: 10
  train_batch_size: 256
  policy_mini_batch_size: 64
  micro_train_batch_size_per_gpu: 8
  update_epochs_per_batch: 1

  # Evaluation
  eval_before_train: true
  eval_interval: 5                   # Evaluate every 5 steps
  eval_batch_size: 256

  # Checkpointing
  ckpt_interval: 10
  ckpt_dir: "${oc.env:HOME}/ckpts"

  # Logging
  logger: tensorboard               # or "wandb"
  project_name: "gsm8k-tutorial"
  run_name: "sglang-grpo"

  # Sequence lengths
  max_prompt_length: 512

# === GENERATOR (SGLang) ===
generator:
  backend: sglang                    # Use SGLang as inference backend

  # Engine configuration
  num_inference_engines: 4           # Number of SGLang engines
  run_engines_locally: true
  inference_engine_tensor_parallel_size: 1

  # Memory
  gpu_memory_utilization: 0.8        # 80% GPU memory for KV cache

  # Performance
  enable_prefix_caching: true        # RadixAttention for prefix reuse
  async_engine: true
  batched: true

  # Weight synchronization
  weight_sync_backend: nccl          # Uses CUDA IPC with colocate_all=true

  # Multi-turn (required for SGLang)
  use_conversation_multi_turn: true

  # Sampling during training
  sampling_params:
    max_generate_length: 1024
    temperature: 1.0                 # Exploration
    top_p: 1.0

  # Sampling during evaluation
  eval_sampling_params:
    max_generate_length: 1024
    temperature: 0.0                 # Greedy (deterministic)

  # Rollouts per prompt
  n_samples_per_prompt: 4            # Generate 4 responses per problem

  # SGLang-specific settings
  engine_init_kwargs:
    attention_backend: "flashinfer"  # Use flashinfer (works on most GPUs)

# === ENVIRONMENT ===
environment:
  env_class: gsm8k                   # GSM8K reward environment
```

### Step 3.2: Configuration Breakdown

**Key concepts:**

| Section | Purpose |
|---------|---------|
| `trainer.policy` | The model being trained |
| `trainer.ref` | Reference model for KL divergence |
| `trainer.algorithm` | RL algorithm (GRPO, GAE, RLOO) |
| `generator` | Inference engine for generating responses |
| `environment` | Reward computation |

**GRPO Algorithm:**
- No critic model needed (unlike PPO)
- Computes advantages relative to group mean
- `n_samples_per_prompt`: Number of responses generated per prompt
- Advantages = (reward - mean) / std within each group

---

## Part 4: Running Training

### Step 4.1: Set Up Environment

```bash
# Activate environment
source .venv/bin/activate

# IMPORTANT: Unset this if SGLang is installed as editable
unset RAY_RUNTIME_ENV_HOOK

# Optional: Set Hugging Face token for gated models
export HF_TOKEN="your-token-here"
```

### Step 4.2: Run Training

**Option A: Using the config file:**

```bash
python -m skyrl_train.entrypoints.main_base \
  --config-path . \
  --config-name my_training_config
```

**Option B: Using command-line overrides with existing experiment:**

```bash
python -m skyrl_train.entrypoints.main_base \
  +experiment=grpo_qwen2.5-0.5b_math500 \
  generator.backend=sglang \
  +generator.engine_init_kwargs.attention_backend=flashinfer \
  trainer.epochs=10 \
  trainer.logger=tensorboard
```

### Step 4.3: Monitor Progress

Training outputs metrics to console:

```
Step 1: train/policy_loss=0.0234, train/kl_loss=0.0012, ...
Step 5: eval/all/pass_at_1=0.2684, eval/all/mean_reward=0.27, ...
Step 10: eval/all/pass_at_1=0.3211, ...
...
Step 50: eval/all/pass_at_1=0.4466, ...  # Significant improvement!
```

**Key metrics:**
- `eval/all/pass_at_1`: Accuracy (fraction of correct answers)
- `train/policy_loss`: Policy gradient loss
- `train/kl_loss`: KL divergence from reference model
- `train/mean_reward`: Average reward across batch

### Step 4.4: View TensorBoard (Optional)

```bash
tensorboard --logdir outputs/
```

Navigate to `http://localhost:6006`

---

## Part 5: Training Loop Explained

Here's what happens during each training step:

```
┌─────────────────────────────────────────────────────────────────┐
│ STEP 1: GENERATION                                              │
│                                                                 │
│  For each math problem in batch:                               │
│    SGLang generates 4 responses (n_samples_per_prompt=4)       │
│    Example:                                                     │
│      Prompt: "Janet has 3 apples. She buys 2 more. How many?"  │
│      Response 1: "3 + 2 = 5. The answer is \\boxed{5}"         │
│      Response 2: "Janet now has 5 apples. \\boxed{5}"          │
│      Response 3: "3 + 2 = 6. \\boxed{6}"  (wrong!)             │
│      Response 4: "The answer is \\boxed{5}"                     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ STEP 2: REWARD                                                  │
│                                                                 │
│  GSM8K environment scores each response:                       │
│    Response 1: reward=1.0 (correct)                            │
│    Response 2: reward=1.0 (correct)                            │
│    Response 3: reward=0.0 (wrong)                              │
│    Response 4: reward=1.0 (correct)                            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ STEP 3: ADVANTAGE COMPUTATION (GRPO)                           │
│                                                                 │
│  Group mean: (1+1+0+1)/4 = 0.75                                │
│  Group std: 0.433                                               │
│                                                                 │
│  Advantages:                                                    │
│    Response 1: (1.0 - 0.75) / 0.433 = +0.58                    │
│    Response 2: (1.0 - 0.75) / 0.433 = +0.58                    │
│    Response 3: (0.0 - 0.75) / 0.433 = -1.73  (discouraged!)    │
│    Response 4: (1.0 - 0.75) / 0.433 = +0.58                    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ STEP 4: POLICY UPDATE                                          │
│                                                                 │
│  Compute policy gradient loss:                                 │
│    - Increase probability of high-advantage responses          │
│    - Decrease probability of low-advantage responses           │
│    - Apply PPO clipping for stability                          │
│    - Add KL penalty to stay close to reference model           │
│                                                                 │
│  Update model weights via gradient descent                     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ STEP 5: WEIGHT SYNC                                            │
│                                                                 │
│  Broadcast updated weights to SGLang engines (~1.9s)           │
│    - Uses CUDA IPC (zero-copy) when colocate_all=true         │
│    - Clears KV cache after update                              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                        REPEAT...
```

---

## Part 6: Using Your Trained Model

### Step 6.1: Load Checkpoint

Checkpoints are saved to `~/ckpts/global_step_*/`:

```bash
ls ~/ckpts/
# global_step_10/  global_step_20/  global_step_30/  ...
```

### Step 6.2: Run Inference

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load your trained model
model_path = "~/ckpts/global_step_50"
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Generate response
messages = [
    {"role": "system", "content": "You are a helpful math tutor. Solve problems step by step."},
    {"role": "user", "content": "Tom has 7 marbles. He gives 3 to his friend. How many does he have left?"}
]

inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to(model.device)
outputs = model.generate(inputs, max_new_tokens=512, temperature=0.0)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

### Step 6.3: Evaluate on Test Set

```bash
python -m skyrl_train.entrypoints.main_base \
  --config-path . \
  --config-name my_training_config \
  trainer.policy.model.path="~/ckpts/global_step_50" \
  trainer.epochs=0 \
  trainer.eval_before_train=true
```

---

## Part 7: Common Customizations

### 7.1: Change Model Size

```yaml
trainer:
  policy:
    model:
      path: "Qwen/Qwen2.5-1.5B-Instruct"  # Larger model

generator:
  inference_engine_tensor_parallel_size: 2  # Split across 2 GPUs
```

### 7.2: Use Different Algorithm

**PPO with GAE (requires critic):**
```yaml
trainer:
  algorithm:
    advantage_estimator: gae
    gamma: 0.99
    lambda_: 0.95

  critic:
    model:
      path: "Qwen/Qwen2.5-0.5B-Instruct"
```

**RLOO (Leave-One-Out baseline):**
```yaml
trainer:
  algorithm:
    advantage_estimator: rloo
```

### 7.3: Enable LoRA Fine-tuning

```yaml
trainer:
  policy:
    model:
      path: "Qwen/Qwen2.5-0.5B-Instruct"
      lora:
        rank: 32
        alpha: 32
        target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]
    optimizer_config:
      lr: 3.0e-5  # Higher LR for LoRA

generator:
  enable_lora: true
  engine_init_kwargs:
    max_lora_rank: 64
```

### 7.4: Multi-GPU Training

```yaml
trainer:
  placement:
    colocate_all: true
    policy_num_gpus_per_node: 8

generator:
  num_inference_engines: 8
  inference_engine_tensor_parallel_size: 1
```

### 7.5: Use Weights & Biases Logging

```yaml
trainer:
  logger: wandb
  project_name: "gsm8k-training"
  run_name: "sglang-grpo-experiment"
```

Then:
```bash
export WANDB_API_KEY="your-key"
python -m skyrl_train.entrypoints.main_base ...
```

---

## Part 8: Troubleshooting

### Issue: FlashAttention Error
```
AssertionError: FlashAttention v3 Backend requires SM>=80
```
**Fix:** Use FlashInfer:
```bash
+generator.engine_init_kwargs.attention_backend=flashinfer
```

### Issue: Ray Worker Crash with Editable SGLang
```
error: Failed to generate package metadata for `sglang @ editable+...`
```
**Fix:**
```bash
unset RAY_RUNTIME_ENV_HOOK
```

### Issue: Out of Memory
**Fix:** Reduce memory usage:
```yaml
generator:
  gpu_memory_utilization: 0.6
  max_num_seqs: 256

trainer:
  micro_train_batch_size_per_gpu: 4
```

### Issue: Weight Sync Timeout
**Fix:** Increase timeout:
```yaml
generator:
  weight_sync_timeout: 120
```

---

## Summary

You've learned how to:

1. **Install** SkyRL with SGLang support
2. **Prepare** datasets in the correct format
3. **Configure** training with GRPO algorithm
4. **Run** RL training with SGLang inference
5. **Monitor** and evaluate training progress
6. **Use** your trained model for inference
7. **Customize** for different scenarios

**Expected Results (Qwen2.5-0.5B on GSM8K):**
- Initial accuracy: ~24%
- After 50 steps: ~45%
- Training time: ~2-3 hours on 4x A100

---

## Next Steps

- [Custom Environments Guide](./CUSTOM_ENVIRONMENTS.md) - Create your own reward functions
- [Full Configuration Reference](./SGLANG_INTEGRATION_GUIDE.md) - All available options
- [FAQ](./FAQ_SGLANG.md) - Common questions answered
- [Examples Directory](../examples/) - More training configurations
