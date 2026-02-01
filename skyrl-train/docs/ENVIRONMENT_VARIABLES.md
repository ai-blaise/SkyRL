# Environment Variables Reference

**Complete reference for environment variables used in SkyRL and SGLang.**

---

## Quick Reference

| Variable | Purpose | Default |
|----------|---------|---------|
| `CUDA_VISIBLE_DEVICES` | GPU selection | All GPUs |
| `NCCL_DEBUG` | NCCL debugging | `WARN` |
| `RAY_DEDUP_LOGS` | Ray log deduplication | `1` |
| `RAY_RUNTIME_ENV_HOOK` | Ray env hooks (unset for editable installs) | Set |
| `SGLANG_LOG_LEVEL` | SGLang logging | `INFO` |
| `HF_TOKEN` | HuggingFace authentication | None |
| `WANDB_API_KEY` | Weights & Biases logging | None |
| `SKYRL_RAY_PG_TIMEOUT_IN_S` | Placement group timeout | `180` |
| `NCCL_CUMEM_ENABLE` | CUDA IPC memory for weight sync | Auto |
| `VLLM_ATTENTION_BACKEND` | vLLM attention backend | Auto |

---

## 1. GPU and CUDA

### CUDA_VISIBLE_DEVICES

Control which GPUs are visible to the application.

```bash
# Use GPUs 0 and 1
export CUDA_VISIBLE_DEVICES=0,1

# Use only GPU 0
export CUDA_VISIBLE_DEVICES=0

# Use GPUs 4-7
export CUDA_VISIBLE_DEVICES=4,5,6,7
```

### CUDA_LAUNCH_BLOCKING

Enable synchronous CUDA operations for debugging.

```bash
# Enable for debugging CUDA errors
export CUDA_LAUNCH_BLOCKING=1

# Default (async, faster)
unset CUDA_LAUNCH_BLOCKING
```

**Note:** Only use for debugging - significantly impacts performance.

### TORCH_CUDA_ALLOC_CONF

Configure PyTorch CUDA memory allocator.

```bash
# Reduce fragmentation
export TORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Set max split size
export TORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

---

## 2. NCCL (Distributed Communication)

### NCCL_DEBUG

Control NCCL debug output level.

```bash
# Levels: WARN, INFO, TRACE
export NCCL_DEBUG=INFO

# Full debugging
export NCCL_DEBUG=TRACE
export NCCL_DEBUG_SUBSYS=ALL
```

### NCCL_TIMEOUT

Set NCCL operation timeout (seconds).

```bash
# Increase timeout for large models (default: 1800)
export NCCL_TIMEOUT=3600  # 1 hour
```

### NCCL_IB_DISABLE

Disable InfiniBand (for debugging network issues).

```bash
# Disable InfiniBand
export NCCL_IB_DISABLE=1

# Use TCP instead
export NCCL_SOCKET_IFNAME=eth0
```

### NCCL_P2P_DISABLE

Disable peer-to-peer GPU communication.

```bash
# Disable P2P (if experiencing GPU communication issues)
export NCCL_P2P_DISABLE=1
```

### NCCL_ASYNC_ERROR_HANDLING

Enable async error handling.

```bash
export NCCL_ASYNC_ERROR_HANDLING=1
```

---

## 3. Ray Configuration

### RAY_DEDUP_LOGS

Control Ray log deduplication.

```bash
# Show all logs (disable deduplication)
export RAY_DEDUP_LOGS=0

# Enable deduplication (default)
export RAY_DEDUP_LOGS=1
```

### RAY_RUNTIME_ENV_HOOK

Control Ray runtime environment hooks.

```bash
# Unset to avoid editable install issues
unset RAY_RUNTIME_ENV_HOOK
```

**Important:** Must unset when using editable SGLang installs to avoid package metadata errors.

### RAY_ADDRESS

Connect to existing Ray cluster.

```bash
# Auto-detect local cluster
export RAY_ADDRESS=auto

# Specific address
export RAY_ADDRESS=ray://192.168.1.100:10001
```

### RAY_memory_monitor_refresh_ms

Memory monitoring interval.

```bash
# Increase monitoring interval
export RAY_memory_monitor_refresh_ms=1000
```

---

## 4. SGLang Configuration

### SGLANG_LOG_LEVEL

Control SGLang logging verbosity.

```bash
# Levels: debug, info, warning, error
export SGLANG_LOG_LEVEL=debug

# Default
export SGLANG_LOG_LEVEL=info
```

### SGLANG_DISABLE_FLASHINFER

Disable FlashInfer backend (for debugging).

```bash
export SGLANG_DISABLE_FLASHINFER=1
```

### SGLANG_ENABLE_TORCH_COMPILE

Enable torch.compile for model optimization.

```bash
export SGLANG_ENABLE_TORCH_COMPILE=1
```

### SGLANG_TORCH_COMPILE_MAX_BS

Maximum batch size for torch.compile.

```bash
export SGLANG_TORCH_COMPILE_MAX_BS=64
```

---

## 5. SkyRL Configuration

### SKYRL_LOG_LEVEL

Control SkyRL logging verbosity.

```bash
# Levels: DEBUG, INFO, WARNING, ERROR
export SKYRL_LOG_LEVEL=DEBUG
```

### SKYRL_CACHE_DIR

Directory for SkyRL cached data.

```bash
export SKYRL_CACHE_DIR=/scratch/skyrl_cache
```

### SKYRL_DISABLE_WANDB

Disable Weights & Biases logging.

```bash
export SKYRL_DISABLE_WANDB=1
```

---

## 6. HuggingFace Configuration

### HF_TOKEN

Authentication token for gated models.

```bash
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxx
```

### HF_HOME

HuggingFace cache directory.

```bash
export HF_HOME=/scratch/huggingface
```

### HF_HUB_ENABLE_HF_TRANSFER

Enable fast file transfers (recommended).

```bash
export HF_HUB_ENABLE_HF_TRANSFER=1
```

### TRANSFORMERS_CACHE

Transformers model cache directory.

```bash
export TRANSFORMERS_CACHE=/scratch/transformers
```

---

## 7. Weights & Biases

### WANDB_API_KEY

W&B API key for experiment tracking.

```bash
export WANDB_API_KEY=xxxxxxxxxxxxxxxxxxxxx
```

### WANDB_PROJECT

Default project name.

```bash
export WANDB_PROJECT=skyrl-experiments
```

### WANDB_ENTITY

W&B team/organization.

```bash
export WANDB_ENTITY=my-team
```

### WANDB_MODE

Control W&B behavior.

```bash
# Disable W&B completely
export WANDB_MODE=disabled

# Offline mode (sync later)
export WANDB_MODE=offline

# Normal mode (default)
export WANDB_MODE=online
```

---

## 8. PyTorch Configuration

### PYTORCH_CUDA_ALLOC_CONF

CUDA memory allocation configuration.

```bash
# Reduce memory fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

### PYTORCH_NO_CUDA_MEMORY_CACHING

Disable CUDA memory caching (for debugging).

```bash
export PYTORCH_NO_CUDA_MEMORY_CACHING=1
```

### TORCH_DISTRIBUTED_DEBUG

Enable distributed debugging.

```bash
# Levels: OFF, INFO, DETAIL
export TORCH_DISTRIBUTED_DEBUG=DETAIL
```

### TORCH_SHOW_CPP_STACKTRACES

Show C++ stack traces on errors.

```bash
export TORCH_SHOW_CPP_STACKTRACES=1
```

---

## 9. Performance Tuning

### OMP_NUM_THREADS

OpenMP thread count.

```bash
# Set based on CPU cores per GPU
export OMP_NUM_THREADS=8
```

### MKL_NUM_THREADS

Intel MKL thread count.

```bash
export MKL_NUM_THREADS=8
```

### TOKENIZERS_PARALLELISM

Control tokenizer parallelism.

```bash
# Disable to avoid fork warnings
export TOKENIZERS_PARALLELISM=false
```

---

## 10. Common Configurations

### Development/Debugging

```bash
#!/bin/bash
# debug_env.sh - Full debugging environment

export CUDA_LAUNCH_BLOCKING=1
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
export RAY_DEDUP_LOGS=0
export SGLANG_LOG_LEVEL=debug
export SKYRL_LOG_LEVEL=DEBUG
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export TORCH_SHOW_CPP_STACKTRACES=1

# Unset Ray hook for editable installs
unset RAY_RUNTIME_ENV_HOOK

echo "Debug environment configured"
```

### Production

```bash
#!/bin/bash
# prod_env.sh - Production environment

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_TIMEOUT=3600
export HF_HUB_ENABLE_HF_TRANSFER=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8

# Performance optimizations
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Production environment configured"
```

### Multi-Node Training

```bash
#!/bin/bash
# multinode_env.sh - Multi-node distributed training

# NCCL configuration
export NCCL_DEBUG=WARN
export NCCL_TIMEOUT=3600
export NCCL_IB_DISABLE=0  # Enable InfiniBand if available

# Network interface (update based on your setup)
export NCCL_SOCKET_IFNAME=eth0
export GLOO_SOCKET_IFNAME=eth0

# Ray configuration
export RAY_ADDRESS=auto

echo "Multi-node environment configured"
```

### Memory-Constrained

```bash
#!/bin/bash
# lowmem_env.sh - Memory-constrained environment

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:256
export CUDA_VISIBLE_DEVICES=0

echo "Low-memory environment configured"
```

---

## 11. Slurm Integration

For Slurm-managed clusters:

```bash
#!/bin/bash
#SBATCH --job-name=skyrl-train
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --time=24:00:00

# Set environment
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_DEBUG=WARN
export NCCL_TIMEOUT=3600
export HF_TOKEN=${HF_TOKEN}
export WANDB_API_KEY=${WANDB_API_KEY}

# Unset Ray hook for editable installs
unset RAY_RUNTIME_ENV_HOOK

# Activate environment
source /path/to/skyrl/.venv/bin/activate

# Run training
python -m skyrl_train.entrypoints.main_base \
  --config-path=/path/to/config \
  --config-name=train
```

---

## 12. Troubleshooting

### NCCL Timeout Issues

```bash
# Increase timeout
export NCCL_TIMEOUT=7200

# Add debugging
export NCCL_DEBUG=INFO
```

### Ray Package Metadata Error

```bash
# Error: Failed to generate package metadata for 'sglang @ editable+...'
# Solution: Unset the hook
unset RAY_RUNTIME_ENV_HOOK
```

### Out of Memory

```bash
# Enable memory optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Limit visible GPUs if needed
export CUDA_VISIBLE_DEVICES=0
```

### FlashInfer Issues

```bash
# Disable for debugging
export SGLANG_DISABLE_FLASHINFER=1
```

---

## 13. SkyRL Internal Variables

### SKYRL_RAY_PG_TIMEOUT_IN_S

Timeout for Ray placement group allocation.

```bash
# Increase timeout for slow cluster startup (default: 180)
export SKYRL_RAY_PG_TIMEOUT_IN_S=300
```

### SKYRL_WEIGHT_SYNC_TIMEOUT

Timeout for weight synchronization operations.

```bash
# Increase for large models (default: 60)
export SKYRL_WEIGHT_SYNC_TIMEOUT=120
```

### SKYRL_ENGINE_STARTUP_TIMEOUT

Timeout for inference engine startup.

```bash
# Increase for large models or slow storage (default: 300)
export SKYRL_ENGINE_STARTUP_TIMEOUT=600
```

---

## 14. NCCL Advanced Configuration

### NCCL_CUMEM_ENABLE

Enable NCCL CUDA IPC memory for weight synchronization.

```bash
# Enable CUDA IPC memory (auto-managed by SkyRL)
export NCCL_CUMEM_ENABLE=1

# Disable if experiencing IPC issues
export NCCL_CUMEM_ENABLE=0
```

**Note:** SkyRL automatically manages this variable for weight sync operations.

### NCCL_NET_GDR_LEVEL

GPU Direct RDMA level for multi-node communication.

```bash
# Levels: 0=disabled, 1=P2P, 2=P2P+network, 3=full
export NCCL_NET_GDR_LEVEL=3
```

---

## 15. vLLM Configuration

When using vLLM as the inference backend:

### VLLM_ATTENTION_BACKEND

Override attention backend selection.

```bash
# Options: FLASH_ATTN, XFORMERS, FLASHINFER
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
```

### VLLM_USE_FLASHINFER_SAMPLER

Enable FlashInfer-based sampling.

```bash
export VLLM_USE_FLASHINFER_SAMPLER=1
```

### VLLM_WORKER_MULTIPROC_METHOD

Multiprocessing method for workers.

```bash
# Options: spawn, fork, forkserver
export VLLM_WORKER_MULTIPROC_METHOD=spawn
```

---

## 16. Alternative Loggers

### MLflow

```bash
# MLflow tracking server
export MLFLOW_TRACKING_URI=http://localhost:5000
export MLFLOW_EXPERIMENT_NAME=skyrl-training

# Use in config
# trainer.logger: "mlflow"
```

### SwanLab

```bash
# SwanLab API key
export SWANLAB_API_KEY=<your_key>

# Use in config
# trainer.logger: "swanlab"
```

### TensorBoard

```bash
# TensorBoard log directory
export TENSORBOARD_LOGDIR=/path/to/logs

# Use in config
# trainer.logger: "tensorboard"
```

---

## 17. Modal Cloud Platform

For Modal deployments (see [Modal integration](../integrations/modal/)):

### MODAL_TOKEN_ID

Modal authentication token ID.

```bash
export MODAL_TOKEN_ID=<your_token_id>
```

### MODAL_TOKEN_SECRET

Modal authentication token secret.

```bash
export MODAL_TOKEN_SECRET=<your_token_secret>
```

### MODAL_ENVIRONMENT

Modal environment selection.

```bash
# Options: main, dev, staging
export MODAL_ENVIRONMENT=main
```

---

## 18. Fully Async Training

For fully asynchronous training mode:

### SKYRL_ASYNC_MAX_STALENESS

Maximum staleness steps for async training.

```bash
# How many steps old weights can be (default: 4)
export SKYRL_ASYNC_MAX_STALENESS=4
```

### SKYRL_ASYNC_GENERATION_WORKERS

Number of parallel generation workers.

```bash
# Default: 768
export SKYRL_ASYNC_GENERATION_WORKERS=768
```

---

## References

- [NCCL Environment Variables](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html)
- [PyTorch Distributed](https://pytorch.org/docs/stable/distributed.html)
- [Ray Configuration](https://docs.ray.io/en/latest/ray-core/configure.html)
- [HuggingFace Environment Variables](https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables)
- [vLLM Environment Variables](https://docs.vllm.ai/en/latest/serving/env_vars.html)
- [SGLang Environment Variables](https://docs.sglang.io/)
