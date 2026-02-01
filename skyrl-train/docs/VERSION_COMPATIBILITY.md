# Version Compatibility Matrix

**Complete version requirements and compatibility information for SkyRL + SGLang.**

---

## Quick Reference

| Component | Required Version | Notes |
|-----------|------------------|-------|
| Python | **3.12.x** (strict) | SkyRL requires exactly Python 3.12 |
| PyTorch | 2.9.1 | For SGLang backend |
| SGLang | 0.4.x+ | With RL training support |
| CUDA | 12.8+ | Required for all backends |
| Ray | 2.51.1 | Distributed execution |
| FlashInfer | 0.5.3 | Attention backend |

---

## 1. Python Version

### SkyRL Requirement

**SkyRL requires Python 3.12.x exactly.**

```toml
# From pyproject.toml
requires-python = "==3.12.*"
```

This is a strict requirement due to:
- Ray compatibility requirements
- Flash attention compilation
- Transformer engine dependencies

### Installation

```bash
# Using pyenv
pyenv install 3.12.7
pyenv local 3.12.7

# Or using conda
conda create -n skyrl python=3.12
conda activate skyrl
```

### Verification

```bash
python --version
# Python 3.12.x
```

---

## 2. PyTorch Versions

### By Backend

| Backend | PyTorch Version | CUDA Version |
|---------|-----------------|--------------|
| SGLang | 2.9.1 | 12.8 |
| vLLM | 2.8.0 | 12.8 |
| Flash RL | 2.7.0 | 12.8 |
| Megatron-Core | 2.8.0 | 12.8 |

### SGLang Backend (Recommended)

```bash
# Install with SGLang backend
cd SkyRL/skyrl-train
uv sync --extra sglang

# Verify
python -c "import torch; print(torch.__version__)"
# 2.9.1+cu128
```

### vLLM Backend

```bash
# Install with vLLM backend
uv sync --extra vllm

# Verify
python -c "import torch; print(torch.__version__)"
# 2.8.0+cu128
```

**Note:** You cannot install both SGLang and vLLM extras simultaneously due to PyTorch version conflicts.

---

## 3. SGLang Version

### Minimum Requirements

- **SGLang 0.4.0+** for RL training support
- Weight sync APIs (`update_weights_from_tensor`, `update_weights_from_distributed`)
- Memory saver (`release_memory_occupation`, `resume_memory_occupation`)

### Version Check

```python
import sglang
print(sglang.__version__)
# 0.4.x or higher
```

### Key APIs by Version

| API | Min SGLang Version |
|-----|-------------------|
| `update_weights_from_tensor` | 0.3.0 |
| `update_weights_from_distributed` | 0.4.0 |
| `release_memory_occupation` | 0.4.0 |
| `resume_memory_occupation` | 0.4.0 |
| `init_weights_update_group` | 0.4.0 |

---

## 4. CUDA and GPU Requirements

### CUDA Version

| Backend | CUDA Version |
|---------|--------------|
| All backends | 12.8+ |

### GPU Compute Capability

| GPU | Compute | FlashInfer | FA3 | Recommended |
|-----|---------|------------|-----|-------------|
| RTX 3090 | 8.6 | Yes | No | flashinfer |
| RTX 4090 | 8.9 | Yes | No | flashinfer |
| A100 | 8.0 | Yes | Yes | fa3 or flashinfer |
| H100 | 9.0 | Yes | Yes | fa3 |
| L4 | 8.9 | Yes | No | flashinfer |
| H200 | 9.0 | Yes | Yes | fa3 |

### GPU Memory Requirements

| Model Size | Minimum VRAM | Recommended VRAM |
|------------|--------------|------------------|
| 0.5B | 8 GB | 16 GB |
| 1.5B | 16 GB | 24 GB |
| 3B | 24 GB | 40 GB |
| 7-8B | 40 GB | 80 GB |
| 14B | 80 GB | 2x 80 GB |
| 32B | 2x 80 GB | 4x 80 GB |
| 70B | 4x 80 GB | 8x 80 GB |

---

## 5. Key Dependencies

### Core Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| transformers | >=4.51.0 | Model loading, tokenizers |
| ray | 2.51.1 | Distributed execution |
| accelerate | latest | Training utilities |
| peft | latest | LoRA support |
| datasets | 4.0.0 | Data loading |
| flash-attn | >=2.8.3 | Attention kernels |
| hydra-core | 1.3.2 | Configuration |

### SGLang-Specific Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| flashinfer-python | 0.5.3 | Attention backend |
| flashinfer-cubin | 0.5.3 | Precompiled kernels |
| torch_memory_saver | 0.0.9 | Memory management |
| sgl-kernel | 0.3.20 | SGLang CUDA kernels |

---

## 6. Complete Backend Configurations

### SGLang Backend (Recommended)

```bash
# Installation
cd SkyRL/skyrl-train
uv sync --extra sglang
source .venv/bin/activate

# Key versions
python -c "
import torch
import sglang
import flashinfer
print(f'PyTorch: {torch.__version__}')
print(f'SGLang: {sglang.__version__}')
print(f'FlashInfer: {flashinfer.__version__}')
print(f'CUDA: {torch.version.cuda}')
"
```

Expected output:
```
PyTorch: 2.9.1+cu128
SGLang: 0.4.x
FlashInfer: 0.5.3
CUDA: 12.8
```

### vLLM Backend

```bash
# Installation
cd SkyRL/skyrl-train
uv sync --extra vllm
source .venv/bin/activate

# Key versions
python -c "
import torch
import vllm
print(f'PyTorch: {torch.__version__}')
print(f'vLLM: {vllm.__version__}')
print(f'CUDA: {torch.version.cuda}')
"
```

Expected output:
```
PyTorch: 2.8.0+cu128
vLLM: 0.11.0
CUDA: 12.8
```

---

## 7. Compatibility Issues and Solutions

### Issue: PyTorch Version Mismatch

**Symptom:**
```
ImportError: torch.cuda.is_available() returns False
```

**Solution:**
```bash
# Reinstall with correct CUDA version
pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
```

### Issue: FlashInfer Not Found

**Symptom:**
```
ModuleNotFoundError: No module named 'flashinfer'
```

**Solution:**
```bash
pip install flashinfer-python==0.5.3 --index-url https://flashinfer.ai/whl/cu128
```

### Issue: CUDA Version Mismatch

**Symptom:**
```
RuntimeError: CUDA error: no kernel image is available
```

**Solution:**
```bash
# Check CUDA version
nvidia-smi  # Look for CUDA Version

# Reinstall packages for correct CUDA
pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
```

### Issue: Python Version Error

**Symptom:**
```
ERROR: Package 'skyrl-train' requires a different Python: 3.11.x not in '==3.12.*'
```

**Solution:**
```bash
# Install Python 3.12
pyenv install 3.12.7
pyenv local 3.12.7

# Or recreate venv
python3.12 -m venv .venv
source .venv/bin/activate
```

---

## 8. Docker Images

### Official Images

| Image | Base | For Backend |
|-------|------|-------------|
| `lmsysorg/sglang:latest` | CUDA 12.8 | SGLang |
| `vllm/vllm-openai:latest` | CUDA 12.x | vLLM |

### Custom Dockerfile

```dockerfile
FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

# Install Python 3.12
RUN apt-get update && apt-get install -y python3.12 python3.12-venv

# Set up environment
WORKDIR /app
COPY . .

# Install SkyRL with SGLang
RUN python3.12 -m venv .venv
RUN .venv/bin/pip install -e ".[sglang]"

ENTRYPOINT [".venv/bin/python", "-m", "skyrl_train.entrypoints.main_base"]
```

---

## 9. Version Verification Script

Use this script to verify your environment:

```python
#!/usr/bin/env python
"""Verify SkyRL + SGLang environment compatibility."""

import sys

def check_version():
    errors = []
    warnings = []

    # Python version
    py_version = sys.version_info
    if py_version.major != 3 or py_version.minor != 12:
        errors.append(f"Python 3.12 required, found {py_version.major}.{py_version.minor}")
    else:
        print(f"[OK] Python {py_version.major}.{py_version.minor}.{py_version.micro}")

    # PyTorch
    try:
        import torch
        if not torch.__version__.startswith("2.9"):
            warnings.append(f"PyTorch 2.9.x recommended, found {torch.__version__}")
        else:
            print(f"[OK] PyTorch {torch.__version__}")

        if not torch.cuda.is_available():
            errors.append("CUDA not available")
        else:
            print(f"[OK] CUDA {torch.version.cuda}")
            print(f"[OK] GPUs: {torch.cuda.device_count()}")
    except ImportError:
        errors.append("PyTorch not installed")

    # SGLang
    try:
        import sglang
        print(f"[OK] SGLang {sglang.__version__}")
    except ImportError:
        warnings.append("SGLang not installed (required for sglang backend)")

    # FlashInfer
    try:
        import flashinfer
        if flashinfer.__version__ != "0.5.3":
            warnings.append(f"FlashInfer 0.5.3 recommended, found {flashinfer.__version__}")
        else:
            print(f"[OK] FlashInfer {flashinfer.__version__}")
    except ImportError:
        warnings.append("FlashInfer not installed")

    # Ray
    try:
        import ray
        print(f"[OK] Ray {ray.__version__}")
    except ImportError:
        errors.append("Ray not installed")

    # Transformers
    try:
        import transformers
        print(f"[OK] Transformers {transformers.__version__}")
    except ImportError:
        errors.append("Transformers not installed")

    # Summary
    print("\n" + "="*50)
    if errors:
        print("ERRORS (must fix):")
        for e in errors:
            print(f"  - {e}")
    if warnings:
        print("WARNINGS (may cause issues):")
        for w in warnings:
            print(f"  - {w}")
    if not errors and not warnings:
        print("Environment is fully compatible!")

    return len(errors) == 0

if __name__ == "__main__":
    sys.exit(0 if check_version() else 1)
```

Save as `check_env.py` and run:
```bash
python check_env.py
```

---

## 10. Upgrade Guide

### Upgrading SGLang

```bash
# In SkyRL directory
cd SkyRL/skyrl-train

# Update SGLang
pip install --upgrade sglang

# Or if using editable install
cd ../sglang/python
git pull origin main
pip install -e .
```

### Upgrading SkyRL

```bash
cd SkyRL/skyrl-train
git pull origin main
uv sync --extra sglang  # or vllm
```

---

## References

- [SGLang Installation](https://docs.sglang.io/get_started/install.html)
- [PyTorch Installation](https://pytorch.org/get-started/locally/)
- [FlashInfer Installation](https://flashinfer.ai/docs/)
- [SkyRL Quickstart](./QUICKSTART_SGLANG.md)
