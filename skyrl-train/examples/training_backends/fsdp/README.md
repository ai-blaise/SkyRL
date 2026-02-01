# FSDP vs FSDP2 Comparison

Compare FSDP (PyTorch 1.x) and FSDP2 (PyTorch 2.x) training strategies.

---

## Overview

This directory contains scripts to compare FSDP and FSDP2 training backends. FSDP2 is the recommended default for new projects.

| Strategy | PyTorch Version | Recommended |
|----------|-----------------|-------------|
| `fsdp` | PyTorch 1.x style | Legacy |
| `fsdp2` | PyTorch 2.x style | **Default** |

---

## Quick Start

```bash
# Prepare dataset
python examples/gsm8k/gsm8k_dataset.py

# Run with FSDP2 (recommended)
bash examples/training_backends/fsdp/run_fsdp2.sh

# Run with FSDP1 (legacy)
bash examples/training_backends/fsdp/run_fsdp.sh
```

---

## Key Differences

### FSDP2 Advantages

| Feature | FSDP | FSDP2 |
|---------|------|-------|
| **Mixed precision** | Manual setup | Native DTensor |
| **Activation checkpointing** | Separate wrapper | Integrated |
| **Memory efficiency** | Good | Better |
| **Debugging** | Harder | Easier |
| **Compile support** | Limited | Full `torch.compile` |

### Configuration

```yaml
# FSDP2 (recommended)
trainer.strategy: fsdp2

# FSDP1 (legacy)
trainer.strategy: fsdp
```

---

## When to Use FSDP1

Use FSDP (v1) only if:
- You have existing FSDP1 checkpoints
- Using PyTorch < 2.0
- Specific compatibility requirements

For all new projects, use `fsdp2`.

---

## Related Documentation

- [Training Strategies](../../../docs/TRAINING_STRATEGIES.md)
- [Training Backends Overview](../README.md)
