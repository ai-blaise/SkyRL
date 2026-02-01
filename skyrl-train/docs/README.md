# SkyRL-Train Documentation

## System Requirements

Before you begin, ensure you have:

| Requirement | Version | Notes |
|-------------|---------|-------|
| **Python** | 3.12.x (strict) | SkyRL requires exactly Python 3.12 |
| **CUDA** | 12.8+ | Required for all backends |
| **GPU Memory** | 16GB+ | For 0.5B-1.5B models; larger models need more |
| **OS** | Linux | Windows/Mac not supported |
| **Package Manager** | `uv` | [Install uv](https://docs.astral.sh/uv/getting-started/installation/) |

Run `bash scripts/preflight_check.sh` after installation to verify your environment.

---

## Choose Your Path

```
What's your situation?

New to both SGLang and SkyRL?
├── Start here → [Quickstart Guide](./QUICKSTART_SGLANG.md) (10 min)
└── Then → [End-to-End Tutorial](./TUTORIAL_SGLANG.md) (30 min)

Switching from vLLM to SGLang?
└── Start here → [Backend Selection Guide](./BACKEND_SELECTION.md)

Having issues or errors?
├── Check → [FAQ](./FAQ_SGLANG.md)
└── Then → [Troubleshooting](./TROUBLESHOOTING.md)

Need all configuration options?
└── Reference → [Full Integration Guide](./SGLANG_INTEGRATION_GUIDE.md)

Creating a custom reward function?
└── Guide → [Custom Environments](./CUSTOM_ENVIRONMENTS.md)

Training a multi-turn agent (chatbot, tool-use)?
└── Guide → [Multi-Turn Training](./MULTI_TURN_TRAINING.md)
```

---

## SGLang Integration

SkyRL supports SGLang as an inference backend for RL training. Complete documentation:

### Getting Started

| Document | Description |
|----------|-------------|
| [**Quickstart Guide**](./QUICKSTART_SGLANG.md) | Get started in under 10 minutes |
| [**End-to-End Tutorial**](./TUTORIAL_SGLANG.md) | Complete walkthrough from install to trained model |

### Reference

| Document | Description |
|----------|-------------|
| [**Full Integration Guide**](./SGLANG_INTEGRATION_GUIDE.md) | Complete configuration reference |
| [**API Reference**](./API_REFERENCE.md) | SGLang APIs for weight sync, generation, memory |
| [**Data Format Specification**](./DATA_FORMAT.md) | Dataset schema and examples |
| [**Feature Support**](./SGLANG_LIMITATIONS.md) | Supported features and capabilities |
| [**Algorithms Guide**](./ALGORITHMS.md) | GRPO, RLOO, DAPO, PPO, and more |
| [**Batch Sizes Guide**](./BATCH_SIZES.md) | Understanding train/mini/micro batch sizes |
| [**Training Strategies**](./TRAINING_STRATEGIES.md) | FSDP2 vs Megatron decision guide |
| [**Backend Selection**](./BACKEND_SELECTION.md) | SGLang vs vLLM decision guide |
| [**Version Compatibility**](./VERSION_COMPATIBILITY.md) | Python, PyTorch, CUDA requirements |
| [**Environment Variables**](./ENVIRONMENT_VARIABLES.md) | All configuration env vars |
| [**Glossary**](./GLOSSARY.md) | Key terms and acronyms defined |

### Guides

| Document | Description |
|----------|-------------|
| [**Multi-Turn Training**](./MULTI_TURN_TRAINING.md) | Train conversational agents and tool-using models |
| [**Custom Environments**](./CUSTOM_ENVIRONMENTS.md) | Create your own reward functions |
| [**Troubleshooting**](./TROUBLESHOOTING.md) | Diagnose and fix common issues |
| [**FAQ**](./FAQ_SGLANG.md) | Common questions answered |

---

## Quick Links

**Getting Started:**
- **New to SGLang + SkyRL?** Start with the [Quickstart Guide](./QUICKSTART_SGLANG.md)
- **Want a complete walkthrough?** See the [End-to-End Tutorial](./TUTORIAL_SGLANG.md)

**Configuration:**
- **Need all options?** See the [Full Integration Guide](./SGLANG_INTEGRATION_GUIDE.md)
- **Creating datasets?** See [Data Format Specification](./DATA_FORMAT.md)
- **Version requirements?** See [Version Compatibility](./VERSION_COMPATIBILITY.md)
- **Batch size confusion?** See [Batch Sizes Guide](./BATCH_SIZES.md)
- **FSDP2 vs Megatron?** See [Training Strategies](./TRAINING_STRATEGIES.md)
- **SGLang vs vLLM?** See [Backend Selection](./BACKEND_SELECTION.md)

**Algorithms:**
- **Understanding RL algorithms?** See the [Algorithms Guide](./ALGORITHMS.md)

**Development:**
- **Creating custom environments?** See [Custom Environments Guide](./CUSTOM_ENVIRONMENTS.md)
- **Using SGLang APIs?** See [API Reference](./API_REFERENCE.md)
- **Environment variables?** See [Environment Variables Reference](./ENVIRONMENT_VARIABLES.md)

**Help:**
- **Having issues?** Check [Troubleshooting](./TROUBLESHOOTING.md) or [FAQ](./FAQ_SGLANG.md)

---

## Documentation Map

```
                    ┌─────────────────────┐
                    │   QUICKSTART_SGLANG │  ← Start here!
                    │   (10 minutes)      │
                    └─────────┬───────────┘
                              │
                              ▼
                    ┌─────────────────────┐
                    │   TUTORIAL_SGLANG   │  ← Complete walkthrough
                    │   (30 minutes)      │
                    └─────────┬───────────┘
                              │
              ┌───────────────┼───────────────┐
              │               │               │
              ▼               ▼               ▼
     ┌────────────────┐ ┌──────────────┐ ┌──────────────┐
     │ CUSTOM_        │ │ DATA_FORMAT  │ │ INTEGRATION_ │
     │ ENVIRONMENTS   │ │ (datasets)   │ │ GUIDE        │
     │ (reward funcs) │ │              │ │ (all config) │
     └────────────────┘ └──────────────┘ └──────────────┘
              │               │               │
              └───────────────┼───────────────┘
                              │
              ┌───────────────┼───────────────┐
              │               │               │
              ▼               ▼               ▼
     ┌────────────────┐ ┌──────────────┐ ┌──────────────┐
     │ API_REFERENCE  │ │ TROUBLESHOOT │ │ FAQ_SGLANG   │
     │ (SGLang APIs)  │ │ (fix issues) │ │ (answers)    │
     └────────────────┘ └──────────────┘ └──────────────┘
```

---

## Example Configurations

See the [examples directory](../examples/) for complete training configurations:

| Directory | Description |
|-----------|-------------|
| `examples/gsm8k/` | Math problem solving with GSM8K |
| `examples/sglang/` | SGLang-specific full configuration |
| `examples/multiply/` | Simple custom environment example |
| `examples/algorithms/` | Different RL algorithms (GRPO, DAPO, RLOO) |
| `examples/text_to_sql/` | Multi-turn SQL generation |
| `examples/livecodebench/` | Code generation |
| `examples/lora/` | LoRA fine-tuning |

---

## Building the Docs

To build the documentation locally:

```bash
bash build.sh
```

If the default port is in use:

```bash
bash build.sh --port 8010
```

---

## SGLang Documentation

For SGLang-specific documentation, see:
- [SGLang Official Docs](https://docs.sglang.io/)
- [SGLang RL Training Integration](https://github.com/sgl-project/sglang/blob/main/docs/advanced_features/rl_training_integration.md)
