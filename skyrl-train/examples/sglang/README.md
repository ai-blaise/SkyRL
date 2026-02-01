# SGLang Configuration Example

This folder contains a comprehensive SGLang configuration example demonstrating all available options.

## Files

- `sglang_full_config_example.yaml` - Complete configuration with all SGLang-specific options documented
- `sglang_speculative_decoding_example.yaml` - Speculative decoding for 2-3x inference speedup

## When to Use

Use this example when you need to:
- See all available SGLang configuration options in one place
- Create a custom configuration for production training
- Understand the relationship between trainer and generator settings

## Quick Start

```bash
# Run with this config directly
python -m skyrl_train.entrypoints.main_base \
  --config-path examples/sglang \
  --config-name sglang_full_config_example

# Or copy and customize
cp examples/sglang/sglang_full_config_example.yaml my_config.yaml
# Edit my_config.yaml, then:
python -m skyrl_train.entrypoints.main_base \
  --config-path . \
  --config-name my_config
```

## Key Configuration Sections

| Section | Purpose |
|---------|---------|
| `trainer.placement` | GPU allocation and colocated training/inference |
| `generator.backend` | Set to `sglang` for SGLang backend |
| `generator.num_inference_engines` | Number of parallel SGLang instances |
| `generator.weight_sync_backend` | `nccl` for fast GPU-to-GPU sync |
| `generator.engine_init_kwargs` | SGLang-specific options (attention backend, LoRA, etc.) |
| `generator.speculative_decoding` | Enable 2-3x inference speedup with draft model |

## Speculative Decoding

Enable speculative decoding for faster inference:

```yaml
generator:
  speculative_decoding:
    enabled: true
    algorithm: "ngram"  # No draft model needed, LoRA compatible
```

Available algorithms:
- **ngram**: Pattern matching (no draft model, LoRA-compatible)
- **eagle/eagle3**: Tree-based with EAGLE-trained draft (fastest)
- **standalone**: Use smaller model from same family

See `sglang_speculative_decoding_example.yaml` for full examples.

## Prometheus Metrics

Enable observability for monitoring training:

```yaml
generator:
  metrics:
    enabled: true  # Exposes /metrics endpoint
    tracing:
      enabled: true
      otlp_endpoint: "localhost:4317"
```

Available features:
- **Prometheus metrics**: 50+ metrics for latency, throughput, cache hits
- **OpenTelemetry tracing**: Distributed request tracing
- **Request logging**: Structured logging of inputs/outputs
- **Per-request export**: Detailed metrics to files

See [API Reference Section 33](../../docs/API_REFERENCE.md) for full documentation.

## Session-Based Generation

Enable KV cache reuse for multi-turn RL environments:

```yaml
generator:
  sessions:
    enabled: true
    default_capacity: 8192  # Tokens per session
    pool_sessions: false
```

Benefits:
- **2-10x speedup** for multi-turn generation
- **KV cache reuse** across conversation turns
- **Branching support** for MCTS/tree search

**Note**: Session API must be called explicitly via custom generator code. See [API Reference Section 23](../../docs/API_REFERENCE.md) for usage examples.

## SGLang vs vLLM

To switch between backends, change:
```yaml
generator:
  backend: sglang  # or "vllm"
```

See [BACKEND_SELECTION.md](../../docs/BACKEND_SELECTION.md) for detailed comparison.

## Related Documentation

- [Quickstart Guide](../../docs/QUICKSTART_SGLANG.md) - Get started in 10 minutes
- [Full Integration Guide](../../docs/SGLANG_INTEGRATION_GUIDE.md) - Complete configuration reference
- [FAQ](../../docs/FAQ_SGLANG.md) - Common questions answered
