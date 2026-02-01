# Remote Inference Engine

Run inference engines on separate servers from the training process.

## When to Use Remote Inference

Use remote inference when:
- Training and inference on different hardware
- Scaling inference independently
- Using pre-existing inference servers
- Memory constraints require separation

**Note:** Weight synchronization is disabled with remote servers. Models won't update during training.

## Prerequisites

- Network connectivity between training and inference servers
- Same model loaded on inference server
- SGLang or vLLM server running

## Quick Start

### 1. Start Inference Server

**SGLang:**
```bash
bash examples/remote_inference_engine/run_sglang_server.sh
```

**vLLM:**
```bash
bash examples/remote_inference_engine/run_vllm_server.sh
```

### 2. Run Training

```bash
bash examples/remote_inference_engine/run_remote.sh
```

## Configuration

```yaml
generator:
  run_engines_locally: false
  remote_inference_engine_urls:
    - "192.168.1.100:8001"
    - "192.168.1.101:8001"
```

## Available Scripts

| Script | Purpose |
|--------|---------|
| `run_sglang_server.sh` | Launch SGLang inference server |
| `run_vllm_server.sh` | Launch vLLM inference server |
| `run_remote.sh` | Training with remote inference |

## Limitations

- **No weight sync:** Remote servers don't receive updated weights
- **Network latency:** Adds overhead to generation
- **Model consistency:** Ensure same model version on all servers

## Related Documentation

- [SGLang Integration Guide](../../docs/SGLANG_INTEGRATION_GUIDE.md)
- [FAQ - Remote Servers](../../docs/FAQ_SGLANG.md)
