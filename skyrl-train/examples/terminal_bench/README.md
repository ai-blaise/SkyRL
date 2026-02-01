# Terminal-Bench Integration (WIP)

Train models on Terminal-Bench tasks for terminal/CLI interaction capabilities.

> **Status:** Work in Progress. Training tasks are currently hard-coded as "hello-world" in the prototype.

---

## Overview

Terminal-Bench is a benchmark for evaluating LLM performance on terminal/command-line tasks. This integration enables RL training on these tasks.

---

## Prerequisites

This integration requires the `harbor` repo:

```bash
cd SkyRL/skyrl-train
git clone https://github.com/laude-institute/harbor.git
```

---

## Quick Start

### Training

```bash
bash examples/terminal_bench/run_tbench.sh
```

### Generation Only

Launch the generator/serving process for rapid debugging (avoids trainer setup overhead):

```bash
bash examples/terminal_bench/run_tbench_gen.sh
```

---

## Directory Structure

| Directory | Purpose |
|-----------|---------|
| `entrypoints/` | Training and generation entry points (`main_tbench.py`, `main_tbench_generate.py`) |
| `generator/` | Terminal-Bench specific generator (`terminal_bench_generator.py`) |
| `terminal_bench_config/` | Configuration files for Terminal-Bench integration |
| `dataset.py` | Dataset preparation for Terminal-Bench tasks |

---

## Current Limitations

- Tasks are hard-coded as "hello-world"
- Full task specification support is in development

---

## Related Documentation

- [Custom Environments](../../docs/CUSTOM_ENVIRONMENTS.md)
- [Mini SWE Agent](../mini_swe_agent/README.md)