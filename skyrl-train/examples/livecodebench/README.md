# LiveCodeBench Training

Train code generation models using the LiveCodeBench benchmark.

## Overview

LiveCodeBench is a contamination-free benchmark for code generation that continuously updates with new problems. This example shows how to train models on LiveCodeBench data for improved code generation capabilities.

## Prerequisites

- 4+ GPUs recommended
- ~50GB disk space for dataset
- `gdown` package for downloading

## Quick Start

### 1. Install Dependencies

```bash
pip install gdown
```

### 2. Download Dataset

```bash
# Download raw data
python examples/livecodebench/lcb_download.py --local_dir ~/data/lcb/download

# Process into training format
python examples/livecodebench/lcb_dataset.py \
  --dataset_dir ~/data/lcb/download \
  --local_dir ~/data/lcb/
```

### 3. Run Training

```bash
export WANDB_API_KEY=<your_key_here>
bash examples/livecodebench/run_lcb.sh
```

## Available Scripts

| Script | Description |
|--------|-------------|
| `lcb_download.py` | Download raw LiveCodeBench data |
| `lcb_dataset.py` | Process data into training format |
| `run_lcb.sh` | Run GRPO training |

## Dataset Notes

- **Large dataset**: The full dataset is very large
- **Streaming recommended**: Use PyArrow streaming for large files
- **JSON format**: Uses JSON instead of parquet for flexibility

```python
# For large files, use streaming:
import pyarrow.json as paj
table = paj.read_json(path, read_options=paj.ReadOptions(block_size=1024*1024))
```

## Configuration Tips

```yaml
# Code generation typically needs longer outputs
generator.sampling_params.max_generate_length: 2048

# Lower temperature for code
generator.sampling_params.temperature: 0.7

# Multiple samples for better coverage
generator.n_samples_per_prompt: 8
```

## Evaluation

After training, evaluate on LiveCodeBench:

```bash
# Run evaluation (implementation depends on your setup)
python eval_livecodebench.py --model_path /path/to/checkpoint
```

## Related Documentation

- [GSM8K Example](../gsm8k/README.md) - Similar single-turn setup
- [Custom Environments](../../docs/CUSTOM_ENVIRONMENTS.md)
