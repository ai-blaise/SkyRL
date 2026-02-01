#!/bin/bash
# NMoE GRPO Training Script
#
# Usage:
#   ./run_training.sh /path/to/nmoe/checkpoint /path/to/train_data.parquet
#
# Example:
#   ./run_training.sh /data/models/nmoe-7b /data/math/train.parquet

set -e

# Check arguments
if [ $# -lt 2 ]; then
    echo "Usage: $0 <model_path> <train_data_path> [additional_args...]"
    echo ""
    echo "Example:"
    echo "  $0 /data/models/nmoe-7b /data/math/train.parquet"
    exit 1
fi

MODEL_PATH="$1"
TRAIN_DATA="$2"
shift 2
EXTRA_ARGS="$@"

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKYRL_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

# Validate paths
if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: Model path does not exist: $MODEL_PATH"
    exit 1
fi

if [ ! -f "$TRAIN_DATA" ]; then
    echo "Error: Training data does not exist: $TRAIN_DATA"
    exit 1
fi

echo "=============================================="
echo "NMoE GRPO Training"
echo "=============================================="
echo "Model: $MODEL_PATH"
echo "Training data: $TRAIN_DATA"
echo "Extra args: $EXTRA_ARGS"
echo "=============================================="

# Change to SkyRL directory
cd "$SKYRL_DIR"

# Run training
python -m skyrl_train.cli \
    --config-path examples/nmoe \
    --config-name config \
    trainer.policy.model.path="$MODEL_PATH" \
    "data.train_data=[\"$TRAIN_DATA\"]" \
    $EXTRA_ARGS
