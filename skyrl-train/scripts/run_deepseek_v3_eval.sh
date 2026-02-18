#!/bin/bash
# Run DeepSeek-V3.2-REAP-345B-A37B evaluation with sglang backend
#
# First run: bash scripts/launch_server.sh
# Then run:  bash scripts/run_deepseek_v3_eval.sh [SGLANG_URL]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

SGLANG_URL="${1:-127.0.0.1:8000}"
MODEL="cerebras/DeepSeek-V3.2-REAP-345B-A37B"
VAL_DATA="${HOME}/data/gsm8k/validation.parquet"
CHAT_TEMPLATE="${PROJECT_DIR}/deepseek_v3_chat_template.jinja2"

# Ensure data exists
if [ ! -f "$VAL_DATA" ]; then
    echo "Validation data not found at $VAL_DATA"
    echo "Run: python examples/gsm8k/gsm8k_dataset.py"
    exit 1
fi

cd "$PROJECT_DIR"
unset RAY_RUNTIME_ENV_HOOK 2>/dev/null || true

exec "${PROJECT_DIR}/.venv/bin/python" -m skyrl_train.entrypoints.main_generate \
  data.val_data="[\"${VAL_DATA}\"]" \
  trainer.policy.model.path="${MODEL}" \
  trainer.placement.colocate_all=false \
  generator.backend=sglang \
  generator.run_engines_locally=false \
  "generator.remote_inference_engine_urls=[\"${SGLANG_URL}\"]" \
  generator.num_inference_engines=1 \
  generator.inference_engine_tensor_parallel_size=8 \
  generator.chat_template.source=file \
  generator.chat_template.name_or_path="${CHAT_TEMPLATE}" \
  generator.eval_sampling_params.max_generate_length=512 \
  generator.eval_sampling_params.temperature=0.7 \
  generator.n_samples_per_prompt=4 \
  generator.sampling_params.logprobs=0 \
  environment.env_class=gsm8k \
  trainer.logger=console
