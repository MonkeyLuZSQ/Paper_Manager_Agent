#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/home/zhengsq/.cache/modelscope/hub/models/Qwen/Qwen3-4B-AWQ}"
if [ ! -f "$MODEL_PATH/model.safetensors" ]; then
  MODEL_PATH="${MODEL_PATH_FALLBACK:-Qwen/Qwen3-4B-AWQ}"
fi

VLLM_BIN="${VLLM_BIN:-$(command -v vllm || true)}"
if [ -z "$VLLM_BIN" ] && [ -x /mnt/e/LLM_Project/vllm_demo/.venv/bin/vllm ]; then
  VLLM_BIN="/mnt/e/LLM_Project/vllm_demo/.venv/bin/vllm"
fi
if [ -z "$VLLM_BIN" ]; then
  echo "Cannot find vllm. Set VLLM_BIN=/path/to/vllm first." >&2
  exit 1
fi

"$VLLM_BIN" serve "$MODEL_PATH" \
  --quantization awq_marlin \
  --dtype float16 \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.70 \
  --attention-backend TRITON_ATTN \
  --enforce-eager \
  --served-model-name qwen3-4b \
  --host "${VLLM_BIND_HOST:-0.0.0.0}" \
  --port 8000
