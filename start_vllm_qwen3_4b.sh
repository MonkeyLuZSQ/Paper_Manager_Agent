#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/home/zhengsq/.cache/modelscope/hub/models/Qwen/Qwen3-4B-AWQ}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-2048}"
GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.70}"
KV_CACHE_DTYPE="${VLLM_KV_CACHE_DTYPE:-auto}"
ENABLE_PREFIX_CACHING="${VLLM_ENABLE_PREFIX_CACHING:-1}"
ENABLE_CHUNKED_PREFILL="${VLLM_ENABLE_CHUNKED_PREFILL:-0}"
MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-}"
MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-4}"
ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-1}"
KV_CACHE_METRICS="${VLLM_KV_CACHE_METRICS:-0}"
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

serve_args=(
  "$VLLM_BIN" serve "$MODEL_PATH"
  --quantization awq_marlin
  --dtype float16
  --max-model-len "$MAX_MODEL_LEN"
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --kv-cache-dtype "$KV_CACHE_DTYPE"
  --max-num-seqs "$MAX_NUM_SEQS"
  --attention-backend TRITON_ATTN
  --served-model-name qwen3-4b
  --host "${VLLM_BIND_HOST:-0.0.0.0}"
  --port 8000
)

if [ "$ENABLE_PREFIX_CACHING" = "1" ]; then
  serve_args+=(--enable-prefix-caching)
fi
if [ "$ENABLE_CHUNKED_PREFILL" = "1" ]; then
  serve_args+=(--enable-chunked-prefill)
fi
if [ -n "$MAX_NUM_BATCHED_TOKENS" ]; then
  serve_args+=(--max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS")
fi
if [ "$ENFORCE_EAGER" = "1" ]; then
  serve_args+=(--enforce-eager)
fi
if [ "$KV_CACHE_METRICS" = "1" ]; then
  serve_args+=(--kv-cache-metrics)
fi

printf 'vLLM max_model_len=%s gpu_memory_utilization=%s kv_cache_dtype=%s max_num_seqs=%s prefix_caching=%s chunked_prefill=%s enforce_eager=%s\n' \
  "$MAX_MODEL_LEN" "$GPU_MEMORY_UTILIZATION" "$KV_CACHE_DTYPE" "$MAX_NUM_SEQS" "$ENABLE_PREFIX_CACHING" "$ENABLE_CHUNKED_PREFILL" "$ENFORCE_EAGER"
"${serve_args[@]}"
