#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

BIND_HOST="${VLLM_BIND_HOST:-0.0.0.0}"
CLIENT_HOST="${VLLM_CLIENT_HOST:-127.0.0.1}"
PORT="${VLLM_PORT:-8000}"
BASE_URL="${VLLM_BASE_URL:-http://${CLIENT_HOST}:${PORT}/v1}"
HEALTH_URL="http://${CLIENT_HOST}:${PORT}/health"
CLIENT_SELECTED=0
MODEL_NAME="${VLLM_MODEL:-qwen3-4b}"
MODEL_PATH="${VLLM_MODEL_PATH:-/home/zhengsq/.cache/modelscope/hub/models/Qwen/Qwen3-4B-AWQ}"
MODEL_FALLBACK="${VLLM_MODEL_FALLBACK:-Qwen/Qwen3-4B-AWQ}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-2048}"
GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.70}"
KV_CACHE_DTYPE="${VLLM_KV_CACHE_DTYPE:-auto}"
ENABLE_PREFIX_CACHING="${VLLM_ENABLE_PREFIX_CACHING:-1}"
ENABLE_CHUNKED_PREFILL="${VLLM_ENABLE_CHUNKED_PREFILL:-0}"
MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-}"
MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-4}"
ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-1}"
KV_CACHE_METRICS="${VLLM_KV_CACHE_METRICS:-0}"
WAIT_SECONDS="${VLLM_WAIT_SECONDS:-900}"
RESTART_STALE="${VLLM_RESTART_STALE:-1}"
FORCE_RESTART="${VLLM_FORCE_RESTART:-0}"
LOG_DIR="${LOG_DIR:-/tmp/paper_agent_logs}"
VLLM_LOG="$LOG_DIR/vllm_${MODEL_NAME}_${PORT}.log"
VLLM_PID_FILE="$LOG_DIR/vllm_${MODEL_NAME}_${PORT}.pid"

PAPER="${1:-}"
if [ -z "$PAPER" ] && [ "${AGENT_MODE:-chat}" = "review" ]; then
  PAPER="$(find paper_rep -maxdepth 1 -type f \( -iname '*.pdf' -o -iname '*.txt' -o -iname '*.md' \) | sort | head -n 1)"
fi
if [ -z "$PAPER" ] && [ "${AGENT_MODE:-chat}" = "review" ]; then
  echo "No paper found in ./paper_rep. Pass a paper file name or add a PDF/TXT/MD file." >&2
  exit 1
fi

CHUNK_CHARS="${AGENT_CHUNK_CHARS:-3000}"
OVERLAP="${AGENT_OVERLAP:-300}"
MAX_TOKENS="${AGENT_MAX_TOKENS:-500}"
AGENT_MODE="${AGENT_MODE:-chat}"

if [ -e "$LOG_DIR" ] && [ ! -d "$LOG_DIR" ]; then
  LOG_DIR="/tmp/paper_agent_logs"
  VLLM_LOG="$LOG_DIR/vllm_${MODEL_NAME}_${PORT}.log"
  VLLM_PID_FILE="$LOG_DIR/vllm_${MODEL_NAME}_${PORT}.pid"
fi
mkdir -p "$LOG_DIR" outputs

log() {
  printf '[%s] %s\n' "$(date '+%H:%M:%S')" "$*"
}

find_vllm_bin() {
  if [ -n "${VLLM_BIN:-}" ] && [ -x "$VLLM_BIN" ]; then
    printf '%s\n' "$VLLM_BIN"
    return 0
  fi
  if command -v vllm >/dev/null 2>&1; then
    command -v vllm
    return 0
  fi
  if [ -x /mnt/e/LLM_Project/vllm_demo/.venv/bin/vllm ]; then
    printf '%s\n' /mnt/e/LLM_Project/vllm_demo/.venv/bin/vllm
    return 0
  fi
  return 1
}

port_listening() {
  ss -ltn 2>/dev/null | awk '{print $4}' | grep -Eq "(^|:)${PORT}$"
}

probe_vllm_url() {
  local candidate_base="$1"
  local candidate_health="$2"
  curl --noproxy '*' -fsS --max-time 5 "${candidate_base}/models" >/dev/null 2>&1 \
    || curl --noproxy '*' -fsS --max-time 5 "$candidate_health" >/dev/null 2>&1
}

api_ready() {
  if probe_vllm_url "$BASE_URL" "$HEALTH_URL"; then
    return 0
  fi

  if [ -n "${VLLM_BASE_URL:-}" ] || [ "$CLIENT_SELECTED" = "1" ]; then
    return 1
  fi

  local ip candidate_base candidate_health
  for ip in $(hostname -I 2>/dev/null || true); do
    candidate_base="http://${ip}:${PORT}/v1"
    candidate_health="http://${ip}:${PORT}/health"
    if probe_vllm_url "$candidate_base" "$candidate_health"; then
      BASE_URL="$candidate_base"
      HEALTH_URL="$candidate_health"
      CLIENT_SELECTED=1
      log "Using reachable vLLM client URL: $BASE_URL"
      return 0
    fi
  done

  return 1
}

model_source() {
  if [ -f "$MODEL_PATH/model.safetensors" ]; then
    printf '%s\n' "$MODEL_PATH"
  else
    printf '%s\n' "$MODEL_FALLBACK"
  fi
}

stop_stale_vllm() {
  if [ "$RESTART_STALE" != "1" ]; then
    return 0
  fi

  local pids
  pids="$(pgrep -f 'vllm serve|VLLM::EngineCore' || true)"
  if [ -z "$pids" ]; then
    return 0
  fi

  log "Found vLLM process(es) but API is not ready. Stopping stale process(es): $pids"
  kill $pids 2>/dev/null || true
  sleep 8
  pids="$(pgrep -f 'vllm serve|VLLM::EngineCore' || true)"
  if [ -n "$pids" ]; then
    log "Some vLLM process(es) did not exit, forcing stop: $pids"
    kill -9 $pids 2>/dev/null || true
  fi
}

start_vllm() {
  local vllm_bin source
  vllm_bin="$(find_vllm_bin)" || {
    echo "Cannot find vllm. Set VLLM_BIN=/path/to/vllm or add vllm to PATH." >&2
    exit 1
  }
  source="$(model_source)"

  log "Starting vLLM with model source: $source"
  log "vLLM log: $VLLM_LOG"
  : > "$VLLM_LOG"

  local serve_args=(
    "$vllm_bin" serve "$source"
    --quantization awq_marlin
    --dtype float16
    --max-model-len "$MAX_MODEL_LEN"
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
    --kv-cache-dtype "$KV_CACHE_DTYPE"
    --max-num-seqs "$MAX_NUM_SEQS"
    --attention-backend TRITON_ATTN
    --served-model-name "$MODEL_NAME"
    --host "$BIND_HOST"
    --port "$PORT"
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

  log "vLLM max_model_len=$MAX_MODEL_LEN gpu_memory_utilization=$GPU_MEMORY_UTILIZATION kv_cache_dtype=$KV_CACHE_DTYPE max_num_seqs=$MAX_NUM_SEQS prefix_caching=$ENABLE_PREFIX_CACHING chunked_prefill=$ENABLE_CHUNKED_PREFILL enforce_eager=$ENFORCE_EAGER"
  nohup "${serve_args[@]}" >"$VLLM_LOG" 2>&1 &

  echo "$!" > "$VLLM_PID_FILE"
  log "vLLM launcher PID: $(cat "$VLLM_PID_FILE")"
}

wait_for_vllm() {
  local start elapsed
  start="$(date +%s)"
  while true; do
    if api_ready; then
      log "vLLM API is ready: $BASE_URL/models"
      return 0
    fi

    elapsed="$(( $(date +%s) - start ))"
    if [ "$elapsed" -ge "$WAIT_SECONDS" ]; then
      echo
      echo "vLLM did not become ready within ${WAIT_SECONDS}s." >&2
      echo "Last 80 lines of $VLLM_LOG:" >&2
      tail -n 80 "$VLLM_LOG" >&2 || true
      echo >&2
      echo "Check whether the model fits GPU memory, or try a smaller model with:" >&2
      echo "  VLLM_MODEL_PATH=Qwen/Qwen3-0.6B VLLM_MODEL=qwen3-0.6b ./run_agent_wsl.sh" >&2
      exit 1
    fi

    if [ "$(( elapsed % 20 ))" -eq 0 ]; then
      log "Waiting for vLLM API... ${elapsed}s/${WAIT_SECONDS}s"
    fi
    sleep 2
  done
}

if [ "$FORCE_RESTART" = "1" ]; then
  log "VLLM_FORCE_RESTART=1; restarting vLLM before running agent."
  RESTART_STALE=1
  stop_stale_vllm
  start_vllm
  wait_for_vllm
elif api_ready; then
  log "Reusing running vLLM API at $BASE_URL"
else
  if ! port_listening; then
    stop_stale_vllm
    start_vllm
  else
    log "Port ${PORT} is listening but vLLM API is not ready."
    stop_stale_vllm
    if ! api_ready; then
      start_vllm
    fi
  fi
  wait_for_vllm
fi

log "Running paper agent mode: $AGENT_MODE"
source .venv/bin/activate
COMMON_ENV=(
  no_proxy="127.0.0.1,localhost,${no_proxy:-}" \
  NO_PROXY="127.0.0.1,localhost,${NO_PROXY:-}" \
  http_proxy= \
  https_proxy= \
  HTTP_PROXY= \
  HTTPS_PROXY= \
  ALL_PROXY= \
  all_proxy= \
  PYTHONUNBUFFERED=1
)

if [ "$AGENT_MODE" = "review" ]; then
  log "Reviewing paper: $PAPER"
  env "${COMMON_ENV[@]}" python -m paper_agent.cli "$PAPER" \
    --model "$MODEL_NAME" \
    --base-url "$BASE_URL" \
    --chunk-chars "$CHUNK_CHARS" \
    --overlap "$OVERLAP" \
    --max-tokens "$MAX_TOKENS" \
    --summary-mode "${SUMMARY_MODE:-quick}"
else
  log "Building local chunk index before chat."
  env "${COMMON_ENV[@]}" python -m paper_agent.cli index
  env "${COMMON_ENV[@]}" python -m paper_agent.cli chat \
    --model "$MODEL_NAME" \
    --base-url "$BASE_URL" \
    --max-tokens "$MAX_TOKENS" \
    --max-input-tokens "${AGENT_MAX_INPUT_TOKENS:-1000}"
fi
