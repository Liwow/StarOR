#!/usr/bin/env bash
set -euo pipefail

export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth0}"
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"

MODEL="${MODEL:-$HOME/model/Qwen/Qwen3-4B-Instruct-2507}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-${MODEL}}"
VLLM_HOST="${VLLM_HOST:-0.0.0.0}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-1}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.95}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-16384}"
VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-false}"
VLLM_ENABLE_PREFIX_CACHING="${VLLM_ENABLE_PREFIX_CACHING:-true}"
SERVER_KIND="${SERVER_KIND:-openai}"
CLEAN_START="${CLEAN_START:-false}"
PORT_WAIT_SEC="${PORT_WAIT_SEC:-45}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

port_in_use() {
  local port="$1"
  if command -v lsof >/dev/null 2>&1; then
    lsof -ti tcp:"${port}" >/dev/null 2>&1 && return 0
  fi
  if command -v ss >/dev/null 2>&1; then
    ss -ltn "sport = :${port}" 2>/dev/null | tail -n +2 | grep -q . && return 0
  fi
  if command -v netstat >/dev/null 2>&1; then
    netstat -tln 2>/dev/null | awk -v p=":${port}" '$4 ~ p {found=1} END {exit !found}' && return 0
  fi
  return 1
}

wait_for_port_free() {
  local port="$1"
  local wait_sec="$2"
  local deadline=$((SECONDS + wait_sec))
  while (( SECONDS < deadline )); do
    if ! port_in_use "${port}"; then
      return 0
    fi
    sleep 1
  done
  return 1
}

echo "========================================"
echo "[vLLM] Environment Check:"
echo "  - CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "  - MODEL_NAME_OR_PATH: ${MODEL_NAME_OR_PATH}"
echo "  - HOST: ${VLLM_HOST} PORT: ${VLLM_PORT}"
echo "  - SERVER_KIND: ${SERVER_KIND}"
echo "  - NCCL_SOCKET_IFNAME: ${NCCL_SOCKET_IFNAME}"
echo "  - NCCL_DEBUG: ${NCCL_DEBUG}"
echo "  - ENFORCE_EAGER: ${VLLM_ENFORCE_EAGER}"
echo "  - ENABLE_PREFIX_CACHING: ${VLLM_ENABLE_PREFIX_CACHING}"
echo "========================================"

if [[ "${CLEAN_START}" == "true" ]] && [[ -x "scripts/stop_vllm_server.sh" ]]; then
  echo "[vLLM] clean start: stopping existing server on port ${VLLM_PORT}"
  PORT="${VLLM_PORT}" WAIT_SEC="${PORT_WAIT_SEC}" bash scripts/stop_vllm_server.sh
fi

if port_in_use "${VLLM_PORT}"; then
  echo "[vLLM][ERROR] port ${VLLM_PORT} is still occupied before startup"
  if ! wait_for_port_free "${VLLM_PORT}" "${PORT_WAIT_SEC}"; then
    echo "[vLLM][ERROR] port ${VLLM_PORT} did not become free within ${PORT_WAIT_SEC}s"
    exit 1
  fi
fi

if [[ "${SERVER_KIND}" == "openai" ]]; then
  echo "[WARN] Launching plain OpenAI API server."
  echo "[WARN] TRL --grpo-vllm-mode=server expects TRL vllm-serve endpoints."
  exec python -m vllm.entrypoints.openai.api_server     --model "${MODEL_NAME_OR_PATH}"     --host "${VLLM_HOST}"     --port "${VLLM_PORT}"     --tensor-parallel-size "${VLLM_TENSOR_PARALLEL_SIZE}"     --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION}"     --max-model-len "${VLLM_MAX_MODEL_LEN}"     --max-num-seqs 3000
fi

if ! command -v trl >/dev/null 2>&1; then
  echo "[ERROR] 'trl' command not found. Install TRL with CLI support first."
  exit 1
fi

HELP_TEXT="$(trl vllm-serve --help 2>&1 || true)"
CMD=(trl vllm-serve --model "${MODEL_NAME_OR_PATH}")

if grep -q -- '--host' <<<"${HELP_TEXT}"; then
  CMD+=(--host "${VLLM_HOST}")
elif grep -q -- '--server-host' <<<"${HELP_TEXT}"; then
  CMD+=(--server-host "${VLLM_HOST}")
elif grep -q -- '--server_host' <<<"${HELP_TEXT}"; then
  CMD+=(--server_host "${VLLM_HOST}")
fi

if grep -q -- '--port' <<<"${HELP_TEXT}"; then
  CMD+=(--port "${VLLM_PORT}")
elif grep -q -- '--server-port' <<<"${HELP_TEXT}"; then
  CMD+=(--server-port "${VLLM_PORT}")
elif grep -q -- '--server_port' <<<"${HELP_TEXT}"; then
  CMD+=(--server_port "${VLLM_PORT}")
fi

if grep -q -- '--tensor-parallel-size' <<<"${HELP_TEXT}"; then
  CMD+=(--tensor-parallel-size "${VLLM_TENSOR_PARALLEL_SIZE}")
elif grep -q -- '--tensor_parallel_size' <<<"${HELP_TEXT}"; then
  CMD+=(--tensor_parallel_size "${VLLM_TENSOR_PARALLEL_SIZE}")
fi

if grep -q -- '--gpu-memory-utilization' <<<"${HELP_TEXT}"; then
  CMD+=(--gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION}")
elif grep -q -- '--gpu_memory_utilization' <<<"${HELP_TEXT}"; then
  CMD+=(--gpu_memory_utilization "${VLLM_GPU_MEMORY_UTILIZATION}")
fi

if grep -q -- '--max-model-len' <<<"${HELP_TEXT}"; then
  CMD+=(--max-model-len "${VLLM_MAX_MODEL_LEN}")
elif grep -q -- '--max_model_len' <<<"${HELP_TEXT}"; then
  CMD+=(--max_model_len "${VLLM_MAX_MODEL_LEN}")
fi

if [[ "${VLLM_ENFORCE_EAGER}" == "true" ]]; then
  if grep -q -- '--enforce-eager' <<<"${HELP_TEXT}"; then
    CMD+=(--enforce-eager)
  elif grep -q -- '--enforce_eager' <<<"${HELP_TEXT}"; then
    CMD+=(--enforce_eager)
  fi
fi

PREFIX_CACHE_ARG=""
if grep -q -- '--enable-prefix-caching' <<<"${HELP_TEXT}"; then
  PREFIX_CACHE_ARG="--enable-prefix-caching"
elif grep -q -- '--enable_prefix_caching' <<<"${HELP_TEXT}"; then
  PREFIX_CACHE_ARG="--enable_prefix_caching"
fi

if [[ -n "${PREFIX_CACHE_ARG}" ]]; then
  if [[ "${VLLM_ENABLE_PREFIX_CACHING}" == "true" ]]; then
    CMD+=("${PREFIX_CACHE_ARG}" "True")
    echo "[INFO] Prefix caching enabled."
  else
    CMD+=("${PREFIX_CACHE_ARG}" "False")
    echo "[INFO] Prefix caching disabled."
  fi
fi

echo "[TRL-vLLM] Running command:"
printf ' %q' "${CMD[@]}"
echo

exec env   NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME}"   NCCL_DEBUG="${NCCL_DEBUG}"   "${CMD[@]}"
