#!/usr/bin/env bash
set -euo pipefail

# export VLLM_USE_V1="${VLLM_USE_V1:-1}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth0}"
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"

# ==================================
# Edit Here: TRL vLLM Server Params
# ==================================
MODEL_NAME_OR_PATH="$HOME/model/Qwen/Qwen2.5-7B-Instruct"

VLLM_HOST="0.0.0.0"
VLLM_PORT=8000
VLLM_TENSOR_PARALLEL_SIZE=1
VLLM_GPU_MEMORY_UTILIZATION=0.45

VLLM_MAX_MODEL_LEN=10240
VLLM_ENFORCE_EAGER=false
VLLM_ENABLE_PREFIX_CACHING=true

SERVER_KIND="trl"   # trl | openai
CLEAN_START=false

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

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

if [[ "${SERVER_KIND}" == "openai" ]]; then
  echo "[WARN] Launching plain OpenAI API server."
  echo "[WARN] TRL --grpo-vllm-mode=server expects TRL vllm-serve endpoints."
  python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_NAME_OR_PATH}" \
    --host "${VLLM_HOST}" \
    --port "${VLLM_PORT}" \
    --tensor-parallel-size "${VLLM_TENSOR_PARALLEL_SIZE}" \
    --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION}" \
    --max-model-len "${VLLM_MAX_MODEL_LEN} \
    --max-num-seqs 32"
  exit 0
fi

if [[ "${CLEAN_START}" == "true" ]]; then
  if [[ -x "scripts/stop_vllm_server.sh" ]]; then
    echo "[vLLM] clean start: stopping existing server on port ${VLLM_PORT}"
    PORT="${VLLM_PORT}" scripts/stop_vllm_server.sh
  fi
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


NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME}" \
NCCL_DEBUG="${NCCL_DEBUG}" \
"${CMD[@]}"

