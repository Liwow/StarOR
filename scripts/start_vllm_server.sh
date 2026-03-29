#!/usr/bin/env bash
set -euo pipefail

export VLLM_USE_V1=0
export NCCL_SOCKET_IFNAME=eth0
export NCCL_DEBUG=INFO
export CUDA_LAUNCH_BLOCKING=1


# ==================================
# Edit Here: TRL vLLM Server Params
# ==================================
CUDA_VISIBLE_DEVICES="1"
MODEL_NAME_OR_PATH="$HOME/model/Qwen/Qwen2.5-7B-Instruct"

VLLM_HOST="0.0.0.0"
VLLM_PORT=8000
VLLM_TENSOR_PARALLEL_SIZE=1
VLLM_GPU_MEMORY_UTILIZATION=0.65
VLLM_MAX_MODEL_LEN=16384

SERVER_KIND="trl"

export CUDA_VISIBLE_DEVICES

# 🔧 [新增] 打印关键环境变量，方便调试确认
echo "========================================"
echo "[vLLM] Environment Check:"
echo "  - CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "  - MODEL_NAME_OR_PATH: ${MODEL_NAME_OR_PATH}"
echo "  - HOST: ${VLLM_HOST} PORT: ${VLLM_PORT}"
echo "  - SERVER_KIND: ${SERVER_KIND}"
echo "  - VLLM_USE_V1: ${VLLM_USE_V1} (v0 engine)"
echo "  - NCCL_SOCKET_IFNAME: ${NCCL_SOCKET_IFNAME}"
echo "  - NCCL_DEBUG: ${NCCL_DEBUG}"
echo "========================================"

if [[ "${SERVER_KIND}" == "openai" ]]; then
  echo "[WARN] You are launching plain OpenAI API server."
  echo "[WARN] TRL --grpo-vllm-mode=server expects TRL vllm-serve endpoints (e.g. init_communicator)."
  python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_NAME_OR_PATH}" \
    --host "${VLLM_HOST}" \
    --port "${VLLM_PORT}" \
    --tensor-parallel-size "${VLLM_TENSOR_PARALLEL_SIZE}" \
    --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION}" \
    --max-model-len "${VLLM_MAX_MODEL_LEN}"
  exit 0
fi

if ! command -v trl >/dev/null 2>&1; then
  echo "[ERROR] 'trl' command not found. Install TRL with CLI support first."
  echo "[ERROR] Example: pip install trl"
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

echo "[TRL-vLLM] Running command:"
printf ' %q' "${CMD[@]}"
echo
echo "----------------------------------------"

# 🔧 [关键修改] 执行命令时显式前置环境变量，确保 100% 传递给子进程
# 即使 export 失效，这里也能兜底
VLLM_USE_V1=0 \
NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME}" \
NCCL_DEBUG="${NCCL_DEBUG}" \
CUDA_LAUNCH_BLOCKING=1 \
"${CMD[@]}"