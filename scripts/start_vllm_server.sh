#!/usr/bin/env bash
set -euo pipefail

# ==================================
# Edit Here: TRL vLLM Server Params
# ==================================
CUDA_VISIBLE_DEVICES="1"
MODEL_NAME_OR_PATH="$HOME/model/Qwen/Qwen3-4B-Instruct-2507"
# MODEL_NAME_OR_PATH="$HOME/model/Qwen/Qwen2.5-7B-Instruct"

VLLM_HOST="0.0.0.0"
VLLM_PORT=8000
VLLM_TENSOR_PARALLEL_SIZE=1
VLLM_GPU_MEMORY_UTILIZATION=0.90
VLLM_MAX_MODEL_LEN=16384

# server kind:
# - trl: required for TRL GRPO server mode
# - openai: plain vLLM OpenAI API server (will NOT work with TRL server mode)

# NOTE: For TRL LoRA training, server mode may fail with NCCL in some notebook/distributed envs.
# If you hit NCCL socket/remote-process errors, prefer run.sh with VLLM_MODE=colocate or enable fallback.
SERVER_KIND="trl"

export CUDA_VISIBLE_DEVICES

echo "[vLLM] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[vLLM] MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH}"
echo "[vLLM] HOST=${VLLM_HOST} PORT=${VLLM_PORT}"
echo "[vLLM] SERVER_KIND=${SERVER_KIND}"

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

"${CMD[@]}"
