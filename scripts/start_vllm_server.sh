#!/usr/bin/env bash
set -euo pipefail

# ==================================
# Edit Here: vLLM Common Parameters
# ==================================
CUDA_VISIBLE_DEVICES="1"
MODEL_NAME_OR_PATH="/path/to/your/model"

VLLM_HOST="0.0.0.0"
VLLM_PORT=8000
VLLM_TENSOR_PARALLEL_SIZE=1
VLLM_GPU_MEMORY_UTILIZATION=0.90
VLLM_MAX_MODEL_LEN=16384

export CUDA_VISIBLE_DEVICES

echo "[vLLM] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[vLLM] MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH}"
echo "[vLLM] HOST=${VLLM_HOST} PORT=${VLLM_PORT}"

python -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_NAME_OR_PATH}" \
  --host "${VLLM_HOST}" \
  --port "${VLLM_PORT}" \
  --tensor-parallel-size "${VLLM_TENSOR_PARALLEL_SIZE}" \
  --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION}" \
  --max-model-len "${VLLM_MAX_MODEL_LEN}"