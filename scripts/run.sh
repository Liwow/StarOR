#!/usr/bin/env bash
set -euo pipefail

# =====================================
# Edit Here: TTRL-OR Common Parameters
# =====================================
CUDA_VISIBLE_DEVICES="1"
BACKEND="trl"                   
MODEL_NAME_OR_PATH="/path/to/your/model"

DATASET_JSONL="data/NL4OPT.jsonl"
DATASET_LIMIT=20
LOG_DIR="logs/run"
OUT_JSON="outputs/run.json"

# MCTS
GROUP_SIZE=8
EXPAND_PER_NODE=3
SIMULATIONS_PER_NODE=4
ROLLOUT_K=3
MAX_NODES_PER_STAGE=12
MCTS_STOP_ON_REWARD_ONE=false

# Reward
CONSENSUS_WINDOW=64
ROBUSTNESS_CASES=3
ENABLE_PERTURB_REWARD=true

# GRPO (common)
GRPO_LR="3e-5"
GRPO_MAX_STEPS=2

# Generation (common)
TEMPERATURE=1.0
TOP_P=0.95
MAX_NEW_TOKENS=4096

# vLLM for TRL
USE_VLLM=true
VLLM_MODE="server"
VLLM_GPU_MEMORY_UTILIZATION=0.85
VLLM_TENSOR_PARALLEL_SIZE=1
VLLM_MAX_MODEL_LEN=16384

# Misc
SEED=7
TORCH_DTYPE="auto"
TRUST_REMOTE_CODE=false

export CUDA_VISIBLE_DEVICES
mkdir -p "$(dirname "${OUT_JSON}")" "${LOG_DIR}"

MODEL_ARG=(--model-name "${MODEL_NAME_OR_PATH}")
if [[ -d "${MODEL_NAME_OR_PATH}" ]]; then
  MODEL_ARG=(--model-path "${MODEL_NAME_OR_PATH}")
fi

CMD=(python -m ttrl_or
  --backend "${BACKEND}"
  "${MODEL_ARG[@]}"
  --seed "${SEED}"
  --dataset-jsonl "${DATASET_JSONL}"
  --dataset-limit "${DATASET_LIMIT}"
  --group-size "${GROUP_SIZE}"
  --expand-per-node "${EXPAND_PER_NODE}"
  --simulations-per-node "${SIMULATIONS_PER_NODE}"
  --rollout-k "${ROLLOUT_K}"
  --max-nodes-per-stage "${MAX_NODES_PER_STAGE}"
  --consensus-window "${CONSENSUS_WINDOW}"
  --robustness-cases "${ROBUSTNESS_CASES}"
  --grpo-lr "${GRPO_LR}"
  --grpo-max-steps "${GRPO_MAX_STEPS}"
  --temperature "${TEMPERATURE}"
  --top-p "${TOP_P}"
  --max-new-tokens "${MAX_NEW_TOKENS}"
  --torch-dtype "${TORCH_DTYPE}"
  --grpo-vllm-mode "${VLLM_MODE}"
  --grpo-vllm-gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION}"
  --grpo-vllm-tensor-parallel-size "${VLLM_TENSOR_PARALLEL_SIZE}"
  --grpo-vllm-max-model-len "${VLLM_MAX_MODEL_LEN}"
  --log-dir "${LOG_DIR}"
  --out "${OUT_JSON}"
)

if [[ "${MCTS_STOP_ON_REWARD_ONE}" == "true" ]]; then
  CMD+=(--mcts-stop-on-reward-one)
fi

if [[ "${ENABLE_PERTURB_REWARD}" != "true" ]]; then
  CMD+=(--disable-perturb-reward)
fi

if [[ "${USE_VLLM}" == "true" ]]; then
  CMD+=(--grpo-use-vllm)
fi

if [[ "${TRUST_REMOTE_CODE}" == "true" ]]; then
  CMD+=(--trust-remote-code)
fi

echo "[TTRL-OR] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[TTRL-OR] BACKEND=${BACKEND} MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH}"
echo "[TTRL-OR] DATASET_JSONL=${DATASET_JSONL} LIMIT=${DATASET_LIMIT}"

echo "[TTRL-OR] Running command:"
printf ' %q' "${CMD[@]}"
echo

"${CMD[@]}"