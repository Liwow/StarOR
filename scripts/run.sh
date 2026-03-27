#!/usr/bin/env bash
set -euo pipefail

# =====================================
# Edit Here: TTRL-OR Common Parameters
# =====================================
CUDA_VISIBLE_DEVICES="0"
BACKEND="trl"
MODEL_NAME_OR_PATH="$HOME/model/Qwen/Qwen3-4B-Instruct-2507"

DATASET_JSONL="data/NL4OPT.jsonl"
DATASET_LIMIT=20
LOG_DIR="logs/run"
OUT_JSON="outputs/run.json"

# MCTS (global-leaf selection)
MAX_ITERATIONS=16
C_PUCT=1.4
MCTS_STOP_ON_REWARD_ONE=true

# Reward
GLOBAL_CONSENSUS_MIN_POOL=3
GLOBAL_CONSENSUS_REL_TOL=0.005
ROBUSTNESS_CASES=3
ENABLE_R3_REWARD=false

# GRPO (common)
GRPO_LR="3e-5"
GRPO_NUM_GENERATIONS=4
GRPO_GENERATION_BATCH_SIZE=0  # 0 = auto align to num_generations
GRPO_MAX_COMPLETION_LEN=2048

# Generation (common)
TEMPERATURE=1.0
TOP_P=0.95
MAX_NEW_TOKENS=2048

# vLLM for TRL
USE_VLLM=true
VLLM_MODE="server" # colocate
VLLM_GPU_MEMORY_UTILIZATION=0.75
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
  --max-iterations "${MAX_ITERATIONS}"
  --c-puct "${C_PUCT}"
  --global-consensus-min-pool "${GLOBAL_CONSENSUS_MIN_POOL}"
  --global-consensus-rel-tol "${GLOBAL_CONSENSUS_REL_TOL}"
  --robustness-cases "${ROBUSTNESS_CASES}"
  --grpo-lr "${GRPO_LR}"
  --grpo-num-generations "${GRPO_NUM_GENERATIONS}"
  --grpo-generation-batch-size "${GRPO_GENERATION_BATCH_SIZE}"
  --grpo-max-completion-len "${GRPO_MAX_COMPLETION_LEN}"
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

if [[ "${ENABLE_R3_REWARD}" != "true" ]]; then
  CMD+=(--disable-r3-reward)
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
