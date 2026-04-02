#!/usr/bin/env bash
set -euo pipefail
# export VLLM_USE_V1=0
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL
export NCCL_SOCKET_IFNAME=eth0
# export NCCL_P2P_DISABLE=1
# export NCCL_IB_DISABLE=1

# =====================================
# Edit Here: TTRL-OR Common Parameters
# =====================================
export CUDA_VISIBLE_DEVICES=0
# Set 1 for single-card, 2/4 for multi-card (must be <= visible GPU count)
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
MASTER_PORT="${MASTER_PORT:-29500}"

BACKEND="${BACKEND:-trl}"
# MODEL_NAME_OR_PATH="$HOME/model/Qwen/Qwen3-4B-Instruct-2507"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-$HOME/model/Qwen/Qwen2.5-7B-Instruct}"

DATASET_JSONL="${DATASET_JSONL:-data/IndustryOR_fixedV2.jsonl}"
DATASET_START_INDEX="${DATASET_START_INDEX:-0}"
DATASET_LIMIT="${DATASET_LIMIT:-0}"
RESUME_SKIP_COMPLETED="${RESUME_SKIP_COMPLETED:-true}"
LOG_DIR="${LOG_DIR:-logs/run}"
OUT_JSON="${OUT_JSON:-outputs/run.json}"

# MCTS
MAX_ITERATIONS=18
C_PUCT=1.414
MCTS_STOP_ON_REWARD_ONE=false
SOLVERLLM_COMPARE_MODE=True

# Reward
GLOBAL_CONSENSUS_REL_TOL=0.005
REWARD_CLUSTER_SCOPE="${REWARD_CLUSTER_SCOPE:-local}"
ROBUSTNESS_CASES=3
ENABLE_R3_REWARD=false

# GRPO (common)
GRPO_LR="1e-4"
GRPO_TRAIN_EPOCHS=1
GRPO_CLIP_EPSILON=0.3
GRPO_CLIP_EPSILON_HIGH=-1   # -1 means disabled
GRPO_NUM_GENERATIONS=8
GRPO_GENERATION_BATCH_SIZE=0  # 0 = auto align to num_generations
GRPO_MAX_COMPLETION_LEN=4096

# Generation (common)
TEMPERATURE=1.0
TOP_P=0.95
MAX_NEW_TOKENS=4096

# vLLM for TRL
USE_VLLM="${USE_VLLM:-true}"
VLLM_MODE="${VLLM_MODE:-server}" # server | colocate
VLLM_GPU_MEMORY_UTILIZATION=0.55
VLLM_TENSOR_PARALLEL_SIZE=1
VLLM_MAX_MODEL_LEN=10240
VLLM_MAX_NUM_BATCHED_TOKENS=32768
GRPO_VLLM_ENABLE_SLEEP_MODE=false


# LoRA
LORA_R=8
LORA_ALPHA=16
LORA_DROPOUT=0.05
LORA_BIAS="none"
LORA_TARGET_MODULES="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"

# Misc
SEED=7
TORCH_DTYPE="auto"
TRUST_REMOTE_CODE=false

mkdir -p "$(dirname "${OUT_JSON}")" "${LOG_DIR}"

visible_gpu_count() {
  local ids="${1// /}"
  if [[ -z "$ids" ]]; then
    echo 0
    return
  fi
  awk -F',' '{print NF}' <<< "$ids"
}

is_true() {
  local raw="${1:-}"
  local lowered="${raw,,}"
  [[ "$lowered" == "true" || "$lowered" == "1" || "$lowered" == "yes" || "$lowered" == "y" ]]
}

if (( NPROC_PER_NODE == 1 && VLLM_TENSOR_PARALLEL_SIZE > 1 )); then
  echo "[WARN] NPROC_PER_NODE=1 but VLLM_TENSOR_PARALLEL_SIZE=${VLLM_TENSOR_PARALLEL_SIZE}."
  echo "[WARN] Runtime may auto-fallback TP to 1."
fi

if (( NPROC_PER_NODE > 1 )) && is_true "${USE_VLLM}" && [[ "${VLLM_MODE}" == "colocate" ]]; then
  echo "[WARN] multi-process training with vLLM colocate is unstable and may duplicate GPU memory/processes."
  echo "[WARN] Auto-disabling USE_VLLM for this run. Use VLLM_MODE=server if you need external vLLM."
  USE_VLLM=false
fi


MODEL_ARG=(--model-name "${MODEL_NAME_OR_PATH}")
if [[ -d "${MODEL_NAME_OR_PATH}" ]]; then
  MODEL_ARG=(--model-path "${MODEL_NAME_OR_PATH}")
fi

BASE_CMD=(-m ttrl_or
  --backend "${BACKEND}"
  "${MODEL_ARG[@]}"
  --seed "${SEED}"
  --dataset-jsonl "${DATASET_JSONL}"
  --dataset-start-index "${DATASET_START_INDEX}"
  --dataset-limit "${DATASET_LIMIT}"
  --max-iterations "${MAX_ITERATIONS}"
  --c-puct "${C_PUCT}"
  --global-consensus-rel-tol "${GLOBAL_CONSENSUS_REL_TOL}"
  --reward-cluster-scope "${REWARD_CLUSTER_SCOPE}"
  --robustness-cases "${ROBUSTNESS_CASES}"
  --grpo-lr "${GRPO_LR}"
  --grpo-train-epochs "${GRPO_TRAIN_EPOCHS}"
  --grpo-clip-epsilon "${GRPO_CLIP_EPSILON}"
  --grpo-clip-epsilon-high "${GRPO_CLIP_EPSILON_HIGH}"
  --grpo-num-generations "${GRPO_NUM_GENERATIONS}"
  --grpo-generation-batch-size "${GRPO_GENERATION_BATCH_SIZE}"
  --grpo-max-completion-len "${GRPO_MAX_COMPLETION_LEN}"
  --temperature "${TEMPERATURE}"
  --top-p "${TOP_P}"
  --max-new-tokens "${MAX_NEW_TOKENS}"
  --lora-r "${LORA_R}"
  --lora-alpha "${LORA_ALPHA}"
  --lora-dropout "${LORA_DROPOUT}"
  --lora-bias "${LORA_BIAS}"
  --lora-target-modules "${LORA_TARGET_MODULES}"
  --torch-dtype "${TORCH_DTYPE}"
  --grpo-vllm-mode "${VLLM_MODE}"
  --grpo-vllm-gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION}"
  --grpo-vllm-tensor-parallel-size "${VLLM_TENSOR_PARALLEL_SIZE}"
  --grpo-vllm-max-model-len "${VLLM_MAX_MODEL_LEN}"
  --grpo-vllm-max-num-batched-tokens "${VLLM_MAX_NUM_BATCHED_TOKENS}"
  --log-dir "${LOG_DIR}"
  --out "${OUT_JSON}"
)

if is_true "${MCTS_STOP_ON_REWARD_ONE}"; then
  BASE_CMD+=(--mcts-stop-on-reward-one)
fi

if is_true "${SOLVERLLM_COMPARE_MODE}"; then
  BASE_CMD+=(--solverllm-compare-mode)
fi

if ! is_true "${ENABLE_R3_REWARD}"; then
  BASE_CMD+=(--disable-r3-reward)
fi

if is_true "${RESUME_SKIP_COMPLETED}"; then
  BASE_CMD+=(--dataset-resume-skip-completed)
else
  BASE_CMD+=(--no-dataset-resume-skip-completed)
fi
if is_true "${USE_VLLM}"; then
  BASE_CMD+=(--grpo-use-vllm)
fi

if is_true "${GRPO_VLLM_ENABLE_SLEEP_MODE}"; then
  BASE_CMD+=(--grpo-vllm-enable-sleep-mode)
fi

if is_true "${TRUST_REMOTE_CODE}"; then
  BASE_CMD+=(--trust-remote-code)
fi

if (( NPROC_PER_NODE > 1 )); then
  CMD=(torchrun --standalone --nproc_per_node "${NPROC_PER_NODE}" --master_port "${MASTER_PORT}" "${BASE_CMD[@]}")
else
  CMD=(python "${BASE_CMD[@]}")
fi

echo "[TTRL-OR] NPROC_PER_NODE=${NPROC_PER_NODE} BACKEND=${BACKEND}"
echo "[TTRL-OR] MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH}"
echo "[TTRL-OR] DATASET_JSONL=${DATASET_JSONL} START=${DATASET_START_INDEX} LIMIT=${DATASET_LIMIT}"
echo "[TTRL-OR] SOLVERLLM_COMPARE_MODE=${SOLVERLLM_COMPARE_MODE}"

echo "[TTRL-OR] Running command:"
printf ' %q' "${CMD[@]}"
echo

"${CMD[@]}"
