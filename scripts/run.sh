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
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
# Launch config
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
MASTER_PORT="${MASTER_PORT:-29500}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MACHINE_RANK="${MACHINE_RANK:-0}"
NUM_MACHINES="${NUM_MACHINES:-1}"
ACCELERATE_MIXED_PRECISION="${ACCELERATE_MIXED_PRECISION:-no}"
ACCELERATE_CONFIG_FILE="${ACCELERATE_CONFIG_FILE:-}"

BACKEND="${BACKEND:-trl}"
# MODEL_NAME_OR_PATH="$HOME/model/Qwen/Qwen3-4B-Instruct-2507"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-$HOME/model/Qwen/Qwen3-4B-Instruct-2507}"

DATASET_JSONL="${DATASET_JSONL:-data/IndustryOR_fixedV2.jsonl}"
DATASET_START_INDEX="${DATASET_START_INDEX:-0}"
DATASET_LIMIT="${DATASET_LIMIT:-0}"
RESUME_SKIP_COMPLETED="${RESUME_SKIP_COMPLETED:-true}"
LOG_DIR="${LOG_DIR:-logs/run}"
OUT_JSON="${OUT_JSON:-outputs/run.json}"

# MCTS
MAX_ITERATIONS=20
C_PUCT=1.414
ENABLE_PRIOR=true
MCTS_STOP_ON_REWARD_ONE=false
SOLVERLLM_COMPARE_MODE=false #true TODO

# Reward
GLOBAL_CONSENSUS_REL_TOL=0.005
REWARD_CLUSTER_SCOPE="${REWARD_CLUSTER_SCOPE:-local}"
ROBUSTNESS_CASES=3
ENABLE_R3_REWARD=false
STRUCTURE_GATE_MIN="${STRUCTURE_GATE_MIN:-0.2}"

# GRPO (common)
GRPO_LR="1e-4"
GRPO_GROUP_SIZE=8
GRPO_KL="0.0"
GRPO_SYNC_REF_MODEL=true
GRPO_REF_MODEL_SYNC_STEPS=5
GRPO_REF_MODEL_MIXUP_ALPHA=0.6
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
VLLM_GPU_MEMORY_UTILIZATION=0.4
VLLM_TENSOR_PARALLEL_SIZE=1
VLLM_MAX_MODEL_LEN=16384
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


is_true() {
  local raw="${1:-}"
  local lowered="${raw,,}"
  [[ "$lowered" == "true" || "$lowered" == "1" || "$lowered" == "yes" || "$lowered" == "y" ]]
}



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
  --structure-gate-min "${STRUCTURE_GATE_MIN}"
  --grpo-lr "${GRPO_LR}"
  --grpo-group-size "${GRPO_GROUP_SIZE}"
  --grpo-kl "${GRPO_KL}"
  --grpo-ref-model-sync-steps "${GRPO_REF_MODEL_SYNC_STEPS}"
  --grpo-ref-model-mixup-alpha "${GRPO_REF_MODEL_MIXUP_ALPHA}"
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

if ! is_true "${ENABLE_PRIOR}"; then
  BASE_CMD+=(--disable-prior)
fi

if is_true "${MCTS_STOP_ON_REWARD_ONE}"; then
  BASE_CMD+=(--mcts-stop-on-reward-one)
fi

if is_true "${SOLVERLLM_COMPARE_MODE}"; then
  BASE_CMD+=(--solverllm-compare-mode)
fi

if ! is_true "${ENABLE_R3_REWARD}"; then
  BASE_CMD+=(--disable-r3-reward)
fi

if is_true "${GRPO_SYNC_REF_MODEL}"; then
  BASE_CMD+=(--grpo-sync-ref-model)
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
  CMD=(accelerate launch --multi_gpu)
  if [[ -n "${ACCELERATE_CONFIG_FILE}" && -f "${ACCELERATE_CONFIG_FILE}" ]]; then
    CMD+=(--config_file "${ACCELERATE_CONFIG_FILE}")
  fi
  CMD+=(
    --num_processes "${NPROC_PER_NODE}"
    --num_machines "${NUM_MACHINES}"
    --machine_rank "${MACHINE_RANK}"
    --main_process_ip "${MASTER_ADDR}"
    --main_process_port "${MASTER_PORT}"
    --mixed_precision "${ACCELERATE_MIXED_PRECISION}"
    "${BASE_CMD[@]}"
  )
else
  CMD=(python "${BASE_CMD[@]}")
fi

echo "[TTRL-OR] NPROC_PER_NODE=${NPROC_PER_NODE} BACKEND=${BACKEND}"
echo "[TTRL-OR] MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT} NUM_MACHINES=${NUM_MACHINES} MACHINE_RANK=${MACHINE_RANK}"
echo "[TTRL-OR] MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH}"
echo "[TTRL-OR] DATASET_JSONL=${DATASET_JSONL} START=${DATASET_START_INDEX} LIMIT=${DATASET_LIMIT}"
echo "[TTRL-OR] SOLVERLLM_COMPARE_MODE=${SOLVERLLM_COMPARE_MODE}"
echo "[TTRL-OR] ACCELERATE_MIXED_PRECISION=${ACCELERATE_MIXED_PRECISION} ACCELERATE_CONFIG_FILE=${ACCELERATE_CONFIG_FILE:-<none>}"
echo "[TTRL-OR] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} USE_VLLM=${USE_VLLM} VLLM_MODE=${VLLM_MODE} VLLM_TP=${VLLM_TENSOR_PARALLEL_SIZE}"

echo "[TTRL-OR] Running command:"
printf ' %q' "${CMD[@]}"
echo

"${CMD[@]}"


