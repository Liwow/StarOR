#!/usr/bin/env bash

trap 'PORT="${VLLM_PORT}" bash "${STOP_SERVER_SCRIPT}"' EXIT
RUN_SCRIPT="scripts/run.sh"
START_SERVER_SCRIPT="scripts/start_vllm_server.sh"
STOP_SERVER_SCRIPT="scripts/stop_vllm_server.sh"

# "data/IndustryOR_fixedV2.jsonl"
DATASET_JSONL="data/OptMATH_Bench_166.jsonl"
CHUNK_SAMPLES=8
offset=137
TOTAL_LIMIT=0   # 0 = run all samples in dataset

LOG_DIR="logs/run"
OUT_JSON="outputs/run_$(basename "${DATASET_JSONL}" .jsonl).json"

USE_VLLM=true
VLLM_MODE="server"
VLLM_PORT=8000
VLLM_GPU_MEMORY_UTILIZATION=0.4
SERVER_STOP_WAIT_SEC=20
SERVER_READY_WAIT_SEC=120
TRAIN_CUDA_VISIBLE_DEVICES="${TRAIN_CUDA_VISIBLE_DEVICES:-0}"
SERVER_CUDA_VISIBLE_DEVICES="${SERVER_CUDA_VISIBLE_DEVICES:-1}"

# pass-through run settings
NPROC_PER_NODE=1
BACKEND="trl"
MODEL_NAME_OR_PATH="$HOME/model/Qwen/Qwen3-4B-Instruct-2507"
STRUCTURE_GATE_MIN="${STRUCTURE_GATE_MIN:-0.2}"
# MODEL_NAME_OR_PATH="$HOME/model/Qwen/Qwen2.5-7B-Instruct"

count_jsonl_lines() {
  local f="$1"
  if [[ ! -f "$f" ]]; then
    echo "[ERROR] dataset not found: $f"
    exit 1
  fi
  wc -l < "$f"
}

count_completed_samples() {
  local dataset_dir="$1"
  if [[ ! -d "$dataset_dir" ]]; then
    echo 0
    return
  fi
  find "$dataset_dir" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l | tr -d ' '
}

restart_server_if_needed() {
  if [[ "${USE_VLLM}" != "true" ]] || [[ "${VLLM_MODE}" != "server" ]]; then
    return
  fi

  # 1. 统一变量名，并确保是绝对路径
  local PORT_NUM="${VLLM_PORT}" # 修正拼写
  local SESSION_NAME="vllm_run_${PORT_NUM}"
  local CONDA_ENV="or"
  local ABS_LOG_DIR=$(mkdir -p "${LOG_DIR}" && realpath "${LOG_DIR}")
  
  # 获取 conda 的绝对路径，防止脚本找不到 conda 命令
  local CONDA_EXE=$(which conda || echo "${CONDA_PREFIX}/bin/conda" || echo "conda")

  echo "[server] === Restarting vLLM Server on Port ${PORT_NUM} ==="

  echo "[server] Cleaning up old processes..."
  env -u TMUX tmux kill-session -t "${SESSION_NAME}" 2>/dev/null || true
  fuser -k "${PORT_NUM}/tcp" 2>/dev/null || true
  pkill -9 -f "trl vllm-serve" || true
  pkill -9 -f "vllm" || true
  pkill -9 -f "trl" || true
  pkill -9 -f "multiprocessing.spawn" || true
  fuser -k "${PORT_NUM}/tcp" 2>/dev/null || true
  sleep 10

  # 3. 核心启动逻辑：完全复刻你的测试命令
  echo "[server] Starting vLLM in tmux session: ${SESSION_NAME}"
  
  # 使用 env -u TMUX 解决嵌套问题
  env -u TMUX tmux new-session -d -s "${SESSION_NAME}" "${CONDA_EXE} run --no-capture-output -n ${CONDA_ENV} bash -c '
    export PYTHONUNBUFFERED=1
    export CUDA_VISIBLE_DEVICES=${SERVER_CUDA_VISIBLE_DEVICES}
    
    echo \"Starting vLLM...\"
    trl vllm-serve \
        --model ${MODEL_NAME_OR_PATH} \
        --host 0.0.0.0 \
        --port ${PORT_NUM} \
        --tensor-parallel-size 1 \
        --gpu-memory-utilization ${VLLM_GPU_MEMORY_UTILIZATION} \
        --max-model-len 16384 \
        --enable-prefix-caching True \
    2>&1 | tee ${ABS_LOG_DIR}/vllm_server.log
  '"

  # 4. 检查 Ready 状态
  echo "[server] Waiting for vLLM to initialize..."
  local start_time=$SECONDS
  while (( SECONDS - start_time < SERVER_READY_WAIT_SEC )); do
    if ! env -u TMUX tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
      echo -e "\n[server][ERROR] Tmux session died. Check log: ${ABS_LOG_DIR}/vllm_server.log"
      exit 1
    fi
    if python3 -c "import socket; s=socket.socket(); s.settimeout(1); exit(0 if s.connect_ex(('127.0.0.1', ${PORT_NUM}))==0 else 1)" 2>/dev/null; then
      echo -e "\n[server] vLLM is READY!"
      return 0
    fi
    local last_msg=$(tail -n 1 "${ABS_LOG_DIR}/vllm_server.log" 2>/dev/null || echo "Starting...")
    echo -ne "[server] Loading... ${SECONDS}s | ${last_msg: -60}\r"
    sleep 2
  done

  echo -e "\n[server][ERROR] vLLM timed out after ${SERVER_READY_WAIT_SEC}s"
  exit 1
}


mkdir -p "${LOG_DIR}" "$(dirname "${OUT_JSON}")"

DATASET_NAME=$(basename "${DATASET_JSONL}" .jsonl)
TOTAL_DATASET_LINES="$(count_jsonl_lines "${DATASET_JSONL}")"
TARGET_TOTAL="${TOTAL_DATASET_LINES}"
if (( TOTAL_LIMIT > 0 && TOTAL_LIMIT < TARGET_TOTAL )); then
  TARGET_TOTAL="${TOTAL_LIMIT}"
fi

if (( CHUNK_SAMPLES < 1 )); then
  echo "[ERROR] CHUNK_SAMPLES must be >= 1"
  exit 1
fi

echo "[server] dataset=${DATASET_JSONL} total_lines=${TOTAL_DATASET_LINES} target_total=${TARGET_TOTAL} chunk=${CHUNK_SAMPLES}"
echo "[server] LOG_DIR=${LOG_DIR} OUT_JSON=${OUT_JSON}"
echo "[server] TRAIN_CUDA_VISIBLE_DEVICES=${TRAIN_CUDA_VISIBLE_DEVICES} SERVER_CUDA_VISIBLE_DEVICES=${SERVER_CUDA_VISIBLE_DEVICES}"

restart_server_if_needed

chunk_id=$(( offset / CHUNK_SAMPLES ))
while (( offset < TARGET_TOTAL )); do
  remain=$(( TARGET_TOTAL - offset ))
  chunk_size="${CHUNK_SAMPLES}"
  if (( remain < chunk_size )); then
    chunk_size="${remain}"
  fi

  echo "[server] chunk=${chunk_id} start=${offset} limit=${chunk_size} train_cuda=${TRAIN_CUDA_VISIBLE_DEVICES}"

  CUDA_VISIBLE_DEVICES="${TRAIN_CUDA_VISIBLE_DEVICES}" \
  DATASET_JSONL="${DATASET_JSONL}" \
  DATASET_START_INDEX="${offset}" \
  DATASET_LIMIT="${chunk_size}" \
  LOG_DIR="${LOG_DIR}" \
  OUT_JSON="${OUT_JSON}" \
  USE_VLLM="${USE_VLLM}" \
  VLLM_MODE="${VLLM_MODE}" \
  NPROC_PER_NODE="${NPROC_PER_NODE}" \
  BACKEND="${BACKEND}" \
  MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH}" \
  STRUCTURE_GATE_MIN="${STRUCTURE_GATE_MIN}" \
  bash "${RUN_SCRIPT}"

  offset=$(( offset + chunk_size ))
  chunk_id=$(( chunk_id + 1 ))
  sleep 5
  if (( offset < TARGET_TOTAL )); then
    restart_server_if_needed
  fi
done

echo "[server] done. chunks=${chunk_id}, processed_samples=${TARGET_TOTAL}"