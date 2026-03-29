#!/usr/bin/env bash
set -euo pipefail

# External periodic-restart runner (minimal invasive):
# - split dataset run into chunks (by sample count)
# - restart TRL vLLM server between chunks

# ==========================
# Edit Here (no CLI/env needed)
# ==========================
RUN_SCRIPT="scripts/run.sh"
START_SERVER_SCRIPT="scripts/start_vllm_server.sh"
STOP_SERVER_SCRIPT="scripts/stop_vllm_server.sh"

DATASET_JSONL="data/IndustryOR_fixedV2.jsonl"
CHUNK_SAMPLES=10          # restart server every N samples
TOTAL_LIMIT=0            # 0 = run all samples in dataset

BASE_LOG_ROOT="logs/periodic"
BASE_OUT_ROOT="outputs/periodic"

USE_VLLM=true
VLLM_MODE="server"
VLLM_PORT=8000

# pass-through run settings
CUDA_VISIBLE_DEVICES="0"
NPROC_PER_NODE=1
BACKEND="trl"
MODEL_NAME_OR_PATH="$HOME/model/Qwen/Qwen2.5-7B-Instruct"

# ==========================
# Helpers
# ==========================
count_jsonl_lines() {
  local f="$1"
  if [[ ! -f "$f" ]]; then
    echo "[ERROR] dataset not found: $f"
    exit 1
  fi
  wc -l < "$f"
}

restart_server_if_needed() {
  if [[ "${USE_VLLM}" != "true" ]] || [[ "${VLLM_MODE}" != "server" ]]; then
    return
  fi

  echo "[periodic] restarting vLLM server (port=${VLLM_PORT}) ..."
  PORT="${VLLM_PORT}" bash "${STOP_SERVER_SCRIPT}" || true
  bash "${START_SERVER_SCRIPT}" &
  local spid=$!
  disown "${spid}" || true
  sleep 20
}

mkdir -p "${BASE_LOG_ROOT}" "${BASE_OUT_ROOT}"

TOTAL_DATASET_LINES="$(count_jsonl_lines "${DATASET_JSONL}")"
if (( TOTAL_LIMIT > 0 )); then
  TARGET_TOTAL="${TOTAL_LIMIT}"
  if (( TARGET_TOTAL > TOTAL_DATASET_LINES )); then
    TARGET_TOTAL="${TOTAL_DATASET_LINES}"
  fi
else
  TARGET_TOTAL="${TOTAL_DATASET_LINES}"
fi

if (( CHUNK_SAMPLES < 1 )); then
  echo "[ERROR] CHUNK_SAMPLES must be >= 1"
  exit 1
fi

echo "[periodic] dataset=${DATASET_JSONL} total_lines=${TOTAL_DATASET_LINES} target_total=${TARGET_TOTAL} chunk=${CHUNK_SAMPLES}"

restart_server_if_needed

offset=0
chunk_id=0
while (( offset < TARGET_TOTAL )); do
  remain=$(( TARGET_TOTAL - offset ))
  chunk_size="${CHUNK_SAMPLES}"
  if (( remain < chunk_size )); then
    chunk_size="${remain}"
  fi

  run_log_dir="${BASE_LOG_ROOT}/chunk_${chunk_id}"
  run_out_json="${BASE_OUT_ROOT}/chunk_${chunk_id}.json"

  echo "[periodic] chunk=${chunk_id} start=${offset} limit=${chunk_size}"

  DATASET_JSONL="${DATASET_JSONL}" \
  DATASET_START_INDEX="${offset}" \
  DATASET_LIMIT="${chunk_size}" \
  LOG_DIR="${run_log_dir}" \
  OUT_JSON="${run_out_json}" \
  USE_VLLM="${USE_VLLM}" \
  VLLM_MODE="${VLLM_MODE}" \
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  NPROC_PER_NODE="${NPROC_PER_NODE}" \
  BACKEND="${BACKEND}" \
  MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH}" \
  bash "${RUN_SCRIPT}"

  offset=$(( offset + chunk_size ))
  chunk_id=$(( chunk_id + 1 ))

  if (( offset < TARGET_TOTAL )); then
    restart_server_if_needed
  fi
done

echo "[periodic] done. chunks=${chunk_id}, processed_samples=${TARGET_TOTAL}"