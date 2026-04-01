#!/usr/bin/env bash
set -euo pipefail

# External periodic-restart runner (minimal invasive):
# - split dataset run into chunks (by sample count)
# - restart TRL vLLM server between chunks
# - keep logs in the same LOG_DIR as normal run.sh (no chunk split)

# ==========================
# Edit Here (internal config only)
# ==========================
RUN_SCRIPT="scripts/run.sh"
START_SERVER_SCRIPT="scripts/start_vllm_server.sh"
STOP_SERVER_SCRIPT="scripts/stop_vllm_server.sh"
# data/OptMATH_Bench_166.jsonl
# data/IndustryOR_fixedV2.jsonl
DATASET_JSONL="data/OptMATH_Bench_166.jsonl"
CHUNK_SAMPLES=10        # restart server every N samples
TOTAL_LIMIT=0            # 0 = run all samples in dataset

# Keep same output paths across chunks (no split)
LOG_DIR="logs/run"
OUT_JSON="outputs/run.json"

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
  PORT="${VLLM_PORT}" bash "${STOP_SERVER_SCRIPT}"
  sleep 2
  bash "${START_SERVER_SCRIPT}" &
  local spid=$!
  disown "${spid}" || true
  sleep 4
}

mkdir -p "${LOG_DIR}" "$(dirname "${OUT_JSON}")"

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
echo "[periodic] LOG_DIR=${LOG_DIR} OUT_JSON=${OUT_JSON} (shared across chunks)"

restart_server_if_needed

offset=0
chunk_id=0
while (( offset < TARGET_TOTAL )); do
  remain=$(( TARGET_TOTAL - offset ))
  chunk_size="${CHUNK_SAMPLES}"
  if (( remain < chunk_size )); then
    chunk_size="${remain}"
  fi

  echo "[periodic] chunk=${chunk_id} start=${offset} limit=${chunk_size}"

  DATASET_JSONL="${DATASET_JSONL}" \
  DATASET_START_INDEX="${offset}" \
  DATASET_LIMIT="${chunk_size}" \
  LOG_DIR="${LOG_DIR}" \
  OUT_JSON="${OUT_JSON}" \
  USE_VLLM="${USE_VLLM}" \
  VLLM_MODE="${VLLM_MODE}" \
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  NPROC_PER_NODE="${NPROC_PER_NODE}" \
  BACKEND="${BACKEND}" \
  MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH}" \
  bash "${RUN_SCRIPT}"

  offset=$(( offset + chunk_size ))
  chunk_id=$(( chunk_id + 1 ))
  sleep 5
  if (( offset < TARGET_TOTAL )); then
    restart_server_if_needed
  fi
done

# PORT="${VLLM_PORT}" bash "${STOP_SERVER_SCRIPT}"
echo "[periodic] done. chunks=${chunk_id}, processed_samples=${TARGET_TOTAL}"