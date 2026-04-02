#!/usr/bin/env bash
set -euo pipefail

# External periodic-restart runner (minimal invasive):
# - split dataset run into chunks (by sample count)
# - restart TRL vLLM server between chunks
# - keep logs in the same LOG_DIR as normal run.sh (no chunk split)
# - auto-detect completed samples for resume
# - auto-detect completed samples for resume

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
# Extract dataset name for log directory
# Extract dataset name for log directory
LOG_DIR="logs/run"
DATASET_NAME=$(basename "${DATASET_JSONL}" .jsonl)
DATASET_DIR_="logs/run/${DATASET_NAME}"
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

# Count completed samples from log directory
count_completed_samples() {
  local DATASET_DIR_="$1"
  if [[ ! -d "$DATASET_DIR_" ]]; then
    echo 0
    return
  fi
  
  # Method 1: Count subdirectories (each sample creates one)
  local count=$(find "$DATASET_DIR_" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l | tr -d ' ')
  
  # Method 2 (fallback): Count unique task_ids from stage_events.jsonl
  if [[ "$count" -eq 0 ]] && [[ -f "$DATASET_DIR_/stage_events.jsonl" ]]; then
    count=$(grep -o '"task_id":"[^"]*"' "$DATASET_DIR_/stage_events.jsonl" 2>/dev/null | sort -u | wc -l | tr -d ' ')
  fi
  
  echo "$count"
}

# Count completed samples from log directory
count_completed_samples() {
  local DATASET_DIR_="$1"
  if [[ ! -d "$DATASET_DIR_" ]]; then
    echo 0
    return
  fi
  
  # Method 1: Count subdirectories (each sample creates one)
  local count=$(find "$DATASET_DIR_" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l | tr -d ' ')
  
  # Method 2 (fallback): Count unique task_ids from stage_events.jsonl
  if [[ "$count" -eq 0 ]] && [[ -f "$DATASET_DIR_/stage_events.jsonl" ]]; then
    count=$(grep -o '"task_id":"[^"]*"' "$DATASET_DIR_/stage_events.jsonl" 2>/dev/null | sort -u | wc -l | tr -d ' ')
  fi
  
  echo "$count"
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

# Auto-detect completed samples for resume
COMPLETED_SAMPLES=$(count_completed_samples "${DATASET_DIR_}")
if (( COMPLETED_SAMPLES > 0 )); then
  echo "[resume] detected ${COMPLETED_SAMPLES} completed samples in ${DATASET_DIR_}"
fi

# Auto-detect completed samples for resume
COMPLETED_SAMPLES=$(count_completed_samples "${DATASET_DIR_}")
if (( COMPLETED_SAMPLES > 0 )); then
  echo "[resume] detected ${COMPLETED_SAMPLES} completed samples in ${DATASET_DIR_}"
fi

echo "[periodic] dataset=${DATASET_JSONL} total_lines=${TOTAL_DATASET_LINES} target_total=${TARGET_TOTAL} chunk=${CHUNK_SAMPLES}"
echo "[periodic] LOG_DIR=${LOG_DIR} OUT_JSON=${OUT_JSON} (shared across chunks)"
echo "[periodic] completed=${COMPLETED_SAMPLES} remaining=$((TARGET_TOTAL - COMPLETED_SAMPLES))"
echo "[periodic] completed=${COMPLETED_SAMPLES} remaining=$((TARGET_TOTAL - COMPLETED_SAMPLES))"

restart_server_if_needed

# Calculate initial offset and chunk_id based on completed samples
offset=${COMPLETED_SAMPLES}
chunk_id=$((COMPLETED_SAMPLES / CHUNK_SAMPLES))
# Calculate initial offset and chunk_id based on completed samples
offset=${COMPLETED_SAMPLES}
chunk_id=$((COMPLETED_SAMPLES / CHUNK_SAMPLES))
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
echo "[periodic] done. chunks=${chunk_id}, processed_samples=${TARGET_TOTAL}"s