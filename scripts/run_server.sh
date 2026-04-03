#!/usr/bin/env bash
set -euo pipefail

# External periodic-restart runner:
# - split dataset run into chunks
# - restart TRL vLLM server between chunks
# - keep logs in the same LOG_DIR as normal run.sh
# - resume by skipping completed samples

# ==========================
# Edit Here (internal config only)
# ==========================
RUN_SCRIPT="scripts/run.sh"
START_SERVER_SCRIPT="scripts/start_vllm_server.sh"
STOP_SERVER_SCRIPT="scripts/stop_vllm_server.sh"

DATASET_JSONL="data/IndustryOR_fixedV2.jsonl"
CHUNK_SAMPLES=8
TOTAL_LIMIT=0   # 0 = run all samples in dataset

LOG_DIR="logs/run"
OUT_JSON="outputs/run_$(basename "${DATASET_JSONL}" .jsonl).json"

USE_VLLM=true
VLLM_MODE="server"
VLLM_PORT=8000
SERVER_STOP_WAIT_SEC=45
SERVER_READY_WAIT_SEC=90
TRAIN_CUDA_VISIBLE_DEVICES="${TRAIN_CUDA_VISIBLE_DEVICES:-0}"
SERVER_CUDA_VISIBLE_DEVICES="${SERVER_CUDA_VISIBLE_DEVICES:-1}"

# pass-through run settings
NPROC_PER_NODE=1
BACKEND="trl"
MODEL_NAME_OR_PATH="$HOME/model/Qwen/Qwen3-4B-Instruct-2507"
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

port_in_use() {
  local port="$1"
  if command -v lsof >/dev/null 2>&1; then
    lsof -ti tcp:"${port}" >/dev/null 2>&1 && return 0
  fi
  if command -v ss >/dev/null 2>&1; then
    ss -ltn "sport = :${port}" 2>/dev/null | tail -n +2 | grep -q . && return 0
  fi
  if command -v netstat >/dev/null 2>&1; then
    netstat -tln 2>/dev/null | awk -v p=":${port}" '$4 ~ p {found=1} END {exit !found}' && return 0
  fi
  return 1
}

wait_for_port_free() {
  local port="$1"
  local wait_sec="$2"
  local deadline=$((SECONDS + wait_sec))
  while (( SECONDS < deadline )); do
    if ! port_in_use "${port}"; then
      return 0
    fi
    sleep 1
  done
  return 1
}

wait_for_server_ready() {
  local host="$1"
  local port="$2"
  local wait_sec="$3"
  local deadline=$((SECONDS + wait_sec))
  while (( SECONDS < deadline )); do
    if python - <<PY >/dev/null 2>&1
import socket
s = socket.socket()
s.settimeout(1.0)
try:
    s.connect(("${host}", int("${port}")))
except Exception:
    raise SystemExit(1)
else:
    raise SystemExit(0)
finally:
    try:
        s.close()
    except Exception:
        pass
PY
    then
      return 0
    fi
    sleep 1
  done
  return 1
}

restart_server_if_needed() {
  if [[ "${USE_VLLM}" != "true" ]] || [[ "${VLLM_MODE}" != "server" ]]; then
    return
  fi

  echo "[server] restarting vLLM server on CUDA_VISIBLE_DEVICES=${SERVER_CUDA_VISIBLE_DEVICES} port=${VLLM_PORT}"
  PORT="${VLLM_PORT}" WAIT_SEC="${SERVER_STOP_WAIT_SEC}" bash "${STOP_SERVER_SCRIPT}"

  if ! wait_for_port_free "${VLLM_PORT}" "${SERVER_STOP_WAIT_SEC}"; then
    echo "[server][ERROR] port ${VLLM_PORT} still occupied after stop"
    exit 1
  fi

  CUDA_VISIBLE_DEVICES="${SERVER_CUDA_VISIBLE_DEVICES}"   MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH}"   VLLM_PORT="${VLLM_PORT}"   CLEAN_START="true"   PORT_WAIT_SEC="${SERVER_STOP_WAIT_SEC}"   bash "${START_SERVER_SCRIPT}" > "${LOG_DIR}/vllm_server.log" 2>&1 &
  local spid=$!
  disown "${spid}" || true

  if ! wait_for_server_ready "127.0.0.1" "${VLLM_PORT}" "${SERVER_READY_WAIT_SEC}"; then
    echo "[server][ERROR] vLLM server did not become ready within ${SERVER_READY_WAIT_SEC}s"
    exit 1
  fi
  echo "[server] vLLM ready on 127.0.0.1:${VLLM_PORT} (pid=${spid})"
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

offset=0
chunk_id=0
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
  bash "${RUN_SCRIPT}"

  offset=$(( offset + chunk_size ))
  chunk_id=$(( chunk_id + 1 ))
  sleep 5
  if (( offset < TARGET_TOTAL )); then
    restart_server_if_needed
  fi
done

echo "[server] done. chunks=${chunk_id}, processed_samples=${TARGET_TOTAL}"
