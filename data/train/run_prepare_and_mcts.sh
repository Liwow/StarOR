#!/usr/bin/env bash
set -euo pipefail

# Run from repo root by default.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

INPUT_JSONL="${INPUT_JSONL:-data/train/train_data.jsonl}"
GEN_OUTPUT_JSONL="${GEN_OUTPUT_JSONL:-data/train/train_data.type_python.jsonl}"
MCTS_OUTPUT_JSONL="${MCTS_OUTPUT_JSONL:-data/train/train_data.mcts_stage.jsonl}"

PARALLEL="${PARALLEL:-4}"
MODEL_NAME="${MODEL_NAME:-${OPENAI_MODEL:-}}"
API_KEY_ENV_NAME="${API_KEY_ENV_NAME:-IDEALAB_API_KEY}"
BASE_URL_ENV_NAME="${BASE_URL_ENV_NAME:-IDEALAB_BASE_URL}"
TEMPERATURE="${TEMPERATURE:-0.4}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-120}"
RUN_TIMEOUT="${RUN_TIMEOUT:-30}"
VERIFY_MODE="${VERIFY_MODE:-run-if-available}" # syntax | run | run-if-available
MCTS_MODE="${MCTS_MODE:-true}"                  # true | false

if [[ -z "${MODEL_NAME}" ]]; then
  echo "ERROR: MODEL_NAME is empty. Set MODEL_NAME or OPENAI_MODEL." >&2
  exit 1
fi

if [[ -z "${!API_KEY_ENV_NAME:-}" ]]; then
  echo "ERROR: API key env '${API_KEY_ENV_NAME}' is empty." >&2
  exit 1
fi

if [[ -z "${!BASE_URL_ENV_NAME:-}" ]]; then
  echo "ERROR: Base URL env '${BASE_URL_ENV_NAME}' is empty." >&2
  exit 1
fi

echo "[step1] generate Type+python with parallel=${PARALLEL}"
python data/train/01_prepare_train_data.py generate \
  --input "${INPUT_JSONL}" \
  --output "${GEN_OUTPUT_JSONL}" \
  --model "${MODEL_NAME}" \
  --api-key-env "${API_KEY_ENV_NAME}" \
  --base-url-env "${BASE_URL_ENV_NAME}" \
  --temperature "${TEMPERATURE}" \
  --request-timeout "${REQUEST_TIMEOUT}" \
  --run-timeout "${RUN_TIMEOUT}" \
  --verification-mode "${VERIFY_MODE}" \
  --parallel "${PARALLEL}" \
  --resume

echo "[step2] convert to mcts/full-prompt format (mcts=${MCTS_MODE})"
python data/train/02_mcts_format.py \
  --input "${GEN_OUTPUT_JSONL}" \
  --output "${MCTS_OUTPUT_JSONL}" \
  --mcts "${MCTS_MODE}" \
  --resume

echo "done: ${MCTS_OUTPUT_JSONL}"

