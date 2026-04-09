#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# ==========================
# Best-of-N default settings
# ==========================
INPUT="${INPUT:-data/IndustryOR_fixedV2.jsonl}"
MODEL="${MODEL:-${OPENAI_MODEL:-}}"
BASE_URL="${BASE_URL:-${VLLM_BASE_URL:-http://127.0.0.1:8000/v1}}"
API_KEY="${API_KEY:-${OPENAI_API_KEY:-EMPTY}}"

N="${N:-8}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.95}"
MAX_TOKENS="${MAX_TOKENS:-128}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-120}"
PARALLEL="${PARALLEL:-4}"

START_INDEX="${START_INDEX:-0}"
LIMIT="${LIMIT:-0}"
ID_KEY="${ID_KEY:-id}"
QUESTION_KEYS="${QUESTION_KEYS:-input,en_question,question,prompt,task}"
ANSWER_KEYS="${ANSWER_KEYS:-answer,en_answer,gt,ground_truth,output}"

VOTE_REL_TOL="${VOTE_REL_TOL:-1e-4}"
VOTE_ABS_TOL="${VOTE_ABS_TOL:-1e-6}"
GT_REL_TOL="${GT_REL_TOL:-1e-4}"
GT_ABS_TOL="${GT_ABS_TOL:-1e-6}"

LOG_DIR="${LOG_DIR:-outputs/best_of_n_logs}"
RESUME="${RESUME:-true}"
SAVE_RAW_TEXT="${SAVE_RAW_TEXT:-false}"

is_true() {
  local raw="${1:-}"
  local lowered="${raw,,}"
  [[ "$lowered" == "true" || "$lowered" == "1" || "$lowered" == "yes" || "$lowered" == "y" ]]
}

if [[ -z "${MODEL}" ]]; then
  echo "[bestofn][ERROR] MODEL is empty. Set MODEL or OPENAI_MODEL."
  exit 1
fi

CMD=(
  python scripts/best_of_n_vllm_infer.py
  --input "${INPUT}"
  --model "${MODEL}"
  --base-url "${BASE_URL}"
  --api-key "${API_KEY}"
  --n "${N}"
  --temperature "${TEMPERATURE}"
  --top-p "${TOP_P}"
  --max-tokens "${MAX_TOKENS}"
  --request-timeout "${REQUEST_TIMEOUT}"
  --parallel "${PARALLEL}"
  --start-index "${START_INDEX}"
  --limit "${LIMIT}"
  --id-key "${ID_KEY}"
  --question-keys "${QUESTION_KEYS}"
  --answer-keys "${ANSWER_KEYS}"
  --vote-rel-tol "${VOTE_REL_TOL}"
  --vote-abs-tol "${VOTE_ABS_TOL}"
  --gt-rel-tol "${GT_REL_TOL}"
  --gt-abs-tol "${GT_ABS_TOL}"
  --log-dir "${LOG_DIR}"
)

if is_true "${RESUME}"; then
  CMD+=(--resume)
else
  CMD+=(--no-resume)
fi

if is_true "${SAVE_RAW_TEXT}"; then
  CMD+=(--save-raw-text)
fi

echo "[bestofn] INPUT=${INPUT}"
echo "[bestofn] MODEL=${MODEL}"
echo "[bestofn] BASE_URL=${BASE_URL}"
echo "[bestofn] N=${N} PARALLEL=${PARALLEL} LOG_DIR=${LOG_DIR}"
echo "[bestofn] Running command:"
printf ' %q' "${CMD[@]}"
echo

"${CMD[@]}"
