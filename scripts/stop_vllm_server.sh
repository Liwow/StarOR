#!/usr/bin/env bash
set -euo pipefail

# Stop TRL/vLLM server processes.
# Usage:
#   PORT=8000 scripts/stop_vllm_server.sh

PORT="${PORT:-8000}"

echo "[vLLM-stop] target port=${PORT}"

# 1) Try kill by listening port.
if command -v lsof >/dev/null 2>&1; then
  PIDS="$(lsof -ti tcp:"${PORT}" || true)"
  if [[ -n "${PIDS}" ]]; then
    echo "[vLLM-stop] kill by port: ${PIDS}"
    kill -9 ${PIDS} || true
  fi
fi

# 2) Fallback by process keywords.
if command -v pkill >/dev/null 2>&1; then
  pkill -f "trl.scripts.vllm_serve" || true
  pkill -f "vllm.entrypoints.openai.api_server" || true
fi

echo "[vLLM-stop] done"
