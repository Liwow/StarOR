#!/usr/bin/env bash
set -euo pipefail

# Stop TRL/vLLM server processes.
# Usage:
#   PORT=8000 scripts/stop_vllm_server.sh

PORT="${PORT:-8000}"

echo "[vLLM-stop] target port=${PORT}"

if command -v lsof >/dev/null 2>&1; then
  PIDS=$(lsof -ti tcp:"${PORT}")
elif command -v ss >/dev/null 2>&1; then
  PIDS=$(ss -tlnp "sport = :${PORT}" | grep -oP '(?<=pid=)\d+' | sort -u)
elif command -v netstat >/dev/null 2>&1; then
  PIDS=$(netstat -tlnp | grep ":${PORT} " | awk '{print $7}' | cut -d'/' -f1 | grep -E '^[0-9]+$')
elif command -v fuser >/dev/null 2>&1; then
  PIDS=$(fuser ${PORT}/tcp 2>/dev/null)
fi

if [[ -n "${PIDS}" ]]; then
  echo "[vLLM-stop] kill by port: ${PIDS}"
  kill -9 ${PIDS} || true
fi


# 2) Fallback by process keywords.
if command -v pkill >/dev/null 2>&1; then
  pkill -f "trl.scripts.vllm_serve" || true
  pkill -f "vllm.entrypoints.openai.api_server" || true
fi

echo "[vLLM-stop] done"
