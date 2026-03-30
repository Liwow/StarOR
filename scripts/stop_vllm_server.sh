#!/usr/bin/env bash
set -euo pipefail

# Stop TRL/vLLM server processes robustly.
# Usage:
#   PORT=8000 scripts/stop_vllm_server.sh
# Optional:
#   WAIT_SEC=20 GRACE_SEC=5 PORT=8000 scripts/stop_vllm_server.sh

PORT="${PORT:-8000}"
WAIT_SEC="${WAIT_SEC:-20}"
GRACE_SEC="${GRACE_SEC:-5}"

collect_port_pids() {
  local out=""

  if command -v lsof >/dev/null 2>&1; then
    out+="$(lsof -ti tcp:"${PORT}" 2>/dev/null || true)"$'\n'
  fi

  if command -v ss >/dev/null 2>&1; then
    out+="$(ss -ltnp "sport = :${PORT}" 2>/dev/null | grep -oE 'pid=[0-9]+' | cut -d= -f2 || true)"$'\n'
  fi

  if command -v netstat >/dev/null 2>&1; then
    out+="$(netstat -tlnp 2>/dev/null | awk -v p=":${PORT}" '$4 ~ p {print $7}' | cut -d/ -f1 | grep -E '^[0-9]+$' || true)"$'\n'
  fi

  if command -v fuser >/dev/null 2>&1; then
    out+="$(fuser "${PORT}/tcp" 2>/dev/null || true)"$'\n'
  fi

  printf '%s\n' "${out}" | tr ' ' '\n' | grep -E '^[0-9]+$' | sort -u || true
}

collect_pattern_pids() {
  local out=""

  if command -v pgrep >/dev/null 2>&1; then
    out+="$(pgrep -f 'trl\.scripts\.vllm_serve' || true)"$'\n'
    out+="$(pgrep -f 'trl vllm-serve' || true)"$'\n'
    out+="$(pgrep -f 'vllm\.entrypoints\.openai\.api_server' || true)"$'\n'
    out+="$(pgrep -f 'python.*vllm_serve' || true)"$'\n'
  else
    out+="$(ps -ef 2>/dev/null | grep -E 'trl\.scripts\.vllm_serve|trl vllm-serve|vllm\.entrypoints\.openai\.api_server|python.*vllm_serve' | grep -v grep | awk '{print $2}' || true)"$'\n'
  fi

  printf '%s\n' "${out}" | grep -E '^[0-9]+$' | sort -u || true
}

collect_all_pids() {
  {
    collect_port_pids
    collect_pattern_pids
  } | grep -E '^[0-9]+$' | sort -u || true
}

kill_pids() {
  local sig="$1"
  shift
  local pids=("$@")
  if ((${#pids[@]} == 0)); then
    return 0
  fi
  kill "-${sig}" "${pids[@]}" 2>/dev/null || true
}

is_alive() {
  local pid="$1"
  kill -0 "${pid}" 2>/dev/null
}

wait_until_gone() {
  local deadline=$((SECONDS + WAIT_SEC))
  while ((SECONDS < deadline)); do
    local alive=0
    while IFS= read -r pid; do
      [[ -z "${pid}" ]] && continue
      if is_alive "${pid}"; then
        alive=1
        break
      fi
    done < <(collect_all_pids)

    if ((alive == 0)); then
      return 0
    fi
    sleep 1
  done
  return 1
}

echo "[vLLM-stop] target port=${PORT} wait=${WAIT_SEC}s grace=${GRACE_SEC}s"

mapfile -t initial_pids < <(collect_all_pids)
if ((${#initial_pids[@]} == 0)); then
  echo "[vLLM-stop] no matching process found"
  exit 0
fi

echo "[vLLM-stop] matched pids: ${initial_pids[*]}"

kill_pids TERM "${initial_pids[@]}"
sleep "${GRACE_SEC}"

mapfile -t remaining_pids < <(collect_all_pids)
if ((${#remaining_pids[@]} > 0)); then
  echo "[vLLM-stop] force kill pids: ${remaining_pids[*]}"
  kill_pids KILL "${remaining_pids[@]}"
fi

if wait_until_gone; then
  echo "[vLLM-stop] done"
  exit 0
fi

echo "[vLLM-stop][ERROR] processes still alive after kill attempts"
collect_all_pids | sed 's/^/[vLLM-stop][ERROR] alive pid=/'
exit 1
