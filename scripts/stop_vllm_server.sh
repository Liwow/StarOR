#!/usr/bin/env bash
set -euo pipefail

PORT="${PORT:-8000}"
WAIT_SEC="${WAIT_SEC:-20}"
GRACE_SEC="${GRACE_SEC:-5}"

collect_port_pids() {
  local out=""
  if command -v lsof >/dev/null 2>&1; then
    out+="$(lsof -ti tcp:${PORT} 2>/dev/null || true)"$'
'
  fi
  if command -v ss >/dev/null 2>&1; then
    out+="$(ss -ltnp "sport = :${PORT}" 2>/dev/null | grep -oE 'pid=[0-9]+' | cut -d= -f2 || true)"$'
'
  fi
  if command -v netstat >/dev/null 2>&1; then
    out+="$(netstat -tlnp 2>/dev/null | awk -v p=":${PORT}" '$4 ~ p {print $7}' | cut -d/ -f1 | grep -E '^[0-9]+$' || true)"$'
'
  fi
  if command -v fuser >/dev/null 2>&1; then
    out+="$(fuser "${PORT}/tcp" 2>/dev/null || true)"$'
'
  fi
  printf '%s
' "${out}" | tr ' ' '
' | grep -E '^[0-9]+$' | sort -u || true
}

collect_pattern_pids() {
  local out=""
  if command -v pgrep >/dev/null 2>&1; then
    out+="$(pgrep -f 'trl\.scripts\.vllm_serve' || true)"$'
'
    out+="$(pgrep -f 'trl vllm-serve' || true)"$'
'
    out+="$(pgrep -f 'vllm\.entrypoints\.openai\.api_server' || true)"$'
'
    out+="$(pgrep -f 'python.*vllm_serve' || true)"$'
'
  else
    out+="$(ps -ef 2>/dev/null | grep -E 'trl\.scripts\.vllm_serve|trl vllm-serve|vllm\.entrypoints\.openai\.api_server|python.*vllm_serve' | grep -v grep | awk '{print $2}' || true)"$'
'
  fi
  printf '%s
' "${out}" | grep -E '^[0-9]+$' | sort -u || true
}

collect_all_pids() {
  {
    collect_port_pids
    collect_pattern_pids
  } | grep -E '^[0-9]+$' | sort -u || true
}

collect_descendant_pids() {
  local roots=("$@")
  if ((${#roots[@]} == 0)); then
    return 0
  fi
  if ! command -v pgrep >/dev/null 2>&1; then
    printf '%s
' "${roots[@]}"
    return 0
  fi
  local seen=""
  local queue=("${roots[@]}")
  local all=("${roots[@]}")
  while ((${#queue[@]} > 0)); do
    local current="${queue[0]}"
    queue=("${queue[@]:1}")
    [[ " ${seen} " == *" ${current} "* ]] && continue
    seen+=" ${current}"
    mapfile -t children < <(pgrep -P "${current}" 2>/dev/null || true)
    if ((${#children[@]} > 0)); then
      for child in "${children[@]}"; do
        if [[ -n "${child}" ]] && [[ ! " ${seen} " == *" ${child} "* ]]; then
          all+=("${child}")
          queue+=("${child}")
        fi
      done
    fi
  done
  printf '%s
' "${all[@]}" | grep -E '^[0-9]+$' | sort -u || true
}

collect_all_targets() {
  mapfile -t roots < <(collect_all_pids)
  if ((${#roots[@]} == 0)); then
    return 0
  fi
  collect_descendant_pids "${roots[@]}"
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
    done < <(collect_all_targets)
    if ((alive == 0)); then
      return 0
    fi
    sleep 1
  done
  return 1
}

echo "[vLLM-stop] target port=${PORT} wait=${WAIT_SEC}s grace=${GRACE_SEC}s"
mapfile -t initial_pids < <(collect_all_targets)
if ((${#initial_pids[@]} == 0)); then
  echo "[vLLM-stop] no matching process found"
  exit 0
fi

echo "[vLLM-stop] matched pids: ${initial_pids[*]}"
ps -fp "${initial_pids[@]}" || true
kill_pids TERM "${initial_pids[@]}"
sleep "${GRACE_SEC}"

mapfile -t remaining_pids < <(collect_all_targets)
if ((${#remaining_pids[@]} > 0)); then
  echo "[vLLM-stop] force kill pids: ${remaining_pids[*]}"
  kill_pids KILL "${remaining_pids[@]}"
fi

if wait_until_gone; then
  echo "[vLLM-stop] done"
  exit 0
fi

echo "[vLLM-stop][ERROR] processes still alive after kill attempts"
collect_all_targets | sed 's/^/[vLLM-stop][ERROR] alive pid=/'
exit 1
