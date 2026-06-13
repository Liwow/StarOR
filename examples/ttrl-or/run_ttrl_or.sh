#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[StarOR] run_ttrl_or.sh is retained for compatibility; forwarding to run_staror.sh." >&2
exec bash "${SCRIPT_DIR}/run_staror.sh" "$@"
