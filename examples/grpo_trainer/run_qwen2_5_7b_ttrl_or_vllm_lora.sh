set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/../ttrl-or/run_qwen2_5_7b_ttrl_or_vllm_lora.sh" "$@"
