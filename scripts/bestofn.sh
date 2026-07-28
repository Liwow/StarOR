set -Eeuo pipefail

cd "$HOME/code/StarOR"

stop_vllm() {
    cd "$HOME/model"
    bash vllm.sh stop
}
trap stop_vllm EXIT

for n in 2 4 8 16 32 40; do
    echo "===== Running Best-of-${n} ====="

    bash scripts/run_bestofn.sh "$n"

    summary="outputs_bon/best_of_n_logs_${n}/voted_OptMATH_Bench_166_qwen3-4b-instruct.summary.json"

    python - "$summary" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path, encoding="utf-8") as f:
    summary = json.load(f)

completed = summary["total_tasks"]
if completed != 166:
    raise SystemExit(f"{path}: expected 166 successful tasks, got {completed}")

print(
    f"validated: tasks={completed}, "
    f"accuracy={summary['overall_accuracy']:.2%}, "
    f"tokens={summary['total_tokens']:,}"
)
PY
done