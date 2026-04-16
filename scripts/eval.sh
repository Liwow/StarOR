cd ~
WORK_DIR="code/TTRL-OR"

# TTRL false
log_dir="code/TTRL-OR/outputs/logs_k-8_TTRL-false_stage-update-true_r3-false_DYNAMIC-R-true_refine-true_repair-2"

# log_dir="code/TTRL-OR/outputs/logs_k-8_TTRL-false_stage-update-true_r3-true_DYNAMIC-R-true_refine-true_repair-2"
# log_dir="code/TTRL-OR/outputs/logs_TTRL-false_r3-true_refine-true_repair-2_codeGATE-true"

# TTRL true
# log_dir="code/TTRL-OR/outputs/logs_k-8_TTRL-true_stage-update-true_r3-true_DYNAMIC-R-true_refine-true_repair-2"
# log_dir="code/TTRL-OR/outputs/logs_TTRL-true_r3-true_refine-true_repair-2_codeGATE-true"


# log_dir="code/TTRL-OR/outputs/ttrl_or_logs"
# log_dir="code/TTRL-OR/logs/run/model_default_Qwen3-4B-Instruct-2507_old_cot"



limit="${1:-0}" #0=all

python ${WORK_DIR}/tools/eval_acc.py --log-root ${log_dir} --limit ${limit}