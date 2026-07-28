#!/bin/bash

# ================= 参数配置 =================
# 核心脚本文件名
SCRIPT_NAME="tools/best_of_n_vllm_infer.py"

MODEL_NAME="qwen3-4b-instruct"

# API 配置
BASE_URL="http://127.0.0.1:8000/v1"
API_KEY="EMPTY"

# 推理参数
# 用法：bash scripts/run_bestofn.sh 8
# 也支持：N_VALUE=8 bash scripts/run_bestofn.sh
N_VALUE="${1:-${N_VALUE:-32}}"  # Best-of-N，默认 32
if ! [[ "$N_VALUE" =~ ^[1-9][0-9]*$ ]]; then
    echo "错误：N_VALUE 必须是正整数，当前值: $N_VALUE" >&2
    exit 2
fi
PARALLEL_SIZE=1        # 并发请求数
TEMPERATURE=0.7
MAX_TOKENS=8192        # 足够长以容纳 Gurobi 代码
EXEC_TIMEOUT=30         # 每个代码运行最长 30 秒
MODEL_PARAMS_B=4.0       # 用于估算 FLOPs：2 * 参数量 * token 数
CODE_PYTHON="${CODE_PYTHON:-/home/ljj/miniconda3/envs/ljj/bin/python}"

# 日志目录
LOG_DIR="outputs_bon/best_of_n_logs_${N_VALUE}"

# 数据集列表
# DATASETS=(
#     "data/ComplexOR.jsonl"
#     "data/IndustryOR_fixedV2.jsonl"
#     "data/OptMATH_Bench_166.jsonl"
#     "data/MAMO_ComplexLP_fixed.jsonl"
#     "data/MAMO_EasyLP_fixed.jsonl"
#     "data/NL4OPT.jsonl"
#     "data/OptiBench.jsonl"
#     "data/NL4LP.jsonl"
# )
DATASETS=(
    "data/OptMATH_Bench_166.jsonl"
)

# ================= 执行逻辑 =================

# 创建日志目录
mkdir -p "$LOG_DIR"

echo "开始批量评估任务..."
echo "模型路径: $MODEL_NAME"
echo "并发数: $PARALLEL_SIZE, N: $N_VALUE"

for DS_PATH in "${DATASETS[@]}"; do
    if [ ! -f "$DS_PATH" ]; then
        echo "跳过：找不到文件 $DS_PATH"
        continue
    fi

    echo "------------------------------------------------"
    echo "正在处理数据集: $DS_PATH"
    echo "------------------------------------------------"

    # 执行 Python 推理脚本
    python "$SCRIPT_NAME" \
        --input "$DS_PATH" \
        --model "$MODEL_NAME" \
        --base-url "$BASE_URL" \
        --api-key "$API_KEY" \
        --n "$N_VALUE" \
        --temperature "$TEMPERATURE" \
        --max-tokens "$MAX_TOKENS" \
        --model-params-billions "$MODEL_PARAMS_B" \
        --parallel "$PARALLEL_SIZE" \
        --exec-timeout "$EXEC_TIMEOUT" \
        --code-python "$CODE_PYTHON" \
        --log-dir "$LOG_DIR" \
        --save-raw-text

    echo "数据集 $(basename "$DS_PATH") 处理完成。"
done

echo "所有任务执行完毕！结果保存在: $LOG_DIR"
