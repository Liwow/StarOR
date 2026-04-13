#!/bin/bash

# ================= 参数配置 =================
# 核心脚本文件名
SCRIPT_NAME="tools/best_of_n_vllm_infer.py"

MODEL_NAME="/home/ljj516475/model/Qwen/Qwen3-4B-Instruct-2507"

# API 配置
BASE_URL="http://127.0.0.1:8000/v1"
API_KEY="EMPTY"

# 推理参数
N_VALUE=16              # Best-of-N
PARALLEL_SIZE=15        # 并发请求数
TEMPERATURE=0.7
MAX_TOKENS=5000         # 足够长以容纳 Gurobi 代码
EXEC_TIMEOUT=30         # 每个代码运行最长 30 秒

# 日志目录
LOG_DIR="outputs/best_of_n_logs"

# 数据集列表
DATASETS=(
    "data/ComplexOR.jsonl"
    "data/IndustryOR_fixedV2.jsonl"
    "data/OptMATH_Bench_166.jsonl"
    "data/MAMO_ComplexLP_fixed.jsonl"
    "data/MAMO_EasyLP_fixed.jsonl"
    "data/NL4OPT.jsonl"
    "data/OptiBench.jsonl"
    "data/NL4LP.jsonl"
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
        --parallel "$PARALLEL_SIZE" \
        --exec-timeout "$EXEC_TIMEOUT" \
        --log-dir "$LOG_DIR" \
        --save-raw-text

    echo "数据集 $(basename "$DS_PATH") 处理完成。"
done

echo "所有任务执行完毕！结果保存在: $LOG_DIR"
