#!/bin/bash

# ================= 参数配置 =================
# 核心脚本文件名（请确保指向你保存 Reflexion 代码的文件）
SCRIPT_NAME="tools/reflection_vllm_infer.py"

# 模型名称/路径
MODEL_NAME="/home/ljj516475/model/Qwen/Qwen3-4B-Instruct-2507"

# API 配置
BASE_URL="http://127.0.0.1:8000/v1"
API_KEY="EMPTY"

# --- Reflexion 核心推理参数 ---
MAX_TRIALS=3            # 每个问题最多允许反思/重试的次数 (建议 3-5)
PARALLEL_SIZE=50        # 同时处理多少个题目 (并发数)
TEMPERATURE=0.4         # 反思模式建议低温度，保持逻辑稳定性
MAX_TOKENS=5000         # 单次生成的最大长度
EXEC_TIMEOUT=40         # 运行 Gurobi 代码的超时时间 (秒)
TIMEOUT=300             # 请求 API 的整体超时时间 (秒)

# 日志与结果输出目录
LOG_DIR="outputs/reflexion_logs"

# 数据集列表 (根据你的实际路径调整)
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

echo "===================================================="
echo "开始 Reflexion (自我反思) 批量评估任务..."
echo "模型: $MODEL_NAME"
echo "并发数: $PARALLEL_SIZE, 最大尝试次数: $MAX_TRIALS"
echo "===================================================="

for DS_PATH in "${DATASETS[@]}"; do
    if [ ! -f "$DS_PATH" ]; then
        echo "跳过：找不到文件 $DS_PATH"
        continue
    fi

    # 提取文件名作为日志标识
    DS_NAME=$(basename "$DS_PATH" .jsonl)
    
    echo "------------------------------------------------"
    echo "正在处理数据集: $DS_NAME"
    echo "路径: $DS_PATH"
    echo "------------------------------------------------"

    # 执行 Python Reflexion 脚本
    # 注意：确保 Python 脚本中的参数名与此处一致
    python "$SCRIPT_NAME" \
        --input "$DS_PATH" \
        --model "$MODEL_NAME" \
        --base-url "$BASE_URL" \
        --api-key "$API_KEY" \
        --max-trials "$MAX_TRIALS" \
        --temperature "$TEMPERATURE" \
        --max-tokens "$MAX_TOKENS" \
        --parallel "$PARALLEL_SIZE" \
        --exec-timeout "$EXEC_TIMEOUT" \
        --log-dir "$LOG_DIR" \
        --timeout "$TIMEOUT"

    echo ">>> 数据集 $DS_NAME 处理完成。"
done

echo "===================================================="
echo "所有 Reflexion 任务执行完毕！"
echo "结果保存在: $LOG_DIR"
echo "===================================================="
