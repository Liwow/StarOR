# 定义变量（根据你之前的配置）
SESSION_NAME="test_vllm_server"
CONDA_ENV="or"
MODEL_PATH="/home/ljj516475/model/Qwen/Qwen3-4B-Instruct-2507"
GPU_ID="1"  # 你的 SERVER_CUDA_VISIBLE_DEVICES
PORT="8000"

# 强杀旧的会话（防止重名）
tmux kill-session -t "$SESSION_NAME" 2>/dev/null || true

# 核心启动测试命令
tmux new-session -d -s "$SESSION_NAME" "conda run --no-capture-output -n $CONDA_ENV bash -c '
    export PYTHONUNBUFFERED=1
    export CUDA_VISIBLE_DEVICES=$GPU_ID
    echo \"Starting vLLM in tmux...\"
    trl vllm-serve \
        --model $MODEL_PATH \
        --host 0.0.0.0 \
        --port $PORT \
        --tensor-parallel-size 1 \
        --gpu-memory-utilization 0.45 \
        --max-model-len 16384 \
        --enable-prefix-caching True \
    2>&1 | tee vllm_test.log
'"

echo "命令已发送至 tmux 会话: $SESSION_NAME"
echo "你可以输入: tmux attach -t $SESSION_NAME 来查看实时日志"
