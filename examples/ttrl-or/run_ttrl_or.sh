set -x
export CUDA_VISIBLE_DEVICES=1
# export RAY_DEBUG_POST_MORTEM=1
DATA_ROOT="${HOME}/code/TTRL-OR/data"
SAMPLE_RUN=false
SAMPLE_SEED=42
sample_size=200
MCTS_CLUSTER_UPDATE=true
CODE_REFINE=true
CODE_REPAIR=2
CODE_ENTRY_SECOND_ATTEMPT=true
CODE_ENTRY_SAME_CLUSTER_SUPPRESS_WEIGHT=0.5
AUTO_COMPLETE=true
FILTER_ROLLOUT=false

USE_TTRL=true

STAGE_UPDATE=true

r3_reward=true
r4_reward=true
DYNAMIC_REWARD=true
EARLY_WEIGHT='[0.2,0.5,0.2,0.1]'
MID_WEIGHT='[0.5,0.3,0.1,0.1]'
FINAL_WEIGHT='[0.6,0.2,0.1,0.1]'

multi_reward=true
k=8

log_dir="outputs/logs_k=${k}_f=${FILTER_ROLLOUT}_ac=${AUTO_COMPLETE}_TTRL-${USE_TTRL}_stage-update=${STAGE_UPDATE}_r3-${r3_reward}_DYNAMIC-R=${DYNAMIC_REWARD}_multi-R=${multi_reward}_refine-${CODE_REFINE}_repair-${CODE_REPAIR}"
EXTRA_ARGS=()
LORA_RANK=16
LORA_ALPHA=32

while [[ $# -gt 0 ]]; do
    case "$1" in
        --sample_run)
            SAMPLE_RUN="${2:-false}"
            shift 2
            ;;
        --sample_seed)
            SAMPLE_SEED="${2:-0}"
            shift 2
            ;;
        --sample_run=*)
            SAMPLE_RUN="${1#*=}"
            shift
            ;;
        --sample_seed=*)
            SAMPLE_SEED="${1#*=}"
            shift
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

if [[ "${multi_reward}" != "true" ]]; then
    echo "using major voting reward"
    EARLY_WEIGHT='[1.0,0,0,0]'
    MID_WEIGHT='[1.0,0,0,0]'
    FINAL_WEIGHT='[1.0,0,0,0]'
    r3_reward=false
    r4_reward=false
fi

SAMPLE_RUN="$(echo "${SAMPLE_RUN}" | tr '[:upper:]' '[:lower:]')"
if [[ "${SAMPLE_RUN}" != "true" && "${SAMPLE_RUN}" != "false" ]]; then
    echo "[run_ttrl_or.sh] --sample_run must be true or false, got: ${SAMPLE_RUN}"
    exit 1
fi
log_dir="${log_dir//=/\-}"
log_dir="${log_dir// /_}"

if [[ "$(echo "${USE_TTRL}" | tr '[:upper:]' '[:lower:]')" == "false" ]]; then
    # Pure-MCTS mode: disable LoRA adapter construction to avoid unnecessary vLLM LoRA paths.
    LORA_RANK=0
    LORA_ALPHA=0
fi

# dataset="NL4OPT.jsonl"
# dataset="NL4LP.jsonl"
# dataset="MAMO_ComplexLP_fixed.jsonl"
dataset="IndustryOR_fixedV2.jsonl"
# dataset="OptMATH_Bench_166.jsonl"

# dataset="MAMO_EasyLP_fixed.jsonl" #sample
# dataset="OptiBench.jsonl"

DATA_PATH="$DATA_ROOT/$dataset"
# For multiple datasets, uncomment and edit the list below.
DATA_SETS="[$DATA_ROOT/IndustryOR_fixedV2.jsonl, $DATA_ROOT/MAMO_ComplexLP_fixed.jsonl,$DATA_ROOT/OptMATH_Bench_166.jsonl, $DATA_ROOT/NL4OPT.jsonl, $DATA_ROOT/NL4LP.jsonl, $DATA_ROOT/ComplexOR.jsonl]"

TRAIN_FILES="$DATA_PATH"

# model_name='Qwen/Qwen2.5-7B-Instruct'
model_name='Qwen/Qwen3-4B-Instruct-2507'
MODEL_NAME_OR_PATH="${HOME}/model/${model_name}"
# MODEL_NAME_OR_PATH="${oss_path}/checkpoint/sft_or_qwen3_4b_int_cp513_v1"

python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.ttrl_or.enable=True \
    algorithm.ttrl_or.log_dir="${log_dir}" \
    algorithm.ttrl_or.mcts.max_iterations=12 \
    algorithm.ttrl_or.mcts.c_puct=1.414 \
    algorithm.ttrl_or.mcts.enable_prior=True \
    algorithm.ttrl_or.mcts.blocked_sibling_soft_weight=0.6 \
    algorithm.ttrl_or.mcts.code_refine=${CODE_REFINE} \
    algorithm.ttrl_or.mcts.code_repair=${CODE_REPAIR} \
    algorithm.ttrl_or.mcts.code_entry_second_attempt=${CODE_ENTRY_SECOND_ATTEMPT} \
    algorithm.ttrl_or.mcts.code_entry_same_cluster_suppress_weight=${CODE_ENTRY_SAME_CLUSTER_SUPPRESS_WEIGHT} \
    algorithm.ttrl_or.mcts.mcts_cluster_update=${MCTS_CLUSTER_UPDATE} \
    algorithm.ttrl_or.mcts.auto_complete=${AUTO_COMPLETE} \
    algorithm.ttrl_or.mcts.filter_rollout=${FILTER_ROLLOUT} \
    algorithm.ttrl_or.grpo.use_ttrl=${USE_TTRL} \
    algorithm.ttrl_or.grpo.stage_update=${STAGE_UPDATE} \
    algorithm.ttrl_or.mcts.solverllm_compare_mode=False \
    algorithm.ttrl_or.reward.enable_r3_reward=${r3_reward} \
    algorithm.ttrl_or.reward.enable_r4_reward=${r4_reward} \
    algorithm.ttrl_or.reward.dynamic_reward=${DYNAMIC_REWARD} \
    algorithm.ttrl_or.reward.early_weight="${EARLY_WEIGHT}" \
    algorithm.ttrl_or.reward.mid_weight="${MID_WEIGHT}" \
    algorithm.ttrl_or.reward.final_weight="${FINAL_WEIGHT}" \
    algorithm.ttrl_or.reward.gurobi_time_limit_sec=30 \
    algorithm.ttrl_or.reward.robustness_cases=3 \
    algorithm.ttrl_or.reward.cluster_scope=local \
    algorithm.ttrl_or.reward.r1_obj_scale_fail_multiplier=0.5 \
    algorithm.ttrl_or.reward.structure_gate_min=1.0 \
    algorithm.ttrl_or.dataset.sample_run=${SAMPLE_RUN} \
    algorithm.ttrl_or.dataset.sample_seed=${SAMPLE_SEED} \
    algorithm.ttrl_or.dataset.sample_size=${sample_size} \
    algorithm.ttrl_or.dataset.resume_skip_completed=True \
    algorithm.ttrl_or.backend.reset_lora_on_begin_episode=True \
    data.train_files="$TRAIN_FILES" \
    data.val_files="$TRAIN_FILES" \
    data.custom_cls.path='pkg://verl.utils.dataset.ttrl_or_dataset' \
    data.custom_cls.name=TTRLORDataset \
    data.ttrl_or_max_numeric_features=32 \
    data.ttrl_or_key_param_top_k=16 \
    data.train_batch_size=1 \
    data.max_prompt_length=3000 \
    data.max_response_length=7000 \
    data.filter_overlong_prompts=False \
    data.truncation='right' \
    data.shuffle=False \
    actor_rollout_ref.model.path=${MODEL_NAME_OR_PATH} \
    actor_rollout_ref.model.use_shm=False \
    actor_rollout_ref.model.target_modules=all-linear \
    actor_rollout_ref.model.lora_rank=${LORA_RANK} \
    actor_rollout_ref.model.lora_alpha=${LORA_ALPHA} \
    actor_rollout_ref.actor.optim.lr=1e-4 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=1 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.prompt_length=8196 \
    actor_rollout_ref.rollout.max_model_len=16384 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.35 \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.n=${k} \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=0.95 \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.val_before_train=False \
    trainer.critic_warmup=0 \
    trainer.use_legacy_worker_impl=disable \
    trainer.logger='["console"]' \
    trainer.project_name='verl_ttrl_or' \
    trainer.experiment_name="${model_name}_verl_ttrl_or" \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.total_epochs=1 "${EXTRA_ARGS[@]}"

