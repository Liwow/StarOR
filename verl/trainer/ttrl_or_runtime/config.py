from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class MCTSConfig:
    max_iterations: int = 20
    c_puct: float = 1.414
    enable_prior: bool = True
    prior_temperature: float = 0.6
    prior_tail_tokens: int = 0
    prior_standardize: bool = True
    prior_use_ref: bool = False
    prior_min_std: float = 1e-4
    stop_on_reward_one: bool = False
    solverllm_compare_mode: bool = False
    final_select_obj_scale_preference: bool = True
    final_select_obj_scale_expand_ratio: float = 0.10
    blocked_sibling_soft_weight: float = 0.6
    # Whether to run CODE terminal refine flow when expanded_to_code is triggered.
    code_refine: bool = True
    # CODE terminal repair rounds (0 means disable repair).
    code_repair: int = 0
    # Enable cluster-linked MCTS backprop update:
    # besides normal selected-path backprop, also update same-cluster siblings
    # (and same-cluster ancestor-layer siblings with decay).
    mcts_cluster_update: bool = True
    # Legacy typo alias, fallback-only compatibility (do not use in new scripts).
    mcts_cluster_updata: bool | None = None


@dataclass(slots=True)
class RewardConfig:
    code_timeout_sec: int = 30
    gurobi_time_limit_sec: float = 30.0
    robustness_cases: int = 3
    global_consensus_rel_tol: float = 0.005
    code_executor_mode: str = "sandbox"  # sandbox or subprocess
    cluster_scope: str = "local"  # global or local

    #  r1 (semantic cluster) settings 
    r1_alpha: float = 0.6  # Smoothing parameter for r1
    r1_min_clusters: int = 3  # Minimum K value for r1 denominator

    enable_r3_reward: bool = False
        
    #  r4 (structural cluster) settings 
    enable_r4_reward: bool = True  # Whether to enable r4
    r4_alpha: float = 0.4  # Smoothing parameter for r4
    r4_k: int = 3  # K value for r4 normalization
    r4_decay: float = 0.95  # Decay factor for historical structural counts

    #  Final reward weights 
    r1_weight: float = 0.6  # Weight for r1 in final reward
    r2_weight: float = 0.1  # Weight for r2 in final reward
    r1_obj_scale_fail_multiplier: float = 0.2  # Multiplier on r1 weight when obj-scale check fails
    r3_weight: float = 0.2  # Weight for r3 in final reward
    r4_weight: float = 0.1  # Weight for r4 in final reward
    structure_gate_min: float = 0.2  # Minimum multiplier when LP structure is incomplete


@dataclass(slots=True)
class GRPOConfig:
    learning_rate: float = 5e-5
    group_size: int = 3  # Alias of num_generations in GRPO literature.
    kl_coef: float = 0.001  # KL penalty coefficient beta.
    sync_ref_model: bool = False  # Whether to periodically refresh the KL reference policy.
    ref_model_sync_steps: int = 5  # Refresh frequency for the KL reference policy.
    ref_model_mixup_alpha: float = 0.6  # Mixup factor when syncing the reference policy.
    train_epochs: float = 1.0
    clip_epsilon: float = 0.3
    clip_epsilon_high: float | None = None
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    num_generations: int = 3
    generation_batch_size: int = 0
    max_prompt_length: int = 10240
    max_completion_length: int = 6144

    use_vllm: bool = False
    vllm_mode: str = "server"
    vllm_gpu_memory_utilization: float = 0.85
    vllm_tensor_parallel_size: int = 1
    vllm_max_model_len: int = 16384
    vllm_max_num_batched_tokens: int = 0
    vllm_enable_sleep_mode: bool = False
    vllm_reset_prefix_cache_after_update: bool = True
    vllm_close_communicator_after_update: bool = True
    vllm_fallback_disable_on_error: bool = True


@dataclass(slots=True)
class DatasetConfig:
    jsonl_path: str = ""
    jsonl_paths: tuple[str, ...] = ()
    start_index: int = 0
    limit: int = 0
    sample_run: bool = False
    sample_seed: int = 0
    sample_size: int = 100
    resume_skip_completed: bool = True
    max_numeric_features: int = 16
    key_param_top_k: int = 8
    mapping_extractor: str = "rule"
    mapping_llm_max_new_tokens: int = 2048
    mapping_llm_temperature: float = 0.3
    mapping_llm_top_p: float = 1.0

    r3_plan_max_new_tokens: int = 2048
    r3_plan_temperature: float = 0.3
    r3_plan_top_p: float = 0.7


@dataclass(slots=True)
class BackendConfig:
    backend: str = "trl"
    model_name_or_path: str = ""
    seed: int = 7
    temperature: float = 0.8
    top_p: float = 0.95
    max_new_tokens: int = 2048
    torch_dtype: str = "auto"
    trust_remote_code: bool = False
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_bias: str = "none"
    lora_target_modules: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )
    reuse_base_model_across_tasks: bool = True
    reset_lora_on_begin_episode: bool = True


@dataclass(slots=True)
class PipelineConfig:
    mcts: MCTSConfig = field(default_factory=MCTSConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    grpo: GRPOConfig = field(default_factory=GRPOConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    backend: BackendConfig = field(default_factory=BackendConfig)
    save_logs: bool = True
    log_dir: str = "logs"
    run_tag: str = ""



