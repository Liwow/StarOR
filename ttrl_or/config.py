from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class MCTSConfig:
    max_iterations: int = 16
    c_puct: float = 1.4
    stop_on_reward_one: bool = False


@dataclass(slots=True)
class RewardConfig:
    code_timeout_sec: int = 30
    robustness_cases: int = 3
    enable_r3_reward: bool = False
    global_consensus_min_pool: int = 3
    global_consensus_rel_tol: float = 0.005
    code_executor_mode: str = "sandbox" #sandbox or subprocess


@dataclass(slots=True)
class GRPOConfig:
    learning_rate: float = 3e-5
    kl_coef: float = 0.0
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    num_generations: int = 3
    generation_batch_size: int = 0
    max_prompt_length: int = 8196
    max_completion_length: int = 2048

    use_vllm: bool = False
    vllm_mode: str = "server"
    vllm_gpu_memory_utilization: float = 0.85
    vllm_tensor_parallel_size: int = 1
    vllm_max_model_len: int = 16384
    vllm_fallback_disable_on_error: bool = True


@dataclass(slots=True)
class DatasetConfig:
    jsonl_path: str = ""
    start_index: int = 0
    limit: int = 0
    max_numeric_features: int = 16
    key_param_top_k: int = 8
    mapping_extractor: str = "rule"
    mapping_llm_max_new_tokens: int = 1024
    mapping_llm_temperature: float = 0.0
    mapping_llm_top_p: float = 1.0


@dataclass(slots=True)
class BackendConfig:
    backend: str = "mock"
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
