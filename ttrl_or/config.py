from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class MCTSConfig:
    expand_per_node: int = 4
    simulations_per_node: int = 5
    max_nodes_per_stage: int = 15
    c_puct: float = 1.4
    rollout_k: int = 3


@dataclass(slots=True)
class RewardConfig:
    code_timeout_sec: int = 6
    robustness_cases: int = 3
    local_consensus_window: int = 32
    enable_perturb_reward: bool = False


@dataclass(slots=True)
class GRPOConfig:
    learning_rate: float = 3e-5
    clip_range: float = 0.2
    kl_coef: float = 0.0
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    num_generations: int = 2
    max_prompt_length: int = 4096
    max_completion_length: int = 256
    max_steps: int = 3


@dataclass(slots=True)
class DatasetConfig:
    jsonl_path: str = "data\IndustryOR_fixedV2.jsonl"
    start_index: int = 0
    limit: int = 0
    max_numeric_features: int = 16
    key_param_top_k: int = 8


@dataclass(slots=True)
class PipelineConfig:
    mcts: MCTSConfig = field(default_factory=MCTSConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    grpo: GRPOConfig = field(default_factory=GRPOConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    group_size: int = 8
    save_logs: bool = True
    log_dir: str = "logs"
