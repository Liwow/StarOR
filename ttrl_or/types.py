from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Stage(str, Enum):
    SCHEMA = "schema"
    SET_PARAM_VAR = "set_param_var"
    OBJ_CONS = "obj_cons"
    CODE = "code"


STAGE_ORDER: tuple[Stage, ...] = (
    Stage.SCHEMA,
    Stage.SET_PARAM_VAR,
    Stage.OBJ_CONS,
    Stage.CODE,
)


@dataclass(slots=True)
class OptimizationTask:
    task_id: str
    description: str
    instance: dict[str, Any] = field(default_factory=dict)
    perturbation_map: dict[str, Any] = field(default_factory=dict)
    gold_answer: str = ""


@dataclass(slots=True)
class Generation:
    text: str
    prior: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ExecutionResult:
    success: bool
    output: Any = None
    stdout: str = ""
    stderr: str = ""
    error_type: str | None = None
    signature: str = ""
    elapsed_sec: float = 0.0


@dataclass(slots=True)
class RewardBreakdown:
    r1: float
    r2: float
    r3: float
    total: float
    consensus_signature: str = ""
    execution_success: bool = False
    robustness_success: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Trajectory:
    trajectory_id: str
    outputs: dict[Stage, str] = field(default_factory=dict)
    priors: dict[Stage, float] = field(default_factory=dict)
    reward: RewardBreakdown | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def text(self, stage: Stage) -> str:
        return self.outputs.get(stage, "")

    @property
    def code(self) -> str:
        return self.outputs.get(Stage.CODE, "")

    def prefix(self, stage: Stage) -> dict[Stage, str]:
        prefix: dict[Stage, str] = {}
        for s in STAGE_ORDER:
            if s == stage:
                break
            if s in self.outputs:
                prefix[s] = self.outputs[s]
        return prefix


@dataclass(slots=True)
class TrainingSample:
    stage: Stage
    prompt: str
    completion: str
    reward: float
    group_id: str
    trajectory_id: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class StageTrace:
    stage: str
    num_frontier_in: int
    num_frontier_out: int
    stage_samples: int
    grpo_report: dict[str, Any]
    mcts_early_stop: bool = False
    mcts_early_stop_info: dict[str, Any] = field(default_factory=dict)
    expansions: list[dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class RunTrace:
    task_id: str
    backend: str
    task_description: str
    instance: dict[str, Any]
    perturbation_map: dict[str, Any] = field(default_factory=dict)
    gold_answer: str = ""
    task_context: dict[str, Any] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)
    stages: list[StageTrace] = field(default_factory=list)
    iteration_logs: list[dict[str, Any]] = field(default_factory=list)
    final_selection: dict[str, Any] = field(default_factory=dict)
    best_trajectory: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)
