from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Stage(str, Enum):
    SCHEMA = "Stage 1: Schema and Modeling Skill Analysis"
    SET_PARAM_VAR = "Stage 2: Set, Parameters, and Variables Construction"
    OBJ_CONS = "Stage 3: Objective and Constraints Modeling"

    TYPE_HINT = "Stage 1: Type and Modeling Hint Analysis"
    SETS = "Stage 2: Sets Construction"
    PARAMETERS = "Stage 3: Parameters Construction"
    VARIABLES = "Stage 4: Variables Construction"
    OBJECTIVE = "Stage 5: Objective Construction"
    CONSTRAINTS = "Stage 6: Constraints Construction"

    CODE = "Problem Python Code with Gurobi"


DEFAULT_STAGE_ORDER: tuple[Stage, ...] = (
    Stage.SCHEMA,
    Stage.SET_PARAM_VAR,
    Stage.OBJ_CONS,
    Stage.CODE,
)

SOLVERLLM_STAGE_ORDER: tuple[Stage, ...] = (
    Stage.TYPE_HINT,
    Stage.SETS,
    Stage.PARAMETERS,
    Stage.VARIABLES,
    Stage.OBJECTIVE,
    Stage.CONSTRAINTS,
    Stage.CODE,
)

STAGE_ORDER: tuple[Stage, ...] = DEFAULT_STAGE_ORDER


_STAGE_TAG_MAP: dict[Stage, str] = {
    Stage.SCHEMA: "Type",
    Stage.SET_PARAM_VAR: "Sets",
    Stage.OBJ_CONS: "Objective",
    Stage.TYPE_HINT: "Type",
    Stage.SETS: "Sets",
    Stage.PARAMETERS: "Parameters",
    Stage.VARIABLES: "Variables",
    Stage.OBJECTIVE: "Objective",
    Stage.CONSTRAINTS: "Constraints",
    Stage.CODE: "python",
}


def get_stage_order(solverllm_compare_mode: bool = False) -> tuple[Stage, ...]:
    return SOLVERLLM_STAGE_ORDER if bool(solverllm_compare_mode) else DEFAULT_STAGE_ORDER


def stage_output_tag(stage: Stage) -> str:
    return _STAGE_TAG_MAP[stage]


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
class ModelInfo:
    """Gurobi model structural information extracted from .lp file."""
    model_sense: int = 0
    num_vars: int = 0
    num_bin_vars: int = 0
    num_int_vars: int = 0
    num_constrs: int = 0
    has_objective: bool = False
    has_constraints: bool = False
    has_variables: bool = False
    extracted: bool = False

    def feature_tuple(self) -> tuple[int, int, int, int]:
        return (self.model_sense, self.num_vars, self.num_bin_vars, self.num_int_vars)


@dataclass(slots=True)
class ExecutionResult:
    success: bool
    output: Any = None
    stdout: str = ""
    stderr: str = ""
    error_type: str | None = None
    signature: str = ""
    elapsed_sec: float = 0.0
    model_info: ModelInfo | None = None


@dataclass(slots=True)
class RewardBreakdown:
    r1: float
    r2: float
    r3: float
    r4: float = 0.0
    total: float = 0.0
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

    def prefix(self, stage: Stage, stage_order: tuple[Stage, ...] = STAGE_ORDER) -> dict[Stage, str]:
        prefix: dict[Stage, str] = {}
        for s in stage_order:
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

