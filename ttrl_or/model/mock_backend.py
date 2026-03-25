from __future__ import annotations

import random
from dataclasses import dataclass, field
from hashlib import md5
from typing import Any

from ttrl_or.config import GRPOConfig
from ttrl_or.model.backend import PolicyBackend
from ttrl_or.types import Generation, OptimizationTask, Stage, TrainingSample


@dataclass(slots=True)
class MockPolicyBackend(PolicyBackend):
    """
    Lightweight backend for local prototyping.
    It does not perform real GRPO updates.
    """

    seed: int = 7
    _stage_bias: dict[Stage, float] = field(default_factory=dict)
    _episode_key: str = ""

    def begin_episode(self, task: OptimizationTask) -> None:
        self._episode_key = task.task_id
        self._stage_bias = {stage: 0.0 for stage in Stage}

    def end_episode(self) -> None:
        self._episode_key = ""
        self._stage_bias = {stage: 0.0 for stage in Stage}

    def generate(self, stage: Stage, prompt: str, n: int) -> list[Generation]:
        cands = self._candidate_bank(stage)
        outputs: list[Generation] = []
        bias = self._stage_bias.get(stage, 0.0)

        for i in range(n):
            rng = self._rng(stage, prompt, i)
            weights = [max(0.001, base + bias) for base, _ in cands]
            idx = _sample_index(rng, weights)
            base_prior, text = cands[idx]
            noisy_prior = base_prior + bias + rng.uniform(-0.05, 0.05)
            outputs.append(
                Generation(
                    text=text,
                    prior=max(0.001, noisy_prior),
                    metadata={"candidate_index": idx},
                )
            )
        return outputs

    def grpo_update(self, samples: list[TrainingSample], config: GRPOConfig, stage: Stage) -> dict[str, Any]:
        return {
            "updated": False,
            "stage": stage.value,
            "num_samples": len(samples),
            "backend": "mock",
            "reason": "MockPolicyBackend does not train. Use TRLPolicyBackend for GRPO updates.",
        }

    def generate_test_instances(self, task: OptimizationTask, k: int) -> list[dict[str, Any]]:
        from ttrl_or.reward.perturbation import generate_perturbed_instances_from_map

        tests = generate_perturbed_instances_from_map(task.instance, task.perturbation_map, k)
        if tests:
            return tests

        fallback: list[dict[str, Any]] = []
        for i in range(k):
            rng = self._rng(Stage.CODE, task.description, 100 + i)
            case: dict[str, Any] = {}
            for key, value in task.instance.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    jitter = 1.0 + rng.uniform(-0.2, 0.2)
                    case[key] = round(value * jitter, 4)
                else:
                    case[key] = value
            fallback.append(case)
        return fallback

    def _rng(self, stage: Stage, prompt: str, salt: int) -> random.Random:
        digest = md5(f"{self.seed}|{self._episode_key}|{stage.value}|{prompt}|{salt}".encode("utf-8")).hexdigest()
        return random.Random(int(digest[:8], 16))

    @staticmethod
    def _candidate_bank(stage: Stage) -> list[tuple[float, str]]:
        if stage == Stage.SCHEMA:
            return [
                (
                    0.68,
                    '{"schema":{"entities":["items"],"data_fields":["cost","value","capacity"],"assumptions":["single objective"]},"skill":{"modeling_patterns":["knapsack-like binary selection"],"decomposition_plan":["extract index set","define value/cost parameters","build capacity constraint"],"solver_tips":["prefer MILP with binary vars"]},"cautions":["units of cost/value must be consistent"]}',
                ),
                (
                    0.61,
                    '{"schema":{"entities":["jobs","machines"],"data_fields":["processing_time","deadline"],"assumptions":["deterministic"]},"skill":{"modeling_patterns":["single-machine scheduling"],"decomposition_plan":["define job index","derive completion variables","encode lateness penalty"],"solver_tips":["use linearized tardiness constraints"]},"cautions":["check due-date inequality directions"]}',
                ),
                (
                    0.49,
                    '{"schema":{"entities":["routes"],"data_fields":["distance","demand","fleet_size"],"assumptions":["static demand"]},"skill":{"modeling_patterns":["vehicle routing style flow model"],"decomposition_plan":["define node/arc sets","add flow conservation","add capacity linking"],"solver_tips":["watch subtour constraints"]},"cautions":["ensure depot degree constraints"]}',
                ),
            ]
        if stage == Stage.SET_PARAM_VAR:
            return [
                (
                    0.67,
                    "Sets\\n- I: item set\\n\\nParameters\\n- cost[i]\\n- value[i]\\n- cap\\n\\nVariables\\n- x[i] in {0,1}",
                ),
                (
                    0.58,
                    "Sets\\n- J: jobs\\n\\nParameters\\n- p[j], d[j]\\n\\nVariables\\n- start[j] >= 0",
                ),
                (
                    0.45,
                    "Sets\\n- N nodes\\n\\nParameters\\n- dist[i,j]\\n\\nVariables\\n- y[i,j] in {0,1}",
                ),
            ]
        if stage == Stage.OBJ_CONS:
            return [
                (
                    0.69,
                    "Objective\\nmaximize sum(value[i] * x[i])\\n\\nConstraints\\n1) sum(cost[i] * x[i]) <= cap",
                ),
                (
                    0.56,
                    "Objective\\nminimize sum(lateness[j])\\n\\nConstraints\\n1) completion[j] = start[j] + p[j]\\n2) lateness[j] >= completion[j] - d[j]",
                ),
                (
                    0.44,
                    "Objective\\nminimize route distance\\n\\nConstraints\\n1) flow conservation\\n2) capacity limit",
                ),
            ]
        return [
            (0.78, _code_variant_sum_numeric()),
            (0.62, _code_variant_weighted_numeric()),
            (0.12, _code_variant_broken()),
        ]


def _sample_index(rng: random.Random, weights: list[float]) -> int:
    total = sum(weights)
    pick = rng.uniform(0.0, total)
    running = 0.0
    for i, w in enumerate(weights):
        running += w
        if pick <= running:
            return i
    return max(0, len(weights) - 1)


def _code_variant_sum_numeric() -> str:
    return """
def solve(instance: dict) -> dict:
    total = 0.0
    for value in instance.values():
        if isinstance(value, (int, float)):
            total += float(value)
    return {"objective": total, "status": "ok"}
""".strip()


def _code_variant_weighted_numeric() -> str:
    return """
def solve(instance: dict) -> dict:
    total = 0.0
    for idx, key in enumerate(sorted(instance.keys())):
        value = instance[key]
        if isinstance(value, (int, float)):
            total += float(value) * (idx + 1)
    return {"objective": total, "status": "ok"}
""".strip()


def _code_variant_broken() -> str:
    return """
def solve(instance: dict) -> dict:
    return {"objective": float(undefined_symbol), "status": "fail"}
""".strip()

