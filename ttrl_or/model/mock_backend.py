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

    def generate(self, stage: Stage, prompt: Any, n: int) -> list[Generation]:
        cands = self._candidate_bank(stage)
        outputs: list[Generation] = []
        bias = self._stage_bias.get(stage, 0.0)
        prompt_key = self._prompt_key(prompt)

        for i in range(n):
            rng = self._rng(stage, prompt_key, i)
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

    def score_action_priors(self, stage: Stage, prompt: Any, candidates: list[str]) -> list[float]:
        if not candidates:
            return []
        prompt_key = self._prompt_key(prompt)
        weights: list[float] = []
        for idx, candidate in enumerate(candidates):
            if not str(candidate or '').strip():
                weights.append(1e-6)
                continue
            rng = self._rng(stage, prompt_key + '||' + str(candidate), 1000 + idx)
            weights.append(max(1e-6, 0.5 + rng.random()))
        total = sum(weights)
        if total <= 0:
            return [1.0 / float(len(candidates))] * len(candidates)
        return [float(w / total) for w in weights]

    def grpo_rollout_group(
        self,
        stage: Stage,
        prompt: Any,
        config: GRPOConfig,
        reward_callback,
    ) -> tuple[list[Generation], dict[str, Any]]:
        k = max(1, int(config.num_generations))
        gens = self.generate(stage, prompt, k)
        out: list[Generation] = []

        batch_score = getattr(reward_callback, "batch_score", None)
        if callable(batch_score):
            rewards = list(batch_score(prompt, [gen.text for gen in gens]))
            if len(rewards) != len(gens):
                rewards = rewards[: len(gens)] + [0.0] * max(0, len(gens) - len(rewards))
            for ridx, gen in enumerate(gens):
                out.append(
                    Generation(
                        text=gen.text,
                        prior=gen.prior,
                        metadata={
                            **dict(gen.metadata),
                            "rollout_index": ridx,
                            "reward_total": float(rewards[ridx]),
                        },
                    )
                )
        else:
            for ridx, gen in enumerate(gens):
                reward_total = float(reward_callback(prompt, gen.text, ridx))
                out.append(
                    Generation(
                        text=gen.text,
                        prior=gen.prior,
                        metadata={
                            **dict(gen.metadata),
                            "rollout_index": ridx,
                            "reward_total": reward_total,
                        },
                    )
                )
        report = {
            "updated": False,
            "stage": stage.value,
            "num_samples": len(out),
            "backend": "mock",
            "group_mode": "internal_rollout_mock",
            "reason": "MockPolicyBackend does not train. Use TRLPolicyBackend for GRPO updates.",
        }
        return out, report

    def grpo_update(self, samples: list[TrainingSample], config: GRPOConfig, stage: Stage) -> dict[str, Any]:
        return {
            "updated": False,
            "stage": stage.value,
            "num_samples": len(samples),
            "backend": "mock",
            "reason": "MockPolicyBackend does not train. Use TRLPolicyBackend for GRPO updates.",
        }

    def generate_auxiliary_text(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 1024,
        temperature: float = 0.0,
        top_p: float = 1.0,
        prefer_vllm: bool = False,
        vllm_mode: str = "",
    ) -> str | None:
        return """<stage_2>
- i: items
</stage_2>

<stage_3>
- c_i: item cost
- v_i: item value
- cap: capacity
</stage_3>

<stage_4>
- x_i: choose item i (BINARY)
</stage_4>

<stage_5>
- maximize_total_value: maximize total selected value
</stage_5>

<stage_6>
- capacity_limit: total cost does not exceed capacity
</stage_6>

<Gurobi_code>
def solve(instance: dict) -> dict:
    total = 0.0
    for value in instance.values():
        if isinstance(value, (int, float)):
            total += float(value)
    print(f"Optimal value: {total}")
    return {"objective": total, "status": "ok"}
</Gurobi_code>"""

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
    def _prompt_key(prompt: Any) -> str:
        if isinstance(prompt, list):
            return "\n\n".join(
                f"[{str(item.get('role', 'user')).upper()}]\n{str(item.get('content', '')).strip()}"
                for item in prompt
                if isinstance(item, dict) and str(item.get("content", "")).strip()
            )
        return str(prompt)

    @staticmethod
    def _candidate_bank(stage: Stage) -> list[tuple[float, str]]:
        if stage == Stage.SCHEMA:
            return [
                (0.68, '{"schema":{"entities":["items"],"goal":"maximize value under capacity"},"skill":{"patterns":["knapsack MILP"]},"cautions":["keep indexing consistent"]}'),
                (0.52, '{"schema":{"entities":["jobs","machines"],"goal":"assign jobs feasibly"},"skill":{"patterns":["assignment / scheduling"]},"cautions":["respect resource balance"]}'),
            ]
        if stage == Stage.SET_PARAM_VAR:
            return [
                (0.67, 'Sets\n- I: item set\n\nParameters\n- cost[i]\n- value[i]\n- cap\n\nVariables\n- x[i] in {0,1}'),
                (0.52, 'Sets\n- J: jobs\n\nParameters\n- p[j], d[j]\n\nVariables\n- start[j] >= 0'),
            ]
        if stage == Stage.OBJ_CONS:
            return [
                (0.69, 'Objective\nmaximize sum(value[i] * x[i])\n\nConstraints\n1) sum(cost[i] * x[i]) <= cap'),
                (0.56, 'Objective\nminimize sum(lateness[j])\n\nConstraints\n1) completion[j] = start[j] + p[j]\n2) lateness[j] >= completion[j] - d[j]'),
            ]
        if stage == Stage.TYPE_HINT:
            return [
                (0.72, '{"type":"MILP","subtype":"resource allocation","modeling_hints":["identify discrete decisions","preserve resource balance"],"cautions":["watch integrality and indexing"]}'),
                (0.54, '{"type":"LP","subtype":"production planning","modeling_hints":["start from flow balance","keep objective linear"],"cautions":["verify bound directions"]}'),
            ]
        if stage == Stage.SETS:
            return [
                (0.68, '- i: items\n- r: resources'),
                (0.52, '- j: jobs\n- m: machines'),
            ]
        if stage == Stage.PARAMETERS:
            return [
                (0.67, '- cost[i]: unit cost\n- value[i]: unit value\n- cap: capacity'),
                (0.53, '- p[j]: processing time\n- d[j]: deadline'),
            ]
        if stage == Stage.VARIABLES:
            return [
                (0.66, '- x[i]: binary selection variable'),
                (0.55, '- start[j]: start time\n- late[j]: tardiness'),
            ]
        if stage == Stage.OBJECTIVE:
            return [
                (0.69, '- maximize total value'),
                (0.54, '- minimize total tardiness'),
            ]
        if stage == Stage.CONSTRAINTS:
            return [
                (0.69, '- capacity limit\n- logical consistency'),
                (0.55, '- timing feasibility\n- non-negativity'),
            ]
        return [
            (0.70, 'def solve(instance: dict) -> dict:\n    total = sum(float(v) for v in instance.values() if isinstance(v, (int, float)))\n    print(f"Optimal value: {total}")\n    return {"objective": total, "status": "ok"}'),
            (0.50, 'def solve(instance: dict) -> dict:\n    return {"objective": 0.0, "status": "mock"}'),
        ]


def _sample_index(rng: random.Random, weights: list[float]) -> int:
    total = sum(weights)
    threshold = rng.uniform(0, total)
    cursor = 0.0
    for idx, weight in enumerate(weights):
        cursor += weight
        if cursor >= threshold:
            return idx
    return len(weights) - 1
