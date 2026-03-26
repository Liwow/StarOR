from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass, field

from ttrl_or.config import RewardConfig
from ttrl_or.model.backend import PolicyBackend
from ttrl_or.reward.base import RewardCalculator
from ttrl_or.reward.executor import PythonCodeExecutor
from ttrl_or.reward.perturbation import generate_perturbed_instances_from_map
from ttrl_or.types import OptimizationTask, RewardBreakdown, Trajectory

_NUMERIC_REL_TOL = 0.005  # 0.5%


@dataclass(slots=True)
class TTRLRewardCalculator(RewardCalculator):
    task: OptimizationTask
    backend: PolicyBackend
    config: RewardConfig
    executor: PythonCodeExecutor = field(init=False)
    _exec_cache: dict[str, object] = field(default_factory=dict, init=False)
    _global_numeric_pool: list[float] = field(default_factory=list, init=False)
    _global_signature_pool: list[str] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self.executor = PythonCodeExecutor(timeout_sec=self.config.code_timeout_sec)

    def provisional_reward(self, trajectory: Trajectory, explored: list[Trajectory]) -> RewardBreakdown:
        execution = self._execute(trajectory)
        r1, consensus = self.compute_r1(execution.success, execution.signature, execution.output)

        if execution.success:
            self._update_global_pool(execution.signature, execution.output)

        if r1 == 1.0:
            r3, r3_meta = self._compute_r3_with_details(trajectory)
            total = self.combine_rewards(r1=r1, r2=0.0, r3=r3)
            return RewardBreakdown(
                r1=r1,
                r2=0.0,
                r3=r3,
                total=total,
                consensus_signature=consensus,
                execution_success=execution.success,
                robustness_success=(r3 == 1.0),
                metadata={"r3": r3_meta},
            )

        r2 = self.compute_r2(execution.success)
        total = self.combine_rewards(r1=r1, r2=r2, r3=0.0)
        return RewardBreakdown(
            r1=r1,
            r2=r2,
            r3=0.0,
            total=total,
            consensus_signature=consensus,
            execution_success=execution.success,
            robustness_success=False,
            metadata={},
        )

    def finalize_group(self, trajectories: list[Trajectory]) -> list[Trajectory]:
        for traj in trajectories:
            exec_result = self._execute(traj)
            r1, consensus = self.compute_r1(exec_result.success, exec_result.signature, exec_result.output)

            if exec_result.success:
                self._update_global_pool(exec_result.signature, exec_result.output)

            if r1 == 1.0:
                r3, r3_meta = self._compute_r3_with_details(traj)
                total = self.combine_rewards(r1=r1, r2=0.0, r3=r3)
                traj.reward = RewardBreakdown(
                    r1=r1,
                    r2=0.0,
                    r3=r3,
                    total=total,
                    consensus_signature=consensus,
                    execution_success=exec_result.success,
                    robustness_success=(r3 == 1.0),
                    metadata={"r3": r3_meta},
                )
            else:
                r2 = self.compute_r2(exec_result.success)
                total = self.combine_rewards(r1=r1, r2=r2, r3=0.0)
                traj.reward = RewardBreakdown(
                    r1=r1,
                    r2=r2,
                    r3=0.0,
                    total=total,
                    consensus_signature=consensus,
                    execution_success=exec_result.success,
                    robustness_success=False,
                    metadata={},
                )
        return trajectories

    def compute_r1(self, execution_success: bool, signature: str, output: object | None = None) -> tuple[float, str]:
        if not execution_success:
            return 0.0, ""

        numeric = self._extract_objective_numeric(output)

        if numeric is not None:
            return self._compute_numeric_r1(numeric)

        return self._compute_signature_r1(signature)

    def _compute_numeric_r1(self, candidate: float) -> tuple[float, str]:
        pool = self._global_numeric_pool
        if len(pool) < 3:
            if not pool:
                return 1.0, f"oom:{self._order_of_magnitude(candidate)}"

            cand_oom = self._order_of_magnitude(candidate)
            consensus = any(self._order_of_magnitude(v) == cand_oom for v in pool)
            return (1.0 if consensus else 0.0), f"oom:{cand_oom}"

        ref, votes = self._majority_numeric_reference(pool)
        if ref is None or votes <= 0:
            return 0.0, ""

        in_consensus = self._within_rel_tol(candidate, ref, _NUMERIC_REL_TOL)
        return (1.0 if in_consensus else 0.0), f"ref:{ref:.6f}|votes:{votes}|tol:{_NUMERIC_REL_TOL}"

    def _compute_signature_r1(self, signature: str) -> tuple[float, str]:
        pool = [s for s in self._global_signature_pool if s and s != "EXEC_ERROR"]
        if len(pool) < 3:
            if not pool:
                return 1.0, signature
            return (1.0 if signature in pool else 0.0), signature

        majority = Counter(pool).most_common(1)[0][0]
        return (1.0 if signature == majority else 0.0), majority

    @staticmethod
    def compute_r2(execution_success: bool) -> float:
        return 1.0 if execution_success else 0.0

    def compute_r3(self, trajectory: Trajectory) -> float:
        score, _ = self._compute_r3_with_details(trajectory)
        return score

    @staticmethod
    def combine_rewards(r1: float, r2: float, r3: float) -> float:
        if r1 == 1.0:
            return r1 * 0.9 + r3 * 0.1
        return r2 * 0.2

    def _compute_r3_with_details(self, trajectory: Trajectory) -> tuple[float, dict]:
        if not self.config.enable_perturb_reward:
            return 1.0, {
                "enabled": False,
                "reason": "disabled_by_config",
                "num_cases": 0,
            }

        tests = self.backend.generate_test_instances(self.task, self.config.robustness_cases)
        source = "backend"
        if not tests:
            tests = generate_perturbed_instances_from_map(self.task.instance, self.task.perturbation_map, self.config.robustness_cases)
            source = "heuristic"

        if not tests:
            return 0.0, {
                "enabled": True,
                "source": source,
                "reason": "no_perturb_cases",
                "num_cases": 0,
            }

        details: list[dict] = []
        for idx, case in enumerate(tests):
            res = self.executor.run(trajectory.code, case)
            case_meta = case.get("__perturbation__") if isinstance(case, dict) else None
            detail = {
                "case_index": idx,
                "success": res.success,
                "signature": res.signature,
                "changes": case_meta.get("changes", []) if isinstance(case_meta, dict) else [],
            }
            details.append(detail)
            if not res.success:
                return 0.0, {
                    "enabled": True,
                    "source": source,
                    "num_cases": len(tests),
                    "failed_case_index": idx,
                    "cases": details,
                }

        return 1.0, {
            "enabled": True,
            "source": source,
            "num_cases": len(tests),
            "cases": details,
        }

    def _execute(self, trajectory: Trajectory):
        if trajectory.trajectory_id in self._exec_cache:
            return self._exec_cache[trajectory.trajectory_id]
        result = self.executor.run(trajectory.code, self.task.instance)
        self._exec_cache[trajectory.trajectory_id] = result
        return result

    def _update_global_pool(self, signature: str, output: object | None) -> None:
        numeric = self._extract_objective_numeric(output)
        if numeric is not None:
            self._global_numeric_pool.append(float(numeric))
            return

        if signature and signature != "EXEC_ERROR":
            self._global_signature_pool.append(signature)

    @staticmethod
    def _extract_objective_numeric(output: object | None) -> float | None:
        if isinstance(output, dict):
            objective = output.get("objective")
            if isinstance(objective, (int, float)) and not isinstance(objective, bool):
                return float(objective)

        if isinstance(output, str):
            try:
                maybe = json.loads(output)
            except Exception:
                return None
            if isinstance(maybe, dict):
                objective = maybe.get("objective")
                if isinstance(objective, (int, float)) and not isinstance(objective, bool):
                    return float(objective)

        return None

    @staticmethod
    def _order_of_magnitude(value: float) -> int:
        if value == 0:
            return 0
        return int(math.floor(math.log10(abs(value))))

    @staticmethod
    def _within_rel_tol(value: float, ref: float, rel_tol: float) -> bool:
        base = max(abs(ref), 1e-12)
        return abs(value - ref) <= rel_tol * base

    def _majority_numeric_reference(self, values: list[float]) -> tuple[float | None, int]:
        if not values:
            return None, 0

        best_members: list[float] = []
        for anchor in values:
            members = [v for v in values if self._within_rel_tol(v, anchor, _NUMERIC_REL_TOL)]
            if len(members) > len(best_members):
                best_members = members

        if not best_members:
            return None, 0

        ref = sum(best_members) / len(best_members)
        return ref, len(best_members)
