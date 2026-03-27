from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field

from ttrl_or.config import RewardConfig
from ttrl_or.model.backend import PolicyBackend
from ttrl_or.reward.base import RewardCalculator
from ttrl_or.reward.executor import PythonCodeExecutor
from ttrl_or.reward.perturbation import generate_perturbed_instances_from_map
from ttrl_or.types import ExecutionResult, OptimizationTask, RewardBreakdown, Trajectory


@dataclass(slots=True)
class TTRLRewardCalculator(RewardCalculator):
    task: OptimizationTask
    backend: PolicyBackend
    config: RewardConfig
    executor: PythonCodeExecutor = field(init=False)
    _exec_cache: dict[str, ExecutionResult] = field(default_factory=dict, init=False)
    _global_numeric_pool: list[float] = field(default_factory=list, init=False)
    _global_signature_pool: list[str] = field(default_factory=list, init=False)
    _gurobi_success_markers: tuple[str, ...] = field(
        default=("optimal solution found", "model is solved to optimality"),
        init=False,
    )

    def __post_init__(self) -> None:
        self.executor = PythonCodeExecutor(
            timeout_sec=self.config.code_timeout_sec,
            mode=self.config.code_executor_mode,
        )

    def provisional_reward(self, trajectory: Trajectory, explored: list[Trajectory]) -> RewardBreakdown:
        execution, exec_cache_hit = self._execute(trajectory)
        strict_success = bool(execution.success)
        r2_success = self._effective_execution_success(execution)
        obj_answer = self._extract_objective_from_execution(execution)
        r1, consensus = self.compute_r1(
            strict_success,
            execution.signature,
            execution.output,
            numeric_override=obj_answer,
        )

        if strict_success:
            self._update_global_pool(execution.signature, execution.output, numeric_override=obj_answer)

        common_meta = {
            "exec_elapsed_sec": float(execution.elapsed_sec),
            "exec_cache_hit": bool(exec_cache_hit),
            "obj_answer": obj_answer,
            "execution": self._execution_summary(execution, obj_answer=obj_answer),
        }

        if r1 == 1.0:
            r3, r3_meta = self._compute_r3_with_details(trajectory)
            total = self.combine_rewards(r1=r1, r2=0.0, r3=r3)
            return RewardBreakdown(
                r1=r1,
                r2=0.0,
                r3=r3,
                total=total,
                consensus_signature=consensus,
                execution_success=strict_success,
                robustness_success=(r3 == 1.0),
                metadata={"r3": r3_meta, **common_meta},
            )

        r2 = self.compute_r2(r2_success)
        total = self.combine_rewards(r1=r1, r2=r2, r3=0.0)
        return RewardBreakdown(
            r1=r1,
            r2=r2,
            r3=0.0,
            total=total,
            consensus_signature=consensus,
            execution_success=strict_success,
            robustness_success=False,
            metadata=common_meta,
        )

    def finalize_group(self, trajectories: list[Trajectory]) -> list[Trajectory]:
        for traj in trajectories:
            exec_result, exec_cache_hit = self._execute(traj)
            strict_success = bool(exec_result.success)
            r2_success = self._effective_execution_success(exec_result)
            obj_answer = self._extract_objective_from_execution(exec_result)
            r1, consensus = self.compute_r1(
                strict_success,
                exec_result.signature,
                exec_result.output,
                numeric_override=obj_answer,
            )

            if strict_success:
                self._update_global_pool(exec_result.signature, exec_result.output, numeric_override=obj_answer)

            common_meta = {
                "exec_elapsed_sec": float(exec_result.elapsed_sec),
                "exec_cache_hit": bool(exec_cache_hit),
                "obj_answer": obj_answer,
                "execution": self._execution_summary(exec_result, obj_answer=obj_answer),
            }

            if r1 == 1.0:
                r3, r3_meta = self._compute_r3_with_details(traj)
                total = self.combine_rewards(r1=r1, r2=0.0, r3=r3)
                traj.reward = RewardBreakdown(
                    r1=r1,
                    r2=0.0,
                    r3=r3,
                    total=total,
                    consensus_signature=consensus,
                    execution_success=strict_success,
                    robustness_success=(r3 == 1.0),
                    metadata={"r3": r3_meta, **common_meta},
                )
            else:
                r2 = self.compute_r2(r2_success)
                total = self.combine_rewards(r1=r1, r2=r2, r3=0.0)
                traj.reward = RewardBreakdown(
                    r1=r1,
                    r2=r2,
                    r3=0.0,
                    total=total,
                    consensus_signature=consensus,
                    execution_success=strict_success,
                    robustness_success=False,
                    metadata=common_meta,
                )
        return trajectories

    def compute_r1(
        self,
        execution_success: bool,
        signature: str,
        output: object | None = None,
        numeric_override: float | None = None,
    ) -> tuple[float, str]:
        if not execution_success:
            return 0.0, ""

        numeric = numeric_override if numeric_override is not None else self._extract_objective_numeric(output)
        if numeric is not None:
            return self._compute_numeric_r1(numeric)

        return self._compute_signature_r1(signature)

    def _compute_numeric_r1(self, candidate: float) -> tuple[float, str]:
        pool = self._global_numeric_pool
        min_pool = max(1, int(self.config.global_consensus_min_pool))
        rel_tol = max(0.0, float(self.config.global_consensus_rel_tol))

        if len(pool) < min_pool:
            if not pool:
                return 1.0, f"oom:{self._order_of_magnitude(candidate)}"

            cand_oom = self._order_of_magnitude(candidate)
            consensus = any(self._order_of_magnitude(v) == cand_oom for v in pool)
            return (1.0 if consensus else 0.0), f"oom:{cand_oom}"

        ref, votes = self._majority_numeric_reference(pool, rel_tol=rel_tol)
        if ref is None or votes <= 0:
            return 0.0, ""

        in_consensus = self._within_rel_tol(candidate, ref, rel_tol)
        return (1.0 if in_consensus else 0.0), f"ref:{ref:.6f}|votes:{votes}|tol:{rel_tol}"

    def _compute_signature_r1(self, signature: str) -> tuple[float, str]:
        pool = [s for s in self._global_signature_pool if s and s != "EXEC_ERROR"]
        min_pool = max(1, int(self.config.global_consensus_min_pool))
        if len(pool) < min_pool:
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
        r3_enabled = bool(self.config.enable_r3_reward)
        if not r3_enabled:
            return 1.0, {
                "enabled": False,
                "enable_r3_reward": bool(self.config.enable_r3_reward),
                "reason": "disabled_by_config",
                "num_cases": 0,
            }

        tests = self.backend.generate_test_instances(self.task, self.config.robustness_cases)
        source = "backend"
        if not tests:
            tests = generate_perturbed_instances_from_map(
                self.task.instance,
                self.task.perturbation_map,
                self.config.robustness_cases,
            )
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
                "elapsed_sec": float(res.elapsed_sec),
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

    def _execute(self, trajectory: Trajectory) -> tuple[ExecutionResult, bool]:
        cache_key = self._execution_cache_key(trajectory.code, self.task.instance)
        if cache_key in self._exec_cache:
            return self._exec_cache[cache_key], True

        result = self.executor.run(trajectory.code, self.task.instance)
        self._exec_cache[cache_key] = result
        return result, False

    @staticmethod
    def _execution_cache_key(code: str, instance: dict) -> str:
        payload = {"code": code, "instance": instance}
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=repr)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _update_global_pool(
        self,
        signature: str,
        output: object | None,
        numeric_override: float | None = None,
    ) -> None:
        numeric = numeric_override if numeric_override is not None else self._extract_objective_numeric(output)
        if numeric is not None:
            self._global_numeric_pool.append(float(numeric))
            return

        if signature and signature != "EXEC_ERROR":
            self._global_signature_pool.append(signature)

    @staticmethod
    def _extract_objective_numeric(output: object | None) -> float | None:
        def _as_float(value: object) -> float | None:
            if isinstance(value, bool):
                return None
            if isinstance(value, (int, float)):
                return float(value)
            if isinstance(value, str):
                text = value.strip().replace(",", "")
                if not text:
                    return None
                try:
                    return float(text)
                except Exception:
                    return None
            return None

        if isinstance(output, dict):
            for key in ("objective", "obj", "optimal", "optimal_value", "best_objective"):
                numeric = _as_float(output.get(key))
                if numeric is not None:
                    return numeric
            nested = output.get("result")
            if isinstance(nested, dict):
                for key in ("objective", "obj", "optimal", "optimal_value", "best_objective"):
                    numeric = _as_float(nested.get(key))
                    if numeric is not None:
                        return numeric

        if isinstance(output, str):
            try:
                maybe = json.loads(output)
            except Exception:
                return TTRLRewardCalculator._extract_objective_from_text(output)
            if isinstance(maybe, dict):
                numeric = TTRLRewardCalculator._extract_objective_numeric(maybe)
                if numeric is not None:
                    return numeric
            return TTRLRewardCalculator._extract_objective_from_text(output)

        return None

    @staticmethod
    def _extract_objective_from_text(text: str | None) -> float | None:
        raw = str(text or "")
        if not raw.strip():
            return None

        patterns = (
            r"(?i)\boptimal\s*value\s*[:=]\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)",
            r"(?i)\bobjective(?:\s*value)?\s*[:=]\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)",
            r"(?i)\bobj(?:ective)?\s*[:=]\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)",
        )
        for pattern in patterns:
            matches = re.findall(pattern, raw, flags=re.IGNORECASE)
            if matches:
                try:
                    return float(matches[-1].replace(",", ""))
                except Exception:
                    pass
        return None

    def _extract_objective_from_execution(self, execution: ExecutionResult) -> float | None:
        from_output = self._extract_objective_numeric(execution.output)
        if from_output is not None:
            return from_output

        from_stdout = self._extract_objective_from_text(execution.stdout)
        if from_stdout is not None:
            return from_stdout

        return self._extract_objective_from_text(execution.stderr)

    @staticmethod
    def _order_of_magnitude(value: float) -> int:
        if value == 0:
            return 0
        return int(math.floor(math.log10(abs(value))))

    @staticmethod
    def _within_rel_tol(value: float, ref: float, rel_tol: float) -> bool:
        base = max(abs(ref), 1e-12)
        return abs(value - ref) <= rel_tol * base

    def _majority_numeric_reference(self, values: list[float], rel_tol: float) -> tuple[float | None, int]:
        if not values:
            return None, 0

        best_members: list[float] = []
        for anchor in values:
            members = [v for v in values if self._within_rel_tol(v, anchor, rel_tol)]
            if len(members) > len(best_members):
                best_members = members

        if not best_members:
            return None, 0

        ref = sum(best_members) / len(best_members)
        return ref, len(best_members)

    def _effective_execution_success(self, execution: ExecutionResult) -> bool:
        if bool(execution.success):
            return True
        stdout_text = str(execution.stdout or "")
        lowered = stdout_text.lower()
        return any(marker in lowered for marker in self._gurobi_success_markers)

    @staticmethod
    def _execution_summary(execution: ExecutionResult, obj_answer: float | None = None) -> dict:
        stdout_text = str(execution.stdout or "")
        marker_hit = "optimal solution found" in stdout_text.lower()
        return {
            "success": bool(execution.success),
            "effective_success": bool(execution.success or marker_hit),
            "solver_success_marker_hit": marker_hit,
            "parsed_obj_answer": obj_answer,
            "signature": str(execution.signature or ""),
            "error_type": str(execution.error_type or ""),
            "output": TTRLRewardCalculator._jsonable(execution.output),
            "stdout_tail": TTRLRewardCalculator._truncate_text(execution.stdout, 2000),
            "stderr_tail": TTRLRewardCalculator._truncate_text(execution.stderr, 2000),
        }

    @staticmethod
    def _truncate_text(text: str, max_len: int) -> str:
        raw = str(text or "")
        if len(raw) <= max_len:
            return raw
        return raw[-max_len:]

    @staticmethod
    def _jsonable(value):
        try:
            json.dumps(value, ensure_ascii=False)
            return value
        except Exception:
            return repr(value)

