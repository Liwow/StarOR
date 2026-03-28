from __future__ import annotations

import hashlib
import math
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from ttrl_or.config import RewardConfig
from ttrl_or.model.backend import PolicyBackend
from ttrl_or.reward.base import RewardCalculator
from ttrl_or.reward.executor import PythonCodeExecutor
from ttrl_or.reward.perturbation import generate_perturbed_instances_from_map
from ttrl_or.types import ExecutionResult, OptimizationTask, RewardBreakdown, Stage, Trajectory


@dataclass(slots=True)
class TTRLRewardCalculator(RewardCalculator):
    task: OptimizationTask
    backend: PolicyBackend
    config: RewardConfig
    executor: PythonCodeExecutor = field(init=False)
    _exec_cache: dict[str, ExecutionResult] = field(default_factory=dict, init=False)
    _stage_numeric_pool: dict[str, list[float]] = field(default_factory=dict, init=False)
    _stage_signature_pool: dict[str, list[str]] = field(default_factory=dict, init=False)
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
        rewards = self.score_rollout_group(stage=Stage.CODE, trajectories=[trajectory], explored=explored)
        return rewards[0]

    def score_rollout_group(
        self,
        stage: Stage,
        trajectories: list[Trajectory],
        explored: list[Trajectory],
    ) -> list[RewardBreakdown]:
        if not trajectories:
            return []

        stage_key = stage.value
        rel_tol = max(0.0, float(self.config.global_consensus_rel_tol))
        base_obj_bounds = self._base_obj_bounds()

        stage_numeric_hist = list(self._stage_numeric_pool.get(stage_key, []))
        stage_signature_hist = list(self._stage_signature_pool.get(stage_key, []))

        evals: list[dict[str, Any]] = []
        for traj in trajectories:
            execution, exec_cache_hit = self._execute(traj)
            strict_success = bool(execution.success)
            effective_success = self._effective_execution_success(execution)
            obj_answer = self._extract_objective_from_execution(execution)
            signature = str(execution.signature or "")
            code_text = str(traj.code or "")
            has_code = len(code_text.strip()) > 20
            has_valid_obj = self._is_valid_objective(obj_answer)
            obj_in_bounds = self._objective_within_bounds(obj_answer, base_obj_bounds)
            r1_eligible = bool(has_code and effective_success and has_valid_obj and obj_in_bounds)
            evals.append(
                {
                    "trajectory": traj,
                    "execution": execution,
                    "exec_cache_hit": bool(exec_cache_hit),
                    "strict_success": strict_success,
                    "effective_success": effective_success,
                    "obj_answer": obj_answer,
                    "signature": signature,
                    "has_code": has_code,
                    "code_len": int(len(code_text.strip())),
                    "has_valid_obj": has_valid_obj,
                    "obj_in_bounds": obj_in_bounds,
                    "r1_eligible": r1_eligible,
                }
            )

        # R1 consensus is numeric-only and only uses rollouts that have:
        # 1) non-trivial code, 2) effective execution success, 3) valid parsed objective.
        group_numeric = [
            float(e["obj_answer"])
            for e in evals
            if e["r1_eligible"] and isinstance(e["obj_answer"], (int, float))
        ]
        group_signature: list[str] = []

        global_numeric_candidates = list(stage_numeric_hist) + list(group_numeric)
        global_signature_candidates = list(stage_signature_hist) + list(group_signature)

        current_numeric_label, current_numeric_votes = self._majority_numeric_reference(group_numeric, rel_tol)
        global_numeric_label, global_numeric_votes = self._majority_numeric_reference(global_numeric_candidates, rel_tol)

        current_signature_label, current_signature_votes = self._majority_signature_reference(group_signature)
        global_signature_label, global_signature_votes = self._majority_signature_reference(global_signature_candidates)

        final_mode = "none"
        final_label_numeric: float | None = None
        final_label_signature: str | None = None

        if current_numeric_label is not None or global_numeric_label is not None:
            final_mode = "numeric"
            final_label_numeric = self._resolve_hierarchical_label_numeric(
                current_label=current_numeric_label,
                global_label=global_numeric_label,
                group_values=group_numeric,
                rel_tol=rel_tol,
            )

        rewards: list[RewardBreakdown] = []
        for e in evals:
            traj = e["trajectory"]
            execution = e["execution"]
            strict_success = bool(e["strict_success"])
            effective_success = bool(e["effective_success"])
            obj_answer = e["obj_answer"]
            signature = str(e["signature"])
            has_code = bool(e.get("has_code", False))
            has_valid_obj = bool(e.get("has_valid_obj", False))
            obj_in_bounds = bool(e.get("obj_in_bounds", True))
            r1_eligible = bool(e.get("r1_eligible", False))
            code_len = int(e.get("code_len", 0))

            r1 = 0.0
            # Hard gate for r1:
            # - no code => r1=0
            # - execution failed => r1=0
            # - objective missing/invalid => r1=0
            if r1_eligible and final_mode == "numeric" and final_label_numeric is not None:
                if isinstance(obj_answer, (int, float)) and self._within_rel_tol(float(obj_answer), final_label_numeric, rel_tol):
                    r1 = 1.0

            common_meta = {
                "r1_debug": {
                    "stage": stage_key,
                    "mode": final_mode,
                    "strict_success": strict_success,
                    "effective_success": effective_success,
                    "obj_answer": obj_answer,
                    "signature": signature,
                    "has_code": has_code,
                    "code_len": int(code_len),
                    "has_valid_obj": has_valid_obj,
                    "obj_in_bounds": obj_in_bounds,
                    "r1_eligible": r1_eligible,
                    "base_obj_bounds": base_obj_bounds,
                    "current_numeric_label": current_numeric_label,
                    "current_numeric_votes": int(current_numeric_votes),
                    "global_numeric_label": global_numeric_label,
                    "global_numeric_votes": int(global_numeric_votes),
                    "final_numeric_label": final_label_numeric,
                    "current_signature_label": current_signature_label,
                    "current_signature_votes": int(current_signature_votes),
                    "global_signature_label": global_signature_label,
                    "global_signature_votes": int(global_signature_votes),
                    "final_signature_label": final_label_signature,
                    "numeric_pool_size_before": int(len(stage_numeric_hist)),
                    "signature_pool_size_before": int(len(stage_signature_hist)),
                    "group_numeric_size": int(len(group_numeric)),
                    "group_signature_size": int(len(group_signature)),
                    "rel_tol": rel_tol,
                },
                "exec_elapsed_sec": float(execution.elapsed_sec),
                "exec_cache_hit": bool(e["exec_cache_hit"]),
                "obj_answer": obj_answer,
                "execution": self._execution_summary(execution, obj_answer=obj_answer, effective_success=effective_success),
            }

            if r1 == 1.0:
                r3, r3_meta = self._compute_r3_with_details(traj)
                total = self.combine_rewards(r1=r1, r2=0.0, r3=r3)
                rewards.append(
                    RewardBreakdown(
                        r1=r1,
                        r2=0.0,
                        r3=r3,
                        total=total,
                        consensus_signature=self._consensus_key(final_mode, final_label_numeric, final_label_signature),
                        execution_success=effective_success,
                        robustness_success=(r3 == 1.0),
                        metadata={"r3": r3_meta, **common_meta},
                    )
                )
            else:
                r2 = self.compute_r2(effective_success)
                total = self.combine_rewards(r1=r1, r2=r2, r3=0.0)
                rewards.append(
                    RewardBreakdown(
                        r1=r1,
                        r2=r2,
                        r3=0.0,
                        total=total,
                        consensus_signature=self._consensus_key(final_mode, final_label_numeric, final_label_signature),
                        execution_success=effective_success,
                        robustness_success=False,
                        metadata=common_meta,
                    )
                )

        self._update_stage_pools(stage_key=stage_key, group_numeric=group_numeric, group_signature=group_signature)
        return rewards

    def finalize_group(self, trajectories: list[Trajectory]) -> list[Trajectory]:
        rewards = self.score_rollout_group(stage=Stage.CODE, trajectories=trajectories, explored=[])
        for traj, reward in zip(trajectories, rewards, strict=False):
            traj.reward = reward
        return trajectories

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

        if bool(self.task.instance.get("__r3_disable__", False)):
            return 1.0, {
                "enabled": False,
                "reason": "disabled_by_precompute_failure",
                "num_cases": 0,
            }

        tests, source = self._load_r3_tests()
        if not tests:
            return 0.0, {
                "enabled": True,
                "source": source,
                "reason": "no_perturb_cases",
                "num_cases": 0,
            }

        details: list[dict] = []
        for idx, raw_case in enumerate(tests):
            case_instance, case_bounds, case_meta = self._normalize_r3_case(raw_case)
            res = self.executor.run(trajectory.code, case_instance)
            effective_success = self._effective_execution_success(res)
            case_obj = self._extract_objective_from_execution(res)
            obj_in_bounds = self._objective_within_bounds(case_obj, case_bounds)

            detail = {
                "case_index": idx,
                "success": bool(res.success),
                "effective_success": bool(effective_success),
                "signature": res.signature,
                "elapsed_sec": float(res.elapsed_sec),
                "obj_answer": case_obj,
                "obj_in_bounds": obj_in_bounds,
                "obj_bounds": case_bounds,
                "changes": list(case_meta.get("changes", [])) if isinstance(case_meta, dict) else [],
                "case_id": str(case_meta.get("case_id", f"case_{idx}")) if isinstance(case_meta, dict) else f"case_{idx}",
            }
            details.append(detail)

            if (not effective_success) or (not obj_in_bounds):
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

    def _load_r3_tests(self) -> tuple[list[dict[str, Any]], str]:
        precomputed = self._precomputed_r3_cases(self.config.robustness_cases)
        if precomputed:
            return precomputed, "precomputed"

        if bool(self.task.instance.get("__r3_precompute_required__", False)):
            # Strict mode: if precompute was required but failed, do not fallback.
            return [], "precompute_required_no_cases"

        tests = self.backend.generate_test_instances(self.task, self.config.robustness_cases)
        if tests:
            return list(tests), "backend"

        tests = generate_perturbed_instances_from_map(
            self.task.instance,
            self.task.perturbation_map,
            self.config.robustness_cases,
        )
        return list(tests), "heuristic"

    def _precomputed_r3_cases(self, k: int) -> list[dict[str, Any]]:
        raw = self.task.instance.get("__r3_test_cases__") if isinstance(self.task.instance, dict) else None
        if not isinstance(raw, list):
            return []
        out: list[dict[str, Any]] = []
        for item in raw[: max(1, int(k))]:
            if isinstance(item, dict):
                out.append(item)
        return out

    def _normalize_r3_case(self, raw_case: dict[str, Any] | Any) -> tuple[dict[str, Any], dict[str, float | None], dict[str, Any]]:
        if isinstance(raw_case, dict) and isinstance(raw_case.get("instance"), dict):
            instance = dict(raw_case.get("instance", {}))
            bounds = self._normalize_bounds_dict(raw_case.get("obj_bounds"))
            meta = {
                "changes": list(raw_case.get("changes", [])) if isinstance(raw_case.get("changes"), list) else [],
                "case_id": raw_case.get("case_id", ""),
            }
            return instance, bounds, meta

        if isinstance(raw_case, dict):
            instance = dict(raw_case)
            bounds = self._normalize_bounds_dict(None)
            case_meta = raw_case.get("__perturbation__") if isinstance(raw_case.get("__perturbation__"), dict) else {}
            meta = {
                "changes": list(case_meta.get("changes", [])) if isinstance(case_meta.get("changes"), list) else [],
                "case_id": case_meta.get("case_id", ""),
            }
            return instance, bounds, meta

        return {}, self._normalize_bounds_dict(None), {}

    def _base_obj_bounds(self) -> dict[str, float | None]:
        from_instance = None
        if isinstance(self.task.instance, dict):
            from_instance = self.task.instance.get("__r3_base_obj_bounds__")
        if from_instance is not None:
            return self._normalize_bounds_dict(from_instance)

        if isinstance(self.task.perturbation_map, dict):
            maybe = self.task.perturbation_map.get("base_obj_bounds")
            if maybe is not None:
                return self._normalize_bounds_dict(maybe)

        return {"lower": None, "upper": None}

    @staticmethod
    def _normalize_bounds_dict(value: Any) -> dict[str, float | None]:
        if not isinstance(value, dict):
            return {"lower": None, "upper": None}

        def _num(x: Any) -> float | None:
            if isinstance(x, bool):
                return None
            if isinstance(x, (int, float)):
                x = float(x)
                return x if math.isfinite(x) else None
            if isinstance(x, str):
                s = x.strip().replace(",", "")
                if not s:
                    return None
                try:
                    v = float(s)
                    return v if math.isfinite(v) else None
                except Exception:
                    return None
            return None

        lo = _num(value.get("lower"))
        hi = _num(value.get("upper"))
        if lo is not None and hi is not None and lo > hi:
            lo, hi = hi, lo
        return {"lower": lo, "upper": hi}

    def _objective_within_bounds(self, obj_answer: float | None, bounds: dict[str, float | None] | None) -> bool:
        if not self._is_valid_objective(obj_answer):
            return False

        if not isinstance(bounds, dict):
            return True
        lo = bounds.get("lower")
        hi = bounds.get("upper")
        val = float(obj_answer)

        eps = 1e-9
        if isinstance(lo, (int, float)) and val < float(lo) - eps:
            return False
        if isinstance(hi, (int, float)) and val > float(hi) + eps:
            return False
        return True

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

    def _update_stage_pools(self, stage_key: str, group_numeric: list[float], group_signature: list[str]) -> None:
        if group_numeric:
            self._stage_numeric_pool.setdefault(stage_key, []).extend(float(x) for x in group_numeric)
        elif group_signature:
            sig_pool = self._stage_signature_pool.setdefault(stage_key, [])
            for s in group_signature:
                if s and s != "EXEC_ERROR":
                    sig_pool.append(s)

    def _resolve_hierarchical_label_numeric(
        self,
        current_label: float | None,
        global_label: float | None,
        group_values: list[float],
        rel_tol: float,
    ) -> float | None:
        if current_label is None and global_label is None:
            return None
        if current_label is None:
            return global_label
        if global_label is None:
            return current_label
        if self._within_rel_tol(current_label, global_label, rel_tol):
            return global_label

        if any(self._within_rel_tol(v, global_label, rel_tol) for v in group_values):
            return global_label
        return current_label

    @staticmethod
    def _resolve_hierarchical_label_signature(
        current_label: str | None,
        global_label: str | None,
        group_values: list[str],
    ) -> str | None:
        if not current_label and not global_label:
            return None
        if not current_label:
            return global_label
        if not global_label:
            return current_label
        if current_label == global_label:
            return global_label

        if any(v == global_label for v in group_values):
            return global_label
        return current_label

    @staticmethod
    def _consensus_key(mode: str, numeric_label: float | None, signature_label: str | None) -> str:
        if mode == "numeric" and numeric_label is not None:
            return f"numeric:{numeric_label:.6f}"
        if mode == "signature" and signature_label:
            return f"signature:{signature_label}"
        return ""

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

    @staticmethod
    def _majority_signature_reference(values: list[str]) -> tuple[str | None, int]:
        if not values:
            return None, 0
        c = Counter(values)
        label, votes = c.most_common(1)[0]
        return label, int(votes)

    def _effective_execution_success(self, execution: ExecutionResult) -> bool:
        if bool(execution.success):
            return True

        # If we can parse a valid objective value from output/stdout/stderr,
        # treat this as effective execution success even when wrapper shape checks fail.
        obj_answer = self._extract_objective_from_execution(execution)
        if self._is_valid_objective(obj_answer):
            return True

        stdout_text = str(execution.stdout or "")
        lowered = stdout_text.lower()
        if any(marker in lowered for marker in self._gurobi_success_markers):
            return True

        # Additional textual fallback for common report formats.
        if "optimal value" in lowered or "objective value" in lowered:
            return True

        return False

    @staticmethod
    def _execution_summary(
        execution: ExecutionResult,
        obj_answer: float | None = None,
        effective_success: bool | None = None,
    ) -> dict:
        stdout_text = str(execution.stdout or "")
        lowered = stdout_text.lower()
        marker_hit = "optimal solution found" in lowered or "model is solved to optimality" in lowered
        obj_hit = TTRLRewardCalculator._is_valid_objective(obj_answer)
        eff = bool(effective_success) if effective_success is not None else bool(execution.success or marker_hit or obj_hit)
        return {
            "success": bool(execution.success),
            "effective_success": eff,
            "solver_success_marker_hit": marker_hit,
            "objective_text_marker_hit": bool("optimal value" in lowered or "objective value" in lowered),
            "parsed_obj_answer": obj_answer,
            "objective_parsed_success": obj_hit,
            "signature": str(execution.signature or ""),
            "error_type": str(execution.error_type or ""),
            "output": TTRLRewardCalculator._jsonable(execution.output),
            "stdout_tail": TTRLRewardCalculator._truncate_text(execution.stdout, 2000),
            "stderr_tail": TTRLRewardCalculator._truncate_text(execution.stderr, 2000),
        }

    @staticmethod
    def _is_valid_objective(value: float | None) -> bool:
        if not isinstance(value, (int, float)):
            return False
        v = float(value)
        return math.isfinite(v)

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
