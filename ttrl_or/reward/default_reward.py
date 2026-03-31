from __future__ import annotations

import hashlib
import math
import json
import re
from dataclasses import dataclass, field
from typing import Any

from ttrl_or.config import RewardConfig
from ttrl_or.model.backend import PolicyBackend
from ttrl_or.reward.base import RewardCalculator
from ttrl_or.reward.clusters import SemanticCluster, StructuralCluster
from ttrl_or.reward.executor import PythonCodeExecutor
from ttrl_or.reward.perturbation import generate_perturbed_instances_from_map
from ttrl_or.types import ExecutionResult, ModelInfo, OptimizationTask, RewardBreakdown, Stage, Trajectory


@dataclass(slots=True)
class TTRLRewardCalculator(RewardCalculator):
    task: OptimizationTask
    backend: PolicyBackend
    config: RewardConfig
    executor: PythonCodeExecutor = field(init=False)
    _exec_cache: dict[str, ExecutionResult] = field(default_factory=dict, init=False)
    # Cluster management for the entire task (episode)
    _semantic_cluster: SemanticCluster = field(init=False)
    _structural_cluster: StructuralCluster = field(init=False)
    _current_iteration: int = field(default=0, init=False)
    _gurobi_success_markers: tuple[str, ...] = field(
        default=("optimal solution found", "model is solved to optimality"),
        init=False,
    )

    def __post_init__(self) -> None:
        self.executor = PythonCodeExecutor(
            timeout_sec=self.config.code_timeout_sec,
            mode=self.config.code_executor_mode,
        )
        # Initialize clusters for this task
        self._semantic_cluster = SemanticCluster(
            rel_tol=self.config.global_consensus_rel_tol,
        )
        self._structural_cluster = StructuralCluster(
            decay=self.config.r4_decay,
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
        """Score a group of trajectories using the new r1/r2/r3/r4 reward system."""
        if not trajectories:
            return []

        self._current_iteration += 1
        current_iter = self._current_iteration
        base_obj_bounds = self._base_obj_bounds()

        # Phase 1: Execute and evaluate all trajectories
        evals: list[dict[str, Any]] = []
        for traj in trajectories:
            execution, exec_cache_hit = self._execute(traj)
            strict_success = bool(execution.success)
            effective_success = self._effective_execution_success(execution)
            obj_answer = self._extract_objective_from_execution(execution)
            code_text = str(traj.code or "")
            has_code = len(code_text.strip()) > 20
            has_valid_obj = self._is_valid_objective(obj_answer)
            obj_in_bounds = self._objective_within_bounds(obj_answer, base_obj_bounds)
            
            # r1 eligibility: code + effective_success + valid_obj + in_bounds
            r1_eligible = bool(has_code and effective_success and has_valid_obj and obj_in_bounds)
            
            # Extract model info for r4
            model_info = execution.model_info
            feature_tuple = model_info.feature_tuple() if model_info and model_info.extracted else None
            
            evals.append({
                "trajectory": traj,
                "execution": execution,
                "exec_cache_hit": bool(exec_cache_hit),
                "strict_success": strict_success,
                "effective_success": effective_success,
                "obj_answer": obj_answer,
                "has_code": has_code,
                "code_len": int(len(code_text.strip())),
                "has_valid_obj": has_valid_obj,
                "obj_in_bounds": obj_in_bounds,
                "r1_eligible": r1_eligible,
                "model_info": model_info,
                "feature_tuple": feature_tuple,
            })

        # Phase 2: Compute rewards for each trajectory
        rewards: list[RewardBreakdown] = []
        
        for e in evals:
            traj = e["trajectory"]
            execution = e["execution"]
            effective_success = bool(e["effective_success"])
            obj_answer = e["obj_answer"]
            r1_eligible = bool(e["r1_eligible"])
            feature_tuple = e["feature_tuple"]
            model_info = e["model_info"]
            
            # ─── r2: Execution success (computed first, r3 depends on it) ───
            r2 = 1.0 if effective_success else 0.0
            
            # ─── r1: Semantic cluster ratio ───
            r1 = 0.0
            r1_debug: dict[str, Any] = {
                "r1_eligible": r1_eligible,
                "obj_answer": obj_answer,
                "effective_success": effective_success,
                "has_valid_obj": e["has_valid_obj"],
                "obj_in_bounds": e["obj_in_bounds"],
            }
            
            if r1_eligible and isinstance(obj_answer, (int, float)):
                # Add to semantic cluster
                leader = self._semantic_cluster.add_sample(
                    obj_value=float(obj_answer),
                    iteration=current_iter,
                )
                # Compute r1
                r1, r1_cluster_debug = self._semantic_cluster.compute_r1(
                    obj_value=float(obj_answer),
                    alpha=self.config.r1_alpha,
                    min_k=self.config.r1_min_clusters,
                )
                r1_debug.update(r1_cluster_debug)
            
            # ─── r4: Structural consensus ───
            r4 = 0.0
            r4_debug: dict[str, Any] = {"enabled": bool(self.config.enable_r4_reward)}
            
            # Add to structural cluster (all samples, including failed)
            self._structural_cluster.add_sample(
                feature_tuple=feature_tuple,
                iteration=current_iter,
            )
            
            if self.config.enable_r4_reward and feature_tuple is not None:
                r4, r4_debug = self._structural_cluster.compute_r4(
                    feature_tuple=feature_tuple,
                    current_iteration=current_iter,
                    alpha=self.config.r4_alpha,
                    k=self.config.r4_k,
                )
            
            # ─── r3: Robustness (only when r2=1.0) ───
            r3 = 0.0
            r3_meta: dict[str, Any] = {"triggered": False}
            
            if r2 == 1.0:
                r3, r3_meta = self._compute_r3_with_details(traj)
                r3_meta["triggered"] = True
            
            # ─── Combine rewards ───
            total = self.combine_rewards(
                r1=r1,
                r2=r2,
                r3=r3,
                r4=r4,
                r3_weight=self.config.r3_weight,
                r4_weight=self.config.r4_weight,
            )
            
            # Build metadata
            common_meta = {
                "r1_debug": r1_debug,
                "r4_debug": r4_debug,
                "r3": r3_meta,
                "iteration": current_iter,
                "exec_elapsed_sec": float(execution.elapsed_sec),
                "exec_cache_hit": bool(e["exec_cache_hit"]),
                "obj_answer": obj_answer,
                "base_obj_bounds": base_obj_bounds,
                "model_info": {
                    "extracted": bool(model_info and model_info.extracted),
                    "model_sense": model_info.model_sense if model_info else None,
                    "num_vars": model_info.num_vars if model_info else None,
                    "num_bin_vars": model_info.num_bin_vars if model_info else None,
                    "num_int_vars": model_info.num_int_vars if model_info else None,
                } if model_info else None,
                "execution": self._execution_summary(
                    execution,
                    obj_answer=obj_answer,
                    effective_success=effective_success,
                ),
            }
            
            rewards.append(RewardBreakdown(
                r1=r1,
                r2=r2,
                r3=r3,
                r4=r4,
                total=total,
                consensus_signature=f"semantic:{r1_debug.get('cluster_leader', '')}" if r1_eligible else "",
                execution_success=effective_success,
                robustness_success=(r3 == 1.0),
                metadata=common_meta,
            ))
        
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
    def combine_rewards(
        r1: float,
        r2: float,
        r3: float,
        r4: float,
        r3_weight: float = 0.3,
        r4_weight: float = 0.2,
    ) -> float:
        """
        Combine rewards using the new formula.
        
        Reward_total = max(0, r1 + r3_weight * r3 * r2 + r4_weight * r4)
        
        Note: r3 is only meaningful when r2=1.0 (execution succeeded)
        """
        total = r1 + r3_weight * r3 * r2 + r4_weight * r4
        return max(0.0, total)

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
