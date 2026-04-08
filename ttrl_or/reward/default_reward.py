from __future__ import annotations

import hashlib
import json
import math
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from ttrl_or.config import RewardConfig
from ttrl_or.model.backend import PolicyBackend
from ttrl_or.reward.base import RewardCalculator
from ttrl_or.reward.clusters import SemanticCluster, StructuralCluster
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
    _exec_cache_lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
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
        self._semantic_cluster = SemanticCluster(
            rel_tol=self.config.global_consensus_rel_tol,
        )
        self._structural_cluster = StructuralCluster(
            decay=self.config.r4_decay,
        )

    def provisional_reward(self, trajectory: Trajectory, explored: list[Trajectory]) -> RewardBreakdown:
        rewards = self.score_rollout_group(
            stage=Stage.CODE,
            trajectories=[trajectory],
            explored=explored,
            commit=True,
        )
        return rewards[0]

    def score_rollout_group(
        self,
        stage: Stage,
        trajectories: list[Trajectory],
        explored: list[Trajectory],
        commit: bool = True,
    ) -> list[RewardBreakdown]:
        """Score one rollout group with configurable global/local cluster comparison."""
        if not trajectories:
            return []

        group_t0 = time.perf_counter()
        current_iter = self._current_iteration + 1
        if commit:
            self._current_iteration = current_iter

        base_obj_scale = self._base_obj_scale()
        local_scope = self._use_local_cluster_scope()

        exec_group_t0 = time.perf_counter()
        execution_pairs = self._execute_group(trajectories)
        execution_group_sec = float(time.perf_counter() - exec_group_t0)

        evals: list[dict[str, Any]] = []
        for traj, (execution, exec_cache_hit) in zip(trajectories, execution_pairs, strict=False):
            effective_success = self._effective_execution_success(execution)
            obj_answer = self._extract_objective_from_execution(execution)
            code_text = str(traj.code or "")
            has_code = len(code_text.strip()) > 20
            has_valid_obj = self._is_valid_objective(obj_answer)
            obj_in_bounds = self._objective_matches_scale(obj_answer, base_obj_scale)
            r1_eligible = bool(has_code and effective_success and has_valid_obj)

            model_info = execution.model_info
            feature_tuple = model_info.feature_tuple() if model_info and model_info.extracted else None

            evals.append(
                {
                    "trajectory": traj,
                    "execution": execution,
                    "exec_cache_hit": bool(exec_cache_hit),
                    "effective_success": bool(effective_success),
                    "obj_answer": obj_answer,
                    "has_code": has_code,
                    "code_len": int(len(code_text.strip())),
                    "has_valid_obj": has_valid_obj,
                    "obj_in_bounds": obj_in_bounds,
                    "r1_eligible": r1_eligible,
                    "model_info": model_info,
                    "feature_tuple": feature_tuple,
                    "semantic_leader": None,
                }
            )

        semantic_t0 = time.perf_counter()
        semantic_group_counts, semantic_group_total = self._prepare_semantic_group(evals, local_scope=local_scope)
        semantic_prepare_sec = float(time.perf_counter() - semantic_t0)
        structural_t0 = time.perf_counter()
        structural_group_counts = self._prepare_structural_group(evals)
        structural_prepare_sec = float(time.perf_counter() - structural_t0)
        structural_group_total = len(evals)

        rewards: list[RewardBreakdown] = []
        reward_loop_t0 = time.perf_counter()
        r3_total_sec = 0.0
        for e in evals:
            traj = e["trajectory"]
            execution = e["execution"]
            effective_success = bool(e["effective_success"])
            obj_answer = e["obj_answer"]
            r1_eligible = bool(e["r1_eligible"])
            feature_tuple = e["feature_tuple"]
            model_info = e["model_info"]
            semantic_leader = e.get("semantic_leader")

            r2 = 1.0 if self._r2_execution_success(execution) else 0.0

            r1 = 0.0
            r1_debug: dict[str, Any] = {
                "cluster_scope": self.config.cluster_scope,
                "r1_eligible": r1_eligible,
                "obj_answer": obj_answer,
                "effective_success": effective_success,
                "has_valid_obj": e["has_valid_obj"],
                "obj_in_bounds": e["obj_in_bounds"],
                "has_code": e["has_code"],
                "code_len": e["code_len"],
                "base_obj_scale": base_obj_scale,
            }
            if r1_eligible and isinstance(obj_answer, (int, float)) and isinstance(semantic_leader, (int, float)):
                if local_scope:
                    r1, r1_cluster_debug = self._compute_local_r1(
                        leader=float(semantic_leader),
                        group_counts=semantic_group_counts,
                        valid_total=semantic_group_total,
                    )
                else:
                    r1, r1_cluster_debug = self._semantic_cluster.preview_r1(
                        leader=float(semantic_leader),
                        alpha=self.config.r1_alpha,
                        min_k=self.config.r1_min_clusters,
                        additional_clusters=semantic_group_counts,
                        additional_total=semantic_group_total,
                    )
                r1_debug.update(r1_cluster_debug)

            r4 = 0.0
            reward_gate, structure_gate_debug = self._structure_gate(model_info)
            r4_debug: dict[str, Any] = {
                "enabled": bool(self.config.enable_r4_reward),
                "cluster_scope": self.config.cluster_scope,
                "structure_gate": reward_gate,
                "structure_gate_debug": structure_gate_debug,
            }
            if self.config.enable_r4_reward and feature_tuple is not None:
                if local_scope:
                    r4, local_r4_debug = self._compute_local_r4(
                        feature_tuple=feature_tuple,
                        group_feature_counts=structural_group_counts,
                        group_total_count=structural_group_total,
                    )
                else:
                    r4, local_r4_debug = self._structural_cluster.preview_r4(
                        feature_tuple=feature_tuple,
                        current_iteration=current_iter,
                        alpha=self.config.r4_alpha,
                        k=self.config.r4_k,
                        group_feature_counts=structural_group_counts,
                        group_total_count=structural_group_total,
                    )
                r4_debug.update(local_r4_debug)

            r3 = 0.0
            r3_meta: dict[str, Any] = {"triggered": False}
            if bool(self.config.enable_r3_reward):
                r3_t0 = time.perf_counter()
                r3, r3_meta = self._compute_r3_with_details(traj)
                r3_total_sec += float(time.perf_counter() - r3_t0)
                r3_meta["triggered"] = True

            r1_weight_scale = 1.0
            r1_obj_scale_penalized = False
            if bool(self.config.enable_r3_reward) and (not bool(e["obj_in_bounds"])):
                r1_weight_scale = float(self.config.r1_obj_scale_fail_multiplier)
                r1_obj_scale_penalized = True

            # If r3 is disabled, fold r3 weight into r1 to keep total weight mass stable.
            r3_enabled = bool(self.config.enable_r3_reward)
            base_r1_weight = float(self.config.r1_weight)
            base_r3_weight = float(self.config.r3_weight)
            r1_weight_effective = base_r1_weight + (0.0 if r3_enabled else base_r3_weight)
            r3_weight_effective = base_r3_weight if r3_enabled else 0.0

            total = self.combine_rewards(
                r1=r1,
                r2=r2,
                r3=r3,
                r4=r4,
                reward_gate=reward_gate,
                r1_weight=r1_weight_effective,
                r2_weight=self.config.r2_weight,
                r1_weight_scale=r1_weight_scale,
                r3_weight=r3_weight_effective,
                r4_weight=self.config.r4_weight,
            )

            common_meta = {
                "reward_cluster_scope": self.config.cluster_scope,
                "r1_debug": r1_debug,
                "r4_debug": r4_debug,
                "reward_gate": reward_gate,
                "total_reward_formula": "total_r = (w1*r1*r1_weight_scale) + (w2*r2) + (w3*r3) + (w4*r4); total=max(0,total_r)",
                "total_reward_weights": {
                    "w1_r1": float(r1_weight_effective),
                    "w2_r2": float(self.config.r2_weight),
                    "r1_obj_scale_fail_multiplier": float(self.config.r1_obj_scale_fail_multiplier),
                    "w3_r3": float(r3_weight_effective),
                    "w4_r4": float(self.config.r4_weight),
                    "w1_base_r1": float(base_r1_weight),
                    "w3_base_r3": float(base_r3_weight),
                    "r3_enabled": bool(r3_enabled),
                    "r3_weight_folded_into_r1": bool(not r3_enabled),
                },
                "total_reward_terms": {
                    "r1": float(r1),
                    "r2": float(r2),
                    "r3": float(r3),
                    "r1_weight_scale": float(r1_weight_scale),
                    "r1_obj_scale_penalized": bool(r1_obj_scale_penalized),
                    "r1_effective_weight": float(r1_weight_effective) * float(r1_weight_scale),
                    "r3_effective": float(r3),
                    "r4": float(r4),
                },
                "structure_gate": structure_gate_debug,
                "r3": r3_meta,
                "iteration": current_iter,
                "iteration_committed": bool(commit),
                "exec_elapsed_sec": float(execution.elapsed_sec),
                "exec_cache_hit": bool(e["exec_cache_hit"]),
                "lp_injection_applied": bool(getattr(execution, "lp_injection_applied", False)),
                "obj_answer": obj_answer,
                "base_obj_scale": base_obj_scale,
                "base_obj_bounds": base_obj_scale,
                "model_info": {
                    "extracted": bool(model_info and model_info.extracted),
                    "model_sense": model_info.model_sense if model_info else None,
                    "num_vars": model_info.num_vars if model_info else None,
                    "num_bin_vars": model_info.num_bin_vars if model_info else None,
                    "num_int_vars": model_info.num_int_vars if model_info else None,
                    "num_constrs": model_info.num_constrs if model_info else None,
                    "has_objective": model_info.has_objective if model_info else None,
                    "has_constraints": model_info.has_constraints if model_info else None,
                    "has_variables": model_info.has_variables if model_info else None,
                    "lp_injection_applied": bool(getattr(execution, "lp_injection_applied", False)),
                }
                if model_info
                else None,
                "execution": self._execution_summary(
                    execution,
                    obj_answer=obj_answer,
                    effective_success=effective_success,
                ),
            }

            rewards.append(
                RewardBreakdown(
                    r1=r1,
                    r2=r2,
                    r3=r3,
                    r4=r4,
                    total=total,
                    consensus_signature=(f"semantic:{r1_debug.get('cluster_leader', '')}" if r1_eligible else ""),
                    execution_success=effective_success,
                    robustness_success=(r3 == 1.0),
                    metadata=common_meta,
                )
            )

        reward_loop_sec = float(time.perf_counter() - reward_loop_t0)
        commit_t0 = time.perf_counter()
        if commit and (not local_scope):
            for e in evals:
                if bool(e["r1_eligible"]) and isinstance(e["obj_answer"], (int, float)) and isinstance(e.get("semantic_leader"), (int, float)):
                    self._semantic_cluster.commit_sample(
                        obj_value=float(e["obj_answer"]),
                        leader=float(e["semantic_leader"]),
                        iteration=current_iter,
                    )

            for e in evals:
                self._structural_cluster.add_sample(
                    feature_tuple=e["feature_tuple"],
                    iteration=current_iter,
                )
        commit_sec = float(time.perf_counter() - commit_t0)
        reward_group_timing = {
            "execution_group_sec": execution_group_sec,
            "semantic_prepare_sec": semantic_prepare_sec,
            "structural_prepare_sec": structural_prepare_sec,
            "reward_loop_sec": reward_loop_sec,
            "r3_total_sec": r3_total_sec,
            "commit_sec": commit_sec,
            "total_sec": float(time.perf_counter() - group_t0),
            "num_trajectories": len(trajectories),
        }
        for reward in rewards:
            if reward.metadata is None:
                reward.metadata = {}
            reward.metadata["reward_timing"] = dict(reward_group_timing)

        return rewards

    def finalize_group(self, trajectories: list[Trajectory]) -> list[Trajectory]:
        rewards = self.score_rollout_group(
            stage=Stage.CODE,
            trajectories=trajectories,
            explored=[],
            commit=False,
        )
        for traj, reward in zip(trajectories, rewards, strict=False):
            traj.reward = reward
        return trajectories

    @staticmethod
    def compute_r2(execution_success: bool) -> float:
        return 1.0 if execution_success else 0.0

    def _r2_execution_success(self, execution: ExecutionResult) -> bool:
        if bool(execution.success):
            return True

        if str(execution.error_type or "") == "Timeout":
            return False

        obj_answer = self._extract_objective_from_execution(execution)
        if self._is_valid_objective(obj_answer):
            return True

        stdout_text = str(execution.stdout or "")
        lowered_stdout = stdout_text.lower()
        if any(marker in lowered_stdout for marker in self._gurobi_success_markers):
            return True
        if "optimal value" in lowered_stdout or "objective value" in lowered_stdout:
            return True

        stderr_text = str(execution.stderr or "")
        lowered_stderr = stderr_text.lower()
        runtime_error_markers = (
            "traceback",
            "syntaxerror",
            "nameerror",
            "typeerror",
            "valueerror",
            "attributeerror",
            "keyerror",
            "indexerror",
            "importerror",
            "modulenotfounderror",
            "zerodivisionerror",
            "assertionerror",
            "runtimeerror",
        )
        if any(marker in lowered_stderr for marker in runtime_error_markers):
            return False

        # If the code ran to normal process exit without runtime/syntax errors,
        # malformed or missing JSON output should still count as r2=1.
        if not lowered_stderr.strip():
            return True
        return False

    def compute_r3(self, trajectory: Trajectory) -> float:
        score, _ = self._compute_r3_with_details(trajectory)
        return score

    @staticmethod
    def combine_rewards(
        r1: float,
        r2: float,
        r3: float,
        r4: float,
        reward_gate: float = 1.0,
        r1_weight: float = 0.6,
        r2_weight: float = 0.1,
        r1_weight_scale: float = 1.0,
        r3_weight: float = 0.2,
        r4_weight: float = 0.1,
    ) -> float:
        # Single point to edit total reward composition:
        # total_r_raw = (w1*r1*r1_weight_scale) + (w2*r2) + (w3*r3) + (w4*r4)
        # no reward gating: total_r = max(0, total_r_raw)
        total_r_raw = (
            float(r1_weight) * float(r1_weight_scale) * float(r1)
            + float(r2_weight) * float(r2)
            + float(r3_weight) * float(r3)
            + float(r4_weight) * float(r4)
        )
        total_r = total_r_raw
        return max(0.0, total_r)

    def _use_local_cluster_scope(self) -> bool:
        return str(self.config.cluster_scope or "global").strip().lower() == "local"

    def _prepare_semantic_group(self, evals: list[dict[str, Any]], *, local_scope: bool) -> tuple[dict[float, int], int]:
        group_counts: dict[float, int] = {}
        group_new_leaders: list[float] = []

        eligible_items: list[tuple[int, float]] = []
        for idx, item in enumerate(evals):
            if bool(item.get("r1_eligible")) and isinstance(item.get("obj_answer"), (int, float)):
                eligible_items.append((idx, float(item["obj_answer"])))

        eligible_items.sort(key=lambda x: (x[1], x[0]))

        for idx, obj_value in eligible_items:
            leader = None if local_scope else self._semantic_cluster.matching_leader(obj_value)
            if leader is None:
                for candidate in group_new_leaders:
                    if self._within_rel_tol(obj_value, candidate, self.config.global_consensus_rel_tol):
                        leader = candidate
                        break
            if leader is None:
                leader = obj_value
                group_new_leaders.append(leader)

            evals[idx]["semantic_leader"] = float(leader)
            group_counts[float(leader)] = group_counts.get(float(leader), 0) + 1

        return group_counts, len(eligible_items)

    def _compute_local_r1(
        self,
        *,
        leader: float,
        group_counts: dict[float, int],
        valid_total: int,
    ) -> tuple[float, dict[str, Any]]:
        count_i = int(group_counts.get(leader, 0))
        n = int(valid_total)
        num_clusters = len(group_counts)
        k = max(num_clusters, int(self.config.r1_min_clusters))

        if n <= 0:
            r1 = 0.0
        else:
            r1 = (count_i + float(self.config.r1_alpha)) / (n + float(self.config.r1_alpha) * k)

        return r1, {
            "cluster_leader": leader,
            "cluster_count": count_i,
            "total_valid": n,
            "num_clusters": num_clusters,
            "k": k,
            "alpha": float(self.config.r1_alpha),
            "r1": r1,
            "scope": "local",
        }

    @staticmethod
    def _prepare_structural_group(evals: list[dict[str, Any]]) -> dict[tuple[int, int, int, int], int]:
        group_counts: dict[tuple[int, int, int, int], int] = {}
        for item in evals:
            feature_tuple = item.get("feature_tuple")
            if isinstance(feature_tuple, tuple) and len(feature_tuple) == 4:
                group_counts[feature_tuple] = group_counts.get(feature_tuple, 0) + 1
        return group_counts

    def _compute_local_r4(
        self,
        *,
        feature_tuple: tuple[int, int, int, int],
        group_feature_counts: dict[tuple[int, int, int, int], int],
        group_total_count: int,
    ) -> tuple[float, dict[str, Any]]:
        n = int(group_total_count)
        if n <= 0:
            return 0.0, {"r4": 0.0, "reason": "no_samples", "total_count": n, "scope": "local"}

        model_sense, num_vars, num_bin_vars, num_int_vars = feature_tuple
        psi_d = 0.0
        psi_nv = 0.0
        psi_nb = 0.0
        psi_ni = 0.0

        for group_tuple, count in group_feature_counts.items():
            if not isinstance(group_tuple, tuple) or len(group_tuple) != 4:
                continue
            g_sense, g_nv, g_nb, g_ni = group_tuple
            if g_sense == model_sense:
                psi_d += float(count)
            if g_nv == num_vars:
                psi_nv += float(count)
            if g_nb == num_bin_vars:
                psi_nb += float(count)
            if g_ni == num_int_vars:
                psi_ni += float(count)

        raw_score = math.sqrt(psi_d) + math.sqrt(psi_nv) + math.sqrt(psi_nb) + math.sqrt(psi_ni)
        tuple_cluster_count = len(group_feature_counts)
        k = max(tuple_cluster_count, int(self.config.r4_k))
        max_score = 4.0 * math.sqrt(n + float(self.config.r4_alpha) * k)
        r4 = raw_score / max_score if max_score > 0 else 0.0
        r4 = min(1.0, max(0.0, r4))

        return r4, {
            "r4": r4,
            "feature_tuple": feature_tuple,
            "psi_direction": psi_d,
            "psi_num_vars": psi_nv,
            "psi_num_bin_vars": psi_nb,
            "psi_num_int_vars": psi_ni,
            "raw_score": raw_score,
            "max_score": max_score,
            "total_count": n,
            "num_structural_samples": sum(group_feature_counts.values()),
            "tuple_cluster_count": tuple_cluster_count,
            "alpha": float(self.config.r4_alpha),
            "k": k,
            "scope": "local",
        }

    def _compute_r3_with_details(self, trajectory: Trajectory) -> tuple[float, dict]:
        r3_t0 = time.perf_counter()
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

        normalize_t0 = time.perf_counter()
        normalized_cases = [self._normalize_r3_case(raw_case) for raw_case in tests]
        normalize_cases_sec = float(time.perf_counter() - normalize_t0)

        exec_t0 = time.perf_counter()
        max_workers = min(len(normalized_cases), max(1, min(8, os.cpu_count() or 1)))
        case_instances = [case_instance for case_instance, _, _ in normalized_cases]
        if len(normalized_cases) <= 1:
            exec_results = self.executor.run_many(trajectory.code, case_instances, max_workers=1) if normalized_cases else []
        else:
            exec_results = self.executor.run_many(trajectory.code, case_instances, max_workers=max_workers)

        details: list[dict[str, Any]] = []
        for idx, ((_, case_bounds, case_meta), res) in enumerate(zip(normalized_cases, exec_results, strict=False)):
            effective_success = self._effective_execution_success(res)
            case_obj = self._extract_objective_from_execution(res)
            obj_in_bounds = self._objective_matches_scale(case_obj, case_bounds)
            details.append(
                {
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
            )
        case_exec_wall_sec = float(time.perf_counter() - exec_t0)

        passed_cases = 0
        weighted_pass_sum = 0.0
        obj_scale_scaled_cases = 0
        failed_case_index = None
        scale = float(self.config.r1_obj_scale_fail_multiplier)
        for detail in details:
            effective = bool(detail.get("effective_success"))
            in_bounds = bool(detail.get("obj_in_bounds"))
            passed = effective
            case_score = 0.0
            scaled_by_obj_in_bounds = False
            if effective:
                if in_bounds:
                    case_score = 1.0
                else:
                    case_score = scale
                    scaled_by_obj_in_bounds = True
                    obj_scale_scaled_cases += 1

            detail["passed"] = passed
            detail["r3_case_score"] = float(case_score)
            detail["r3_scaled_by_obj_in_bounds"] = bool(scaled_by_obj_in_bounds)
            weighted_pass_sum += float(case_score)

            if passed:
                passed_cases += 1
            elif failed_case_index is None:
                failed_case_index = int(detail.get("case_index", -1))

        num_cases = len(details)
        r3_score = (float(weighted_pass_sum) / float(num_cases)) if num_cases > 0 else 0.0

        return r3_score, {
            "enabled": True,
            "source": source,
            "num_cases": len(details),
            "requested_cases": int(self.config.robustness_cases),
            "passed_cases": passed_cases,
            "weighted_pass_sum": float(weighted_pass_sum),
            "obj_scale_scaled_cases": int(obj_scale_scaled_cases),
            "failed_case_index": failed_case_index,
            "pass_ratio": r3_score,
            "r3_case_score_mode": "effective_success_with_obj_scale_multiplier",
            "obj_scale_fail_multiplier": float(self.config.r1_obj_scale_fail_multiplier),
            "cases": details,
            "timing": {
                "normalize_cases_sec": normalize_cases_sec,
                "case_exec_wall_sec": case_exec_wall_sec,
                "case_exec_elapsed_sum_sec": float(sum(float(d.get("elapsed_sec", 0.0) or 0.0) for d in details)),
                "num_cases": len(details),
                "max_workers": max_workers,
                "total_sec": float(time.perf_counter() - r3_t0),
            },
        }

    def _execute_group(self, trajectories: list[Trajectory]) -> list[tuple[ExecutionResult, bool]]:
        if not trajectories:
            return []
        if len(trajectories) == 1:
            return [self._execute(trajectories[0])]

        max_workers = min(len(trajectories), max(1, min(8, os.cpu_count() or 1)))
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            return list(pool.map(self._execute, trajectories))

    def _load_r3_tests(self) -> tuple[list[dict[str, Any]], str]:
        precomputed = self._precomputed_r3_cases()
        if precomputed:
            return precomputed, "precomputed"

        if bool(self.task.instance.get("__r3_precompute_required__", False)):
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

    def _precomputed_r3_cases(self) -> list[dict[str, Any]]:
        raw = self.task.instance.get("__r3_test_cases__") if isinstance(self.task.instance, dict) else None
        if not isinstance(raw, list):
            return []
        out: list[dict[str, Any]] = []
        for item in raw:
            if isinstance(item, dict):
                out.append(item)
        return out

    def _normalize_r3_case(self, raw_case: dict[str, Any] | Any) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        if isinstance(raw_case, dict) and isinstance(raw_case.get("instance"), dict):
            instance = dict(raw_case.get("instance", {}))
            scale = self._normalize_scale_spec(raw_case.get("obj_scale") or raw_case.get("obj_bounds"))
            meta = {
                "changes": list(raw_case.get("changes", [])) if isinstance(raw_case.get("changes"), list) else [],
                "patches": list(raw_case.get("patches", [])) if isinstance(raw_case.get("patches"), list) else [],
                "case_id": raw_case.get("case_id", ""),
            }
            return instance, scale, meta

        if isinstance(raw_case, dict):
            instance = dict(raw_case)
            scale = self._normalize_scale_spec(None)
            case_meta = raw_case.get("__perturbation__") if isinstance(raw_case.get("__perturbation__"), dict) else {}
            meta = {
                "changes": list(case_meta.get("changes", [])) if isinstance(case_meta.get("changes"), list) else [],
                "patches": list(case_meta.get("patches", [])) if isinstance(case_meta.get("patches"), list) else [],
                "case_id": case_meta.get("case_id", ""),
            }
            return instance, scale, meta

        return {}, self._normalize_scale_spec(None), {}

    def _base_obj_scale(self) -> dict[str, Any]:
        from_instance = None
        if isinstance(self.task.instance, dict):
            from_instance = self.task.instance.get("__r3_base_obj_scale__")
            if from_instance is None:
                from_instance = self.task.instance.get("__r3_base_obj_bounds__")
        if from_instance is not None:
            return self._normalize_scale_spec(from_instance)

        if isinstance(self.task.perturbation_map, dict):
            maybe = self.task.perturbation_map.get("base_obj_scale")
            if maybe is None:
                maybe = self.task.perturbation_map.get("base_obj_bounds")
            if maybe is not None:
                return self._normalize_scale_spec(maybe)

        return {"kind": "interval", "lower": None, "upper": None, "sign_relation": "any", "magnitude": {"min_order": None, "max_order": None, "use_abs": True}, "reject_exact": []}

    @staticmethod
    def _normalize_scale_spec(value: Any) -> dict[str, Any]:
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

        def _normalize_sign_relation(raw: Any) -> str:
            text = str(raw or "any").strip().lower()
            aliases = {
                "gt0": "positive",
                ">0": "positive",
                "positive": "positive",
                "ge0": "nonnegative",
                ">=0": "nonnegative",
                "nonnegative": "nonnegative",
                "lt0": "negative",
                "<0": "negative",
                "negative": "negative",
                "le0": "nonpositive",
                "<=0": "nonpositive",
                "nonpositive": "nonpositive",
                "ne0": "nonzero",
                "!=0": "nonzero",
                "nonzero": "nonzero",
                "any": "any",
            }
            return aliases.get(text, "any")

        def _normalize_magnitude(raw: Any) -> dict[str, Any]:
            if not isinstance(raw, dict):
                raw = {}
            min_order = _num(raw.get("min_order"))
            if min_order is None:
                min_order = _num(raw.get("order_min"))
            if min_order is None:
                min_order = _num(raw.get("lower_order"))
            max_order = _num(raw.get("max_order"))
            if max_order is None:
                max_order = _num(raw.get("order_max"))
            if max_order is None:
                max_order = _num(raw.get("upper_order"))
            use_abs = bool(raw.get("use_abs", True))
            min_order_i = int(min_order) if min_order is not None and math.isfinite(min_order) else None
            max_order_i = int(max_order) if max_order is not None and math.isfinite(max_order) else None
            if min_order_i is not None and max_order_i is not None and min_order_i > max_order_i:
                min_order_i, max_order_i = max_order_i, min_order_i
            return {"min_order": min_order_i, "max_order": max_order_i, "use_abs": use_abs}

        if not isinstance(value, dict):
            return {"kind": "interval", "lower": None, "upper": None, "sign_relation": "any", "magnitude": {"min_order": None, "max_order": None, "use_abs": True}, "reject_exact": []}

        kind = str(value.get("kind") or "interval").strip().lower()
        reject_exact: list[float] = []
        if isinstance(value.get("reject_exact"), list):
            for item in value.get("reject_exact", []):
                num = _num(item)
                if num is not None:
                    reject_exact.append(float(num))
        sign_relation = _normalize_sign_relation(
            value.get("sign_relation")
            or value.get("zero_relation")
            or value.get("relation_to_zero")
            or value.get("sign")
        )
        magnitude = _normalize_magnitude(value.get("magnitude"))
        if magnitude["min_order"] is None and magnitude["max_order"] is None:
            magnitude = _normalize_magnitude(value)

        if kind == "point":
            point = _num(value.get("point"))
            tol_abs = _num(value.get("tol_abs"))
            tol_rel = _num(value.get("tol_rel"))
            return {
                "kind": "point",
                "point": point,
                "tol_abs": tol_abs,
                "tol_rel": tol_rel,
                "sign_relation": sign_relation,
                "magnitude": magnitude,
                "reject_exact": reject_exact,
            }

        if kind == "union":
            intervals: list[dict[str, float | None]] = []
            if isinstance(value.get("intervals"), list):
                for item in value.get("intervals", []):
                    if not isinstance(item, dict):
                        continue
                    lo = _num(item.get("lower"))
                    hi = _num(item.get("upper"))
                    if lo is not None and hi is not None and lo > hi:
                        lo, hi = hi, lo
                    if lo is None and hi is None:
                        continue
                    intervals.append({"lower": lo, "upper": hi})
            return {"kind": "union", "intervals": intervals, "sign_relation": sign_relation, "magnitude": magnitude, "reject_exact": reject_exact}

        lo = _num(value.get("lower"))
        hi = _num(value.get("upper"))
        if lo is not None and hi is not None and lo > hi:
            lo, hi = hi, lo
        return {"kind": "interval", "lower": lo, "upper": hi, "sign_relation": sign_relation, "magnitude": magnitude, "reject_exact": reject_exact}

    def _objective_matches_scale(self, obj_answer: float | None, scale: dict[str, Any] | None) -> bool:
        if not self._is_valid_objective(obj_answer):
            return False

        if not isinstance(scale, dict):
            return True

        val = float(obj_answer)
        abs_val = abs(val)
        eps = 1e-9
        reject_exact = scale.get("reject_exact") if isinstance(scale.get("reject_exact"), list) else []
        for item in reject_exact:
            if isinstance(item, (int, float)) and abs(val - float(item)) <= eps:
                return False

        sign_relation = str(scale.get("sign_relation") or "any").strip().lower()
        if sign_relation == "positive" and not (val > 0.0):
            return False
        if sign_relation == "nonnegative" and not (val >= 0.0):
            return False
        if sign_relation == "negative" and not (val < 0.0):
            return False
        if sign_relation == "nonpositive" and not (val <= 0.0):
            return False
        if sign_relation == "nonzero" and abs(val) <= eps:
            return False

        magnitude = scale.get("magnitude") if isinstance(scale.get("magnitude"), dict) else {}
        min_order = magnitude.get("min_order") if isinstance(magnitude.get("min_order"), int) else None
        max_order = magnitude.get("max_order") if isinstance(magnitude.get("max_order"), int) else None
        use_abs = bool(magnitude.get("use_abs", True))
        magnitude_val = abs_val if use_abs else val
        if min_order is not None or max_order is not None:
            if magnitude_val <= eps:
                return False
            order = math.log10(magnitude_val)
            if min_order is not None and order < float(min_order) - eps:
                return False
            if max_order is not None and order > float(max_order) + eps:
                return False

        kind = str(scale.get("kind") or "interval").strip().lower()
        if kind == "point":
            point = scale.get("point")
            if not isinstance(point, (int, float)):
                return True
            tol_abs = scale.get("tol_abs") if isinstance(scale.get("tol_abs"), (int, float)) else 0.0
            tol_rel = scale.get("tol_rel") if isinstance(scale.get("tol_rel"), (int, float)) else 0.0
            tol = max(float(tol_abs), abs(float(point)) * float(tol_rel))
            return abs(val - float(point)) <= tol + eps

        if kind == "union":
            intervals = scale.get("intervals") if isinstance(scale.get("intervals"), list) else []
            for item in intervals:
                if not isinstance(item, dict):
                    continue
                lo = item.get("lower")
                hi = item.get("upper")
                ok = True
                if isinstance(lo, (int, float)) and val < float(lo) - eps:
                    ok = False
                if isinstance(hi, (int, float)) and val > float(hi) + eps:
                    ok = False
                if ok:
                    return True
            return False if intervals else True

        lo = scale.get("lower")
        hi = scale.get("upper")
        if isinstance(lo, (int, float)) and val < float(lo) - eps:
            return False
        if isinstance(hi, (int, float)) and val > float(hi) + eps:
            return False
        return True

    def _objective_within_bounds(self, obj_answer: float | None, bounds: dict[str, Any] | None) -> bool:
        return self._objective_matches_scale(obj_answer, bounds)

    def _execute(self, trajectory: Trajectory) -> tuple[ExecutionResult, bool]:
        cache_key = self._execution_cache_key(trajectory.code, self.task.instance)
        with self._exec_cache_lock:
            cached = self._exec_cache.get(cache_key)
        if cached is not None:
            return cached, True

        result = self.executor.run(trajectory.code, self.task.instance)
        with self._exec_cache_lock:
            existing = self._exec_cache.get(cache_key)
            if existing is not None:
                return existing, True
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
                numeric = float(value)
                return numeric if math.isfinite(numeric) else None
            if isinstance(value, str):
                text = value.strip().replace(",", "")
                if not text:
                    return None
                try:
                    numeric = float(text)
                    return numeric if math.isfinite(numeric) else None
                except Exception:
                    return None
            return None

        # Direct scalar outputs are common in script-style candidates where
        # the runtime picks module.optimal / module.obj / module.objective.
        direct_numeric = _as_float(output)
        if direct_numeric is not None:
            return direct_numeric

        objective_keys = (
            "objective",
            "obj",
            "optimal",
            "optimal_value",
            "objective_value",
            "obj_value",
            "best_objective",
            "optimal_obj",
            "objective_val",
        )

        if isinstance(output, dict):
            for key in objective_keys:
                numeric = _as_float(output.get(key))
                if numeric is not None:
                    return numeric

            for wrapper_key in ("result", "value", "val", "data"):
                nested = output.get(wrapper_key)
                if nested is not None:
                    numeric = TTRLRewardCalculator._extract_objective_numeric(nested)
                    if numeric is not None:
                        return numeric

            repr_value = output.get("repr")
            if repr_value is not None:
                numeric = TTRLRewardCalculator._extract_objective_from_text(str(repr_value))
                if numeric is not None:
                    return numeric

            # Fallback: recursively scan nested dict/list payloads for common
            # objective keys before giving up.
            for _, value in output.items():
                if isinstance(value, (dict, list, tuple)):
                    numeric = TTRLRewardCalculator._extract_objective_numeric(value)
                    if numeric is not None:
                        return numeric

        if isinstance(output, (list, tuple)):
            for item in output:
                numeric = TTRLRewardCalculator._extract_objective_numeric(item)
                if numeric is not None:
                    return numeric

        if isinstance(output, str):
            try:
                maybe = json.loads(output)
            except Exception:
                return TTRLRewardCalculator._extract_objective_from_text(output)
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
        if self._r2_execution_success(execution):
            return True

        obj_answer = self._extract_objective_from_execution(execution)
        if self._is_valid_objective(obj_answer):
            return True

        stdout_text = str(execution.stdout or "")
        lowered = stdout_text.lower()
        if any(marker in lowered for marker in self._gurobi_success_markers):
            return True

        if "optimal value" in lowered or "objective value" in lowered:
            return True

        return False
    def _execution_summary(
        self,
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
            "r2_success": bool(self._r2_execution_success(execution)),
            "effective_success": eff,
            "solver_success_marker_hit": marker_hit,
            "objective_text_marker_hit": bool("optimal value" in lowered or "objective value" in lowered),
            "parsed_obj_answer": obj_answer,
            "objective_parsed_success": obj_hit,
            "signature": str(execution.signature or ""),
            "error_type": str(execution.error_type or ""),
            "lp_injection_applied": bool(getattr(execution, "lp_injection_applied", False)),
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

    def _structure_gate(self, model_info: Any) -> tuple[float, dict[str, Any]]:
        gate_min = float(max(0.0, min(1.0, getattr(self.config, 'structure_gate_min', 0.2))))
        if model_info is None or not bool(getattr(model_info, 'extracted', False)):
            return gate_min, {
                'name': 'structure_gate',
                'gate_min': gate_min,
                'extracted': False,
                'has_objective': False,
                'num_constrs': 0,
                'num_vars': 0,
                'passes': False,
            }
        has_objective = bool(getattr(model_info, 'has_objective', False))
        num_constrs = int(getattr(model_info, 'num_constrs', 0) or 0)
        num_vars = int(getattr(model_info, 'num_vars', 0) or 0)
        passes = bool(has_objective and num_constrs > 0 and num_vars > 0)
        return (1.0 if passes else gate_min), {
            'name': 'structure_gate',
            'gate_min': gate_min,
            'extracted': True,
            'has_objective': has_objective,
            'num_constrs': num_constrs,
            'num_vars': num_vars,
            'passes': passes,
        }














