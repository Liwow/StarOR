from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from ttrl_or.config import RewardConfig
from ttrl_or.model.backend import PolicyBackend
from ttrl_or.reward.base import RewardCalculator
from ttrl_or.reward.executor import PythonCodeExecutor
from ttrl_or.reward.perturbation import generate_perturbed_instances_from_map
from ttrl_or.types import OptimizationTask, RewardBreakdown, Trajectory


@dataclass(slots=True)
class TTRLRewardCalculator(RewardCalculator):
    task: OptimizationTask
    backend: PolicyBackend
    config: RewardConfig
    executor: PythonCodeExecutor = field(init=False)
    _exec_cache: dict[str, object] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self.executor = PythonCodeExecutor(timeout_sec=self.config.code_timeout_sec)

    def provisional_reward(self, trajectory: Trajectory, explored: list[Trajectory]) -> RewardBreakdown:
        execution = self._execute(trajectory)

        # Stage-local sliding window for consensus (no cross-stage leakage).
        scoped = self._windowed_explored(explored)
        consensus = self._majority_signature(self._collect_signatures(scoped))

        r1 = self.compute_r1(execution.success, execution.signature, consensus)
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
        signatures: list[str] = []
        exec_results = {}
        for traj in trajectories:
            res = self._execute(traj)
            exec_results[traj.trajectory_id] = res
            if res.success:
                signatures.append(res.signature)

        consensus = self._majority_signature(signatures)

        for traj in trajectories:
            exec_result = exec_results[traj.trajectory_id]
            r1 = self.compute_r1(exec_result.success, exec_result.signature, consensus)
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

    def compute_r1(self, execution_success: bool, signature: str, consensus: str) -> float:
        if not execution_success:
            return 0.0
        if not consensus:
            return 0.0
        return 1.0 if signature == consensus else 0.0

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

    def _windowed_explored(self, explored: list[Trajectory]) -> list[Trajectory]:
        window = self.config.local_consensus_window
        if window <= 0:
            return explored
        return explored[-window:]

    def _collect_signatures(self, trajectories: list[Trajectory]) -> list[str]:
        signatures: list[str] = []
        for traj in trajectories:
            res = self._execute(traj)
            if res.success:
                signatures.append(res.signature)
        return signatures

    @staticmethod
    def _majority_signature(signatures: list[str]) -> str:
        valid = [s for s in signatures if s and s != "EXEC_ERROR"]
        if not valid:
            return ""
        return Counter(valid).most_common(1)[0][0]

