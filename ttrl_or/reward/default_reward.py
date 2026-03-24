from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from ttrl_or.config import RewardConfig
from ttrl_or.model.backend import PolicyBackend
from ttrl_or.reward.base import RewardCalculator
from ttrl_or.reward.executor import PythonCodeExecutor
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
            r3 = self.compute_r3(trajectory)
            total = self.combine_rewards(r1=r1, r2=0.0, r3=r3)
            return RewardBreakdown(
                r1=r1,
                r2=0.0,
                r3=r3,
                total=total,
                consensus_signature=consensus,
                execution_success=execution.success,
                robustness_success=(r3 == 1.0),
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
                r3 = self.compute_r3(traj)
                total = self.combine_rewards(r1=r1, r2=0.0, r3=r3)
                traj.reward = RewardBreakdown(
                    r1=r1,
                    r2=0.0,
                    r3=r3,
                    total=total,
                    consensus_signature=consensus,
                    execution_success=exec_result.success,
                    robustness_success=(r3 == 1.0),
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
        tests = self.backend.generate_test_instances(self.task, self.config.robustness_cases)
        if not tests:
            return 0.0

        for case in tests:
            res = self.executor.run(trajectory.code, case)
            if not res.success:
                return 0.0
        return 1.0

    @staticmethod
    def combine_rewards(r1: float, r2: float, r3: float) -> float:
        if r1 == 1.0:
            return r1 * 0.9 + r3 * 0.1
        return r2 * 0.2

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
