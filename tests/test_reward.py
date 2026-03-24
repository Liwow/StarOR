from ttrl_or.config import RewardConfig
from ttrl_or.model import MockPolicyBackend
from ttrl_or.reward.default_reward import TTRLRewardCalculator
from ttrl_or.types import OptimizationTask, Stage, Trajectory


_GOOD_CODE = """
def solve(instance: dict) -> dict:
    total = 0.0
    for v in instance.values():
        if isinstance(v, (int, float)):
            total += float(v)
    return {"objective": total, "status": "ok"}
""".strip()

_WEIGHT_CODE = """
def solve(instance: dict) -> dict:
    total = 0.0
    for idx, key in enumerate(sorted(instance.keys())):
        v = instance[key]
        if isinstance(v, (int, float)):
            total += float(v) * (idx + 1)
    return {"objective": total, "status": "ok"}
""".strip()

_BAD_CODE = """
def solve(instance: dict) -> dict:
    return {"objective": float(missing_symbol), "status": "bad"}
""".strip()


def test_reward_combination_formula():
    assert TTRLRewardCalculator.combine_rewards(r1=1.0, r2=0.0, r3=1.0) == 1.0
    assert TTRLRewardCalculator.combine_rewards(r1=1.0, r2=0.0, r3=0.0) == 0.9
    assert TTRLRewardCalculator.combine_rewards(r1=0.0, r2=1.0, r3=0.0) == 0.2


def test_finalize_group_majority_and_execution():
    task = OptimizationTask(
        task_id="reward-1",
        description="Simple numeric objective",
        instance={"a": 2, "b": 3},
    )
    backend = MockPolicyBackend(seed=5)
    backend.begin_episode(task)

    try:
        rewarder = TTRLRewardCalculator(task=task, backend=backend, config=RewardConfig())
        trajectories = [
            Trajectory(trajectory_id="t1", outputs={Stage.CODE: _GOOD_CODE}),
            Trajectory(trajectory_id="t2", outputs={Stage.CODE: _GOOD_CODE}),
            Trajectory(trajectory_id="t3", outputs={Stage.CODE: _BAD_CODE}),
        ]
        updated = rewarder.finalize_group(trajectories)

        assert updated[0].reward.r1 == 1.0
        assert updated[1].reward.r1 == 1.0
        assert updated[2].reward.r1 == 0.0
        assert updated[2].reward.r2 == 0.0
    finally:
        backend.end_episode()


def test_provisional_reward_uses_stage_local_consensus_window():
    task = OptimizationTask(
        task_id="reward-2",
        description="Stage-local consensus test",
        instance={"a": 2, "b": 3},
    )
    backend = MockPolicyBackend(seed=7)
    backend.begin_episode(task)

    try:
        config = RewardConfig(local_consensus_window=1)
        rewarder = TTRLRewardCalculator(task=task, backend=backend, config=config)

        # Full history majority would favor _GOOD_CODE (2 vs 1),
        # but window=1 should only look at latest explored trajectory (_WEIGHT_CODE).
        explored = [
            Trajectory(trajectory_id="e1", outputs={Stage.CODE: _GOOD_CODE}),
            Trajectory(trajectory_id="e2", outputs={Stage.CODE: _GOOD_CODE}),
            Trajectory(trajectory_id="e3", outputs={Stage.CODE: _WEIGHT_CODE}),
        ]
        candidate = Trajectory(trajectory_id="cand", outputs={Stage.CODE: _WEIGHT_CODE})

        reward = rewarder.provisional_reward(candidate, explored)
        assert reward.r1 == 1.0
    finally:
        backend.end_episode()
