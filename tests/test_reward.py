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


def _const_code(value: float) -> str:
    return f"""
def solve(instance: dict) -> dict:
    return {{"objective": {float(value)}, "status": "ok"}}
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


def test_r1_global_pool_uses_order_of_magnitude_when_pool_lt_three():
    task = OptimizationTask(
        task_id="reward-oom",
        description="Global pool OOM consensus",
        instance={"x": 1},
    )
    backend = MockPolicyBackend(seed=17)
    backend.begin_episode(task)

    try:
        rewarder = TTRLRewardCalculator(task=task, backend=backend, config=RewardConfig())

        t1 = Trajectory(trajectory_id="o1", outputs={Stage.CODE: _const_code(2.0)})
        t2 = Trajectory(trajectory_id="o2", outputs={Stage.CODE: _const_code(8.0)})
        t3 = Trajectory(trajectory_id="o3", outputs={Stage.CODE: _const_code(6.0)})

        r1 = rewarder.provisional_reward(t1, explored=[])
        r2 = rewarder.provisional_reward(t2, explored=[])
        r3 = rewarder.provisional_reward(t3, explored=[])

        assert r1.r1 == 1.0
        assert r2.r1 == 1.0
        assert r3.r1 == 1.0
    finally:
        backend.end_episode()


def test_r1_global_pool_majority_vote_with_relative_tolerance():
    task = OptimizationTask(
        task_id="reward-majority",
        description="Global pool majority consensus",
        instance={"x": 1},
    )
    backend = MockPolicyBackend(seed=19)
    backend.begin_episode(task)

    try:
        rewarder = TTRLRewardCalculator(task=task, backend=backend, config=RewardConfig())

        seeds = [100.0, 100.2, 99.9, 120.0]
        for i, value in enumerate(seeds, start=1):
            traj = Trajectory(trajectory_id=f"m{i}", outputs={Stage.CODE: _const_code(value)})
            rewarder.provisional_reward(traj, explored=[])

        in_band = Trajectory(trajectory_id="m_in", outputs={Stage.CODE: _const_code(100.3)})
        out_band = Trajectory(trajectory_id="m_out", outputs={Stage.CODE: _const_code(101.0)})

        r_in = rewarder.provisional_reward(in_band, explored=[])
        r_out = rewarder.provisional_reward(out_band, explored=[])

        assert r_in.r1 == 1.0
        assert r_out.r1 == 0.0
    finally:
        backend.end_episode()


def test_disable_perturb_reward_short_circuits_r3():
    task = OptimizationTask(
        task_id="reward-3",
        description="Disable perturb reward",
        instance={"a": 2, "b": 3},
    )
    backend = MockPolicyBackend(seed=9)
    backend.begin_episode(task)

    try:
        config = RewardConfig(enable_perturb_reward=False)
        rewarder = TTRLRewardCalculator(task=task, backend=backend, config=config)

        explored = [Trajectory(trajectory_id="e1", outputs={Stage.CODE: _GOOD_CODE})]
        candidate = Trajectory(trajectory_id="cand", outputs={Stage.CODE: _GOOD_CODE})
        reward = rewarder.provisional_reward(candidate, explored)

        assert reward.r1 == 1.0
        assert reward.r3 == 1.0
        assert reward.total == 1.0
        assert reward.metadata["r3"]["enabled"] is False
    finally:
        backend.end_episode()
