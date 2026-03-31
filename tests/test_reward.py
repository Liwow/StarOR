from ttrl_or.config import RewardConfig
from ttrl_or.model import MockPolicyBackend
from ttrl_or.reward.default_reward import TTRLRewardCalculator
from ttrl_or.types import OptimizationTask, Stage, Trajectory


# Note: In the new reward system:
# - r1 is now a continuous cluster-ratio value (0 to 1), not binary
# - r4 is a new structural consensus reward
# - Formula: total = max(0, r1 + r3_weight * r3 * r2 + r4_weight * r4)


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
    # New formula: total = max(0, r1 + r3_weight * r3 * r2 + r4_weight * r4)
    # Default weights: r3_weight=0.3, r4_weight=0.2
    
    # r1=1.0, r2=0.0, r3=1.0, r4=0.0 => 1.0 + 0.3*1.0*0.0 + 0.2*0.0 = 1.0
    assert TTRLRewardCalculator.combine_rewards(r1=1.0, r2=0.0, r3=1.0, r4=0.0) == 1.0
    
    # r1=1.0, r2=1.0, r3=1.0, r4=0.0 => 1.0 + 0.3*1.0*1.0 + 0.2*0.0 = 1.3
    assert TTRLRewardCalculator.combine_rewards(r1=1.0, r2=1.0, r3=1.0, r4=0.0) == 1.3
    
    # r1=0.0, r2=1.0, r3=0.0, r4=1.0 => 0.0 + 0.3*0.0*1.0 + 0.2*1.0 = 0.2
    assert TTRLRewardCalculator.combine_rewards(r1=0.0, r2=1.0, r3=0.0, r4=1.0) == 0.2
    
    # r1=0.5, r2=1.0, r3=1.0, r4=0.5 => 0.5 + 0.3*1.0*1.0 + 0.2*0.5 = 0.9
    assert abs(TTRLRewardCalculator.combine_rewards(r1=0.5, r2=1.0, r3=1.0, r4=0.5) - 0.9) < 1e-9
    
    # Negative result should be clamped to 0
    # r1=-0.5, r2=0.0, r3=0.0, r4=0.0 => max(0, -0.5) = 0.0
    assert TTRLRewardCalculator.combine_rewards(r1=-0.5, r2=0.0, r3=0.0, r4=0.0) == 0.0


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

        # In the new system, r1 is cluster-ratio (continuous), not binary
        # t1 and t2 produce the same objective, so they belong to the same cluster
        # Formula: r1 = (count_i + alpha) / (N + alpha * K)
        # After processing, with 2 successful samples in same cluster:
        # Both should have positive r1 since they're in the majority cluster
        assert updated[0].reward.r1 > 0.0
        assert updated[1].reward.r1 > 0.0
        # t3 failed execution, so r1 should be 0
        assert updated[2].reward.r1 == 0.0
        assert updated[2].reward.r2 == 0.0
    finally:
        backend.end_episode()


def test_r1_semantic_clustering_different_values():
    """Test that different objective values create different clusters."""
    task = OptimizationTask(
        task_id="reward-cluster",
        description="Semantic clustering test",
        instance={"x": 1},
    )
    backend = MockPolicyBackend(seed=17)
    backend.begin_episode(task)

    try:
        rewarder = TTRLRewardCalculator(task=task, backend=backend, config=RewardConfig())

        # These values differ by more than 0.5% relative tolerance,
        # so they should form different clusters
        t1 = Trajectory(trajectory_id="o1", outputs={Stage.CODE: _const_code(2.0)})
        t2 = Trajectory(trajectory_id="o2", outputs={Stage.CODE: _const_code(8.0)})
        t3 = Trajectory(trajectory_id="o3", outputs={Stage.CODE: _const_code(6.0)})

        r1 = rewarder.provisional_reward(t1, explored=[])
        r2 = rewarder.provisional_reward(t2, explored=[])
        r3 = rewarder.provisional_reward(t3, explored=[])

        # All should have positive r1 (they're valid executions)
        # r1 is now cluster-ratio, not binary
        assert r1.r1 > 0.0
        assert r2.r1 > 0.0
        assert r3.r1 > 0.0
        
        # All should have r2=1.0 (execution success)
        assert r1.r2 == 1.0
        assert r2.r2 == 1.0
        assert r3.r2 == 1.0
    finally:
        backend.end_episode()


def test_r1_semantic_clustering_with_relative_tolerance():
    """Test that values within 0.5% relative tolerance form the same cluster."""
    task = OptimizationTask(
        task_id="reward-majority",
        description="Semantic clustering with tolerance",
        instance={"x": 1},
    )
    backend = MockPolicyBackend(seed=19)
    backend.begin_episode(task)

    try:
        rewarder = TTRLRewardCalculator(task=task, backend=backend, config=RewardConfig())

        # 100.0, 100.2, 99.9 are within 0.5% of each other (0.5% of 100 = 0.5)
        # 120.0 is too far away to be in the same cluster
        seeds = [100.0, 100.2, 99.9, 120.0]
        for i, value in enumerate(seeds, start=1):
            traj = Trajectory(trajectory_id=f"m{i}", outputs={Stage.CODE: _const_code(value)})
            rewarder.provisional_reward(traj, explored=[])

        # 100.3 is within 0.5% tolerance of 100.0 => joins the large cluster
        in_band = Trajectory(trajectory_id="m_in", outputs={Stage.CODE: _const_code(100.3)})
        # 101.0 is > 0.5% away from 100.0 (|101-100|/100 = 1% > 0.5%) => new cluster
        out_band = Trajectory(trajectory_id="m_out", outputs={Stage.CODE: _const_code(101.0)})

        r_in = rewarder.provisional_reward(in_band, explored=[])
        r_out = rewarder.provisional_reward(out_band, explored=[])

        # r_in joins the majority cluster (4 samples: 100.0, 100.2, 99.9, 100.3)
        # r_out creates a new smaller cluster
        # So r_in should have higher r1 than r_out
        assert r_in.r1 > r_out.r1
        
        # Both should have positive r1 (valid executions)
        assert r_in.r1 > 0.0
        assert r_out.r1 > 0.0
    finally:
        backend.end_episode()


def test_disable_r3_reward_short_circuits_r3():
    task = OptimizationTask(
        task_id="reward-3",
        description="Disable perturb reward",
        instance={"a": 2, "b": 3},
    )
    backend = MockPolicyBackend(seed=9)
    backend.begin_episode(task)

    try:
        config = RewardConfig(enable_r3_reward=False)
        rewarder = TTRLRewardCalculator(task=task, backend=backend, config=config)

        explored = [Trajectory(trajectory_id="e1", outputs={Stage.CODE: _GOOD_CODE})]
        candidate = Trajectory(trajectory_id="cand", outputs={Stage.CODE: _GOOD_CODE})
        reward = rewarder.provisional_reward(candidate, explored)

        # r1 is now cluster-ratio, positive for valid execution
        assert reward.r1 > 0.0
        # r3 should be 1.0 when disabled (default pass)
        assert reward.r3 == 1.0
        # r2 should be 1.0 for successful execution
        assert reward.r2 == 1.0
        # Check r3 metadata indicates disabled
        assert reward.metadata["r3"]["enabled"] is False
    finally:
        backend.end_episode()

