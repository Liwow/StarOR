from __future__ import annotations

from dataclasses import dataclass

from ttrl_or.config import MCTSConfig
from ttrl_or.mcts import FourStageMCTS
from ttrl_or.prompts import DEFAULT_TEMPLATES, PromptBuilder
from ttrl_or.types import Generation, OptimizationTask, RewardBreakdown, Stage, Trajectory


@dataclass
class _FakeBackend:
    def generate(self, stage: Stage, prompt: str, n: int):
        return [Generation(text=f"{stage.value}_draft", prior=0.9) for _ in range(n)]


@dataclass
class _AlwaysPerfectRewarder:
    def provisional_reward(self, trajectory: Trajectory, explored: list[Trajectory]) -> RewardBreakdown:
        return RewardBreakdown(
            r1=1.0,
            r2=0.0,
            r3=1.0,
            total=1.0,
            consensus_signature="ok",
            execution_success=True,
            robustness_success=True,
        )


def _build_task() -> OptimizationTask:
    return OptimizationTask(
        task_id="mcts-stop-test",
        description="maximize profit under capacity",
        instance={"capacity": 10},
    )


def test_mcts_can_stop_early_on_reward_one():
    config = MCTSConfig(
        expand_per_node=3,
        simulations_per_node=5,
        max_nodes_per_stage=8,
        rollout_k=4,
        stop_on_reward_one=True,
    )
    mcts = FourStageMCTS(
        backend=_FakeBackend(),
        prompt_builder=PromptBuilder(templates=DEFAULT_TEMPLATES),
        rewarder=_AlwaysPerfectRewarder(),
        config=config,
    )

    root = mcts.root()
    frontier, records, early_stop_info = mcts.expand_stage(
        task=_build_task(),
        stage=Stage.SCHEMA,
        parent_nodes=[root],
        rollout_archive=[],
    )

    assert early_stop_info is not None
    assert len(records) == 1
    assert records[0].hit_reward_one is True
    assert len(frontier) >= 1


def test_mcts_without_early_stop_runs_full_simulations():
    config = MCTSConfig(
        expand_per_node=3,
        simulations_per_node=5,
        max_nodes_per_stage=8,
        rollout_k=4,
        stop_on_reward_one=False,
    )
    mcts = FourStageMCTS(
        backend=_FakeBackend(),
        prompt_builder=PromptBuilder(templates=DEFAULT_TEMPLATES),
        rewarder=_AlwaysPerfectRewarder(),
        config=config,
    )

    root = mcts.root()
    _, records, early_stop_info = mcts.expand_stage(
        task=_build_task(),
        stage=Stage.SCHEMA,
        parent_nodes=[root],
        rollout_archive=[],
    )

    assert early_stop_info is None
    assert len(records) == config.simulations_per_node
