from __future__ import annotations

from dataclasses import dataclass

from ttrl_or.config import GRPOConfig, MCTSConfig
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
    config = MCTSConfig(max_iterations=16, stop_on_reward_one=True)
    mcts = FourStageMCTS(
        backend=_FakeBackend(),
        prompt_builder=PromptBuilder(templates=DEFAULT_TEMPLATES),
        rewarder=_AlwaysPerfectRewarder(),
        config=config,
    )

    result = mcts.search(task=_build_task(), grpo_config=GRPOConfig(num_generations=4))

    assert result.stop_info.get("reason") == "reward_one"
    assert len(result.records) == 1
    assert result.records[0].stage == Stage.SCHEMA
    assert result.records[0].hit_reward_one is True


def test_mcts_stops_after_reaching_code_when_no_reward_one_stop():
    config = MCTSConfig(max_iterations=16, stop_on_reward_one=False)
    mcts = FourStageMCTS(
        backend=_FakeBackend(),
        prompt_builder=PromptBuilder(templates=DEFAULT_TEMPLATES),
        rewarder=_AlwaysPerfectRewarder(),
        config=config,
    )

    result = mcts.search(task=_build_task(), grpo_config=GRPOConfig(num_generations=4))

    assert result.stop_info.get("reason") == "expanded_to_code"
    assert any(record.stage == Stage.CODE for record in result.records)
    assert result.best_trajectory is not None
    assert result.best_reward == 1.0
