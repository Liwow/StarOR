from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Protocol

from ttrl_or.config import MCTSConfig
from ttrl_or.mcts.node import SearchNode
from ttrl_or.mcts.puct import PUCTSelector
from ttrl_or.prompts import PromptBuilder
from ttrl_or.types import OptimizationTask, RewardBreakdown, STAGE_ORDER, Stage, Trajectory


class ProvisionalRewarder(Protocol):
    def provisional_reward(self, trajectory: Trajectory, explored: list[Trajectory]) -> RewardBreakdown:
        ...


@dataclass(slots=True)
class StageExpansionRecord:
    stage: Stage
    node_id: str
    parent_id: str
    prompt: str
    completion: str
    reward: float
    trajectory: Trajectory
    prior: float = 0.0
    was_expanded: bool = False
    child_q_before: float = 0.0
    child_visits_before: int = 0
    child_q_after: float = 0.0
    child_visits_after: int = 0
    parent_q_before: float = 0.0
    parent_visits_before: int = 0
    parent_q_after: float = 0.0
    parent_visits_after: int = 0
    rollout_details: list[dict] = field(default_factory=list)


class FourStageMCTS:
    def __init__(
        self,
        backend,
        prompt_builder: PromptBuilder,
        rewarder: ProvisionalRewarder,
        config: MCTSConfig,
        selector: PUCTSelector | None = None,
    ) -> None:
        self.backend = backend
        self.prompt_builder = prompt_builder
        self.rewarder = rewarder
        self.config = config
        self.selector = selector or PUCTSelector(c_puct=config.c_puct)

    def root(self) -> SearchNode:
        return SearchNode(stage=None, text="<ROOT>", prior=1.0)

    def expand_stage(
        self,
        task: OptimizationTask,
        stage: Stage,
        parent_nodes: list[SearchNode],
        rollout_archive: list[Trajectory],
    ) -> tuple[list[SearchNode], list[StageExpansionRecord]]:
        records: list[StageExpansionRecord] = []

        for parent in parent_nodes:
            for _ in range(self.config.simulations_per_node):
                parent_q_before = parent.q_value
                parent_visits_before = parent.visits

                if len(parent.children) < self.config.expand_per_node:
                    parent_traj = None if parent.stage is None else parent.to_partial_trajectory()
                    prompt = self.prompt_builder.build(task, stage, parent_traj)
                    generation = self.backend.generate(stage, prompt, 1)[0]
                    child = SearchNode(
                        stage=stage,
                        text=generation.text,
                        prior=generation.prior,
                        parent=parent,
                        prompt=prompt,
                    )
                    parent.add_child(child)
                    was_expanded = True
                else:
                    child = self.selector.select(parent)
                    was_expanded = False

                child_q_before = child.q_value
                child_visits_before = child.visits

                reward_values: list[float] = []
                best_reward = float("-inf")
                best_trajectory: Trajectory | None = None
                rollout_details: list[dict] = []

                for ridx in range(max(1, self.config.rollout_k)):
                    completed = self.rollout_to_code(task, child)
                    reward = self.rewarder.provisional_reward(completed, rollout_archive)
                    rollout_archive.append(completed)
                    reward_values.append(reward.total)

                    rollout_details.append(
                        {
                            "rollout_index": ridx,
                            "trajectory_id": completed.trajectory_id,
                            "reward": {
                                "r1": reward.r1,
                                "r2": reward.r2,
                                "r3": reward.r3,
                                "total": reward.total,
                                "consensus_signature": reward.consensus_signature,
                                "execution_success": reward.execution_success,
                                "robustness_success": reward.robustness_success,
                            },
                            "priors": {s.value: p for s, p in completed.priors.items()},
                        }
                    )

                    if reward.total > best_reward:
                        best_reward = reward.total
                        best_trajectory = completed

                mean_reward = sum(reward_values) / max(1, len(reward_values))
                child.update(mean_reward)
                parent.update(mean_reward)

                if best_trajectory is None:
                    best_trajectory = child.to_partial_trajectory()

                records.append(
                    StageExpansionRecord(
                        stage=stage,
                        node_id=child.node_id,
                        parent_id=parent.node_id,
                        prompt=child.prompt,
                        completion=child.text,
                        reward=mean_reward,
                        trajectory=best_trajectory,
                        prior=child.prior,
                        was_expanded=was_expanded,
                        child_q_before=child_q_before,
                        child_visits_before=child_visits_before,
                        child_q_after=child.q_value,
                        child_visits_after=child.visits,
                        parent_q_before=parent_q_before,
                        parent_visits_before=parent_visits_before,
                        parent_q_after=parent.q_value,
                        parent_visits_after=parent.visits,
                        rollout_details=rollout_details,
                    )
                )

        candidates: list[SearchNode] = []
        for parent in parent_nodes:
            candidates.extend(parent.children)

        dedup: dict[str, SearchNode] = {node.node_id: node for node in candidates}
        ranked = sorted(
            dedup.values(),
            key=lambda node: (node.q_value, node.visits, node.prior),
            reverse=True,
        )
        return ranked[: self.config.max_nodes_per_stage], records

    def rollout_to_code(self, task: OptimizationTask, from_node: SearchNode) -> Trajectory:
        partial = from_node.to_partial_trajectory()
        partial.trajectory_id = str(uuid.uuid4())

        if from_node.stage is None:
            start_idx = -1
        else:
            start_idx = STAGE_ORDER.index(from_node.stage)

        for next_stage in STAGE_ORDER[start_idx + 1 :]:
            prompt = self.prompt_builder.build(task, next_stage, partial)
            generation = self.backend.generate(next_stage, prompt, 1)[0]
            partial.outputs[next_stage] = generation.text
            partial.priors[next_stage] = generation.prior

        return partial

    @staticmethod
    def pick_group_trajectories(code_nodes: list[SearchNode], group_size: int) -> list[Trajectory]:
        ranked = sorted(code_nodes, key=lambda node: (node.q_value, node.visits), reverse=True)
        picked: list[Trajectory] = []
        for node in ranked[:group_size]:
            picked.append(node.to_partial_trajectory())
        return picked
