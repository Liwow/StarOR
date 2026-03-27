from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from ttrl_or.config import GRPOConfig, MCTSConfig
from ttrl_or.mcts.node import SearchNode
from ttrl_or.mcts.puct import PUCTSelector
from ttrl_or.prompts import PromptBuilder
from ttrl_or.types import Generation, OptimizationTask, RewardBreakdown, STAGE_ORDER, Stage, Trajectory


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
    was_expanded: bool = True
    hit_reward_one: bool = False
    child_q_before: float = 0.0
    child_visits_before: int = 0
    child_q_after: float = 0.0
    child_visits_after: int = 0
    parent_q_before: float = 0.0
    parent_visits_before: int = 0
    parent_q_after: float = 0.0
    parent_visits_after: int = 0
    rollout_details: list[dict] = field(default_factory=list)
    simulation_index: int = 0
    rollout_index: int = 0
    group_id: str = ""
    grpo_report: dict[str, Any] = field(default_factory=dict)
    iteration: int = 0


@dataclass(slots=True)
class SearchRunResult:
    root: SearchNode
    records: list[StageExpansionRecord]
    stop_info: dict[str, Any]
    best_trajectory: Trajectory | None
    best_reward: float
    code_nodes: list[SearchNode]
    iteration_logs: list[dict[str, Any]] = field(default_factory=list)


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

    def search(
        self,
        task: OptimizationTask,
        grpo_config: GRPOConfig,
    ) -> SearchRunResult:
        root = self.root()
        records: list[StageExpansionRecord] = []
        iteration_logs: list[dict[str, Any]] = []
        best_trajectory: Trajectory | None = None
        best_reward = float("-inf")

        stage_archives: dict[Stage, list[Trajectory]] = {stage: [] for stage in STAGE_ORDER}

        stop_info: dict[str, Any] = {
            "reason": "max_iterations",
            "iteration": -1,
            "stage": "",
            "node_id": "",
            "trajectory_id": "",
            "reward_total": None,
        }

        for iter_idx in range(max(1, int(self.config.max_iterations))):
            leaves = [node for node in self._iter_leaves(root) if self._next_stage(node.stage) is not None]
            if not leaves:
                stop_info = {
                    "reason": "no_expandable_leaf",
                    "iteration": iter_idx,
                    "stage": "",
                    "node_id": "",
                    "trajectory_id": "",
                    "reward_total": None,
                }
                break

            ranked_leaves = self._rank_leaves(leaves)
            selected = ranked_leaves[0][0]
            selected_score = float(ranked_leaves[0][1])

            next_stage = self._next_stage(selected.stage)
            if next_stage is None:
                continue

            selected_traj = None if selected.stage is None else selected.to_partial_trajectory()
            prompt = self.prompt_builder.build(task, next_stage, selected_traj)
            group_id = f"iter:{iter_idx}:{selected.node_id}:{next_stage.value}"
            group_rollouts: list[dict[str, Any]] = []
            stage_archive = stage_archives[next_stage]

            def _reward_callback(prompt_text: str, completion_text: str, ridx: int) -> float:
                child = SearchNode(
                    stage=next_stage,
                    text=self._extract_stage_payload(next_stage, completion_text),
                    prior=1.0,
                    parent=selected,
                    prompt=prompt,
                )
                completed = self._complete_for_reward(task, child)
                reward = self.rewarder.provisional_reward(completed, stage_archive)
                completed.reward = reward
                stage_archive.append(completed)

                group_rollouts.append(
                    {
                        "rollout_index": ridx,
                        "child": child,
                        "trajectory": completed,
                        "reward_obj": reward,
                        "reward_total": float(reward.total),
                    }
                )
                return float(reward.total)

            generations, grpo_report = self._run_internal_grpo_rollout(
                stage=next_stage,
                prompt=prompt,
                grpo_config=grpo_config,
                reward_callback=_reward_callback,
            )

            if not group_rollouts:
                continue

            hit_reward_one = False
            reward_one_payload: dict[str, Any] = {}
            rollout_summaries: list[dict[str, Any]] = []
            processed_group_rollouts: list[dict[str, Any]] = []

            for ridx, rollout in enumerate(group_rollouts):
                child = rollout["child"]
                completed = rollout["trajectory"]
                reward_obj = rollout["reward_obj"]
                reward_total = float(rollout["reward_total"])

                processed_group_rollouts.append(rollout)

                if ridx < len(generations):
                    child.prior = max(1e-6, float(generations[ridx].prior))

                parent_q_before = selected.q_value
                parent_visits_before = selected.visits
                child_q_before = child.q_value
                child_visits_before = child.visits

                selected.add_child(child)
                self._backpropagate(child, reward_total)

                rollout_detail = {
                    "rollout_index": ridx,
                    "trajectory_id": completed.trajectory_id,
                    "reward": {
                        "r1": reward_obj.r1,
                        "r2": reward_obj.r2,
                        "r3": reward_obj.r3,
                        "total": reward_obj.total,
                        "consensus_signature": reward_obj.consensus_signature,
                        "execution_success": reward_obj.execution_success,
                        "robustness_success": reward_obj.robustness_success,
                    },
                    "priors": {s.value: p for s, p in completed.priors.items()},
                }

                current_hit_reward_one = bool(self.config.stop_on_reward_one and reward_total >= 1.0)

                records.append(
                    StageExpansionRecord(
                        stage=next_stage,
                        node_id=child.node_id,
                        parent_id=selected.node_id,
                        prompt=prompt,
                        completion=child.text,
                        reward=reward_total,
                        trajectory=completed,
                        prior=child.prior,
                        was_expanded=True,
                        hit_reward_one=current_hit_reward_one,
                        child_q_before=child_q_before,
                        child_visits_before=child_visits_before,
                        child_q_after=child.q_value,
                        child_visits_after=child.visits,
                        parent_q_before=parent_q_before,
                        parent_visits_before=parent_visits_before,
                        parent_q_after=selected.q_value,
                        parent_visits_after=selected.visits,
                        rollout_details=[rollout_detail],
                        simulation_index=iter_idx,
                        rollout_index=ridx,
                        group_id=group_id,
                        grpo_report=dict(grpo_report),
                        iteration=iter_idx,
                    )
                )

                rollout_summaries.append(
                    {
                        "rollout_index": ridx,
                        "node_id": child.node_id,
                        "reward": {
                            "r1": reward_obj.r1,
                            "r2": reward_obj.r2,
                            "r3": reward_obj.r3,
                            "total": reward_obj.total,
                        },
                        "prior": child.prior,
                        "completion_preview": child.text[:240],
                    }
                )

                if reward_total > best_reward:
                    best_reward = reward_total
                    best_trajectory = completed

                if current_hit_reward_one:
                    hit_reward_one = True
                    reward_one_payload = {
                        "reason": "reward_one",
                        "iteration": iter_idx,
                        "stage": next_stage.value,
                        "node_id": child.node_id,
                        "trajectory_id": completed.trajectory_id,
                        "reward_total": reward_total,
                    }
                    break

            if not processed_group_rollouts:
                continue

            best_rollout = max(processed_group_rollouts, key=lambda x: float(x.get("reward_total", float("-inf"))))
            best_rollout_obj = best_rollout["reward_obj"]
            best_rollout_child = best_rollout["child"]

            iteration_logs.append(
                {
                    "iteration": iter_idx,
                    "stage": next_stage.value,
                    "selected_leaf": {
                        "node_id": selected.node_id,
                        "leaf_stage": selected.stage.value if selected.stage else "<ROOT>",
                        "score": selected_score,
                        "q": selected.q_value,
                        "visits": selected.visits,
                    },
                    "leaf_candidates": [
                        {
                            "node_id": node.node_id,
                            "leaf_stage": node.stage.value if node.stage else "<ROOT>",
                            "score": float(score),
                            "q": node.q_value,
                            "visits": node.visits,
                            "parent_id": node.parent.node_id if node.parent else "",
                        }
                        for node, score in ranked_leaves
                    ],
                    "group_id": group_id,
                    "num_rollouts": len(rollout_summaries),
                    "rollouts": rollout_summaries,
                    "best_rollout": {
                        "rollout_index": int(best_rollout.get("rollout_index", -1)),
                        "node_id": best_rollout_child.node_id,
                        "reward": {
                            "r1": best_rollout_obj.r1,
                            "r2": best_rollout_obj.r2,
                            "r3": best_rollout_obj.r3,
                            "total": best_rollout_obj.total,
                        },
                        "prior": best_rollout_child.prior,
                        "completion": best_rollout_child.text,
                    },
                    "grpo_report": dict(grpo_report),
                }
            )

            if hit_reward_one:
                stop_info = reward_one_payload
                return SearchRunResult(
                    root=root,
                    records=records,
                    stop_info=stop_info,
                    best_trajectory=best_trajectory,
                    best_reward=best_reward,
                    code_nodes=self._collect_code_nodes(root),
                    iteration_logs=iteration_logs,
                )

            if next_stage == Stage.CODE:
                stop_info = {
                    "reason": "expanded_to_code",
                    "iteration": iter_idx,
                    "stage": next_stage.value,
                    "node_id": selected.node_id,
                    "trajectory_id": best_trajectory.trajectory_id if best_trajectory else "",
                    "reward_total": (best_trajectory.reward.total if best_trajectory and best_trajectory.reward else None),
                }
                break

        return SearchRunResult(
            root=root,
            records=records,
            stop_info=stop_info,
            best_trajectory=best_trajectory,
            best_reward=best_reward,
            code_nodes=self._collect_code_nodes(root),
            iteration_logs=iteration_logs,
        )

    def _run_internal_grpo_rollout(
        self,
        stage: Stage,
        prompt: str,
        grpo_config: GRPOConfig,
        reward_callback: Callable[[str, str, int], float],
    ) -> tuple[list[Generation], dict[str, Any]]:
        method = getattr(self.backend, "grpo_rollout_group", None)
        if callable(method):
            return method(stage, prompt, grpo_config, reward_callback)

        group_n = max(1, int(grpo_config.num_generations))
        generations = self.backend.generate(stage, prompt, group_n)
        for ridx, gen in enumerate(generations):
            reward_total = float(reward_callback(prompt, gen.text, ridx))
            gen.metadata["reward_total"] = reward_total
        report = {
            "updated": False,
            "backend": type(self.backend).__name__,
            "num_samples": len(generations),
            "reason": "backend_has_no_internal_grpo_rollout",
        }
        return generations, report

    def _complete_for_reward(self, task: OptimizationTask, from_node: SearchNode) -> Trajectory:
        if from_node.stage == Stage.CODE:
            partial = from_node.to_partial_trajectory()
            partial.trajectory_id = str(uuid.uuid4())
            return partial
        return self.rollout_to_code(task, from_node)

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
            partial.outputs[next_stage] = self._extract_stage_payload(next_stage, generation.text)
            partial.priors[next_stage] = generation.prior

        return partial

    def _rank_leaves(self, leaves: list[SearchNode]) -> list[tuple[SearchNode, float]]:
        ranked = [(node, self._leaf_score(node)) for node in leaves]
        ranked.sort(key=lambda item: item[1], reverse=True)
        return ranked

    def _leaf_score(self, node: SearchNode) -> float:
        if node.parent is None:
            return float(node.q_value)
        return float(self.selector.score(node.parent, node))

    @staticmethod
    def _backpropagate(node: SearchNode, reward: float) -> None:
        cur: SearchNode | None = node
        while cur is not None:
            cur.update(reward)
            cur = cur.parent

    @staticmethod
    def _iter_leaves(root: SearchNode) -> list[SearchNode]:
        leaves: list[SearchNode] = []
        stack: list[SearchNode] = [root]
        while stack:
            cur = stack.pop()
            if not cur.children:
                leaves.append(cur)
                continue
            stack.extend(cur.children)
        return leaves

    @staticmethod
    def _collect_code_nodes(root: SearchNode) -> list[SearchNode]:
        nodes: list[SearchNode] = []
        stack: list[SearchNode] = [root]
        while stack:
            cur = stack.pop()
            if cur.stage == Stage.CODE:
                nodes.append(cur)
            stack.extend(cur.children)
        return nodes

    @staticmethod
    def _next_stage(stage: Stage | None) -> Stage | None:
        if stage is None:
            return STAGE_ORDER[0]
        idx = STAGE_ORDER.index(stage)
        if idx >= len(STAGE_ORDER) - 1:
            return None
        return STAGE_ORDER[idx + 1]

    @staticmethod
    def _extract_stage_payload(stage: Stage, text: str) -> str:
        cleaned = (text or "").strip()
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            if len(lines) >= 2 and lines[-1].strip().startswith("```"):
                cleaned = "\n".join(lines[1:-1]).strip()

        if stage == Stage.SCHEMA:
            l = cleaned.find("{")
            r = cleaned.rfind("}")
            if l >= 0 and r > l:
                return cleaned[l : r + 1].strip()
            return cleaned

        if stage == Stage.SET_PARAM_VAR:
            keys = ["Sets", "Parameters", "Variables"]
            lines = [ln.rstrip() for ln in cleaned.splitlines()]
            keep = [ln for ln in lines if any(k.lower() in ln.lower() for k in keys) or ln.strip().startswith("-")]
            return "\n".join(keep).strip() or cleaned

        if stage == Stage.OBJ_CONS:
            keys = ["Objective", "Constraint"]
            lines = [ln.rstrip() for ln in cleaned.splitlines()]
            keep = [
                ln
                for ln in lines
                if any(k.lower() in ln.lower() for k in keys)
                or ln.strip().startswith(tuple(str(i) + ")" for i in range(1, 10)))
            ]
            return "\n".join(keep).strip() or cleaned

        if stage == Stage.CODE:
            return FourStageMCTS._sanitize_code_payload(cleaned)

        return cleaned

    @staticmethod
    def _sanitize_code_payload(text: str) -> str:
        cleaned = (text or "").strip()
        if not cleaned:
            return cleaned

        lines = cleaned.splitlines()

        in_block = False
        block_lines: list[str] = []
        blocks: list[str] = []
        for line in lines:
            if line.strip().startswith("```"):
                if in_block:
                    blocks.append("\n".join(block_lines).strip())
                    block_lines = []
                    in_block = False
                else:
                    in_block = True
                continue
            if in_block:
                block_lines.append(line)

        if in_block and block_lines:
            blocks.append("\n".join(block_lines).strip())

        if blocks:
            cleaned = max(blocks, key=len)
        else:
            cleaned = "\n".join(ln for ln in lines if not ln.strip().startswith("```"))

        code_lines = cleaned.splitlines()
        start_idx = None
        for idx, line in enumerate(code_lines):
            stripped = line.strip()
            if stripped.startswith(("import ", "from ", "def solve(", "@", "class ")):
                start_idx = idx
                break

        if start_idx is not None:
            code_lines = code_lines[start_idx:]

        cleaned = "\n".join(code_lines).strip()
        return cleaned
