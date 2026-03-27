from __future__ import annotations

import re
import time
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
        iteration_callback: Callable[[dict[str, Any]], None] | None = None,
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
            iter_t0 = time.perf_counter()
            selection_t0 = time.perf_counter()
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
            selection_sec = float(time.perf_counter() - selection_t0)

            prompt_t0 = time.perf_counter()
            selected_traj = None if selected.stage is None else selected.to_partial_trajectory()
            base_prompt = self.prompt_builder.build(task, next_stage, selected_traj)
            prompt = self.prompt_builder.build_rollout(task, next_stage, selected_traj)
            prompt_build_sec = float(time.perf_counter() - prompt_t0)

            group_id = f"iter:{iter_idx}:{selected.node_id}:{next_stage.value}"
            group_rollouts: list[dict[str, Any]] = []
            stage_archive = stage_archives[next_stage]
            callback_timings: list[dict[str, Any]] = []

            def _reward_callback(prompt_text: str, completion_text: str, ridx: int) -> float:
                callback_t0 = time.perf_counter()

                parse_t0 = time.perf_counter()
                current_stage_text, rollout_suffix = self._split_rollout_completion(completion_text)
                parsed_text = self._extract_stage_payload(next_stage, current_stage_text)
                parse_sec = float(time.perf_counter() - parse_t0)

                child = SearchNode(
                    stage=next_stage,
                    text=parsed_text,
                    prior=1.0,
                    parent=selected,
                    prompt=base_prompt,
                )

                complete_t0 = time.perf_counter()
                completed = self._complete_for_reward_from_rollout(task, child, rollout_suffix)
                complete_sec = float(time.perf_counter() - complete_t0)

                reward_t0 = time.perf_counter()
                reward = self.rewarder.provisional_reward(completed, stage_archive)
                reward_sec = float(time.perf_counter() - reward_t0)
                completed.reward = reward
                stage_archive.append(completed)

                exec_sec = float((reward.metadata or {}).get("exec_elapsed_sec", 0.0) or 0.0)
                callback_total_sec = float(time.perf_counter() - callback_t0)
                timing_payload = {
                    "completion_parse_sec": parse_sec,
                    "rollout_to_code_sec": complete_sec,
                    "reward_compute_sec": reward_sec,
                    "code_execution_sec": exec_sec,
                    "callback_total_sec": callback_total_sec,
                }
                callback_timings.append({"rollout_index": ridx, **timing_payload})

                group_rollouts.append(
                    {
                        "rollout_index": ridx,
                        "child": child,
                        "trajectory": completed,
                        "reward_obj": reward,
                        "reward_total": float(reward.total),
                        "timing": timing_payload,
                        "prompt_base": base_prompt,
                        "prompt_full": prompt,
                        "completion_full": completion_text,
                        "answer_current_stage": current_stage_text,
                        "answer_rollout_suffix": rollout_suffix,
                    }
                )
                return float(reward.total)

            rollout_group_t0 = time.perf_counter()
            generations, grpo_report = self._run_internal_grpo_rollout(
                stage=next_stage,
                prompt=prompt,
                grpo_config=grpo_config,
                reward_callback=_reward_callback,
            )
            rollout_group_wall_sec = float(time.perf_counter() - rollout_group_t0)

            callback_total_sec = float(sum(t.get("callback_total_sec", 0.0) for t in callback_timings))
            callback_exec_sec = float(sum(t.get("code_execution_sec", 0.0) for t in callback_timings))
            callback_complete_sec = float(sum(t.get("rollout_to_code_sec", 0.0) for t in callback_timings))
            callback_reward_sec = float(sum(t.get("reward_compute_sec", 0.0) for t in callback_timings))
            callback_parse_sec = float(sum(t.get("completion_parse_sec", 0.0) for t in callback_timings))
            model_sampling_update_sec = float(max(0.0, rollout_group_wall_sec - callback_total_sec))

            if not group_rollouts:
                continue

            hit_reward_one = False
            reward_one_payload: dict[str, Any] = {}
            rollout_summaries: list[dict[str, Any]] = []
            processed_group_rollouts: list[dict[str, Any]] = []
            backprop_total_sec = 0.0

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

                backprop_t0 = time.perf_counter()
                selected.add_child(child)
                self._backpropagate(child, reward_total)
                backprop_sec = float(time.perf_counter() - backprop_t0)
                backprop_total_sec += backprop_sec

                rollout_timing = dict(rollout.get("timing", {}))
                rollout_timing["backprop_sec"] = backprop_sec
                completed.metadata["iter"] = int(iter_idx)
                completed.metadata["stage"] = next_stage.value
                completed.metadata["group_id"] = group_id
                completed.metadata["parent_node"] = {
                    "node_id": selected.node_id,
                    "stage": (selected.stage.value if selected.stage else "<ROOT>"),
                    "value": float(selected.q_value),
                    "visits": int(selected.visits),
                    "content": selected.text,
                }

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
                    "timing": rollout_timing,
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

                rollout["timing"] = rollout_timing
                rollout["update"] = {
                    "prior": float(child.prior),
                    "child_q_before": float(child_q_before),
                    "child_q_after": float(child.q_value),
                    "child_visits_before": int(child_visits_before),
                    "child_visits_after": int(child.visits),
                    "parent_q_before": float(parent_q_before),
                    "parent_q_after": float(selected.q_value),
                    "parent_visits_before": int(parent_visits_before),
                    "parent_visits_after": int(selected.visits),
                    "backprop_sec": float(backprop_sec),
                }

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
                        "obj_answer": (reward_obj.metadata or {}).get("obj_answer"),
                        "timing": rollout_timing,
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
            best_rollout_traj = best_rollout["trajectory"]
            best_rollout_timing = dict(best_rollout.get("timing", {}))
            best_rollout_update = dict(best_rollout.get("update", {}))

            grpo_train_runtime_sec = float((grpo_report or {}).get("train_runtime", 0.0) or 0.0)
            iter_total_sec = float(time.perf_counter() - iter_t0)

            iter_payload = {
                "iter": int(iter_idx),
                "stage": next_stage.value,
                "selection": {
                    "selected_parent": {
                        "node_id": selected.node_id,
                        "stage": selected.stage.value if selected.stage else "<ROOT>",
                        "value": float(selected.q_value),
                        "visits": int(selected.visits),
                        "puct_score": float(selected_score),
                        "content": selected.text,
                    },
                    "leaf_candidates": [
                        {
                            "node_id": node.node_id,
                            "stage": node.stage.value if node.stage else "<ROOT>",
                            "puct_score": float(score),
                            "value": float(node.q_value),
                            "visits": int(node.visits),
                            "parent_id": node.parent.node_id if node.parent else "",
                            "content": node.text,
                        }
                        for node, score in ranked_leaves
                    ],
                },
                "expansion": {
                    "group_id": group_id,
                    "k": len(rollout_summaries),
                    "rollout_stage": next_stage.value,
                },
                "best_rollout": {
                    "rollout_index": int(best_rollout.get("rollout_index", -1)),
                    "child_node_id": best_rollout_child.node_id,
                    "parent_node": {
                        "node_id": selected.node_id,
                        "stage": selected.stage.value if selected.stage else "<ROOT>",
                        "value": float(selected.q_value),
                        "visits": int(selected.visits),
                        "content": selected.text,
                    },
                    "prompt": {
                        "base": str(best_rollout.get("prompt_base", "")),
                        "full": str(best_rollout.get("prompt_full", "")),
                    },
                    "answer": {
                        "full": str(best_rollout.get("completion_full", "")),
                        "current_stage": str(best_rollout.get("answer_current_stage", "")),
                        "rollout_suffix": str(best_rollout.get("answer_rollout_suffix", "")),
                    },
                    "trajectory_content": {
                        stage.value: best_rollout_traj.outputs.get(stage, "")
                        for stage in STAGE_ORDER
                    },
                    "code": best_rollout_traj.code,
                    "code_execution": (best_rollout_obj.metadata or {}).get("execution", {}),
                    "gt": str(task.gold_answer or ""),
                    "reward": {
                        "r1": float(best_rollout_obj.r1),
                        "r2": float(best_rollout_obj.r2),
                        "r3": float(best_rollout_obj.r3),
                        "total": float(best_rollout_obj.total),
                        "obj_answer": (best_rollout_obj.metadata or {}).get("obj_answer"),
                    },
                    "timing": best_rollout_timing,
                    "update": best_rollout_update,
                },
                "timing": {
                    "selection_sec": selection_sec,
                    "prompt_build_sec": prompt_build_sec,
                    "rollout_group_wall_sec": rollout_group_wall_sec,
                    "grpo_train_runtime_sec": grpo_train_runtime_sec,
                    "reward_callback_total_sec": callback_total_sec,
                    "completion_parse_total_sec": callback_parse_sec,
                    "rollout_to_code_total_sec": callback_complete_sec,
                    "reward_compute_total_sec": callback_reward_sec,
                    "code_execution_total_sec": callback_exec_sec,
                    "model_sampling_update_excluding_reward_callback_sec": model_sampling_update_sec,
                    "backprop_total_sec": backprop_total_sec,
                    "iteration_total_sec": iter_total_sec,
                },
                "grpo_update": dict(grpo_report),
            }
            iteration_logs.append(iter_payload)
            if iteration_callback is not None:
                iteration_callback(iter_payload)

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

    @staticmethod
    def _split_rollout_completion(text: str) -> tuple[str, str]:
        marker = "### ROLLOUT_CONTINUATION"
        raw = text or ""
        idx = raw.find(marker)
        if idx < 0:
            return raw.strip(), ""
        current = raw[:idx].strip()
        suffix = raw[idx + len(marker) :].strip()
        return current, suffix

    @staticmethod
    def _extract_rollout_stage_block(rollout_suffix: str, stage: Stage) -> str:
        if not rollout_suffix.strip():
            return ""
        tag = f"ROLLOUT_STAGE_{stage.value.upper()}"
        pattern = rf"<\s*{tag}\s*>(.*?)<\s*/\s*{tag}\s*>"
        blocks = re.findall(pattern, rollout_suffix, flags=re.IGNORECASE | re.DOTALL)
        if blocks:
            return max((b.strip() for b in blocks), key=len, default="")
        if stage == Stage.CODE:
            return rollout_suffix.strip()
        return ""

    def _complete_for_reward_from_rollout(
        self,
        task: OptimizationTask,
        from_node: SearchNode,
        rollout_suffix: str,
    ) -> Trajectory:
        partial = from_node.to_partial_trajectory()
        partial.trajectory_id = str(uuid.uuid4())

        if from_node.stage is None:
            return partial
        if from_node.stage == Stage.CODE:
            return partial

        start_idx = STAGE_ORDER.index(from_node.stage)
        for next_stage in STAGE_ORDER[start_idx + 1 :]:
            block = self._extract_rollout_stage_block(rollout_suffix, next_stage)
            if not block:
                continue
            parsed = self._extract_stage_payload(next_stage, block)
            if not parsed:
                continue
            partial.outputs[next_stage] = parsed
            partial.priors[next_stage] = from_node.prior

        if partial.outputs.get(Stage.CODE, "").strip():
            return partial

        # Fallback for non-compliant completion formats.
        return self._complete_for_reward(task, from_node)

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

        # Unified extraction policy:
        # 1) Prefer explicit <Gurobi_code> ... </Gurobi_code> blocks.
        # 2) Fallback to fenced code block extraction.
        tag_blocks = re.findall(
            r"<\s*gurobi_code\s*>(.*?)<\s*/\s*gurobi_code\s*>",
            cleaned,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if tag_blocks:
            cleaned = max((block.strip() for block in tag_blocks), key=len, default="")
            if not cleaned:
                cleaned = (text or "").strip()


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
