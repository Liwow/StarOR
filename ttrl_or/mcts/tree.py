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
        stage_order: tuple[Stage, ...] | None = None,
        split_rollout_completion: bool = False,
    ) -> None:
        self.backend = backend
        self.prompt_builder = prompt_builder
        self.rewarder = rewarder
        self.config = config
        self.selector = selector or PUCTSelector(c_puct=config.c_puct)
        self.stage_order = tuple(stage_order or STAGE_ORDER)
        self.split_rollout_completion = bool(split_rollout_completion)
        self._active_grpo_config: GRPOConfig | None = None

    def root(self) -> SearchNode:
        return SearchNode(stage=None, text="<ROOT>", prior=1.0)

    def search(
        self,
        task: OptimizationTask,
        grpo_config: GRPOConfig,
        iteration_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> SearchRunResult:
        self._active_grpo_config = grpo_config
        root = self.root()
        records: list[StageExpansionRecord] = []
        iteration_logs: list[dict[str, Any]] = []
        best_trajectory: Trajectory | None = None
        best_reward = float("-inf")

        stage_archives: dict[Stage, list[Trajectory]] = {stage: [] for stage in self.stage_order}

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
            ranked_leaves = self._rank_expandable_leaves(root)
            if not ranked_leaves:
                stop_info = {
                    "reason": "no_expandable_leaf",
                    "iteration": iter_idx,
                    "stage": "",
                    "node_id": "",
                    "trajectory_id": "",
                    "reward_total": None,
                }
                break

            selected, selected_score, selection_path = self._select_leaf_recursive(root)

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

            def _prepare_rollout_item(completion_text: str, ridx: int) -> dict[str, Any]:
                callback_t0 = time.perf_counter()

                parse_t0 = time.perf_counter()
                parsed_text = self._extract_stage_payload(next_stage, completion_text)
                parse_sec = float(time.perf_counter() - parse_t0)

                child = SearchNode(
                    stage=next_stage,
                    text=parsed_text,
                    prior=1.0,
                    parent=selected,
                    prompt=base_prompt,
                )

                complete_t0 = time.perf_counter()
                completed = self._complete_for_reward_from_rollout(task, child, completion_text)
                complete_sec = float(time.perf_counter() - complete_t0)

                return {
                    "rollout_index": ridx,
                    "callback_started": callback_t0,
                    "child": child,
                    "trajectory": completed,
                    "completion_full": completion_text,
                    "answer_current_stage": parsed_text,
                    "answer_rollout_suffix": "",
                    "parse_sec": parse_sec,
                    "complete_sec": complete_sec,
                }

            def _score_prepared_group(prepared_items: list[dict[str, Any]]) -> list[float]:
                if not prepared_items:
                    return []

                reward_t0 = time.perf_counter()
                trajectories = [item["trajectory"] for item in prepared_items]
                score_group = getattr(self.rewarder, "score_rollout_group", None)
                if callable(score_group):
                    reward_list = list(score_group(stage=next_stage, trajectories=trajectories, explored=stage_archive))
                else:
                    reward_list = [self.rewarder.provisional_reward(t, stage_archive) for t in trajectories]
                reward_sec_total = float(time.perf_counter() - reward_t0)

                if len(reward_list) != len(prepared_items):
                    reward_list = reward_list[: len(prepared_items)]
                    while len(reward_list) < len(prepared_items):
                        reward_list.append(RewardBreakdown(r1=0.0, r2=0.0, r3=0.0, r4=0.0, total=0.0))

                rewards: list[float] = []
                per_item_reward_sec = reward_sec_total / max(1, len(prepared_items))

                for item, reward in zip(prepared_items, reward_list, strict=False):
                    completed = item["trajectory"]
                    completed.reward = reward
                    stage_archive.append(completed)

                    exec_sec = float((reward.metadata or {}).get("exec_elapsed_sec", 0.0) or 0.0)
                    callback_total_sec = float(time.perf_counter() - float(item["callback_started"]))
                    timing_payload = {
                        "completion_parse_sec": float(item["parse_sec"]),
                        "rollout_to_code_sec": float(item["complete_sec"]),
                        "reward_compute_sec": float(per_item_reward_sec),
                        "code_execution_sec": exec_sec,
                        "callback_total_sec": callback_total_sec,
                    }
                    callback_timings.append({"rollout_index": int(item["rollout_index"]), **timing_payload})

                    group_rollouts.append(
                        {
                            "rollout_index": int(item["rollout_index"]),
                            "child": item["child"],
                            "trajectory": completed,
                            "reward_obj": reward,
                            "reward_total": float(reward.total),
                            "timing": timing_payload,
                            "prompt_base": base_prompt,
                            "prompt_full": prompt,
                            "completion_full": str(item["completion_full"]),
                            "answer_current_stage": str(item["answer_current_stage"]),
                            "answer_rollout_suffix": str(item["answer_rollout_suffix"]),
                        }
                    )
                    rewards.append(float(reward.total))

                return rewards

            def _batch_reward_callback(prompt_text: str, completion_texts: list[str]) -> list[float]:
                prepared = [
                    _prepare_rollout_item(completion_text=text, ridx=ridx)
                    for ridx, text in enumerate(completion_texts)
                ]
                return _score_prepared_group(prepared)

            def _reward_callback(prompt_text: str, completion_text: str, ridx: int) -> float:
                rewards = _batch_reward_callback(prompt_text, [completion_text])
                return float(rewards[0]) if rewards else 0.0

            setattr(_reward_callback, "batch_score", _batch_reward_callback)

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

            resolved_priors, prior_source = self._resolve_child_priors(
                stage=next_stage,
                prompt=base_prompt,
                rollouts=group_rollouts,
                fallback_generations=generations,
            )
            for ridx, rollout in enumerate(group_rollouts):
                child = rollout["child"]
                prior_value = float(resolved_priors[ridx]) if ridx < len(resolved_priors) else max(1e-6, float(child.prior))
                child.prior = max(1e-6, prior_value)
                rollout["prior_source"] = prior_source
                rollout["resolved_prior"] = float(child.prior)
                rollout["trajectory"].priors[next_stage] = float(child.prior)

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
                        "r4": reward_obj.r4,
                        "total": reward_obj.total,
                        "consensus_signature": reward_obj.consensus_signature,
                        "execution_success": reward_obj.execution_success,
                        "robustness_success": reward_obj.robustness_success,
                    },
                    "timing": rollout_timing,
                    "priors": {s.value: p for s, p in completed.priors.items()},
                    "prior_source": str(rollout.get("prior_source", "fallback_generation")),
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
                            "r4": reward_obj.r4,
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
                    "selection_path": selection_path,
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
                        for stage in self.stage_order
                    },
                    "code": best_rollout_traj.code,
                    "code_execution": (best_rollout_obj.metadata or {}).get("execution", {}),
                    "gt": str(task.gold_answer or ""),
                    "reward": {
                        "r1": float(best_rollout_obj.r1),
                        "r2": float(best_rollout_obj.r2),
                        "r3": float(best_rollout_obj.r3),
                        "r4": float(best_rollout_obj.r4),
                        "total": float(best_rollout_obj.total),
                        "obj_answer": (best_rollout_obj.metadata or {}).get("obj_answer"),
                        "r1_debug": (best_rollout_obj.metadata or {}).get("r1_debug", {}),
                        "r4_debug": (best_rollout_obj.metadata or {}).get("r4_debug", {}),
                    },
                    "prior": {
                        "source": str(best_rollout.get("prior_source", "fallback_generation")),
                        "resolved_prior": float(best_rollout_child.prior),
                    },
                    "timing": best_rollout_timing,
                    "update": best_rollout_update,
                },
                "rollout_group": rollout_summaries,
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
        cleaned = (text or "").strip()
        return cleaned, cleaned

    @staticmethod
    def _tag_for_stage(stage: Stage) -> str:
        if stage in (Stage.SCHEMA, Stage.TYPE_HINT):
            return "stage_1"
        if stage in (Stage.SET_PARAM_VAR, Stage.SETS):
            return "stage_2"
        if stage in (Stage.OBJ_CONS, Stage.PARAMETERS):
            return "stage_3"
        if stage == Stage.VARIABLES:
            return "stage_4"
        if stage == Stage.OBJECTIVE:
            return "stage_5"
        if stage == Stage.CONSTRAINTS:
            return "stage_6"
        return "Gurobi_code"

    @staticmethod
    def _extract_tag_block(text: str, tag: str, min_len: int = 0) -> str:
        """Extract content from the FIRST valid closed tag pair.

        Supports two formats (in priority order):
        1. <tag>...</tag>  (angle brackets, preferred)
        2. [tag]...[/tag]  (square brackets, fallback)

        Logic (from front to back):
        1. Find the first closing tag
        2. Find the nearest opening tag before it
        3. If content length > min_len, return it
        4. Otherwise, continue to the next closing tag
        """
        raw = (text or "").strip()
        if not raw:
            return ""

        required_len = max(21, int(min_len))

        result = FourStageMCTS._extract_tag_block_with_delimiters(
            raw, tag, "<", ">", required_len
        )
        if result:
            return result

        return FourStageMCTS._extract_tag_block_with_delimiters(
            raw, tag, "[", "]", required_len
        )

    @staticmethod
    def _extract_tag_block_with_delimiters(
        text: str,
        tag: str,
        open_delim: str,
        close_delim: str,
        required_len: int,
    ) -> str:
        """Extract content using specific delimiters (e.g., < > or [ ])."""
        od = re.escape(open_delim)
        cd = re.escape(close_delim)

        open_re = re.compile(
            rf"{od}\s*{re.escape(tag)}\s*{cd}",
            flags=re.IGNORECASE,
        )
        close_re = re.compile(
            rf"{od}\s*/\s*{re.escape(tag)}\s*{cd}",
            flags=re.IGNORECASE,
        )

        open_matches = list(open_re.finditer(text))
        close_matches = list(close_re.finditer(text))

        if not open_matches or not close_matches:
            return ""

        for close_match in close_matches:
            close_start = close_match.start()

            nearest_open = None
            for open_match in reversed(open_matches):
                if open_match.end() <= close_start:
                    nearest_open = open_match
                    break

            if nearest_open is None:
                continue

            content = text[nearest_open.end():close_start]
            cleaned = content.strip()
            if len(cleaned) >= required_len:
                return cleaned

        return ""

    @staticmethod
    def _extract_rollout_stage_block(rollout_text: str, stage: Stage) -> str:
        tag = FourStageMCTS._tag_for_stage(stage)
        return FourStageMCTS._extract_tag_block(rollout_text, tag=tag, min_len=21)

    def _complete_for_reward_from_rollout(
        self,
        task: OptimizationTask,
        from_node: SearchNode,
        full_completion: str = "",
    ) -> Trajectory:
        if self.split_rollout_completion and from_node.stage != Stage.CODE:
            return self._complete_for_reward_split(task, from_node)
        return self._trajectory_from_rollout_completion(from_node, full_completion)

    def _trajectory_from_rollout_completion(self, from_node: SearchNode, full_completion: str = "") -> Trajectory:
        partial = from_node.to_partial_trajectory()
        partial.trajectory_id = str(uuid.uuid4())

        if from_node.stage is None:
            return partial
        if from_node.stage == Stage.CODE:
            return partial

        full_text = str(full_completion or "")
        start_idx = self.stage_order.index(from_node.stage)
        for next_stage in self.stage_order[start_idx + 1 :]:
            block = self._extract_rollout_stage_block(full_text, next_stage)
            if not block:
                continue

            if next_stage == Stage.CODE:
                parsed = self._extract_stage_payload(next_stage, block)
            else:
                parsed = self._normalize_text_block(block)
                if len((parsed or "").strip()) <= 20:
                    parsed = ""
            if not parsed:
                continue
            partial.outputs[next_stage] = parsed
            partial.priors[next_stage] = from_node.prior

        if not partial.outputs.get(Stage.CODE, "").strip():
            partial.metadata["rollout_missing_code"] = True
        return partial

    def _complete_for_reward_split(self, task: OptimizationTask, from_node: SearchNode) -> Trajectory:
        partial = from_node.to_partial_trajectory()
        partial.trajectory_id = str(uuid.uuid4())
        completion_prompt = self.prompt_builder.build_completion(task, from_node.stage, partial)
        completion_text = ""
        if completion_prompt:
            completion_text = str(
                self.backend.generate_auxiliary_text(
                    completion_prompt,
                    max_new_tokens=int(getattr(self.backend, "max_new_tokens", 2048) or 2048),
                    temperature=float(getattr(self.backend, "temperature", 0.0) or 0.0),
                    top_p=float(getattr(self.backend, "top_p", 1.0) or 1.0),
                    prefer_vllm=bool(self._active_grpo_config.use_vllm) if self._active_grpo_config is not None else False,
                    vllm_mode=str(self._active_grpo_config.vllm_mode) if self._active_grpo_config is not None else "",
                )
                or ""
            )
        if not completion_text.strip():
            return self.rollout_to_code(task, from_node)
        return self._trajectory_from_rollout_completion(from_node, completion_text)

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
            start_idx = self.stage_order.index(from_node.stage)

        for next_stage in self.stage_order[start_idx + 1 :]:
            prompt = self.prompt_builder.build(task, next_stage, partial)
            generation = self.backend.generate(next_stage, prompt, 1)[0]
            partial.outputs[next_stage] = self._extract_stage_payload(next_stage, generation.text)
            partial.priors[next_stage] = generation.prior

        return partial

    def _resolve_child_priors(
        self,
        stage: Stage,
        prompt: str,
        rollouts: list[dict[str, Any]],
        fallback_generations: list[Generation],
    ) -> tuple[list[float], str]:
        fallback: list[float] = []
        for ridx, rollout in enumerate(rollouts):
            if ridx < len(fallback_generations):
                fallback.append(max(1e-6, float(fallback_generations[ridx].prior)))
            else:
                fallback.append(max(1e-6, float(rollout['child'].prior)))

        score_method = getattr(self.backend, 'score_action_priors', None)
        if not callable(score_method):
            return self._normalize_priors(fallback), 'fallback_generation'

        candidates = [str(rollout['child'].text or '') for rollout in rollouts]
        try:
            priors = list(score_method(stage=stage, prompt=prompt, candidates=candidates))
        except TypeError:
            priors = list(score_method(stage, prompt, candidates))
        except Exception:
            priors = []

        if len(priors) != len(rollouts):
            return self._normalize_priors(fallback), 'fallback_generation'

        normalized = self._normalize_priors(priors)
        if not normalized:
            return self._normalize_priors(fallback), 'fallback_generation'
        return normalized, 'teacher_forcing_lora'

    def _select_leaf_recursive(self, root: SearchNode) -> tuple[SearchNode, float, list[dict[str, Any]]]:
        cur = root
        selection_path: list[dict[str, Any]] = []

        while cur.children:
            candidate_scores: list[tuple[SearchNode, float]] = []
            for child in cur.children:
                if not self._subtree_has_expandable_leaf(child):
                    continue
                candidate_scores.append((child, float(self.selector.score(cur, child))))

            if not candidate_scores:
                break

            candidate_scores.sort(key=lambda item: item[1], reverse=True)
            chosen, chosen_score = candidate_scores[0]
            selection_path.append(
                {
                    'parent_node_id': cur.node_id,
                    'parent_stage': cur.stage.value if cur.stage else '<ROOT>',
                    'candidates': [
                        {
                            'node_id': node.node_id,
                            'stage': node.stage.value if node.stage else '<ROOT>',
                            'puct_score': float(score),
                            'value': float(node.q_value),
                            'visits': int(node.visits),
                            'prior': float(node.prior),
                            'content': node.text,
                        }
                        for node, score in candidate_scores
                    ],
                    'selected_child_id': chosen.node_id,
                    'selected_child_score': float(chosen_score),
                }
            )
            cur = chosen

        return cur, float(self._leaf_score(cur)), selection_path

    def _rank_expandable_leaves(self, root: SearchNode) -> list[tuple[SearchNode, float]]:
        leaves = [node for node in self._iter_leaves(root) if self._next_stage(node.stage) is not None]
        return self._rank_leaves(leaves)

    def _subtree_has_expandable_leaf(self, node: SearchNode) -> bool:
        if not node.children:
            return self._next_stage(node.stage) is not None
        return any(self._subtree_has_expandable_leaf(child) for child in node.children)

    @staticmethod
    def _normalize_priors(values: list[float]) -> list[float]:
        if not values:
            return []
        cleaned = [max(0.0, float(v)) if v == v else 0.0 for v in values]
        total = sum(cleaned)
        if total <= 0:
            return [1.0 / float(len(cleaned))] * len(cleaned)
        return [float(v / total) for v in cleaned]

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

    def _next_stage(self, stage: Stage | None) -> Stage | None:
        if stage is None:
            return self.stage_order[0]
        idx = self.stage_order.index(stage)
        if idx >= len(self.stage_order) - 1:
            return None
        return self.stage_order[idx + 1]

    @staticmethod
    def _extract_stage_payload(stage: Stage, text: str) -> str:
        cleaned = FourStageMCTS._normalize_text_block(text)

        if stage == Stage.CODE:
            code = FourStageMCTS._sanitize_code_payload(cleaned)
            if len((code or "").strip()) > 20:
                return code
            return ""

        tag = FourStageMCTS._tag_for_stage(stage)
        block = FourStageMCTS._extract_tag_block(cleaned, tag=tag, min_len=21)
        if block:
            return block.strip()

        return ""

    @staticmethod
    def _normalize_text_block(text: str) -> str:
        cleaned = (text or "").strip()
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            if len(lines) >= 2 and lines[-1].strip().startswith("```"):
                cleaned = "\n".join(lines[1:-1]).strip()
        return cleaned

    @staticmethod
    def _strip_rollout_region(text: str) -> str:
        raw = text or ""

        marker = "### ROLLOUT_CONTINUATION"
        idx = raw.find(marker)
        if idx >= 0:
            raw = raw[:idx]

        tag_match = re.search(r"<\s*ROLLOUT_STAGE_[A-Z_]+\s*>", raw, flags=re.IGNORECASE)
        if tag_match:
            raw = raw[: int(tag_match.start())]

        return raw.strip()

    @staticmethod
    def _extract_named_sections(text: str, section_headers: list[str], stop_headers: list[str]) -> str:
        lines = text.splitlines()
        if not lines:
            return ""

        lower_lines = [ln.strip().lower() for ln in lines]
        all_headers = [h.lower() for h in (section_headers + stop_headers)]

        blocks: list[str] = []
        for header in section_headers:
            h = header.lower()
            start = -1
            for i, ln in enumerate(lower_lines):
                if ln.startswith(h):
                    start = i
                    break
            if start < 0:
                continue

            end = len(lines)
            for j in range(start + 1, len(lines)):
                ln = lower_lines[j]
                if any(ln.startswith(x) for x in all_headers):
                    end = j
                    break

            chunk = "\n".join(lines[start:end]).strip()
            if chunk:
                blocks.append(chunk)

        return "\n\n".join(blocks).strip()

    @staticmethod
    def _looks_like_code(text: str) -> bool:
        raw = (text or "").strip()
        if not raw:
            return False
        patterns = (
            r"\bdef\s+solve\s*\(",
            r"\bimport\s+gurobipy\b",
            r"\bfrom\s+gurobipy\s+import\b",
            r"\bmodel\s*=",
        )
        return any(re.search(p, raw, flags=re.IGNORECASE) for p in patterns)

    @staticmethod
    def _sanitize_code_payload(text: str) -> str:
        cleaned = (text or "").strip()
        if not cleaned:
            return cleaned

        # Unified extraction policy:
        # 1) Prefer explicit <Gurobi_code> ... </Gurobi_code> with first-close truncation.
        # 2) Fallback to fenced code block extraction.
        by_tag = FourStageMCTS._extract_tag_block(cleaned, tag="Gurobi_code", min_len=21)
        if by_tag:
            cleaned = by_tag

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



