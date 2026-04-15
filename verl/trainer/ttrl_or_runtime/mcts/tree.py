from __future__ import annotations

import json
import math
import re
import string
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from verl.trainer.ttrl_or_runtime.config import GRPOConfig, MCTSConfig
from verl.trainer.ttrl_or_runtime.mcts.node import SearchNode
from verl.trainer.ttrl_or_runtime.mcts.puct import PUCTSelector
from verl.trainer.ttrl_or_runtime.prompts import PromptBuilder
from verl.trainer.ttrl_or_runtime.prompts.refine_code import (
    CODE_ERROR_PROMPT_TEMPLATE,
    CODE_INFEASIBLE_PROMPT_TEMPLATE,
    CODE_REFINE_PROMPT_TEMPLATE,
)
from verl.trainer.ttrl_or_runtime.types import Generation, OptimizationTask, RewardBreakdown, STAGE_ORDER, Stage, Trajectory


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
        selection_history: list[tuple[tuple[str, str], str]] = []
        code_entry_attempt_total: int = 0
        code_entry_one_shot_suppression: dict[str, Any] | None = None
        # Once MCTS first attempts to enter CODE (including deferred gate),
        # dynamic reward should immediately switch to late-phase weights.
        dynamic_reward_force_stage3: bool = False

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
            active_code_suppress_node_ids: set[str] = set()
            active_code_suppress_weight = 1.0
            active_code_suppress_info: dict[str, Any] = {}
            if code_entry_one_shot_suppression is not None:
                target_iter = int(code_entry_one_shot_suppression.get("target_iter", -1))
                if int(iter_idx) == target_iter:
                    active_code_suppress_node_ids = {
                        str(x) for x in (code_entry_one_shot_suppression.get("node_ids", []) or []) if str(x).strip()
                    }
                    active_code_suppress_weight = float(
                        code_entry_one_shot_suppression.get("weight", self._code_entry_suppress_weight())
                    )
                    active_code_suppress_info = dict(code_entry_one_shot_suppression)
                elif int(iter_idx) > target_iter:
                    code_entry_one_shot_suppression = None
            blocked_repeat_threshold = self._blocked_sibling_threshold()
            stuck_state = self._blocked_sibling_state(selection_history, blocked_repeat_threshold)
            blocked_group = stuck_state["group"] if stuck_state is not None else None
            soft_block_weight = self._blocked_sibling_soft_weight()
            force_anchor_node_id = str(stuck_state.get("anchor_node_id", "")) if stuck_state is not None else ""

            force_applied = False
            selection_root = root
            if force_anchor_node_id:
                anchor_node = self._find_node_by_id(root, force_anchor_node_id)
                if anchor_node is not None and self._subtree_has_expandable_leaf(anchor_node):
                    selection_root = anchor_node
                    force_applied = True

            ranked_leaves = self._rank_expandable_leaves(
                selection_root,
                soft_block_group=blocked_group,
                soft_block_weight=soft_block_weight,
                extra_suppress_node_ids=active_code_suppress_node_ids,
                extra_suppress_weight=active_code_suppress_weight,
            )
            if not ranked_leaves and force_applied:
                selection_root = root
                force_applied = False
                ranked_leaves = self._rank_expandable_leaves(
                    selection_root,
                    soft_block_group=blocked_group,
                    soft_block_weight=soft_block_weight,
                    extra_suppress_node_ids=active_code_suppress_node_ids,
                    extra_suppress_weight=active_code_suppress_weight,
                )
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

            selected, selected_score, selection_path = self._select_leaf_recursive(
                selection_root,
                soft_block_group=blocked_group,
                soft_block_weight=soft_block_weight,
                extra_suppress_node_ids=active_code_suppress_node_ids,
                extra_suppress_weight=active_code_suppress_weight,
            )
            if code_entry_one_shot_suppression is not None and int(iter_idx) == int(
                code_entry_one_shot_suppression.get("target_iter", -1)
            ):
                code_entry_one_shot_suppression = None

            next_stage = self._next_stage(selected.stage)
            if next_stage is None:
                continue
            if next_stage == Stage.CODE:
                dynamic_reward_force_stage3 = True
            selected_group_key = self._selection_group_key(selected)
            if selected_group_key is not None:
                selection_history.append((selected_group_key, str(selected.node_id)))
            selection_sec = float(time.perf_counter() - selection_t0)
            selected_q_before_group = float(selected.q_value)
            selected_visits_before_group = int(selected.visits)

            if next_stage == Stage.CODE and self._second_code_entry_enabled():
                selected_node_id = str(selected.node_id)
                code_entry_attempt_total = int(code_entry_attempt_total) + 1
                attempt = int(code_entry_attempt_total)
                if attempt < 2:
                    cluster_suppress_node_ids, suppress_meta = self._build_global_same_cluster_node_ids(
                        selected=selected,
                        records=records,
                    )
                    path_suppress_node_ids = self._collect_path_node_ids(selected, include_root=False)
                    suppress_node_ids = set(cluster_suppress_node_ids) | set(path_suppress_node_ids)
                    suppress_weight = self._code_entry_suppress_weight()
                    code_entry_one_shot_suppression = {
                        "target_iter": int(iter_idx + 1),
                        "node_ids": sorted(suppress_node_ids),
                        "cluster_node_ids": sorted(cluster_suppress_node_ids),
                        "path_node_ids": sorted(path_suppress_node_ids),
                        "weight": float(suppress_weight),
                        "source_node_id": selected_node_id,
                        "source_stage": selected.stage.value if selected.stage else "<ROOT>",
                        "source_attempt": int(attempt),
                        "scope": "cluster_plus_first_code_path_once",
                        "cluster_debug": dict(suppress_meta),
                    }
                    iter_payload = {
                        "iter": int(iter_idx),
                        "stage": "obj-con-code-gate",
                        "selection": {
                            "selected_parent": {
                                "node_id": selected.node_id,
                                "stage": selected.stage.value if selected.stage else "<ROOT>",
                                "puct_score": float(selected_score),
                                "value_before_group": float(selected_q_before_group),
                                "visits_before_group": int(selected_visits_before_group),
                                "content": selected.text,
                            },
                            "selection_path": selection_path,
                        },
                        "code_entry_gate": {
                            "enabled": True,
                            "second_attempt_required": True,
                            "decision": "defer_first_attempt",
                            "attempt": int(attempt),
                            "attempt_scope": "global_per_sample",
                            "target_node_id": selected_node_id,
                            "one_shot_suppression": dict(code_entry_one_shot_suppression),
                            "active_suppression_this_iter": dict(active_code_suppress_info),
                        },
                        "timing": {
                            "mcts_selection_sec": float(time.perf_counter() - selection_t0),
                            "iteration_total_sec": float(time.perf_counter() - iter_t0),
                        },
                    }
                    iteration_logs.append(iter_payload)
                    if iteration_callback is not None:
                        iteration_callback(iter_payload)
                    continue

            prompt_t0 = time.perf_counter()
            selected_traj = None if selected.stage is None else selected.to_partial_trajectory()
            base_prompt_messages = self.prompt_builder.build_messages(task, next_stage, selected_traj, prompt_kind="stage")
            prompt_messages = self.prompt_builder.build_messages(task, next_stage, selected_traj, prompt_kind="rollout")
            base_prompt = self.prompt_builder.build(task, next_stage, selected_traj)
            prompt = self.prompt_builder.build_rollout(task, next_stage, selected_traj)
            if next_stage == Stage.CODE and bool(getattr(self.config, "code_refine", True)):
                code_refine_prompt = self._build_code_refine_rollout_prompt(
                    task=task,
                    selected=selected,
                    records=records,
                )
                if code_refine_prompt.strip():
                    base_prompt_messages = [{"role": "user", "content": code_refine_prompt}]
                    prompt_messages = [{"role": "user", "content": code_refine_prompt}]
                    base_prompt = code_refine_prompt
                    prompt = code_refine_prompt
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
                )

                return {
                    "rollout_index": ridx,
                    "callback_started": callback_t0,
                    "child": child,
                    "trajectory": None,
                    "completion_full": completion_text,
                    "answer_current_stage": parsed_text,
                    "answer_rollout_suffix": "",
                    "parse_sec": parse_sec,
                    "complete_sec": 0.0,
                }

            def _complete_prepared_group(prepared_items: list[dict[str, Any]]) -> None:
                if not prepared_items:
                    return

                complete_t0 = time.perf_counter()
                average_complete_sec = 0.0

                if self.split_rollout_completion and next_stage != Stage.CODE:
                    prompts: list[str] = []
                    prompt_indices: list[int] = []
                    for idx, item in enumerate(prepared_items):
                        child = item["child"]
                        partial = child.to_partial_trajectory()
                        partial.trajectory_id = str(uuid.uuid4())
                        completion_prompt = self.prompt_builder.build_completion(task, child.stage, partial)
                        item["split_completion_prompt"] = completion_prompt
                        if completion_prompt.strip():
                            prompts.append(completion_prompt)
                            prompt_indices.append(idx)

                    completion_texts: list[str | None] = []
                    if prompts:
                        batch_method = getattr(self.backend, "generate_auxiliary_texts", None)
                        if callable(batch_method):
                            completion_texts = list(
                                batch_method(
                                    prompts,
                                    max_new_tokens=int(getattr(self.backend, "max_new_tokens", 2048) or 2048),
                                    temperature=float(getattr(self.backend, "temperature", 0.0) or 0.0),
                                    top_p=float(getattr(self.backend, "top_p", 1.0) or 1.0),
                                    prefer_vllm=bool(self._active_grpo_config.use_vllm) if self._active_grpo_config is not None else False,
                                    vllm_mode=str(self._active_grpo_config.vllm_mode) if self._active_grpo_config is not None else "",
                                )
                            )
                        else:
                            completion_texts = []

                    mapped_texts: list[str | None] = [None for _ in prepared_items]
                    for local_idx, prepared_idx in enumerate(prompt_indices):
                        mapped_texts[prepared_idx] = completion_texts[local_idx] if local_idx < len(completion_texts) else None

                    total_complete_sec = float(time.perf_counter() - complete_t0)
                    average_complete_sec = total_complete_sec / max(1, len(prepared_items))

                    for idx, item in enumerate(prepared_items):
                        child = item["child"]
                        completion_text = str(mapped_texts[idx] or "")
                        item["answer_rollout_suffix"] = completion_text
                        item["complete_sec"] = average_complete_sec
                        if completion_text.strip():
                            item["trajectory"] = self._trajectory_from_rollout_completion(child, completion_text)
                        else:
                            item["trajectory"] = self.rollout_to_code(task, child)
                    return

                for item in prepared_items:
                    child = item["child"]
                    completion_text = str(item.get("completion_full", ""))
                    item["trajectory"] = self._complete_for_reward_from_rollout(task, child, completion_text)
                total_complete_sec = float(time.perf_counter() - complete_t0)
                average_complete_sec = total_complete_sec / max(1, len(prepared_items))
                for item in prepared_items:
                    item["complete_sec"] = average_complete_sec

            def _score_prepared_group(prepared_items: list[dict[str, Any]]) -> list[float]:
                if not prepared_items:
                    return []

                _complete_prepared_group(prepared_items)

                reward_t0 = time.perf_counter()
                trajectories = [item["trajectory"] for item in prepared_items]
                # Attach per-iteration dynamic reward context for reward calculator.
                iter_num = int(iter_idx) + 1
                for t in trajectories:
                    if t is None:
                        continue
                    meta = t.metadata if isinstance(t.metadata, dict) else {}
                    meta["__mcts_iter__"] = int(iter_num)
                    meta["__dynamic_reward_force_stage3__"] = bool(dynamic_reward_force_stage3)
                    meta["__mcts_stage__"] = str(next_stage.value)
                    t.metadata = meta
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
            if next_stage == Stage.CODE:
                code_k = max(1, int(grpo_config.num_generations))
                generations = list(
                    self._backend_generate(
                        next_stage,
                        prompt_messages or prompt,
                        code_k,
                        no_lora_adapter=True,
                    )
                )
                completion_texts = [str(gen.text or "") for gen in generations]
                if completion_texts:
                    rewards = list(_batch_reward_callback(prompt, completion_texts))
                else:
                    rewards = []
                for ridx, gen in enumerate(generations):
                    reward_total = float(rewards[ridx]) if ridx < len(rewards) else 0.0
                    gen.metadata["reward_total"] = reward_total
                    gen.metadata["rollout_index"] = int(ridx)
                grpo_report = {
                    "updated": False,
                    "backend": type(self.backend).__name__,
                    "stage": next_stage.value,
                    "num_samples": len(generations),
                    "reason": "code_stage_generate_only_no_grpo_update",
                }
            else:
                generations, grpo_report = self._run_internal_grpo_rollout(
                    stage=next_stage,
                    prompt=prompt_messages or prompt,
                    grpo_config=grpo_config,
                    reward_callback=_reward_callback,
                )
            rollout_group_wall_sec = float(time.perf_counter() - rollout_group_t0)

            # Callback timing is measured per rollout item. Since reward callbacks are executed in parallel,
            # wall time should be approximated by max(item_time) instead of sum(item_time).
            callback_total_sum_sec = float(sum(t.get("callback_total_sec", 0.0) for t in callback_timings))
            callback_total_sec = float(max([0.0] + [float(t.get("callback_total_sec", 0.0) or 0.0) for t in callback_timings]))
            callback_exec_sec = float(sum(t.get("code_execution_sec", 0.0) for t in callback_timings))
            callback_complete_sec = float(sum(t.get("rollout_to_code_sec", 0.0) for t in callback_timings))
            callback_reward_sec = float(sum(t.get("reward_compute_sec", 0.0) for t in callback_timings))
            callback_parse_sec = float(sum(t.get("completion_parse_sec", 0.0) for t in callback_timings))
            model_sampling_update_sec = float(max(0.0, rollout_group_wall_sec - callback_total_sec))

            if not group_rollouts:
                if next_stage == Stage.CODE:
                    iter_payload = {
                        "iter": int(iter_idx),
                        "stage": next_stage.value,
                        "timing": {
                            "mcts_selection_sec": selection_sec,
                            "iteration_total_sec": float(time.perf_counter() - iter_t0),
                        },
                        "code_terminal": {
                            "enabled": False,
                            "reason": "empty_code_rollout_group",
                            "fallback_to_original_logic": True,
                        },
                    }
                    iteration_logs.append(iter_payload)
                    if iteration_callback is not None:
                        iteration_callback(iter_payload)
                    # CODE produced no rollout candidates: do not early-stop.
                    # Continue search so sibling branches can still be explored.
                    continue
                continue

            resolved_priors, prior_source = self._resolve_child_priors(
                stage=next_stage,
                prompt=base_prompt_messages or base_prompt,
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
            reward_one_candidates: list[dict[str, Any]] = []
            rollout_summaries: list[dict[str, Any]] = []
            processed_group_rollouts: list[dict[str, Any]] = []
            pending_record_inputs: list[dict[str, Any]] = []
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
                child.update(reward_total)
                child_backprop_sec = float(time.perf_counter() - backprop_t0)
                backprop_total_sec += child_backprop_sec

                rollout_timing = dict(rollout.get("timing", {}))
                rollout_timing["child_backprop_sec"] = child_backprop_sec
                rollout["timing"] = rollout_timing

                current_hit_reward_one = bool(self.config.stop_on_reward_one and reward_total >= 0.9)

                pending_record_inputs.append(
                    {
                        "rollout_index": ridx,
                        "rollout": rollout,
                        "child": child,
                        "trajectory": completed,
                        "reward_obj": reward_obj,
                        "reward_total": reward_total,
                        "parent_q_before": parent_q_before,
                        "parent_visits_before": parent_visits_before,
                        "child_q_before": child_q_before,
                        "child_visits_before": child_visits_before,
                        "hit_reward_one": current_hit_reward_one,
                    }
                )

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
                        "effective_success": bool(((reward_obj.metadata or {}).get("execution", {}) or {}).get("effective_success", False)),
                        "execution": ((reward_obj.metadata or {}).get("execution", {}) or {}),
                        "r1_debug": ((reward_obj.metadata or {}).get("r1_debug", {}) or {}),
                        "r3_debug": ((reward_obj.metadata or {}).get("r3", {}) or {}),
                        "r4_debug": ((reward_obj.metadata or {}).get("r4_debug", {}) or {}),
                        "timing": rollout_timing,
                        "prior": child.prior,
                        "completion_preview": child.text[:240],
                    }
                )

                if reward_total > best_reward:
                    best_reward = reward_total
                    best_trajectory = completed

                if current_hit_reward_one:
                    reward_one_candidates.append(
                        {
                            "node_id": child.node_id,
                            "trajectory_id": completed.trajectory_id,
                            "reward_total": reward_total,
                            "prior": float(child.prior),
                        }
                    )

            if not processed_group_rollouts:
                continue

            if reward_one_candidates:
                chosen_reward_one = max(
                    reward_one_candidates,
                    key=lambda item: (float(item.get("reward_total", 0.0)), float(item.get("prior", 0.0))),
                )
                hit_reward_one = True
                reward_one_payload = {
                    "reason": "reward_one",
                    "iteration": iter_idx,
                    "stage": next_stage.value,
                    "node_id": str(chosen_reward_one.get("node_id", "")),
                    "trajectory_id": str(chosen_reward_one.get("trajectory_id", "")),
                    "reward_total": float(chosen_reward_one.get("reward_total", 0.0)),
                    "threshold": 0.9,
                    "tie_breaker": "prior_when_reward_equal",
                }

            group_reward_mean = float(
                sum(float(rollout.get("reward_total", 0.0)) for rollout in processed_group_rollouts)
                / max(1, len(processed_group_rollouts))
            )
            group_backprop_t0 = time.perf_counter()
            self._backpropagate(selected, group_reward_mean)
            group_backprop_sec = float(time.perf_counter() - group_backprop_t0)
            backprop_total_sec += group_backprop_sec
            shared_group_backprop_sec = group_backprop_sec / max(1, len(processed_group_rollouts))

            cluster_lineage_update: dict[str, Any] = {
                "enabled": bool(self._mcts_cluster_update_enabled()),
                "updated": 0,
                "levels": [],
                "node_ids": [],
                "sec": 0.0,
            }
            if self._mcts_cluster_update_enabled():
                cluster_lineage_update = self._propagate_cluster_lineage_from_selected(
                    selected=selected,
                    records=records,
                    reward=group_reward_mean,
                )
                backprop_total_sec += float(cluster_lineage_update.get("sec", 0.0))
            shared_cluster_backprop_sec = float(cluster_lineage_update.get("sec", 0.0)) / max(1, len(processed_group_rollouts))

            for pending in pending_record_inputs:
                ridx = int(pending["rollout_index"])
                rollout = pending["rollout"]
                child = pending["child"]
                completed = pending["trajectory"]
                reward_obj = pending["reward_obj"]
                reward_total = float(pending["reward_total"])
                rollout_timing = dict(rollout.get("timing", {}))
                rollout_timing["group_backprop_sec"] = float(shared_group_backprop_sec)
                rollout_timing["cluster_linked_backprop_sec"] = float(shared_cluster_backprop_sec)
                rollout_timing["group_reward_mean"] = float(group_reward_mean)
                rollout["timing"] = rollout_timing
                completed.metadata["iter"] = int(iter_idx)
                completed.metadata["stage"] = next_stage.value
                completed.metadata["group_id"] = group_id
                completed.metadata["parent_node"] = {
                    "node_id": selected.node_id,
                    "stage": (selected.stage.value if selected.stage else "<ROOT>"),
                    "value_before_group": float(selected_q_before_group),
                    "visits_before_group": int(selected_visits_before_group),
                    "value_after_group": float(selected.q_value),
                    "visits_after_group": int(selected.visits),
                    "group_reward_mean": float(group_reward_mean),
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
                        hit_reward_one=bool(pending["hit_reward_one"]),
                        child_q_before=float(pending["child_q_before"]),
                        child_visits_before=int(pending["child_visits_before"]),
                        child_q_after=child.q_value,
                        child_visits_after=child.visits,
                        parent_q_before=float(pending["parent_q_before"]),
                        parent_visits_before=int(pending["parent_visits_before"]),
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

                rollout["update"] = {
                    "prior": float(child.prior),
                    "child_q_before": float(pending["child_q_before"]),
                    "child_q_after": float(child.q_value),
                    "child_visits_before": int(pending["child_visits_before"]),
                    "child_visits_after": int(child.visits),
                    "parent_q_before": float(pending["parent_q_before"]),
                    "parent_q_after": float(selected.q_value),
                    "parent_visits_before": int(pending["parent_visits_before"]),
                    "parent_visits_after": int(selected.visits),
                    "child_backprop_sec": float(rollout_timing.get("child_backprop_sec", 0.0)),
                    "group_backprop_sec": float(shared_group_backprop_sec),
                    "cluster_linked_backprop_sec": float(shared_cluster_backprop_sec),
                    "group_reward_mean": float(group_reward_mean),
                }
                if 0 <= ridx < len(rollout_summaries):
                    rollout_summaries[ridx]["timing"] = rollout_timing
                    rollout_summaries[ridx]["group_reward_mean"] = float(group_reward_mean)
                    rollout_summaries[ridx]["parent_visits_after_group"] = int(selected.visits)
                    rollout_summaries[ridx]["parent_value_after_group"] = float(selected.q_value)
            best_rollout = max(
                processed_group_rollouts,
                key=lambda x: (
                    float(x.get("reward_total", float("-inf"))),
                    float((x.get("child").prior if x.get("child") is not None else 0.0)),
                ),
            )
            best_rollout_obj = best_rollout["reward_obj"]
            best_rollout_child = best_rollout["child"]
            best_rollout_traj = best_rollout["trajectory"]
            best_rollout_timing = dict(best_rollout.get("timing", {}))
            best_rollout_update = dict(best_rollout.get("update", {}))

            grpo_timing = (
                dict((grpo_report or {}).get("timing", {}))
                if isinstance((grpo_report or {}).get("timing", {}), dict)
                else {}
            )
            rollout_vllm_infer_sec = float(grpo_timing.get("rollout_vllm_infer_sec", model_sampling_update_sec) or 0.0)
            forward_compute_sec = float(grpo_timing.get("old_log_prob_forward_sec", 0.0) or 0.0)
            grpo_update_sec = float(grpo_timing.get("actor_update_sec", 0.0) or 0.0)
            grpo_group_total_sec = float(grpo_timing.get("grpo_group_total_sec", 0.0) or 0.0)

            # Backward-compatible alias: prefer explicit train_runtime if present,
            # otherwise use actor update time as the GRPO update time proxy.
            grpo_train_runtime_sec = float((grpo_report or {}).get("train_runtime", 0.0) or 0.0)
            if grpo_train_runtime_sec <= 0.0:
                grpo_train_runtime_sec = float(grpo_update_sec)

            # Reward calculation time includes rollout->code completion/parsing,
            # code execution and reward computation callback path.
            reward_calculation_total_sec = float(callback_total_sec)
            iter_total_sec = float(time.perf_counter() - iter_t0)

            iter_payload = {
                "iter": int(iter_idx),
                "stage": next_stage.value,
                "selection": {
                    "blocked_sibling_group": {
                        "parent_node_id": (blocked_group[0] if blocked_group else ""),
                        "stage": (blocked_group[1] if blocked_group else ""),
                        "applied": bool(blocked_group is not None),
                        "threshold": int(blocked_repeat_threshold),
                        "mode": "soft_penalty",
                        "soft_weight": float(soft_block_weight),
                        "forced_anchor_node_id": force_anchor_node_id,
                        "force_into_anchor_subtree_applied": bool(force_applied),
                    },
                    "selected_parent": {
                        "node_id": selected.node_id,
                        "stage": selected.stage.value if selected.stage else "<ROOT>",
                        "value_before_group": float(selected_q_before_group),
                        "visits_before_group": int(selected_visits_before_group),
                        "value_after_group": float(selected.q_value),
                        "visits_after_group": int(selected.visits),
                        "puct_score": float(selected_score),
                        "content": selected.text,
                    },
                    "selection_path": selection_path,
                    "code_entry_suppression_active": dict(active_code_suppress_info),
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
                        "value_before_group": float(selected_q_before_group),
                        "visits_before_group": int(selected_visits_before_group),
                        "value_after_group": float(selected.q_value),
                        "visits_after_group": int(selected.visits),
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
                    "reward": {
                        "r1": float(best_rollout_obj.r1),
                        "r2": float(best_rollout_obj.r2),
                        "r3": float(best_rollout_obj.r3),
                        "r4": float(best_rollout_obj.r4),
                        "total": float(best_rollout_obj.total),
                        "obj_answer": (best_rollout_obj.metadata or {}).get("obj_answer"),
                        "r1_debug": (best_rollout_obj.metadata or {}).get("r1_debug", {}),
                        "r3_debug": (best_rollout_obj.metadata or {}).get("r3", {}),
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
                    # Canonical timing fields (explicit names)
                    "mcts_selection_sec": selection_sec,
                    "vllm_infer_generation_sec": rollout_vllm_infer_sec,
                    "reward_calculation_total_sec": reward_calculation_total_sec,
                    "grpo_update_sec": grpo_update_sec,
                    "forward_compute_sec": forward_compute_sec,

                    # Backward-compatible / detailed timing fields
                    "selection_sec": selection_sec,
                    "prompt_build_sec": prompt_build_sec,
                    "rollout_group_wall_sec": rollout_group_wall_sec,
                    "rollout_vllm_infer_sec": rollout_vllm_infer_sec,
                    "grpo_train_runtime_sec": grpo_train_runtime_sec,
                    "grpo_group_total_sec": grpo_group_total_sec,
                    "reward_callback_total_sec": callback_total_sec,
                    "reward_callback_total_sum_sec": callback_total_sum_sec,
                    "old_log_prob_forward_sec": forward_compute_sec,
                    "actor_update_sec": grpo_update_sec,
                    "completion_parse_total_sec": callback_parse_sec,
                    "rollout_to_code_total_sec": callback_complete_sec,
                    "reward_compute_total_sec": callback_reward_sec,
                    "code_execution_total_sec": callback_exec_sec,
                    "model_sampling_update_excluding_reward_callback_sec": model_sampling_update_sec,
                    "backprop_total_sec": backprop_total_sec,
                    "cluster_linked_backprop_total_sec": float(cluster_lineage_update.get("sec", 0.0)),
                    "iteration_total_sec": iter_total_sec,
                },
                "cluster_lineage_update": cluster_lineage_update,
                "grpo_update": dict(grpo_report),
            }
            iteration_logs.append(iter_payload)

            if next_stage == Stage.CODE:
                terminal_payload = self._run_code_terminal_refine(
                    task=task,
                    selected=selected,
                    records=records,
                    code_rollouts=processed_group_rollouts,
                    stage_archive=stage_archive,
                    current_iter=iter_idx,
                )
                iter_payload["code_terminal"] = dict(terminal_payload)
                if iteration_callback is not None:
                    iteration_callback(iter_payload)
                fallback_to_original = bool(terminal_payload.get("fallback_to_original_logic", False))
                if not fallback_to_original:
                    final_tid = str(terminal_payload.get("trajectory_id", ""))
                    final_reward = terminal_payload.get("reward_total")
                    stop_info = {
                        "reason": "expanded_to_code",
                        "iteration": iter_idx,
                        "stage": next_stage.value,
                        "node_id": selected.node_id,
                        "trajectory_id": final_tid,
                        "reward_total": final_reward,
                        "code_terminal": dict(terminal_payload),
                    }
                    return self._finalize_result(
                        root=root,
                        records=records,
                        stop_info=stop_info,
                        iteration_logs=iteration_logs,
                    )
                # CODE(plus optional repair) still has no valid obj: continue MCTS search.
                # This lets sibling branches keep exploring until max_iterations or a CODE-success appears.
                continue

            if iteration_callback is not None:
                iteration_callback(iter_payload)

            recent_consensus_stop = self._check_recent_obj_consensus(records=records, current_iter=iter_idx)
            if recent_consensus_stop is not None:
                stop_info = {
                    "reason": "recent_obj_consensus",
                    "iteration": iter_idx,
                    "stage": next_stage.value,
                    "node_id": selected.node_id,
                    "trajectory_id": str(recent_consensus_stop.get("trajectory_id", "")),
                    "reward_total": recent_consensus_stop.get("reward_total"),
                    "obj_leader": recent_consensus_stop.get("obj_leader"),
                    "count": recent_consensus_stop.get("count"),
                    "window_rollouts": recent_consensus_stop.get("window_rollouts"),
                    "ratio": recent_consensus_stop.get("ratio"),
                    "selected_iteration": recent_consensus_stop.get("selected_iteration"),
                    "obj_scale_mode": "expanded",
                    "obj_scale_expand_ratio": recent_consensus_stop.get("obj_scale_expand_ratio"),
                    "recent_top_obj_gate": recent_consensus_stop.get("recent_top_obj_gate", {}),
                    "tie_breaker": "prior_when_reward_equal",
                }
                return self._finalize_result(
                    root=root,
                    records=records,
                    stop_info=stop_info,
                    iteration_logs=iteration_logs,
                )

            if hit_reward_one:
                stop_info = reward_one_payload
                return self._finalize_result(
                    root=root,
                    records=records,
                    stop_info=stop_info,
                    iteration_logs=iteration_logs,
                )

        return self._finalize_result(
            root=root,
            records=records,
            stop_info=stop_info,
            iteration_logs=iteration_logs,
        )

    def _run_internal_grpo_rollout(
        self,
        stage: Stage,
        prompt: Any,
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

    def _backend_generate(
        self,
        stage: Stage,
        prompt: Any,
        n: int,
        *,
        no_lora_adapter: bool = False,
    ) -> list[Generation]:
        method = getattr(self.backend, "generate")
        try:
            return list(method(stage=stage, prompt=prompt, n=n, no_lora_adapter=bool(no_lora_adapter)))
        except TypeError:
            try:
                return list(method(stage, prompt, n, no_lora_adapter=bool(no_lora_adapter)))
            except TypeError:
                return list(method(stage, prompt, n))

    @staticmethod
    def _split_rollout_completion(text: str) -> tuple[str, str]:
        cleaned = (text or "").strip()
        return cleaned, cleaned

    @staticmethod
    def _tags_for_stage(stage: Stage) -> list[str]:
        if stage == Stage.SCHEMA:
            return ["Type", "Sets"]
        if stage == Stage.TYPE_HINT:
            return ["Type"]
        if stage == Stage.SET_PARAM_VAR:
            return ["Parameters", "Variables"]
        if stage == Stage.OBJ_CONS:
            return ["Objective", "Constraints"]
        if stage == Stage.SETS:
            return ["Sets"]
        if stage == Stage.PARAMETERS:
            return ["Parameters"]
        if stage == Stage.VARIABLES:
            return ["Variables"]
        if stage == Stage.OBJECTIVE:
            return ["Objective"]
        if stage == Stage.CONSTRAINTS:
            return ["Constraints"]
        return ["python"]

    @staticmethod
    def _extract_tag_block(text: str, tag: str, min_len: int = 0) -> str:
        """Extract content from the FIRST valid closed tag pair.

        Supports two formats (in priority order):
        1. <tag>...</tag>  (angle brackets, preferred)
        2. [tag]...[/tag]  (square brackets, fallback)
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
        return FourStageMCTS._extract_stage_payload(stage, rollout_text)

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
            parsed = self._extract_stage_payload(next_stage, full_text)
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
            prompt_messages = self.prompt_builder.build_messages(task, next_stage, partial, prompt_kind="stage")
            prompt = self.prompt_builder.build(task, next_stage, partial)
            generation = self.backend.generate(next_stage, prompt_messages or prompt, 1)[0]
            partial.outputs[next_stage] = self._extract_stage_payload(next_stage, generation.text)
            partial.priors[next_stage] = generation.prior

        return partial

    def _resolve_child_priors(
        self,
        stage: Stage,
        prompt: Any,
        rollouts: list[dict[str, Any]],
        fallback_generations: list[Generation],
    ) -> tuple[list[float], str]:
        if not bool(getattr(self.config, 'enable_prior', True)):
            uniform = [1.0 / float(max(1, len(rollouts)))] * len(rollouts)
            return uniform, 'uniform_disabled'

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

    def _select_leaf_recursive(
        self,
        root: SearchNode,
        soft_block_group: tuple[str, str] | None = None,
        soft_block_weight: float = 0.6,
        extra_suppress_node_ids: set[str] | None = None,
        extra_suppress_weight: float = 1.0,
    ) -> tuple[SearchNode, float, list[dict[str, Any]]]:
        cur = root
        selection_path: list[dict[str, Any]] = []
        suppress_ids = {str(x) for x in (extra_suppress_node_ids or set()) if str(x).strip()}
        suppress_weight = float(max(0.0, min(1.0, extra_suppress_weight)))

        while cur.children:
            candidate_scores: list[tuple[SearchNode, float]] = []
            for child in cur.children:
                if not self._subtree_has_expandable_leaf(child):
                    continue
                score = float(self.selector.score(cur, child))
                if self._node_matches_selection_group(child, soft_block_group):
                    score *= float(max(0.0, soft_block_weight))
                suppress_applied = False
                if suppress_ids and self._node_is_under_node_ids(child, suppress_ids):
                    score *= suppress_weight
                    suppress_applied = True
                candidate_scores.append((child, score))

            if not candidate_scores:
                break

            candidate_scores.sort(key=lambda item: item[1], reverse=True)
            chosen, chosen_score = candidate_scores[0]
            selection_path.append(
                {
                    'parent_node_id': cur.node_id,
                    'parent_stage': cur.stage.value if cur.stage else '<ROOT>',
                    'soft_block_group': {
                        'parent_node_id': (soft_block_group[0] if soft_block_group else ''),
                        'stage': (soft_block_group[1] if soft_block_group else ''),
                        'soft_weight': float(max(0.0, soft_block_weight)),
                    },
                    'one_shot_suppression': {
                        'enabled': bool(suppress_ids),
                        'node_count': int(len(suppress_ids)),
                        'weight': float(suppress_weight),
                    },
                    'candidates': [
                        {
                            'node_id': node.node_id,
                            'stage': node.stage.value if node.stage else '<ROOT>',
                            'puct_score': float(score),
                            'value': float(node.q_value),
                            'visits': int(node.visits),
                            'prior': float(node.prior),
                            'soft_block_applied': bool(self._node_matches_selection_group(node, soft_block_group)),
                            'one_shot_suppression_applied': bool(
                                suppress_ids and self._node_is_under_node_ids(node, suppress_ids)
                            ),
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

    def _rank_expandable_leaves(
        self,
        root: SearchNode,
        soft_block_group: tuple[str, str] | None = None,
        soft_block_weight: float = 0.6,
        extra_suppress_node_ids: set[str] | None = None,
        extra_suppress_weight: float = 1.0,
    ) -> list[tuple[SearchNode, float]]:
        leaves = [
            node
            for node in self._iter_leaves(root)
            if self._next_stage(node.stage) is not None
        ]
        ranked = self._rank_leaves(leaves)
        if soft_block_group is None and not extra_suppress_node_ids:
            return ranked
        out: list[tuple[SearchNode, float]] = []
        weight = float(max(0.0, soft_block_weight))
        suppress_ids = {str(x) for x in (extra_suppress_node_ids or set()) if str(x).strip()}
        suppress_weight = float(max(0.0, min(1.0, extra_suppress_weight)))
        for node, score in ranked:
            adjusted = float(score)
            if self._leaf_is_under_soft_block_group(node, soft_block_group):
                adjusted *= weight
            if suppress_ids and self._node_is_under_node_ids(node, suppress_ids):
                adjusted *= suppress_weight
            out.append((node, adjusted))
        out.sort(key=lambda item: item[1], reverse=True)
        return out

    def _subtree_has_expandable_leaf(
        self,
        node: SearchNode,
    ) -> bool:
        if not node.children:
            return self._next_stage(node.stage) is not None
        return any(self._subtree_has_expandable_leaf(child) for child in node.children)

    @staticmethod
    def _selection_group_key(node: SearchNode) -> tuple[str, str] | None:
        if node.parent is None or node.stage is None:
            return None
        return (str(node.parent.node_id), str(node.stage.value))

    @staticmethod
    def _node_matches_selection_group(
        node: SearchNode,
        blocked_group: tuple[str, str] | None,
    ) -> bool:
        if blocked_group is None:
            return False
        group_key = FourStageMCTS._selection_group_key(node)
        return group_key == blocked_group

    @staticmethod
    def _leaf_is_under_soft_block_group(
        node: SearchNode,
        soft_block_group: tuple[str, str] | None,
    ) -> bool:
        cur: SearchNode | None = node
        while cur is not None:
            if FourStageMCTS._node_matches_selection_group(cur, soft_block_group):
                return True
            cur = cur.parent
        return False

    @staticmethod
    def _node_is_under_node_ids(
        node: SearchNode,
        node_ids: set[str],
    ) -> bool:
        if not node_ids:
            return False
        cur: SearchNode | None = node
        while cur is not None:
            if str(cur.node_id) in node_ids:
                return True
            cur = cur.parent
        return False

    def _blocked_sibling_threshold(self) -> int:
        group_k = max(2, int(getattr(self._active_grpo_config, 'num_generations', 0) or 0)) if self._active_grpo_config is not None else 1
        return max(2, group_k // 2)

    @staticmethod
    def _blocked_sibling_state(
        selection_history: list[tuple[tuple[str, str], str]],
        threshold: int,
    ) -> dict[str, Any] | None:
        if threshold <= 0 or len(selection_history) < threshold:
            return None
        last_group = selection_history[-1][0]
        run_len = 0
        for group_key, _ in reversed(selection_history):
            if group_key != last_group:
                break
            run_len += 1
        if run_len < threshold:
            return None

        anchor_index = max(0, int(len(selection_history) - run_len))
        anchor_node_id = str(selection_history[anchor_index][1]) if selection_history[anchor_index][1] else ""
        return {
            "group": last_group,
            "run_len": int(run_len),
            "anchor_node_id": anchor_node_id,
        }

    def _blocked_sibling_soft_weight(self) -> float:
        raw = getattr(self.config, "blocked_sibling_soft_weight", 0.6)
        try:
            return max(0.0, min(1.0, float(raw)))
        except Exception:
            return 0.6

    def _second_code_entry_enabled(self) -> bool:
        raw = getattr(self.config, "code_entry_second_attempt", True)
        try:
            return bool(raw)
        except Exception:
            return True

    def _code_entry_suppress_weight(self) -> float:
        raw = getattr(self.config, "code_entry_same_cluster_suppress_weight", 0.7)
        try:
            return max(0.0, min(1.0, float(raw)))
        except Exception:
            return 0.7

    def _build_global_same_cluster_node_ids(
        self,
        *,
        selected: SearchNode,
        records: list[StageExpansionRecord],
    ) -> tuple[set[str], dict[str, Any]]:
        selected_node_id = str(selected.node_id)
        latest_by_node: dict[str, StageExpansionRecord] = {}
        for rec in records:
            node_id = str(rec.node_id)
            prev = latest_by_node.get(node_id)
            if prev is None or int(rec.iteration) > int(prev.iteration):
                latest_by_node[node_id] = rec
            elif int(rec.iteration) == int(prev.iteration):
                if float(self._record_reward_total(rec)) > float(self._record_reward_total(prev)):
                    latest_by_node[node_id] = rec

        selected_rec = latest_by_node.get(selected_node_id)
        selected_obj = self._record_obj_answer(selected_rec) if selected_rec is not None else None
        if selected_obj is None:
            return {
                selected_node_id,
            }, {
                "mode": "fallback_selected_only_no_obj",
                "selected_obj": None,
                "matched": 1,
                "considered_nodes": int(len(latest_by_node)),
            }

        node_ids: set[str] = {selected_node_id}
        for node_id, rec in latest_by_node.items():
            obj = self._record_obj_answer(rec)
            if obj is None:
                continue
            if self._within_rel_tol(float(obj), float(selected_obj)):
                node_ids.add(str(node_id))

        return node_ids, {
            "mode": "global_cross_stage_obj_cluster",
            "selected_obj": float(selected_obj),
            "matched": int(len(node_ids)),
            "considered_nodes": int(len(latest_by_node)),
        }

    @staticmethod
    def _find_node_by_id(root: SearchNode, node_id: str) -> SearchNode | None:
        target = str(node_id or "").strip()
        if not target:
            return None
        stack: list[SearchNode] = [root]
        while stack:
            cur = stack.pop()
            if str(cur.node_id) == target:
                return cur
            stack.extend(cur.children)
        return None

    @staticmethod
    def _collect_path_node_ids(node: SearchNode, *, include_root: bool = False) -> set[str]:
        out: set[str] = set()
        cur: SearchNode | None = node
        while cur is not None:
            if include_root or cur.parent is not None:
                out.add(str(cur.node_id))
            cur = cur.parent
        return out

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

    def _mcts_cluster_update_enabled(self) -> bool:
        # Canonical key first; legacy typo key is fallback-only compatibility.
        canonical = getattr(self.config, "mcts_cluster_update", None)
        if canonical is not None:
            return bool(canonical)
        legacy = getattr(self.config, "mcts_cluster_updata", None)
        if legacy is not None:
            return bool(legacy)
        return True

    def _latest_record_for_node(
        self,
        records: list[StageExpansionRecord],
        *,
        node_id: str,
        stage: Stage,
    ) -> StageExpansionRecord | None:
        target = str(node_id or "")
        chosen: StageExpansionRecord | None = None
        for rec in records:
            if rec.stage != stage:
                continue
            if str(rec.node_id) != target:
                continue
            if chosen is None:
                chosen = rec
                continue
            if int(rec.iteration) > int(chosen.iteration):
                chosen = rec
            elif int(rec.iteration) == int(chosen.iteration):
                if float(self._record_reward_total(rec)) > float(self._record_reward_total(chosen)):
                    chosen = rec
        return chosen

    def _propagate_cluster_lineage_from_selected(
        self,
        *,
        selected: SearchNode,
        records: list[StageExpansionRecord],
        reward: float,
    ) -> dict[str, Any]:
        coef_raw = getattr(self.config, "cluster_propagate_coef", 0.6)
        decay_raw = getattr(self.config, "cluster_propagate_ancestor_decay", 0.8)
        try:
            cluster_coef = max(0.0, float(coef_raw))
        except Exception:
            cluster_coef = 0.6
        try:
            ancestor_decay = max(0.0, min(1.0, float(decay_raw)))
        except Exception:
            ancestor_decay = 0.8

        t0 = time.perf_counter()
        total_updated = 0
        touched_ids: list[str] = []
        levels: list[dict[str, Any]] = []
        seen_ids: set[str] = set()

        cursor: SearchNode | None = selected
        level = 0
        while cursor is not None and cursor.parent is not None:
            if cursor.stage is None:
                cursor = cursor.parent
                level += 1
                continue

            rec = self._latest_record_for_node(records, node_id=str(cursor.node_id), stage=cursor.stage)
            if rec is None:
                cursor = cursor.parent
                level += 1
                continue
            cursor_obj = self._record_obj_answer(rec)
            if cursor_obj is None:
                cursor = cursor.parent
                level += 1
                continue

            level_weight = float(cluster_coef * (ancestor_decay ** max(0, int(level))))
            level_reward = float(reward) * float(level_weight)

            peer_ids: list[str] = []
            updated_this_level = 0
            for sibling in cursor.parent.children:
                sibling_id = str(sibling.node_id)
                if sibling_id == str(cursor.node_id):
                    continue
                if sibling.stage != cursor.stage:
                    continue
                sibling_rec = self._latest_record_for_node(records, node_id=sibling_id, stage=sibling.stage)
                if sibling_rec is None:
                    continue
                sibling_obj = self._record_obj_answer(sibling_rec)
                if sibling_obj is None:
                    continue
                if not self._within_rel_tol(float(cursor_obj), float(sibling_obj)):
                    continue
                peer_ids.append(sibling_id)
                if sibling_id in seen_ids:
                    continue
                sibling.update(level_reward)
                seen_ids.add(sibling_id)
                touched_ids.append(sibling_id)
                total_updated += 1
                updated_this_level += 1

            levels.append(
                {
                    "target_node_id": str(cursor.node_id),
                    "stage": str(cursor.stage.value),
                    "obj_leader": float(cursor_obj),
                    "level_index": int(level),
                    "level_weight": float(level_weight),
                    "level_reward": float(level_reward),
                    "peer_node_ids": peer_ids,
                    "updated": int(updated_this_level),
                }
            )

            cursor = cursor.parent
            level += 1

        return {
            "enabled": True,
            "updated": int(total_updated),
            "node_ids": touched_ids,
            "levels": levels,
            "coef": float(cluster_coef),
            "ancestor_decay": float(ancestor_decay),
            "sec": float(time.perf_counter() - t0),
        }

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

    def _finalize_result(
        self,
        *,
        root: SearchNode,
        records: list[StageExpansionRecord],
        stop_info: dict[str, Any],
        iteration_logs: list[dict[str, Any]],
    ) -> SearchRunResult:
        best_trajectory, best_reward = self._resolve_best_trajectory(
            root=root,
            records=records,
            iteration_logs=iteration_logs,
            stop_info=stop_info,
        )
        return SearchRunResult(
            root=root,
            records=records,
            stop_info=stop_info,
            best_trajectory=best_trajectory,
            best_reward=best_reward,
            code_nodes=self._collect_code_nodes(root),
            iteration_logs=iteration_logs,
        )

    def _resolve_best_trajectory(
        self,
        *,
        root: SearchNode,
        records: list[StageExpansionRecord],
        iteration_logs: list[dict[str, Any]],
        stop_info: dict[str, Any],
    ) -> tuple[Trajectory | None, float]:
        stop_reason = str((stop_info or {}).get("reason", "") or "").strip()
        explicit_tid = str((stop_info or {}).get("trajectory_id", "") or "").strip()

        if explicit_tid:
            explicit_record = self._record_by_trajectory_id(records, explicit_tid)
            if explicit_record is not None:
                chosen_reward = self._record_reward_total(explicit_record)
                self._annotate_final_selection(
                    explicit_record.trajectory,
                    reason_code=f"stop_reason_{stop_reason or 'explicit_trajectory'}",
                    reason="selected trajectory provided by stop_info",
                    judgement_condition={
                        "stop_reason": stop_reason,
                        "trajectory_id": explicit_tid,
                        "selection_mode": "explicit_stop_info_trajectory",
                    },
                )
                return explicit_record.trajectory, chosen_reward

        if stop_reason in {"max_iterations", "no_expandable_leaf"}:
            chosen = self._choose_for_exhaustive_stop(root=root, records=records)
            if chosen is not None:
                chosen_reward = self._record_reward_total(chosen)
                node_visits = self._node_visits_from_root(root)
                visit_count = int(node_visits.get(str(chosen.node_id), int(chosen.child_visits_after)))
                ratio_raw = getattr(self.config, "final_select_obj_scale_expand_ratio", 0.10)
                try:
                    expand_ratio = max(0.0, float(ratio_raw))
                except Exception:
                    expand_ratio = 0.10
                self._annotate_final_selection(
                    chosen.trajectory,
                    reason_code="obj_scale_majority_visit",
                    reason=(
                        "stop at max_iterations/no_expandable_leaf: keep only records within expanded obj-scale, "
                        "cluster by objective, choose largest cluster (tie with size>2 uses cluster average prior; "
                        "if all clusters are singletons choose global max prior), then choose highest-visit record "
                        "in selected cluster; tie-break by reward then prior"
                    ),
                    judgement_condition={
                        "stop_reason": stop_reason,
                        "obj_scale_mode": "expanded",
                        "obj_scale_expand_ratio": float(expand_ratio),
                        "cluster_choice": "majority_by_count",
                        "record_choice": "max_visit_then_reward_then_prior",
                        "selected_node_visits": visit_count,
                        "tie_breaker": "prior_when_reward_equal",
                    },
                )
                return chosen.trajectory, chosen_reward

        eligible_records: list[StageExpansionRecord] = []
        clusters: list[dict[str, Any]] = []

        for rec in records:
            reward = rec.trajectory.reward
            meta = (reward.metadata or {}) if reward is not None else {}
            execution_meta = meta.get('execution', {}) if isinstance(meta.get('execution', {}), dict) else {}
            obj = meta.get('obj_answer')
            if not isinstance(obj, (int, float)) or not math.isfinite(float(obj)):
                continue
            if not bool(execution_meta.get('effective_success', False)):
                continue

            eligible_records.append(rec)
            matched = None
            for cluster in clusters:
                if self._within_rel_tol(float(obj), float(cluster['leader'])):
                    matched = cluster
                    break
            if matched is None:
                matched = {'leader': float(obj), 'records': []}
                clusters.append(matched)
            matched['records'].append(rec)

        total_valid = len(eligible_records)
        if total_valid > 0:
            qualified_clusters: list[dict[str, Any]] = []
            for cluster in clusters:
                ratio = float(len(cluster['records'])) / float(total_valid)
                if ratio >= 0.4:
                    qualified_clusters.append(cluster)
            if qualified_clusters:
                qualified_clusters.sort(
                    key=lambda cluster: (
                        int(len(cluster['records'])),
                        max(
                            self._record_reward_total(rec)
                            for rec in cluster['records']
                        ),
                    ),
                    reverse=True,
                )
                chosen_cluster = qualified_clusters[0]
                ordered_cluster_records = sorted(
                    list(chosen_cluster["records"]),
                    key=lambda rec: (float(self._record_reward_total(rec)), float(self._record_prior(rec))),
                    reverse=True,
                )
                chosen, obj_scale_meta = self._pick_with_obj_scale_preference(ordered_cluster_records)
                chosen_reward = self._record_reward_total(chosen)
                chosen_cluster_ratio = float(len(chosen_cluster["records"])) / float(total_valid)
                self._annotate_final_selection(
                    chosen.trajectory,
                    reason_code="qualified_obj_consensus_cluster",
                    reason=(
                        "selected from objective-consensus cluster "
                        "(effective_success + finite objective, cluster ratio >= 0.4; tie-break: cluster size, then max reward), "
                        "then prefer first candidate whose objective is within expanded obj-scale bounds"
                    ),
                    judgement_condition={
                        "effective_success_required": True,
                        "finite_objective_required": True,
                        "consensus_ratio_threshold": 0.4,
                        "total_valid_records": int(total_valid),
                        "selected_cluster_leader": float(chosen_cluster.get("leader", 0.0)),
                        "selected_cluster_size": int(len(chosen_cluster["records"])),
                        "selected_cluster_ratio": float(chosen_cluster_ratio),
                        "cluster_tie_breaker": "cluster_size_then_cluster_max_reward",
                        "record_selection_rule": "max_reward_with_obj_scale_preference_within_selected_cluster",
                        **obj_scale_meta,
                    },
                )
                return chosen.trajectory, chosen_reward

        if records:
            if iteration_logs:
                last_iter = max(int(item.get('iter', -1)) for item in iteration_logs)
                last_iter_records = [rec for rec in records if int(rec.iteration) == last_iter]
                if last_iter_records:
                    ordered_last_iter_records = sorted(
                        list(last_iter_records),
                        key=lambda rec: (float(self._record_reward_total(rec)), float(self._record_prior(rec))),
                        reverse=True,
                    )
                    chosen, obj_scale_meta = self._pick_with_obj_scale_preference(ordered_last_iter_records)
                    chosen_reward = self._record_reward_total(chosen)
                    self._annotate_final_selection(
                        chosen.trajectory,
                        reason_code="last_iteration_max_reward_fallback",
                        reason=(
                            "no qualified objective-consensus cluster, fallback to last-iteration candidates; "
                            "prefer first objective within expanded obj-scale bounds, otherwise fallback to first candidate"
                        ),
                        judgement_condition={
                            "fallback_from": "qualified_obj_consensus_cluster",
                            "last_iteration": int(last_iter),
                            "candidate_records_in_last_iteration": int(len(last_iter_records)),
                            "record_selection_rule": "max_reward_with_obj_scale_preference",
                            **obj_scale_meta,
                        },
                    )
                    return chosen.trajectory, chosen_reward
            ordered_records = sorted(
                list(records),
                key=lambda rec: (float(self._record_reward_total(rec)), float(self._record_prior(rec))),
                reverse=True,
            )
            chosen, obj_scale_meta = self._pick_with_obj_scale_preference(ordered_records)
            chosen_reward = self._record_reward_total(chosen)
            self._annotate_final_selection(
                chosen.trajectory,
                reason_code="global_max_reward_fallback",
                reason=(
                    "no last-iteration candidates, fallback to global candidates; "
                    "prefer first objective within expanded obj-scale bounds, otherwise fallback to first candidate"
                ),
                judgement_condition={
                    "fallback_from": "last_iteration_max_reward_fallback",
                    "candidate_records_global": int(len(records)),
                    "record_selection_rule": "max_reward_with_obj_scale_preference",
                    **obj_scale_meta,
                },
            )
            return chosen.trajectory, chosen_reward

        return None, float('-inf')

    @staticmethod
    def _record_reward_total(rec: StageExpansionRecord) -> float:
        return float(rec.trajectory.reward.total if rec.trajectory.reward is not None else rec.reward)

    @staticmethod
    def _record_prior(rec: StageExpansionRecord) -> float:
        return float(rec.prior)

    @staticmethod
    def _record_obj_answer(rec: StageExpansionRecord) -> float | None:
        reward = rec.trajectory.reward
        meta = (reward.metadata or {}) if reward is not None else {}
        obj = meta.get("obj_answer")
        if isinstance(obj, (int, float)) and math.isfinite(float(obj)):
            return float(obj)
        return None

    @staticmethod
    def _record_effective_success(rec: StageExpansionRecord) -> bool:
        reward = rec.trajectory.reward
        meta = (reward.metadata or {}) if reward is not None else {}
        execution_meta = meta.get("execution", {}) if isinstance(meta.get("execution", {}), dict) else {}
        return bool(execution_meta.get("effective_success", False))

    @staticmethod
    def _record_by_trajectory_id(records: list[StageExpansionRecord], trajectory_id: str) -> StageExpansionRecord | None:
        target = str(trajectory_id or "")
        for rec in records:
            if str(rec.trajectory.trajectory_id) == target:
                return rec
        return None

    def _cluster_records_by_obj(self, records: list[StageExpansionRecord]) -> list[dict[str, Any]]:
        clusters: list[dict[str, Any]] = []
        for rec in records:
            obj = self._record_obj_answer(rec)
            if obj is None:
                continue
            matched = None
            for cluster in clusters:
                if self._within_rel_tol(float(obj), float(cluster["leader"])):
                    matched = cluster
                    break
            if matched is None:
                matched = {"leader": float(obj), "records": []}
                clusters.append(matched)
            matched["records"].append(rec)
        return clusters

    @staticmethod
    def _pick_record_max_reward_prior(records: list[StageExpansionRecord]) -> StageExpansionRecord | None:
        if not records:
            return None
        return max(records, key=lambda rec: (float(FourStageMCTS._record_reward_total(rec)), float(FourStageMCTS._record_prior(rec))))

    @staticmethod
    def _node_visits_from_root(root: SearchNode) -> dict[str, int]:
        visits: dict[str, int] = {}
        stack: list[SearchNode] = [root]
        while stack:
            cur = stack.pop()
            visits[str(cur.node_id)] = int(cur.visits)
            stack.extend(cur.children)
        return visits

    def _choose_for_exhaustive_stop(self, *, root: SearchNode, records: list[StageExpansionRecord]) -> StageExpansionRecord | None:
        ratio_raw = getattr(self.config, "final_select_obj_scale_expand_ratio", 0.10)
        try:
            expand_ratio = max(0.0, float(ratio_raw))
        except Exception:
            expand_ratio = 0.10

        in_scale_records: list[StageExpansionRecord] = []
        for rec in records:
            in_bounds, _ = self._record_obj_in_expanded_scale(rec, expand_ratio=expand_ratio)
            if not in_bounds:
                continue
            if self._record_obj_answer(rec) is None:
                continue
            in_scale_records.append(rec)

        if not in_scale_records:
            return None

        clusters = self._cluster_records_by_obj(in_scale_records)
        if not clusters:
            return None

        node_visits = self._node_visits_from_root(root)
        max_cluster_size = max(int(len(cluster["records"])) for cluster in clusters)
        top_clusters = [cluster for cluster in clusters if int(len(cluster["records"])) == max_cluster_size]

        if max_cluster_size == 1:
            # All clusters are singletons: choose highest-prior record directly.
            return max(
                in_scale_records,
                key=lambda rec: (
                    float(self._record_prior(rec)),
                    float(self._record_reward_total(rec)),
                    int(node_visits.get(str(rec.node_id), int(rec.child_visits_after))),
                ),
            )

        def _cluster_avg_prior(cluster: dict[str, Any]) -> float:
            recs = list(cluster["records"])
            if not recs:
                return float("-inf")
            return float(sum(float(self._record_prior(rec)) for rec in recs) / max(1, len(recs)))

        if len(top_clusters) > 1 and max_cluster_size > 2:
            top_clusters.sort(
                key=lambda cluster: (
                    float(_cluster_avg_prior(cluster)),
                    max(float(self._record_reward_total(rec)) for rec in cluster["records"]),
                ),
                reverse=True,
            )
            chosen_cluster = top_clusters[0]
        else:
            top_clusters.sort(
                key=lambda cluster: (
                    max(float(self._record_reward_total(rec)) for rec in cluster["records"]),
                    float(_cluster_avg_prior(cluster)),
                ),
                reverse=True,
            )
            chosen_cluster = top_clusters[0]

        chosen_cluster_records = list(chosen_cluster["records"])
        chosen_cluster_records.sort(
            key=lambda rec: (
                int(node_visits.get(str(rec.node_id), int(rec.child_visits_after))),
                float(self._record_reward_total(rec)),
                float(self._record_prior(rec)),
            ),
            reverse=True,
        )
        return chosen_cluster_records[0]

    def _pick_with_obj_scale_preference(
        self,
        ordered_records: list[StageExpansionRecord],
    ) -> tuple[StageExpansionRecord, dict[str, Any]]:
        if not ordered_records:
            raise ValueError("ordered_records must be non-empty")

        enabled = bool(getattr(self.config, "final_select_obj_scale_preference", True))
        ratio_raw = getattr(self.config, "final_select_obj_scale_expand_ratio", 0.10)
        try:
            expand_ratio = max(0.0, float(ratio_raw))
        except Exception:
            expand_ratio = 0.10

        if not enabled:
            return ordered_records[0], {
                "obj_scale_filter_applied": False,
                "obj_scale_expand_ratio": float(expand_ratio),
                "obj_scale_candidate_count": int(len(ordered_records)),
                "obj_scale_selected_rank": 0,
                "obj_scale_selected_by_in_bounds": False,
                "obj_scale_fallback_to_first": True,
                "obj_scale_candidates_preview": [],
            }

        status_list: list[dict[str, Any]] = []
        for idx, rec in enumerate(ordered_records):
            in_expanded_bounds, status = self._record_obj_in_expanded_scale(rec, expand_ratio=expand_ratio)
            status_item = {
                "rank": int(idx),
                "trajectory_id": str(rec.trajectory.trajectory_id),
                "obj_answer": status.get("obj_answer"),
                "obj_in_expanded_bounds": bool(in_expanded_bounds),
                "reason": str(status.get("reason", "")),
            }
            status_list.append(status_item)
            if in_expanded_bounds:
                return rec, {
                    "obj_scale_filter_applied": True,
                    "obj_scale_expand_ratio": float(expand_ratio),
                    "obj_scale_candidate_count": int(len(ordered_records)),
                    "obj_scale_selected_rank": int(idx),
                    "obj_scale_selected_by_in_bounds": True,
                    "obj_scale_fallback_to_first": False,
                    "obj_scale_candidates_preview": status_list[:12],
                }

        return ordered_records[0], {
            "obj_scale_filter_applied": True,
            "obj_scale_expand_ratio": float(expand_ratio),
            "obj_scale_candidate_count": int(len(ordered_records)),
            "obj_scale_selected_rank": 0,
            "obj_scale_selected_by_in_bounds": False,
            "obj_scale_fallback_to_first": True,
            "obj_scale_candidates_preview": status_list[:12],
        }

    def _record_obj_in_expanded_scale(
        self,
        rec: StageExpansionRecord,
        *,
        expand_ratio: float,
    ) -> tuple[bool, dict[str, Any]]:
        reward = rec.trajectory.reward
        meta = (reward.metadata or {}) if reward is not None else {}
        obj = meta.get("obj_answer")
        if not isinstance(obj, (int, float)) or not math.isfinite(float(obj)):
            return False, {"reason": "obj_not_finite", "obj_answer": obj}

        base_scale = meta.get("base_obj_scale")
        if not isinstance(base_scale, dict):
            base_scale = meta.get("base_obj_bounds")
        if not isinstance(base_scale, dict):
            return False, {"reason": "base_obj_scale_missing", "obj_answer": float(obj)}

        expanded_scale = self._expand_obj_scale_margin(base_scale, ratio=expand_ratio)
        in_bounds = self._objective_matches_scale(float(obj), expanded_scale)
        return bool(in_bounds), {
            "reason": ("in_expanded_bounds" if in_bounds else "out_of_expanded_bounds"),
            "obj_answer": float(obj),
        }

    @staticmethod
    def _expand_obj_scale_margin(scale: dict[str, Any], ratio: float = 0.10) -> dict[str, Any]:
        margin_ratio = max(0.0, float(ratio))
        kind = str(scale.get("kind") or "interval").strip().lower()

        if kind == "point":
            # Margin expansion is only applied to explicit lower/upper bounds.
            # Point-based scales stay unchanged.
            return dict(scale)

        if kind == "union":
            out = dict(scale)
            intervals_out: list[dict[str, float | None]] = []
            intervals = scale.get("intervals") if isinstance(scale.get("intervals"), list) else []
            for item in intervals:
                if not isinstance(item, dict):
                    continue
                lo = item.get("lower")
                hi = item.get("upper")
                lo_num = float(lo) if isinstance(lo, (int, float)) and math.isfinite(float(lo)) else None
                hi_num = float(hi) if isinstance(hi, (int, float)) and math.isfinite(float(hi)) else None
                if lo_num is not None:
                    lo_num = lo_num - abs(lo_num) * margin_ratio
                if hi_num is not None:
                    hi_num = hi_num + abs(hi_num) * margin_ratio
                if lo_num is not None and hi_num is not None and lo_num > hi_num:
                    lo_num, hi_num = hi_num, lo_num
                intervals_out.append({"lower": lo_num, "upper": hi_num})
            out["intervals"] = intervals_out
            return out

        out = dict(scale)
        lo = scale.get("lower")
        hi = scale.get("upper")
        lo_num = float(lo) if isinstance(lo, (int, float)) and math.isfinite(float(lo)) else None
        hi_num = float(hi) if isinstance(hi, (int, float)) and math.isfinite(float(hi)) else None
        if lo_num is not None:
            lo_num = lo_num - abs(lo_num) * margin_ratio
        if hi_num is not None:
            hi_num = hi_num + abs(hi_num) * margin_ratio
        if lo_num is not None and hi_num is not None and lo_num > hi_num:
            lo_num, hi_num = hi_num, lo_num
        out["lower"] = lo_num
        out["upper"] = hi_num
        return out

    def _objective_matches_scale(self, obj_answer: float, scale: dict[str, Any]) -> bool:
        checker = getattr(self.rewarder, "_objective_matches_scale", None)
        if callable(checker):
            try:
                return bool(checker(obj_answer, scale))
            except Exception:
                pass

        kind = str(scale.get("kind") or "interval").strip().lower()
        eps = 1e-9
        if kind == "union":
            intervals = scale.get("intervals") if isinstance(scale.get("intervals"), list) else []
            for item in intervals:
                if not isinstance(item, dict):
                    continue
                lo = item.get("lower")
                hi = item.get("upper")
                if isinstance(lo, (int, float)) and obj_answer < float(lo) - eps:
                    continue
                if isinstance(hi, (int, float)) and obj_answer > float(hi) + eps:
                    continue
                return True
            return False if intervals else True

        if kind == "point":
            point = scale.get("point")
            if not isinstance(point, (int, float)):
                return True
            tol_abs = float(scale.get("tol_abs", 0.0) or 0.0) if isinstance(scale.get("tol_abs"), (int, float)) else 0.0
            tol_rel = float(scale.get("tol_rel", 0.0) or 0.0) if isinstance(scale.get("tol_rel"), (int, float)) else 0.0
            tol = max(tol_abs, abs(float(point)) * tol_rel)
            return abs(obj_answer - float(point)) <= tol + eps

        lo = scale.get("lower")
        hi = scale.get("upper")
        if isinstance(lo, (int, float)) and obj_answer < float(lo) - eps:
            return False
        if isinstance(hi, (int, float)) and obj_answer > float(hi) + eps:
            return False
        return True

    @staticmethod
    def _annotate_final_selection(
        trajectory: Trajectory,
        *,
        reason_code: str,
        reason: str,
        judgement_condition: dict[str, Any],
    ) -> None:
        metadata = trajectory.metadata if isinstance(trajectory.metadata, dict) else {}
        iter_value = metadata.get("iter")
        stage_value = metadata.get("stage")
        metadata["final_selection"] = {
            "reason_code": str(reason_code),
            "reason": str(reason),
            "judgement_condition": dict(judgement_condition or {}),
            "iteration": (int(iter_value) if isinstance(iter_value, (int, float)) else None),
            "stage": (str(stage_value) if stage_value is not None else ""),
        }
        trajectory.metadata = metadata

    def _check_recent_obj_consensus(
        self,
        *,
        records: list[StageExpansionRecord],
        current_iter: int,
    ) -> dict[str, Any] | None:
        if current_iter < 2:
            return None

        top_obj_gate = self._check_recent_top_obj_alignment(records=records, current_iter=current_iter)
        if top_obj_gate is None:
            return None

        min_iter = max(0, int(current_iter) - 2)
        recent_records = [rec for rec in records if int(rec.iteration) >= min_iter and int(rec.iteration) <= int(current_iter)]
        if not recent_records:
            return None

        ratio_raw = getattr(self.config, "final_select_obj_scale_expand_ratio", 0.10)
        try:
            expand_ratio = max(0.0, float(ratio_raw))
        except Exception:
            expand_ratio = 0.10

        eligible: list[StageExpansionRecord] = []
        for rec in recent_records:
            if self._record_obj_answer(rec) is None:
                continue
            if not self._record_effective_success(rec):
                continue
            in_scale, _ = self._record_obj_in_expanded_scale(rec, expand_ratio=expand_ratio)
            if not in_scale:
                continue
            eligible.append(rec)

        total_eligible = len(eligible)
        if total_eligible <= 0:
            return None

        clusters = self._cluster_records_by_obj(eligible)
        if not clusters:
            return None

        clusters.sort(
            key=lambda cluster: (
                float(len(cluster["records"])) / float(total_eligible),
                int(len(cluster["records"])),
                max(float(self._record_reward_total(rec)) for rec in cluster["records"]),
            ),
            reverse=True,
        )
        chosen_cluster = clusters[0]
        ratio = float(len(chosen_cluster["records"])) / float(total_eligible)
        if ratio < 0.5:
            return None

        latest_iter = max(int(rec.iteration) for rec in chosen_cluster["records"])
        latest_records = [rec for rec in chosen_cluster["records"] if int(rec.iteration) == int(latest_iter)]
        chosen = self._pick_record_max_reward_prior(latest_records)
        if chosen is None:
            return None

        return {
            "obj_leader": float(chosen_cluster["leader"]),
            "count": int(len(chosen_cluster["records"])),
            "window_rollouts": int(total_eligible),
            "ratio": float(ratio),
            "selected_iteration": int(latest_iter),
            "trajectory_id": str(chosen.trajectory.trajectory_id),
            "reward_total": float(self._record_reward_total(chosen)),
            "obj_scale_expand_ratio": float(expand_ratio),
            "recent_top_obj_gate": dict(top_obj_gate),
        }

    def _check_recent_top_obj_alignment(
        self,
        *,
        records: list[StageExpansionRecord],
        current_iter: int,
    ) -> dict[str, Any] | None:
        if current_iter < 2:
            return None

        target_iters = [int(current_iter) - 2, int(current_iter) - 1, int(current_iter)]
        top_records: list[StageExpansionRecord] = []
        top_objs: list[float] = []
        top_stages: list[str] = []

        for iter_id in target_iters:
            iter_records = [
                rec
                for rec in records
                if int(rec.iteration) == int(iter_id) and self._record_obj_answer(rec) is not None
            ]
            if not iter_records:
                return None

            top_rec = self._pick_record_max_reward_prior(iter_records)
            if top_rec is None:
                return None

            top_obj = self._record_obj_answer(top_rec)
            if top_obj is None:
                return None

            top_records.append(top_rec)
            top_objs.append(float(top_obj))
            top_stages.append(str(top_rec.stage.value))

        # New gate: recent 3 iterations must come from 3 different stages.
        # If stage repeats across the 3 consecutive iterations, do not stop.
        if len(set(top_stages)) < 3:
            return None

        leader = float(top_objs[0])
        for value in top_objs[1:]:
            if not self._within_rel_tol(float(value), leader):
                return None

        return {
            "iterations": [int(x) for x in target_iters],
            "stages": [str(x) for x in top_stages],
            "top_obj_values": [float(x) for x in top_objs],
            "top_obj_leader": float(leader),
            "trajectory_ids": [str(rec.trajectory.trajectory_id) for rec in top_records],
            "rule": "same_top_obj_across_last_3_iterations_with_rel_tol_and_distinct_3_stages",
        }

    def _check_code_stage_consensus(
        self,
        *,
        records: list[StageExpansionRecord],
        current_iter: int,
    ) -> dict[str, Any] | None:
        current_code_records = [
            rec
            for rec in records
            if rec.stage == Stage.CODE and int(rec.iteration) == int(current_iter)
        ]
        if not current_code_records:
            return None

        ratio_raw = getattr(self.config, "final_select_obj_scale_expand_ratio", 0.10)
        try:
            expand_ratio = max(0.0, float(ratio_raw))
        except Exception:
            expand_ratio = 0.10

        eligible: list[StageExpansionRecord] = []
        for rec in current_code_records:
            if self._record_obj_answer(rec) is None:
                continue
            in_bounds, _ = self._record_obj_in_expanded_scale(rec, expand_ratio=expand_ratio)
            if not in_bounds:
                continue
            eligible.append(rec)

        if len(eligible) < 2:
            return None

        clusters = self._cluster_records_by_obj(eligible)
        if not clusters:
            return None

        clusters.sort(
            key=lambda cluster: (
                float(len(cluster["records"])) / float(len(eligible)),
                int(len(cluster["records"])),
                max(float(self._record_reward_total(rec)) for rec in cluster["records"]),
            ),
            reverse=True,
        )
        chosen_cluster = clusters[0]
        ratio = float(len(chosen_cluster["records"])) / float(len(eligible))
        if ratio < 0.5:
            return None

        chosen = self._pick_record_max_reward_prior(chosen_cluster["records"])
        if chosen is None:
            return None

        return {
            "obj_leader": float(chosen_cluster["leader"]),
            "count": int(len(chosen_cluster["records"])),
            "eligible_count": int(len(eligible)),
            "ratio": float(ratio),
            "trajectory_id": str(chosen.trajectory.trajectory_id),
            "reward_total": float(self._record_reward_total(chosen)),
            "obj_scale_expand_ratio": float(expand_ratio),
        }

    def _run_code_terminal_refine(
        self,
        *,
        task: OptimizationTask,
        selected: SearchNode,
        records: list[StageExpansionRecord],
        code_rollouts: list[dict[str, Any]],
        stage_archive: list[Trajectory],
        current_iter: int,
    ) -> dict[str, Any]:
        terminal_group_id = f"terminal:{current_iter}:{selected.node_id}"
        partial_model = selected.to_partial_trajectory()
        model_text = self._compose_model_text(partial_model)

        candidate_items: list[dict[str, Any]] = []
        for item in code_rollouts:
            traj = item.get("trajectory")
            if traj is None:
                continue
            reward_obj = traj.reward if traj.reward is not None else item.get("reward_obj")
            if reward_obj is None:
                continue
            if traj.reward is None:
                traj.reward = reward_obj
            child = item.get("child")
            prior_val = float(getattr(child, "prior", 1.0) if child is not None else item.get("resolved_prior", 1.0) or 1.0)
            candidate_items.append(
                {
                    "trajectory": traj,
                    "reward": reward_obj,
                    "prior": prior_val,
                    "obj": self._reward_obj_answer(reward_obj),
                }
            )

        if not candidate_items:
            return {
                "enabled": False,
                "reason": "empty_code_candidates",
                "fallback_to_original_logic": True,
            }

        # CODE candidate selection rule: reward first, then prior.
        # Do not force obj-cluster consensus at this step.
        chosen_item = max(
            candidate_items,
            key=lambda x: (float(x["reward"].total), float(x["prior"])),
        )
        valid_obj_count = sum(
            1 for item in candidate_items if isinstance(item.get("obj"), (int, float)) and math.isfinite(float(item.get("obj")))
        )
        selection_debug: dict[str, Any] = {
            "mode": "max_reward_then_prior",
            "candidate_count": int(len(candidate_items)),
            "valid_obj_count": int(valid_obj_count),
            "tie_breaker": "prior_when_reward_equal",
        }

        candidate_traj = chosen_item["trajectory"]
        candidate_reward = chosen_item["reward"]
        repair_rounds = max(0, int(getattr(self.config, "code_repair", 0) or 0))
        terminal_trace: dict[str, Any] = {
            "enabled": True,
            "code_refine_enabled": bool(getattr(self.config, "code_refine", True)),
            "code_repair_max": int(repair_rounds),
            "obj_cons_node_id": str(selected.node_id),
            "selection": selection_debug,
            "initial_candidate": self._terminal_candidate_summary(candidate_traj),
            "steps": [],
        }

        issue_kind, issue_reason = self._classify_terminal_issue(candidate_reward)
        need_repair = (not self._reward_has_valid_obj(candidate_reward)) or issue_kind in {"error", "infeasible"}
        terminal_trace["initial_issue"] = {"kind": issue_kind, "reason": issue_reason}
        terminal_trace["initial_need_repair"] = bool(need_repair)
        for repair_idx in range(repair_rounds):
            if not need_repair:
                break

            current_code = str(candidate_traj.code or "")
            current_exec_text = self._execution_text_from_reward(candidate_reward)
            if issue_kind == "infeasible":
                repair_prompt = self._safe_template_render(
                    CODE_INFEASIBLE_PROMPT_TEMPLATE,
                    task_description=str(task.description or ""),
                    model_text=model_text,
                    code_text=current_code,
                    execution_text=current_exec_text,
                )
                repair_kind = "repair_infeasible"
            else:
                repair_prompt = self._safe_template_render(
                    CODE_ERROR_PROMPT_TEMPLATE,
                    task_description=str(task.description or ""),
                    code_text=current_code,
                    error_info=current_exec_text,
                )
                repair_kind = "repair_error"

            answer_text, repaired_code, repaired_prior = self._generate_code_candidate(repair_prompt)
            step_payload = {
                "kind": repair_kind,
                "round": int(repair_idx + 1),
                "prompt": repair_prompt,
                "answer": answer_text,
                "parsed_code_len": int(len(str(repaired_code or "").strip())),
                "prior": float(repaired_prior),
                "issue_before": {"kind": issue_kind, "reason": issue_reason},
            }

            if repaired_code.strip():
                repaired_traj = self._trajectory_with_code(partial_model, repaired_code, repaired_prior)
                repaired_reward = self._score_terminal_trajectory(repaired_traj, stage_archive)
                repaired_traj.reward = repaired_reward
                candidate_traj = repaired_traj
                candidate_reward = repaired_reward

            issue_kind, issue_reason = self._classify_terminal_issue(candidate_reward)
            need_repair = (not self._reward_has_valid_obj(candidate_reward)) or issue_kind in {"error", "infeasible"}
            step_payload["candidate"] = self._terminal_candidate_summary(candidate_traj)
            step_payload["reward"] = {
                "r1": float(candidate_reward.r1 if candidate_reward is not None else 0.0),
                "r2": float(candidate_reward.r2 if candidate_reward is not None else 0.0),
                "r3": float(candidate_reward.r3 if candidate_reward is not None else 0.0),
                "r4": float(candidate_reward.r4 if candidate_reward is not None else 0.0),
                "total": float(candidate_reward.total if candidate_reward is not None else 0.0),
            }
            step_payload["issue_after"] = {"kind": issue_kind, "reason": issue_reason}
            terminal_trace["steps"].append(step_payload)

        if self._record_by_trajectory_id(records, str(candidate_traj.trajectory_id)) is None:
            final_reward_total = float(candidate_reward.total if candidate_reward is not None else 0.0)
            records.append(
                StageExpansionRecord(
                    stage=Stage.CODE,
                    node_id=str(uuid.uuid4()),
                    parent_id=str(selected.node_id),
                    prompt=str((terminal_trace.get("steps", [{}])[-1] or {}).get("prompt", "")),
                    completion=str(candidate_traj.code or ""),
                    reward=final_reward_total,
                    trajectory=candidate_traj,
                    prior=float(candidate_traj.priors.get(Stage.CODE, 1.0)),
                    was_expanded=False,
                    simulation_index=int(current_iter),
                    rollout_index=-1,
                    group_id=terminal_group_id,
                    grpo_report={"terminal_refine": True},
                    iteration=int(current_iter),
                )
            )

        final_has_valid_obj = bool(self._reward_has_valid_obj(candidate_reward))
        final_issue_blocking = issue_kind in {"error", "infeasible"}
        # Only stop at CODE when both conditions are met:
        # 1) objective is valid; 2) final issue is not error/infeasible.
        fallback_to_original = (not final_has_valid_obj) or final_issue_blocking
        terminal_trace["final_candidate"] = self._terminal_candidate_summary(candidate_traj)
        terminal_trace["final_issue"] = {"kind": issue_kind, "reason": issue_reason}
        terminal_trace["final_has_valid_obj"] = bool(final_has_valid_obj)
        terminal_trace["final_issue_blocking"] = bool(final_issue_blocking)
        terminal_trace["fallback_to_original_logic"] = bool(fallback_to_original)
        terminal_trace["repair_logs"] = list(terminal_trace.get("steps", []))

        return {
            "enabled": True,
            "trajectory_id": str(candidate_traj.trajectory_id),
            "reward_total": float(candidate_reward.total if candidate_reward is not None else 0.0),
            "obj_answer": self._reward_obj_answer(candidate_reward),
            "fallback_to_original_logic": bool(fallback_to_original),
            "trace": terminal_trace,
        }

    def _build_code_refine_rollout_prompt(
        self,
        *,
        task: OptimizationTask,
        selected: SearchNode,
        records: list[StageExpansionRecord],
    ) -> str:
        partial_model = selected.to_partial_trajectory()
        model_text = self._compose_model_text(partial_model)
        selected_rec = (
            self._latest_record_for_node(records, node_id=str(selected.node_id), stage=selected.stage)
            if selected.stage is not None
            else None
        )
        seed_code = str((selected_rec.trajectory.code if selected_rec is not None else "") or "")
        seed_execution_text = self._execution_text_from_reward(selected_rec.trajectory.reward if selected_rec is not None else None)
        return self._safe_template_render(
            CODE_REFINE_PROMPT_TEMPLATE,
            task_description=str(task.description or ""),
            model_text=model_text,
            code_text=seed_code,
            execution_text=seed_execution_text,
        )

    def _compose_model_text(self, partial: Trajectory) -> str:
        blocks: list[str] = []
        stage_rank = {stage: idx for idx, stage in enumerate(self.stage_order)}
        ordered_items = sorted(
            list((partial.outputs or {}).items()),
            key=lambda item: int(stage_rank.get(item[0], 10_000)),
        )
        for stage, raw_text in ordered_items:
            if stage == Stage.CODE:
                continue
            text = str(raw_text or "").strip()
            if text:
                blocks.append(text)
        return "\n\n".join(blocks).strip()

    @staticmethod
    def _safe_template_render(template: str, **kwargs: Any) -> str:
        class _SafeMap(dict):
            def __missing__(self, key):  # type: ignore[override]
                return ""

        text = str(template or "")
        merged: dict[str, Any] = dict(kwargs or {})
        for _, field_name, _, _ in string.Formatter().parse(text):
            if not field_name:
                continue
            if field_name not in merged:
                merged[field_name] = ""
        try:
            return text.format_map(_SafeMap(merged))
        except Exception:
            rendered = text
            for key, value in merged.items():
                rendered = rendered.replace("{" + str(key) + "}", str(value))
            return rendered

    def _generate_code_candidate(self, prompt: Any) -> tuple[str, str, float]:
        try:
            generations = self._backend_generate(Stage.CODE, prompt, 1, no_lora_adapter=True)
        except Exception as exc:  # noqa: BLE001
            return f"[generation_error] {type(exc).__name__}: {exc}", "", 0.0
        if not generations:
            return "", "", 0.0
        generation = generations[0]
        answer_text = str(generation.text or "")
        code_text = self._extract_stage_payload(Stage.CODE, answer_text)
        if not code_text.strip() and self._looks_like_code(answer_text):
            code_text = self._sanitize_code_payload(answer_text)
        return answer_text, str(code_text or ""), float(getattr(generation, "prior", 1.0) or 1.0)

    @staticmethod
    def _trajectory_with_code(partial: Trajectory, code_text: str, code_prior: float = 1.0) -> Trajectory:
        outputs = dict(partial.outputs)
        priors = dict(partial.priors)
        outputs[Stage.CODE] = str(code_text or "")
        priors[Stage.CODE] = float(max(1e-6, code_prior))
        return Trajectory(
            trajectory_id=str(uuid.uuid4()),
            outputs=outputs,
            priors=priors,
            metadata=dict(partial.metadata or {}),
        )

    def _score_terminal_trajectory(self, trajectory: Trajectory, stage_archive: list[Trajectory]) -> RewardBreakdown:
        score_group = getattr(self.rewarder, "score_rollout_group", None)
        if callable(score_group):
            try:
                scored = list(score_group(stage=Stage.CODE, trajectories=[trajectory], explored=stage_archive, commit=False))
                if scored:
                    trajectory.reward = scored[0]
                    return scored[0]
            except Exception:
                pass
        reward = self.rewarder.provisional_reward(trajectory, stage_archive)
        trajectory.reward = reward
        return reward

    @staticmethod
    def _execution_text_from_reward(reward: RewardBreakdown | None) -> str:
        if reward is None:
            return "{}"
        meta = reward.metadata if isinstance(reward.metadata, dict) else {}
        execution = meta.get("execution", {})
        payload = {
            "execution": execution,
            "obj_answer": meta.get("obj_answer"),
            "r1": reward.r1,
            "r2": reward.r2,
            "r3": reward.r3,
            "r4": reward.r4,
            "total": reward.total,
        }
        try:
            return json.dumps(payload, ensure_ascii=False, indent=2)
        except Exception:
            return str(payload)

    @staticmethod
    def _terminal_candidate_summary(trajectory: Trajectory | None) -> dict[str, Any]:
        if trajectory is None:
            return {"trajectory_id": "", "reward_total": 0.0, "obj_answer": None, "code_len": 0}
        reward = trajectory.reward
        meta = (reward.metadata or {}) if reward is not None else {}
        return {
            "trajectory_id": str(trajectory.trajectory_id),
            "reward_total": float(reward.total if reward is not None else 0.0),
            "obj_answer": meta.get("obj_answer"),
            "code_len": int(len(str(trajectory.code or "").strip())),
            "execution": meta.get("execution", {}),
        }

    @staticmethod
    def _reward_obj_answer(reward: RewardBreakdown | None) -> float | None:
        if reward is None:
            return None
        meta = reward.metadata if isinstance(reward.metadata, dict) else {}
        obj = meta.get("obj_answer")
        if isinstance(obj, (int, float)) and math.isfinite(float(obj)):
            return float(obj)
        return None

    @staticmethod
    def _reward_has_valid_obj(reward: RewardBreakdown | None) -> bool:
        return FourStageMCTS._reward_obj_answer(reward) is not None

    @staticmethod
    def _classify_terminal_issue(reward: RewardBreakdown | None) -> tuple[str, str]:
        if reward is None:
            return "error", "missing_reward"
        meta = reward.metadata if isinstance(reward.metadata, dict) else {}
        execution = meta.get("execution", {}) if isinstance(meta.get("execution", {}), dict) else {}
        output = execution.get("output")
        status_text = ""
        if isinstance(output, dict):
            status_text = str(output.get("status", "") or "")
        elif output is not None:
            status_text = str(output)
        err_type = str(execution.get("error_type", "") or "").strip()
        stdout_tail = str(execution.get("stdout_tail", "") or "")
        stderr_tail = str(execution.get("stderr_tail", "") or "")
        haystack = " ".join([status_text, stdout_tail, stderr_tail]).lower()
        infeasible_markers = ("infeasible", "inf_or_unbd", "infeasible or unbounded", "unbounded")
        if any(marker in haystack for marker in infeasible_markers):
            return "infeasible", "infeasible_marker"
        if err_type:
            return "error", err_type
        eff = bool(execution.get("effective_success", False))
        obj = meta.get("obj_answer")
        if eff and isinstance(obj, (int, float)) and math.isfinite(float(obj)):
            return "ok", "effective_success_with_obj"
        if not bool(execution.get("r2_success", False)) and not eff:
            return "error", "execution_failed"
        return "ok", "default"

    def _semantic_rel_tol(self) -> float:
        reward_cfg = getattr(self.rewarder, 'config', None)
        tol = getattr(reward_cfg, 'global_consensus_rel_tol', 0.005)
        try:
            return float(tol)
        except Exception:
            return 0.005

    def _within_rel_tol(self, a: float, b: float) -> bool:
        scale = max(abs(float(a)), abs(float(b)), 1.0)
        return abs(float(a) - float(b)) <= self._semantic_rel_tol() * scale

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

        blocks: list[str] = []
        for tag in FourStageMCTS._tags_for_stage(stage):
            block = FourStageMCTS._extract_tag_block(cleaned, tag=tag, min_len=21)
            if not block:
                continue
            blocks.append(f"<{tag}>\n{block.strip()}\n</{tag}>")

        return "\n\n".join(blocks).strip()

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
        # 1) Prefer explicit <python> ... </python> with first-close truncation.
        # 2) Fallback to legacy <Gurobi_code> ... </Gurobi_code>.
        # 3) Fallback to fenced code block extraction.
        by_tag = FourStageMCTS._extract_tag_block(cleaned, tag="python", min_len=21)
        if not by_tag:
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





