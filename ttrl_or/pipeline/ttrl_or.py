from __future__ import annotations

import json
import re
import time
import uuid
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ttrl_or.config import PipelineConfig
from ttrl_or.mcts import FourStageMCTS
from ttrl_or.model.backend import PolicyBackend
from ttrl_or.prompts import PromptBuilder, get_prompt_profile
from ttrl_or.reward import TTRLRewardCalculator
from ttrl_or.types import OptimizationTask, RunTrace, Stage, StageTrace, Trajectory


@dataclass(slots=True)
class TaskRunResult:
    task_id: str
    stage_reports: dict[str, dict]
    trajectories: list[Trajectory]
    best_trajectory: Trajectory | None
    trace: RunTrace | None = None



def _safe_path_component(name: str) -> str:
    raw = str(name or "").strip()
    if not raw:
        return "task"
    safe = re.sub(r'[\\/:*?"<>|]+', "_", raw)
    safe = safe.strip(" .")
    return safe or "task"


class TTRLORRunner:
    def __init__(self, backend: PolicyBackend, config: PipelineConfig | None = None) -> None:
        self.backend = backend
        self.config = config or PipelineConfig()
        profile = get_prompt_profile(self.config.mcts.solverllm_compare_mode)
        self.stage_order = profile.stage_order
        self.prompt_builder = PromptBuilder(
            templates=profile.templates,
            rollout_templates=profile.rollout_templates,
            completion_templates=profile.completion_templates,
            stage_order=profile.stage_order,
            system_instruction=profile.system_instruction,
        )

    def run_task(self, task: OptimizationTask, human_gold_answer: str | None = None) -> TaskRunResult:
        run_t0 = time.perf_counter()
        self.backend.begin_episode(task)
        task_context = self.backend.prepare_task_context(task, self.config.dataset)
        if human_gold_answer:
            task_context["gold_answer"] = str(human_gold_answer)

        backend_name = type(self.backend).__name__
        trace = RunTrace(
            task_id=task.task_id,
            backend=backend_name,
            task_description=task.description,
            instance=task.instance,
            perturbation_map=task.perturbation_map,
            task_context=task_context,
            config={
                "mcts": asdict(self.config.mcts),
                "reward": asdict(self.config.reward),
                "grpo": asdict(self.config.grpo),
                "dataset": asdict(self.config.dataset),
                "backend": asdict(self.config.backend),
            },
        )

        iter_live_writer = None
        try:
            rewarder = TTRLRewardCalculator(task=task, backend=self.backend, config=self.config.reward)
            mcts = FourStageMCTS(
                backend=self.backend,
                prompt_builder=self.prompt_builder,
                rewarder=rewarder,
                config=self.config.mcts,
                stage_order=self.stage_order,
                split_rollout_completion=self.config.mcts.solverllm_compare_mode,
            )

            if self.config.save_logs:
                run_dir = Path(self.config.log_dir) / _safe_path_component(task.task_id)
                run_dir.mkdir(parents=True, exist_ok=True)
                iter_live_writer = (run_dir / "mcts_iterations.md").open("w", encoding="utf-8")

            def _on_iteration_log(payload: dict[str, Any]) -> None:
                if iter_live_writer is None:
                    return
                iter_live_writer.write(self._format_iteration_markdown(payload))
                iter_live_writer.flush()

                best = payload.get("best_rollout", {}) if isinstance(payload, dict) else {}
                reward_obj = best.get("reward", {}) if isinstance(best, dict) else {}
                timing = payload.get("timing", {}) if isinstance(payload, dict) else {}
                iter_idx = payload.get("iter", payload.get("iteration", -1))
                stage_name = payload.get("stage", "")
                best_reward = reward_obj.get("total")
                obj_answer = reward_obj.get("obj_answer")
                gold_answer = str(task_context.get("gold_answer", ""))
                print(
                    f"[MCTS][task={task.task_id}] iter={iter_idx} stage={stage_name} "
                    f"best_reward={best_reward} obj={obj_answer} gt={gold_answer} "
                    f"iter_sec={timing.get('iteration_total_sec', 'n/a')} "
                    f"rollout_sec={timing.get('rollout_group_wall_sec', 'n/a')} "
                    f"exec_sec={timing.get('code_execution_total_sec', 'n/a')}"
                )

            search_result = mcts.search(task=task, grpo_config=self.config.grpo, iteration_callback=_on_iteration_log)
            if iter_live_writer is not None:
                iter_live_writer.close()
                iter_live_writer = None
            records = search_result.records
            stop_info = dict(search_result.stop_info or {})
            iteration_logs = list(search_result.iteration_logs or [])
            trace.iteration_logs = iteration_logs

            stage_reports: dict[str, dict] = {}

            for stage in self.stage_order:
                stage_records = [r for r in records if r.stage == stage]
                stage_samples = len(stage_records)
                stage_update_reports = self._collect_group_reports(stage_records)
                grpo_report = self._summarize_stage_grpo(
                    stage=stage,
                    stage_samples=stage_samples,
                    update_reports=stage_update_reports,
                )

                stage_stop = bool(stop_info.get("stage", "") == stage.value)
                stage_report = dict(grpo_report)
                stage_report["mcts_stop"] = stage_stop
                if stage_stop:
                    stage_report["mcts_stop_info"] = dict(stop_info)
                stage_reports[stage.value] = stage_report

                trace.stages.append(
                    StageTrace(
                        stage=stage.value,
                        num_frontier_in=len({r.parent_id for r in stage_records}),
                        num_frontier_out=len({r.node_id for r in stage_records}),
                        stage_samples=stage_samples,
                        grpo_report=grpo_report,
                        mcts_early_stop=stage_stop,
                        mcts_early_stop_info=(dict(stop_info) if stage_stop else {}),
                        expansions=[
                            {
                                "iteration": record.iteration,
                                "node_id": record.node_id,
                                "parent_id": record.parent_id,
                                "group_id": record.group_id,
                                "simulation_index": record.simulation_index,
                                "rollout_index": record.rollout_index,
                                "prior": record.prior,
                                "was_expanded": record.was_expanded,
                                "hit_reward_one": record.hit_reward_one,
                                "reward": record.reward,
                                "child_q_before": record.child_q_before,
                                "child_q_after": record.child_q_after,
                                "child_visits_before": record.child_visits_before,
                                "child_visits_after": record.child_visits_after,
                                "parent_q_before": record.parent_q_before,
                                "parent_q_after": record.parent_q_after,
                                "parent_visits_before": record.parent_visits_before,
                                "parent_visits_after": record.parent_visits_after,
                                "rollout_details": record.rollout_details,
                                "grpo_report": record.grpo_report,
                                "completion_preview": record.completion[:300],
                            }
                            for record in stage_records
                        ],
                    )
                )

            best = search_result.best_trajectory
            if best is not None:
                group_trajectories = [best]
            elif search_result.code_nodes:
                group_trajectories = [node.to_partial_trajectory() for node in search_result.code_nodes]
                group_trajectories = rewarder.finalize_group(group_trajectories)
                best = self._pick_best_by_reward(group_trajectories, search_result.code_nodes)
            else:
                group_trajectories = []

            runtime_summary = self._build_runtime_summary(
                iteration_logs=iteration_logs,
                total_elapsed_sec=float(time.perf_counter() - run_t0),
            )

            final_selection = {
                "selection_basis": "global_search_reward",
                "stop_info": stop_info,
                "num_records": len(records),
                "num_code_nodes": len(search_result.code_nodes),
                "best_reward": (best.reward.total if best and best.reward else None),
                "best_trajectory_id": (best.trajectory_id if best else ""),
            }
            stage_reports["final_selection"] = final_selection

            mcts_stats = self._build_mcts_stats(trace)
            stage_reports["mcts_stats"] = mcts_stats
            stage_reports["runtime"] = runtime_summary

            trace.final_selection = {**final_selection, "runtime": runtime_summary}
            trace.best_trajectory = self._best_trace(best)

            if self.config.save_logs:
                artifacts = self._save_trace_artifacts(
                    trace,
                    group_trajectories,
                    best,
                    mcts_stats,
                    iteration_logs,
                    runtime_summary,
                )
                trace.artifacts = artifacts
            return TaskRunResult(
                task_id=task.task_id,
                stage_reports=stage_reports,
                trajectories=group_trajectories,
                best_trajectory=best,
                trace=trace,
            )
        finally:
            if iter_live_writer is not None:
                iter_live_writer.close()
            self.backend.end_episode()

    def run_from_text(
        self,
        description: str,
        instance: dict | None = None,
        task_id: str | None = None,
        gold_answer: str | None = None,
    ) -> TaskRunResult:
        task = OptimizationTask(
            task_id=task_id or str(uuid.uuid4()),
            description=description,
            instance=instance or {},
        )
        return self.run_task(task, human_gold_answer=gold_answer)

    @staticmethod
    def _pick_best_by_reward(trajectories: list[Trajectory], selected_nodes) -> Trajectory | None:
        if not trajectories:
            return None

        node_by_text = {(node.text or "").strip(): node for node in selected_nodes}

        def _key(traj: Trajectory) -> tuple[float, float, int]:
            reward = traj.reward.total if traj.reward else float("-inf")
            node = node_by_text.get((traj.code or "").strip())
            q = node.q_value if node is not None else 0.0
            visits = node.visits if node is not None else 0
            return reward, q, visits

        return max(trajectories, key=_key)

    def _save_trace_artifacts(
        self,
        trace: RunTrace,
        trajectories: list[Trajectory],
        best: Trajectory | None,
        mcts_stats: dict,
        iteration_logs: list[dict[str, Any]],
        runtime_summary: dict[str, Any],
    ) -> dict[str, str]:
        run_dir = Path(self.config.log_dir) / _safe_path_component(trace.task_id)
        run_dir.mkdir(parents=True, exist_ok=True)

        summary_path = run_dir / "run_summary.json"
        stages_path = run_dir / "stage_events.json"
        stages_md_path = run_dir / "stage_events.md"
        trajectories_path = run_dir / "final_trajectories.json"
        best_code_path = run_dir / "best_code.py"
        mcts_stats_path = run_dir / "mcts_stats.json"
        iter_logs_path = run_dir / "mcts_iterations.json"
        iter_logs_md_path = run_dir / "mcts_iterations.md"
        result_path = run_dir / "result.json"
        runtime_path = run_dir / "runtime_summary.json"
        runtime_md_path = run_dir / "runtime_summary.md"
        selected_trace_path = run_dir / "selected_trajectory.json"
        config_path = Path(self.config.log_dir) / "run_config.json"
        if not config_path.exists():
            config_path = Path(self.config.log_dir).parent / "run_config.json"

        summary_payload = {
            "task_id": trace.task_id,
            "backend": trace.backend,
            "task_description": trace.task_description,
            "instance": trace.instance,
            "perturbation_map": trace.perturbation_map,
            "task_context": trace.task_context,
            "config": trace.config,
            "final_selection": trace.final_selection,
            "best_trajectory": trace.best_trajectory,
            "runtime": runtime_summary,
        }
        summary_path.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")

        stage_events_payload = [asdict(stage_trace) for stage_trace in trace.stages]
        stages_path.write_text(json.dumps(stage_events_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        stages_md_path.write_text(self._format_stage_events_markdown(stage_events_payload), encoding="utf-8")

        iter_logs_path.write_text(json.dumps(iteration_logs, ensure_ascii=False, indent=2), encoding="utf-8")
        iter_logs_md_path.write_text(
            "".join(self._format_iteration_markdown(item) for item in iteration_logs),
            encoding="utf-8",
        )

        traj_payload = [
            {
                "trajectory_id": traj.trajectory_id,
                "priors": {s.value: p for s, p in traj.priors.items()},
                "reward": asdict(traj.reward) if traj.reward else None,
                "code": traj.code,
            }
            for traj in trajectories
        ]
        trajectories_path.write_text(json.dumps(traj_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        mcts_stats_path.write_text(json.dumps(mcts_stats, ensure_ascii=False, indent=2), encoding="utf-8")

        if best is not None:
            best_code_path.write_text(best.code, encoding="utf-8")
        else:
            best_code_path.write_text("", encoding="utf-8")

        result_payload = {
            "best_code": best.code if best is not None else "",
            "obj_answer": self._best_obj_answer(best),
            "gold_answer": str(trace.task_context.get("gold_answer", "")),
        }
        result_path.write_text(json.dumps(result_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        runtime_path.write_text(json.dumps(runtime_summary, ensure_ascii=False, indent=2), encoding="utf-8")
        runtime_md_path.write_text(self._format_runtime_summary_markdown(runtime_summary), encoding="utf-8")

        selected_payload = {
            "selected_iter": (best.metadata.get("iter") if best is not None else None),
            "max_iter": int(((trace.config or {}).get("mcts") or {}).get("max_iterations", 0)),
            "gt": str(trace.task_context.get("gold_answer", "")),
            "trajectory_id": (best.trajectory_id if best is not None else ""),
            "reward": (asdict(best.reward) if best is not None and best.reward else None),
            "content": (
                {stage.value: best.outputs.get(stage, "") for stage in self.stage_order}
                if best is not None
                else {}
            ),
            "code_execution": (
                ((best.reward.metadata or {}).get("execution", {}))
                if best is not None and best.reward
                else {}
            ),
        }
        selected_trace_path.write_text(json.dumps(selected_payload, ensure_ascii=False, indent=2), encoding="utf-8")

        return {
            "run_dir": str(run_dir.resolve()),
            "run_summary": str(summary_path.resolve()),
            "stage_events": str(stages_path.resolve()),
            "stage_events_md": str(stages_md_path.resolve()),
            "mcts_iterations": str(iter_logs_path.resolve()),
            "mcts_iterations_md": str(iter_logs_md_path.resolve()),
            "mcts_stats": str(mcts_stats_path.resolve()),
            "final_trajectories": str(trajectories_path.resolve()),
            "best_code": str(best_code_path.resolve()),
            "result_json": str(result_path.resolve()),
            "runtime_summary": str(runtime_path.resolve()),
            "runtime_summary_md": str(runtime_md_path.resolve()),
            "selected_trajectory": str(selected_trace_path.resolve()),
            "config_path": str(config_path.resolve()) if config_path.exists() else "",
        }
    @staticmethod
    def _format_iteration_markdown(payload: dict[str, Any]) -> str:
        iter_idx = payload.get("iter", -1)
        stage = payload.get("stage", "")
        selection = payload.get("selection", {}) if isinstance(payload, dict) else {}
        best = payload.get("best_rollout", {}) if isinstance(payload, dict) else {}
        timing = payload.get("timing", {}) if isinstance(payload, dict) else {}
        reward = best.get("reward", {}) if isinstance(best, dict) else {}
        parent = best.get("parent_node", {}) if isinstance(best, dict) else {}
        leaf_candidates = selection.get("leaf_candidates", []) if isinstance(selection, dict) else []
        selection_path = selection.get("selection_path", []) if isinstance(selection, dict) else []
        best_prior = best.get("prior", {}) if isinstance(best, dict) else {}
        rollout_group = payload.get("rollout_group", []) if isinstance(payload, dict) else []

        lines = [
            f"# Iter {iter_idx} | Stage {stage}",
            "",
            "## Selection",
            f"- selected_node: {parent.get('node_id', '')}",
            f"- selected_stage: {parent.get('stage', '')}",
            f"- selected_value: {parent.get('value', '')}",
            f"- selected_visits: {parent.get('visits', '')}",
            "",
            "### Recursive Selection Path",
        ]
        if selection_path:
            for step_idx, step in enumerate(selection_path):
                lines.append(
                    f"- step={step_idx} parent={step.get('parent_node_id', '')} stage={step.get('parent_stage', '')} "
                    f"selected_child={step.get('selected_child_id', '')} selected_puct={step.get('selected_child_score', '')}"
                )
                for cand in step.get('candidates', []) or []:
                    lines.append(
                        f"  child={cand.get('node_id', '')} stage={cand.get('stage', '')} "
                        f"puct={cand.get('puct_score', '')} q={cand.get('value', '')} visits={cand.get('visits', '')} prior={cand.get('prior', '')}"
                    )
        else:
            lines.append("- root selected directly")

        lines.extend([
            "",
            "### Expandable Leaf Ranking",
        ])
        if leaf_candidates:
            for c in leaf_candidates:
                lines.append(
                    f"- node={c.get('node_id', '')} stage={c.get('stage', '')} "
                    f"puct={c.get('puct_score', '')} value={c.get('value', '')} visits={c.get('visits', '')}"
                )
        else:
            lines.append("- none")

        lines.extend([
            "",
            "## Rollout Group Summary",
        ])
        if rollout_group:
            for item in rollout_group:
                reward_i = item.get('reward', {}) if isinstance(item, dict) else {}
                timing_i = item.get('timing', {}) if isinstance(item, dict) else {}
                lines.append(
                    f"- rollout={item.get('rollout_index', '')} node={item.get('node_id', '')} "
                    f"prior={item.get('prior', '')} prior_source={item.get('prior_source', '')} "
                    f"obj={item.get('obj_answer', '')} total={reward_i.get('total', '')} "
                    f"r1={reward_i.get('r1', '')} r2={reward_i.get('r2', '')} r3={reward_i.get('r3', '')} r4={reward_i.get('r4', '')} "
                    f"backprop_sec={timing_i.get('backprop_sec', '')}"
                )
        else:
            lines.append("- none")

        lines.extend(
            [
                "",
                "## Best Rollout",
                f"- rollout_index: {best.get('rollout_index', '')}",
                f"- resolved_prior: {best_prior.get('resolved_prior', '')}",
                f"- prior_source: {best_prior.get('source', '')}",
                f"- obj: {reward.get('obj_answer', '')}",
                f"- gt: {best.get('gt', '')}",
                f"- reward: r1={reward.get('r1', '')}, r2={reward.get('r2', '')}, r3={reward.get('r3', '')}, r4={reward.get('r4', '')}, total={reward.get('total', '')}",
                "",
                "### Prompt",
                "```text",
                str((best.get('prompt', {}) or {}).get("full", "")),
                "```",
                "",
                "### Answer",
                "```text",
                str((best.get('answer', {}) or {}).get("full", "")),
                "```",
                "",
                "### Timing (sec)",
                f"- iteration_total_sec: {timing.get('iteration_total_sec', '')}",
                f"- selection_sec: {timing.get('selection_sec', '')}",
                f"- rollout_group_wall_sec: {timing.get('rollout_group_wall_sec', '')}",
                f"- grpo_train_runtime_sec: {timing.get('grpo_train_runtime_sec', '')}",
                f"- reward_callback_total_sec: {timing.get('reward_callback_total_sec', '')}",
                f"- code_execution_total_sec: {timing.get('code_execution_total_sec', '')}",
                f"- backprop_total_sec: {timing.get('backprop_total_sec', '')}",
                "",
                "---",
                "",
            ]
        )
        return "\n".join(lines)

    @staticmethod
    def _format_stage_events_markdown(stage_events_payload: list[dict[str, Any]]) -> str:
        lines = ["# Stage Events", ""]
        for stage in stage_events_payload:
            stage_name = stage.get("stage", "")
            lines.extend(
                [
                    f"## {stage_name}",
                    f"- num_frontier_in: {stage.get('num_frontier_in', '')}",
                    f"- num_frontier_out: {stage.get('num_frontier_out', '')}",
                    f"- stage_samples: {stage.get('stage_samples', '')}",
                    f"- mcts_early_stop: {stage.get('mcts_early_stop', '')}",
                    "",
                ]
            )
        return "\n".join(lines)
    @staticmethod
    def _best_obj_answer(best: Trajectory | None) -> Any:
        if best is None or best.reward is None:
            return None
        metadata = best.reward.metadata or {}
        return metadata.get("obj_answer")

    @staticmethod
    def _best_trace(best: Trajectory | None) -> dict:
        if best is None:
            return {}
        return {
            "trajectory_id": best.trajectory_id,
            "priors": {s.value: p for s, p in best.priors.items()},
            "reward": asdict(best.reward) if best.reward else None,
        }

    @staticmethod
    def _collect_group_reports(records) -> list[dict[str, Any]]:
        by_group: dict[str, dict[str, Any]] = {}
        for record in records:
            if not record.group_id:
                continue
            if record.group_id in by_group:
                continue
            report = dict(record.grpo_report or {})
            report["group_id"] = record.group_id
            report["iteration"] = record.iteration
            by_group[record.group_id] = report
        return list(by_group.values())

    @staticmethod
    def _summarize_stage_grpo(stage: Stage, stage_samples: int, update_reports: list[dict[str, Any]]) -> dict[str, Any]:
        train_losses = [float(r["train_loss"]) for r in update_reports if "train_loss" in r]
        train_runtimes = [float(r["train_runtime"]) for r in update_reports if "train_runtime" in r]

        summary: dict[str, Any] = {
            "mode": "global_leaf_select_internal_group_rollout",
            "stage": stage.value,
            "num_updates": len(update_reports),
            "num_samples": stage_samples,
            "updated": any(bool(r.get("updated", False)) for r in update_reports),
            "updates": update_reports,
        }
        if train_losses:
            summary["train_loss_mean"] = sum(train_losses) / len(train_losses)
        if train_runtimes:
            summary["train_runtime_sum"] = sum(train_runtimes)
        return summary

    @staticmethod
    def _build_runtime_summary(iteration_logs: list[dict[str, Any]], total_elapsed_sec: float) -> dict[str, Any]:
        per_iteration: list[dict[str, Any]] = []
        reward_totals: list[float] = []
        iter_secs: list[float] = []

        for item in iteration_logs:
            timing = item.get("timing", {}) if isinstance(item, dict) else {}
            best = item.get("best_rollout", {}) if isinstance(item, dict) else {}
            reward = best.get("reward", {}) if isinstance(best, dict) else {}

            reward_total = reward.get("total")
            iter_sec = timing.get("iteration_total_sec")

            if isinstance(reward_total, (int, float)):
                reward_totals.append(float(reward_total))
            if isinstance(iter_sec, (int, float)):
                iter_secs.append(float(iter_sec))

            per_iteration.append(
                {
                    "iter": item.get("iter"),
                    "stage": item.get("stage"),
                    "reward_total": reward_total,
                    "r1": reward.get("r1"),
                    "r2": reward.get("r2"),
                    "r3": reward.get("r3"),
                    "r4": reward.get("r4"),
                    "iter_sec": iter_sec,
                    "rollout_group_wall_sec": timing.get("rollout_group_wall_sec"),
                    "grpo_train_runtime_sec": timing.get("grpo_train_runtime_sec"),
                    "code_execution_total_sec": timing.get("code_execution_total_sec"),
                    "reward_callback_total_sec": timing.get("reward_callback_total_sec"),
                }
            )

        return {
            "sample_total_sec": float(total_elapsed_sec),
            "num_iterations": len(iteration_logs),
            "iter_time_sum_sec": float(sum(iter_secs)) if iter_secs else 0.0,
            "iter_time_mean_sec": (float(sum(iter_secs) / len(iter_secs)) if iter_secs else None),
            "best_reward": (max(reward_totals) if reward_totals else None),
            "last_reward": (reward_totals[-1] if reward_totals else None),
            "per_iteration": per_iteration,
        }

    @staticmethod
    def _format_runtime_summary_markdown(runtime_summary: dict[str, Any]) -> str:
        lines = [
            "# Runtime Summary",
            "",
            f"- sample_total_sec: {runtime_summary.get('sample_total_sec')}",
            f"- num_iterations: {runtime_summary.get('num_iterations')}",
            f"- iter_time_sum_sec: {runtime_summary.get('iter_time_sum_sec')}",
            f"- iter_time_mean_sec: {runtime_summary.get('iter_time_mean_sec')}",
            f"- best_reward: {runtime_summary.get('best_reward')}",
            f"- last_reward: {runtime_summary.get('last_reward')}",
            "",
            "## Per Iteration",
            "| iter | stage | reward_total | r1 | r2 | r3 | r4 | iter_sec | rollout_group_wall_sec | grpo_train_runtime_sec | code_execution_total_sec | reward_callback_total_sec |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for item in runtime_summary.get("per_iteration", []):
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(item.get("iter", "")),
                        str(item.get("stage", "")),
                        str(item.get("reward_total", "")),
                        str(item.get("r1", "")),
                        str(item.get("r2", "")),
                        str(item.get("r3", "")),
                        str(item.get("r4", "")),
                        str(item.get("iter_sec", "")),
                        str(item.get("rollout_group_wall_sec", "")),
                        str(item.get("grpo_train_runtime_sec", "")),
                        str(item.get("code_execution_total_sec", "")),
                        str(item.get("reward_callback_total_sec", "")),
                    ]
                )
                + " |"
            )
        lines.append("")
        return "\n".join(lines)
    @staticmethod
    def _build_mcts_stats(trace: RunTrace) -> dict:
        per_stage: dict[str, dict] = {}
        total_expansions = 0
        total_rollouts = 0
        total_grpo_updates = 0
        total_grpo_samples = 0

        for stage_trace in trace.stages:
            rollout_by_node: dict[str, int] = defaultdict(int)
            expanded_nodes: list[str] = []
            reused_nodes: list[str] = []

            for exp in stage_trace.expansions:
                total_expansions += 1
                node_id = str(exp.get("node_id", ""))
                num_rollouts = len(exp.get("rollout_details", []))
                total_rollouts += num_rollouts
                rollout_by_node[node_id] += num_rollouts

                if exp.get("was_expanded", False):
                    expanded_nodes.append(node_id)
                else:
                    reused_nodes.append(node_id)

            grpo_report = stage_trace.grpo_report or {}
            stage_grpo_updates = int(grpo_report.get("num_updates", 0))
            stage_grpo_samples = int(grpo_report.get("num_samples", 0))
            total_grpo_updates += stage_grpo_updates
            total_grpo_samples += stage_grpo_samples

            per_stage[stage_trace.stage] = {
                "frontier_in": stage_trace.num_frontier_in,
                "frontier_out": stage_trace.num_frontier_out,
                "simulations": len(stage_trace.expansions),
                "expanded_count": len(expanded_nodes),
                "reused_count": len(reused_nodes),
                "expanded_node_ids": expanded_nodes,
                "reused_node_ids": reused_nodes,
                "rollouts_total": sum(rollout_by_node.values()),
                "rollouts_by_node": dict(rollout_by_node),
                "grpo_updates": stage_grpo_updates,
                "grpo_samples": stage_grpo_samples,
                "early_stop": bool(stage_trace.mcts_early_stop),
                "early_stop_info": dict(stage_trace.mcts_early_stop_info),
            }

        return {
            "total_expansions": total_expansions,
            "total_rollouts": total_rollouts,
            "total_grpo_updates": total_grpo_updates,
            "total_grpo_samples": total_grpo_samples,
            "per_stage": per_stage,
        }

