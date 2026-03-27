from __future__ import annotations

import json
import uuid
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ttrl_or.config import PipelineConfig
from ttrl_or.mcts import FourStageMCTS
from ttrl_or.model.backend import PolicyBackend
from ttrl_or.prompts import DEFAULT_TEMPLATES, PromptBuilder
from ttrl_or.reward import TTRLRewardCalculator
from ttrl_or.types import OptimizationTask, RunTrace, STAGE_ORDER, Stage, StageTrace, Trajectory


@dataclass(slots=True)
class TaskRunResult:
    task_id: str
    stage_reports: dict[str, dict]
    trajectories: list[Trajectory]
    best_trajectory: Trajectory | None
    trace: RunTrace | None = None


class TTRLORRunner:
    def __init__(self, backend: PolicyBackend, config: PipelineConfig | None = None) -> None:
        self.backend = backend
        self.config = config or PipelineConfig()
        self.prompt_builder = PromptBuilder(templates=DEFAULT_TEMPLATES)

    def run_task(self, task: OptimizationTask) -> TaskRunResult:
        self.backend.begin_episode(task)
        task_context = self.backend.prepare_task_context(task, self.config.dataset)
        if task.gold_answer:
            task_context["gold_answer"] = task.gold_answer

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

        try:
            rewarder = TTRLRewardCalculator(task=task, backend=self.backend, config=self.config.reward)
            mcts = FourStageMCTS(
                backend=self.backend,
                prompt_builder=self.prompt_builder,
                rewarder=rewarder,
                config=self.config.mcts,
            )

            search_result = mcts.search(task=task, grpo_config=self.config.grpo)
            records = search_result.records
            stop_info = dict(search_result.stop_info or {})
            iteration_logs = list(search_result.iteration_logs or [])
            trace.iteration_logs = iteration_logs

            stage_reports: dict[str, dict] = {}

            for stage in STAGE_ORDER:
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

            trace.final_selection = final_selection
            trace.best_trajectory = self._best_trace(best)

            if self.config.save_logs:
                artifacts = self._save_trace_artifacts(trace, group_trajectories, best, mcts_stats, iteration_logs)
                trace.artifacts = artifacts

            return TaskRunResult(
                task_id=task.task_id,
                stage_reports=stage_reports,
                trajectories=group_trajectories,
                best_trajectory=best,
                trace=trace,
            )
        finally:
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
            gold_answer=(gold_answer or ""),
        )
        return self.run_task(task)

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
    ) -> dict[str, str]:
        run_dir = Path(self.config.log_dir) / trace.task_id
        run_dir.mkdir(parents=True, exist_ok=True)

        summary_path = run_dir / "run_summary.json"
        stages_path = run_dir / "stage_events.jsonl"
        trajectories_path = run_dir / "final_trajectories.json"
        best_code_path = run_dir / "best_code.py"
        mcts_stats_path = run_dir / "mcts_stats.json"
        iter_logs_path = run_dir / "mcts_iterations.jsonl"
        result_path = run_dir / "result.json"

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
        }
        summary_path.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")

        with stages_path.open("w", encoding="utf-8") as f:
            for stage_trace in trace.stages:
                f.write(json.dumps(asdict(stage_trace), ensure_ascii=False) + "\n")

        with iter_logs_path.open("w", encoding="utf-8") as f:
            for item in iteration_logs:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

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

        return {
            "run_dir": str(run_dir.resolve()),
            "run_summary": str(summary_path.resolve()),
            "stage_events": str(stages_path.resolve()),
            "mcts_iterations": str(iter_logs_path.resolve()),
            "mcts_stats": str(mcts_stats_path.resolve()),
            "final_trajectories": str(trajectories_path.resolve()),
            "best_code": str(best_code_path.resolve()),
            "result_json": str(result_path.resolve()),
        }

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
