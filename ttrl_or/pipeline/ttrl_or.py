from __future__ import annotations

import json
import uuid
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

from ttrl_or.config import PipelineConfig
from ttrl_or.mcts import FourStageMCTS
from ttrl_or.model.backend import PolicyBackend
from ttrl_or.prompts import DEFAULT_TEMPLATES, PromptBuilder
from ttrl_or.reward import TTRLRewardCalculator
from ttrl_or.types import OptimizationTask, RunTrace, STAGE_ORDER, Stage, StageTrace, TrainingSample, Trajectory


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
                "group_size": self.config.group_size,
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

            root = mcts.root()
            frontier = [root]
            stage_reports: dict[str, dict] = {}
            early_stop_candidate: Trajectory | None = None
            stopped_early_stage: Stage | None = None
            stopped_early_info: dict = {}

            # Four-layer loop: after each stage expansion, do one GRPO update.
            for stage in STAGE_ORDER:
                frontier_in = len(frontier)

                # Keep provisional consensus local to the current stage only.
                stage_rollout_archive: list[Trajectory] = []
                frontier, records, early_stop_info = mcts.expand_stage(
                    task=task,
                    stage=stage,
                    parent_nodes=frontier,
                    rollout_archive=stage_rollout_archive,
                )

                stage_samples = [
                    TrainingSample(
                        stage=stage,
                        prompt=record.prompt,
                        completion=record.completion,
                        reward=record.reward,
                        group_id=f"{stage.value}:{record.parent_id}",
                        trajectory_id=record.trajectory.trajectory_id,
                        metadata={"node_id": record.node_id},
                    )
                    for record in records
                ]
                grpo_report = self.backend.grpo_update(stage_samples, self.config.grpo, stage)

                stage_report = dict(grpo_report)
                stage_report["mcts_early_stop"] = early_stop_info is not None
                if early_stop_info is not None:
                    stage_report["mcts_early_stop_info"] = early_stop_info
                stage_reports[stage.value] = stage_report

                trace.stages.append(
                    StageTrace(
                        stage=stage.value,
                        num_frontier_in=frontier_in,
                        num_frontier_out=len(frontier),
                        stage_samples=len(stage_samples),
                        grpo_report=grpo_report,
                        mcts_early_stop=(early_stop_info is not None),
                        mcts_early_stop_info=(early_stop_info or {}),
                        expansions=[
                            {
                                "node_id": record.node_id,
                                "parent_id": record.parent_id,
                                "prior": record.prior,
                                "was_expanded": record.was_expanded,
                                "hit_reward_one": record.hit_reward_one,
                                "mean_reward": record.reward,
                                "child_q_before": record.child_q_before,
                                "child_q_after": record.child_q_after,
                                "child_visits_before": record.child_visits_before,
                                "child_visits_after": record.child_visits_after,
                                "parent_q_before": record.parent_q_before,
                                "parent_q_after": record.parent_q_after,
                                "parent_visits_before": record.parent_visits_before,
                                "parent_visits_after": record.parent_visits_after,
                                "rollout_details": record.rollout_details,
                                "completion_preview": record.completion[:300],
                            }
                            for record in records
                        ],
                    )
                )

                if early_stop_info is not None:
                    stopped_early_stage = stage
                    stopped_early_info = dict(early_stop_info)
                    target_tid = str(early_stop_info.get("trajectory_id", ""))
                    matched = [record.trajectory for record in records if record.trajectory.trajectory_id == target_tid]
                    if matched:
                        early_stop_candidate = matched[0]
                    elif records:
                        early_stop_candidate = max(records, key=lambda x: x.reward).trajectory
                    break

            if early_stop_candidate is not None:
                group_trajectories = rewarder.finalize_group([early_stop_candidate])
                best = self._pick_best_by_reward(group_trajectories, [])
                final_selection = {
                    "num_code_nodes": len(group_trajectories),
                    "num_selected": len(group_trajectories),
                    "selection_basis": "reward",
                    "stopped_early": True,
                    "stop_stage": stopped_early_stage.value if stopped_early_stage else "",
                    "stop_info": stopped_early_info,
                    "selected_nodes": [],
                }
            else:
                code_nodes = frontier
                ranked_code_nodes = sorted(code_nodes, key=lambda node: (node.q_value, node.visits), reverse=True)
                selected_nodes = ranked_code_nodes[: self.config.group_size]

                # Final answer is selected by finalized reward (not by Q value).
                group_trajectories = [node.to_partial_trajectory() for node in selected_nodes]
                group_trajectories = rewarder.finalize_group(group_trajectories)
                best = self._pick_best_by_reward(group_trajectories, selected_nodes)

                final_selection = {
                    "num_code_nodes": len(code_nodes),
                    "num_selected": len(selected_nodes),
                    "selection_basis": "reward",
                    "stopped_early": False,
                    "selected_nodes": [
                        {
                            "node_id": node.node_id,
                            "q_value": node.q_value,
                            "visits": node.visits,
                            "prior": node.prior,
                        }
                        for node in selected_nodes
                    ],
                }

            stage_reports["final_selection"] = final_selection

            mcts_stats = self._build_mcts_stats(trace)
            stage_reports["mcts_stats"] = mcts_stats

            trace.final_selection = final_selection
            trace.best_trajectory = self._best_trace(best)

            if self.config.save_logs:
                artifacts = self._save_trace_artifacts(trace, group_trajectories, best, mcts_stats)
                trace.artifacts = artifacts

            return TaskRunResult(
                task_id=task.task_id,
                stage_reports=stage_reports,
                trajectories=group_trajectories,
                best_trajectory=best,
                trace=trace,
            )
        finally:
            # Drop temporary LoRA/adapters before the next instance.
            self.backend.end_episode()

    def run_from_text(
        self,
        description: str,
        instance: dict | None = None,
        task_id: str | None = None,
    ) -> TaskRunResult:
        task = OptimizationTask(
            task_id=task_id or str(uuid.uuid4()),
            description=description,
            instance=instance or {},
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
    ) -> dict[str, str]:
        run_dir = Path(self.config.log_dir) / trace.task_id
        run_dir.mkdir(parents=True, exist_ok=True)

        summary_path = run_dir / "run_summary.json"
        stages_path = run_dir / "stage_events.jsonl"
        trajectories_path = run_dir / "final_trajectories.json"
        best_code_path = run_dir / "best_code.py"
        mcts_stats_path = run_dir / "mcts_stats.json"

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

        return {
            "run_dir": str(run_dir.resolve()),
            "run_summary": str(summary_path.resolve()),
            "stage_events": str(stages_path.resolve()),
            "mcts_stats": str(mcts_stats_path.resolve()),
            "final_trajectories": str(trajectories_path.resolve()),
            "best_code": str(best_code_path.resolve()),
        }

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
    def _build_mcts_stats(trace: RunTrace) -> dict:
        per_stage: dict[str, dict] = {}
        total_expansions = 0
        total_rollouts = 0

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
                "early_stop": bool(stage_trace.mcts_early_stop),
                "early_stop_info": dict(stage_trace.mcts_early_stop_info),
            }

        return {
            "total_expansions": total_expansions,
            "total_rollouts": total_rollouts,
            "per_stage": per_stage,
        }
