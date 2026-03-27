from __future__ import annotations

from dataclasses import dataclass, field

from ttrl_or.types import STAGE_ORDER, OptimizationTask, Stage, Trajectory


@dataclass(slots=True)
class PromptBuilder:
    templates: dict[Stage, str] = field(default_factory=dict)
    rollout_templates: dict[Stage, str] = field(default_factory=dict)

    def build(self, task: OptimizationTask, stage: Stage, trajectory: Trajectory | None = None) -> str:
        history = self._history_text(trajectory, stage)
        template = self.templates[stage]
        # Keep literal braces in prompt templates untouched.
        return (
            template.replace("{task_description}", task.description)
            .replace("{history}", history)
        )

    def build_rollout(self, task: OptimizationTask, stage: Stage, trajectory: Trajectory | None = None) -> str:
        base_prompt = self.build(task, stage, trajectory)
        rollout_suffix = (self.rollout_templates.get(stage, "") or "").strip()
        if not rollout_suffix:
            return base_prompt

        remaining = self._remaining_stages(stage)
        remaining_names = ", ".join(s.value for s in remaining) if remaining else "none"
        suffix = rollout_suffix.replace("{remaining_stages}", remaining_names)
        return f"{base_prompt}\n\n{suffix}".strip()

    @staticmethod
    def _remaining_stages(stage: Stage) -> list[Stage]:
        idx = STAGE_ORDER.index(stage)
        return list(STAGE_ORDER[idx + 1 :])

    @staticmethod
    def _history_text(trajectory: Trajectory | None, stage: Stage) -> str:
        if trajectory is None:
            return ""

        output_stage1 = trajectory.outputs.get(Stage.SCHEMA, "")
        output_stage2 = trajectory.outputs.get(Stage.SET_PARAM_VAR, "")
        output_stage3 = trajectory.outputs.get(Stage.OBJ_CONS, "")

        # Required history policy:
        # Stage 1: ""
        # Stage 2: output_stage1
        # Stage 3: output_stage1 + "\n\n" + output_stage2
        # Stage 4: output_stage1 + "\n\n" + output_stage2 + "\n\n" + output_stage3
        if stage == Stage.SCHEMA:
            return ""
        if stage == Stage.SET_PARAM_VAR:
            return output_stage1
        if stage == Stage.OBJ_CONS:
            return output_stage1 + "\n\n" + output_stage2
        if stage == Stage.CODE:
            return output_stage1 + "\n\n" + output_stage2 + "\n\n" + output_stage3
        return ""
