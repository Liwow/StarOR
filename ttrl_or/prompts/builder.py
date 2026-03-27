from __future__ import annotations

from dataclasses import dataclass, field

from ttrl_or.types import OptimizationTask, Stage, Trajectory


@dataclass(slots=True)
class PromptBuilder:
    templates: dict[Stage, str] = field(default_factory=dict)

    def build(self, task: OptimizationTask, stage: Stage, trajectory: Trajectory | None = None) -> str:
        history = self._history_text(trajectory, stage)
        template = self.templates[stage]
        # Keep literal braces in prompt templates untouched.
        return (
            template.replace("{task_description}", task.description)
            .replace("{history}", history)
        )

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
