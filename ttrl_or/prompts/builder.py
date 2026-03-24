from __future__ import annotations

from dataclasses import dataclass, field

from ttrl_or.types import OptimizationTask, Stage, Trajectory


@dataclass(slots=True)
class PromptBuilder:
    templates: dict[Stage, str] = field(default_factory=dict)

    def build(self, task: OptimizationTask, stage: Stage, trajectory: Trajectory | None = None) -> str:
        history = self._history_text(trajectory)
        template = self.templates[stage]
        return template.format(task_description=task.description, history=history)

    @staticmethod
    def _history_text(trajectory: Trajectory | None) -> str:
        if trajectory is None:
            return "<EMPTY>"

        chunks: list[str] = []
        for stage in (Stage.SCHEMA, Stage.SET_PARAM_VAR, Stage.OBJ_CONS):
            if stage in trajectory.outputs:
                chunks.append(f"[{stage.value}]\\n{trajectory.outputs[stage]}")
        return "\\n\\n".join(chunks) if chunks else "<EMPTY>"
