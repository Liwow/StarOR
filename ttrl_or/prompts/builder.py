from __future__ import annotations

from dataclasses import dataclass, field

from ttrl_or.types import OptimizationTask, Stage, Trajectory

from .notice_prompts import SYSTEM_INSTRUCTION


@dataclass(slots=True)
class PromptBuilder:
    templates: dict[Stage, str] = field(default_factory=dict)
    rollout_templates: dict[Stage, str] = field(default_factory=dict)
    completion_templates: dict[Stage, str] = field(default_factory=dict)
    stage_order: tuple[Stage, ...] = field(default_factory=tuple)
    system_instruction: str = SYSTEM_INSTRUCTION

    def build(self, task: OptimizationTask, stage: Stage, trajectory: Trajectory | None = None) -> str:
        history = self._history_text(trajectory, stage)
        template = self.templates[stage]
        body = (
            template.replace('{task_description}', task.description)
            .replace('{history}', history)
        )
        system_text = (self.system_instruction or '').strip()
        if not system_text:
            return body
        return f"[SYSTEM]\n{system_text}\n\n[USER]\n{body}".strip()

    def build_rollout(self, task: OptimizationTask, stage: Stage, trajectory: Trajectory | None = None) -> str:
        base_prompt = self.build(task, stage, trajectory)
        rollout_suffix = (self.rollout_templates.get(stage, '') or '').strip()
        if not rollout_suffix:
            return base_prompt
        remaining = self._remaining_stages(stage)
        remaining_names = ' --> '.join(s.value for s in remaining) if remaining else 'none'
        suffix = rollout_suffix.replace('{remaining_stages}', remaining_names)
        return f"{base_prompt}\n\n{suffix}".strip()

    def build_completion(self, task: OptimizationTask, stage: Stage, trajectory: Trajectory | None = None) -> str:
        template = (self.completion_templates.get(stage, '') or '').strip()
        if not template:
            return ''
        history = self._history_text(trajectory, None)
        remaining = self._remaining_stages(stage)
        remaining_names = ' --> '.join(s.value for s in remaining) if remaining else 'none'
        body = (
            template.replace('{task_description}', task.description)
            .replace('{history}', history)
            .replace('{current_stage}', stage.value)
            .replace('{remaining_stages}', remaining_names)
        )
        system_text = (self.system_instruction or '').strip()
        if not system_text:
            return body
        return f"[SYSTEM]\n{system_text}\n\n[USER]\n{body}".strip()

    def _remaining_stages(self, stage: Stage) -> list[Stage]:
        idx = self.stage_order.index(stage)
        return list(self.stage_order[idx + 1 :])

    def _history_text(self, trajectory: Trajectory | None, stage: Stage | None) -> str:
        if trajectory is None:
            return ''
        chunks: list[str] = []
        for s in self.stage_order:
            if stage is not None and s == stage:
                break
            text = (trajectory.outputs.get(s, '') or '').strip()
            if text:
                chunks.append(text)
        return '\n\n'.join(chunks)
