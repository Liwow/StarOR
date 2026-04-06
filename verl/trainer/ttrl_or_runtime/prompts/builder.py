from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from verl.trainer.ttrl_or_runtime.types import OptimizationTask, Stage, Trajectory

from .notice_prompts import SYSTEM_INSTRUCTION


@dataclass(slots=True)
class PromptBuilder:
    templates: dict[Stage, str] = field(default_factory=dict)
    rollout_templates: dict[Stage, str] = field(default_factory=dict)
    completion_templates: dict[Stage, str] = field(default_factory=dict)
    stage_order: tuple[Stage, ...] = field(default_factory=tuple)
    system_instruction: str = SYSTEM_INSTRUCTION

    def build(self, task: OptimizationTask, stage: Stage, trajectory: Trajectory | None = None) -> str:
        messages = self.build_messages(task, stage, trajectory, prompt_kind="stage")
        return self._messages_to_text(messages)

    def build_rollout(self, task: OptimizationTask, stage: Stage, trajectory: Trajectory | None = None) -> str:
        messages = self.build_messages(task, stage, trajectory, prompt_kind="rollout")
        return self._messages_to_text(messages)

    def build_completion(self, task: OptimizationTask, stage: Stage, trajectory: Trajectory | None = None) -> str:
        messages = self.build_messages(task, stage, trajectory, prompt_kind="completion")
        return self._messages_to_text(messages)

    def build_messages(
        self,
        task: OptimizationTask,
        stage: Stage,
        trajectory: Trajectory | None = None,
        prompt_kind: str = "stage",
    ) -> list[dict[str, str]]:
        template = self._resolve_template(stage, prompt_kind)
        if not template:
            return []
        body = self._render_template(template, task, stage, trajectory, prompt_kind)
        messages: list[dict[str, str]] = []
        system_text = (self.system_instruction or "").strip()
        if system_text:
            messages.append({"role": "system", "content": system_text})
        messages.append({"role": "user", "content": body})
        return messages

    def _resolve_template(self, stage: Stage, prompt_kind: str) -> str:
        if prompt_kind == "rollout":
            return (self.rollout_templates.get(stage) or self.templates.get(stage) or "").strip()
        if prompt_kind == "completion":
            return (self.completion_templates.get(stage) or "").strip()
        return (self.templates.get(stage) or "").strip()

    def _render_template(
        self,
        template: str,
        task: OptimizationTask,
        stage: Stage,
        trajectory: Trajectory | None,
        prompt_kind: str,
    ) -> str:
        include_current = prompt_kind == "completion"
        history_components = self._history_components(trajectory, include_current=include_current, current_stage=stage)
        remaining = self._remaining_stages(stage)
        remaining_names = " -> ".join(s.value for s in remaining) if remaining else "none"

        replacements: dict[str, str] = {
            "task_description": task.description,
            "current_stage": stage.value,
            "remaining_stages": remaining_names,
        }
        replacements.update(history_components)

        body = template
        for key, value in replacements.items():
            body = body.replace("{" + key + "}", value)
        return body.strip()

    def _remaining_stages(self, stage: Stage) -> list[Stage]:
        idx = self.stage_order.index(stage)
        return list(self.stage_order[idx + 1 :])

    def _history_components(
        self,
        trajectory: Trajectory | None,
        *,
        include_current: bool,
        current_stage: Stage,
    ) -> dict[str, str]:
        components = {
            "schema_skill_str": "",
            "set_param_var_str": "",
            "obj_cons_str": "",
            "type_str": "",
            "instructions_str": "",
            "sets_str": "",
            "parameters_str": "",
            "variables_str": "",
            "objective_str": "",
            "constraints_str": "",
        }
        if trajectory is None:
            return components

        for stage, text in trajectory.outputs.items():
            if not include_current and stage == current_stage:
                continue
            clean = (text or "").strip()
            if not clean:
                continue
            if stage == Stage.TYPE_HINT:
                type_str, instructions_str = self._split_type_hint(clean)
                if "type_str" in components:
                    components["type_str"] = type_str
                if "instructions_str" in components:
                    components["instructions_str"] = instructions_str
                continue
            if stage == Stage.SCHEMA and "schema_skill_str" in components:
                components["schema_skill_str"] = clean
                continue
            if stage == Stage.SET_PARAM_VAR and "set_param_var_str" in components:
                components["set_param_var_str"] = clean
                continue
            if stage == Stage.OBJ_CONS and "obj_cons_str" in components:
                components["obj_cons_str"] = clean
                continue
            if stage == Stage.SETS and "sets_str" in components:
                components["sets_str"] = clean
                continue
            if stage == Stage.PARAMETERS and "parameters_str" in components:
                components["parameters_str"] = clean
                continue
            if stage == Stage.VARIABLES and "variables_str" in components:
                components["variables_str"] = clean
                continue
            if stage == Stage.OBJECTIVE and "objective_str" in components:
                components["objective_str"] = clean
                continue
            if stage == Stage.CONSTRAINTS and "constraints_str" in components:
                components["constraints_str"] = clean
                continue
        return components

    @staticmethod
    def _split_type_hint(text: str) -> tuple[str, str]:
        type_match = re.search(
            r"###\s*Type Analysis:\s*(.*?)(?=###\s*Modeling Hints:|$)",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        hints_match = re.search(
            r"###\s*Modeling Hints:\s*(.*?)(?=###\s*Cautions:|$)",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        cautions_match = re.search(
            r"###\s*Cautions:\s*(.*)$",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )

        type_str = (type_match.group(1) if type_match else text).strip()
        instructions_parts: list[str] = []
        if hints_match:
            instructions_parts.append(hints_match.group(1).strip())
        if cautions_match:
            instructions_parts.append(cautions_match.group(1).strip())
        instructions_str = "\n\n".join(part for part in instructions_parts if part).strip()
        if not instructions_str:
            instructions_str = text.strip()
        return type_str, instructions_str

    @staticmethod
    def _messages_to_text(messages: list[dict[str, str]]) -> str:
        if not messages:
            return ""
        chunks: list[str] = []
        for message in messages:
            role = str(message.get("role", "user") or "user").strip().upper()
            content = str(message.get("content", "") or "").strip()
            if not content:
                continue
            chunks.append(f"[{role}]\n{content}")
        return "\n\n".join(chunks).strip()

