from __future__ import annotations

from ttrl_or.types import Stage


DEFAULT_TEMPLATES: dict[Stage, str] = {
    Stage.SCHEMA: """
You are given an optimization problem in natural language.
Task:
{task_description}

Step 1 (SCHEMA + SKILL): output a compact JSON object that includes both structured schema and modeling skill hints.
Requirements:
- only JSON
- include keys: schema, skill
- schema must include: entities, data_fields, assumptions
- skill must include: modeling_patterns, decomposition_plan, solver_tips and cautions (a list of likely modeling pitfalls for this instance) 
""".strip(),
    Stage.SET_PARAM_VAR: """
You are modeling an optimization task.
Task:
{task_description}

Previous stages:
{history}

Step 2 (SET/PARAM/VAR): define sets, parameters, and decision variables in structured markdown.
Include sections:
- Sets
- Parameters
- Variables
""".strip(),
    Stage.OBJ_CONS: """
You are modeling an optimization task.
Task:
{task_description}

Previous stages:
{history}

Step 3 (Objective + Constraints): write a concise math-style optimization model.
Include:
- Objective
- Constraints
- Optional notes for linearization
""".strip(),
    Stage.CODE: """
You are solving an optimization task by code.
Task:
{task_description}

Previous stages:
{history}

Step 4 (CODE): output Python code only.
Rules:
- define function solve(instance: dict) -> dict
- return at least {{"objective": float, "status": str}}
- no markdown fences
""".strip(),
}
