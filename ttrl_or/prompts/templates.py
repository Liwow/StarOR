from __future__ import annotations

from ttrl_or.types import Stage


DEFAULT_TEMPLATES: dict[Stage, str] = {
    Stage.SCHEMA: """
You are an OR modeling assistant.
Given one natural-language optimization task, produce only a compact JSON object for Stage 1.

Task:
{task_description}

Stage 1 = schema + skill
Output JSON only, with exact top-level keys:
- schema
- skill

Required structure:
{{
  "schema": {{
    "entities": ["..."],
    "data_fields": ["..."],
    "assumptions": ["..."]
  }},
  "skill": {{
    "modeling_patterns": ["..."],
    "decomposition_plan": ["..."],
    "solver_tips": ["..."],
    "cautions": ["..."]
  }}
}}

Rules:
- no markdown fences
- no explanation text outside JSON
- keep each list concise but informative
""".strip(),
    Stage.SET_PARAM_VAR: """
You are continuing a 4-stage OR modeling pipeline.

Task:
{task_description}

Previous stages:
{history}

Stage 2 = set + parameter + var
Output must contain exactly these section headers:
- "### Sets Definition"
- "### Parameters Definition"
- "### Variables Definition"

Formatting rules:
1. Use short lowercase set/index names.
2. Parameters must align with defined sets/indexes.
3. Variables must include clear domain (NONNEGATIVE CONTINUOUS / NONNEGATIVE INTEGER / BINARY).
4. Include key numeric values from the task when available.
5. No chain-of-thought, output final structured result only.
""".strip(),
    Stage.OBJ_CONS: """
You are continuing a 4-stage OR modeling pipeline.

Task:
{task_description}

Previous stages:
{history}

Stage 3 = obj + con
Output must contain exactly these section headers:
- "### Objective Definition"
- "### Constraints Definition"

Formatting rules:
1. Objective must clearly state minimize/maximize and use symbols defined earlier.
2. Constraints should be complete (resource/bound/logic/non-negativity as needed).
3. Keep expressions linear if possible; if not, briefly note required linearization.
4. Do not introduce undefined symbols.
5. No chain-of-thought, output final result only.
""".strip(),
    Stage.CODE: """
You are continuing a 4-stage OR modeling pipeline.

Task:
{task_description}

Previous stages:
{history}

Stage 4 = model2code
Output only executable Python code.

Hard rules:
1. First non-empty line must be Python code, not natural language.
2. Define exactly: def solve(instance: dict) -> dict
3. Return at least: {{"objective": float, "status": str}}
4. No markdown fences, no XML tags, no extra commentary.
5. Keep code self-contained and runnable.
""".strip(),
}
