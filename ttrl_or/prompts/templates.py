from __future__ import annotations

from ttrl_or.types import Stage

from .notice_prompts import CODE_NOTICE, OBJ_CON_NOTICE, SCHEMA_SKILL_NOTICE, SET_PARA_VAR_NOTICE


def _append_notice(base: str, notice: str) -> str:
    base_text = (base or "").strip()
    notice_text = (notice or "").strip()
    if not notice_text:
        return base_text
    return f"{base_text}\n\n{notice_text}".strip()


SCHEMA_TEMPLATE = """
You are the first stage of a 4-stage OR modeling pipeline.

Task:
{task_description}

Stage 1 = schema + skill

Your job is to extract a stable OR modeling blueprint from the natural language task.
This stage is for structural modeling analysis only.

Do NOT define full sets, parameters, or variables.
Do NOT write the objective function.
Do NOT write formal constraints.
Do NOT generate code.
""".strip()

SET_PARAM_VAR_TEMPLATE = """
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

Under "### Sets Definition", output exactly this subheader:
- "## Set"

Under "### Parameters Definition", output exactly this subheader:
- "## Parameters"

Under "### Variables Definition", output exactly this subheader:
- "## Variables"

Your job in this stage is to convert the schema from previous stages into:
1. formal sets,
2. formal parameters,
3. formal decision variables.

Do NOT write the objective function.
Do NOT write the constraints.
Do NOT generate code.
""".strip()

OBJ_CONS_TEMPLATE = """
You are continuing a 4-stage OR modeling pipeline.

Task:
{task_description}

Previous stages:
{history}

Stage 3 = objective + constraints

Output must contain exactly these section headers:
- "### Objective Definition"
- "### Constraints Definition"

Under "### Objective Definition", output exactly this subheader:
- "## Objective"

Under "### Constraints Definition", output exactly this subheader:
- "## Constraints"

Your job in this stage is to construct:
1. the formal optimization objective,
2. the complete set of core constraints,

STRICTLY based on the sets, parameters, and variables defined in previous stages.

Do NOT redefine sets/parameters/variables unless absolutely necessary for consistency.
Do NOT generate code.
""".strip()

CODE_TEMPLATE = """
You are continuing a 4-stage OR modeling pipeline.

Task:
{task_description}

Previous stages:
{history}

Stage 4 = model2code

Your job is to translate the finalized optimization model into executable Gurobi Python code.

You must faithfully translate the existing model.
You are NOT allowed to correct, reinterpret, simplify, or redesign the model.
Even if the model seems imperfect, you must still translate it as given.
""".strip()

DEFAULT_TEMPLATES: dict[Stage, str] = {
    Stage.SCHEMA: _append_notice(SCHEMA_TEMPLATE, SCHEMA_SKILL_NOTICE),
    Stage.SET_PARAM_VAR: _append_notice(SET_PARAM_VAR_TEMPLATE, SET_PARA_VAR_NOTICE),
    Stage.OBJ_CONS: _append_notice(OBJ_CONS_TEMPLATE, OBJ_CON_NOTICE),
    Stage.CODE: _append_notice(CODE_TEMPLATE, CODE_NOTICE),
}


SCHEMA_ROLLOUT_TEMPLATE = """
### ROLLOUT_CONTINUATION
After completing the current-stage output above, continue and finish downstream stages in order: {remaining_stages}.
Do not rewrite previous-stage content.
Use exact tags below for machine parsing:
<ROLLOUT_STAGE_SET_PARAM_VAR>
...stage-2 content...
</ROLLOUT_STAGE_SET_PARAM_VAR>
<ROLLOUT_STAGE_OBJ_CONS>
...stage-3 content...
</ROLLOUT_STAGE_OBJ_CONS>
<ROLLOUT_STAGE_CODE>
<Gurobi_code>
...python code...
</Gurobi_code>
</ROLLOUT_STAGE_CODE>
""".strip()

SET_PARAM_VAR_ROLLOUT_TEMPLATE = """
### ROLLOUT_CONTINUATION
After completing the current-stage output above, continue and finish downstream stages in order: {remaining_stages}.
Use exact tags below for machine parsing:
<ROLLOUT_STAGE_OBJ_CONS>
...stage-3 content...
</ROLLOUT_STAGE_OBJ_CONS>
<ROLLOUT_STAGE_CODE>
<Gurobi_code>
...python code...
</Gurobi_code>
</ROLLOUT_STAGE_CODE>
""".strip()

OBJ_CONS_ROLLOUT_TEMPLATE = """
### ROLLOUT_CONTINUATION
After completing the current-stage output above, continue and finish downstream stage: {remaining_stages}.
Use exact tags below for machine parsing:
<ROLLOUT_STAGE_CODE>
<Gurobi_code>
...python code...
</Gurobi_code>
</ROLLOUT_STAGE_CODE>
""".strip()

DEFAULT_ROLLOUT_TEMPLATES: dict[Stage, str] = {
    Stage.SCHEMA: _append_notice(SCHEMA_ROLLOUT_TEMPLATE, CODE_NOTICE),
    Stage.SET_PARAM_VAR: _append_notice(SET_PARAM_VAR_ROLLOUT_TEMPLATE, CODE_NOTICE),
    Stage.OBJ_CONS: _append_notice(OBJ_CONS_ROLLOUT_TEMPLATE, CODE_NOTICE),
    Stage.CODE: "",
}
