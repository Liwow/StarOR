from __future__ import annotations

from ttrl_or.types import Stage

from .notice_prompts import CODE_NOTICE, OBJ_CON_NOTICE, SCHEMA_SKILL_NOTICE, SET_PARA_VAR_NOTICE


def _append_notice(base: str, notice: str) -> str:
    base_text = (base or "").strip()
    notice_text = (notice or "").strip()
    if not notice_text:
        return base_text
    return f"{base_text}\n\n{notice_text}".strip()


TAG_RULES_COMMON = """
Output tag protocol (strict):
- Stage 1 output must be only inside <stage_1>...</stage_1>
- Stage 2 output must be only inside <stage_2>...</stage_2>
- Stage 3 output must be only inside <stage_3>...</stage_3>
- Stage 4 code must be only inside <Gurobi_code>...</Gurobi_code>
Do not output content outside the required tag for the current stage.
""".strip()


SCHEMA_TEMPLATE = """
You are the first stage of a 4-stage OR modeling pipeline.
In First Stage (Schema and Modeling Skill Analysis), your goal is to provide a conceptual abstraction and logical blueprint of the optimization task.

Task:
{task_description}

Stage 1 = schema + skill

Your job is to extract a stable OR modeling blueprint from the natural language task.
This stage is for structural modeling analysis only.

Do NOT define full sets, parameters, or variables.
Do NOT write the objective function.
Do NOT write formal constraints.
Do NOT generate code.

Output requirement:
- Return exactly one block: <stage_1> ... </stage_1>
""".strip()

SET_PARAM_VAR_TEMPLATE = """
You are continuing a 4-stage OR modeling pipeline, and you are in second stage.
In Second Stage (Set, Parameters, and Variables Construction), your goal is to construct Set, Parameters, and Variables based on the content from previous stages.

Task:
{task_description}

Previous stages:
{history}

Stage 2 = set + parameter + var

Your job in this stage is to convert the schema from previous stages into:
1. formal sets,
2. formal parameters,
3. formal decision variables.

Do NOT write the objective function.
Do NOT write the constraints.
Do NOT generate code.

Output requirement:
- Return exactly one block: <stage_2> ... </stage_2>
""".strip()

OBJ_CONS_TEMPLATE = """
You are continuing a 4-stage OR modeling pipeline, and you are in third stage.
In Third Stage (Objective and Constraints Modeling), your goal is to construct Objective and Constraints based on the content from previous stages.

Task:
{task_description}

Previous stages:
{history}

Stage 3 = objective + constraints

Your job in this stage is to construct:
1. the formal optimization objective,
2. the complete set of core constraints,
STRICTLY based on the sets, parameters, and variables defined in previous stages.

Do NOT redefine sets/parameters/variables unless absolutely necessary for consistency.
Do NOT generate code.

Output requirement:
- Return exactly one block: <stage_3> ... </stage_3>
""".strip()

CODE_TEMPLATE = """
You are continuing a 4-stage OR modeling pipeline, and you are in final stage.
In Final Stage, your goal is to write Python code with gurobi based on the content from previous stages.

Task:
{task_description}

Previous stages:
{history}

Stage 4 = model2code

Your job is to translate the finalized optimization model into executable Gurobi Python code.

You must faithfully translate the existing model.
You are NOT allowed to correct, reinterpret, simplify, or redesign the model.
Even if the model seems imperfect, you must still translate it as given.

Output requirement:
- Return exactly one block: <Gurobi_code> ... </Gurobi_code>
""".strip()


DEFAULT_TEMPLATES: dict[Stage, str] = {
    Stage.SCHEMA: _append_notice(_append_notice(SCHEMA_TEMPLATE, TAG_RULES_COMMON), SCHEMA_SKILL_NOTICE),
    Stage.SET_PARAM_VAR: _append_notice(_append_notice(SET_PARAM_VAR_TEMPLATE, TAG_RULES_COMMON), SET_PARA_VAR_NOTICE),
    Stage.OBJ_CONS: _append_notice(_append_notice(OBJ_CONS_TEMPLATE, TAG_RULES_COMMON), OBJ_CON_NOTICE),
    Stage.CODE: _append_notice(_append_notice(CODE_TEMPLATE, TAG_RULES_COMMON), CODE_NOTICE),
}


ROLLOUT_APPEND_TEMPLATE = """
After finishing the current stage, continue to complete the remaining stages in order: {remaining_stages}.
You must still follow strict tags:
<stage_1>...</stage_1>
<stage_2>...</stage_2>
<stage_3>...</stage_3>
<Gurobi_code>...</Gurobi_code>
Final code must be in <Gurobi_code> only.
""".strip()


DEFAULT_ROLLOUT_TEMPLATES: dict[Stage, str] = {
    Stage.SCHEMA: _append_notice(ROLLOUT_APPEND_TEMPLATE, CODE_NOTICE),
    Stage.SET_PARAM_VAR: _append_notice(ROLLOUT_APPEND_TEMPLATE, CODE_NOTICE),
    Stage.OBJ_CONS: _append_notice(ROLLOUT_APPEND_TEMPLATE, CODE_NOTICE),
    Stage.CODE: "",
}
