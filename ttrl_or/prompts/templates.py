from __future__ import annotations

from dataclasses import dataclass

from ttrl_or.types import DEFAULT_STAGE_ORDER, SOLVERLLM_STAGE_ORDER, Stage

from .notice_prompts import CODE_NOTICE, SYSTEM_INSTRUCTION


def _append_notice(base: str, notice: str) -> str:
    base_text = (base or "").strip()
    notice_text = (notice or "").strip()
    if not notice_text:
        return base_text
    return f"{base_text}\n\n{notice_text}".strip()


TAG_OUTPUT_NOTICE = """
Tag rules:
- Output only the required tagged block(s).
- Do not repeat the same tag multiple times.
- As soon as one required block is finished, close the tag and move on.
- Do not continue generating duplicate content after the first valid closing tag.
""".strip()


SOLVERLLM_TYPE_HINT_NOTICE = """
!!! MANDATORY FORMAT RULES !!!
1. MUST return exactly one block: <stage_1>...</stage_1>
2. This stage is only for optimization type + modeling hint analysis.
3. Do NOT define sets, parameters, variables, objective, constraints, or code.
4. Output concise, reusable, downstream-helpful modeling guidance only.

Inside <stage_1>, organize the content as:
### Type Analysis:
- type: LP / MILP / NLP / MINLP
- subtype: classic OR family if identifiable

### Modeling Hints:
- key_decisions:
- likely_sets:
- likely_parameters:
- likely_variable_types:
- likely_constraint_families:

### Cautions:
- pitfalls:
- ambiguity_handling:

Quality check:
- Type matches the problem semantics.
- Hints are reusable by later stages.
- No downstream algebra or code appears here.
""".strip()


SOLVERLLM_SETS_NOTICE = """
!!! MANDATORY FORMAT RULES !!!
1. MUST contain ### Sets Definition: and ## Set:
2. Use bullet format: - set_name: description
3. Set names must be short, lowercase, and reusable downstream.
4. Enumerate elements only when they are explicitly small and given in the task.
5. Do NOT generate parameters, variables, objective, constraints, or code.

Quality check:
- Every decision-relevant object type has a set if needed.
- No redundant sets.
- Names remain stable for later stages.
""".strip()


SOLVERLLM_PARAMETERS_NOTICE = """
!!! MANDATORY FORMAT RULES !!!
1. MUST contain ### Parameters Definition: and ## Parameters:
2. Format:
   - Indexed: - param_index: description [unit][indexed by set] (data type): value_or_semantic_value
   - Global: - param: description [unit] (data type): value_or_semantic_value
3. Parameter names must align with the set names and the original entities.
4. Use exact values from the problem when available; otherwise use semantic values without inventing unsupported numbers.
5. Do NOT generate variables, objective, constraints, or code.

Quality check:
- All coefficients needed later are captured.
- Every indexed parameter references an existing set.
- No parameter is actually a decision.
""".strip()


SOLVERLLM_VARIABLES_NOTICE = """
!!! MANDATORY FORMAT RULES !!!
1. MUST contain ### Variables Definition: and ## Variables:
2. Format:
   - Indexed: - x_index: description (NONNEGATIVE CONTINUOUS / NONNEGATIVE INTEGER / BINARY)
   - Global: - x: description (domain)
3. Variable names must remain consistent with sets and parameters.
4. Domains must match the physical meaning of the decision.
5. Do NOT generate objective, constraints, or code.

Quality check:
- Every required decision is represented.
- Domains are valid.
- No unnecessary auxiliary variables unless clearly needed.
""".strip()


SOLVERLLM_OBJECTIVE_NOTICE = """
!!! MANDATORY FORMAT RULES !!!
1. MUST contain ### Objective Definition: and ## Objective:
2. Format: - objective_name: description: $LaTeX expression$
3. Use only previously defined symbols.
4. Write exactly one main objective.
5. Do NOT generate constraints or code.

Quality check:
- Objective direction is correct.
- Symbols all come from previous stages.
- No hard-coded coefficients when symbolic parameters already exist.
""".strip()


SOLVERLLM_CONSTRAINTS_NOTICE = """
!!! MANDATORY FORMAT RULES !!!
1. MUST contain ### Constraints Definition: and ## Constraints:
2. Format: - constraint_name: description: $LaTeX expression$ (type: Equality/Inequality)
3. Use only previously defined symbols.
4. Cover the complete core constraints needed by the task.
5. Do NOT generate code.

Quality check:
- Inequality directions match the task language.
- The core feasibility/resource/linking logic is complete.
- No contradictory or undefined expressions.
""".strip()


SPLIT_COMPLETION_NOTICE = """
Completion rules:
- The current node content is already fixed. Do NOT rewrite it.
- Output only the missing later-stage blocks, in the required order.
- Each missing tag may appear at most once.
- Stop after the first valid </Gurobi_code>.
""".strip()


@dataclass(slots=True)
class PromptProfile:
    name: str
    stage_order: tuple[Stage, ...]
    templates: dict[Stage, str]
    rollout_templates: dict[Stage, str]
    completion_templates: dict[Stage, str]
    system_instruction: str = SYSTEM_INSTRUCTION


DEFAULT_TEMPLATES: dict[Stage, str] = {
    Stage.SCHEMA: _append_notice(
        """
You are the first stage of a 4-stage OR modeling pipeline.

Task:
{task_description}

Current stage:
Stage 1 = schema + skill

Extract a stable OR modeling blueprint from the natural language task.
Do NOT define full sets, parameters, or variables.
Do NOT write the objective function.
Do NOT write formal constraints.
Do NOT generate code.

Output requirement:
- Return exactly one block: <stage_1> ... </stage_1>
""".strip(),
        TAG_OUTPUT_NOTICE,
    ),
    Stage.SET_PARAM_VAR: _append_notice(
        """
You are continuing a 4-stage OR modeling pipeline.

Task:
{task_description}

Previous stages:
{history}

Current stage:
Stage 2 = set + parameter + var

Construct sets, parameters, and decision variables based on previous stages.
Do NOT write the objective function.
Do NOT write the constraints.
Do NOT generate code.

Output requirement:
- Return exactly one block: <stage_2> ... </stage_2>
""".strip(),
        TAG_OUTPUT_NOTICE,
    ),
    Stage.OBJ_CONS: _append_notice(
        """
You are continuing a 4-stage OR modeling pipeline.

Task:
{task_description}

Previous stages:
{history}

Current stage:
Stage 3 = objective + constraints

Construct the formal objective and the complete core constraints strictly based on previous stages.
Do NOT redefine previous components unless absolutely necessary.
Do NOT generate code.

Output requirement:
- Return exactly one block: <stage_3> ... </stage_3>
""".strip(),
        TAG_OUTPUT_NOTICE,
    ),
    Stage.CODE: _append_notice(
        """
You are in the final code-generation stage of a 4-stage OR modeling pipeline.

Task:
{task_description}

Previous stages:
{history}

Translate the finalized optimization model into executable Gurobi Python code.
Do not reinterpret or redesign the model.

Output requirement:
- Return exactly one block: <Gurobi_code> ... </Gurobi_code>
""".strip(),
        f"{TAG_OUTPUT_NOTICE}\n\n{CODE_NOTICE}",
    ),
}


DEFAULT_ROLLOUT_TEMPLATES: dict[Stage, str] = {
    Stage.SCHEMA: _append_notice(
        """
After finishing the current stage block, continue the remaining stages in one response.
You must output the current stage block first, then complete the remaining stages, and end with <Gurobi_code> ... </Gurobi_code>.
Remaining stages: {remaining_stages}
""".strip(),
        f"{TAG_OUTPUT_NOTICE}\n\n{CODE_NOTICE}",
    ),
    Stage.SET_PARAM_VAR: _append_notice(
        """
After finishing the current stage block, continue the remaining stages in one response.
You must output the current stage block first, then complete the remaining stages, and end with <Gurobi_code> ... </Gurobi_code>.
Remaining stages: {remaining_stages}
""".strip(),
        f"{TAG_OUTPUT_NOTICE}\n\n{CODE_NOTICE}",
    ),
    Stage.OBJ_CONS: _append_notice(
        """
After finishing the current stage block, continue the remaining stages in one response.
You must output the current stage block first, then complete the remaining stages, and end with <Gurobi_code> ... </Gurobi_code>.
Remaining stages: {remaining_stages}
""".strip(),
        f"{TAG_OUTPUT_NOTICE}\n\n{CODE_NOTICE}",
    ),
    Stage.CODE: "",
}


DEFAULT_COMPLETION_TEMPLATES: dict[Stage, str] = {}


SOLVERLLM_TEMPLATES: dict[Stage, str] = {
    Stage.TYPE_HINT: _append_notice(
        """
You are generating the first component of a staged optimization formulation.

Problem description:
{task_description}

Current stage:
Type + Modeling Hint Analysis

Analyze the problem type and extract downstream-useful modeling hints only.
Do not move into later-stage symbolic formulation.

Return exactly one block: <stage_1> ... </stage_1>
""".strip(),
        f"{TAG_OUTPUT_NOTICE}\n\n{SOLVERLLM_TYPE_HINT_NOTICE}",
    ),
    Stage.SETS: _append_notice(
        """
You are generating the SETS component for a staged optimization formulation.

Problem description:
{task_description}

Previously defined components:
{history}

Current stage:
Sets Construction

Return exactly one block: <stage_2> ... </stage_2>
""".strip(),
        f"{TAG_OUTPUT_NOTICE}\n\n{SOLVERLLM_SETS_NOTICE}",
    ),
    Stage.PARAMETERS: _append_notice(
        """
You are generating the PARAMETERS component for a staged optimization formulation.

Problem description:
{task_description}

Previously defined components:
{history}

Current stage:
Parameters Construction

Return exactly one block: <stage_3> ... </stage_3>
""".strip(),
        f"{TAG_OUTPUT_NOTICE}\n\n{SOLVERLLM_PARAMETERS_NOTICE}",
    ),
    Stage.VARIABLES: _append_notice(
        """
You are generating the VARIABLES component for a staged optimization formulation.

Problem description:
{task_description}

Previously defined components:
{history}

Current stage:
Variables Construction

Return exactly one block: <stage_4> ... </stage_4>
""".strip(),
        f"{TAG_OUTPUT_NOTICE}\n\n{SOLVERLLM_VARIABLES_NOTICE}",
    ),
    Stage.OBJECTIVE: _append_notice(
        """
You are generating the OBJECTIVE component for a staged optimization formulation.

Problem description:
{task_description}

Previously defined components:
{history}

Current stage:
Objective Construction

Return exactly one block: <stage_5> ... </stage_5>
""".strip(),
        f"{TAG_OUTPUT_NOTICE}\n\n{SOLVERLLM_OBJECTIVE_NOTICE}",
    ),
    Stage.CONSTRAINTS: _append_notice(
        """
You are generating the CONSTRAINTS component for a staged optimization formulation.

Problem description:
{task_description}

Previously defined components:
{history}

Current stage:
Constraints Construction

Return exactly one block: <stage_6> ... </stage_6>
""".strip(),
        f"{TAG_OUTPUT_NOTICE}\n\n{SOLVERLLM_CONSTRAINTS_NOTICE}",
    ),
    Stage.CODE: _append_notice(
        """
You are generating solver-ready Gurobi Python code from a completed mathematical formulation.

Problem description:
{task_description}

Previously defined components:
{history}

Current stage:
Code Generation

Return exactly one block: <Gurobi_code> ... </Gurobi_code>
Translate the formulation faithfully into executable Gurobi code.
""".strip(),
        f"{TAG_OUTPUT_NOTICE}\n\n{CODE_NOTICE}",
    ),
}


SOLVERLLM_ROLLOUT_TEMPLATES: dict[Stage, str] = {
    Stage.TYPE_HINT: "",
    Stage.SETS: "",
    Stage.PARAMETERS: "",
    Stage.VARIABLES: "",
    Stage.OBJECTIVE: "",
    Stage.CONSTRAINTS: "",
    Stage.CODE: "",
}


SOLVERLLM_COMPLETION_TEMPLATE = _append_notice(
    """
You are in the simulation/completion step of an MCTS-based staged optimization search.

Problem description:
{task_description}

Fixed components so far:
{history}

The current node has already fixed the component for: {current_stage}
Now complete only the remaining components in order: {remaining_stages}

Return only the missing later-stage blocks in order and end with <Gurobi_code> ... </Gurobi_code>.
Do not rewrite or modify already fixed earlier components.
""".strip(),
    f"{TAG_OUTPUT_NOTICE}\n\n{SPLIT_COMPLETION_NOTICE}\n\n{CODE_NOTICE}",
)


SOLVERLLM_COMPLETION_TEMPLATES: dict[Stage, str] = {
    Stage.TYPE_HINT: SOLVERLLM_COMPLETION_TEMPLATE,
    Stage.SETS: SOLVERLLM_COMPLETION_TEMPLATE,
    Stage.PARAMETERS: SOLVERLLM_COMPLETION_TEMPLATE,
    Stage.VARIABLES: SOLVERLLM_COMPLETION_TEMPLATE,
    Stage.OBJECTIVE: SOLVERLLM_COMPLETION_TEMPLATE,
    Stage.CONSTRAINTS: SOLVERLLM_COMPLETION_TEMPLATE,
    Stage.CODE: "",
}


def get_prompt_profile(solverllm_compare_mode: bool = False) -> PromptProfile:
    if bool(solverllm_compare_mode):
        return PromptProfile(
            name="solverllm_like",
            stage_order=SOLVERLLM_STAGE_ORDER,
            templates=SOLVERLLM_TEMPLATES,
            rollout_templates=SOLVERLLM_ROLLOUT_TEMPLATES,
            completion_templates=SOLVERLLM_COMPLETION_TEMPLATES,
            system_instruction=SYSTEM_INSTRUCTION,
        )
    return PromptProfile(
        name="default4",
        stage_order=DEFAULT_STAGE_ORDER,
        templates=DEFAULT_TEMPLATES,
        rollout_templates=DEFAULT_ROLLOUT_TEMPLATES,
        completion_templates=DEFAULT_COMPLETION_TEMPLATES,
        system_instruction=SYSTEM_INSTRUCTION,
    )
