from __future__ import annotations

from dataclasses import dataclass

from ttrl_or.types import DEFAULT_STAGE_ORDER, SOLVERLLM_STAGE_ORDER, Stage

from .notice_prompts import CODE_NOTICE, OBJ_CON_NOTICE, SET_PARA_VAR_NOTICE, SYSTEM_INSTRUCTION


TAG_OUTPUT_NOTICE = """
Tag rules:
- Output only the required tagged block(s).
- Do not repeat the same tag multiple times.
- As soon as one required block is finished, close the tag and move on.
- Do not continue generating duplicate content after the first valid closing tag.
""".strip()


SPLIT_COMPLETION_NOTICE = """
Completion rules:
- The current node content is already fixed. Do NOT rewrite it.
- Output only the missing later-stage blocks, in the required order.
- Each missing tag may appear at most once.
- Stop after the first valid </python>.
""".strip()


SOLVERLLM_TYPE_HINT_NOTICE = """
!!! MANDATORY FORMAT RULES !!!
1. MUST return exactly one block: <Type>...</Type>
2. This stage is only for optimization type + modeling hint analysis.
3. Do NOT define sets, parameters, variables, objective, constraints, or code.
4. Output concise, reusable, downstream-helpful modeling guidance only.

Inside <Type>, organize the content as:
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
""".strip()


SOLVERLLM_SETS_NOTICE = """
!!! MANDATORY FORMAT RULES !!!
1. MUST contain only one <Sets>...</Sets> block.
2. Use bullet format: - set_name: description
3. Set names must be short, lowercase, and reusable downstream.
4. Enumerate elements only when they are explicitly small and given in the task.
5. Do NOT generate parameters, variables, objective, constraints, or code.
""".strip()


SOLVERLLM_PARAMETERS_NOTICE = """
!!! MANDATORY FORMAT RULES !!!
1. MUST contain only one <Parameters>...</Parameters> block.
2. Format:
   - Indexed: - param_index: description [unit][indexed by set] (data type): value_or_semantic_value
   - Global: - param: description [unit] (data type): value_or_semantic_value
3. Parameter names must align with the set names and the original entities.
4. Use exact values from the problem when available; otherwise use semantic values without inventing unsupported numbers.
5. Do NOT generate variables, objective, constraints, or code.
""".strip()


SOLVERLLM_VARIABLES_NOTICE = """
!!! MANDATORY FORMAT RULES !!!
1. MUST contain only one <Variables>...</Variables> block.
2. Format:
   - Indexed: - x_index: description (NONNEGATIVE CONTINUOUS / NONNEGATIVE INTEGER / BINARY)
   - Global: - x: description (domain)
3. Variable names must remain consistent with sets and parameters.
4. Domains must match the physical meaning of the decision.
5. Do NOT generate objective, constraints, or code.
""".strip()


SOLVERLLM_OBJECTIVE_NOTICE = """
!!! MANDATORY FORMAT RULES !!!
1. MUST contain only one <Objective>...</Objective> block.
2. Format: - objective_name: description: $LaTeX expression$
3. Use only previously defined symbols.
4. Write exactly one main objective.
5. Do NOT generate constraints or code.
""".strip()


SOLVERLLM_CONSTRAINTS_NOTICE = """
!!! MANDATORY FORMAT RULES !!!
1. MUST contain only one <Constraints>...</Constraints> block.
2. Format: - constraint_name: description: $LaTeX expression$ (type: Equality/Inequality)
3. Use only previously defined symbols.
4. Cover the complete core constraints needed by the task.
5. Do NOT generate code.
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
    Stage.SCHEMA: f"""
You are the first stage of a 4-stage OR modeling pipeline. I need youto help me generate the TYPE component for amathematical optimization formulation. You are also required to determine which category of classic optimization problem the given instance belongs to.

Task:
{{task_description}}


Please judge the type of this optimization problem and output the subtype of 
Return exactly one block: <Type> ... </Type>
Do not define sets, parameters, variables, objective, constraints, or code.

First, you need to determine whether the problem is linear. The rules are as follows:
* If the objective and constraints ofthe model involve non-linear terms (such as power functions, multiplication, non-linear probability models, etc.), then the problem is non-linear and returns directly to NLP.
* If the objective and constraint ofthe model are bothlinear, then the problem is linear. Furthermore,you need to determinewhether the problem is LPor MILP

Second, you should 
""".strip(),
    Stage.SET_PARAM_VAR: f"""
You are the second stage of a 4-stage OR modeling pipeline.

Task:
{{task_description}}

Here is the schema and modeling skill that have already been defined:
{{schema_skill_str}}

Please provide the sets, parameters, and variables needed for this optimization problem.
Return exactly these three blocks in order:
<Sets> ... </Sets>
<Parameters> ... </Parameters>
<Variables> ... </Variables>
Do not write the objective, constraints, or code.

{TAG_OUTPUT_NOTICE}

{SET_PARA_VAR_NOTICE}
""".strip(),
    Stage.OBJ_CONS: f"""
You are the third stage of a 4-stage OR modeling pipeline.

Task:
{{task_description}}

Here is the schema and modeling skill that have already been defined:
{{schema_skill_str}}

Here are the sets, parameters, and variables that have already been defined:
{{set_param_var_str}}

Please provide the objective and constraints needed for this optimization problem.
Return exactly these two blocks in order:
<Objective> ... </Objective>
<Constraints> ... </Constraints>
Do not generate code.

{TAG_OUTPUT_NOTICE}

{OBJ_CON_NOTICE}
""".strip(),
    Stage.CODE: f"""
You are the final stage of a 4-stage OR modeling pipeline.

Task:
{{task_description}}

Here is the schema and modeling skill that have already been defined:
{{schema_skill_str}}

Here are the sets, parameters, and variables that have already been defined:
{{set_param_var_str}}

Here are the objective and constraints that have already been defined:
{{obj_cons_str}}

Please provide the executable Gurobi Python code needed for this optimization problem.
Return exactly one block: <python> ... </python>
Translate the finalized model faithfully and do not redesign it.

{TAG_OUTPUT_NOTICE}

{CODE_NOTICE}
""".strip(),
}


DEFAULT_ROLLOUT_TEMPLATES: dict[Stage, str] = {
    Stage.SCHEMA: f"""
You are the first stage of a 4-stage OR modeling pipeline.

Task:
{{task_description}}

No earlier components have been fixed yet.

Please first provide the schema and modeling skill analysis for this optimization problem.
Then continue to finish the remaining stages in the same response and end with <python> ... </python>.
Output the current stage block first, followed by the later-stage blocks in order: {{remaining_stages}}.

{TAG_OUTPUT_NOTICE}

{SCHEMA_SKILL_NOTICE}

{CODE_NOTICE}
""".strip(),
    Stage.SET_PARAM_VAR: f"""
You are the second stage of a 4-stage OR modeling pipeline.

Task:
{{task_description}}

Here is the schema and modeling skill that have already been defined:
{{schema_skill_str}}

Please first provide the sets, parameters, and variables needed for this optimization problem.
Then continue to finish the remaining stages in the same response and end with <python> ... </python>.
Output the current stage block first, followed by the later-stage blocks in order: {{remaining_stages}}.

{TAG_OUTPUT_NOTICE}

{SET_PARA_VAR_NOTICE}

{CODE_NOTICE}
""".strip(),
    Stage.OBJ_CONS: f"""
You are the third stage of a 4-stage OR modeling pipeline.

Task:
{{task_description}}

Here is the schema and modeling skill that have already been defined:
{{schema_skill_str}}

Here are the sets, parameters, and variables that have already been defined:
{{set_param_var_str}}

Please first provide the objective and constraints needed for this optimization problem.
Then continue to finish the remaining stages in the same response and end with <python> ... </python>.
Output the current stage block first, followed by the later-stage blocks in order: {{remaining_stages}}.

{TAG_OUTPUT_NOTICE}

{OBJ_CON_NOTICE}

{CODE_NOTICE}
""".strip(),
    Stage.CODE: DEFAULT_TEMPLATES[Stage.CODE],
}


DEFAULT_COMPLETION_TEMPLATES: dict[Stage, str] = {
    Stage.SCHEMA: f"""
You are in the simulation/completion step of a 4-stage OR modeling pipeline.

Task:
{{task_description}}

Here is the schema and modeling skill that has already been fixed:
{{schema_skill_str}}

Please provide only the missing later-stage blocks in order: {{remaining_stages}}.
End with <python> ... </python> and do not rewrite earlier components.

{TAG_OUTPUT_NOTICE}

{SPLIT_COMPLETION_NOTICE}

{CODE_NOTICE}
""".strip(),
    Stage.SET_PARAM_VAR: f"""
You are in the simulation/completion step of a 4-stage OR modeling pipeline.

Task:
{{task_description}}

Here is the schema and modeling skill that has already been defined:
{{schema_skill_str}}

Here are the sets, parameters, and variables that have already been fixed:
{{set_param_var_str}}

Please provide only the missing later-stage blocks in order: {{remaining_stages}}.
End with <python> ... </python> and do not rewrite earlier components.

{TAG_OUTPUT_NOTICE}

{SPLIT_COMPLETION_NOTICE}

{CODE_NOTICE}
""".strip(),
    Stage.OBJ_CONS: f"""
You are in the simulation/completion step of a 4-stage OR modeling pipeline.

Task:
{{task_description}}

Here is the schema and modeling skill that has already been defined:
{{schema_skill_str}}

Here are the sets, parameters, and variables that have already been defined:
{{set_param_var_str}}

Here are the objective and constraints that have already been fixed:
{{obj_cons_str}}

Please provide only the missing later-stage blocks in order: {{remaining_stages}}.
End with <python> ... </python> and do not rewrite earlier components.

{TAG_OUTPUT_NOTICE}

{SPLIT_COMPLETION_NOTICE}

{CODE_NOTICE}
""".strip(),
    Stage.CODE: "",
}


SOLVERLLM_TEMPLATES: dict[Stage, str] = {
    Stage.TYPE_HINT: f"""
You are responsible for the type and modeling hint stage of a staged optimization formulation pipeline.

Problem description:
{{task_description}}

No earlier components have been fixed yet.

Please provide the type analysis and modeling hints needed for this optimization problem.
Return exactly one block: <Type> ... </Type>
Do not define sets, parameters, variables, objective, constraints, or code.

{TAG_OUTPUT_NOTICE}

{SOLVERLLM_TYPE_HINT_NOTICE}
""".strip(),
    Stage.SETS: f"""
You are responsible for the sets stage of a staged optimization formulation pipeline.

Problem description:
{{task_description}}

Here is the type of problem that has already been defined:
{{type_str}}

Here are some instructions for solving this problem:
{{instructions_str}}

Please provide the sets needed for this optimization problem.
Return exactly one block: <Sets> ... </Sets>
Do not generate parameters, variables, objective, constraints, or code.

{TAG_OUTPUT_NOTICE}

{SOLVERLLM_SETS_NOTICE}
""".strip(),
    Stage.PARAMETERS: f"""
You are responsible for the parameters stage of a staged optimization formulation pipeline.

Problem description:
{{task_description}}

Here is the type of problem that has already been defined:
{{type_str}}

Here are some instructions for solving this problem:
{{instructions_str}}

Here are the sets that have already been defined:
{{sets_str}}

Please provide the parameters needed for this optimization problem.
Return exactly one block: <Parameters> ... </Parameters>
Do not generate variables, objective, constraints, or code.

{TAG_OUTPUT_NOTICE}

{SOLVERLLM_PARAMETERS_NOTICE}
""".strip(),
    Stage.VARIABLES: f"""
You are responsible for the variables stage of a staged optimization formulation pipeline.

Problem description:
{{task_description}}

Here is the type of problem that has already been defined:
{{type_str}}

Here are some instructions for solving this problem:
{{instructions_str}}

Here are the sets that have already been defined:
{{sets_str}}

Here are the parameters that have already been defined:
{{parameters_str}}

Please provide the variables needed for this optimization problem.
Return exactly one block: <Variables> ... </Variables>
Do not generate objective, constraints, or code.

{TAG_OUTPUT_NOTICE}

{SOLVERLLM_VARIABLES_NOTICE}
""".strip(),
    Stage.OBJECTIVE: f"""
You are responsible for the objective stage of a staged optimization formulation pipeline.

Problem description:
{{task_description}}

Here is the type of problem that has already been defined:
{{type_str}}

Here are some instructions for solving this problem:
{{instructions_str}}

Here are the sets that have already been defined:
{{sets_str}}

Here are the parameters that have already been defined:
{{parameters_str}}

Here are the variables that have already been defined:
{{variables_str}}

Please provide the objective needed for this optimization problem.
Return exactly one block: <Objective> ... </Objective>
Do not generate constraints or code.

{TAG_OUTPUT_NOTICE}

{SOLVERLLM_OBJECTIVE_NOTICE}
""".strip(),
    Stage.CONSTRAINTS: f"""
You are responsible for the constraints stage of a staged optimization formulation pipeline.

Problem description:
{{task_description}}

Here is the type of problem that has already been defined:
{{type_str}}

Here are some instructions for solving this problem:
{{instructions_str}}

Here are the sets that have already been defined:
{{sets_str}}

Here are the parameters that have already been defined:
{{parameters_str}}

Here are the variables that have already been defined:
{{variables_str}}

Here is the objective that has already been defined:
{{objective_str}}

Please provide the constraints needed for this optimization problem.
Return exactly one block: <Constraints> ... </Constraints>
Do not generate code.

{TAG_OUTPUT_NOTICE}

{SOLVERLLM_CONSTRAINTS_NOTICE}
""".strip(),
    Stage.CODE: f"""
You are responsible for the final code generation stage of a staged optimization formulation pipeline.

Problem description:
{{task_description}}

Here is the type of problem that has already been defined:
{{type_str}}

Here are some instructions for solving this problem:
{{instructions_str}}

Here are the sets that have already been defined:
{{sets_str}}

Here are the parameters that have already been defined:
{{parameters_str}}

Here are the variables that have already been defined:
{{variables_str}}

Here is the objective that has already been defined:
{{objective_str}}

Here are the constraints that have already been defined:
{{constraints_str}}

Please provide the executable Gurobi Python code needed for this optimization problem.
Return exactly one block: <python> ... </python>
Translate the formulation faithfully and do not redesign it.

{TAG_OUTPUT_NOTICE}

{CODE_NOTICE}
""".strip(),
}


SOLVERLLM_ROLLOUT_TEMPLATES: dict[Stage, str] = {
    Stage.TYPE_HINT: SOLVERLLM_TEMPLATES[Stage.TYPE_HINT],
    Stage.SETS: SOLVERLLM_TEMPLATES[Stage.SETS],
    Stage.PARAMETERS: SOLVERLLM_TEMPLATES[Stage.PARAMETERS],
    Stage.VARIABLES: SOLVERLLM_TEMPLATES[Stage.VARIABLES],
    Stage.OBJECTIVE: SOLVERLLM_TEMPLATES[Stage.OBJECTIVE],
    Stage.CONSTRAINTS: SOLVERLLM_TEMPLATES[Stage.CONSTRAINTS],
    Stage.CODE: SOLVERLLM_TEMPLATES[Stage.CODE],
}


SOLVERLLM_COMPLETION_TEMPLATE = f"""
You are in the simulation/completion step of an MCTS-based staged optimization search.

Problem description:
{{task_description}}

Here is the type of problem that has already been defined:
{{type_str}}

Here are some instructions for solving this problem:
{{instructions_str}}

Here are the sets that have already been defined:
{{sets_str}}

Here are the parameters that have already been defined:
{{parameters_str}}

Here are the variables that have already been defined:
{{variables_str}}

Here is the objective that has already been defined:
{{objective_str}}

Here are the constraints that have already been defined:
{{constraints_str}}

The current node has already fixed the component for: {{current_stage}}
Please provide only the missing later-stage blocks in order: {{remaining_stages}}.
End with <python> ... </python> and do not rewrite earlier components.

{TAG_OUTPUT_NOTICE}

{SPLIT_COMPLETION_NOTICE}

{CODE_NOTICE}
""".strip()


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
