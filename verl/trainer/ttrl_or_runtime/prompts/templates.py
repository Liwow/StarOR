from __future__ import annotations

from dataclasses import dataclass

from verl.trainer.ttrl_or_runtime.types import DEFAULT_STAGE_ORDER, SOLVERLLM_STAGE_ORDER, Stage

from .notice_prompts import CODE_NOTICE, OBJ_CON_NOTICE, PARA_VAR_NOTICE, SYSTEM_INSTRUCTION, TYPE_SET_NOTICE

is_think = True

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
- subtype: classical OR family if identifiable

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
You are a professional optimization problem analyst, proficient in extracting key elements from optimization problems described in natural language.
I need you to help me generate the TYPE and SETS components for a mathematical optimization formulation.

Here is the specific description of the optimization problem:
{{task_description}}

Please provide the type analysis and the core sets needed for this optimization problem.
Return exactly these two blocks in order:
<Type> ... </Type>
<Sets> ... </Sets>
Do not define parameters, variables, objective, constraints, or code.
The required order from this point is:
1. <Type>
2. <Sets>

{TYPE_SET_NOTICE}
""".strip(),
    Stage.SET_PARAM_VAR: f"""
You are a professional optimization problem analyst, proficient in extracting key elements from optimization problems described in natural language.
I need you to help me generate the PARAMETERS and VARIABLES components for a mathematical optimization formulation.

Here is the specific description of the optimization problem:
{{task_description}}

Here are the type analysis and sets that have already been defined:
{{schema_skill_str}}

Please provide the parameters and variables needed for this optimization problem.
Return exactly these two blocks in order:
<Parameters> ... </Parameters>
<Variables> ... </Variables>
Do not write the objective, constraints, or code.
The required order from this point is:
1. <Parameters>
2. <Variables>

{PARA_VAR_NOTICE}
""".strip(),
    Stage.OBJ_CONS: f"""
You are a professional optimization problem analyst, proficient in extracting key elements from optimization problems described in natural language.
I need you to help me generate the Objectives and Constraints components for a mathematical optimization formulation.

Here is the specific description of the optimization problem:
{{task_description}}

Here are the type analysis and sets that have already been defined:
{{schema_skill_str}}

Here are the parameters and variables that have already been defined:
{{set_param_var_str}}

Please provide the objective and constraints needed for this optimization problem.
Return exactly these two blocks in order:
<Objective> ... </Objective>
<Constraints> ... </Constraints>
Do not generate code.
The required order from this point is:
1. <Objective>
2. <Constraints>


{OBJ_CON_NOTICE}
""".strip(),
    Stage.CODE: f"""
You are an optimization expert. You should solve the optimization problem and only Provide the corresponding Gurobi Python code to implement the model within <python> and </python>

Here is the specific description of the optimization problem:
{{task_description}}

Here are the type analysis and sets that have already been defined:
{{schema_skill_str}}

Here are the parameters and variables that have already been defined:
{{set_param_var_str}}

Here are the objective and constraints that have already been defined:
{{obj_cons_str}}

Please provide the executable Gurobi Python code needed for this optimization problem within <python>.
Translate the finalized model faithfully and do not redesign it.
The required order from this point is:
1. <python>

{CODE_NOTICE}
""".strip(),
}

if is_think:
    DEFAULT_ROLLOUT_TEMPLATES: dict[Stage, str] = {
        Stage.SCHEMA: f"""
You are a professional optimization problem analyst, proficient in extracting key elements from optimization problems described in natural language.

Here is the specific description of the optimization problem:
{{task_description}}

1. You should first think step by step in <thought> and carefully analyze the problem structure.

2. Then you should generate the TYPE and SETS components first and in order:
<Type> ... </Type>
<Sets> ... </Sets>

3. After finishing the current stage, continue the formulation and finally provide the corresponding Gurobi Python code within <python> ... </python>.

The required order from this point is:
1. <thought>
2. <Type>
3. <Sets>
4. <python>

{TYPE_SET_NOTICE}

{CODE_NOTICE}
**NOTE**: You should output the problem type in <Type> and the Sets in <Sets>, and then MUST output the complete optimization problem modeling code in <python>.
IMPORTANT: Do not skip any steps. AFTER the TYPE and SET sections are fully completed, you must output the full modeling and solving Python code inside <python>. Proceed with the task now.
""".strip(),
        Stage.SET_PARAM_VAR: f"""
You are a professional optimization problem analyst, proficient in extracting key elements from optimization problems described in natural language.

Here is the specific description of the optimization problem:
{{task_description}}

Here are the type analysis and sets that have already been defined:
{{schema_skill_str}}

1. You should first think step by step in <thought> and carefully analyze the parameters and decisions needed by the formulation.

2. Then you should generate the PARAMETERS and VARIABLES components first and in order:
<Parameters> ... </Parameters>
<Variables> ... </Variables>

3. After finishing the current stage, continue the formulation and finally provide the corresponding Gurobi Python code within <python> ... </python>.

The required order from this point is:
1. <thought>
2. <Parameters>
3. <Variables>
4. <python>

{PARA_VAR_NOTICE}

{CODE_NOTICE}
**NOTE**: You should output the problem Parameters in <Parameters> and the problem Variables in <Variables>, and then MUST output the complete optimization problem modeling code in <python>.
IMPORTANT: Do not skip any steps. AFTER the Parameters and Variables sections are fully completed, you must output the full modeling and solving Python code inside <python>. Proceed with the task now.
""".strip(),
        Stage.OBJ_CONS: f"""
You are a professional optimization problem analyst, proficient in extracting key elements from optimization problems described in natural language.

Here is the specific description of the optimization problem:
{{task_description}}

Here are the type analysis and sets that have already been defined:
{{schema_skill_str}}

Here are the parameters and variables that have already been defined:
{{set_param_var_str}}

1. You should first think step by step in <thought> and carefully analyze the mathematical objective and constraints.

2. Then you should generate the Objectives and Constraints components first and in order:
<Objective> ... </Objective>
<Constraints> ... </Constraints>

3. After finishing the current stage, continue and provide the corresponding Gurobi Python code within <python> ... </python>.

The required order from this point is:
1. <thought>
2. <Objective>
3. <Constraints>
4. <python>

{OBJ_CON_NOTICE}

{CODE_NOTICE}
**NOTE**: You should output the problem Objective in <Objective> and the problem Constraints in <Variables>, and then MUST output the complete optimization problem modeling code in <python>.
IMPORTANT: Do not skip any steps. AFTER the Objective and Constraints sections are fully completed, you must output the full modeling and solving Python code inside <python>. Proceed with the task now.
""".strip(),
        Stage.CODE: f"""
You are an optimization expert. You should solve the optimization problem and only Provide the corresponding Gurobi Python code to implement the model within <python> and </python>

Here is the specific description of the optimization problem:
{{task_description}}

Here are the type analysis and sets that have already been defined:
{{schema_skill_str}}

Here are the parameters and variables that have already been defined:
{{set_param_var_str}}

Here are the objective and constraints that have already been defined:
{{obj_cons_str}}

You should think first within <thought> and then only provide the executable Gurobi Python code needed for this optimization problem within <python>.
Translate the finalized model faithfully and do not redesign it.
The required order from this point is:
1. <thought>
2. <python>

{CODE_NOTICE}
""".strip(),
}

else:
    DEFAULT_ROLLOUT_TEMPLATES: dict[Stage, str] = {
        Stage.SCHEMA: f"""
You are a professional optimization problem analyst, proficient in extracting key elements from optimization problems described in natural language.

Here is the specific description of the optimization problem:
{{task_description}}

1. You should generate the TYPE and SETS components first and in order:
<Type> ... </Type>
<Sets> ... </Sets>

2. After finishing the current stage, continue the formulation and finally provide the corresponding Gurobi Python code within <python> ... </python>.

The required order from this point is:
1. <Type>
2. <Sets>
3. <python>

{TYPE_SET_NOTICE}

{CODE_NOTICE}
**NOTE**: You should output the problem type in <Type> and the Sets in <Sets>, and then MUST output the complete optimization problem modeling code in <python>.
IMPORTANT: Do not skip any steps. AFTER the TYPE and SET sections are fully completed, you must output the full modeling and solving Python code inside <python>. Proceed with the task now.
    """.strip(),
        Stage.SET_PARAM_VAR: f"""
You are a professional optimization problem analyst, proficient in extracting key elements from optimization problems described in natural language.

Here is the specific description of the optimization problem:
{{task_description}}

Here are the type analysis and sets that have already been defined:
{{schema_skill_str}}

1. You should generate the PARAMETERS and VARIABLES components first and in order:
<Parameters> ... </Parameters>
<Variables> ... </Variables>

2. After finishing the current stage, continue the formulation and finally provide the corresponding Gurobi Python code within <python> ... </python>.

The required order from this point is:
1. <Parameters>
2. <Variables>
3. <python>

{PARA_VAR_NOTICE}

{CODE_NOTICE}
**NOTE**: You should output the problem Parameters in <Parameters> and the problem Variables in <Variables>, and then MUST output the complete optimization problem modeling code in <python>.
IMPORTANT: Do not skip any steps. AFTER the Parameters and Variables sections are fully completed, you must output the full modeling and solving Python code inside <python>. Proceed with the task now.
""".strip(),
        Stage.OBJ_CONS: f"""
You are a professional optimization problem analyst, proficient in extracting key elements from optimization problems described in natural language.

Here is the specific description of the optimization problem:
{{task_description}}

Here are the type analysis and sets that have already been defined:
{{schema_skill_str}}

Here are the parameters and variables that have already been defined:
{{set_param_var_str}}

1. You should generate the Objectives and Constraints components first and in order:
<Objective> ... </Objective>
<Constraints> ... </Constraints>

2. After finishing the current stage, continue and provide the corresponding Gurobi Python code within <python> ... </python>.

The required order from this point is:
1. <Objective>
2. <Constraints>
3. <python>

{OBJ_CON_NOTICE}

{CODE_NOTICE}
**NOTE**: You should output the problem Objective in <Objective> and the problem Constraints in <Variables>, and then MUST output the complete optimization problem modeling code in <python>.
IMPORTANT: Do not skip any steps. AFTER the Objective and Constraints sections are fully completed, you must output the full modeling and solving Python code inside <python>. Proceed with the task now.
""".strip(),
        Stage.CODE: f"""
You are an optimization expert. You should solve the optimization problem and only Provide the corresponding Gurobi Python code to implement the model within <python> and </python>

Here is the specific description of the optimization problem:
{{task_description}}

Here are the type analysis and sets that have already been defined:
{{schema_skill_str}}

Here are the parameters and variables that have already been defined:
{{set_param_var_str}}

Here are the objective and constraints that have already been defined:
{{obj_cons_str}}

Please only provide the executable Gurobi Python code needed for this optimization problem within <python>.
Translate the finalized model faithfully and do not redesign it.

{CODE_NOTICE}
""".strip(),
    }


DEFAULT_COMPLETION_TEMPLATES: dict[Stage, str] = {
        Stage.SCHEMA: f"""
    You are an optimization expert. The current stage has already fixed the type analysis and sets for this optimization problem.

    Here is the specific description of the optimization problem:
    {{task_description}}

    Here are the type analysis and sets that have already been fixed:
    {{schema_skill_str}}

    Please complete the remaining formulation by defining parameters, variables, objective, constraints, and finally the executable Gurobi Python code within <python> ... </python>.

    {CODE_NOTICE}
    """.strip(),
        Stage.SET_PARAM_VAR: f"""
    You are an optimization expert. The current stage has already fixed the parameters and variables for this optimization problem.

    Here is the specific description of the optimization problem:
    {{task_description}}

    Here are the type analysis and sets that have already been defined:
    {{schema_skill_str}}

    Here are the parameters and variables that have already been fixed:
    {{set_param_var_str}}

    Please complete the remaining formulation by defining the objective, constraints, and finally the executable Gurobi Python code within <python> ... </python>.

    {CODE_NOTICE}
    """.strip(),
        Stage.OBJ_CONS: f"""
    You are an optimization expert. The mathematical formulation has already been fixed.

    Here is the specific description of the optimization problem:
    {{task_description}}

    Here are the type analysis and sets that have already been defined:
    {{schema_skill_str}}

    Here are the parameters and variables that have already been defined:
    {{set_param_var_str}}

    Here are the objective and constraints that have already been fixed:
    {{obj_cons_str}}

    Please provide the corresponding executable Gurobi Python code within <python> ... </python>.

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


