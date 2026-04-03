from __future__ import annotations

from dataclasses import dataclass

from ttrl_or.types import DEFAULT_STAGE_ORDER, SOLVERLLM_STAGE_ORDER, Stage

from .notice_prompts import CODE_NOTICE, OBJ_CON_NOTICE, SET_PARA_VAR_NOTICE, SYSTEM_INSTRUCTION


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
I need you to help me generate the TYPE component for a mathematical optimization formulation. You are also required to determine which category of classical optimization problem the given instance belongs to.

Here is the specific description of the optimization problem:
{{task_description}}


1. MUST return exactly one block: <Type>...</Type>
2. This stage is only for optimization type + modeling hint analysis.
3. Do NOT define sets, parameters, variables, objective, constraints, or code.
4. Output concise, reusable, downstream-helpful modeling guidance only.

Inside <Type>, organize the content as:
### Type Analysis:
- type: LP / MILP / NLP / MINLP
- subtype: classical OR family if identifiable, like TSP, SetCover, CVRP and so on.

### Modeling Hints

""".strip(),
    Stage.SET_PARAM_VAR: f"""You are a professional optimization problem analyst, proficient in extracting key elements from optimization problems described in natural language.
I need you to help me generate the SETS, PARAMETERS and VARIABLES components for a mathematical optimization formulation.

Here is the specific description of the optimization problem:
{{task_description}}

Here is the analysis of the classical category for the optimization problem:
{{schema_skill_str}}

Please provide the sets, parameters, and variables needed for this optimization problem.
Return exactly these three blocks in order:
<Sets> ... </Sets>
<Parameters> ... </Parameters>
<Variables> ... </Variables>
Do not write the objective, constraints, or code.

{SET_PARA_VAR_NOTICE}
""".strip(),
    Stage.OBJ_CONS: f"""
You are a professional optimization problem analyst, proficient in extracting key elements from optimization problems described in natural language.
I need you to help me generate the Objectives and Constraints components for a mathematical optimization formulation.

Here is the specific description of the optimization problem:
{{task_description}}

Here is the analysis of the classical category for the optimization problem:
{{schema_skill_str}}

Here are the sets, parameters, and variables that have already been defined:
{{set_param_var_str}}

Please provide the objective and constraints needed for this optimization problem.
Return exactly these two blocks in order:
<Objective> ... </Objective>
<Constraints> ... </Constraints>
Do not generate code.

{OBJ_CON_NOTICE}
""".strip(),
    Stage.CODE: f"""
You are an optimization expert. You should solve the optimization problem and only Provide the corresponding Gurobi Python code to implement the model within <python> and </python>

Here is the specific description of the optimization problem:
{{task_description}}

Here is the analysis of the classical category for the optimization problem:
{{schema_skill_str}}

Here are the sets, parameters, and variables that have already been defined:
{{set_param_var_str}}

Here are the objective and constraints that have already been defined:
{{obj_cons_str}}

Please provide the executable Gurobi Python code needed for this optimization problem within <python>.
Translate the finalized model faithfully and do not redesign it.

{CODE_NOTICE}
""".strip(),
}


DEFAULT_ROLLOUT_TEMPLATES: dict[Stage, str] = {
    Stage.SCHEMA: f"""
You are a professional optimization problem analyst, proficient in extracting key elements from optimization problems described in natural language.
Here is the specific description of the optimization problem:
{{task_description}}

1. You should first think step by step in <thought> and Carefully analyze the problem to identify decision variables, objective, and constraints.

2. Then You should generate the TYPE component for a mathematical optimization formulation within <Type> and </Type>. You are also required to determine which category of classical optimization problem the given instance belongs to.

Inside <Type>, organize the content as:
### Type Analysis:
- type: LP / MILP / NLP / MINLP
- subtype: classical OR family if identifiable, like TSP, SetCover, CVRP and so on.

### Modeling Hints

3. Finally, you should Develop a complete mathematical model step bt step, and  Provide the corresponding Gurobi Python code to implement the model in <python>.

The required order from this point is:
1. <thought>
2. <Type>
3. <python>

{CODE_NOTICE}
""".strip(),
    Stage.SET_PARAM_VAR: f"""You are a professional optimization problem analyst, proficient in extracting key elements from optimization problems described in natural language.

Here is the specific description of the optimization problem:
{{task_description}}

Here is the analysis of the classical category for the optimization problem:
{{schema_skill_str}}

1. You should first think step by step in <thought> and Carefully analyze the problem to identify decision variables, objective, and constraints.

2. Then You should generate the SETS, PARAMETERS and VARIABLES components for a mathematical optimization formulation.

- Return exactly these three blocks first and in order:
<Sets> ... </Sets>
<Parameters> ... </Parameters>
<Variables> ... </Variables>

3. Finally, you should Develop a complete mathematical model step bt step, and  Provide the corresponding Gurobi Python code to implement the model in <python>.

The required order from this point is:
1. <thought>
2. <Sets>
3. <Parameters>
4. <Variables>
5. <python>

{SET_PARA_VAR_NOTICE}

{CODE_NOTICE}
""".strip(),
    Stage.OBJ_CONS: f"""
You are a professional optimization problem analyst, proficient in extracting key elements from optimization problems described in natural language.

Here is the specific description of the optimization problem:
{{task_description}}

Here is the analysis of the classical category for the optimization problem:
{{schema_skill_str}}

Here are the sets, parameters, and variables that have already been defined:
{{set_param_var_str}}

1. You should first think step by step in <thought> and Carefully analyze the problem to identify decision variables, objective, and constraints.

2. Then You should generate the Objectives and Constraints components for a mathematical optimization formulation.

- Return exactly these two blocks first and in order:
<Objective> ... </Objective>
<Constraints> ... </Constraints>
- Do not stop after the current stage; continue to produce <python>.

3. Finally, you should Develop a complete mathematical model step bt step, and  Provide the corresponding Gurobi Python code to implement the model in <python>.

The required order from this point is:
1. <thought>
2. <Objective>
3. <Constraints>
4. <python>

{OBJ_CON_NOTICE}

{CODE_NOTICE}
""".strip(),
    Stage.CODE: f"""
You are an optimization expert. You should solve the optimization problem and only Provide the corresponding Gurobi Python code to implement the model within <python> and </python>

Here is the specific description of the optimization problem:
{{task_description}}

Here is the analysis of the classical category for the optimization problem:
{{schema_skill_str}}

Here are the sets, parameters, and variables that have already been defined:
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
You are an optimization expert. You should solve the optimization problem and only Provide the corresponding Gurobi Python code to implement the model within <python> and </python>

Here is the specific description of the optimization problem:
{{task_description}}

Here is the analysis of the classical category for the optimization problem:
{{schema_skill_str}}

please think and Develop a complete mathematical model step by step, explicitly defining: * Sets * Parameters * Decision Variables (and their types) * Objective Function * Constraints 
Finally, Provide the corresponding Gurobi Python code to implement the model in <python></python>.

{CODE_NOTICE}
""".strip(),
    Stage.SET_PARAM_VAR: f"""
You are an optimization expert. You should solve the optimization problem and only Provide the corresponding Gurobi Python code to implement the model within <python> and </python>

Here is the specific description of the optimization problem:
{{task_description}}

Here is the analysis of the classical category for the optimization problem:
{{schema_skill_str}}

Here are the sets, parameters, and variables that have already been defined:
{{set_param_var_str}}

please think and Develop a complete mathematical model step by step, explicitly defining: * Objective Function * Constraints 
Finally, Provide the corresponding Gurobi Python code to implement the model in <python></python>.

{CODE_NOTICE}
""".strip(),
    Stage.OBJ_CONS: f"""
You are an optimization expert. You should solve the optimization problem and only Provide the corresponding Gurobi Python code to implement the model within <python> and </python>

Here is the specific description of the optimization problem:
{{task_description}}

Here is the analysis of the classical category for the optimization problem:
{{schema_skill_str}}

Here are the sets, parameters, and variables that have already been defined:
{{set_param_var_str}}

Here are the objective and constraints that have already been fixed:
{{obj_cons_str}}

And you should Provide the corresponding Gurobi Python code to implement the model in <python></python>.

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
