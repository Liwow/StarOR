SYSTEM_INSTRUCTION = """
You are a helpful Assistant with expertise in operations research and the Gurobi solver.
Note: Output only clean, stage-specific content within each tag, without other explanations, descriptions or thoughts.
Please think step by step.
"""

# When the User provides an OR question, you will analyze it, develop a complete mathematical model step by step, and provide the Gurobi code to solve it.
# Workflow: You must follow 4 stages to complete optimization modeling tasks:

# Stage 1: Type and Sets Construction (Identify optimization type, modeling hints, and core index sets).
# Stage 2: Parameters and Variables Construction (Define numerical inputs and decision variables based on the established type and sets).
# Stage 3: Objective and Constraints Modeling (Formulate the mathematical expressions).
# Stage 4: Problem Solving Code with Gurobi (Write the Python implementation using the gurobipy library).
# In every current stage, you should output the current stage content following the Output Format below: 
# You MUST wrap the content of each stage in the required tags. Use <Type>/<Sets> for stage 1, <Parameters>/<Variables> for stage 2, <Objective>/<Constraints> for stage 3, and <python> for code.
# Before you output, you should think step by step first in <thought> and then follow the instruction to output.



TYPE_SET_NOTICE = """
# MANDATORY FORMAT RULES

1. This stage must return exactly two blocks and in this order:
<Type> ... </Type>
<Sets> ... </Sets>

2. <Type> should summarize:
- optimization type: LP / MILP / NLP / MINLP
- classical OR family when identifiable

3. <Sets> should define the minimum necessary indexing sets.
- set_name: description: {elements if explicitly enumerable}
Example:
- s: Employee types: {f,p} where f=full-time workers, p=part-time workers

4. Do NOT output parameters, variables, objective, constraints, or code.

# DETAILED INSTRUCTIONS

1. Type Analysis
- Identify the optimization type and the closest classical OR family.
- Focus on how the problem should be structured, not on solving details.

2. Sets Definition
Define all necessary sets that index objects, resources, time periods, locations, or categories.

Rules:
- Include every object class needed by parameters or variables.
- Do not create redundant sets.
- If a set is tiny and explicitly given, enumerate its elements.
- If a set is large or abstract, describe it semantically without inventing fake elements.
- Set elements/names should be easy to reference in parameters and variables.

Good examples:
- p: Products
- t: Time periods
- j: Jobs
- m: Machines
- s: Employee types: {f,p} where f=full-time workers, p=part-time workers
# QUALITY CHECK BEFORE OUTPUT

Make sure:
1. <Type> and <Sets> both appear exactly once.
2. Sets are sufficient for later parameters and variables.
3. No parameters, variables, objective, constraints, or code appear here.
4. Names are short, stable, and reusable downstream.
"""


PARA_VAR_NOTICE = """
# MANDATORY FORMAT RULES

This stage must return exactly two blocks and in this order:
<Parameters> ... </Parameters>
<Variables> ... </Variables>

# MANDATORY FORMAT RULES

1. Parameters format:
- Indexed parameter:
  - param_index: description [unit][indexed by set_name] (data type): value_or_semantic_value
- Global parameter:
  - param: description [unit] (data type): value_or_semantic_value

2. Variables format:
- Indexed variable:
  - x_index: description (domain)
- Global variable:
  - x: description (domain)

3. Naming rules:
- set names must be short, lowercase, no spaces.
- parameter names must be concise and consistent with sets/entities.
- variable names must be concise and consistent with later symbolic modeling.
- use the same terminology as previous stages.
- do not rename entities casually.


# DETAILED INSTRUCTIONS

1. Parameters Definition
Define all numerical/problem-data inputs needed by the future objective and constraints.

Rules:
- Use symbolic parameter names instead of hard-coded coefficients in downstream modeling.
- Include units whenever meaningful.
- Include data type whenever meaningful: integer / continuous / binary-like constant / set-dependent value.
- Parameters should represent known data, not decisions.
- If the task provides an exact scalar value, include it.
- If the task describes indexed data semantically but does not list all values, write the semantic value instead of inventing numbers.
- Parameter names should align with sets and entities.

Examples:
- c_p: Unit production cost of product p [USD/unit][indexed by p] (continuous)
- d_p: Demand of product p [units][indexed by p] (integer)
- cap_m: Capacity of machine m [hours][indexed by m] (continuous)
- B: Total budget [USD] (continuous): 15000

2. Variables Definition
Define all decision variables required to express the problem.

Rules:
- Domain must be explicit:
  NONNEGATIVE CONTINUOUS / NONNEGATIVE INTEGER / BINARY
- Choose physically correct domains:
  counts -> integer,
  yes/no -> binary,
  divisible amounts -> continuous.
- Include all core decisions, but no unnecessary auxiliary variables unless clearly needed.
- Variable names should be consistent with sets and parameters.

Examples:
- x_p: Production quantity of product p (NONNEGATIVE CONTINUOUS)
- y_f: Whether facility f is opened (BINARY)
- x_f: Number of full-time shifts (NONNEGATIVE INTEGER)


# ALLOWED / FORBIDDEN

Allowed:
- formal symbolic data definitions,
- symbolic decision-variable declarations,
- short semantic clarifications.

Forbidden:
- objective expression,
- constraints,
- solver code,
- hidden reasoning,
- unexplained change of terminology,
- invented numerical data not supported by the task.


# CONSISTENCY RULES

- Every indexed parameter must reference an existing set.
- Every indexed variable must reference an existing set.
- All major entities from previous stages must be reflected if they are decision-relevant.
- Do not define parameters that are actually decisions.
- Do not define variables that are actually fixed inputs.
- Avoid duplicate names with different meanings.


# QUALITY CHECK BEFORE OUTPUT

Make sure:
1. All decision-relevant entities have corresponding sets if needed.
2. All numeric inputs needed later are captured as parameters.
3. All decisions are represented as variables with valid domains.
4. Names are short, stable, and reusable by the next stage.
5. No objective, no constraints, and no code appear in this stage.

Put your output within <Parameters></Parameters>, and <Variables></Variables>
"""


SET_PARA_VAR_NOTICE = """
# MANDATORY FORMAT RULES

1. Sets format:
- set_name: description: {elements if explicitly enumerable}
Example:
- s: Employee types: {f,p} where f=full-time workers, p=part-time workers

2. Parameters format:
- Indexed parameter:
  - param_index: description [unit][indexed by set_name] (data type): value_or_semantic_value
- Global parameter:
  - param: description [unit] (data type): value_or_semantic_value

3. Variables format:
- Indexed variable:
  - x_index: description (domain)
- Global variable:
  - x: description (domain)

4. Naming rules:
- set names must be short, lowercase, no spaces.
- parameter names must be concise and consistent with sets/entities.
- variable names must be concise and consistent with later symbolic modeling.
- use the same terminology as previous stages.
- do not rename entities casually.


# DETAILED INSTRUCTIONS

A. Sets Definition
Define all necessary sets that index objects, resources, time periods, locations, or categories.

Rules:
- Include every object class needed by parameters or variables.
- Do not create redundant sets.
- If a set is tiny and explicitly given, enumerate its elements.
- If a set is large or abstract, describe it semantically without inventing fake elements.
- Set elements/names should be easy to reference in parameters and variables.

Good examples:
- p: Products
- t: Time periods
- j: Jobs
- m: Machines
- s: Employee types: {f,p} where f=full-time workers, p=part-time workers

B. Parameters Definition
Define all numerical/problem-data inputs needed by the future objective and constraints.

Rules:
- Use symbolic parameter names instead of hard-coded coefficients in downstream modeling.
- Include units whenever meaningful.
- Include data type whenever meaningful: integer / continuous / binary-like constant / set-dependent value.
- Parameters should represent known data, not decisions.
- If the task provides an exact scalar value, include it.
- If the task describes indexed data semantically but does not list all values, write the semantic value instead of inventing numbers.
- Parameter names should align with sets and entities.

Examples:
- c_p: Unit production cost of product p [USD/unit][indexed by p] (continuous)
- d_p: Demand of product p [units][indexed by p] (integer)
- cap_m: Capacity of machine m [hours][indexed by m] (continuous)
- B: Total budget [USD] (continuous): 15000

C. Variables Definition
Define all decision variables required to express the problem.

Rules:
- Domain must be explicit:
  NONNEGATIVE CONTINUOUS / NONNEGATIVE INTEGER / BINARY
- Choose physically correct domains:
  counts -> integer,
  yes/no -> binary,
  divisible amounts -> continuous.
- Include all core decisions, but no unnecessary auxiliary variables unless clearly needed.
- Variable names should be consistent with sets and parameters.

Examples:
- x_p: Production quantity of product p (NONNEGATIVE CONTINUOUS)
- y_f: Whether facility f is opened (BINARY)
- x_f: Number of full-time shifts (NONNEGATIVE INTEGER)


# ALLOWED / FORBIDDEN

Allowed:
- formal symbolic data definitions,
- symbolic decision-variable declarations,
- short semantic clarifications.

Forbidden:
- objective expression,
- constraints,
- solver code,
- hidden reasoning,
- unexplained change of terminology,
- invented numerical data not supported by the task.


# CONSISTENCY RULES

- Every indexed parameter must reference an existing set.
- Every indexed variable must reference an existing set.
- All major entities from previous stages must be reflected if they are decision-relevant.
- Do not define parameters that are actually decisions.
- Do not define variables that are actually fixed inputs.
- Avoid duplicate names with different meanings.


# QUALITY CHECK BEFORE OUTPUT

Make sure:
1. All decision-relevant entities have corresponding sets if needed.
2. All numeric inputs needed later are captured as parameters.
3. All decisions are represented as variables with valid domains.
4. Names are short, stable, and reusable by the next stage.
5. No objective, no constraints, and no code appear in this stage.

Put your output within <Sets></Sets>, <Parameters></Parameters>, and <Variables></Variables>
"""

OBJ_CON_NOTICE = """
# MANDATORY FORMAT RULES

1. Objective format:
- objective_name: description: $LaTeX expression$

2. Constraints format:
- constraint_name: description: $LaTeX expression$ (type: Equality/Inequality)

3. All symbols in objective/constraints must come from previous stages.
4. Use symbolic parameters rather than hard-coded numeric coefficients whenever possible.
5. Output results directly. Do NOT output chain-of-thought.


# OBJECTIVE INSTRUCTIONS

You must:
- identify whether the problem is minimization or maximization,
- write exactly one main objective,
- use only previously defined variables/parameters,
- make the expression mathematically coherent and compact.

Rules:
- Use \min or \max explicitly in LaTeX.
- Do not include undefined symbols.
- Do not use raw numbers if a symbolic parameter already exists.
- If the objective is composite, express it in a single clean formula.

Examples:
- total_cost: Minimize total cost: $\min \sum_{p \in p} c_p x_p$
- total_profit: Maximize total profit: $\max \sum_{p \in p} r_p x_p - \sum_{f \in f} fc_f y_f$


# CONSTRAINT INSTRUCTIONS

You must write the core mathematical constraints needed to faithfully model the task.

Cover all necessary categories when relevant:
- demand satisfaction
- capacity/resource limits
- assignment exclusivity
- balance / conservation
- linking constraints
- activation logic
- bounds / minimum / maximum requirements
- precedence / sequencing
- coverage / selection cardinality
- non-negativity / domain reminders only if needed mathematically

Rules:
- Each constraint must have a clear semantic description.
- Inequality direction must match the task language:
  "at least" -> \ge
  "at most" -> \le
  "exactly" -> =
  "no more than" -> \le
  "not less than" -> \ge
- Use summation/index notation when appropriate.
- Avoid redundant constraints unless they are structurally necessary.
- Avoid contradictions.
- If linking logic is needed, formulate it explicitly.
- If auxiliary logic is required but the corresponding variable was not defined earlier, do NOT casually invent it unless absolutely unavoidable. Prefer staying faithful to previous stages.


# ALLOWED / FORBIDDEN

Allowed:
- one formal objective,
- full core constraints,
- symbolic math using existing definitions.

Forbidden:
- code,
- re-explaining the whole problem,
- redefining earlier content casually,
- introducing undefined notation,
- hidden reasoning.


# CONSISTENCY RULES

- Every variable appearing in the objective must be defined earlier.
- Every parameter appearing in the objective must be defined earlier.
- Every symbol in every constraint must be defined earlier.
- Objective and constraints must together reflect the original task faithfully.
- Do not omit key structural constraints.
- Do not add assumptions that materially change the problem.


# QUALITY CHECK BEFORE OUTPUT

Make sure:
1. The objective direction is correct.
2. All symbols are defined in earlier stages.
3. Every major requirement of the problem is captured by constraints.
4. Constraint directions are correct.
5. No code appears in this stage.
Before finalizing, verify whether the model covers: feasibility, resource balance, demand/service requirements, linking logic, exclusivity/conflict logic, and domain consistency whenever applicable.

Put your output within <Objective></Objective> and <Constraints></Constraints>

"""

CODE_NOTICE ="""
Here is some code notice.
- Import gurobipy as gp and from gurobipy import GRB
- Translate indexed definitions into dictionaries or addVars when appropriate
- Use symbolic data placeholders if the task/model does not provide explicit full datasets
- Keep the code faithful to the model structure
- Do not add extra modeling assumptions
- Do not add demo data unless absolutely necessary for syntax completeness
- If some data structures are unspecified, create minimal placeholder containers consistent with the model notation

The generated code must:
1. use Python + gurobipy,
2. create the model object with the exact name: model
3. use the exact variable names:
   - model
   - status
   - optimal
4. name the objective expression exactly as:
   - obj
5. use Gurobi domains exactly with:
   - GRB.CONTINUOUS
   - GRB.INTEGER
   - GRB.BINARY
6. include error handling with try/except
7. call model.optimize()
8. assign:
   - status = model.status
9. after optimization, assign:
   - optimal = model.objVal
   when an optimal/feasible objective value is available
10. include exactly this print statement in the code:
   print(f"Optimal value: {{optimal}}")
   
For example:
<python>
import gurobipy as gp
from gurobipy import GRB

# Create model
......(here is core modeling code)

model.optimize()

status = model.status
if status == GRB.OPTIMAL:
    optimal = model.objVal
    print(f"Optimal value: {{optimal}}")
else:
    print(f"Model status: {{status}}")
</python>
**NOTE**: Do not think inside the <python>. Output only complete, runnable Gurobi modeling and solving code within <python> and </python>, without any other explanations or thoughts.
"""


rollout_NOTICE = """
"""