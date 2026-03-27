from __future__ import annotations

from ttrl_or.types import Stage


DEFAULT_TEMPLATES: dict[Stage, str] = {
    Stage.SCHEMA: """
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

========================================
OUTPUT FORMAT REQUIREMENTS
========================================
Output must contain exactly these section headers:
- "### Problem Schema"
- "### Modeling Skills"

Under "### Problem Schema", output exactly these subheaders:
- "## Problem Type"
- "## Decision Structure"
- "## Core Entities"
- "## Resources / Limits"
- "## Time / Space Structure"
- "## Optimization Goal"
- "## Anticipated Constraint Families"
- "## Assumption Boundary"

Under "### Modeling Skills", output exactly these subheaders:
- "## Core Skills"
- "## Potential Pitfalls"

========================================
STAGE GOAL
========================================
You need to convert the task description into a modeling schema that is:
1. structurally correct,
2. minimal but sufficient,
3. easy for downstream stages to formalize.

Focus on:
- what must be decided,
- what entities are decision-relevant,
- what resources or limits matter,
- what structural relations are likely needed,
- what modeling skills will be required later.

========================================
DETAILED INSTRUCTIONS
========================================
1. Problem Schema

"## Problem Type"
- Classify the problem into one or more OR families, such as:
  assignment, scheduling, routing, packing, covering, partitioning,
  resource allocation, network flow, facility location, production planning,
  matching, sequencing, inventory, or hybrid combinations.
- If hybrid, explicitly state the combination.

"## Decision Structure"
- Describe the essential decisions in compact OR language.
- Focus on what is being chosen, assigned, scheduled, activated, routed, or allocated.
- Do not define variables yet.

"## Core Entities"
- List only the major decision-relevant object types.
- Examples: customers, facilities, products, jobs, machines, workers, periods, locations, vehicles.
- Do not include background details unless they are model-relevant.

"## Resources / Limits"
- Identify the main capacities, budgets, demands, quotas, availabilities, or logical limits.
- Include only limits that are likely to appear in the formal model.

"## Time / Space Structure"
- State whether the problem is:
  single-period, multi-period, sequential, stage-based, network-based, spatial, or has no explicit time/space structure.
- If none, explicitly say "None".

"## Optimization Goal"
- State the optimization goal precisely:
  minimize cost, maximize profit, minimize completion time, maximize coverage, etc.

"## Anticipated Constraint Families"
- List the main constraint families likely needed in the formal model.
- Use concise OR-style phrases, such as:
  demand satisfaction,
  capacity limits,
  assignment exclusivity,
  flow conservation,
  linking / activation,
  precedence,
  coverage,
  budget restriction,
  balance constraints,
  lower/upper bound requirements.
- This is not the place to write equations.

"## Assumption Boundary"
- Distinguish clearly between:
  1. what is explicit in the task,
  2. what is strongly implied,
  3. what is ambiguous or missing.
- Do not invent extra assumptions.
- If some modeling detail is unclear, state it briefly and conservatively.

2. Modeling Skills

"## Core Skills"
- Provide a concise bullet list of reusable OR modeling patterns required by this task.
- Skills must be specific and operational, not generic.
- Good examples:
  - exact-one assignment
  - capacity aggregation
  - binary activation and linking
  - time-indexed scheduling
  - flow balance
  - precedence enforcement
  - coverage modeling
  - fixed-charge formulation
  - either-or logic
  - cardinality control
- Avoid vague items like "optimization modeling" or "mathematical reasoning".

"## Potential Pitfalls"
- Provide a concise bullet list of likely formulation errors.
- Focus on structural mistakes, such as:
  - confusing decisions with input data
  - missing an indexing dimension
  - omitting linking logic
  - reversing inequality direction
  - failing to model exclusivity or conservation
  - introducing unsupported assumptions
  - over-modeling irrelevant details

========================================
GLOBAL RULES
========================================
- Output results directly. Do NOT output chain-of-thought.
- Be concise but complete.
- Preserve the original task semantics.
- Prefer structural abstraction over surface paraphrase.
- Include only decision-relevant content.
- Do not drift into downstream stages.

========================================
QUALITY CHECK BEFORE OUTPUT
========================================
Make sure:
1. the decision structure is clear,
2. the main entities and limits are captured,
3. the optimization goal is explicit,
4. the anticipated constraint families are meaningful,
5. the skills are reusable OR patterns,
6. no sets, parameters, variables, equations, or code appear in this stage.
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

========================================
MANDATORY FORMAT RULES
========================================
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

========================================
DETAILED INSTRUCTIONS
========================================
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

========================================
ALLOWED / FORBIDDEN
========================================
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

========================================
CONSISTENCY RULES
========================================
- Every indexed parameter must reference an existing set.
- Every indexed variable must reference an existing set.
- All major entities from previous stages must be reflected if they are decision-relevant.
- Do not define parameters that are actually decisions.
- Do not define variables that are actually fixed inputs.
- Avoid duplicate names with different meanings.

========================================
QUALITY CHECK BEFORE OUTPUT
========================================
Make sure:
1. All decision-relevant entities have corresponding sets if needed.
2. All numeric inputs needed later are captured as parameters.
3. All decisions are represented as variables with valid domains.
4. Names are short, stable, and reusable by the next stage.
5. No objective, no constraints, and no code appear in this stage.
""".strip(),
    Stage.OBJ_CONS: """
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

========================================
MANDATORY FORMAT RULES
========================================
1. Objective format:
- objective_name: description: $LaTeX expression$

2. Constraints format:
- constraint_name: description: $LaTeX expression$ (type: Equality/Inequality)

3. All symbols in objective/constraints must come from previous stages.
4. Use symbolic parameters rather than hard-coded numeric coefficients whenever possible.
5. Output results directly. Do NOT output chain-of-thought.

========================================
OBJECTIVE INSTRUCTIONS
========================================
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

========================================
CONSTRAINT INSTRUCTIONS
========================================
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

========================================
LATEX RULES
========================================
- Write mathematically clean LaTeX.
- Use proper summation/index notation.
- Keep notation consistent with previous stages.
- Do not mix informal text into formulas.

========================================
ALLOWED / FORBIDDEN
========================================
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

========================================
CONSISTENCY RULES
========================================
- Every variable appearing in the objective must be defined earlier.
- Every parameter appearing in the objective must be defined earlier.
- Every symbol in every constraint must be defined earlier.
- Objective and constraints must together reflect the original task faithfully.
- Do not omit key structural constraints.
- Do not add assumptions that materially change the problem.

========================================
QUALITY CHECK BEFORE OUTPUT
========================================
Make sure:
1. The objective direction is correct.
2. All symbols are defined in earlier stages.
3. Every major requirement of the problem is captured by constraints.
4. Constraint directions are correct.
5. No code appears in this stage.
Before finalizing, verify whether the model covers: feasibility, resource balance, demand/service requirements, linking logic, exclusivity/conflict logic, and domain consistency whenever applicable.
""".strip(),
    Stage.CODE: """
ou are continuing a 4-stage OR modeling pipeline.

Task:
{task_description}

Previous stages:
{history}

Stage 4 = model2code

Your job is to translate the finalized optimization model into executable Gurobi Python code.

You must faithfully translate the existing model.
You are NOT allowed to correct, reinterpret, simplify, or redesign the model.
Even if the model seems imperfect, you must still translate it as given.

========================================
MANDATORY OUTPUT RULES
========================================
1. Output must contain exactly one top-level section header:
- "### Gurobi Code"

2. Under that header, output code STRICTLY inside:
<Gurobi_code>
...
</Gurobi_code>

3. Do NOT output any explanation before or after the code.
4. Do NOT output chain-of-thought.
5. The generated code must be minimal, executable, and non-redundant.

========================================
MANDATORY CODE REQUIREMENTS
========================================
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

========================================
CODE STYLE REQUIREMENTS
========================================
- Import gurobipy as gp and from gurobipy import GRB
- Create variables and constraints using names consistent with the mathematical model
- Translate indexed definitions into dictionaries or addVars when appropriate
- Use symbolic data placeholders if the task/model does not provide explicit full datasets
- Keep the code faithful to the model structure
- Do not add extra modeling assumptions
- Do not add demo data unless absolutely necessary for syntax completeness
- If some data structures are unspecified, create minimal placeholder containers consistent with the model notation

========================================
TRANSLATION PRINCIPLES
========================================
- Translate, do not repair.
- Preserve objective direction exactly.
- Preserve every constraint category exactly.
- Preserve variable domains exactly.
- Preserve indexing logic exactly.
- Do not silently remove constraints.
- Do not silently add new constraints.
- Do not change parameter meanings.

========================================
ROBUSTNESS REQUIREMENTS
========================================
The code should:
- handle optimization status safely,
- avoid crashing on missing optimal solution reporting logic,
- print a clear result when solved,
- remain syntactically correct Python code.

A typical safe pattern is:
- build model
- set objective
- add constraints
- optimize
- check status
- if solvable, set optimal and print
- otherwise print model status information

========================================
FINAL CHECK BEFORE OUTPUT
========================================
Make sure:
1. only one "### Gurobi Code" section exists,
2. code is only inside <Gurobi_code> and </Gurobi_code>,
3. model / status / optimal / obj use the exact required names,
4. the code is a faithful translation of previous stages,
5. there is no explanation outside the code block.
""".strip(),
}
