CODE_REFINE_PROMPT_TEMPLATE = """
You are a Senior Operations Research Architect.

Your task is to refine and repair the candidate solver code for the final code layer.


Your goal is to produce an improved and executable Gurobi Python program by performing a joint consistency check over:
- problem description <--> mathematical model
- mathematical model <--> Python code
- execution result <--> problem description

You must use the execution result as diagnostic evidence, not just as extra context. If the execution result reveals infeasibility, unboundedness, index mismatch, missing constraints, wrong objective direction, degenerate zero solution, or other suspicious behaviors, you should revise the code accordingly.

Think carefully before writing the final code.

### Output Constraints
Output ONLY the following format:

<thought>
Verification and refinement summary:
1. Problem → Model consistency:
   - Check whether all entities in the task description (sets, parameters, decision variables, objective, and constraints) are correctly represented in the mathematical model.
   - Identify any missing, redundant, ambiguous, or misinterpreted modeling components.

2. Model → Code consistency:
   - Check whether the Python/Gurobi code faithfully implements the mathematical model.
   - Verify index domains, summation scopes, variable types, objective sense, constraint directions, boundary conditions, and linking logic.
   - Identify any implementation errors, omissions, or mismatches.

3. Execution result → Intended problem consistency:
   - Analyze whether the execution result is consistent with the intended optimization problem.
   - Pay special attention to infeasible, unbounded, abnormal zero-valued, trivially optimal, or otherwise suspicious results.
   - Infer the most likely causes of the observed behavior.

4. Refinement decision:
   - Decide what must be fixed in the code.
   - Prioritize semantic correctness over superficial code repair.
   - Ensure the final code is self-consistent, executable, and aligned with the task description and mathematical model.
</thought>

<python>
import gurobipy as gp
from gurobipy import GRB

# Revised Gurobi modeling code
...
model.optimize()

status = model.status
if status == GRB.OPTIMAL:
    optimal = model.objVal
    print(f"Optimal value: {{optimal}}")
else:
    print(f"Model status: {{status}}")
</python>

### Input Data

Task Description:
{task_description}

Mathematical Model:
{model_text}

The code you should refine:
<python>
{code_text}
</python>

Execution Result of the candidate code:
{execution_text}

### Evaluation Criteria

1. Structural Alignment
Every important modeling element in the task description must be reflected in the mathematical model.

2. Implementation Fidelity
The final Python code must faithfully implement the mathematical model, including:
- correct index sets
- correct parameter usage
- correct variable domains
- correct objective sense
- correct constraint expressions
- correct logical linking among variables and constraints

3. Execution-grounded Validity
The final code must address issues exposed by the execution result.
Do not ignore solver feedback. Use it to diagnose and refine the formulation and implementation.

4. End-to-end Consistency
The final code should be jointly consistent with:
- the task description,
- the mathematical model,
- and the execution behavior.

### Important Requirements
- Do not merely polish syntax.
- Do not preserve incorrect code if it conflicts with the task description or mathematical model.
- Prefer semantic repair over local patching.
- Output the final improved code only in the required format.
""".strip()


CODE_ERROR_PROMPT_TEMPLATE = """
You are an experienced operations research algorithm engineer. You are presented with an operations research problem and a previous attempt to model and code a solution. That attempt resulted in an error.
Problem Description:
{task_description}

Previous Code Solution Attempt:
<python>
{code_text}
</python>

After running the provided code from the previous attempt, the following error occurred:
{error_info}

Your task:
Based on the information above, please perform the following:
1. Analyze Root Cause & Identify Pitfalls 
- Thoroughly analyze the root cause of the error.
- Summarize potential pitfalls or common mistakes related to this type of code error.
2. Provide Corrected Gurobi Code:
- Write the complete and corrected Python code using the 'gurobipy' library to accurately solve the problem.

Please structure your response strictly as follows:
## Cause of the Error and Potential Pitfalls:
<thought> (Your detailed analysis of the error's cause and a summary of potential pitfalls.) </thought>
## Corrected Gurobi Code:
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
Please think step by step.
"""


CODE_INFEASIBLE_PROMPT_TEMPLATE = """
You are an optimization code repair specialist.
Do NOT change model design. Only repair python implementation bugs.

Task description:
{task_description}

Fixed model blocks (unchanged):
{model_text}

Current code:
<python>
{code_text}
</python>

Execution error:
{execution_text}

You are an experienced operations research algorithm engineer. You are presented with an operations research problem and a previous attempt to model and code a solution. That attempt resulted in an infeasible solution.
Problem Description:
{task_description}

Previous Model:
{model_text}

Code Solution Attempt:
<python>
{code_text}
</python>

After running the provided code from the previous attempt, the answer could not provide a feasible solution.

Your task:
Based on the information above, please perform the following:

1. Analyze Root Cause & Identify Pitfalls
- Thoroughly analyze the root cause of the infeasibility.
- Summarize potential pitfalls or common mistakes related to this type of infeasibility.

2. Provide an Improved Mathematical Model: 
- Develop a mathematical model for correctly
- modeling this OR problem. This should address the flaws in the previous attempt.

3. Provide Corrected Gurobi Code:
Write the complete and corrected Python code associated with the mathematical model using the 'gurobipy' library to accurately solve the problem.

Please structure your response strictly as follows:
## Cause of the Error and Potential Pitfalls:
<thought> (Your detailed analysis of the error's cause and a summary of potential pitfalls.) </thought>

## Corrected Mathematical Model:
<Type>
[Identify the problem class: LP, MILP, NLP, etc.]
</Type>

<Sets>
[Define all indices and sets with clear descriptions]
</Sets>

<Parameters>
[Define all constants and data structures, including units]
</Parameters>

<Variables>
[Define decision variables, their domains (Binary, Non-negative, etc.), and physical meanings]
</Variables>

<Objective>
[Mathematical expression of the objective function with Max/Min direction]
</Objective>

<Constraints>
[List all mathematical constraints. Ensure they are indexed correctly (e.g., ∀i ∈ I) and clearly explained]
</Constraints>

## Corrected Gurobi Code:
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
Note: Do not rewrite the model from scratch; instead, surgically patch the existing model and code by addressing valid feedback while critically filtering out any incorrect or redundant signals.
Please think step by step.
""".strip()

