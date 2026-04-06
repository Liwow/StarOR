# schema + skill
# set + parameter + var
# obj + con
# model2code

prompt_schema = """
"""

prompt_skill = """
"""

prompt_set = '''
!!! MANDATORY FORMAT RULES !!!
1. MUST contain ### Sets Definition: header + ## Set: subheader
2. List items: - set_name: description (e.g., "- s: Employee types: {f,p} where f=full-time, p=part-time")
3. NO mathematical symbols/units; set names/indexes: short, lowercase, no spaces (e.g., "s" not "employee_set")
4. All sets MUST be referenced by subsequent parameters/variables; output results directly (no inference process)

### Key Requirements:
- Explicitly clarify element-object correspondence (e.g., {f,p} where f=full-time, p=part-time)
- Include all object types (decision objects, resources, time periods; no missing/redundant elements)
- Set elements are short (consistent with parameter/variable names, e.g., "c" for cleansing chemical)
- Self-check before output: 鈶?All decision objects included? 鈶?Names/elements short for reference? 鈶?No inconsistency with problem?

### Correct Example:
problem: An accounting firm uses full-time (f) and part-time (p) workers.
### Sets Definition:
## Set:
- s: Employee types: {f,p} where f=full-time workers, p=part-time workers

'''

prompt_parameters = '''
!!! MANDATORY FORMAT RULES !!!
1. MUST contain ### Parameters Definition: header + ## Parameters: subheader
2. Format:
   - Indexed: - param_setElement: description [unit][indexed by SetName] (data type) (e.g., h_f: Hours per full-time shift [hours][indexed by s] (integer))
   - Global (no index): - param: description [unit] (data type) (e.g., T: Total labor required [hours] (integer))
3. Indexes strictly match set elements; parameter names use consistent prefixes (e.g., r_1, r_2 for red liquid)
4. Include specific numerical values; output results directly (no inference process)

### Key Requirements:
- Coefficients use exact values from the problem (e.g., "3 units" 鈫?"3 [units]")
- Parameter names correspond to set elements (e.g., set {a,c} 鈫?fat_a, fat_c; same prefixes)
- Descriptions match problem semantics (use exact names, e.g., "lemon mix" not "mix")
- No missing/redundant parameters (only those affecting objectives/constraints)
- Validation before output: 1. Indexes exist in set? 2. Values/units/data types complete? 3. Names consistent? 4. No missing/redundant?

### Correct Example:
problem: An accounting firm uses full-time (f: 8h/shift, $300/shift) and part-time (p: 4h/shift, $100/shift) workers; needs 450h labor, $15000 budget.
### Mathematical Optimization Model: 
## Set: - s: Employee types: {f,p} where f=full-time, p=part-time
## Parameters:  
- h_f: Hours per full-time shift [hours][indexed by s] (integer): 8
- h_p: Hours per part-time shift [hours][indexed by s] (integer): 4
- w_f: Wage per full-time shift [USD][indexed by s] (integer): 300
- w_p: Wage per part-time shift [USD][indexed by s] (integer): 100
- T: Total labor required [hours] (integer): 450
- B: Total budget [USD] (integer): 15000

'''

prompt_variable = '''
!!! MANDATORY FORMAT RULES !!!
1. MUST contain ### Variables Definition: header + ## Variables: subheader
2. Format:
   - Indexed: - x_setElement: description (domain) (e.g., x_f: Full-time shifts (NONNEGATIVE INTEGER))
   - Global: - x_name: description (domain)
3. Domain: GRB types (CONTINUOUS/INTEGER/BINARY) + NONNEGATIVE; indexes match set elements
4. Output results directly (no inference process)

### Key Requirements:
- Domain includes non-negativity (e.g., "NONNEGATIVE INTEGER" not just "INTEGER")
- Descriptions match problem scenario (e.g., "provided" not "produced" if for food supply)
- Names correspond to sets/parameters (e.g., set {c,o} 鈫?x_c, x_o; matches time_c, cost_c)
- Include all decision variables; domain is physically reasonable (countable鈫扞NTEGER, continuous鈫扖ONTINUOUS)
- Self-check: 鈶?Indexes match sets/parameters? 鈶?Domain fits physical meaning? 鈶?All variables included?

### Correct Example:
problem: An accounting firm schedules full-time (f) and part-time (p) worker shifts.
### Mathematical Optimization Model: 
## Set: - s: Employee types: {f,p} where f=full-time, p=part-time
## Variables:  
- x_f: Number of full-time shifts (NONNEGATIVE INTEGER)
- x_p: Number of part-time shifts (NONNEGATIVE INTEGER)

'''

prompt_objective = '''
!!! MANDATORY FORMAT RULES !!!
1. MUST contain ### Objective Definition: header + ## Objective: subheader
2. Specify direction (maximize/minimize) + LaTeX expression (e.g., $\\min(x_f + x_p)$)
3. Use defined parameters/variables (no undefined symbols); coefficients = parameters (no hard-coded numbers)
4. Output results directly (no inference process)

### Key Requirements:
- Direction/expression match problem goal (e.g., "minimize time" 鈫?$\\min(time_c x_c + time_o x_o)$)
- Include all variables affecting the goal; no redundant symbols (e.g., no double "max")
- Validation before output: 鈶?Direction matches goal? 鈶?Coefficients = parameters? 鈶?All variables included? 鈶?LaTeX correct?

### Correct Example:
problem: Minimize total worker shifts for an accounting firm.
### Mathematical Optimization Model: 
## Set: - s: Employee types: {f,p} where f=full-time, p=part-time
## Variables: - x_f: Full-time shifts (NONNEGATIVE INTEGER); x_p: Part-time shifts (NONNEGATIVE INTEGER)
## Objective:  
- minimize_total_shifts: Minimize total worker shifts: $\\min(x_f + x_p)$

'''

prompt_constrains = '''
!!! MANDATORY FORMAT RULES !!!
1. MUST contain ### Constraints Definition: header + ## Constraints: subheader
2. Format: - constraint_name: description: $LaTeX$ (type: Equality/Inequality)
3. Simplify to integer coefficients (e.g., x 鈮?(1/2)y 鈫?2x 鈮?y); use defined parameters/variables
4. Output results directly (no inference process)

### Key Requirements:
- Include all core constraints: 鈶?Bounds (鈮?鈮? e.g., x 鈮?100) 鈶?Resource/demand (total x1+x2 鈮?300) 鈶?Proportion (A 鈮?2B) 鈶?Non-negativity (all variables 鈮?0)
- Inequality direction accurate: "at least"鈫掆墺, "at most"鈫掆墹, "exactly"鈫?, "more than (integer)"鈫掆墺y+1
- No contradictory constraints (e.g., x鈮?00 and x鈮?00)
- Validation before output: 鈶?All constraint types covered? 鈶?Direction matches problem? 鈶?Symbols are defined? 鈶?No contradictions?

### Correct Example:
problem: An accounting firm needs 鈮?50h labor, 鈮?15000 budget; f=8h/$300, p=4h/$100.
### Mathematical Optimization Model: 
## Variables: - x_f: Full-time shifts; x_p: Part-time shifts
## Constraints:  
- labor_demand: Total hours 鈮?required: $8x_f + 4x_p 鈮?450$ (Inequality)
- budget_limit: Total wages 鈮?budget: $300x_f + 100x_p 鈮?15000$ (Inequality)
- non_negativity: Shifts 鈮?0: $x_f 鈮?0, x_p 鈮?0$ (Inequality)

'''

prompt_model2code = '''
!!! MANDATORY FORMAT RULES !!!
    1. MUST contain ### Gurobi Code: header
    2. Use EXACT variable names: model, status, optimal
    3. Variable types MUST use GRB.CONTINUOUS etc.
    4. Objective MUST be named 'obj'
    5. MUST include error handling
    6.Do not output the inference process, and output the results directly in the format I require
    7.The generated code strictly maintains the following form:<Gurobi_code></Gurobi_code>,Code must generated between two tags!!It should not contain redundant code,
    8.The final output format of the code must follow the format in the example I gave.
    9.This code must be included in it: print(f"Optimal value: {{optimal}}")

    You need to Generate the code based on the model I gave you in the same form as above.
    You can't modify the model, whether the model definition is correct or incorrect, you can't modify the model, you just need to transform
        Model Description:

'''

