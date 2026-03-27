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
- Self-check before output: ① All decision objects included? ② Names/elements short for reference? ③ No inconsistency with problem?

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
- Coefficients use exact values from the problem (e.g., "3 units" → "3 [units]")
- Parameter names correspond to set elements (e.g., set {a,c} → fat_a, fat_c; same prefixes)
- Descriptions match problem semantics (use exact names, e.g., "lemon mix" not "mix")
- No missing/redundant parameters (only those affecting objectives/constraints)
- Validation before output: ① Indexes exist in set? ② Values/units/data types complete? ③ Names consistent? ④ No missing/redundant?

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
- Names correspond to sets/parameters (e.g., set {c,o} → x_c, x_o; matches time_c, cost_c)
- Include all decision variables; domain is physically reasonable (countable→INTEGER, continuous→CONTINUOUS)
- Self-check: ① Indexes match sets/parameters? ② Domain fits physical meaning? ③ All variables included?

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
- Direction/expression match problem goal (e.g., "minimize time" → $\\min(time_c x_c + time_o x_o)$)
- Include all variables affecting the goal; no redundant symbols (e.g., no double "max")
- Validation before output: ① Direction matches goal? ② Coefficients = parameters? ③ All variables included? ④ LaTeX correct?

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
3. Simplify to integer coefficients (e.g., x ≤ (1/2)y → 2x ≤ y); use defined parameters/variables
4. Output results directly (no inference process)

### Key Requirements:
- Include all core constraints: ① Bounds (≥/≤, e.g., x ≥ 100) ② Resource/demand (total x1+x2 ≤ 300) ③ Proportion (A ≤ 2B) ④ Non-negativity (all variables ≥ 0)
- Inequality direction accurate: "at least"→≥, "at most"→≤, "exactly"→=, "more than (integer)"→≥y+1
- No contradictory constraints (e.g., x≥200 and x≤100)
- Validation before output: ① All constraint types covered? ② Direction matches problem? ③ Symbols are defined? ④ No contradictions?

### Correct Example:
problem: An accounting firm needs ≥450h labor, ≤$15000 budget; f=8h/$300, p=4h/$100.
### Mathematical Optimization Model: 
## Variables: - x_f: Full-time shifts; x_p: Part-time shifts
## Constraints:  
- labor_demand: Total hours ≥ required: $8x_f + 4x_p ≥ 450$ (Inequality)
- budget_limit: Total wages ≤ budget: $300x_f + 100x_p ≤ 15000$ (Inequality)
- non_negativity: Shifts ≥ 0: $x_f ≥ 0, x_p ≥ 0$ (Inequality)

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
