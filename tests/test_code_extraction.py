from ttrl_or.mcts.tree import FourStageMCTS


def test_code_extraction_prefers_gurobi_tags():
    text = '''
### Gurobi Code
<Gurobi_code>
import gurobipy as gp
from gurobipy import GRB

def solve(instance: dict) -> dict:
    return {"objective": 1.0, "status": "ok"}
</Gurobi_code>
extra trailing text
'''.strip()

    code = FourStageMCTS._sanitize_code_payload(text)
    assert code.startswith("import gurobipy as gp")
    assert "def solve(instance: dict) -> dict:" in code
    assert "<Gurobi_code>" not in code
    assert "</Gurobi_code>" not in code


def test_code_extraction_fallback_to_fenced_block():
    text = '''
notes
```python
import math

def solve(instance: dict) -> dict:
    return {"objective": 2.0, "status": "ok"}
```
'''.strip()

    code = FourStageMCTS._sanitize_code_payload(text)
    assert code.startswith("import math")
    assert "def solve(instance: dict) -> dict:" in code


def test_code_extraction_tag_has_priority_over_fence():
    text = '''
```python
# wrong candidate
print("bad")
```

<Gurobi_code>
from gurobipy import GRB

def solve(instance: dict) -> dict:
    return {"objective": 3.0, "status": "ok"}
</Gurobi_code>
'''.strip()

    code = FourStageMCTS._sanitize_code_payload(text)
    assert code.startswith("from gurobipy import GRB")
    assert "print(\"bad\")" not in code
