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


def test_code_extraction_prefers_last_valid_tag():
    """Test that extraction picks the LAST valid (>20 chars) tag, not the first."""
    text = '''
<Gurobi_code>
short
</Gurobi_code>

some middle text

<Gurobi_code>
import gurobipy as gp
from gurobipy import GRB

def solve(instance: dict) -> dict:
    model = gp.Model()
    return {"objective": 42.0, "status": "optimal"}
</Gurobi_code>
'''.strip()

    code = FourStageMCTS._sanitize_code_payload(text)
    # Should extract the LAST valid tag content, not the first short one
    assert "import gurobipy as gp" in code
    assert "def solve(instance: dict)" in code
    assert "short" not in code


def test_code_extraction_skips_short_tags_from_back():
    """Test that short tags are skipped when iterating from back."""
    text = '''
<Gurobi_code>
import gurobipy as gp
from gurobipy import GRB

def solve(instance: dict) -> dict:
    return {"objective": 1.0, "status": "ok"}
</Gurobi_code>

<Gurobi_code>
short2
</Gurobi_code>

<Gurobi_code>
short3
</Gurobi_code>
'''.strip()

    code = FourStageMCTS._sanitize_code_payload(text)
    # The last two tags are too short, should skip back to the first valid one
    assert "import gurobipy as gp" in code
    assert "short2" not in code
    assert "short3" not in code


def test_code_extraction_square_brackets_fallback():
    """Test that [tag]...[/tag] is used as fallback when <tag> not found."""
    text = '''
Some explanation text here.

[Gurobi_code]
import gurobipy as gp
from gurobipy import GRB

def solve(instance: dict) -> dict:
    model = gp.Model()
    return {"objective": 99.0, "status": "optimal"}
[/Gurobi_code]
'''.strip()

    code = FourStageMCTS._sanitize_code_payload(text)
    assert "import gurobipy as gp" in code
    assert "def solve(instance: dict)" in code
    assert "[Gurobi_code]" not in code
    assert "[/Gurobi_code]" not in code


def test_code_extraction_angle_brackets_preferred_over_square():
    """Test that <tag> takes priority over [tag] when both exist."""
    text = '''
[Gurobi_code]
# This is the square bracket version (should NOT be extracted)
print("wrong")
[/Gurobi_code]

<Gurobi_code>
import gurobipy as gp
from gurobipy import GRB

def solve(instance: dict) -> dict:
    return {"objective": 42.0, "status": "ok"}
</Gurobi_code>
'''.strip()

    code = FourStageMCTS._sanitize_code_payload(text)
    # Angle brackets should be preferred
    assert "import gurobipy as gp" in code
    assert "print(\"wrong\")" not in code
