def solve(instance: dict) -> dict:
    total = 0.0
    for value in instance.values():
        if isinstance(value, (int, float)):
            total += float(value)
    return {"objective": total, "status": "ok"}