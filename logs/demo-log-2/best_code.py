def solve(instance: dict) -> dict:
    total = 0.0
    for idx, key in enumerate(sorted(instance.keys())):
        value = instance[key]
        if isinstance(value, (int, float)):
            total += float(value) * (idx + 1)
    return {"objective": total, "status": "ok"}