from __future__ import annotations

from copy import deepcopy
from typing import Any

_KEYWORD_TOKENS: tuple[str, ...] = (
    "obj",
    "objective",
    "cost",
    "profit",
    "revenue",
    "capacity",
    "demand",
    "budget",
    "limit",
    "bound",
    "penalty",
    "price",
    "time",
    "distance",
    "resource",
)


def build_perturbation_map(base_instance: dict[str, Any]) -> dict[str, Any]:
    numeric_keys = _numeric_top_level_keys(base_instance)
    focus_keys = _focus_keys(base_instance, numeric_keys)

    base_values: dict[str, int | float] = {}
    for key in numeric_keys:
        value = base_instance.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            base_values[key] = value

    return {
        "strategy": "key_param_numeric_perturb",
        "numeric_keys": numeric_keys,
        "focus_keys": focus_keys,
        "base_values": base_values,
        "num_numeric_keys": len(numeric_keys),
        "num_focus_keys": len(focus_keys),
    }


def generate_perturbed_instances_from_map(
    base_instance: dict[str, Any],
    perturbation_map: dict[str, Any] | None,
    k: int,
) -> list[dict[str, Any]]:
    if k <= 0:
        return []

    pmap = perturbation_map or build_perturbation_map(base_instance)
    numeric_keys = [
        str(kv)
        for kv in pmap.get("numeric_keys", [])
        if isinstance(kv, str) and kv in base_instance and isinstance(base_instance.get(kv), (int, float))
    ]
    if not numeric_keys:
        numeric_keys = _numeric_top_level_keys(base_instance)
    if not numeric_keys:
        return []

    focus_keys = [
        str(kv)
        for kv in pmap.get("focus_keys", [])
        if isinstance(kv, str) and kv in numeric_keys
    ]
    if not focus_keys:
        focus_keys = _focus_keys(base_instance, numeric_keys)
    if not focus_keys:
        focus_keys = numeric_keys[:]

    base_values = pmap.get("base_values", {}) if isinstance(pmap, dict) else {}

    cases: list[dict[str, Any]] = []
    max_width = min(3, len(focus_keys))

    for case_idx in range(k):
        case = deepcopy(base_instance)
        width = 1 + (case_idx % max(1, max_width))
        start = case_idx % len(focus_keys)

        picked: list[str] = []
        for step in range(width):
            picked.append(focus_keys[(start + step) % len(focus_keys)])

        changes: list[dict[str, Any]] = []
        for local_idx, key in enumerate(picked):
            old_value = None
            if isinstance(base_values, dict):
                old_value = base_values.get(key)
            if not isinstance(old_value, (int, float)):
                old_value = case.get(key)

            if not isinstance(old_value, (int, float)) or isinstance(old_value, bool):
                continue

            new_value = _perturb_value(old_value, case_idx, local_idx)
            case[key] = new_value
            changes.append({"key": key, "old": old_value, "new": new_value})

        case["__perturbation__"] = {
            "case_index": case_idx,
            "num_keys": len(changes),
            "changes": changes,
            "strategy": str(pmap.get("strategy", "key_param_numeric_perturb")),
            "focus_keys": focus_keys,
        }
        cases.append(case)

    return cases


def generate_perturbed_instances(base_instance: dict[str, Any], k: int) -> list[dict[str, Any]]:
    pmap = build_perturbation_map(base_instance)
    return generate_perturbed_instances_from_map(base_instance, pmap, k)


def _numeric_top_level_keys(instance: dict[str, Any]) -> list[str]:
    keys: list[str] = []
    for key, value in instance.items():
        if key.startswith("__"):
            continue
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            keys.append(key)
    return keys


def _focus_keys(instance: dict[str, Any], numeric_keys: list[str]) -> list[str]:
    hinted = instance.get("__key_param_keys__")
    if isinstance(hinted, list):
        selected = [str(k) for k in hinted if isinstance(k, str) and k in numeric_keys]
        if selected:
            return selected

    ranked = sorted(numeric_keys, key=_score_key_name, reverse=True)
    return ranked[: min(12, len(ranked))]


def _score_key_name(key: str) -> int:
    lowered = key.lower()
    score = 0
    for token in _KEYWORD_TOKENS:
        if token in lowered:
            score += 2
    if lowered.startswith("num_"):
        score += 1
    return score


def _perturb_value(value: int | float, case_idx: int, local_idx: int) -> int | float:
    is_int = isinstance(value, int) and not isinstance(value, bool)
    direction = -1.0 if (case_idx + local_idx) % 2 == 0 else 1.0

    # Deterministic relative scales: +/- 8%, 12%, 16%.
    magnitude = 0.08 + 0.04 * ((case_idx + local_idx) % 3)

    if not is_int and abs(float(value)) <= 1.0:
        delta = direction * (0.03 + 0.02 * ((case_idx + local_idx) % 3))
        candidate = float(value) + delta
        if 0.0 <= float(value) <= 1.0:
            candidate = min(1.0, max(0.0, candidate))
        return round(candidate, 6)

    candidate = float(value) * (1.0 + direction * magnitude)

    if is_int:
        new_int = int(round(candidate))
        old_int = int(value)
        if new_int == old_int:
            new_int = old_int + (1 if direction > 0 else -1)
        if old_int >= 0 and new_int < 0:
            new_int = 0
        return new_int

    if float(value) >= 0 and candidate < 0:
        candidate = 0.0
    return round(candidate, 6)
