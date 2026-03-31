from __future__ import annotations

import json
import math
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from ttrl_or.reward.perturbation import build_perturbation_map, generate_perturbed_instances_from_map


@dataclass(slots=True)
class R3SamplePlan:
    sample_id: str
    source: str
    analysis: str
    base_obj_bounds: dict[str, float | None]
    test_cases: list[dict[str, Any]]
    mapping: list[dict[str, Any]]
    llm_raw_preview: str = ""


def build_r3_planner_prompt(
    *,
    sample_id: str,
    description: str,
    instance: dict[str, Any],
    num_tests: int,
) -> str:
    instance_view = _compact_instance(instance)
    payload = json.dumps(instance_view, ensure_ascii=False, indent=2)

    return (
        "You are an OR robustness planner.\\n"
        "Given a natural-language optimization task and its parsed numeric data, output ONLY a JSON object.\\n"
        "Do not output markdown or extra text.\\n\\n"
        f"sample_id: {sample_id}\\n"
        "Task description:\\n"
        f"{description}\\n\\n"
        "Parsed data instance (JSON):\\n"
        f"{payload}\\n\\n"
        "Return JSON with fields:\\n"
        "1) analysis: short text about problem understanding and key risk points.\\n"
        "2) base_obj_bounds: {lower: number|null, upper: number|null} for original data, broad but meaningful.\\n"
        f"3) tests: list with at most {num_tests} items. Each item should contain:\\n"
        "   - case_id: string\\n"
        "   - changes: list of {key, new_value} modifying existing numeric keys in instance\\n"
        "   - obj_bounds: {lower: number|null, upper: number|null} broad but meaningful\\n"
        "   - rationale: short text\\n"
        "Rules:\\n"
        "- Keep bounds broad, but exclude absurd objectives (e.g., gt around 10000 and obj!=0).\\n"
        "- Prefer edge/boundary stress tests.\\n"
        "- Use only keys that exist in the given instance.\\n"
        "You should think step by step to output\\n"
    )


def build_sample_r3_plan(
    *,
    sample_id: str,
    description: str,
    instance: dict[str, Any],
    reference_answer: str,
    robustness_cases: int,
    llm_text: str | None,
    allow_heuristic_fallback: bool = True,
) -> R3SamplePlan:
    parsed = _parse_json_object(llm_text) if llm_text else None
    if isinstance(parsed, dict):
        plan = _normalize_llm_plan(
            sample_id=sample_id,
            parsed=parsed,
            base_instance=instance,
            robustness_cases=robustness_cases,
            reference_answer=reference_answer,
            raw_preview=_preview(llm_text),
        )
        if plan is not None:
            return plan

    if allow_heuristic_fallback:
        return _heuristic_plan(
            sample_id=sample_id,
            base_instance=instance,
            robustness_cases=robustness_cases,
            reference_answer=reference_answer,
            raw_preview=_preview(llm_text),
        )

    return R3SamplePlan(
        sample_id=sample_id,
        source="disabled",
        analysis="r3 precompute disabled due to extraction failure",
        base_obj_bounds={"lower": None, "upper": None},
        test_cases=[],
        mapping=[],
        llm_raw_preview=_preview(llm_text),
    )


def attach_r3_plan_to_instance(instance: dict[str, Any], plan: R3SamplePlan) -> dict[str, Any]:
    out = dict(instance)
    out["__r3_source__"] = plan.source
    out["__r3_analysis__"] = plan.analysis
    out["__r3_base_obj_bounds__"] = plan.base_obj_bounds
    out["__r3_test_cases__"] = plan.test_cases
    out["__r3_mapping__"] = plan.mapping
    out["__r3_llm_raw_preview__"] = plan.llm_raw_preview
    out["__r3_precompute_required__"] = True
    out["__r3_precompute_ok__"] = bool(plan.source != "disabled" and len(plan.test_cases) > 0)
    out["__r3_disable__"] = not bool(plan.source != "disabled" and len(plan.test_cases) > 0)
    return out


def _normalize_llm_plan(
    *,
    sample_id: str,
    parsed: dict[str, Any],
    base_instance: dict[str, Any],
    robustness_cases: int,
    reference_answer: str,
    raw_preview: str,
) -> R3SamplePlan | None:
    analysis = str(parsed.get("analysis", "")).strip()
    base_bounds = _coerce_bounds(parsed.get("base_obj_bounds"))
    if not _has_valid_bounds(base_bounds):
        base_bounds = _default_bounds_from_reference(reference_answer)

    tests_raw = parsed.get("tests")
    if not isinstance(tests_raw, list):
        tests_raw = []

    numeric_keys = {
        k for k, v in base_instance.items() if not str(k).startswith("__") and isinstance(v, (int, float)) and not isinstance(v, bool)
    }

    test_cases: list[dict[str, Any]] = []
    mapping: list[dict[str, Any]] = []

    for idx, test in enumerate(tests_raw[: max(1, robustness_cases)]):
        if not isinstance(test, dict):
            continue

        case_id = str(test.get("case_id") or f"llm_case_{idx}")
        changes_raw = test.get("changes")
        if not isinstance(changes_raw, list):
            changes_raw = []

        case_instance = deepcopy(base_instance)
        norm_changes: list[dict[str, Any]] = []
        for item in changes_raw:
            if not isinstance(item, dict):
                continue
            key = str(item.get("key", "")).strip()
            if key not in numeric_keys:
                continue
            new_value = _to_number(item.get("new_value"))
            if new_value is None or not math.isfinite(new_value):
                continue
            if isinstance(case_instance.get(key), int) and not isinstance(case_instance.get(key), bool):
                case_instance[key] = int(round(new_value))
            else:
                case_instance[key] = float(new_value)
            norm_changes.append({"key": key, "old": base_instance.get(key), "new": case_instance[key]})

        if not norm_changes:
            continue

        obj_bounds = _coerce_bounds(test.get("obj_bounds"))
        if not _has_valid_bounds(obj_bounds):
            obj_bounds = dict(base_bounds)

        case_instance["__perturbation__"] = {
            "strategy": "llm_r3_batch",
            "case_id": case_id,
            "changes": norm_changes,
        }
        case_entry = {
            "case_id": case_id,
            "instance": case_instance,
            "obj_bounds": obj_bounds,
            "rationale": str(test.get("rationale", "")).strip(),
            "changes": norm_changes,
        }
        test_cases.append(case_entry)
        mapping.append({"case_id": case_id, "changes": norm_changes, "source": "llm"})

    if not test_cases:
        return None

    return R3SamplePlan(
        sample_id=sample_id,
        source="llm",
        analysis=analysis,
        base_obj_bounds=base_bounds,
        test_cases=test_cases,
        mapping=mapping,
        llm_raw_preview=raw_preview,
    )


def _heuristic_plan(
    *,
    sample_id: str,
    base_instance: dict[str, Any],
    robustness_cases: int,
    reference_answer: str,
    raw_preview: str,
) -> R3SamplePlan:
    pmap = build_perturbation_map(base_instance)
    generated = generate_perturbed_instances_from_map(base_instance, pmap, max(1, robustness_cases))
    base_bounds = _default_bounds_from_reference(reference_answer)

    test_cases: list[dict[str, Any]] = []
    mapping: list[dict[str, Any]] = []
    for idx, case in enumerate(generated[: max(1, robustness_cases)]):
        meta = case.get("__perturbation__") if isinstance(case, dict) else {}
        case_id = f"heur_case_{idx}"
        bounds = _expand_bounds(base_bounds, factor=1.3)
        test_cases.append(
            {
                "case_id": case_id,
                "instance": case,
                "obj_bounds": bounds,
                "rationale": "heuristic numeric perturbation",
                "changes": list(meta.get("changes", [])) if isinstance(meta, dict) else [],
            }
        )
        mapping.append(
            {
                "case_id": case_id,
                "changes": list(meta.get("changes", [])) if isinstance(meta, dict) else [],
                "source": "heuristic",
            }
        )

    return R3SamplePlan(
        sample_id=sample_id,
        source="heuristic",
        analysis="heuristic fallback plan",
        base_obj_bounds=base_bounds,
        test_cases=test_cases,
        mapping=mapping,
        llm_raw_preview=raw_preview,
    )


def _default_bounds_from_reference(reference_answer: str) -> dict[str, float | None]:
    gt = _to_number(reference_answer)
    if gt is None or not math.isfinite(gt):
        return {"lower": None, "upper": None}

    span = max(abs(gt), 1.0)
    lower = gt - 10.0 * span
    upper = gt + 10.0 * span
    if gt > 0:
        lower = max(lower, 0.01 * gt)
    if gt < 0:
        upper = min(upper, 0.01 * gt)
    return {"lower": float(lower), "upper": float(upper)}


def _expand_bounds(bounds: dict[str, float | None], factor: float = 1.2) -> dict[str, float | None]:
    lower = bounds.get("lower") if isinstance(bounds, dict) else None
    upper = bounds.get("upper") if isinstance(bounds, dict) else None
    if isinstance(lower, (int, float)) and isinstance(upper, (int, float)):
        center = 0.5 * (float(lower) + float(upper))
        half = 0.5 * (float(upper) - float(lower))
        new_half = max(1.0, half * float(factor))
        return {"lower": center - new_half, "upper": center + new_half}
    return dict(bounds)


def _compact_instance(instance: dict[str, Any], max_items: int = 160) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    count = 0
    for k, v in instance.items():
        key = str(k)
        if key.startswith("__"):
            continue
        if isinstance(v, bool):
            continue
        if isinstance(v, (int, float, str)):
            compact[key] = v
            count += 1
        if count >= max_items:
            break
    return compact


def _parse_json_object(text: str | None) -> dict[str, Any] | None:
    raw = str(text or "").strip()
    if not raw:
        return None

    if raw.startswith("```"):
        lines = raw.splitlines()
        if len(lines) >= 2 and lines[-1].strip().startswith("```"):
            raw = "\n".join(lines[1:-1]).strip()

    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        pass

    start = raw.find("{")
    end = raw.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        parsed = json.loads(raw[start : end + 1])
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None


def _coerce_bounds(value: Any) -> dict[str, float | None]:
    if not isinstance(value, dict):
        return {"lower": None, "upper": None}
    lower = _to_number(value.get("lower"))
    upper = _to_number(value.get("upper"))
    if lower is not None and upper is not None and lower > upper:
        lower, upper = upper, lower
    return {
        "lower": (float(lower) if lower is not None and math.isfinite(float(lower)) else None),
        "upper": (float(upper) if upper is not None and math.isfinite(float(upper)) else None),
    }


def _has_valid_bounds(bounds: dict[str, float | None]) -> bool:
    lower = bounds.get("lower")
    upper = bounds.get("upper")
    return isinstance(lower, (int, float)) or isinstance(upper, (int, float))


def _to_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip().replace(",", "")
        if not text:
            return None
        try:
            return float(text)
        except Exception:
            return None
    return None


def _preview(text: str | None, max_len: int = 320) -> str:
    raw = str(text or "").strip().replace("\n", " ")
    if len(raw) <= max_len:
        return raw
    return raw[: max_len - 3] + "..."
