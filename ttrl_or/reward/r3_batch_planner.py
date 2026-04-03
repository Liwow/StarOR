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
    base_obj_bounds: dict[str, Any]
    test_cases: list[dict[str, Any]]
    mapping: list[dict[str, Any]]
    feature_catalog: list[dict[str, Any]]
    llm_raw_preview: str = ""
def build_r3_planner_prompt(
    *,
    sample_id: str,
    description: str,
    instance: dict[str, Any],
    num_tests: int,
) -> str:
    feature_catalog = _compact_feature_catalog(_feature_catalog_from_instance(instance))
    instance_view = _compact_instance(instance)
    skeleton = _plan_skeleton(num_tests=num_tests, feature_catalog=feature_catalog)
    return (
        "You are an OR robustness planner.\n"
        "Focus on analysis quality. We already have a rule-built JSON skeleton and runtime mapping.\n"
        "Do NOT invent internal keys such as num_3 or tbl_0_r2_cost. Use feature ids only.\n"
        "Output exactly three tagged blocks and nothing else:\n"
        "<analysis>...text...</analysis>\n"
        "<base_scale>{...JSON...}</base_scale>\n"
        "<tests>[...JSON list...]</tests>\n\n"
        f"sample_id: {sample_id}\n\n"
        "Task description:\n"
        f"{description}\n\n"
        "Parsed numeric data snapshot (for context only):\n"
        f"{json.dumps(instance_view, ensure_ascii=False, indent=2)}\n\n"
        "Feature catalog (use fid only when proposing changes):\n"
        f"{json.dumps(feature_catalog, ensure_ascii=False, indent=2)}\n\n"
        "Rule-built output skeleton reference:\n"
        f"{json.dumps(skeleton, ensure_ascii=False, indent=2)}\n\n"
        "Scale rules:\n"
        "- base_scale / obj_scale must be easy for filter(obj, scale) to use.\n"
        "- Prefer one of these JSON forms:\n"
        '  {"kind":"interval","lower":number|null,"upper":number|null,"reject_exact":[...]}\n'
        '  {"kind":"point","point":number,"tol_abs":number,"reject_exact":[...]}\n'
        '  {"kind":"union","intervals":[{"lower":...,"upper":...}, ...],"reject_exact":[...]}\n'
        "- Keep the scale meaningful: not too loose, but broad enough to allow modeling variation.\n"
        "- Explicitly reject absurd objectives such as 0 when it is clearly impossible.\n\n"
        "Test-case rules:\n"
        "- Each test item should contain case_id, patches, obj_scale, rationale.\n"
        "- patches should usually be a short list of {fid, new_value}.\n"
        "- If needed you may also use {fid, op, value} with op in [replace, scale, shift].\n"
        "- Choose stress cases that are realistic and useful for robustness testing.\n"
        "- Each proposed fid must exist in the feature catalog.\n"
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
    feature_catalog = _feature_catalog_from_instance(instance)
    parsed = _parse_tagged_plan(llm_text) if llm_text else None
    if isinstance(parsed, dict):
        plan = _normalize_llm_plan(
            sample_id=sample_id,
            parsed=parsed,
            base_instance=instance,
            feature_catalog=feature_catalog,
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
            feature_catalog=feature_catalog,
            robustness_cases=robustness_cases,
            reference_answer=reference_answer,
            raw_preview=_preview(llm_text),
        )
    return R3SamplePlan(
        sample_id=sample_id,
        source="disabled",
        analysis="r3 precompute disabled due to extraction failure",
        base_obj_bounds=_default_scale_from_reference(reference_answer),
        test_cases=[],
        mapping=[],
        feature_catalog=feature_catalog,
        llm_raw_preview=_preview(llm_text),
    )
def attach_r3_plan_to_instance(instance: dict[str, Any], plan: R3SamplePlan) -> dict[str, Any]:
    out = dict(instance)
    out["__r3_source__"] = plan.source
    out["__r3_analysis__"] = plan.analysis
    out["__r3_base_obj_bounds__"] = plan.base_obj_bounds
    out["__r3_base_obj_scale__"] = plan.base_obj_bounds
    out["__r3_test_cases__"] = plan.test_cases
    out["__r3_mapping__"] = plan.mapping
    out["__r3_feature_catalog__"] = plan.feature_catalog
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
    feature_catalog: list[dict[str, Any]],
    robustness_cases: int,
    reference_answer: str,
    raw_preview: str,
) -> R3SamplePlan | None:
    analysis = str(parsed.get("analysis", "")).strip()
    feature_map = _feature_catalog_map(feature_catalog)
    base_bounds = _coerce_scale(parsed.get("base_scale"))
    if not _has_valid_scale(base_bounds):
        base_bounds = _default_scale_from_reference(reference_answer)
    tests_raw = parsed.get("tests")
    if not isinstance(tests_raw, list):
        tests_raw = []
    test_cases: list[dict[str, Any]] = []
    mapping: list[dict[str, Any]] = []
    for idx, test in enumerate(tests_raw[: max(1, robustness_cases)]):
        if not isinstance(test, dict):
            continue
        case_id = str(test.get("case_id") or f"llm_case_{idx + 1}")
        patches_raw = test.get("patches")
        if not isinstance(patches_raw, list):
            patches_raw = []
        case_instance = deepcopy(base_instance)
        norm_patches: list[dict[str, Any]] = []
        for item in patches_raw:
            patch = _normalize_patch(item, feature_map, case_instance)
            if patch is None:
                continue
            case_instance[patch["key"]] = patch["new"]
            norm_patches.append(patch)
        if not norm_patches:
            continue
        obj_scale = _coerce_scale(test.get("obj_scale") or test.get("obj_bounds"))
        if not _has_valid_scale(obj_scale):
            obj_scale = _expand_scale(base_bounds, factor=1.2)
        case_instance["__perturbation__"] = {
            "strategy": "llm_r3_batch",
            "case_id": case_id,
            "patches": norm_patches,
            "feature_ids": [patch["fid"] for patch in norm_patches],
        }
        case_entry = {
            "case_id": case_id,
            "instance": case_instance,
            "obj_scale": obj_scale,
            "obj_bounds": obj_scale,
            "rationale": str(test.get("rationale", "")).strip(),
            "patches": norm_patches,
            "changes": norm_patches,
        }
        test_cases.append(case_entry)
        mapping.append(
            {
                "case_id": case_id,
                "patches": norm_patches,
                "feature_ids": [patch["fid"] for patch in norm_patches],
                "source": "llm",
            }
        )
    if not test_cases:
        return None
    return R3SamplePlan(
        sample_id=sample_id,
        source="llm",
        analysis=analysis,
        base_obj_bounds=base_bounds,
        test_cases=test_cases,
        mapping=mapping,
        feature_catalog=feature_catalog,
        llm_raw_preview=raw_preview,
    )
def _heuristic_plan(
    *,
    sample_id: str,
    base_instance: dict[str, Any],
    feature_catalog: list[dict[str, Any]],
    robustness_cases: int,
    reference_answer: str,
    raw_preview: str,
) -> R3SamplePlan:
    pmap = build_perturbation_map(base_instance)
    generated = generate_perturbed_instances_from_map(base_instance, pmap, max(1, robustness_cases))
    base_bounds = _default_scale_from_reference(reference_answer)
    fid_lookup = _key_to_fid_map(feature_catalog)
    test_cases: list[dict[str, Any]] = []
    mapping: list[dict[str, Any]] = []
    for idx, case in enumerate(generated[: max(1, robustness_cases)]):
        meta = case.get("__perturbation__") if isinstance(case, dict) else {}
        case_id = str(meta.get("case_id") or f"heur_case_{idx + 1}") if isinstance(meta, dict) else f"heur_case_{idx + 1}"
        obj_scale = _expand_scale(base_bounds, factor=1.25)
        raw_changes = list(meta.get("changes", [])) if isinstance(meta, dict) else []
        patches = []
        for change in raw_changes:
            if not isinstance(change, dict):
                continue
            key = str(change.get("key", "")).strip()
            if not key:
                continue
            patches.append(
                {
                    "fid": fid_lookup.get(key, ""),
                    "key": key,
                    "old": change.get("old"),
                    "new": change.get("new"),
                    "source": "heuristic",
                }
            )
        test_cases.append(
            {
                "case_id": case_id,
                "instance": case,
                "obj_scale": obj_scale,
                "obj_bounds": obj_scale,
                "rationale": "heuristic numeric perturbation",
                "patches": patches,
                "changes": patches,
            }
        )
        mapping.append(
            {
                "case_id": case_id,
                "patches": patches,
                "feature_ids": [patch.get("fid", "") for patch in patches if patch.get("fid")],
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
        feature_catalog=feature_catalog,
        llm_raw_preview=raw_preview,
    )
def _feature_catalog_from_instance(instance: dict[str, Any]) -> list[dict[str, Any]]:
    raw = instance.get("__feature_catalog__") if isinstance(instance, dict) else None
    if isinstance(raw, list):
        catalog = [dict(item) for item in raw if isinstance(item, dict)]
        if catalog:
            return catalog
    raw_meta = instance.get("__feature_meta__") if isinstance(instance, dict) else None
    catalog: list[dict[str, Any]] = []
    if isinstance(raw_meta, list):
        for idx, item in enumerate(raw_meta, start=1):
            if not isinstance(item, dict):
                continue
            key = str(item.get("key", "")).strip()
            if not key:
                continue
            catalog.append(
                {
                    "fid": f"F{idx:02d}",
                    "key": key,
                    "value": instance.get(key),
                    "source": str(item.get("source", "")),
                    "score": int(item.get("score", 0)),
                    "snippet": str(item.get("snippet", "")).strip(),
                }
            )
    return catalog
def _compact_feature_catalog(catalog: list[dict[str, Any]], max_items: int = 20) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for row in catalog[: max(1, int(max_items))]:
        compact.append(
            {
                "fid": row.get("fid"),
                "value": row.get("value"),
                "source": row.get("source"),
                "snippet": row.get("snippet"),
            }
        )
    return compact
def _feature_catalog_map(feature_catalog: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for item in feature_catalog:
        if not isinstance(item, dict):
            continue
        fid = str(item.get("fid", "")).strip()
        key = str(item.get("key", "")).strip()
        if fid and key:
            out[fid] = item
    return out
def _key_to_fid_map(feature_catalog: list[dict[str, Any]]) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in feature_catalog:
        if not isinstance(item, dict):
            continue
        fid = str(item.get("fid", "")).strip()
        key = str(item.get("key", "")).strip()
        if fid and key:
            out[key] = fid
    return out
def _plan_skeleton(*, num_tests: int, feature_catalog: list[dict[str, Any]]) -> dict[str, Any]:
    sample_fids = [str(row.get("fid", "")) for row in feature_catalog[:2] if isinstance(row, dict) and row.get("fid")]
    tests = []
    for idx in range(max(1, int(num_tests))):
        fid = sample_fids[min(idx, len(sample_fids) - 1)] if sample_fids else "F01"
        tests.append(
            {
                "case_id": f"case_{idx + 1}",
                "patches": [{"fid": fid, "new_value": 0}],
                "obj_scale": {"kind": "interval", "lower": None, "upper": None, "reject_exact": []},
                "rationale": "",
            }
        )
    return {
        "analysis": "",
        "base_scale": {"kind": "interval", "lower": None, "upper": None, "reject_exact": []},
        "tests": tests,
    }
def _normalize_patch(item: Any, feature_map: dict[str, dict[str, Any]], target_instance: dict[str, Any]) -> dict[str, Any] | None:
    if not isinstance(item, dict):
        return None
    fid = str(item.get("fid") or item.get("feature_id") or "").strip()
    feature = feature_map.get(fid)
    if feature is None:
        return None
    key = str(feature.get("key", "")).strip()
    if not key or key not in target_instance:
        return None
    old_value = target_instance.get(key)
    if not isinstance(old_value, (int, float)) or isinstance(old_value, bool):
        return None
    op = str(item.get("op", "replace") or "replace").strip().lower()
    new_value: float | None = None
    if op == "scale":
        factor = _to_number(item.get("value"))
        if factor is None or not math.isfinite(factor):
            return None
        new_value = float(old_value) * float(factor)
    elif op == "shift":
        delta = _to_number(item.get("value"))
        if delta is None or not math.isfinite(delta):
            return None
        new_value = float(old_value) + float(delta)
    else:
        new_value = _to_number(item.get("new_value"))
        if new_value is None:
            new_value = _to_number(item.get("value"))
        if new_value is None or not math.isfinite(new_value):
            return None
        op = "replace"
    if isinstance(old_value, int) and not isinstance(old_value, bool):
        normalized_new: int | float = int(round(float(new_value)))
    else:
        normalized_new = float(new_value)
    return {
        "fid": fid,
        "key": key,
        "old": old_value,
        "new": normalized_new,
        "op": op,
        "source": str(feature.get("source", "")),
        "snippet": str(feature.get("snippet", "")).strip(),
    }
def _parse_tagged_plan(text: str | None) -> dict[str, Any] | None:
    raw = str(text or "")
    if not raw.strip():
        return None
    analysis = _extract_tag_text(raw, "analysis")
    base_scale = _parse_json_object(_extract_tag_text(raw, "base_scale"))
    tests = _parse_json_object(_extract_tag_text(raw, "tests"))
    parsed: dict[str, Any] = {"analysis": analysis or ""}
    if isinstance(base_scale, dict):
        parsed["base_scale"] = base_scale
    if isinstance(tests, list):
        parsed["tests"] = tests
    elif isinstance(tests, dict) and isinstance(tests.get("tests"), list):
        parsed["tests"] = tests.get("tests")
    return parsed if (analysis or "base_scale" in parsed or "tests" in parsed) else None
def _extract_tag_text(text: str, tag: str, min_len: int = 2) -> str:
    raw = str(text or "")
    lower_raw = raw.lower()
    open_tag = f"<{tag.lower()}>"
    close_tag = f"</{tag.lower()}>"
    start = 0
    while True:
        close_idx = lower_raw.find(close_tag, start)
        if close_idx < 0:
            return ""
        open_idx = lower_raw.rfind(open_tag, 0, close_idx)
        if open_idx < 0:
            start = close_idx + len(close_tag)
            continue
        content = raw[open_idx + len(open_tag) : close_idx].strip()
        if len(content) >= int(min_len):
            return content
        start = close_idx + len(close_tag)
def _default_scale_from_reference(reference_answer: str) -> dict[str, Any]:
    gt = _to_number(reference_answer)
    if gt is None or not math.isfinite(gt):
        return {"kind": "interval", "lower": None, "upper": None, "reject_exact": []}
    span = max(abs(gt), 1.0)
    lower = gt - 1.0 * span
    upper = gt + 1.0 * span
    reject_exact: list[float] = []
    if gt > 0:
        lower = max(lower, 0.1 * gt)
        reject_exact = [0.0]
    elif gt < 0:
        upper = min(upper, 0.1 * gt)
        reject_exact = [0.0]
    return {
        "kind": "interval",
        "lower": float(lower),
        "upper": float(upper),
        "reject_exact": reject_exact,
    }
def _expand_scale(scale: dict[str, Any], factor: float = 1.2) -> dict[str, Any]:
    normalized = _coerce_scale(scale)
    kind = str(normalized.get("kind", "interval"))
    if kind == "point":
        point = _to_number(normalized.get("point"))
        tol_abs = _to_number(normalized.get("tol_abs"))
        if point is not None:
            tol_abs = max(abs(point) * 0.2, tol_abs or 1.0) * float(factor)
            return {
                "kind": "point",
                "point": float(point),
                "tol_abs": float(tol_abs),
                "reject_exact": list(normalized.get("reject_exact", [])) if isinstance(normalized.get("reject_exact"), list) else [],
            }
    if kind == "union":
        intervals = []
        for item in normalized.get("intervals", []):
            if not isinstance(item, dict):
                continue
            lo = _to_number(item.get("lower"))
            hi = _to_number(item.get("upper"))
            if lo is None and hi is None:
                continue
            if lo is not None and hi is not None:
                center = 0.5 * (lo + hi)
                half = 0.5 * (hi - lo)
                new_half = max(1.0, abs(half) * float(factor))
                intervals.append({"lower": center - new_half, "upper": center + new_half})
            else:
                intervals.append({"lower": lo, "upper": hi})
        return {
            "kind": "union",
            "intervals": intervals,
            "reject_exact": list(normalized.get("reject_exact", [])) if isinstance(normalized.get("reject_exact"), list) else [],
        }
    lo = _to_number(normalized.get("lower"))
    hi = _to_number(normalized.get("upper"))
    if lo is not None and hi is not None:
        center = 0.5 * (lo + hi)
        half = 0.5 * (hi - lo)
        new_half = max(1.0, abs(half) * float(factor))
        lo, hi = center - new_half, center + new_half
    return {
        "kind": "interval",
        "lower": lo,
        "upper": hi,
        "reject_exact": list(normalized.get("reject_exact", [])) if isinstance(normalized.get("reject_exact"), list) else [],
    }
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
def _parse_json_object(text: str | None) -> Any:
    raw = str(text or "").strip()
    if not raw:
        return None
    if raw.startswith("```"):
        lines = raw.splitlines()
        if len(lines) >= 2 and lines[-1].strip().startswith("```"):
            raw = "\n".join(lines[1:-1]).strip()
    try:
        return json.loads(raw)
    except Exception:
        pass
    start_brace = raw.find("{")
    end_brace = raw.rfind("}")
    if start_brace >= 0 and end_brace > start_brace:
        try:
            return json.loads(raw[start_brace : end_brace + 1])
        except Exception:
            pass
    start_bracket = raw.find("[")
    end_bracket = raw.rfind("]")
    if start_bracket >= 0 and end_bracket > start_bracket:
        try:
            return json.loads(raw[start_bracket : end_bracket + 1])
        except Exception:
            pass
    return None
def _coerce_scale(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        kind = str(value.get("kind") or "interval").strip().lower()
        reject_exact = []
        raw_reject = value.get("reject_exact")
        if isinstance(raw_reject, list):
            for item in raw_reject:
                num = _to_number(item)
                if num is not None and math.isfinite(num):
                    reject_exact.append(float(num))
        if kind == "point":
            point = _to_number(value.get("point"))
            tol_abs = _to_number(value.get("tol_abs"))
            tol_rel = _to_number(value.get("tol_rel"))
            return {
                "kind": "point",
                "point": (float(point) if point is not None and math.isfinite(point) else None),
                "tol_abs": (float(tol_abs) if tol_abs is not None and math.isfinite(tol_abs) else None),
                "tol_rel": (float(tol_rel) if tol_rel is not None and math.isfinite(tol_rel) else None),
                "reject_exact": reject_exact,
            }
        if kind == "union":
            intervals: list[dict[str, float | None]] = []
            raw_intervals = value.get("intervals")
            if isinstance(raw_intervals, list):
                for item in raw_intervals:
                    if not isinstance(item, dict):
                        continue
                    lo = _to_number(item.get("lower"))
                    hi = _to_number(item.get("upper"))
                    if lo is not None and hi is not None and lo > hi:
                        lo, hi = hi, lo
                    if lo is None and hi is None:
                        continue
                    intervals.append({
                        "lower": (float(lo) if lo is not None and math.isfinite(lo) else None),
                        "upper": (float(hi) if hi is not None and math.isfinite(hi) else None),
                    })
            return {"kind": "union", "intervals": intervals, "reject_exact": reject_exact}
        lower = _to_number(value.get("lower"))
        upper = _to_number(value.get("upper"))
        if lower is not None and upper is not None and lower > upper:
            lower, upper = upper, lower
        return {
            "kind": "interval",
            "lower": (float(lower) if lower is not None and math.isfinite(lower) else None),
            "upper": (float(upper) if upper is not None and math.isfinite(upper) else None),
            "reject_exact": reject_exact,
        }
    return {"kind": "interval", "lower": None, "upper": None, "reject_exact": []}
def _has_valid_scale(scale: dict[str, Any]) -> bool:
    kind = str(scale.get("kind") or "interval")
    if kind == "point":
        return isinstance(scale.get("point"), (int, float))
    if kind == "union":
        intervals = scale.get("intervals")
        return isinstance(intervals, list) and len(intervals) > 0
    return isinstance(scale.get("lower"), (int, float)) or isinstance(scale.get("upper"), (int, float))
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
