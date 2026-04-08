from __future__ import annotations
import json
import math
from copy import deepcopy
from dataclasses import dataclass
from typing import Any
from ttrl_or.reward.perturbation import build_perturbation_map, generate_perturbed_instances_from_map


R3_BASE_SCALE_FEWSHOT = """Example A (transportation minimization)
Task sketch:
- Boat capacity 10, canoe capacity 8
- Boat time 20, canoe time 40
- At most 12 boat trips, at least 60% canoe, demand 300
Expected style:
<analysis>
Objective is total transport time, so it should stay positive and far from 0. Demand 300 with per-trip capacity around 8 to 10 implies dozens of trips, each costing tens of minutes, so a rough 10^3 scale is reasonable.
</analysis>
<base_scale>{"kind":"interval","lower":600,"upper":3000,"sign_relation":"positive","magnitude":{"min_order":2,"max_order":4,"use_abs":true},"reject_exact":[0]}</base_scale>

Example B (production planning)
Task sketch:
- Multi-week demand with labor, overtime, inventory, and delay penalties
Expected style:
<analysis>
Total cost should stay positive. Repeated labor and penalty terms across many periods usually put the objective in the 10^5 to 10^6 family, so the filter should be tolerant but still reject near-zero values.
</analysis>
<base_scale>{"kind":"interval","lower":100000,"upper":600000,"sign_relation":"positive","magnitude":{"min_order":5,"max_order":6,"use_abs":true},"reject_exact":[0]}</base_scale>
"""


R3_TESTS_FEWSHOT = """Example A (transportation minimization)
Task sketch:
- Boat capacity 10, canoe capacity 8
- Boat time 20, canoe time 40
- At most 12 boat trips, at least 60% canoe, demand 300
Feature catalog snippet:
- F03: boat capacity = 10
- F05: demand = 300
- F07: minimum canoe share = 0.60
Expected style:
<analysis>
Use realistic harder and easier cases. Prefer coordinated multi-patch edits when one scenario naturally changes several related quantities.
</analysis>
<tests>[{"case_id":"demand_up_capacity_tighten","patches":[{"fid":"F05","new_value":360},{"fid":"F03","new_value":9}],"obj_scale":{"kind":"interval","lower":800,"upper":4200,"sign_relation":"positive","magnitude":{"min_order":2,"max_order":4,"use_abs":true},"reject_exact":[0]},"rationale":"Higher demand with tighter fast-boat capacity is one coherent harder case."},{"case_id":"storm_rule_stress","patches":[{"fid":"F05","new_value":330},{"fid":"F03","new_value":9},{"fid":"F07","new_value":0.7}],"obj_scale":{"kind":"interval","lower":900,"upper":5200,"sign_relation":"positive","magnitude":{"min_order":2,"max_order":4,"use_abs":true},"reject_exact":[0]},"rationale":"A weather stress case can jointly raise demand pressure, reduce effective capacity, and tighten the canoe-share rule."}]</tests>

Example B (production planning)
Task sketch:
- Multi-week demand with training, overtime, wages, inventory, and delay penalties
Feature catalog snippet:
- F12: one late-week demand entry
- F18: overtime wage
- F21: regular-time weekly capacity
Expected style:
<analysis>
Perturb demand-like, cost-like, and capacity-like numbers in plausible business combinations. Keep obj_scale positive and in the same broad order family unless the case is intentionally extreme.
</analysis>
<tests>[{"case_id":"rush_week_tighter_capacity","patches":[{"fid":"F12","op":"scale","value":1.12},{"fid":"F18","op":"scale","value":1.05},{"fid":"F21","op":"scale","value":0.92}],"obj_scale":{"kind":"interval","lower":150000,"upper":900000,"sign_relation":"positive","magnitude":{"min_order":5,"max_order":6,"use_abs":true},"reject_exact":[0]},"rationale":"A rush-week scenario can combine higher late demand, costlier overtime, and tighter regular capacity."},{"case_id":"demand_relief_and_capacity_ease","patches":[{"fid":"F12","op":"scale","value":0.94},{"fid":"F21","op":"scale","value":1.06}],"obj_scale":{"kind":"interval","lower":90000,"upper":520000,"sign_relation":"positive","magnitude":{"min_order":5,"max_order":6,"use_abs":true},"reject_exact":[0]},"rationale":"Lower late demand with easier capacity is one coherent relief case."}]</tests>
"""

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
    llm_base_preview: str = ""
    llm_tests_preview: str = ""


def build_r3_base_scale_prompt(
    *,
    description: str,
    instance: dict[str, Any],
) -> str:
    feature_catalog = _compact_feature_catalog(_feature_catalog_from_instance(instance))
    instance_view = _compact_instance(instance)
    skeleton = _base_scale_skeleton()
    return (
        "You are an OR objective-scale analyst. Return exactly two tagged blocks and nothing else:\n"
        "<analysis>...</analysis>\n<base_scale>{...JSON...}</base_scale>\n\n"
        "Goal: produce a runtime filter(obj, scale) for the ORIGINAL sample.\n"
        "base_scale should capture: bounds, sign_relation, magnitude, and obvious exact rejects.\n"
        "Schema keys: kind(interval|point|union), sign_relation(positive|nonnegative|negative|nonpositive|nonzero|any), magnitude({min_order,max_order,use_abs:true}), reject_exact(list).\n"
        "Keep the scale informative: broad enough for modeling variation, narrow enough to reject absurd objectives.\n\n"
        "Few-shot reference:\n"
        f"{R3_BASE_SCALE_FEWSHOT}\n\n"
        "Task description:\n"
        f"{description}\n\n"
        "Numeric snapshot:\n"
        f"{_compact_json_text(instance_view)}\n\n"
        "Feature catalog (context only, no patches needed here):\n"
        f"{_compact_json_text(feature_catalog)}\n\n"
        "Output skeleton:\n"
        f"{_compact_json_text(skeleton)}\n"
    )


def build_r3_tests_prompt(
    *,
    description: str,
    instance: dict[str, Any],
    num_tests: int,
) -> str:
    feature_catalog = _compact_feature_catalog(_feature_catalog_from_instance(instance))
    instance_view = _compact_instance(instance)
    skeleton = _tests_skeleton(num_tests=num_tests, feature_catalog=feature_catalog)
    return (
        "You are an OR robustness test designer. Return exactly two tagged blocks and nothing else:\n"
        "<analysis>...</analysis>\n<tests>[...JSON list...]</tests>\n\n"
        "Each test must include case_id, patches, obj_scale, rationale. Use feature ids only.\n"
        "Each case may use 1 to 3 patches; prefer coordinated edits when they describe one plausible scenario.\n"
        "Include a mix of harder and easier cases when possible, keep units/semantics consistent, and make obj_scale describe bounds, sign_relation, and magnitude.\n"
        "Prefer small but meaningful perturbations.\n\n"
        "Few-shot reference:\n"
        f"{R3_TESTS_FEWSHOT}\n\n"
        "Task description:\n"
        f"{description}\n\n"
        "Numeric snapshot:\n"
        f"{_compact_json_text(instance_view)}\n\n"
        "Feature catalog (use fid only in patches):\n"
        f"{_compact_json_text(feature_catalog)}\n\n"
        "Output skeleton:\n"
        f"{_compact_json_text(skeleton)}\n"
    )


def build_r3_planner_prompt(
    *,
    description: str,
    instance: dict[str, Any],
    num_tests: int,
) -> str:
    return build_r3_tests_prompt(
        description=description,
        instance=instance,
        num_tests=num_tests,
    )


def build_sample_r3_plan(
    *,
    sample_id: str,
    description: str,
    instance: dict[str, Any],
    robustness_cases: int,
    llm_base_text: str | None = None,
    llm_tests_text: str | None = None,
    llm_text: str | None = None,
    allow_heuristic_fallback: bool = True,
) -> R3SamplePlan:
    feature_catalog = _feature_catalog_from_instance(instance)
    parsed: dict[str, Any] | None = None
    source = "disabled"

    base_parsed = _parse_tagged_base_scale_plan(llm_base_text) if llm_base_text else None
    tests_parsed = _parse_tagged_tests_plan(llm_tests_text) if llm_tests_text else None
    if isinstance(base_parsed, dict) or isinstance(tests_parsed, dict):
        parsed = {}
        analyses = []
        if isinstance(base_parsed, dict):
            if isinstance(base_parsed.get("base_scale"), dict):
                parsed["base_scale"] = base_parsed["base_scale"]
            if str(base_parsed.get("analysis", "")).strip():
                analyses.append(f"[base_scale] {str(base_parsed.get('analysis', '')).strip()}")
        if isinstance(tests_parsed, dict):
            if isinstance(tests_parsed.get("tests"), list):
                parsed["tests"] = tests_parsed["tests"]
            if str(tests_parsed.get("analysis", "")).strip():
                analyses.append(f"[tests] {str(tests_parsed.get('analysis', '')).strip()}")
        parsed["analysis"] = "\n\n".join(analyses).strip()
        source = "llm_split"
    elif llm_text:
        parsed = _parse_tagged_plan(llm_text)
        if isinstance(parsed, dict):
            source = "llm"

    if isinstance(parsed, dict):
        plan = _normalize_llm_plan(
            sample_id=sample_id,
            description=description,
            parsed=parsed,
            base_instance=instance,
            feature_catalog=feature_catalog,
            robustness_cases=robustness_cases,
            raw_preview=_compose_raw_preview(llm_base_text, llm_tests_text, llm_text),
        )
        if plan is not None:
            plan.source = source or plan.source
            plan.llm_base_preview = _preview(llm_base_text)
            plan.llm_tests_preview = _preview(llm_tests_text)
            return plan
    if allow_heuristic_fallback:
        return _heuristic_plan(
            sample_id=sample_id,
            description=description,
            base_instance=instance,
            feature_catalog=feature_catalog,
            robustness_cases=robustness_cases,
            raw_preview=_compose_raw_preview(llm_base_text, llm_tests_text, llm_text),
        )
    return R3SamplePlan(
        sample_id=sample_id,
        source="disabled",
        analysis="r3 precompute disabled due to extraction failure",
        base_obj_bounds=_default_scale_from_context(description, instance, feature_catalog),
        test_cases=[],
        mapping=[],
        feature_catalog=feature_catalog,
        llm_raw_preview=_compose_raw_preview(llm_base_text, llm_tests_text, llm_text),
        llm_base_preview=_preview(llm_base_text),
        llm_tests_preview=_preview(llm_tests_text),
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
    out["__r3_llm_base_preview__"] = plan.llm_base_preview
    out["__r3_llm_tests_preview__"] = plan.llm_tests_preview
    out["__r3_precompute_required__"] = True
    out["__r3_precompute_ok__"] = bool(plan.source != "disabled" and len(plan.test_cases) > 0)
    out["__r3_disable__"] = not bool(plan.source != "disabled" and len(plan.test_cases) > 0)
    return out


def _normalize_llm_plan(
    *,
    sample_id: str,
    description: str,
    parsed: dict[str, Any],
    base_instance: dict[str, Any],
    feature_catalog: list[dict[str, Any]],
    robustness_cases: int,
    raw_preview: str,
) -> R3SamplePlan | None:
    analysis = str(parsed.get("analysis", "")).strip()
    feature_map = _feature_catalog_map(feature_catalog)
    base_bounds = _coerce_scale(parsed.get("base_scale"))
    if not _has_valid_scale(base_bounds):
        base_bounds = _default_scale_from_context(description, base_instance, feature_catalog)
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
    description: str,
    base_instance: dict[str, Any],
    feature_catalog: list[dict[str, Any]],
    robustness_cases: int,
    raw_preview: str,
) -> R3SamplePlan:
    pmap = build_perturbation_map(base_instance)
    generated = generate_perturbed_instances_from_map(base_instance, pmap, max(1, robustness_cases))
    base_bounds = _default_scale_from_context(description, base_instance, feature_catalog)
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
def _short_snippet(text: Any, max_len: int = 56) -> str:
    raw = " ".join(str(text or "").split())
    if len(raw) <= max_len:
        return raw
    return raw[: max(8, max_len - 3)].rstrip() + "..."


def _compact_json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))

def _compact_feature_catalog(catalog: list[dict[str, Any]], max_items: int = 12) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for row in catalog[: max(1, int(max_items))]:
        compact.append(
            {
                "fid": row.get("fid"),
                "value": row.get("value"),
                "source": row.get("source"),
                "snippet": _short_snippet(row.get("snippet"), max_len=48),
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


def _collect_numeric_feature_values(instance: dict[str, Any], feature_catalog: list[dict[str, Any]]) -> list[float]:
    values: list[float] = []
    seen_keys: set[str] = set()
    for item in feature_catalog:
        if not isinstance(item, dict):
            continue
        key = str(item.get("key", "")).strip()
        if key:
            seen_keys.add(key)
        num = _to_number(item.get("value"))
        if num is not None and math.isfinite(num):
            values.append(float(num))
    if isinstance(instance, dict):
        for key, value in instance.items():
            text_key = str(key).strip()
            if text_key.startswith("__") or text_key in seen_keys:
                continue
            num = _to_number(value)
            if num is not None and math.isfinite(num):
                values.append(float(num))
    return values


def _default_scale_from_context(description: str, instance: dict[str, Any], feature_catalog: list[dict[str, Any]]) -> dict[str, Any]:
    numeric_values = [abs(v) for v in _collect_numeric_feature_values(instance, feature_catalog) if math.isfinite(v) and abs(v) > 1e-12]
    if not numeric_values:
        return {
            "kind": "interval",
            "lower": None,
            "upper": None,
            "sign_relation": "any",
            "magnitude": {"min_order": None, "max_order": None, "use_abs": True},
            "reject_exact": [],
        }

    numeric_values.sort()
    top_values = numeric_values[-min(8, len(numeric_values)) :]
    anchor = max(1.0, float(sum(top_values)), float(numeric_values[-1]))
    smallest = float(numeric_values[0])
    max_order = int(math.floor(math.log10(max(anchor, 1e-12)))) + 2
    min_order = int(math.floor(math.log10(max(smallest, 1e-12)))) - 1
    lowered = str(description or "").lower()
    sign_relation = "any"
    if any(token in lowered for token in ("minimize", "maximize", "cost", "profit", "revenue", "penalty", "budget", "time", "transport", "production")):
        sign_relation = "nonnegative"
    lower = 0.0 if sign_relation == "nonnegative" else -4.0 * anchor
    upper = 4.0 * anchor
    return {
        "kind": "interval",
        "lower": float(lower),
        "upper": float(upper),
        "sign_relation": sign_relation,
        "magnitude": {"min_order": min_order, "max_order": max_order, "use_abs": True},
        "reject_exact": [],
    }
def _base_scale_skeleton() -> dict[str, Any]:
    return {
        "analysis": "",
        "base_scale": {
            "kind": "interval",
            "lower": None,
            "upper": None,
            "sign_relation": "any",
            "magnitude": {"min_order": None, "max_order": None, "use_abs": True},
            "reject_exact": [],
        },
    }


def _tests_skeleton(*, num_tests: int, feature_catalog: list[dict[str, Any]]) -> dict[str, Any]:
    sample_fids = [str(row.get("fid", "")) for row in feature_catalog[:3] if isinstance(row, dict) and row.get("fid")]
    tests = []
    for idx in range(max(1, int(num_tests))):
        patch_fids = sample_fids[: min(len(sample_fids), 3)] if sample_fids else ["F01"]
        patches = []
        for patch_idx, fid in enumerate(patch_fids):
            patches.append(
                {
                    "fid": fid,
                    "op": "scale",
                    "value": (1.08 if patch_idx == 0 else (0.94 if patch_idx == 1 else 1.03)),
                }
            )
        tests.append(
            {
                "case_id": ("stress_case" if idx % 2 == 0 else "relief_case") + f"_{idx + 1}",
                "patches": patches,
                "obj_scale": {
                    "kind": "interval",
                    "lower": None,
                    "upper": None,
                    "sign_relation": "any",
                    "magnitude": {"min_order": None, "max_order": None, "use_abs": True},
                    "reject_exact": [],
                },
                "rationale": "",
            }
        )
    return {"analysis": "", "tests": tests}


def _plan_skeleton(*, num_tests: int, feature_catalog: list[dict[str, Any]]) -> dict[str, Any]:
    out = _base_scale_skeleton()
    out["tests"] = _tests_skeleton(num_tests=num_tests, feature_catalog=feature_catalog)["tests"]
    return out
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
def _parse_tagged_base_scale_plan(text: str | None) -> dict[str, Any] | None:
    raw = str(text or "")
    if not raw.strip():
        return None
    analysis = _extract_tag_text(raw, "analysis")
    base_scale = _parse_json_object(_extract_tag_text(raw, "base_scale"))
    parsed: dict[str, Any] = {"analysis": analysis or ""}
    if isinstance(base_scale, dict):
        parsed["base_scale"] = base_scale
    return parsed if (analysis or "base_scale" in parsed) else None


def _parse_tagged_tests_plan(text: str | None) -> dict[str, Any] | None:
    raw = str(text or "")
    if not raw.strip():
        return None
    analysis = _extract_tag_text(raw, "analysis")
    tests = _parse_json_object(_extract_tag_text(raw, "tests"))
    parsed: dict[str, Any] = {"analysis": analysis or ""}
    if isinstance(tests, list):
        parsed["tests"] = tests
    elif isinstance(tests, dict) and isinstance(tests.get("tests"), list):
        parsed["tests"] = tests.get("tests")
    return parsed if (analysis or "tests" in parsed) else None


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
def _expand_scale(scale: dict[str, Any], factor: float = 1.2) -> dict[str, Any]:
    normalized = _coerce_scale(scale)
    sign_relation = str(normalized.get("sign_relation") or "any")
    magnitude = _expand_magnitude(dict(normalized.get("magnitude") or {}), step=1)
    reject_exact = list(normalized.get("reject_exact", [])) if isinstance(normalized.get("reject_exact"), list) else []
    kind = str(normalized.get("kind", "interval"))
    if kind == "point":
        point = _to_number(normalized.get("point"))
        tol_abs = _to_number(normalized.get("tol_abs"))
        tol_rel = _to_number(normalized.get("tol_rel"))
        if point is not None:
            tol_abs = max(abs(point) * 0.2, tol_abs or 1.0) * float(factor)
            return {
                "kind": "point",
                "point": float(point),
                "tol_abs": float(tol_abs),
                "tol_rel": (float(tol_rel) if tol_rel is not None and math.isfinite(tol_rel) else None),
                "sign_relation": sign_relation,
                "magnitude": magnitude,
                "reject_exact": reject_exact,
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
            "sign_relation": sign_relation,
            "magnitude": magnitude,
            "reject_exact": reject_exact,
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
        "sign_relation": sign_relation,
        "magnitude": magnitude,
        "reject_exact": reject_exact,
    }

def _default_magnitude_from_value(value: float | None) -> dict[str, Any]:
    if value is None or not math.isfinite(value) or abs(value) < 1e-12:
        return {"min_order": None, "max_order": None, "use_abs": True}
    order = int(math.floor(math.log10(abs(float(value)))))
    return {"min_order": order - 1, "max_order": order + 1, "use_abs": True}

def _expand_magnitude(magnitude: dict[str, Any], step: int = 1) -> dict[str, Any]:
    min_order = magnitude.get("min_order") if isinstance(magnitude.get("min_order"), int) else _to_number(magnitude.get("min_order"))
    max_order = magnitude.get("max_order") if isinstance(magnitude.get("max_order"), int) else _to_number(magnitude.get("max_order"))
    use_abs = bool(magnitude.get("use_abs", True))
    min_order = int(min_order) if min_order is not None and math.isfinite(float(min_order)) else None
    max_order = int(max_order) if max_order is not None and math.isfinite(float(max_order)) else None
    if min_order is not None:
        min_order -= int(step)
    if max_order is not None:
        max_order += int(step)
    if min_order is not None and max_order is not None and min_order > max_order:
        min_order, max_order = max_order, min_order
    return {"min_order": min_order, "max_order": max_order, "use_abs": use_abs}

def _compact_instance(instance: dict[str, Any], max_items: int = 96) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    count = 0
    for k, v in instance.items():
        key = str(k)
        if key.startswith("__"):
            continue
        if isinstance(v, bool):
            continue
        if isinstance(v, (int, float)):
            compact[key] = v
            count += 1
        elif isinstance(v, str):
            compact[key] = _short_snippet(v, max_len=80)
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
    def _normalize_sign_relation(raw: Any) -> str:
        text = str(raw or "any").strip().lower()
        aliases = {
            "gt0": "positive",
            ">0": "positive",
            "positive": "positive",
            "ge0": "nonnegative",
            ">=0": "nonnegative",
            "nonnegative": "nonnegative",
            "lt0": "negative",
            "<0": "negative",
            "negative": "negative",
            "le0": "nonpositive",
            "<=0": "nonpositive",
            "nonpositive": "nonpositive",
            "ne0": "nonzero",
            "!=0": "nonzero",
            "nonzero": "nonzero",
            "any": "any",
        }
        return aliases.get(text, "any")

    def _normalize_magnitude(raw: Any) -> dict[str, Any]:
        if not isinstance(raw, dict):
            raw = {}
        min_order = _to_number(raw.get("min_order"))
        if min_order is None:
            min_order = _to_number(raw.get("order_min"))
        if min_order is None:
            min_order = _to_number(raw.get("lower_order"))
        max_order = _to_number(raw.get("max_order"))
        if max_order is None:
            max_order = _to_number(raw.get("order_max"))
        if max_order is None:
            max_order = _to_number(raw.get("upper_order"))
        use_abs = bool(raw.get("use_abs", True))
        min_order_i = int(min_order) if min_order is not None and math.isfinite(min_order) else None
        max_order_i = int(max_order) if max_order is not None and math.isfinite(max_order) else None
        if min_order_i is not None and max_order_i is not None and min_order_i > max_order_i:
            min_order_i, max_order_i = max_order_i, min_order_i
        return {"min_order": min_order_i, "max_order": max_order_i, "use_abs": use_abs}

    if isinstance(value, dict):
        kind = str(value.get("kind") or "interval").strip().lower()
        reject_exact = []
        raw_reject = value.get("reject_exact")
        if isinstance(raw_reject, list):
            for item in raw_reject:
                num = _to_number(item)
                if num is not None and math.isfinite(num):
                    reject_exact.append(float(num))
        sign_relation = _normalize_sign_relation(
            value.get("sign_relation")
            or value.get("zero_relation")
            or value.get("relation_to_zero")
            or value.get("sign")
        )
        magnitude = _normalize_magnitude(value.get("magnitude"))
        if magnitude["min_order"] is None and magnitude["max_order"] is None:
            magnitude = _normalize_magnitude(value)
        if kind == "point":
            point = _to_number(value.get("point"))
            tol_abs = _to_number(value.get("tol_abs"))
            tol_rel = _to_number(value.get("tol_rel"))
            return {
                "kind": "point",
                "point": (float(point) if point is not None and math.isfinite(point) else None),
                "tol_abs": (float(tol_abs) if tol_abs is not None and math.isfinite(tol_abs) else None),
                "tol_rel": (float(tol_rel) if tol_rel is not None and math.isfinite(tol_rel) else None),
                "sign_relation": sign_relation,
                "magnitude": magnitude,
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
            return {"kind": "union", "intervals": intervals, "sign_relation": sign_relation, "magnitude": magnitude, "reject_exact": reject_exact}
        lower = _to_number(value.get("lower"))
        upper = _to_number(value.get("upper"))
        if lower is not None and upper is not None and lower > upper:
            lower, upper = upper, lower
        return {
            "kind": "interval",
            "lower": (float(lower) if lower is not None and math.isfinite(lower) else None),
            "upper": (float(upper) if upper is not None and math.isfinite(upper) else None),
            "sign_relation": sign_relation,
            "magnitude": magnitude,
            "reject_exact": reject_exact,
        }
    return {
        "kind": "interval",
        "lower": None,
        "upper": None,
        "sign_relation": "any",
        "magnitude": {"min_order": None, "max_order": None, "use_abs": True},
        "reject_exact": [],
    }

def _has_valid_scale(scale: dict[str, Any]) -> bool:
    kind = str(scale.get("kind") or "interval")
    if kind == "point" and isinstance(scale.get("point"), (int, float)):
        return True
    if kind == "union":
        intervals = scale.get("intervals")
        if isinstance(intervals, list) and len(intervals) > 0:
            return True
    if isinstance(scale.get("lower"), (int, float)) or isinstance(scale.get("upper"), (int, float)):
        return True
    sign_relation = str(scale.get("sign_relation") or "any")
    if sign_relation != "any":
        return True
    magnitude = scale.get("magnitude") if isinstance(scale.get("magnitude"), dict) else {}
    return isinstance(magnitude.get("min_order"), int) or isinstance(magnitude.get("max_order"), int)

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
def summarize_scale(scale: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(scale, dict):
        return {"kind": "interval", "range": "unbounded", "sign_relation": "any", "magnitude": "unspecified", "reject_exact": []}
    kind = str(scale.get("kind") or "interval")
    sign_relation = str(scale.get("sign_relation") or "any")
    reject_exact = list(scale.get("reject_exact", [])) if isinstance(scale.get("reject_exact"), list) else []
    magnitude = scale.get("magnitude") if isinstance(scale.get("magnitude"), dict) else {}
    min_order = magnitude.get("min_order") if isinstance(magnitude.get("min_order"), int) else None
    max_order = magnitude.get("max_order") if isinstance(magnitude.get("max_order"), int) else None
    mag_text = "unspecified"
    if min_order is not None or max_order is not None:
        mag_text = f"10^[{min_order if min_order is not None else '-inf'}, {max_order if max_order is not None else '+inf'}]"
    if kind == "point":
        point = scale.get("point")
        tol_abs = scale.get("tol_abs")
        tol_rel = scale.get("tol_rel")
        range_text = f"point={point}, tol_abs={tol_abs}, tol_rel={tol_rel}"
    elif kind == "union":
        intervals = scale.get("intervals") if isinstance(scale.get("intervals"), list) else []
        parts = []
        for item in intervals:
            if isinstance(item, dict):
                parts.append(f"[{item.get('lower')}, {item.get('upper')}]")
        range_text = " union ".join(parts) if parts else "unbounded"
    else:
        range_text = f"[{scale.get('lower')}, {scale.get('upper')}]"
    return {
        "kind": kind,
        "range": range_text,
        "sign_relation": sign_relation,
        "magnitude": mag_text,
        "reject_exact": reject_exact,
    }


def summarize_test_case(case: dict[str, Any]) -> dict[str, Any]:
    patches = list(case.get("patches", [])) if isinstance(case.get("patches"), list) else []
    patch_summary = [
        {
            "fid": patch.get("fid"),
            "key": patch.get("key"),
            "op": patch.get("op", "replace"),
            "old": patch.get("old"),
            "new": patch.get("new"),
        }
        for patch in patches[:8]
        if isinstance(patch, dict)
    ]
    return {
        "case_id": case.get("case_id", ""),
        "rationale": str(case.get("rationale", "")).strip(),
        "num_patches": len(patches),
        "patches": patch_summary,
        "obj_scale_summary": summarize_scale(case.get("obj_scale") or case.get("obj_bounds")),
    }


def _preview(text: str | None, max_len: int = 320) -> str:
    raw = str(text or "").strip().replace("\n", " ")
    if len(raw) <= max_len:
        return raw
    return raw[: max_len - 3] + "..."


def _compose_raw_preview(
    llm_base_text: str | None,
    llm_tests_text: str | None,
    llm_text: str | None,
    *,
    max_len: int = 1200,
) -> str:
    parts: list[str] = []
    if str(llm_base_text or "").strip():
        parts.append(f"[base_scale]\n{str(llm_base_text).strip()}")
    if str(llm_tests_text or "").strip():
        parts.append(f"[tests]\n{str(llm_tests_text).strip()}")
    if str(llm_text or "").strip():
        parts.append(f"[legacy]\n{str(llm_text).strip()}")
    merged = "\n\n".join(parts).strip()
    if len(merged) <= max_len:
        return merged
    return merged[: max_len - 3] + "..."
