from __future__ import annotations
import json
import math
from copy import deepcopy
from dataclasses import dataclass
from typing import Any
from verl.trainer.ttrl_or_runtime.reward.perturbation import build_perturbation_map, generate_perturbed_instances_from_map

OBJ_SCALE_RANGE_MARGIN_RATIO = 0.10


R3_BASE_SCALE_FEWSHOT = """Example A (duck transport minimization)
Task description:
There has been an oil spill in the ocean and ducks need to be taken to shore to be cleaned either by boat or by canoe. A boat can take 10 ducks per trip while a canoe can take 8 ducks per trip. Since the boats are motor powered, they take 20 minutes per trip while the canoes take 40 minutes per trip. In order to avoid further environmental damage, there can be at most 12 boat trips and at least 60% of the trips should be by canoe. If at least 300 ducks need to be taken to shore, how many of each transportation method should be used to minimize the total amount of time needed to transport the ducks?
Ground-truth objective (reference only): 1160.0
Expected output:
<analysis>
The objective is total transport time, so it must be positive and clearly away from zero. With demand at least 300 and per-trip capacity around 8 to 10, the total number of trips is on the order of dozens. Each trip contributes tens of minutes, so the objective should typically lie in the 10^3 family.
</analysis>
<base_scale>{"kind":"interval","lower":500,"upper":2800,"sign_relation":"positive","magnitude":{"min_order":2,"max_order":4,"use_abs":true},"reject_exact":[0]}</base_scale>

Example B (farmland allocation profit maximization)
Task description:
A farmer has a field of 100 acres on which he can grow wheat and corn. The farmer needs to decide how many acres to dedicate to each crop in order to maximize his profits. Each acre of wheat yields a profit of $20, while each acre of corn yields a profit of $30. However, due to market demands and farming regulations, the combined output from 3 acres of wheat and 2 acres of corn must be at least 200 units (a unit represents the yield from an acre). Given these conditions and aiming for whole numbers of acres due to practicalities, what is the maximum total profit that the farmer can make by optimally allocating his land between wheat and corn? Provide your answer in dollars, rounded to the nearest whole number.
Ground-truth objective (reference only): 3000
Expected output:
<analysis>
The objective is total profit, so it should be positive. With up to 100 acres and per-acre profit around 20 to 30, the optimal objective is naturally around thousands. A practical filter should keep broad room for feasible variations while excluding implausible near-zero values.
</analysis>
<base_scale>{"kind":"interval","lower":1200,"upper":3800,"sign_relation":"positive","magnitude":{"min_order":3,"max_order":4,"use_abs":true},"reject_exact":[0]}</base_scale>
"""


R3_TESTS_FEWSHOT = """Example A (vitamin capsules minimization)
Task description:
Kevin needs vitamins to supplement his diet. He needs to get at least 25 units of vitamin A and 40 units of vitamin B everyday. In order to do so, he can buy capsules named Special Formula and One Daily. Each capsule of Special Formula contains 4 units of vitamin A and 5 units of vitamin B. Each capsule of One Daily contains 3 units of vitamin A and 7 units of vitamin B. If the cost per Special Formula capsule is $0.50 and the cost per One Daily capsule is $0.20, how many of each should he buy to minimize his cost?
Ground-truth objective (reference only): 1.8
Feature catalog snippet (illustrative):
- F01: min vitamin A requirement
- F02: min vitamin B requirement
- F03: A units in Special Formula
- F04: B units in Special Formula
- F05: A units in One Daily
- F06: B units in One Daily
- F07: cost of Special Formula
- F08: cost of One Daily
Expected output:
<analysis>
Create both harder and easier realistic perturbation cases. Keep nutritional semantics coherent and avoid unrealistic large jumps. Objective should remain positive and usually in low single-digit cost range.
</analysis>
<tests>[
    {"case_id":"higher_requirements_and_cost_pressure","patches":[{"fid":"F01","new_value":30},{"fid":"F02","new_value":45},{"fid":"F08","new_value":0.25}],"obj_scale":{"kind":"interval","lower":2.1,"upper":3.6,"sign_relation":"positive","magnitude":{"min_order":0,"max_order":1,"use_abs":true},"reject_exact":[0]},"rationale":"Higher vitamin requirements and more expensive One Daily should increase the optimal daily cost."},
    {"case_id":"milder_requirements_and_cheaper_one_daily","patches":[{"fid":"F01","new_value":22},{"fid":"F02","new_value":36},{"fid":"F08","new_value":0.18}],"obj_scale":{"kind":"interval","lower":1.2,"upper":2.2,"sign_relation":"positive","magnitude":{"min_order":0,"max_order":1,"use_abs":true},"reject_exact":[0]},"rationale":"Lower requirements and cheaper One Daily should reduce total cost while preserving model structure."}
]
</tests>

Example B (5-city TSP minimization)
Task description:
Imagine a scenario where a salesperson has to visit five different cities, labeled as City 1, City 2, City 3, City 4, and City 5. The objective for the salesperson is to minimize the travel costs associated with visiting each city exactly once and then returning to the starting city. The salesperson can begin their journey from any of the cities.
Here are the travel costs between the cities:
- Traveling from City 1 to City 2 costs 46 units, to City 3 costs 63 units, to City 4 costs 54 units, and to City 5 costs 45 units.
- From City 2, it costs 46 units to reach City 1, 48 units to get to City 3, 50 units to reach City 4, and 51 units to travel to City 5.
- From City 3, the travel costs are 63 units to City 1, 48 units to City 2, 31 units to City 4, and 64 units to City 5.
- From City 4, the expenses are 54 units to City 1, 50 units to City 2, 31 units to City 3, and 94 units to get to City 5.
- Lastly, traveling from City 5 involves costs of 45 units to City 1, 51 units to City 2, 64 units to City 3, and 94 units to City 4.
What is the minimum total travel cost for the salesperson to visit each city exactly once and return to the starting point?
Ground-truth objective (reference only): 229.0
Feature catalog snippet (illustrative):
- F01..F10: selected directed/symmetric edge costs among city pairs
Expected output:
<analysis>
Use coherent edge-cost perturbations to create one harder and one easier routing scenario. Keep all travel costs positive and preserve TSP semantics. Objective should remain in the 10^2 scale.
</analysis>
<tests>[
    {"case_id":"harder_route_key_edges_up","patches":[{"fid":"F04","new_value":58},{"fid":"F08","new_value":42},{"fid":"F07","new_value":60}],"obj_scale":{"kind":"interval","lower":245,"upper":340,"sign_relation":"positive","magnitude":{"min_order":2,"max_order":3,"use_abs":true},"reject_exact":[0]},"rationale":"Increasing several low-impact bridge edges should push the best Hamiltonian cycle cost upward."},
    {"case_id":"easier_route_key_edges_down","patches":[{"fid":"F04","new_value":38},{"fid":"F08","new_value":24},{"fid":"F05","new_value":42}],"obj_scale":{"kind":"interval","lower":180,"upper":245,"sign_relation":"positive","magnitude":{"min_order":2,"max_order":3,"use_abs":true},"reject_exact":[0]},"rationale":"Reducing selected key edges should create cheaper tours and lower the optimum."}
]
</tests>
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

        "Goal: Produce a runtime filter(obj, scale) for the ORIGINAL sample.\n"
        "Priority: PRIORITIZE CORRECTNESS OVER PRECISION. The range should be wide enough to encompass all mathematically plausible optimal solutions, while narrow enough to reject 'absurd' or 'dimensionally impossible' values (e.g., a cost of 10^15 for a 10-item knapsack).\n\n"

        "Analysis logic in <analysis>:\n"
        "1. Identify the objective type (Min/Max) and its physical meaning.\n"
        "2. Calculate the 'Theoretical Floor' and 'Theoretical Ceiling' using extreme values(e.g., sum of all max costs, or min possible demand).\n"
        "3. Justify why the chosen scale is 'Safe' (won't reject the true optimum).\n"
        "4. While being conservative, ensure that any logically impossible values are strictly excluded from the scale.\n\n"

        "base_scale schema constraints:\n"
        "- kind: 'interval' (preferred for robustness), 'point', or 'union'.\n"
        "- sign_relation: Use 'nonnegative' or 'nonpositive' unless the math strictly forbids zero or specific signs.\n"
        "- magnitude: Use {min_order, max_order} to define a 'safety envelope'.\n"
        "- reject_exact: Only include values that are mathematically impossible (e.g., 0 in a strictly positive cost sum).\n\n"

        "Task description:\n"
        f"{description}\n\n"
        "Numeric snapshot:\n"
        f"{_compact_json_text(instance_view)}\n\n"
        "Feature catalog (context only):\n"
        f"{_compact_json_text(feature_catalog)}\n\n"
        "Output skeleton:\n"
        f"{_compact_json_text(skeleton)}\n"
        "Few-shot reference:\n"
        f"{R3_BASE_SCALE_FEWSHOT}\n\n"
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
        "You are an expert Operations Research (OR) Stress-Test Engineer. Your goal is to design robustness tests that challenge a model's sensitivity and feasibility limits. Return exactly two tagged blocks and nothing else:\n"
        "<analysis>...</analysis>\n<tests>[...JSON list...]</tests>\n\n"

        "Analysis Logic in <analysis>:\n"
        "1. Identify Sensitive Parameters: Which features (fid) impact the objective or feasibility most (e.g., bottleneck capacities, high-cost coefficients)?\n"
        "2. Scenario Design: Plan 3-5 scenarios (e.g., 'Resource Scarcity', 'Extreme Cost Variation', 'Boundary Feasibility').\n"
        "3. Objective Forecasting: For each scenario, estimate the new objective's range. Define a 'Safe Envelope' that is wide enough to be correct but strictly excludes logically impossible values (e.g., negative distances, zero cost when a base fee exists).\n\n"

        "Requirements for <tests>:\n"
        "- case_id: Descriptive (e.g., 'tight_capacity_limit_01').\n"
        "- patches: 1-3 coordinated edits using fids. Ensure edits are realistic (e.g., increasing demand while tightening supply).\n"
        "- obj_scale: Must be a robust interval. Use the schema: {kind, sign_relation, magnitude:{min_order, max_order}, reject_exact:[...]} to capture the 'Safe Envelope'.\n"
        "- rationale: Explain the 'Why' (e.g., 'Testing if the model handles 95% utilization without crashing').\n\n"

        "Principles:\n"
        "- Prefer small, meaningful perturbations over random massive changes.\n"
        "- Ensure unit consistency and semantic integrity across all patches.\n"
        "- EXCLUDE IMPOSSIBLE VALUES: The obj_scale must reflect physical reality (e.g., non-negative costs).\n\n"

        "Task description:\n"
        f"{description}\n\n"
        "Numeric snapshot:\n"
        f"{_compact_json_text(instance_view)}\n\n"
        "Feature catalog (context only):\n"
        f"{_compact_json_text(feature_catalog)}\n\n"
        "Output skeleton:\n"
        f"{_compact_json_text(skeleton)}\n"
        "Few-shot reference:\n"
        f"{R3_TESTS_FEWSHOT}\n\n"
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
        base_obj_bounds=_expand_obj_scale_margin(_default_scale_from_context(description, instance, feature_catalog)),
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
    base_bounds = _expand_obj_scale_margin(base_bounds)
    tests_raw = parsed.get("tests")
    if not isinstance(tests_raw, list):
        tests_raw = []
    test_cases: list[dict[str, Any]] = []
    mapping: list[dict[str, Any]] = []
    for idx, test in enumerate(tests_raw):
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
        obj_scale = _expand_obj_scale_margin(obj_scale)
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
    base_bounds = _expand_obj_scale_margin(_default_scale_from_context(description, base_instance, feature_catalog))
    fid_lookup = _key_to_fid_map(feature_catalog)
    test_cases: list[dict[str, Any]] = []
    mapping: list[dict[str, Any]] = []
    for idx, case in enumerate(generated[: max(1, robustness_cases)]):
        meta = case.get("__perturbation__") if isinstance(case, dict) else {}
        case_id = str(meta.get("case_id") or f"heur_case_{idx + 1}") if isinstance(meta, dict) else f"heur_case_{idx + 1}"
        obj_scale = _expand_obj_scale_margin(_expand_scale(base_bounds, factor=1.25))
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
def _expand_obj_scale_margin(scale: dict[str, Any], ratio: float = OBJ_SCALE_RANGE_MARGIN_RATIO) -> dict[str, Any]:
    normalized = _coerce_scale(scale)
    margin_ratio = max(0.0, float(ratio))
    kind = str(normalized.get("kind") or "interval")

    if kind == "point":
        # Margin expansion is only applied to explicit lower/upper bounds.
        # Point-based scales must stay unchanged.
        return dict(normalized)

    if kind == "union":
        out = dict(normalized)
        intervals_out: list[dict[str, float | None]] = []
        for item in normalized.get("intervals", []):
            if not isinstance(item, dict):
                continue
            lo = _to_number(item.get("lower"))
            hi = _to_number(item.get("upper"))
            if lo is not None:
                lo = float(lo) - abs(float(lo)) * margin_ratio
            if hi is not None:
                hi = float(hi) + abs(float(hi)) * margin_ratio
            if lo is not None and hi is not None and lo > hi:
                lo, hi = hi, lo
            intervals_out.append({"lower": lo, "upper": hi})
        out["intervals"] = intervals_out
        return out

    out = dict(normalized)
    lo = _to_number(normalized.get("lower"))
    hi = _to_number(normalized.get("upper"))
    if lo is not None:
        lo = float(lo) - abs(float(lo)) * margin_ratio
    if hi is not None:
        hi = float(hi) + abs(float(hi)) * margin_ratio
    if lo is not None and hi is not None and lo > hi:
        lo, hi = hi, lo
    out["lower"] = lo
    out["upper"] = hi
    return out
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


def _display_instance(instance: dict[str, Any]) -> dict[str, Any]:
    display: dict[str, Any] = {}
    if not isinstance(instance, dict):
        return display
    for key, value in instance.items():
        text_key = str(key)
        if text_key.startswith("__"):
            continue
        display[text_key] = value
    return display


def build_readable_test_case(case: dict[str, Any]) -> dict[str, Any]:
    return {
        "case_id": str(case.get("case_id", "")),
        "rationale": str(case.get("rationale", "")).strip(),
        "patches": list(case.get("patches", [])) if isinstance(case.get("patches"), list) else [],
        "obj_scale": case.get("obj_scale") or case.get("obj_bounds") or {},
        "obj_scale_summary": summarize_scale(case.get("obj_scale") or case.get("obj_bounds")),
        "perturbed_instance": _display_instance(case.get("instance", {})),
    }


def format_r3_precompute_markdown(
    *,
    sample_id: str,
    gold_answer: str,
    source: str,
    analysis: str,
    base_obj_bounds: dict[str, Any],
    tests: list[dict[str, Any]],
) -> str:
    lines: list[str] = []
    lines.append(f"# R3 Precompute: {sample_id}")
    lines.append("")
    lines.append(f"- gt: {gold_answer}")
    lines.append(f"- source: {source}")
    lines.append(f"- num_tests: {len(tests)}")
    lines.append(f"- base_obj_scale: {json.dumps(base_obj_bounds, ensure_ascii=False)}")
    lines.append(f"- base_scale_summary: {json.dumps(summarize_scale(base_obj_bounds), ensure_ascii=False)}")
    lines.append("")
    lines.append("## Analysis")
    lines.append(analysis or "")
    for idx, case in enumerate(tests, start=1):
        readable = build_readable_test_case(case)
        lines.append("")
        lines.append(f"## Test Case {idx}: {readable['case_id']}")
        lines.append(f"- rationale: {readable['rationale']}")
        lines.append(f"- obj_scale: {json.dumps(readable['obj_scale'], ensure_ascii=False)}")
        lines.append(f"- obj_scale_summary: {json.dumps(readable['obj_scale_summary'], ensure_ascii=False)}")
        lines.append("- patches:")
        patch_lines = json.dumps(readable["patches"], ensure_ascii=False, indent=2)
        lines.append("```json")
        lines.append(patch_lines)
        lines.append("```")
        lines.append("- perturbed_instance:")
        instance_lines = json.dumps(readable["perturbed_instance"], ensure_ascii=False, indent=2)
        lines.append("```json")
        lines.append(instance_lines)
        lines.append("```")
    lines.append("")
    return "\n".join(lines)


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







