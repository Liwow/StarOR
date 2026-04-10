#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


NUM_RE = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


@dataclass
class SampleEval:
    sample_id: str
    run_seed: str
    obj_answer: float | None
    gold_answer: float | None
    rel_error: float | None
    within_tol: bool


def _parse_float(value: Any) -> float | None:
    if value is None:
        return None

    if isinstance(value, (int, float)):
        x = float(value)
        return x if math.isfinite(x) else None

    text = str(value).strip().replace(",", "")
    if not text:
        return None

    try:
        x = float(text)
        return x if math.isfinite(x) else None
    except ValueError:
        pass

    m = NUM_RE.search(text)
    if not m:
        return None
    try:
        x = float(m.group(0))
        return x if math.isfinite(x) else None
    except ValueError:
        return None


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _extract_obj_gt(sample_dir: Path) -> tuple[float | None, float | None]:
    result_path = sample_dir / "result.json"
    if not result_path.exists():
        return None, None

    result = _read_json(result_path)
    obj = _parse_float(result.get("obj_answer"))
    gt = _parse_float(result.get("gold_answer"))

    if obj is not None and gt is not None:
        return obj, gt

    selected_path = sample_dir / "selected_trajectory.json"
    if selected_path.exists():
        try:
            selected = _read_json(selected_path)
            if gt is None:
                gt = _parse_float(selected.get("gt"))
            if obj is None:
                reward = selected.get("reward") or {}
                meta = reward.get("metadata") or {}
                obj = _parse_float(meta.get("obj_answer"))
                if obj is None:
                    exec_info = meta.get("execution") or {}
                    obj = _parse_float(exec_info.get("parsed_obj_answer"))
        except Exception:
            pass

    return obj, gt


def _rel_error(obj: float, gt: float) -> float:
    denom = max(abs(gt), 1e-12)
    return abs(obj - gt) / denom


def _infer_model_root(dataset_dir: Path, log_root: Path) -> str:
    try:
        rel = dataset_dir.resolve().relative_to(log_root.resolve())
        parts = rel.parts
        if len(parts) >= 2 and parts[0].startswith("model_"):
            return parts[0]
    except Exception:
        pass
    parent = dataset_dir.parent.name
    if parent.startswith("model_"):
        return parent
    return ""


def _iter_sample_result_dirs(dataset_dir: Path) -> list[tuple[str, str, Path]]:
    out: list[tuple[str, str, Path]] = []
    try:
        for sample_dir in sorted(dataset_dir.iterdir()):
            if not sample_dir.is_dir():
                continue
            direct_result = sample_dir / "result.json"
            if direct_result.exists():
                out.append((sample_dir.name, "", sample_dir))
                continue
            for run_dir in sorted(sample_dir.iterdir()):
                if not run_dir.is_dir():
                    continue
                if not run_dir.name.startswith("run_"):
                    continue
                if (run_dir / "result.json").exists():
                    out.append((sample_dir.name, run_dir.name, run_dir))
    except Exception:
        return out
    return out


def _compute_metrics(rows: list[SampleEval], tol: float, limit: int = 0) -> dict[str, Any]:
    total = len(rows)
    numeric = sum(1 for r in rows if r.gold_answer is not None)
    hits = sum(1 for r in rows if r.within_tol)
    return {
        "tol": float(tol),
        "limit": int(limit),
        "num_samples": int(total),
        "num_numeric_pairs": int(numeric),
        "num_within_tol": int(hits),
        "accuracy_over_all": (hits / total if total else 0.0),
        "accuracy_over_numeric": (hits / numeric if numeric else 0.0),
    }


def eval_dataset_dir(dataset_dir: Path, tol: float, log_root: Path, limit: int = 0) -> dict[str, Any]:
    rows: list[SampleEval] = []

    sample_dirs = _iter_sample_result_dirs(dataset_dir)
    if limit > 0:
        sample_dirs = sample_dirs[:limit]

    for sample_id, run_seed, item in sample_dirs:
        if not item.is_dir():
            continue
        if not (item / "result.json").exists():
            continue

        obj, gt = _extract_obj_gt(item)
        rel = _rel_error(obj, gt) if obj is not None and gt is not None else None
        ok = bool(gt is not None and obj is not None and rel is not None and rel <= tol)
        rows.append(
            SampleEval(
                sample_id=sample_id,
                run_seed=run_seed,
                obj_answer=obj,
                gold_answer=gt,
                rel_error=rel,
                within_tol=ok,
            )
        )

    metrics_all = _compute_metrics(rows, tol=tol, limit=limit)
    rows_no_seed = [r for r in rows if not r.run_seed]
    metrics_no_seed = _compute_metrics(rows_no_seed, tol=tol, limit=limit)

    run_seed_groups: dict[str, list[SampleEval]] = {}
    for row in rows:
        if not row.run_seed:
            continue
        run_seed_groups.setdefault(row.run_seed, []).append(row)
    run_seed_summaries = []
    for run_seed in sorted(run_seed_groups.keys()):
        seed_rows = run_seed_groups[run_seed]
        seed_metrics = _compute_metrics(seed_rows, tol=tol, limit=limit)
        run_seed_summaries.append(
            {
                "run_seed": run_seed,
                **seed_metrics,
                "samples": [
                    {
                        "sample_id": r.sample_id,
                        "sample_key": f"{r.sample_id}/{run_seed}",
                        "run_seed": run_seed,
                        "obj_answer": r.obj_answer,
                        "gold_answer": r.gold_answer,
                        "rel_error": r.rel_error,
                        "within_tol": r.within_tol,
                    }
                    for r in seed_rows
                ],
            }
        )

    return {
        "dataset": dataset_dir.name,
        "model_root": _infer_model_root(dataset_dir, log_root),
        "dataset_dir": str(dataset_dir.resolve()),
        **metrics_all,
        "has_run_seed": bool(run_seed_summaries),
        "no_run_seed": {
            **metrics_no_seed,
            "samples": [
                {
                    "sample_id": r.sample_id,
                    "sample_key": r.sample_id,
                    "run_seed": "",
                    "obj_answer": r.obj_answer,
                    "gold_answer": r.gold_answer,
                    "rel_error": r.rel_error,
                    "within_tol": r.within_tol,
                }
                for r in rows_no_seed
            ],
        },
        "run_seeds": run_seed_summaries,
        "samples": [
            {
                "sample_id": r.sample_id,
                "sample_key": (f"{r.sample_id}/{r.run_seed}" if r.run_seed else r.sample_id),
                "run_seed": r.run_seed,
                "obj_answer": r.obj_answer,
                "gold_answer": r.gold_answer,
                "rel_error": r.rel_error,
                "within_tol": r.within_tol,
            }
            for r in rows
        ],
    }


def _dataset_names_from_args(dataset_json: list[str], dataset_jsons: str) -> list[str]:
    values = list(dataset_json)
    if dataset_jsons.strip():
        values.extend([s.strip() for s in dataset_jsons.split(",") if s.strip()])

    names: list[str] = []
    for v in values:
        p = Path(v)
        name = p.stem if p.suffix else p.name
        if name.endswith(".jsonl"):
            name = name[:-6]
        names.append(name)
    return names


def _looks_like_sample_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    if (path / "result.json").exists():
        return True
    try:
        return any(
            x.is_dir() and x.name.startswith("run_") and (x / "result.json").exists()
            for x in path.iterdir()
        )
    except Exception:
        return False


def _looks_like_dataset_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    try:
        return any(_looks_like_sample_dir(x) for x in path.iterdir() if x.is_dir())
    except Exception:
        return False


def _candidate_search_roots(log_root: Path) -> list[Path]:
    roots: list[Path] = []
    if log_root.exists() and log_root.is_dir():
        roots.append(log_root)
        try:
            for child in sorted(log_root.iterdir()):
                if child.is_dir() and child.name.startswith("model_"):
                    roots.append(child)
        except Exception:
            pass
    # deduplicate while preserving order
    out: list[Path] = []
    seen: set[str] = set()
    for item in roots:
        key = str(item.resolve())
        if key not in seen:
            seen.add(key)
            out.append(item)
    return out


def _find_named_dataset_dirs_under(root: Path, dataset_names: list[str]) -> list[Path]:
    found: list[Path] = []
    for name in dataset_names:
        direct = root / name
        if _looks_like_dataset_dir(direct):
            found.append(direct)
    return found


def _discover_dataset_dirs_under(root: Path) -> list[Path]:
    out: list[Path] = []
    if _looks_like_dataset_dir(root):
        return [root]
    try:
        for child in sorted(root.iterdir()):
            if _looks_like_dataset_dir(child):
                out.append(child)
    except Exception:
        pass
    return out


def _find_dataset_dirs(log_root: Path, dataset_names: list[str]) -> list[Path]:
    search_roots = _candidate_search_roots(log_root)
    found: list[Path] = []

    if dataset_names:
        for root in search_roots:
            found.extend(_find_named_dataset_dirs_under(root, dataset_names))
    else:
        for root in search_roots:
            found.extend(_discover_dataset_dirs_under(root))

    # deduplicate resolved paths while preserving order
    out: list[Path] = []
    seen: set[str] = set()
    for item in found:
        key = str(item.resolve())
        if key not in seen:
            seen.add(key)
            out.append(item)
    return out


def _print_table(models: list[dict[str, Any]]) -> None:
    headers = [
        "level",
        "model_root",
        "dataset",
        "run_seed",
        "tol",
        "limit",
        "num_samples",
        "num_numeric_pairs",
        "num_within_tol",
        "acc_all",
        "acc_numeric",
    ]
    print("\t".join(headers))
    for model in models:
        if not isinstance(model, dict):
            continue
        print(
            "\t".join(
                [
                    "model_total",
                    str(model.get("model_root", "")),
                    "-",
                    "-",
                    str(model.get("tol", "")),
                    str(model.get("limit", "")),
                    str(model.get("num_samples", 0)),
                    str(model.get("num_numeric_pairs", 0)),
                    str(model.get("num_within_tol", 0)),
                    f"{float(model.get('accuracy_over_all', 0.0)):.4f}",
                    f"{float(model.get('accuracy_over_numeric', 0.0)):.4f}",
                ]
            )
        )
        for s in (model.get("datasets", []) if isinstance(model.get("datasets", []), list) else []):
            if not isinstance(s, dict):
                continue
            print(
                "\t".join(
                    [
                        "dataset_total",
                        str(s.get("model_root", "")),
                        str(s["dataset"]),
                        "-",
                        str(s.get("tol", "")),
                        str(s.get("limit", "")),
                        str(s["num_samples"]),
                        str(s["num_numeric_pairs"]),
                        str(s["num_within_tol"]),
                        f"{float(s['accuracy_over_all']):.4f}",
                        f"{float(s['accuracy_over_numeric']):.4f}",
                    ]
                )
            )

            no_seed = s.get("no_run_seed", {}) if isinstance(s.get("no_run_seed", {}), dict) else {}
            if int(no_seed.get("num_samples", 0) or 0) > 0:
                print(
                    "\t".join(
                        [
                            "dataset_no_run_seed",
                            str(s.get("model_root", "")),
                            str(s["dataset"]),
                            "(none)",
                            str(no_seed.get("tol", s.get("tol", ""))),
                            str(no_seed.get("limit", s.get("limit", ""))),
                            str(no_seed.get("num_samples", 0)),
                            str(no_seed.get("num_numeric_pairs", 0)),
                            str(no_seed.get("num_within_tol", 0)),
                            f"{float(no_seed.get('accuracy_over_all', 0.0)):.4f}",
                            f"{float(no_seed.get('accuracy_over_numeric', 0.0)):.4f}",
                        ]
                    )
                )

            for run_info in (s.get("run_seeds", []) if isinstance(s.get("run_seeds", []), list) else []):
                if not isinstance(run_info, dict):
                    continue
                print(
                    "\t".join(
                        [
                            "dataset_run_seed",
                            str(s.get("model_root", "")),
                            str(s["dataset"]),
                            str(run_info.get("run_seed", "")),
                            str(run_info.get("tol", s.get("tol", ""))),
                            str(run_info.get("limit", s.get("limit", ""))),
                            str(run_info.get("num_samples", 0)),
                            str(run_info.get("num_numeric_pairs", 0)),
                            str(run_info.get("num_within_tol", 0)),
                            f"{float(run_info.get('accuracy_over_all', 0.0)):.4f}",
                            f"{float(run_info.get('accuracy_over_numeric', 0.0)):.4f}",
                        ]
                    )
                )


def _group_by_model(summaries: list[dict[str, Any]], tol: float, limit: int) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for s in summaries:
        model = str(s.get("model_root", "") or "")
        grouped.setdefault(model, []).append(s)

    out: list[dict[str, Any]] = []
    for model in sorted(grouped.keys()):
        items = grouped[model]
        model_total_samples = sum(int(x.get("num_samples", 0) or 0) for x in items)
        model_total_numeric = sum(int(x.get("num_numeric_pairs", 0) or 0) for x in items)
        model_total_hits = sum(int(x.get("num_within_tol", 0) or 0) for x in items)
        model_payload = {
            "model_root": model,
            "tol": float(tol),
            "limit": int(limit),
            "num_datasets": len(items),
            "num_samples": int(model_total_samples),
            "num_numeric_pairs": int(model_total_numeric),
            "num_within_tol": int(model_total_hits),
            "accuracy_over_all": (model_total_hits / model_total_samples if model_total_samples else 0.0),
            "accuracy_over_numeric": (model_total_hits / model_total_numeric if model_total_numeric else 0.0),
            "datasets": sorted(items, key=lambda x: str(x.get("dataset", ""))),
        }
        out.append(model_payload)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute per-dataset accuracy from TTRL-OR logs: |obj-gt|/|gt| <= tol"
    )
    parser.add_argument("--log-root", type=str, default="logs/run", help="Root log directory")
    parser.add_argument(
        "--dataset-json",
        action="append",
        default=[],
        help="Dataset json/jsonl path. Repeatable.",
    )
    parser.add_argument(
        "--dataset-jsons",
        type=str,
        default="",
        help="Comma-separated dataset json/jsonl paths.",
    )
    parser.add_argument("--tol", type=float, default=0.01, help="Relative error threshold")
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Maximum number of samples to evaluate per dataset (0 = all).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="",
        help="Optional output json path (default: <log-root>/accuracy_summary.json)",
    )
    args = parser.parse_args()

    log_root = Path(args.log_root)
    if not log_root.exists():
        raise FileNotFoundError(f"log root not found: {log_root}")

    dataset_jsons_arg = args.dataset_jsons
    if not dataset_jsons_arg.strip():
        dataset_jsons_arg = os.environ.get("DATASET_JSON", "")

    dataset_names = _dataset_names_from_args(args.dataset_json, dataset_jsons_arg)
    dataset_dirs = _find_dataset_dirs(log_root, dataset_names)
    if not dataset_dirs:
        print("No dataset log dirs found.")
        return 1

    summaries = [
        eval_dataset_dir(d, tol=float(args.tol), log_root=log_root, limit=max(0, int(args.limit)))
        for d in dataset_dirs
    ]

    total_samples = sum(int(s["num_samples"]) for s in summaries)
    total_numeric = sum(int(s["num_numeric_pairs"]) for s in summaries)
    total_hits = sum(int(s["num_within_tol"]) for s in summaries)

    overall = {
        "tol": float(args.tol),
        "limit": max(0, int(args.limit)),
        "num_datasets": len(summaries),
        "num_samples": total_samples,
        "num_numeric_pairs": total_numeric,
        "num_within_tol": total_hits,
        "accuracy_over_all": (total_hits / total_samples if total_samples else 0.0),
        "accuracy_over_numeric": (total_hits / total_numeric if total_numeric else 0.0),
    }

    models = _group_by_model(summaries, tol=float(args.tol), limit=max(0, int(args.limit)))

    payload = {
        "overall": overall,
        "models": models,
        "datasets": summaries,
    }

    _print_table(models)
    print(
        "overall\t-\t-\t-\t{tol}\t{limit}\t{num_samples}\t{num_numeric_pairs}\t{num_within_tol}\t{accuracy_over_all:.4f}\t{accuracy_over_numeric:.4f}".format(
            **overall
        )
    )

    out_path = Path(args.out) if args.out else (log_root / "accuracy_summary.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved -> {out_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
