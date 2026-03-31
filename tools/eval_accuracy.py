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


def eval_dataset_dir(dataset_dir: Path, tol: float) -> dict[str, Any]:
    rows: list[SampleEval] = []

    for item in sorted(dataset_dir.iterdir()):
        if not item.is_dir():
            continue
        if not (item / "result.json").exists():
            continue

        obj, gt = _extract_obj_gt(item)
        rel = _rel_error(obj, gt) if obj is not None and gt is not None else None
        ok = bool(rel is not None and rel <= tol)
        rows.append(
            SampleEval(
                sample_id=item.name,
                obj_answer=obj,
                gold_answer=gt,
                rel_error=rel,
                within_tol=ok,
            )
        )

    total = len(rows)
    numeric = sum(1 for r in rows if r.rel_error is not None)
    hits = sum(1 for r in rows if r.within_tol)

    return {
        "dataset": dataset_dir.name,
        "dataset_dir": str(dataset_dir.resolve()),
        "tol": tol,
        "num_samples": total,
        "num_numeric_pairs": numeric,
        "num_within_tol": hits,
        "accuracy_over_all": (hits / total if total else 0.0),
        "accuracy_over_numeric": (hits / numeric if numeric else 0.0),
        "samples": [
            {
                "sample_id": r.sample_id,
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


def _find_dataset_dirs(log_root: Path, dataset_names: list[str]) -> list[Path]:
    if dataset_names:
        out: list[Path] = []
        for name in dataset_names:
            d = log_root / name
            if d.exists() and d.is_dir():
                out.append(d)
        return out

    # Fallback: auto-discover all dataset folders under log_root
    out = []
    for d in sorted(log_root.iterdir()):
        if not d.is_dir():
            continue
        has_result = any((x / "result.json").exists() for x in d.iterdir() if x.is_dir())
        if has_result:
            out.append(d)
    return out


def _print_table(summaries: list[dict[str, Any]]) -> None:
    headers = [
        "dataset",
        "num_samples",
        "num_numeric_pairs",
        "num_within_tol",
        "acc_all",
        "acc_numeric",
    ]
    print("\t".join(headers))
    for s in summaries:
        print(
            "\t".join(
                [
                    str(s["dataset"]),
                    str(s["num_samples"]),
                    str(s["num_numeric_pairs"]),
                    str(s["num_within_tol"]),
                    f"{float(s['accuracy_over_all']):.4f}",
                    f"{float(s['accuracy_over_numeric']):.4f}",
                ]
            )
        )


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

    summaries = [eval_dataset_dir(d, tol=float(args.tol)) for d in dataset_dirs]

    total_samples = sum(int(s["num_samples"]) for s in summaries)
    total_numeric = sum(int(s["num_numeric_pairs"]) for s in summaries)
    total_hits = sum(int(s["num_within_tol"]) for s in summaries)

    overall = {
        "tol": float(args.tol),
        "num_datasets": len(summaries),
        "num_samples": total_samples,
        "num_numeric_pairs": total_numeric,
        "num_within_tol": total_hits,
        "accuracy_over_all": (total_hits / total_samples if total_samples else 0.0),
        "accuracy_over_numeric": (total_hits / total_numeric if total_numeric else 0.0),
    }

    payload = {
        "overall": overall,
        "datasets": summaries,
    }

    _print_table(summaries)
    print("overall\t{num_samples}\t{num_numeric_pairs}\t{num_within_tol}\t{accuracy_over_all:.4f}\t{accuracy_over_numeric:.4f}".format(**overall))

    out_path = Path(args.out) if args.out else (log_root / "accuracy_summary.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved -> {out_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
