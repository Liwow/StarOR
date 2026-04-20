#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any


NUM_RE = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


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

    out: list[Path] = []
    seen: set[str] = set()
    for item in roots:
        key = str(item.resolve())
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _discover_dataset_dirs(log_root: Path, dataset_names: list[str]) -> list[Path]:
    found: list[Path] = []
    roots = _candidate_search_roots(log_root)
    for root in roots:
        if dataset_names:
            for ds_name in dataset_names:
                candidate = root / ds_name
                if _looks_like_dataset_dir(candidate):
                    found.append(candidate)
            continue
        if _looks_like_dataset_dir(root):
            found.append(root)
        try:
            for child in sorted(root.iterdir()):
                if _looks_like_dataset_dir(child):
                    found.append(child)
        except Exception:
            pass

    out: list[Path] = []
    seen: set[str] = set()
    for item in found:
        key = str(item.resolve())
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _collect_dataset_samples(dataset_dir: Path, tol: float) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for sample_id, run_seed, item in _iter_sample_result_dirs(dataset_dir):
        obj, gt = _extract_obj_gt(item)
        rel = _rel_error(obj, gt) if obj is not None and gt is not None else None
        ok = bool(gt is not None and obj is not None and rel is not None and rel <= tol)
        sample_key = f"{sample_id}/{run_seed}" if run_seed else sample_id
        rows[sample_key] = {
            "sample_id": sample_id,
            "run_seed": run_seed,
            "sample_key": sample_key,
            "obj_answer": obj,
            "gold_answer": gt,
            "rel_error": rel,
            "within_tol": ok,
            "path": str(item.resolve()),
        }
    return rows


def _collect_all(log_root: Path, dataset_names: list[str], tol: float) -> dict[str, dict[str, dict[str, Any]]]:
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for dataset_dir in _discover_dataset_dirs(log_root, dataset_names):
        ds = dataset_dir.name
        out.setdefault(ds, {})
        samples = _collect_dataset_samples(dataset_dir, tol=tol)
        # Merge by sample_key; if duplicated across multiple model roots, latest wins.
        out[ds].update(samples)
    return out


def _dataset_names_from_args(
    dataset_names: list[str],
    datasets_csv: str,
    dataset_json: list[str],
    dataset_jsons: str,
) -> list[str]:
    names: list[str] = []

    for v in list(dataset_names):
        raw = str(v or "").strip()
        if raw:
            names.append(raw)

    if datasets_csv.strip():
        names.extend([s.strip() for s in datasets_csv.split(",") if s.strip()])

    values = list(dataset_json)
    if dataset_jsons.strip():
        values.extend([s.strip() for s in dataset_jsons.split(",") if s.strip()])
    for v in values:
        p = Path(v)
        name = p.stem if p.suffix else p.name
        if name.endswith(".jsonl"):
            name = name[:-6]
        if name:
            names.append(name)

    # De-duplicate while preserving order.
    out: list[str] = []
    seen: set[str] = set()
    for n in names:
        key = n.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(n)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare two log roots and find samples with inconsistent correctness per dataset."
    )
    parser.add_argument("--folder1", type=str, default="", help="First log folder, e.g. outputs/log1")
    parser.add_argument("--folder2", type=str, default="", help="Second log folder, e.g. outputs/log2")
    parser.add_argument("--log1", type=str, default="outputs/logs_k-8_f-true_ac-true_TTRL-true_stage-update-true_r3-true_DYNAMIC-R-true_multi-R-true_refine-true_repair-2", help="Alias of --folder1")
    parser.add_argument("--log2", type=str, default="outputs/logs_TTRL-true_r3-true_refine-true_repair-2_codeGATE-true", help="Alias of --folder2")
    parser.add_argument("--tol", type=float, default=1e-4, help="Relative error threshold for correctness")
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        help="Dataset folder name. Repeatable, e.g. --dataset OptMATH --dataset NL4OPT",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="IndustryOR_fixedV2",
        help="Comma-separated dataset folder names, e.g. OptMATH,NL4OPT",
    )
    parser.add_argument(
        "--dataset-json",
        action="append",
        default=[],
        help="Dataset json/jsonl path. Repeatable. Used to filter dataset names.",
    )
    parser.add_argument(
        "--dataset-jsons",
        type=str,
        default="",
        help="Comma-separated dataset json/jsonl paths. Used to filter dataset names.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="",
        help="Output json path. Default: outputs/compare_inconsistent_log1_vs_log2.json",
    )
    parser.add_argument(
        "--show-missing",
        action="store_true",
        help="Also include samples only present in one side.",
    )
    args = parser.parse_args()

    folder1 = str(args.folder1 or "").strip()
    folder2 = str(args.folder2 or "").strip()
    log1_arg = str(args.log1 or "").strip()
    log2_arg = str(args.log2 or "").strip()
    log1 = Path(folder1 or log1_arg)
    log2 = Path(folder2 or log2_arg)
    if not str(log1).strip() or not str(log2).strip():
        raise ValueError("Please provide both folders: --folder1 <path> --folder2 <path> (or --log1/--log2).")
    if not log1.exists():
        raise FileNotFoundError(f"log1 not found: {log1}")
    if not log2.exists():
        raise FileNotFoundError(f"log2 not found: {log2}")

    dataset_names = _dataset_names_from_args(
        dataset_names=args.dataset,
        datasets_csv=args.datasets,
        dataset_json=args.dataset_json,
        dataset_jsons=args.dataset_jsons,
    )

    left = _collect_all(log1, dataset_names=dataset_names, tol=float(args.tol))
    right = _collect_all(log2, dataset_names=dataset_names, tol=float(args.tol))

    datasets = sorted(set(left.keys()) | set(right.keys()))
    result_datasets: list[dict[str, Any]] = []
    total_inconsistent = 0

    for ds in datasets:
        a = left.get(ds, {})
        b = right.get(ds, {})
        keys_a = set(a.keys())
        keys_b = set(b.keys())
        common_keys = sorted(keys_a & keys_b)
        only_a = sorted(keys_a - keys_b)
        only_b = sorted(keys_b - keys_a)

        inconsistent: list[dict[str, Any]] = []
        for key in common_keys:
            ra = a[key]
            rb = b[key]
            if bool(ra.get("within_tol", False)) == bool(rb.get("within_tol", False)):
                continue
            inconsistent.append(
                {
                    "sample_key": key,
                    "sample_id": ra.get("sample_id") or rb.get("sample_id"),
                    "run_seed": ra.get("run_seed") or rb.get("run_seed"),
                    "log1": {
                        "within_tol": bool(ra.get("within_tol", False)),
                        "obj_answer": ra.get("obj_answer"),
                        "gold_answer": ra.get("gold_answer"),
                        "rel_error": ra.get("rel_error"),
                        "path": ra.get("path"),
                    },
                    "log2": {
                        "within_tol": bool(rb.get("within_tol", False)),
                        "obj_answer": rb.get("obj_answer"),
                        "gold_answer": rb.get("gold_answer"),
                        "rel_error": rb.get("rel_error"),
                        "path": rb.get("path"),
                    },
                }
            )

        total_inconsistent += len(inconsistent)
        payload: dict[str, Any] = {
            "dataset": ds,
            "num_log1": len(keys_a),
            "num_log2": len(keys_b),
            "num_common": len(common_keys),
            "num_inconsistent": len(inconsistent),
            "inconsistent_samples": inconsistent,
        }
        if args.show_missing:
            payload["only_in_log1"] = [a[k] for k in only_a]
            payload["only_in_log2"] = [b[k] for k in only_b]
        result_datasets.append(payload)

        print(
            f"[dataset] {ds}: common={len(common_keys)} inconsistent={len(inconsistent)} "
            f"log1={len(keys_a)} log2={len(keys_b)}"
        )

    output = {
        "log1": str(log1.resolve()),
        "log2": str(log2.resolve()),
        "tol": float(args.tol),
        "dataset_filter": dataset_names,
        "num_datasets": len(result_datasets),
        "total_inconsistent": int(total_inconsistent),
        "datasets": result_datasets,
    }

    out_path = Path(args.out) if args.out else Path("outputs/compare_inconsistent_log1_vs_log2.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[saved] {out_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
