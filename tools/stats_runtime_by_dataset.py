#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


RUNTIME_FILE_PRIORITY: tuple[str, ...] = (
    "runtime_summary.json",
    "runtime.json",
    "runtime_summary.md",
    "runtime.md",
    "run_summary.json",
)


def _to_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        v = float(value)
        return v if v == v else None
    if isinstance(value, str):
        text = value.strip().replace(",", "")
        if not text:
            return None
        try:
            v = float(text)
            return v if v == v else None
        except Exception:
            return None
    return None


def _to_int(value: Any) -> int | None:
    v = _to_float(value)
    if v is None:
        return None
    try:
        return int(round(v))
    except Exception:
        return None


def _pick_runtime_file(run_dir: Path) -> Path | None:
    for name in RUNTIME_FILE_PRIORITY:
        p = run_dir / name
        if p.exists() and p.is_file():
            return p
    return None


def _looks_like_sample_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    if _pick_runtime_file(path) is not None:
        return True
    if (path / "result.json").exists():
        return True
    try:
        for child in path.iterdir():
            if child.is_dir() and (_pick_runtime_file(child) is not None or (child / "result.json").exists()):
                return True
    except Exception:
        return False
    return False


def _looks_like_dataset_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    try:
        return any(_looks_like_sample_dir(x) for x in path.iterdir() if x.is_dir())
    except Exception:
        return False


def _candidate_search_roots(input_dir: Path) -> list[Path]:
    roots: list[Path] = []
    if input_dir.exists() and input_dir.is_dir():
        roots.append(input_dir)
        try:
            for child in sorted(input_dir.iterdir()):
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


def _discover_dataset_dirs(input_dir: Path) -> list[Path]:
    found: list[Path] = []
    for root in _candidate_search_roots(input_dir):
        if _looks_like_dataset_dir(root):
            found.append(root)
            continue
        try:
            for child in sorted(root.iterdir()):
                if child.is_dir() and _looks_like_dataset_dir(child):
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


def _iter_sample_run_dirs(dataset_dir: Path) -> list[tuple[str, str, Path]]:
    out: list[tuple[str, str, Path]] = []
    try:
        for sample_dir in sorted(dataset_dir.iterdir()):
            if not sample_dir.is_dir():
                continue
            direct_runtime = _pick_runtime_file(sample_dir)
            if direct_runtime is not None or (sample_dir / "result.json").exists():
                out.append((sample_dir.name, "", sample_dir))
                continue
            for run_dir in sorted(sample_dir.iterdir()):
                if not run_dir.is_dir():
                    continue
                if _pick_runtime_file(run_dir) is not None or (run_dir / "result.json").exists():
                    out.append((sample_dir.name, run_dir.name, run_dir))
    except Exception:
        return out
    return out


def _extract_runtime_from_json(path: Path) -> tuple[int | None, float | None]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None, None

    runtime: dict[str, Any] = {}
    if isinstance(payload, dict):
        if isinstance(payload.get("runtime"), dict):
            runtime = dict(payload.get("runtime", {}))
        else:
            runtime = dict(payload)
    else:
        return None, None

    num_iterations = _to_int(runtime.get("num_iterations"))
    if num_iterations is None and isinstance(runtime.get("per_iteration"), list):
        num_iterations = len(runtime.get("per_iteration", []))

    sample_total_sec = _to_float(runtime.get("sample_total_sec"))
    if sample_total_sec is None:
        sample_total_sec = _to_float(runtime.get("total_elapsed_sec"))
    if sample_total_sec is None:
        sample_total_sec = _to_float(runtime.get("total_sec"))
    if sample_total_sec is None:
        sample_total_sec = _to_float(runtime.get("elapsed_sec"))
    if sample_total_sec is None:
        sample_total_sec = _to_float(runtime.get("iter_time_sum_sec"))

    return num_iterations, sample_total_sec


def _extract_runtime_from_markdown(path: Path) -> tuple[int | None, float | None]:
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return None, None

    def _find_scalar(key: str) -> str | None:
        # e.g. "- sample_total_sec: 12.34"
        pattern = re.compile(rf"^\s*-\s*{re.escape(key)}\s*:\s*(.+?)\s*$", flags=re.MULTILINE | re.IGNORECASE)
        m = pattern.search(text)
        return m.group(1).strip() if m else None

    num_iterations = _to_int(_find_scalar("num_iterations"))
    sample_total_sec = _to_float(_find_scalar("sample_total_sec"))
    if sample_total_sec is None:
        sample_total_sec = _to_float(_find_scalar("total_elapsed_sec"))
    if sample_total_sec is None:
        sample_total_sec = _to_float(_find_scalar("iter_time_sum_sec"))
    return num_iterations, sample_total_sec


def _read_runtime(path: Path) -> tuple[int | None, float | None]:
    if path.suffix.lower() == ".json":
        return _extract_runtime_from_json(path)
    if path.suffix.lower() == ".md":
        return _extract_runtime_from_markdown(path)
    return None, None


def _collapse_sample_like_groups(rows: list[dict[str, Any]], root_name: str) -> list[dict[str, Any]]:
    # If grouping exploded into one-row groups (likely sample_id), collapse to one dataset row.
    if not rows:
        return rows
    if len(rows) <= 20:
        return rows
    single_count_groups = sum(1 for r in rows if int(r.get("num_runs_total", 0) or 0) <= 1)
    if single_count_groups < int(0.8 * len(rows)):
        return rows

    total_runs = sum(int(r.get("num_runs_total", 0) or 0) for r in rows)
    iter_values = [float(r["avg_iterations"]) for r in rows if isinstance(r.get("avg_iterations"), (int, float))]
    time_values = [float(r["avg_sample_total_sec"]) for r in rows if isinstance(r.get("avg_sample_total_sec"), (int, float))]

    collapsed = {
        "dataset": root_name,
        "dataset_dir": "",
        "num_runs_total": int(total_runs),
        "num_runs_with_iterations": int(sum(int(r.get("num_runs_with_iterations", 0) or 0) for r in rows)),
        "num_runs_with_time": int(sum(int(r.get("num_runs_with_time", 0) or 0) for r in rows)),
        "avg_iterations": (sum(iter_values) / len(iter_values)) if iter_values else None,
        "avg_sample_total_sec": (sum(time_values) / len(time_values)) if time_values else None,
        "num_runtime_missing": int(sum(int(r.get("num_runtime_missing", 0) or 0) for r in rows)),
        "num_runtime_parse_failed": int(sum(int(r.get("num_runtime_parse_failed", 0) or 0) for r in rows)),
        "collapsed_from_sample_like_groups": True,
    }
    return [collapsed]


def collect_runtime_stats(input_dir: Path) -> dict[str, Any]:
    dataset_dirs = _discover_dataset_dirs(input_dir)
    summaries: list[dict[str, Any]] = []

    for dataset_dir in dataset_dirs:
        runs = _iter_sample_run_dirs(dataset_dir)
        iter_values: list[float] = []
        total_sec_values: list[float] = []
        missing_runtime = 0
        parse_failed = 0

        for _, _, run_dir in runs:
            runtime_file = _pick_runtime_file(run_dir)
            if runtime_file is None:
                missing_runtime += 1
                continue
            num_iterations, sample_total_sec = _read_runtime(runtime_file)
            if num_iterations is None and sample_total_sec is None:
                parse_failed += 1
                continue
            if isinstance(num_iterations, int):
                iter_values.append(float(num_iterations))
            if isinstance(sample_total_sec, (int, float)):
                total_sec_values.append(float(sample_total_sec))

        summaries.append(
            {
                "dataset": dataset_dir.name,
                "dataset_dir": str(dataset_dir.resolve()),
                "num_runs_total": int(len(runs)),
                "num_runs_with_iterations": int(len(iter_values)),
                "num_runs_with_time": int(len(total_sec_values)),
                "avg_iterations": (sum(iter_values) / len(iter_values)) if iter_values else None,
                "avg_sample_total_sec": (sum(total_sec_values) / len(total_sec_values)) if total_sec_values else None,
                "num_runtime_missing": int(missing_runtime),
                "num_runtime_parse_failed": int(parse_failed),
            }
        )

    summaries = _collapse_sample_like_groups(summaries, root_name=input_dir.name)

    all_iter_values = [float(item["avg_iterations"]) for item in summaries if isinstance(item.get("avg_iterations"), (int, float))]
    all_time_values = [float(item["avg_sample_total_sec"]) for item in summaries if isinstance(item.get("avg_sample_total_sec"), (int, float))]

    return {
        "input_dir": str(input_dir.resolve()),
        "num_datasets": int(len(summaries)),
        "datasets": summaries,
        "overall": {
            "dataset_avg_iterations_mean": (sum(all_iter_values) / len(all_iter_values)) if all_iter_values else None,
            "dataset_avg_sample_total_sec_mean": (sum(all_time_values) / len(all_time_values)) if all_time_values else None,
        },
    }


def _print_table(payload: dict[str, Any]) -> None:
    rows = list(payload.get("datasets", []))
    print(f"input_dir: {payload.get('input_dir', '')}")
    print(f"num_datasets: {payload.get('num_datasets', 0)}")
    print(
        "| dataset | runs | avg_iterations | avg_sample_total_sec | with_iter | with_time | runtime_missing | parse_failed |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        print(
            "| "
            + " | ".join(
                [
                    str(row.get("dataset", "")),
                    str(row.get("num_runs_total", "")),
                    str(row.get("avg_iterations", "")),
                    str(row.get("avg_sample_total_sec", "")),
                    str(row.get("num_runs_with_iterations", "")),
                    str(row.get("num_runs_with_time", "")),
                    str(row.get("num_runtime_missing", "")),
                    str(row.get("num_runtime_parse_failed", "")),
                ]
            )
            + " |"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "统计给定日志目录下各数据集的平均代次数和平均耗时。"
            "自动读取 runtime_summary.json/md 或 runtime.json/md。"
        )
    )
    parser.add_argument("input_dir", type=str, help="数据集日志文件夹（可为单数据集目录或包含多个数据集的上层目录）")
    parser.add_argument("--out", type=str, default="", help="可选：输出汇总 JSON 路径")
    args = parser.parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    if not input_dir.exists() or not input_dir.is_dir():
        raise SystemExit(f"input_dir is not a directory: {input_dir}")

    payload = collect_runtime_stats(input_dir)
    _print_table(payload)

    if str(args.out or "").strip():
        out_path = Path(args.out).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nSaved JSON summary to: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
