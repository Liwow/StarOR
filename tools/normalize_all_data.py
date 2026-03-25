from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ttrl_or.dataset import normalize_dataset_to_jsonl


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Batch normalize all JSONL files in input directory to unified JSONL files."
    )
    parser.add_argument("--input-dir", type=str, default="data", help="Input directory containing *.jsonl")
    parser.add_argument("--output-dir", type=str, default="data/unified", help="Output directory")
    parser.add_argument(
        "--glob",
        type=str,
        default="*.jsonl",
        help="Glob pattern for input files (default: *.jsonl)",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(in_dir.glob(args.glob))
    files = [p for p in files if p.is_file() and p.parent.resolve() != out_dir.resolve()]

    if not files:
        print(json.dumps({"message": "No files matched.", "input_dir": str(in_dir.resolve())}, ensure_ascii=False, indent=2))
        return 0

    per_file: list[dict] = []
    total = 0
    total_table = 0
    total_inline = 0

    for src in files:
        dst = out_dir / f"{src.stem}.unified.jsonl"
        stats = normalize_dataset_to_jsonl(src, dst)
        per_file.append(stats)
        total += int(stats.get("total", 0))
        total_table += int(stats.get("table_mode", 0))
        total_inline += int(stats.get("inline_mode", 0))

    summary = {
        "input_dir": str(in_dir.resolve()),
        "output_dir": str(out_dir.resolve()),
        "files": len(per_file),
        "total_samples": total,
        "total_table_mode": total_table,
        "total_inline_mode": total_inline,
        "per_file": per_file,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
