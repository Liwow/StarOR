from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


_TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")
_SEPARATOR_RE = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)+\|?\s*$")
_NUMBER_RE = re.compile(r"[-+]?\d{1,3}(?:,\d{3})*(?:\.\d+)?|[-+]?\d+(?:\.\d+)?")

_KEYWORD_HINTS: tuple[str, ...] = (
    "minimize",
    "maximize",
    "objective",
    "constraint",
    "subject to",
    "at least",
    "at most",
    "no more than",
    "no less than",
    "cannot exceed",
    "must",
    "capacity",
    "demand",
    "budget",
    "cost",
    "profit",
    "revenue",
    "penalty",
    "bound",
    "limit",
)

_RELATION_HINTS: tuple[str, ...] = ("<=", ">=", "<", ">", "=", "at least", "at most")


@dataclass(slots=True)
class UnifiedSample:
    sample_id: str
    dataset: str
    question: str
    answer: str
    difficulty: str | None = None
    is_checked: int | None = None
    param_mode: str = "inline"
    tables: list[dict[str, Any]] = field(default_factory=list)
    inline_numbers: list[float] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("raw", None)
        return payload


@dataclass(slots=True)
class RawTaskSample:
    sample_id: str
    dataset: str
    question: str
    answer: str
    instance: dict[str, Any]
    param_mode: str
    tables: list[dict[str, Any]] = field(default_factory=list)
    inline_numbers: list[float] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)


def load_jsonl_dataset(path: str | Path) -> list[UnifiedSample]:
    p = Path(path)
    dataset_name = p.stem
    samples: list[UnifiedSample] = []

    with p.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            question = str(item.get("en_question", item.get("question", ""))).strip()
            answer = str(item.get("en_answer", item.get("answer", ""))).strip()

            tables = extract_markdown_tables(question)
            param_mode = "table" if tables else "inline"
            inline_numbers = [] if tables else extract_inline_numbers(question)

            sample_id = _build_sample_id(item=item, dataset_name=dataset_name, fallback_index=idx)

            samples.append(
                UnifiedSample(
                    sample_id=sample_id,
                    dataset=dataset_name,
                    question=question,
                    answer=answer,
                    difficulty=_as_optional_str(item.get("difficulty")),
                    is_checked=_as_optional_int(item.get("is_checked")),
                    param_mode=param_mode,
                    tables=tables,
                    inline_numbers=inline_numbers,
                    raw=item,
                )
            )

    return samples


def load_raw_task_dataset(
    path: str | Path,
    *,
    start_index: int = 0,
    limit: int | None = None,
    max_numeric_features: int = 256,
    key_param_top_k: int = 16,
) -> list[RawTaskSample]:
    p = Path(path)
    dataset_name = p.stem
    samples: list[RawTaskSample] = []

    with p.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx < start_index:
                continue
            if limit is not None and len(samples) >= limit:
                break

            line = line.strip()
            if not line:
                continue

            item = json.loads(line)
            question = str(item.get("en_question", item.get("question", ""))).strip()
            answer = str(item.get("en_answer", item.get("answer", ""))).strip()
            sample_id = _build_sample_id(item=item, dataset_name=dataset_name, fallback_index=idx)

            tables = extract_markdown_tables(question)
            inline_numbers = extract_inline_numbers(question)
            param_mode = "table" if tables else "inline"

            instance = build_instance_from_question(
                question,
                tables=tables,
                inline_numbers=inline_numbers,
                max_numeric_features=max_numeric_features,
                key_param_top_k=key_param_top_k,
            )
            instance["__sample_id__"] = sample_id
            instance["__dataset__"] = dataset_name

            samples.append(
                RawTaskSample(
                    sample_id=sample_id,
                    dataset=dataset_name,
                    question=question,
                    answer=answer,
                    instance=instance,
                    param_mode=param_mode,
                    tables=tables,
                    inline_numbers=inline_numbers,
                    raw=item,
                )
            )

    return samples


def build_instance_from_question(
    question: str,
    *,
    tables: list[dict[str, Any]] | None = None,
    inline_numbers: list[float] | None = None,
    max_numeric_features: int = 256,
    key_param_top_k: int = 16,
) -> dict[str, Any]:
    parsed_tables = tables if tables is not None else extract_markdown_tables(question)
    parsed_inline_numbers = inline_numbers if inline_numbers is not None else extract_inline_numbers(question)

    instance: dict[str, Any] = {}
    feature_meta: list[dict[str, Any]] = []

    mentions = _extract_number_mentions(question)
    for idx, mention in enumerate(mentions):
        if len(feature_meta) >= max_numeric_features:
            break
        key = f"num_{idx}"
        instance[key] = mention["value"]
        feature_meta.append(
            {
                "key": key,
                "value": mention["value"],
                "source": "question",
                "score": mention["score"],
                "snippet": mention["snippet"],
                "order": len(feature_meta),
            }
        )

    for table_entry in _extract_table_numeric_entries(parsed_tables):
        if len(feature_meta) >= max_numeric_features:
            break
        key = f"tbl_{table_entry['table_index']}_r{table_entry['row_index']}_{table_entry['header_slug']}"
        suffix = 1
        unique_key = key
        while unique_key in instance:
            unique_key = f"{key}_{suffix}"
            suffix += 1

        instance[unique_key] = table_entry["value"]
        feature_meta.append(
            {
                "key": unique_key,
                "value": table_entry["value"],
                "source": "table",
                "score": table_entry["score"],
                "snippet": table_entry["snippet"],
                "order": len(feature_meta),
            }
        )

    ranked = sorted(feature_meta, key=lambda x: (-int(x["score"]), int(x["order"])))
    key_params: list[str] = []
    for row in ranked:
        key = str(row["key"])
        if key not in key_params:
            key_params.append(key)
        if len(key_params) >= key_param_top_k:
            break

    if not key_params:
        key_params = [row["key"] for row in feature_meta[: min(key_param_top_k, len(feature_meta))]]

    feature_catalog = _build_feature_catalog(feature_meta)

    instance["__param_mode__"] = "table" if parsed_tables else "inline"
    instance["__inline_numbers__"] = parsed_inline_numbers
    instance["__table_count__"] = len(parsed_tables)
    instance["__num_numeric_features__"] = len(feature_meta)
    instance["__key_param_keys__"] = key_params
    instance["__feature_meta__"] = [
        {
            "key": row["key"],
            "source": row["source"],
            "score": row["score"],
            "snippet": row["snippet"],
        }
        for row in feature_meta[:64]
    ]
    instance["__feature_catalog__"] = feature_catalog
    instance["__feature_fid_to_key__"] = {
        str(row["fid"]): str(row["key"])
        for row in feature_catalog
        if isinstance(row, dict) and isinstance(row.get("fid"), str) and isinstance(row.get("key"), str)
    }
    return instance


def _build_feature_catalog(feature_meta: list[dict[str, Any]], max_items: int = 64) -> list[dict[str, Any]]:
    catalog: list[dict[str, Any]] = []
    for idx, row in enumerate(feature_meta[: max(1, int(max_items))], start=1):
        key = str(row.get("key", "")).strip()
        if not key:
            continue
        catalog.append(
            {
                "fid": f"F{idx:02d}",
                "key": key,
                "value": row.get("value"),
                "source": str(row.get("source", "")),
                "score": int(row.get("score", 0)),
                "snippet": str(row.get("snippet", "")).strip(),
            }
        )
    return catalog


def normalize_dataset_to_jsonl(input_path: str | Path, output_path: str | Path) -> dict[str, Any]:
    samples = load_jsonl_dataset(input_path)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    table_count = 0
    inline_count = 0

    with out.open("w", encoding="utf-8") as f:
        for sample in samples:
            if sample.param_mode == "table":
                table_count += 1
            else:
                inline_count += 1
            f.write(json.dumps(sample.to_json(), ensure_ascii=False) + "\n")

    return {
        "input": str(Path(input_path).resolve()),
        "output": str(out.resolve()),
        "total": len(samples),
        "table_mode": table_count,
        "inline_mode": inline_count,
    }


def extract_markdown_tables(text: str) -> list[dict[str, Any]]:
    lines = text.splitlines()
    blocks: list[list[str]] = []
    cur: list[str] = []

    for line in lines:
        if _TABLE_ROW_RE.match(line):
            cur.append(line.strip())
        else:
            if cur:
                blocks.append(cur)
                cur = []
    if cur:
        blocks.append(cur)

    tables: list[dict[str, Any]] = []
    for block in blocks:
        parsed = _parse_table_block(block)
        if parsed:
            tables.append(parsed)
    return tables


def extract_inline_numbers(text: str) -> list[float]:
    values: list[float] = []
    seen: set[float] = set()

    for token in _NUMBER_RE.findall(text):
        cleaned = token.replace(",", "")
        try:
            value = float(cleaned)
        except ValueError:
            continue
        if value in seen:
            continue
        seen.add(value)
        values.append(value)
    return values


def _extract_number_mentions(text: str) -> list[dict[str, Any]]:
    mentions: list[dict[str, Any]] = []

    for match in _NUMBER_RE.finditer(text):
        token = match.group(0)
        cleaned = token.replace(",", "")
        try:
            value = float(cleaned)
        except ValueError:
            continue

        left = max(0, match.start() - 60)
        right = min(len(text), match.end() + 60)
        snippet = text[left:right].replace("\n", " ")
        lowered = snippet.lower()

        score = 1
        if any(k in lowered for k in _KEYWORD_HINTS):
            score += 3
        if any(op in lowered for op in _RELATION_HINTS):
            score += 2
        if "%" in snippet:
            score += 1

        mentions.append(
            {
                "value": value,
                "score": score,
                "snippet": snippet.strip(),
            }
        )

    return mentions


def _extract_table_numeric_entries(tables: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for table_idx, table in enumerate(tables):
        headers = [str(h) for h in table.get("headers", [])]
        data_rows = table.get("rows", [])
        for row_idx, row in enumerate(data_rows):
            if not isinstance(row, dict):
                continue

            row_text = " ".join(str(v) for v in row.values())
            for col_idx, header in enumerate(headers):
                cell = row.get(header, "")
                number = _parse_first_number(cell)
                if number is None:
                    continue

                header_slug = _slugify(header)
                snippet = f"{header}: {cell}"
                score = 2 + _keyword_score(header) + _keyword_score(row_text)
                rows.append(
                    {
                        "table_index": table_idx,
                        "row_index": row_idx,
                        "col_index": col_idx,
                        "header_slug": header_slug or f"c{col_idx}",
                        "value": number,
                        "score": score,
                        "snippet": snippet,
                    }
                )

    return rows


def _parse_first_number(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    token = str(value)
    match = _NUMBER_RE.search(token)
    if not match:
        return None
    try:
        return float(match.group(0).replace(",", ""))
    except ValueError:
        return None


def _keyword_score(text: str) -> int:
    lowered = text.lower()
    score = 0
    for token in _KEYWORD_HINTS:
        if token in lowered:
            score += 1
    return score


def _slugify(text: str) -> str:
    lowered = text.lower().strip()
    lowered = re.sub(r"[^a-z0-9]+", "_", lowered)
    lowered = lowered.strip("_")
    if not lowered:
        return ""
    return lowered[:40]


def _parse_table_block(block: list[str]) -> dict[str, Any] | None:
    if len(block) < 2:
        return None

    header_cells = _split_row(block[0])
    sep_cells = _split_row(block[1])
    if not header_cells or not sep_cells:
        return None

    if not _is_separator_row(sep_cells):
        return None

    rows: list[dict[str, str]] = []
    for row_line in block[2:]:
        row_cells = _split_row(row_line)
        if not row_cells:
            continue

        if len(row_cells) < len(header_cells):
            row_cells += [""] * (len(header_cells) - len(row_cells))
        elif len(row_cells) > len(header_cells):
            row_cells = row_cells[: len(header_cells)]

        row = {header_cells[i]: row_cells[i] for i in range(len(header_cells))}
        rows.append(row)

    if not rows:
        return None

    return {
        "headers": header_cells,
        "rows": rows,
        "n_rows": len(rows),
        "n_cols": len(header_cells),
    }


def _split_row(line: str) -> list[str]:
    stripped = line.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    cells = [cell.strip() for cell in stripped.split("|")]
    return cells


def _is_separator_row(cells: list[str]) -> bool:
    for cell in cells:
        token = cell.replace(" ", "")
        if not token:
            return False
        if not re.fullmatch(r":?-{2,}:?", token):
            return False
    return True


def _build_sample_id(item: dict[str, Any], dataset_name: str, fallback_index: int) -> str:
    if "sample_id" in item:
        return str(item["sample_id"])
    if "id" in item:
        return f"{dataset_name}:{item['id']}"
    if "index" in item:
        return f"{dataset_name}:{item['index']}"
    return f"{dataset_name}:{fallback_index}"


def _as_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _as_optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
