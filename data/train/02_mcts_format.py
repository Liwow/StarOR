from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ttrl_or.prompts import PromptBuilder
from ttrl_or.prompts.notice_prompts import SYSTEM_INSTRUCTION
from ttrl_or.prompts.templates import DEFAULT_ROLLOUT_TEMPLATES, DEFAULT_TEMPLATES
from ttrl_or.types import DEFAULT_STAGE_ORDER, OptimizationTask, Stage, Trajectory

FULL_PROMPT_TEMPLATE = """You are a professional optimization problem analyst, proficient in extracting key elements from optimization problems described in natural language.
I need you to help me to build  a detailed mathematical model and  provide a gurobi python code to solve it.
Please define the type, set, parameters, variables, objective, constraints, and finally the executable Gurobi Python code within <python> ... </python>.
Here is the specific description of the optimization problem:
{task_description}. 
The required order from this point is:
1. <Type>
2. <Sets>
3. <Parameters>
4. <Variables>
5. <Objective>
6. <Constraints>
7. <python>"""

STAGE_OUTPUT_ORDER: list[tuple[Stage, tuple[str, ...], str]] = [
    (Stage.SCHEMA, ("Type", "Sets"), "schema"),
    (Stage.SET_PARAM_VAR, ("Parameters", "Variables"), "set_param_var"),
    (Stage.OBJ_CONS, ("Objective", "Constraints"), "obj_cons"),
    (Stage.CODE, ("python",), "code"),
]


def sanitize_prompt_without_thought(prompt_text: str) -> str:
    """Remove <thought>/think-step instructions to match no-thought prompt policy."""
    text = str(prompt_text or "")
    lines = text.splitlines()
    cleaned: list[str] = []
    for line in lines:
        lower = line.strip().lower()
        if "<thought>" in lower:
            continue
        if "think step by step" in lower:
            continue
        if "you should think first" in lower:
            continue
        if "before you output, you should think" in lower:
            continue
        cleaned.append(line)

    text = "\n".join(cleaned)
    # Renumber bullet order lines like "2. <Type>" -> "1. <Type>" after thought removal.
    text = re.sub(r"(?m)^\s*[2-9]\.\s*<", lambda m: f"{int(m.group(0).strip().split('.')[0]) - 1}. <", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


def iter_jsonl(path: Path):
    # Use utf-8-sig to tolerate BOM-prefixed JSONL files.
    with path.open("r", encoding="utf-8-sig") as fh:
        for idx, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            yield idx, json.loads(line)



def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
        fh.flush()



def load_seen_record_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    seen: set[str] = set()
    for _, row in iter_jsonl(path):
        rid = str(row.get("record_id", "")).strip()
        if rid:
            seen.add(rid)
    return seen



def stable_record_id(record: dict[str, Any], source_index: int, *, suffix: str = "") -> str:
    base = json.dumps(
        {
            "source_index": source_index,
            "input": record.get("input", ""),
            "output": record.get("output", ""),
            "suffix": suffix,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha1(base.encode("utf-8")).hexdigest()[:20]



def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    raw = str(value).strip().lower()
    if raw in {"1", "true", "yes", "y", "on"}:
        return True
    if raw in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"invalid boolean value: {value}")



def extract_tag_block(text: str, tag: str) -> str:
    pattern = re.compile(rf"<{tag}>\s*(.*?)\s*</{tag}>", flags=re.IGNORECASE | re.DOTALL)
    match = pattern.search(text)
    return f"<{tag}>\n{match.group(1).strip()}\n</{tag}>" if match else ""



def extract_required_blocks(output_text: str) -> dict[str, str]:
    blocks: dict[str, str] = {}
    for tag in ["Type", "Sets", "Parameters", "Variables", "Objective", "Constraints", "python"]:
        block = extract_tag_block(output_text, tag)
        if not block:
            raise ValueError(f"missing <{tag}> block")
        blocks[tag] = block
    return blocks



def build_stage_outputs(blocks: dict[str, str]) -> dict[Stage, str]:
    return {
        Stage.SCHEMA: "\n\n".join([blocks["Type"], blocks["Sets"]]),
        Stage.SET_PARAM_VAR: "\n\n".join([blocks["Parameters"], blocks["Variables"]]),
        Stage.OBJ_CONS: "\n\n".join([blocks["Objective"], blocks["Constraints"]]),
        Stage.CODE: blocks["python"],
    }


def build_rollout_supervision_outputs(canonical_stage_outputs: dict[Stage, str]) -> dict[Stage, str]:
    python_block = canonical_stage_outputs[Stage.CODE]
    rollout_outputs: dict[Stage, str] = {
        Stage.CODE: python_block,
    }
    for stage in [Stage.SCHEMA, Stage.SET_PARAM_VAR, Stage.OBJ_CONS]:
        rollout_outputs[stage] = "\n\n".join([canonical_stage_outputs[stage], python_block]).strip()
    return rollout_outputs


def _clamp_probability(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _deterministic_unit_interval(record: dict[str, Any], source_index: int, suffix: str) -> float:
    key = stable_record_id(record, source_index, suffix=suffix)
    hv = hashlib.sha1(key.encode("utf-8")).hexdigest()
    numerator = int(hv[:16], 16)
    denominator = float(16**16 - 1)
    if denominator <= 0:
        return 0.0
    return float(numerator / denominator)


def should_keep_code_stage(record: dict[str, Any], source_index: int, code_stage_prob: float) -> bool:
    prob = _clamp_probability(code_stage_prob)
    if prob <= 0.0:
        return False
    if prob >= 1.0:
        return True
    u = _deterministic_unit_interval(record, source_index, suffix="code_stage_sampling")
    return bool(u < prob)



def build_mcts_records(record: dict[str, Any], source_index: int, code_stage_prob: float = 0.3) -> list[dict[str, Any]]:
    task_text = str(record.get("input", "") or "").strip()
    output_text = str(record.get("output", "") or "").strip()
    if not task_text:
        raise ValueError("missing input task text")
    if not output_text:
        raise ValueError("missing output text")

    blocks = extract_required_blocks(output_text)
    canonical_stage_outputs = build_stage_outputs(blocks)
    rollout_stage_outputs = build_rollout_supervision_outputs(canonical_stage_outputs)
    keep_code_stage = should_keep_code_stage(record, source_index, code_stage_prob=code_stage_prob)
    task = OptimizationTask(
        task_id=str(record.get("record_id", "") or f"source_{source_index}"),
        description=task_text,
    )
    builder = PromptBuilder(
        templates=DEFAULT_TEMPLATES,
        rollout_templates=DEFAULT_ROLLOUT_TEMPLATES,
        stage_order=DEFAULT_STAGE_ORDER,
        system_instruction=SYSTEM_INSTRUCTION,
    )

    records: list[dict[str, Any]] = []
    trajectory = Trajectory(trajectory_id=f"mcts_format_{source_index}")
    for stage_index, (stage, output_tags, stage_name) in enumerate(STAGE_OUTPUT_ORDER, start=1):
        if stage == Stage.CODE and not keep_code_stage:
            continue
        prompt_text = builder.build_rollout(task, stage, trajectory if trajectory.outputs else None)
        prompt_text = sanitize_prompt_without_thought(prompt_text)
        stage_output = rollout_stage_outputs[stage]
        record_id = stable_record_id(record, source_index, suffix=stage_name)
        records.append(
            {
                "record_id": record_id,
                "source_record_id": str(record.get("record_id", "") or ""),
                "source_index": source_index,
                "mode": "mcts",
                "stage_index": stage_index,
                "stage_name": stage_name,
                "stage_enum": stage.value,
                "output_tags": list(output_tags),
                "input": prompt_text,
                "output": stage_output,
                "code_stage_prob": _clamp_probability(code_stage_prob),
                "code_stage_sampled": bool(keep_code_stage),
                "task": task_text,
                "full_output": output_text,
            }
        )
        # Keep canonical stage-only content in trajectory history so later-stage prompts
        # receive the same structured context as runtime MCTS.
        trajectory.outputs[stage] = canonical_stage_outputs[stage]
    return records



def build_full_prompt_record(record: dict[str, Any], source_index: int) -> dict[str, Any]:
    task_text = str(record.get("input", "") or "").strip()
    output_text = str(record.get("output", "") or "").strip()
    if not task_text:
        raise ValueError("missing input task text")
    if not output_text:
        raise ValueError("missing output text")
    return {
        "record_id": stable_record_id(record, source_index, suffix="full_prompt"),
        "source_record_id": str(record.get("record_id", "") or ""),
        "source_index": source_index,
        "mode": "full_prompt",
        "input": sanitize_prompt_without_thought(FULL_PROMPT_TEMPLATE.format(task_description=task_text)),
        "output": output_text,
        "task": task_text,
    }



def resolve_output_path(args: argparse.Namespace) -> Path:
    if args.output:
        return Path(args.output)
    suffix = "mcts_stage" if parse_bool(args.mcts) else "full_prompt"
    return Path(f"data/train/train_data.{suffix}.jsonl")



def process(args: argparse.Namespace) -> None:
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(
            f"input file not found: {input_path}. "
            "If you want MCTS-formatted data from generated <Type>/<python> records, "
            "run `python data/train/prepare_train_data.py generate` first or pass --input explicitly."
        )
    output_path = resolve_output_path(args)
    error_path = output_path.with_suffix(output_path.suffix + ".errors.jsonl")
    resume = bool(args.resume)
    seen = load_seen_record_ids(output_path) if resume else set()
    processed = 0
    skipped = 0
    mcts_mode = parse_bool(args.mcts)

    for source_index, record in iter_jsonl(input_path):
        if args.limit is not None and processed >= args.limit:
            break
        try:
            payloads = (
                build_mcts_records(record, source_index, code_stage_prob=float(args.code_stage_prob))
                if mcts_mode
                else [build_full_prompt_record(record, source_index)]
            )
            pending = [payload for payload in payloads if str(payload.get("record_id", "")) not in seen]
            if not pending:
                skipped += 1
                continue
            for payload in pending:
                append_jsonl(output_path, payload)
                seen.add(str(payload.get("record_id", "")))
            processed += 1
            print(
                f"[mcts-format] wrote source_index={source_index} record_count={len(pending)} mode={'mcts' if mcts_mode else 'full'}",
                flush=True,
            )
        except Exception as exc:
            append_jsonl(
                error_path,
                {
                    "record_id": stable_record_id(record, source_index, suffix="error"),
                    "source_index": source_index,
                    "mode": "mcts" if mcts_mode else "full_prompt",
                    "error": str(exc),
                    "input_preview": str(record.get("input", ""))[:500],
                    "output_preview": str(record.get("output", ""))[:1000],
                },
            )
            print(f"[mcts-format] failed source_index={source_index}: {exc}", flush=True)

    print(
        f"[mcts-format] done processed={processed} skipped={skipped} output={output_path}",
        flush=True,
    )



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert prepared TTRL-OR data into MCTS-stage or full-prompt SFT format.")
    parser.add_argument("--input", default="data/train/train_data.type_python.jsonl")
    parser.add_argument("--output", default="data/train/train_data_full.jsonl")
    parser.add_argument("--mcts", default="false")
    parser.add_argument(
        "--code-stage-prob",
        type=float,
        default=0.3,
        help="When --mcts=true, probability to keep the Stage.CODE sample for each source record.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", dest="resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    return parser



def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    process(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
