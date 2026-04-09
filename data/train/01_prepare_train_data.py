from __future__ import annotations
import argparse
import ast
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import concurrent.futures
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ttrl_or.prompts.notice_prompts import CODE_NOTICE, SYSTEM_INSTRUCTION, TYPE_SET_NOTICE

SECTION_SPECS: list[tuple[str, str]] = [
    ("Set", "Sets"),
    ("Parameters", "Parameters"),
    ("Variables", "Variables"),
    ("Objective", "Objective"),
    ("Constraints", "Constraints"),
]
SECTION_NAME_TO_TAG = {name.lower(): tag for name, tag in SECTION_SPECS}
SECTION_HEADER_RE = re.compile(r"##\s*(Set|Parameters|Variables|Objective|Constraints)\s*:", flags=re.IGNORECASE)
TAG_BLOCK_RE = re.compile(r"<(?P<tag>[A-Za-z]+)>\s*(?P<body>.*?)\s*</(?P=tag)>", flags=re.IGNORECASE | re.DOTALL)
TYPE_BLOCK_RE = re.compile(r"<Type>\s*(.*?)\s*</Type>", flags=re.IGNORECASE | re.DOTALL)
PYTHON_BLOCK_RE = re.compile(r"<python>\s*(.*?)\s*</python>", flags=re.IGNORECASE | re.DOTALL)

TYPE_PROMPT_NOTICE = f"""We must follow the project's current DEFAULT Type-stage requirements, but for this data generation task only output the <Type> block.

The current DEFAULT Type-stage rules are:
{TYPE_SET_NOTICE}

For this dataset generation variant:
- Output exactly one block: <Type> ... </Type>
- Do NOT output <Sets>, because canonical sets are already fixed below.
- Do NOT output <thought>.
- Inside <Type>, use exactly these three bullet items in this order:
  - optimization type: LP / MILP / NLP / MINLP
  - classical OR family: closest OR family if identifiable
  - Explanation: one brief sentence explaining the modeling rationale
""".strip()

CODE_PROMPT_NOTICE = f"""We must follow the project's current DEFAULT Code-stage requirements, but for this data generation task only output the <python> block and do NOT output <thought>.

The current DEFAULT Code-stage rules are:
{CODE_NOTICE}
""".strip()


class VerificationError(RuntimeError):
    """Raised when generated code fails verification with full context payload."""

    def __init__(self, message: str, *, detail: dict[str, Any] | None = None):
        super().__init__(message)
        self.detail = detail or {}


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


def stable_record_id(record: dict[str, Any], source_index: int) -> str:
    base = json.dumps(
        {
            "source_index": source_index,
            "input": record.get("input", ""),
            "output": record.get("output", ""),
            "math_model": record.get("### Mathematical Optimization Model"),
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha1(base.encode("utf-8")).hexdigest()[:16]


def _normalize_model_dict_key(raw_key: str) -> str | None:
    match = re.search(r"##\s*(Set|Parameters|Variables|Objective|Constraints)\s*:", str(raw_key), flags=re.IGNORECASE)
    if not match:
        return None
    return match.group(1).lower()


def _clean_section_body(body: str) -> str:
    body = body.replace("\r\n", "\n").replace("\r", "\n").strip()
    return body.strip('"').strip()


def extract_sections_from_model_dict(model_dict: dict[str, Any]) -> dict[str, str]:
    sections: dict[str, str] = {}
    for raw_key, value in model_dict.items():
        norm_key = _normalize_model_dict_key(str(raw_key))
        if norm_key is None:
            continue
        if isinstance(value, list):
            body = "\n".join(str(item).strip() for item in value if str(item).strip())
        elif isinstance(value, str):
            body = value.strip()
        else:
            continue
        body = _clean_section_body(body)
        if body:
            sections[norm_key] = body
    return sections


def extract_sections_from_output_text(text: str) -> dict[str, str]:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    sections: dict[str, str] = {}
    matches = list(SECTION_HEADER_RE.finditer(text))
    for idx, match in enumerate(matches):
        name = match.group(1).lower()
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        body = _clean_section_body(text[start:end])
        if body:
            sections[name] = body
    return sections


def build_tagged_sections(record: dict[str, Any]) -> str:
    if isinstance(record.get("### Mathematical Optimization Model"), dict):
        sections = extract_sections_from_model_dict(record["### Mathematical Optimization Model"])
    else:
        output_text = str(record.get("output", "") or "")
        existing_blocks = {m.group("tag").lower(): m.group("body").strip() for m in TAG_BLOCK_RE.finditer(output_text)}
        if existing_blocks:
            sections = {}
            for _, tag in SECTION_SPECS:
                body = existing_blocks.get(tag.lower())
                if body:
                    sections[tag.lower()] = body
        else:
            sections = extract_sections_from_output_text(output_text)

    missing = [name for name, _ in SECTION_SPECS if name.lower() not in sections]
    if missing:
        raise ValueError(f"missing sections: {missing}")

    blocks = []
    for name, tag in SECTION_SPECS:
        body = sections[name.lower()].strip()
        blocks.append(f"<{tag}>\n{body}\n</{tag}>")
    return "\n\n".join(blocks)


def build_format_record(record: dict[str, Any], source_index: int) -> dict[str, Any]:
    tagged = build_tagged_sections(record)
    return {
        "record_id": stable_record_id(record, source_index),
        "source_index": source_index,
        "input": record.get("input", ""),
        "output": tagged,
        "raw_output": record.get("output", ""),
    }


def build_generation_prompt(task_text: str, canonical_blocks: str) -> str:
    sets_block = extract_tag_block(canonical_blocks, "Sets")
    sets_block_text = sets_block if sets_block else "<Sets>\n...\n</Sets>"
    return (
        "You are a professional optimization problem analyst and an optimization expert.\n"
        "We are preparing supervision data that must match the project's current DEFAULT prompt style.\n"
        "Return exactly two blocks in this order and nothing else:\n"
        "<Type> ... </Type>\n"
        "<python> ... </python>\n\n"
        f"{TYPE_PROMPT_NOTICE}\n\n"
        f"{CODE_PROMPT_NOTICE}\n\n"
        "Natural-language task:\n"
        f"{task_text.strip()}\n\n"
        "The canonical sets have already been fixed and must NOT be regenerated:\n"
        f"{sets_block_text}\n\n"
        "The canonical formulation blocks below are fixed and the code must implement them faithfully:\n"
        f"{canonical_blocks}\n\n"
        "Important reminders:\n"
        "- Do not output <Sets>, <Parameters>, <Variables>, <Objective>, or <Constraints>.\n"
        "- Do not output <thought>.\n"
        "- The <Type> block must follow the current DEFAULT Type style, not the solverllm hint style.\n"
        "- The <python> block must follow the current DEFAULT Code rules exactly.\n"
    )


class OpenAICompatClient:
    def __init__(self, *, model: str, api_key: str, base_url: str, timeout: int = 120):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        if self.base_url.endswith("/chat/completions"):
            self.endpoint = self.base_url
        elif self.base_url.endswith("/v1"):
            self.endpoint = self.base_url + "/chat/completions"
        else:
            self.endpoint = self.base_url + "/v1/chat/completions"
        self.timeout = timeout

    def chat(self, *, system_prompt: str, user_prompt: str, temperature: float = 0.2) -> str:
        payload = {
            "model": self.model,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.endpoint,
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"URL error: {exc}") from exc
        try:
            return body["choices"][0]["message"]["content"]
        except Exception as exc:
            raise RuntimeError(f"unexpected API response: {body}") from exc


def extract_tag_block(text: str, tag: str) -> str:
    pattern = re.compile(rf"<{tag}>\s*(.*?)\s*</{tag}>", flags=re.IGNORECASE | re.DOTALL)
    match = pattern.search(text)
    return match.group(0).strip() if match else ""


def verify_type_block(type_block: str) -> dict[str, Any]:
    type_match = TYPE_BLOCK_RE.search(type_block)
    if not type_match:
        raise ValueError("missing <Type> block during verification")
    body = type_match.group(1).strip()
    forbidden = re.findall(r"</?(Sets|Parameters|Variables|Objective|Constraints|python|thought)>", body, flags=re.IGNORECASE)
    if forbidden:
        raise ValueError(f"<Type> block contains forbidden nested tags: {sorted(set(x.lower() for x in forbidden))}")

    required_patterns = {
        "optimization_type": r"(?im)^\s*-\s*optimization\s+type\s*:",
        "classical_or_family": r"(?im)^\s*-\s*classical\s+or\s+family\s*:",
        "explanation": r"(?im)^\s*-\s*explanation\s*:",
    }
    missing = [name for name, regex in required_patterns.items() if not re.search(regex, body)]
    if missing:
        raise ValueError(f"<Type> block is missing required default-format fields: {missing}")

    return {
        "format": "default_type",
        "required_fields": list(required_patterns),
        "body_preview": body[:500],
    }


def extract_type_and_python(response_text: str) -> tuple[str, str]:
    type_match = TYPE_BLOCK_RE.search(response_text)
    py_match = PYTHON_BLOCK_RE.search(response_text)
    if not type_match:
        raise ValueError("missing <Type> block")
    if not py_match:
        raise ValueError("missing <python> block")
    type_block = f"<Type>\n{type_match.group(1).strip()}\n</Type>"
    python_block = f"<python>\n{py_match.group(1).strip()}\n</python>"
    return type_block, python_block


def extract_python_block_from_record(record: dict[str, Any]) -> str:
    direct = str(record.get("python_block", "") or "").strip()
    if direct:
        return direct
    output_text = str(record.get("output", "") or "")
    py = extract_tag_block(output_text, "python")
    if py:
        return py
    raise VerificationError(
        "missing <python> block in existing record",
        detail={"stage": "extract_existing_python"},
    )


def gurobipy_available() -> bool:
    return importlib.util.find_spec("gurobipy") is not None


def verify_python_block(python_block: str, *, mode: str, timeout: int) -> dict[str, Any]:
    py_match = PYTHON_BLOCK_RE.search(python_block)
    if not py_match:
        raise VerificationError(
            "missing <python> block during verification",
            detail={"stage": "extract_python", "mode": mode},
        )
    code = py_match.group(1).strip()
    try:
        ast.parse(code)
        compile(code, "<generated_gurobi>", "exec")
    except Exception as exc:
        raise VerificationError(
            f"syntax verification failed: {exc}",
            detail={
                "stage": "syntax",
                "mode": mode,
                "error": repr(exc),
                "python_code": code,
            },
        ) from exc
    result: dict[str, Any] = {"syntax_ok": True, "run_ok": None, "mode": mode}
    if mode == "syntax":
        result["run_ok"] = False
        return result
    if mode == "run-if-available" and not gurobipy_available():
        result["mode"] = "syntax-fallback"
        result["run_ok"] = False
        return result
    with tempfile.TemporaryDirectory(prefix="verify_gurobi_") as tmpdir:
        tmp_path = Path(tmpdir) / "candidate.py"
        tmp_path.write_text(code, encoding="utf-8")
        try:
            proc = subprocess.run(
                [sys.executable, str(tmp_path)],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            raise VerificationError(
                f"runtime verification timeout ({timeout}s)",
                detail={
                    "stage": "runtime",
                    "mode": mode,
                    "timeout_sec": int(timeout),
                    "stdout": str(exc.stdout or ""),
                    "stderr": str(exc.stderr or ""),
                    "python_code": code,
                },
            ) from exc
    result.update(
        {
            "run_ok": proc.returncode == 0,
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }
    )
    if proc.returncode != 0:
        raise VerificationError(
            f"runtime verification failed with return code {proc.returncode}",
            detail={
                "stage": "runtime",
                "mode": mode,
                "returncode": int(proc.returncode),
                "stdout": proc.stdout,
                "stderr": proc.stderr,
                "python_code": code,
            },
        )
    return result


def build_retry_feedback(
    *,
    task_text: str,
    canonical_blocks: str,
    previous_type_block: str,
    previous_python_block: str,
    verification_error: Exception,
) -> str:
    detail: dict[str, Any] = {}
    if isinstance(verification_error, VerificationError):
        detail = dict(verification_error.detail or {})

    error_block = {
        "error_class": type(verification_error).__name__,
        "error_message": str(verification_error),
        "detail": detail,
    }
    return (
        "Your previous answer was rejected. Regenerate from scratch and fix all issues.\n\n"
        "=== FULL CONTEXT ===\n"
        f"[Task]\n{task_text.strip()}\n\n"
        f"[Canonical Modeling Blocks]\n{canonical_blocks.strip()}\n\n"
        f"[Previous <Type>]\n{previous_type_block.strip() or '<empty>'}\n\n"
        f"[Previous <python>]\n{previous_python_block.strip() or '<empty>'}\n\n"
        "[Verification Error]\n"
        f"{json.dumps(error_block, ensure_ascii=False, indent=2)}\n\n"
        "Regeneration requirements:\n"
        "- Keep canonical modeling semantics unchanged.\n"
        "- Output exactly two blocks in order: <Type> then <python>.\n"
        "- Do not output <thought>.\n"
    )


def build_generation_record(
    *,
    source_record: dict[str, Any],
    source_index: int,
    record_id: str | None,
    tagged_output: str,
    type_block: str,
    python_block: str,
    verification: dict[str, Any],
    raw_response: str,
) -> dict[str, Any]:
    combined_output = "\n\n".join(
        [
            type_block.strip(),
            tagged_output.strip(),
            python_block.strip(),
        ]
    )
    return {
        "record_id": str(record_id or stable_record_id(source_record, source_index)),
        "source_index": source_index,
        "input": source_record.get("input", ""),
        "output": combined_output,
        "tagged_output": tagged_output,
        "type_verification": verification.get("type_verification", {}),
        "type_block": type_block,
        "python_block": python_block,
        "verification": verification,
        "raw_response_preview": raw_response[:2000],
    }


def _generate_one_record(
    *,
    source_index: int,
    record: dict[str, Any],
    rid: str,
    client: "OpenAICompatClient",
    system_prompt: str,
    temperature: float,
    max_retries: int,
    retry_sleep: float,
    verification_mode: str,
    run_timeout: int,
    retry_until_success: bool,
) -> dict[str, Any]:
    try:
        tagged_output = build_tagged_sections(record)
    except Exception as exc:
        return {
            "ok": False,
            "record_id": rid,
            "source_index": source_index,
            "stage": "formatting",
            "error_payload": {
                "record_id": rid,
                "source_index": source_index,
                "stage": "formatting",
                "error": str(exc),
                "input_preview": str(record.get("input", ""))[:500],
            },
            "attempt_errors": [],
        }

    feedback = ""
    last_exc: Exception | None = None
    attempt_errors: list[str] = []
    attempt = 0
    while True:
        attempt += 1
        if (not retry_until_success) and attempt > max_retries:
            break
        user_prompt = build_generation_prompt(str(record.get("input", "")), tagged_output)
        if feedback:
            user_prompt += "\n\n" + feedback + "\n"
        try:
            response_text = client.chat(system_prompt=system_prompt, user_prompt=user_prompt, temperature=temperature)
            type_block, python_block = extract_type_and_python(response_text)
            type_verification = verify_type_block(type_block)
            verification = verify_python_block(python_block, mode=verification_mode, timeout=run_timeout)
            verification["type_verification"] = type_verification
            payload = build_generation_record(
                source_record=record,
                source_index=source_index,
                record_id=rid,
                tagged_output=tagged_output,
                type_block=type_block,
                python_block=python_block,
                verification=verification,
                raw_response=response_text,
            )
            return {
                "ok": True,
                "record_id": rid,
                "source_index": source_index,
                "attempt": attempt,
                "payload": payload,
                "verification_mode": verification.get("mode"),
                "attempt_errors": attempt_errors,
            }
        except Exception as exc:
            last_exc = exc
            previous_type = locals().get("type_block", "")
            previous_python = locals().get("python_block", "")
            feedback = build_retry_feedback(
                task_text=str(record.get("input", "")),
                canonical_blocks=tagged_output,
                previous_type_block=str(previous_type or ""),
                previous_python_block=str(previous_python or ""),
                verification_error=exc,
            )
            attempt_total = ("inf" if retry_until_success else str(max_retries))
            attempt_errors.append(f"attempt={attempt}/{attempt_total}: {exc}")
            time.sleep(retry_sleep)

    return {
        "ok": False,
        "record_id": rid,
        "source_index": source_index,
        "stage": "generation",
        "error_payload": {
            "record_id": rid,
            "source_index": source_index,
            "stage": "generation",
            "error": str(last_exc) if last_exc else "unknown generation error",
            "input_preview": str(record.get("input", ""))[:500],
            "tagged_output_preview": tagged_output[:1000],
        },
        "attempt_errors": attempt_errors,
    }


def process_format(args: argparse.Namespace) -> None:
    input_path = Path(args.input)
    output_path = Path(args.output)
    seen = load_seen_record_ids(output_path) if args.resume else set()
    processed = 0
    skipped = 0
    for source_index, record in iter_jsonl(input_path):
        rid = stable_record_id(record, source_index)
        if rid in seen:
            skipped += 1
            continue
        if args.limit is not None and processed >= args.limit:
            break
        try:
            payload = build_format_record(record, source_index)
            append_jsonl(output_path, payload)
            seen.add(rid)
            processed += 1
            print(f"[format] wrote source_index={source_index} record_id={rid}", flush=True)
        except Exception as exc:
            error_path = output_path.with_suffix(output_path.suffix + ".errors.jsonl")
            append_jsonl(
                error_path,
                {
                    "record_id": rid,
                    "source_index": source_index,
                    "error": str(exc),
                    "input_preview": str(record.get("input", ""))[:500],
                },
            )
            print(f"[format] failed source_index={source_index} record_id={rid}: {exc}", flush=True)
    print(f"[format] done processed={processed} skipped={skipped} output={output_path}", flush=True)


def process_generate(args: argparse.Namespace) -> None:
    input_path = Path(args.input)
    output_path = Path(args.output)
    error_path = output_path.with_suffix(output_path.suffix + ".errors.jsonl")
    seen = load_seen_record_ids(output_path) if args.resume else set()
    parallel = max(1, int(args.parallel))

    api_key = os.getenv(args.api_key_env, "")
    if not api_key:
        raise SystemExit(f"missing API key env: {args.api_key_env}")
    model = args.model or os.getenv(args.model_env, "")
    if not model:
        raise SystemExit("missing model: pass --model or set the model env")
    base_url = os.getenv(args.base_url_env, args.base_url)
    client = OpenAICompatClient(model=model, api_key=api_key, base_url=base_url, timeout=args.request_timeout)

    processed = 0
    skipped = 0
    source_iter = iter_jsonl(input_path)
    exhausted = False
    system_prompt = SYSTEM_INSTRUCTION.strip() + "\nOnly output the requested final tagged blocks. Do not output <thought>."
    retry_until_success = bool(args.retry_until_success) or int(args.max_retries) <= 0

    def next_candidate() -> tuple[int, dict[str, Any], str] | None:
        nonlocal skipped, exhausted
        if exhausted:
            return None
        for source_index, record in source_iter:
            rid = stable_record_id(record, source_index)
            if rid in seen:
                skipped += 1
                continue
            return source_index, record, rid
        exhausted = True
        return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=parallel) as executor:
        inflight: dict[concurrent.futures.Future, tuple[int, str]] = {}

        while True:
            remaining_needed = None if args.limit is None else max(0, int(args.limit) - processed)
            if remaining_needed == 0:
                break
            max_inflight = parallel if remaining_needed is None else min(parallel, remaining_needed)

            while len(inflight) < max_inflight:
                candidate = next_candidate()
                if candidate is None:
                    break
                source_index, record, rid = candidate
                fut = executor.submit(
                    _generate_one_record,
                    source_index=source_index,
                    record=record,
                    rid=rid,
                    client=client,
                    system_prompt=system_prompt,
                    temperature=args.temperature,
                    max_retries=args.max_retries,
                    retry_sleep=args.retry_sleep,
                    verification_mode=args.verification_mode,
                    run_timeout=args.run_timeout,
                    retry_until_success=retry_until_success,
                )
                inflight[fut] = (source_index, rid)

            if not inflight:
                break

            done, _ = concurrent.futures.wait(inflight.keys(), return_when=concurrent.futures.FIRST_COMPLETED)
            for fut in done:
                source_index, rid = inflight.pop(fut)
                try:
                    result = fut.result()
                except Exception as exc:
                    append_jsonl(
                        error_path,
                        {
                            "record_id": rid,
                            "source_index": source_index,
                            "stage": "worker_crash",
                            "error": str(exc),
                        },
                    )
                    print(f"[generate] worker failed source_index={source_index} record_id={rid}: {exc}", flush=True)
                    continue

                for retry_msg in result.get("attempt_errors", []):
                    print(
                        f"[generate] retry source_index={result.get('source_index', source_index)} "
                        f"record_id={result.get('record_id', rid)} {retry_msg}",
                        flush=True,
                    )

                if bool(result.get("ok")):
                    append_jsonl(output_path, result["payload"])
                    seen.add(rid)
                    processed += 1
                    print(
                        f"[generate] wrote source_index={result['source_index']} record_id={result['record_id']} "
                        f"attempt={result.get('attempt')} verification={result.get('verification_mode')}",
                        flush=True,
                    )
                else:
                    append_jsonl(error_path, result["error_payload"])
                    print(
                        f"[generate] {result.get('stage', 'generation')} failed source_index={result['source_index']} "
                        f"record_id={result['record_id']}: {result['error_payload'].get('error')}",
                        flush=True,
                    )

    print(f"[generate] done processed={processed} skipped={skipped} output={output_path}", flush=True)


def process_repair(args: argparse.Namespace) -> None:
    input_path = Path(args.input)
    output_path = Path(args.output)
    error_path = output_path.with_suffix(output_path.suffix + ".errors.jsonl")
    seen = load_seen_record_ids(output_path) if args.resume else set()

    system_prompt = SYSTEM_INSTRUCTION.strip() + "\nOnly output the requested final tagged blocks. Do not output <thought>."
    retry_until_success = bool(args.retry_until_success) or int(args.max_retries) <= 0
    processed = 0
    skipped = 0
    repaired = 0
    verified_ok = 0
    client: OpenAICompatClient | None = None

    def _get_client() -> OpenAICompatClient:
        nonlocal client
        if client is not None:
            return client
        api_key = os.getenv(args.api_key_env, "")
        if not api_key:
            raise SystemExit(f"missing API key env: {args.api_key_env}")
        model = args.model or os.getenv(args.model_env, "")
        if not model:
            raise SystemExit("missing model: pass --model or set the model env")
        base_url = os.getenv(args.base_url_env, args.base_url)
        client = OpenAICompatClient(model=model, api_key=api_key, base_url=base_url, timeout=args.request_timeout)
        return client

    for source_index, record in iter_jsonl(input_path):
        rid = str(record.get("record_id", "") or stable_record_id(record, source_index))
        if rid in seen:
            skipped += 1
            continue
        if args.limit is not None and processed >= args.limit:
            break

        try:
            existing_python = extract_python_block_from_record(record)
            verification = verify_python_block(existing_python, mode=args.verification_mode, timeout=args.run_timeout)
            payload = dict(record)
            payload["record_id"] = rid
            payload["python_block"] = existing_python
            payload["verification_recheck"] = verification
            append_jsonl(output_path, payload)
            verified_ok += 1
            processed += 1
            seen.add(rid)
            print(f"[repair] verified source_index={source_index} record_id={rid}", flush=True)
            continue
        except Exception as verify_exc:
            print(f"[repair] need_regen source_index={source_index} record_id={rid}: {verify_exc}", flush=True)

        result = _generate_one_record(
            source_index=source_index,
            record=record,
            rid=rid,
            client=_get_client(),
            system_prompt=system_prompt,
            temperature=args.temperature,
            max_retries=args.max_retries,
            retry_sleep=args.retry_sleep,
            verification_mode=args.verification_mode,
            run_timeout=args.run_timeout,
            retry_until_success=retry_until_success,
        )
        for retry_msg in result.get("attempt_errors", []):
            print(
                f"[repair] retry source_index={result.get('source_index', source_index)} "
                f"record_id={result.get('record_id', rid)} {retry_msg}",
                flush=True,
            )

        if bool(result.get("ok")):
            append_jsonl(output_path, result["payload"])
            seen.add(rid)
            processed += 1
            repaired += 1
            print(
                f"[repair] repaired source_index={result['source_index']} record_id={result['record_id']} "
                f"attempt={result.get('attempt')} verification={result.get('verification_mode')}",
                flush=True,
            )
        else:
            append_jsonl(error_path, result["error_payload"])
            print(
                f"[repair] failed source_index={result['source_index']} record_id={result['record_id']}: "
                f"{result['error_payload'].get('error')}",
                flush=True,
            )

    print(
        f"[repair] done processed={processed} skipped={skipped} verified_ok={verified_ok} repaired={repaired} output={output_path}",
        flush=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare TTRL-OR training data with tag formatting and Type/Python generation.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    format_parser = subparsers.add_parser("format", help="Convert legacy markdown sections into tagged blocks.")
    format_parser.add_argument("--input", default="data/train/train_data.jsonl")
    format_parser.add_argument("--output", default="data/train/train_data.tagged.jsonl")
    format_parser.add_argument("--limit", type=int, default=None)
    format_parser.add_argument("--resume", dest="resume", action="store_true", default=True)
    format_parser.add_argument("--no-resume", dest="resume", action="store_false")
    format_parser.set_defaults(func=process_format)
    

    generate_parser = subparsers.add_parser("generate", help="Call an LLM to generate <Type> and <python> blocks.")
    generate_parser.add_argument("--input", default="data/train/train_data.jsonl")
    generate_parser.add_argument("--output", default="data/train/train_data.type_python.jsonl")
    generate_parser.add_argument("--model", default="")
    generate_parser.add_argument("--base-url", default=os.getenv("IDEALAB_BASE_URL", ""))
    generate_parser.add_argument("--api-key-env", default="IDEALAB_API_KEY")
    generate_parser.add_argument("--base-url-env", default="IDEALAB_BASE_URL")
    generate_parser.add_argument("--model-env", default="OPENAI_MODEL")
    generate_parser.add_argument("--temperature", type=float, default=0.4)
    generate_parser.add_argument("--request-timeout", type=int, default=120)
    generate_parser.add_argument("--run-timeout", type=int, default=30)
    generate_parser.add_argument("--verification-mode", choices=["syntax", "run", "run-if-available"], default="run-if-available")
    generate_parser.add_argument("--max-retries", type=int, default=4)
    generate_parser.add_argument(
        "--retry-until-success",
        action="store_true",
        default=False,
        help="Keep retrying a sample until verification succeeds. If set, --max-retries is ignored.",
    )
    generate_parser.add_argument("--retry-sleep", type=float, default=1.5)
    generate_parser.add_argument("--parallel", type=int, default=4, help="Number of concurrent generation workers.")
    generate_parser.add_argument("--limit", type=int, default=None)
    generate_parser.add_argument("--resume", dest="resume", action="store_true", default=True)
    generate_parser.add_argument("--no-resume", dest="resume", action="store_false")
    generate_parser.set_defaults(func=process_generate)

    repair_parser = subparsers.add_parser(
        "repair",
        help="Re-verify existing generated <python> blocks, keep valid records, and regenerate failed ones.",
    )
    repair_parser.add_argument("--input", default="data/train/train_data.type_python.jsonl")
    repair_parser.add_argument("--output", default="data/train/train_data.type_python.repaired.jsonl")
    repair_parser.add_argument("--model", default="")
    repair_parser.add_argument("--base-url", default=os.getenv("IDEALAB_BASE_URL", ""))
    repair_parser.add_argument("--api-key-env", default="IDEALAB_API_KEY")
    repair_parser.add_argument("--base-url-env", default="IDEALAB_BASE_URL")
    repair_parser.add_argument("--model-env", default="OPENAI_MODEL")
    repair_parser.add_argument("--temperature", type=float, default=0.2)
    repair_parser.add_argument("--request-timeout", type=int, default=120)
    repair_parser.add_argument("--run-timeout", type=int, default=30)
    repair_parser.add_argument("--verification-mode", choices=["syntax", "run", "run-if-available"], default="run")
    repair_parser.add_argument("--max-retries", type=int, default=8)
    repair_parser.add_argument(
        "--retry-until-success",
        action="store_true",
        default=False,
        help="Keep retrying a sample until verification succeeds.",
    )
    repair_parser.add_argument("--retry-sleep", type=float, default=1.5)
    repair_parser.add_argument("--limit", type=int, default=None)
    repair_parser.add_argument("--resume", dest="resume", action="store_true", default=True)
    repair_parser.add_argument("--no-resume", dest="resume", action="store_false")
    repair_parser.set_defaults(func=process_repair)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
