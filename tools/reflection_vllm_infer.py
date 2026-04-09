from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


NUMBER_RE = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


def iter_jsonl(path: Path):
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


def sanitize_component(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(text or "").strip())
    cleaned = cleaned.strip("._-")
    return cleaned or "unknown"


def is_close(a: float, b: float, *, rel_tol: float, abs_tol: float) -> bool:
    return abs(float(a) - float(b)) <= float(abs_tol) + float(rel_tol) * max(abs(float(a)), abs(float(b)), 1.0)


def parse_numeric(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        num = float(value)
        return num if math.isfinite(num) else None
    if value is None:
        return None
    text = str(value).strip().replace(",", "")
    if not text:
        return None
    try:
        num = float(text)
        return num if math.isfinite(num) else None
    except Exception:
        pass
    matches = NUMBER_RE.findall(text)
    if not matches:
        return None
    try:
        num = float(matches[-1])
        return num if math.isfinite(num) else None
    except Exception:
        return None


def extract_objective(text: str) -> float | None:
    raw = str(text or "")
    if not raw.strip():
        return None

    patterns = [
        r"(?i)\bobjective(?:\s*value)?\s*[:=]\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)",
        r"(?i)\boptimal(?:\s*value)?\s*[:=]\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)",
        r"(?i)\bobj(?:ective)?\s*[:=]\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)",
        r"(?i)\bfinal\s*answer\s*[:=]\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)",
        r"(?i)\banswer\s*[:=]\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)",
        r"(?i)\bobjectivevalue\s*[:=]?\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)",
    ]

    for pat in patterns:
        m = re.search(pat, raw)
        if m:
            try:
                v = float(m.group(1))
                if math.isfinite(v):
                    return v
            except Exception:
                pass

    stripped = raw.strip()
    if re.fullmatch(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", stripped):
        try:
            v = float(stripped)
            if math.isfinite(v):
                return v
        except Exception:
            pass

    tail = raw[-800:]
    nums = NUMBER_RE.findall(tail)
    if nums:
        try:
            v = float(nums[-1])
            if math.isfinite(v):
                return v
        except Exception:
            pass

    return None


def pick_first_nonempty(row: dict[str, Any], keys: list[str]) -> tuple[str, Any]:
    for key in keys:
        value = row.get(key)
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return key, value
    return "", None


class OpenAICompatClient:
    def __init__(self, *, model: str, api_key: str, base_url: str, timeout: int = 120):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = int(timeout)

        if self.base_url.endswith("/chat/completions"):
            self.endpoint = self.base_url
        elif self.base_url.endswith("/v1"):
            self.endpoint = self.base_url + "/chat/completions"
        else:
            self.endpoint = self.base_url + "/v1/chat/completions"

    def chat(self, *, messages: list[dict[str, str]], n: int, temperature: float, top_p: float, max_tokens: int) -> list[str]:
        payload = {
            "model": self.model,
            "temperature": float(temperature),
            "top_p": float(top_p),
            "max_tokens": int(max_tokens),
            "n": int(max(1, n)),
            "messages": messages,
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

        choices = body.get("choices", []) if isinstance(body, dict) else []
        outputs: list[str] = []
        for item in choices:
            content = None
            if isinstance(item, dict):
                msg = item.get("message")
                if isinstance(msg, dict):
                    content = msg.get("content")
                if content is None and isinstance(item.get("text"), str):
                    content = item.get("text")
            outputs.append(str(content or ""))

        if not outputs:
            raise RuntimeError(f"unexpected API response: {body}")
        return outputs


def build_initial_messages(question: str) -> list[dict[str, str]]:
    system_prompt = (
        "You are an optimization expert. Solve the optimization problem carefully. "
        "Return only one line in the exact format: ObjectiveValue: <number>. "
        "Do not output any extra text."
    )
    user_prompt = (
        "Problem:\n"
        f"{question.strip()}\n\n"
        "Output format requirement (strict):\n"
        "ObjectiveValue: <number>"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def build_reflection_prompt(round_index: int) -> str:
    return (
        f"Reflection round {round_index}. Re-check your previous answer for numerical or modeling mistakes. "
        "If needed, correct it. Output only one line exactly: ObjectiveValue: <number>."
    )


def choose_final_obj(round_payloads: list[dict[str, Any]]) -> tuple[float | None, int | None, str]:
    for item in reversed(round_payloads):
        obj = item.get("obj")
        if isinstance(obj, (int, float)) and math.isfinite(float(obj)):
            return float(obj), int(item.get("round", -1)), "last_valid_round"
    return None, None, "no_valid_obj"


def process_one(
    *,
    source_index: int,
    row: dict[str, Any],
    client: OpenAICompatClient,
    args: argparse.Namespace,
    question_keys: list[str],
    answer_keys: list[str],
) -> dict[str, Any]:
    q_key, q_val = pick_first_nonempty(row, question_keys)
    a_key, a_val = pick_first_nonempty(row, answer_keys)

    question = str(q_val or "").strip()
    if not question:
        return {
            "source_index": int(source_index),
            "sample_id": row.get(args.id_key, source_index),
            "status": "failed",
            "error": "missing question text",
            "question_key": q_key,
            "answer_key": a_key,
        }

    gt_numeric = parse_numeric(a_val)
    rounds = max(0, int(args.reflection_rounds))

    messages = build_initial_messages(question)
    round_outputs: list[dict[str, Any]] = []

    t0 = time.perf_counter()
    first = client.chat(
        messages=messages,
        n=1,
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        max_tokens=int(args.max_tokens),
    )[0]
    first_obj = extract_objective(first)
    round_outputs.append(
        {
            "round": 0,
            "phase": "initial",
            "obj": (float(first_obj) if isinstance(first_obj, (int, float)) and math.isfinite(float(first_obj)) else None),
            "text": (first if bool(args.save_raw_text) else first[:400]),
        }
    )
    messages.append({"role": "assistant", "content": first})

    for ridx in range(1, rounds + 1):
        messages.append({"role": "user", "content": build_reflection_prompt(round_index=ridx)})
        reflected = client.chat(
            messages=messages,
            n=1,
            temperature=float(args.temperature),
            top_p=float(args.top_p),
            max_tokens=int(args.max_tokens),
        )[0]
        reflected_obj = extract_objective(reflected)
        round_outputs.append(
            {
                "round": int(ridx),
                "phase": "reflection",
                "obj": (float(reflected_obj) if isinstance(reflected_obj, (int, float)) and math.isfinite(float(reflected_obj)) else None),
                "text": (reflected if bool(args.save_raw_text) else reflected[:400]),
            }
        )
        messages.append({"role": "assistant", "content": reflected})

    elapsed = float(time.perf_counter() - t0)

    final_obj, final_round, final_policy = choose_final_obj(round_outputs)

    hit = None
    abs_err = None
    rel_err = None
    if final_obj is not None and gt_numeric is not None:
        abs_err = abs(float(final_obj) - float(gt_numeric))
        rel_err = abs_err / max(abs(float(gt_numeric)), 1.0)
        hit = bool(is_close(float(final_obj), float(gt_numeric), rel_tol=float(args.gt_rel_tol), abs_tol=float(args.gt_abs_tol)))

    return {
        "source_index": int(source_index),
        "sample_id": row.get(args.id_key, source_index),
        "status": "ok",
        "question_key": q_key,
        "answer_key": a_key,
        "question_preview": question[:300],
        "gt_raw": a_val,
        "gt_obj": gt_numeric,
        "final_obj": final_obj,
        "final_round": final_round,
        "final_policy": final_policy,
        "rounds_requested": int(rounds),
        "rounds_executed": int(len(round_outputs) - 1),
        "hit": hit,
        "abs_error": abs_err,
        "rel_error": rel_err,
        "latency_sec": elapsed,
        "round_outputs": round_outputs,
    }


def load_seen_indices(path: Path) -> set[int]:
    if not path.exists():
        return set()
    seen: set[int] = set()
    for _, row in iter_jsonl(path):
        idx = row.get("source_index")
        if isinstance(idx, int):
            seen.add(idx)
    return seen


def build_summary(log_path: Path) -> dict[str, Any]:
    total = 0
    ok = 0
    failed = 0
    with_pred = 0
    with_gt = 0
    comparable = 0
    hits = 0
    abs_errors: list[float] = []

    for _, row in iter_jsonl(log_path):
        total += 1
        status = str(row.get("status", ""))
        if status == "ok":
            ok += 1
        else:
            failed += 1

        pred = row.get("final_obj")
        gt = row.get("gt_obj")
        if isinstance(pred, (int, float)) and math.isfinite(float(pred)):
            with_pred += 1
        if isinstance(gt, (int, float)) and math.isfinite(float(gt)):
            with_gt += 1
        if isinstance(pred, (int, float)) and isinstance(gt, (int, float)):
            if math.isfinite(float(pred)) and math.isfinite(float(gt)):
                comparable += 1
                ae = abs(float(pred) - float(gt))
                abs_errors.append(ae)
                if bool(row.get("hit", False)):
                    hits += 1

    mae = (sum(abs_errors) / float(len(abs_errors))) if abs_errors else None
    return {
        "total": int(total),
        "ok": int(ok),
        "failed": int(failed),
        "with_pred": int(with_pred),
        "with_gt": int(with_gt),
        "comparable": int(comparable),
        "hits": int(hits),
        "hit_rate": (float(hits) / float(comparable) if comparable > 0 else None),
        "mae": mae,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Reflection inference via local vLLM OpenAI-compatible API.")
    parser.add_argument("--input", required=True, help="Input dataset jsonl path under data/.")
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", ""), help="Model name served by vLLM.")
    parser.add_argument("--base-url", default=os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8000/v1"))
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY", "EMPTY"))
    parser.add_argument("--reflection-rounds", type=int, default=2, help="Number of reflection rounds after initial answer.")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--request-timeout", type=int, default=120)
    parser.add_argument("--parallel", type=int, default=4)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0, help="0 means no limit.")
    parser.add_argument("--id-key", default="id")
    parser.add_argument("--question-keys", default="input,en_question,question,prompt,task")
    parser.add_argument("--answer-keys", default="answer,en_answer,gt,ground_truth,output")
    parser.add_argument("--gt-rel-tol", type=float, default=1e-4)
    parser.add_argument("--gt-abs-tol", type=float, default=1e-6)
    parser.add_argument("--save-raw-text", action="store_true", help="Save full round text in logs.")
    parser.add_argument("--log-dir", default="outputs/reflection_logs")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"input not found: {input_path}")
    if not args.model:
        raise SystemExit("missing model: pass --model or set OPENAI_MODEL")

    dataset_tag = sanitize_component(input_path.stem)
    model_tag = sanitize_component(args.model)
    out_dir = Path(args.log_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log_path = out_dir / f"reflection_{dataset_tag}__{model_tag}.jsonl"
    summary_path = out_dir / f"reflection_{dataset_tag}__{model_tag}.summary.json"

    seen = load_seen_indices(log_path) if args.resume else set()

    question_keys = [k.strip() for k in str(args.question_keys).split(",") if k.strip()]
    answer_keys = [k.strip() for k in str(args.answer_keys).split(",") if k.strip()]

    tasks: list[tuple[int, dict[str, Any]]] = []
    for idx, row in iter_jsonl(input_path):
        if int(idx) < int(args.start_index):
            continue
        if args.resume and idx in seen:
            continue
        tasks.append((idx, row))
        if int(args.limit) > 0 and len(tasks) >= int(args.limit):
            break

    client = OpenAICompatClient(
        model=str(args.model),
        api_key=str(args.api_key),
        base_url=str(args.base_url),
        timeout=int(args.request_timeout),
    )

    print(
        f"[reflection] input={input_path} model={args.model} rounds={args.reflection_rounds} "
        f"parallel={args.parallel} tasks={len(tasks)} log={log_path}",
        flush=True,
    )

    processed = 0
    with ThreadPoolExecutor(max_workers=max(1, int(args.parallel))) as pool:
        fut_map = {
            pool.submit(
                process_one,
                source_index=idx,
                row=row,
                client=client,
                args=args,
                question_keys=question_keys,
                answer_keys=answer_keys,
            ): idx
            for idx, row in tasks
        }
        for fut in as_completed(fut_map):
            idx = fut_map[fut]
            try:
                payload = fut.result()
            except Exception as exc:  # noqa: BLE001
                payload = {
                    "source_index": int(idx),
                    "sample_id": idx,
                    "status": "failed",
                    "error": f"worker exception: {type(exc).__name__}: {exc}",
                }
            append_jsonl(log_path, payload)
            processed += 1
            if processed % 10 == 0 or processed == len(tasks):
                print(f"[reflection] processed={processed}/{len(tasks)}", flush=True)

    summary = {
        "dataset": str(input_path),
        "dataset_tag": dataset_tag,
        "model": str(args.model),
        "model_tag": model_tag,
        "reflection_rounds": int(args.reflection_rounds),
        "temperature": float(args.temperature),
        "top_p": float(args.top_p),
        "max_tokens": int(args.max_tokens),
        "parallel": int(args.parallel),
        "log_path": str(log_path.resolve()),
        "timestamp": int(time.time()),
    }
    summary.update(build_summary(log_path))
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[reflection] done log={log_path} summary={summary_path}", flush=True)
    print(
        f"[reflection] comparable={summary.get('comparable')} hit_rate={summary.get('hit_rate')} mae={summary.get('mae')}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
