from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
import urllib.error
import urllib.request
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

SYSTEM_PROMPT = """
You are a helpful Assistant with expertise in operations research and the Gurobi solver. 
When the User provides an OR question, you will analyze it, build a detailed mathematical model, and provide the Gurobi code to solve it.
"""

def code_prompt(question):
    prompt = f"""Answer the following mathematical modeling question:
{question}
You should think first within a reasoning process in <thinking> and then directly output the final python code using Gurobi to solve the mathematical modeling question within <python>.
You output should be in the following format:
<thinking>
[Your reasoning here]
</thinking>
<python>
[Your Python code here]
</python>

code example:
<python>
import gurobipy as gp
from gurobipy import GRB

# Create model
model = gp.Model()
......(here is core modeling code)

model.optimize()

status = model.status
if status == GRB.OPTIMAL:
    optimal = model.objVal
    print(f"Optimal value: {{optimal}}")
else:
    print(f"Model status: {{status}}")
</python>
"""
    return prompt

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
    try:
        return abs(float(a) - float(b)) <= float(abs_tol) + float(rel_tol) * max(abs(float(a)), abs(float(b)), 1.0)
    except:
        return False

def parse_numeric(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        num = float(value)
        return num if math.isfinite(num) else None
    if value is None:
        return None
    text = str(value).strip().replace(",", "")
    try:
        return float(text)
    except:
        matches = NUMBER_RE.findall(text)
        if matches:
            return float(matches[-1])
    return None

def extract_python_code(text: str) -> str | None:
    match = re.search(r"<python>(.*?)</python>", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None
    
def execute_and_extract_optimal(
    code: str,
    timeout: int = 30,
    python_executable: str | None = None,
) -> float | None:
    # 使用 delete=False 确保在多线程环境下文件写入和读取的稳定性
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tf:
        tf.write(code)
        temp_file_path = tf.name

    try:
        result = subprocess.run(
            [python_executable or sys.executable, temp_file_path],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        output = result.stdout
        
        num_pattern = r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"
        opt_match = re.search(rf"Optimal value:?\s*({num_pattern})", output, re.IGNORECASE)
        if opt_match:
            return float(opt_match.group(1))

        best_obj_match = re.search(rf"Best objective\s+:?\s*({num_pattern})", output, re.IGNORECASE)
        if best_obj_match:
            return float(best_obj_match.group(1))

    except Exception:
        pass
    finally:
        if os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except:
                pass
    return None

def cluster_vote(
    values: list[float],
    *,
    rel_tol: float,
    abs_tol: float,
) -> tuple[float | None, int, list[dict[str, Any]]]:
    if not values:
        return None, 0, []

    clusters: list[dict[str, Any]] = []
    for idx, val in enumerate(values):
        matched = None
        for c in clusters:
            if is_close(val, c["center"], rel_tol=rel_tol, abs_tol=abs_tol):
                matched = c
                break
        if matched is None:
            matched = {"center": float(val), "members": []}
            clusters.append(matched)
        matched["members"].append((idx, float(val)))
        matched["center"] = sum(v for _, v in matched["members"]) / float(len(matched["members"]))

    clusters.sort(key=lambda c: (len(c["members"]), -c["members"][0][0]), reverse=True)
    winner = clusters[0]
    vals = sorted(v for _, v in winner["members"])
    mid = len(vals) // 2
    voted = float(vals[mid]) if len(vals) % 2 == 1 else float((vals[mid - 1] + vals[mid]) / 2.0)

    debug_clusters = [
        {"center": c["center"], "size": len(c["members"]), "members": [v for _, v in c["members"]]}
        for c in clusters
    ]
    return voted, len(winner["members"]), debug_clusters

def pick_first_nonempty(row: dict[str, Any], keys: list[str]) -> tuple[str, Any]:
    for key in keys:
        value = row.get(key)
        if value is not None and (not isinstance(value, str) or value.strip()):
            return key, value
    return "", None

# --- 新增：代码执行的工作单元 ---
def execution_worker(
    idx: int,
    text: str,
    timeout: int,
    save_raw: bool,
    code_python: str | None = None,
) -> dict[str, Any]:
    code = extract_python_code(text)
    obj = None
    exec_success = False
    execution_wall_clock_sec = 0.0
    
    if code:
        exec_start = time.perf_counter()
        obj = execute_and_extract_optimal(
            code,
            timeout=timeout,
            python_executable=code_python,
        )
        execution_wall_clock_sec = time.perf_counter() - exec_start
        if obj is not None:
            exec_success = True
            
    return {
        "index": idx,
        "has_code": bool(code),
        "exec_success": exec_success,
        "obj": obj,
        "execution_wall_clock_sec": execution_wall_clock_sec,
        "text_preview": text if save_raw else text[:200]
    }

class OpenAICompatClient:
    def __init__(self, *, model: str, api_key: str, base_url: str, timeout: int = 600):
        self.model = model
        self.api_key = api_key
        base = base_url.strip().rstrip("/")
        if not base.endswith("/v1"):
            base += "/v1"
        self.endpoint = f"{base}/chat/completions"
        self.timeout = int(timeout)

    def chat_n(self, max_retries: int = 3, **kwargs) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "n": int(kwargs.get("n", 1)),
            "temperature": float(kwargs.get("temperature", 0.7)),
            "top_p": float(kwargs.get("top_p", 0.95)),
            "max_tokens": int(kwargs.get("max_tokens", 512)),
            "messages": [
                {"role": "system", "content": kwargs.get("system_prompt", "")},
                {"role": "user", "content": kwargs.get("user_prompt", "")},
            ],
        }
        
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.endpoint,
            data=data,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
            method="POST",
        )

        for attempt in range(max_retries):
            try:
                start_time = time.perf_counter()
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                    api_duration = time.perf_counter() - start_time
                    choices = body.get("choices", [])
                    usage = body.get("usage", {})
                    prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
                    completion_tokens = int(usage.get("completion_tokens", 0) or 0)
                    total_tokens = usage.get("total_tokens")
                    return {
                        "contents": [c["message"]["content"] for c in choices],
                        "api_duration": api_duration,
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": int(total_tokens) if total_tokens is not None else prompt_tokens + completion_tokens,
                    }
            except Exception as e:
                if attempt == max_retries - 1: raise e
                time.sleep(2 ** attempt)
        return {
            "contents": [],
            "api_duration": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

def process_one(
    *,
    source_index: int,
    row: dict[str, Any],
    client: OpenAICompatClient,
    args: argparse.Namespace,
    question_keys: list[str],
    answer_keys: list[str],
) -> dict[str, Any]:
    task_start_time = time.perf_counter()
    
    _, q_val = pick_first_nonempty(row, question_keys)
    _, a_val = pick_first_nonempty(row, answer_keys)
    question = str(q_val or "").strip()
    gt_numeric = parse_numeric(a_val)
    
    system_prompt, user_prompt = build_prompt(question)

    # 1. 获取 LLM 生成 (N 个结果)
    inference_start_time = time.perf_counter()
    try:
        api_res = client.chat_n(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            n=args.n,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
        )
        inference_wall_clock_sec = time.perf_counter() - inference_start_time
        outputs = api_res["contents"]
        api_duration = api_res["api_duration"]
        prompt_tokens = api_res["prompt_tokens"]
        completion_tokens = api_res["completion_tokens"]
        total_tokens = api_res["total_tokens"]
    except Exception as e:
        return {
            "source_index": source_index,
            "status": "failed",
            "error": str(e),
            "metrics": {
                "inference_wall_clock_sec": time.perf_counter() - inference_start_time,
            },
        }

    # 2. 并行执行提取到的 Python 代码 (沙盒执行)
    candidates_detail = []
    candidate_objs = []
    
    # 在单个样本内启动线程池以并行执行 N 个代码
    # 使用 min(len(outputs), 16) 防止线程过多
    code_execution_start_time = time.perf_counter()
    if outputs:
        with ThreadPoolExecutor(max_workers=len(outputs)) as executor:
            futures = [
                executor.submit(
                    execution_worker,
                    i,
                    text,
                    args.exec_timeout,
                    args.save_raw_text,
                    args.code_python,
                )
                for i, text in enumerate(outputs)
            ]
            for f in as_completed(futures):
                res = f.result()
                candidates_detail.append(res)
                if res["exec_success"] and res["obj"] is not None:
                    candidate_objs.append(res["obj"])
    code_execution_wall_clock_sec = time.perf_counter() - code_execution_start_time
    if not any(item["has_code"] for item in candidates_detail):
        code_execution_wall_clock_sec = 0.0
    code_execution_elapsed_sum_sec = sum(
        item["execution_wall_clock_sec"] for item in candidates_detail
    )

    # 按原始索引排序详情
    candidates_detail.sort(key=lambda x: x["index"])

    # 3. 多数投票
    voted_obj, vote_count, vote_clusters = cluster_vote(
        candidate_objs,
        rel_tol=args.vote_rel_tol,
        abs_tol=args.vote_abs_tol,
    )

    # 4. 验证结果
    hit = None
    if voted_obj is not None and gt_numeric is not None:
        hit = is_close(voted_obj, gt_numeric, rel_tol=args.gt_rel_tol, abs_tol=args.gt_abs_tol)

    task_wall_clock_sec = time.perf_counter() - task_start_time
    estimated_inference_flops = 2.0 * args.model_params_billions * 1e9 * total_tokens
    
    return {
        "source_index": source_index,
        "sample_id": row.get(args.id_key, source_index),
        "status": "ok",
        "gt_obj": gt_numeric,
        "voted_obj": voted_obj,
        "vote_count": vote_count,
        "n_received": len(outputs),
        "n_executed_ok": len(candidate_objs),
        "hit": hit,
        "candidates": candidates_detail,
        "vote_clusters": vote_clusters,
        "metrics": {
            "task_wall_clock_sec": task_wall_clock_sec,
            "inference_wall_clock_sec": inference_wall_clock_sec,
            "code_execution_wall_clock_sec": code_execution_wall_clock_sec,
            "code_execution_elapsed_sum_sec": code_execution_elapsed_sum_sec,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "estimated_inference_flops": estimated_inference_flops,
            "flops_estimation": "2 * model_parameters * total_tokens",
            "model_params_billions": args.model_params_billions,
            "completion_throughput_tokens_per_sec": (
                completion_tokens / inference_wall_clock_sec if inference_wall_clock_sec > 0 else 0
            ),
            "total_throughput_tokens_per_sec": (
                total_tokens / inference_wall_clock_sec if inference_wall_clock_sec > 0 else 0
            ),
            "task_duration": task_wall_clock_sec,
            "api_call_duration": api_duration,
            "total_completion_tokens": completion_tokens,
            "avg_tokens_per_gen": completion_tokens / len(outputs) if outputs else 0
        }
    }

def build_prompt(question: str) -> tuple[str, str]:
    return SYSTEM_PROMPT, code_prompt(question)

def load_seen_indices(path: Path) -> set[int]:
    if not path.exists(): return set()
    seen = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try: seen.add(json.loads(line)["source_index"])
            except: continue
    return seen

def build_summary(log_path: Path) -> dict[str, Any]:
    total, comparable, hits = 0, 0, 0
    executed_ok = 0
    sum_task_duration = 0.0
    sum_inference_wall_clock = 0.0
    sum_code_execution_wall_clock = 0.0
    sum_code_execution_elapsed = 0.0
    sum_prompt_tokens = 0
    sum_completion_tokens = 0
    sum_total_tokens = 0
    sum_estimated_inference_flops = 0.0
    total_generations = 0
    
    for _, row in iter_jsonl(log_path):
        if row.get("status") != "ok": continue
        total += 1
        if row.get("n_executed_ok", 0) > 0: executed_ok += 1
        if row.get("voted_obj") is not None and row.get("gt_obj") is not None:
            comparable += 1
            if row.get("hit"): hits += 1
            
        m = row.get("metrics", {})
        sum_task_duration += m.get("task_wall_clock_sec", m.get("task_duration", 0))
        sum_inference_wall_clock += m.get("inference_wall_clock_sec", m.get("api_call_duration", 0))
        sum_code_execution_wall_clock += m.get("code_execution_wall_clock_sec", 0)
        sum_code_execution_elapsed += m.get("code_execution_elapsed_sum_sec", 0)
        sum_prompt_tokens += m.get("prompt_tokens", 0)
        sum_completion_tokens += m.get("completion_tokens", m.get("total_completion_tokens", 0))
        sum_total_tokens += m.get(
            "total_tokens",
            m.get("prompt_tokens", 0) + m.get("total_completion_tokens", 0),
        )
        sum_estimated_inference_flops += m.get("estimated_inference_flops", 0)
        total_generations += row.get("n_received", 0)
            
    return {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_tasks": total,
        "tasks_with_successful_exec": executed_ok,
        "comparable_with_gt": comparable,
        "total_hits": hits,
        "overall_accuracy": (hits / total if total > 0 else 0),
        "avg_task_wall_clock_sec": (sum_task_duration / total if total > 0 else 0),
        "avg_inference_wall_clock_sec": (sum_inference_wall_clock / total if total > 0 else 0),
        "avg_code_execution_wall_clock_sec": (sum_code_execution_wall_clock / total if total > 0 else 0),
        "avg_task_completion_time": (sum_task_duration / total if total > 0 else 0),
        "cumulative_inference_wall_clock_sec": sum_inference_wall_clock,
        "cumulative_code_execution_wall_clock_sec": sum_code_execution_wall_clock,
        "cumulative_code_execution_elapsed_sum_sec": sum_code_execution_elapsed,
        "total_prompt_tokens": sum_prompt_tokens,
        "total_completion_tokens": sum_completion_tokens,
        "total_tokens": sum_total_tokens,
        "avg_prompt_tokens_per_task": (sum_prompt_tokens / total if total > 0 else 0),
        "avg_completion_tokens_per_task": (sum_completion_tokens / total if total > 0 else 0),
        "avg_total_tokens_per_task": (sum_total_tokens / total if total > 0 else 0),
        "estimated_inference_flops": sum_estimated_inference_flops,
        "completion_throughput_tokens_per_sec": (
            sum_completion_tokens / sum_inference_wall_clock if sum_inference_wall_clock > 0 else 0
        ),
        "total_throughput_tokens_per_sec": (
            sum_total_tokens / sum_inference_wall_clock if sum_inference_wall_clock > 0 else 0
        ),
        "avg_completion_tokens_per_gen": (sum_completion_tokens / total_generations if total_generations > 0 else 0)
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--n", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--model-params-billions", type=float, default=4.0)
    parser.add_argument("--exec-timeout", type=int, default=30)
    parser.add_argument("--code-python", default=sys.executable)
    parser.add_argument("--parallel", type=int, default=10) # 同时处理多少个题目
    parser.add_argument("--id-key", default="id")
    parser.add_argument("--question-keys", default="input,question,en_question,prompt")
    parser.add_argument("--answer-keys", default="answer,en_answer,gt,output")
    parser.add_argument("--vote-rel-tol", type=float, default=1e-6)
    parser.add_argument("--vote-abs-tol", type=float, default=1e-6)
    parser.add_argument("--gt-rel-tol", type=float, default=1e-4)
    parser.add_argument("--gt-abs-tol", type=float, default=1e-6)
    parser.add_argument("--log-dir", default="outputs/best_of_n_logs")
    parser.add_argument("--save-raw-text", action="store_true")
    args = parser.parse_args()

    input_path = Path(args.input)
    base_log_name = f"voted_{sanitize_component(input_path.stem)}_{sanitize_component(args.model)}"
    log_path = Path(args.log_dir) / f"{base_log_name}.jsonl"
    summary_path = Path(args.log_dir) / f"{base_log_name}.summary.json"
    
    seen = load_seen_indices(log_path)
    question_keys = args.question_keys.split(",")
    answer_keys = args.answer_keys.split(",")

    tasks = [(idx, row) for idx, row in iter_jsonl(input_path) if idx not in seen]
    client = OpenAICompatClient(model=args.model, api_key=args.api_key, base_url=args.base_url)

    print(f"Starting: {len(tasks)} new tasks, logging to {log_path}")

    try:
        # 外部 ThreadPoolExecutor 用于并行处理不同的数据行
        with ThreadPoolExecutor(max_workers=args.parallel) as pool:
            future_to_idx = {
                pool.submit(process_one, source_index=idx, row=row, client=client, args=args, 
                            question_keys=question_keys, answer_keys=answer_keys): idx 
                for idx, row in tasks
            }
            
            completed_count = 0
            for future in as_completed(future_to_idx):
                res = future.result()
                append_jsonl(log_path, res)
                completed_count += 1
                
                if completed_count % 5 == 0 or completed_count == len(tasks):
                    current_stats = build_summary(log_path)
                    print(f"Progress: {completed_count}/{len(tasks)} | "
                          f"Acc: {current_stats['overall_accuracy']:.2%} | "
                          f"AvgTaskWall: {current_stats['avg_task_wall_clock_sec']:.2f}s | "
                          f"AvgInfer: {current_stats['avg_inference_wall_clock_sec']:.2f}s | "
                          f"AvgCode: {current_stats['avg_code_execution_wall_clock_sec']:.2f}s | "
                          f"Tokens(P/C/T): {current_stats['total_prompt_tokens']:,}/"
                          f"{current_stats['total_completion_tokens']:,}/"
                          f"{current_stats['total_tokens']:,} | "
                          f"AvgTok/Task: {current_stats['avg_total_tokens_per_task']:.1f} | "
                          f"GenTok/s: {current_stats['completion_throughput_tokens_per_sec']:.1f}")

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        if log_path.exists():
            summary = build_summary(log_path)
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=4, ensure_ascii=False)
            print(f"\nFinal Accuracy: {summary['overall_accuracy']:.4f}")

if __name__ == "__main__":
    main()
