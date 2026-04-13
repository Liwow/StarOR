from __future__ import annotations

import argparse
import json
import math
import os
import re
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
You should directly output the final python code using Gurobi to solve the mathematical modeling question within <python>.
For example:
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
    return abs(float(a) - float(b)) <= float(abs_tol) + float(rel_tol) * max(abs(float(a)), abs(float(b)), 1.0)

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
    """提取 <python> 标签中的代码"""
    match = re.search(r"<python>(.*?)</python>", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None
    
def execute_and_extract_optimal(code: str, timeout: int = 30) -> float | None:
    """运行 Python 代码并提取 Optimal value 或 Best objective"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tf:
        tf.write(code)
        temp_file_path = tf.name

    try:
        # 运行子进程
        result = subprocess.run(
            ["python", temp_file_path],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        output = result.stdout
        stderr = result.stderr # 有时错误信息也很重要
        
        # 定义数字部分的正则（支持整数、浮点数、科学计数法）
        num_pattern = r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"

        # 策略 1: 寻找用户代码中 print(f"Optimal value: {optimal}") 的输出
        # 兼容可能有空格或冒号的情况
        opt_match = re.search(rf"Optimal value:?\s*({num_pattern})", output, re.IGNORECASE)
        if opt_match:
            try:
                return float(opt_match.group(1))
            except ValueError:
                pass

        # 策略 2: 寻找 Gurobi 标准输出日志中的 "Best objective 3.040000000000e+04"
        # Gurobi 输出通常形如: Best objective 3.040000000000e+04, best bound ...
        best_obj_match = re.search(rf"Best objective\s+:?\s*({num_pattern})", output, re.IGNORECASE)
        if best_obj_match:
            try:
                return float(best_obj_match.group(1))
            except ValueError:
                pass

        # 如果 stdout 没找到，尝试在 stderr 中找（有时 logging 会输出到 stderr）
        best_obj_match_err = re.search(rf"Best objective\s+:?\s*({num_pattern})", stderr, re.IGNORECASE)
        if best_obj_match_err:
            try:
                return float(best_obj_match_err.group(1))
            except ValueError:
                pass

    except subprocess.TimeoutExpired:
        print(f"Execution timed out after {timeout}s")
    except Exception as e:
        print(f"Execution error: {e}")
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
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

import time
import socket  # 导入用于捕获底层超时

class OpenAICompatClient:
    def __init__(self, *, model: str, api_key: str, base_url: str, timeout: int = 600):
        self.model = model
        self.api_key = api_key
        base = base_url.strip().rstrip("/")
        if not base.endswith("/v1"):
            base += "/v1"
        self.endpoint = f"{base}/chat/completions"
        self.timeout = int(timeout)

    def chat_n(self, max_retries: int = 3, **kwargs) -> list[str]:
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
        proxy_handler = urllib.request.ProxyHandler({}) 
        opener = urllib.request.build_opener(proxy_handler)
        
        req = urllib.request.Request(
            self.endpoint,
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )

        last_error = None
        for attempt in range(max_retries):
            try:
                # 每次重试稍微增加一点超时时间
                current_timeout = self.timeout + (attempt * 60)
                with opener.open(req, timeout=current_timeout) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                    choices = body.get("choices", [])
                    if not choices:
                        raise RuntimeError("API returned empty choices")
                    return [c["message"]["content"] for c in choices]
            
            except (urllib.error.HTTPError, urllib.error.URLError, socket.timeout, Exception) as e:
                last_error = e
                # 如果是 429 (Rate Limit) 或超时，才进行重试
                status_code = getattr(e, 'code', None)
                
                # 打印重试信息
                print(f"  [Attempt {attempt+1}/{max_retries}] Error: {e}. Retrying...")
                
                if attempt < max_retries - 1:
                    # 指数退避：等待 2, 4, 8 秒...
                    time.sleep(2 ** (attempt + 1))
                    continue
                else:
                    break

        raise RuntimeError(f"Failed after {max_retries} attempts. Last error: {last_error}")

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
    gt_numeric = parse_numeric(a_val)
    
    system_prompt, user_prompt = build_prompt(question)

    # 1. 获取 LLM 生成的 N 个候选回复
    try:
        outputs = client.chat_n(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            n=args.n,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
        )
    except Exception as e:
        return {"source_index": source_index, "status": "failed", "error": str(e)}

    # 2. 依次提取代码并执行
    candidate_objs: list[float] = []
    candidates_detail = []

    for idx, text in enumerate(outputs):
        code = extract_python_code(text)
        obj = None
        exec_success = False
        
        if code:
            obj = execute_and_extract_optimal(code, timeout=args.exec_timeout)
            if obj is not None:
                candidate_objs.append(obj)
                exec_success = True
        
        candidates_detail.append({
            "index": idx,
            "has_code": bool(code),
            "exec_success": exec_success,
            "obj": obj,
            "text_preview": text[:200] if not args.save_raw_text else text
        })

    # 3. 多数投票 (使用执行结果)
    voted_obj, vote_count, vote_clusters = cluster_vote(
        candidate_objs,
        rel_tol=args.vote_rel_tol,
        abs_tol=args.vote_abs_tol,
    )

    # 4. 验证结果
    hit = None
    if voted_obj is not None and gt_numeric is not None:
        hit = is_close(voted_obj, gt_numeric, rel_tol=args.gt_rel_tol, abs_tol=args.gt_abs_tol)

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
        "vote_clusters": vote_clusters
    }

def build_prompt(question: str) -> tuple[str, str]:
    return SYSTEM_PROMPT, code_prompt(question)

def load_seen_indices(path: Path) -> set[int]:
    if not path.exists(): return set()
    seen = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                seen.add(json.loads(line)["source_index"])
            except: continue
    return seen

def build_summary(log_path: Path) -> dict[str, Any]:
    total, comparable, hits = 0, 0, 0
    executed_ok = 0
    for _, row in iter_jsonl(log_path):
        if row.get("status") != "ok": continue
        total += 1
        if row.get("n_executed_ok", 0) > 0:
            executed_ok += 1
        if row.get("voted_obj") is not None and row.get("gt_obj") is not None:
            comparable += 1
            if row.get("hit"): hits += 1
            
    return {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_tasks": total,
        "tasks_with_successful_exec": executed_ok,
        "comparable_with_gt": comparable,
        "total_hits": hits,
        "exec_rate": (executed_ok / total if total > 0 else 0),
        "hit_rate_on_comparable": (hits / comparable if comparable > 0 else 0),
        "overall_accuracy": (hits / total if total > 0 else 0)
    }

def main():
    parser = argparse.ArgumentParser()
    # ... (参数部分保持不变)
    parser.add_argument("--input", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--n", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--exec-timeout", type=int, default=30)
    parser.add_argument("--parallel", type=int, default=10)
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
    summary_path = Path(args.log_dir) / f"{base_log_name}.summary.json" # 新增摘要路径
    
    seen = load_seen_indices(log_path)
    question_keys = args.question_keys.split(",")
    answer_keys = args.answer_keys.split(",")

    tasks = [(idx, row) for idx, row in iter_jsonl(input_path) if idx not in seen]
    client = OpenAICompatClient(model=args.model, api_key=args.api_key, base_url=args.base_url)

    print(f"Starting: {len(tasks)} new tasks, logging to {log_path}")

    try:
        with ThreadPoolExecutor(max_workers=args.parallel) as pool:
            future_to_idx = {
                pool.submit(process_one, source_index=idx, row=row, client=client, args=args, 
                            question_keys=question_keys, answer_keys=answer_keys): idx 
                for idx, row in tasks
            }
            
            completed_count = 0
            for i, future in enumerate(as_completed(future_to_idx)):
                res = future.result()
                append_jsonl(log_path, res)
                completed_count += 1
                
                # 每 5 个打印一次实时统计
                if completed_count % 5 == 0 or completed_count == len(tasks):
                    current_stats = build_summary(log_path)
                    print(f"Progress: {completed_count}/{len(tasks)} | "
                          f"ExecRate: {current_stats['exec_rate']:.2%} | "
                          f"HitRate: {current_stats['hit_rate_on_comparable']:.2%} "
                          f"({current_stats['total_hits']}/{current_stats['comparable_with_gt']})")

    except KeyboardInterrupt:
        print("\nInterrupted by user. Calculating final stats for completed tasks...")
    finally:
        # 无论成功、失败或中途退出，都保存并打印摘要
        if log_path.exists():
            summary = build_summary(log_path)
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=4, ensure_ascii=False)
            
            print("\n" + "="*30)
            print(f"Final Summary Saved to: {summary_path}")
            for k, v in summary.items():
                print(f"{k}: {v}")
            print("="*30)

if __name__ == "__main__":
    main()