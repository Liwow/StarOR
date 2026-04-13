import argparse
import json
import re
import subprocess
import tempfile
import os
import time
import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Optional, Tuple
import urllib.request
import urllib.error

# --- API 客户端 ---

class OpenAICompatClient:
    def __init__(self, *, model: str, api_key: str, base_url: str, timeout: int = 120):
        self.model = model
        self.api_key = api_key
        base = base_url.strip().rstrip("/")
        if not base.endswith("/v1"): base += "/v1"
        self.endpoint = f"{base}/chat/completions"
        self.timeout = timeout

    def chat(self, messages: list, **kwargs) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 2048),
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.endpoint, data=data,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
            method="POST"
        )
        
        for attempt in range(3):
            try:
                with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(req, timeout=self.timeout) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                    return body["choices"][0]["message"]["content"]
            except urllib.error.HTTPError as e:
                err_content = e.read().decode("utf-8")
                print(f"\n[HTTP Error {e.code}]: {err_content}")
                if attempt == 2: raise
                time.sleep(1)
            except Exception as e:
                if attempt == 2: raise
                time.sleep(2 ** attempt)
        return ""

# --- 提示词模板 ---

SYSTEM_PROMPT = "You are an expert Operations Research engineer and Gurobi optimizer."

def get_initial_prompt(question):
    return f"Answer the following mathematical modeling question:\n{question}\nYou should directly output the final python code using Gurobi within <python> tags."

def get_reflection_prompt(code, error):
    return f"Your previous Gurobi code failed.\n[Code]:\n{code}\n[Error]:\n{error}\nAnalyze why it failed and provide a fix."

def get_retry_prompt(question, reflection):
    return f"Question: {question}\n[Reflection]: {reflection}\nProvide the corrected Gurobi code in <python> tags."

# --- 执行引擎 ---

def execute_code(code: str, timeout: int = 30) -> Tuple[bool, Optional[float], str]:
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tf:
        tf.write(code)
        path = tf.name
    try:
        result = subprocess.run(["python", path], capture_output=True, text=True, timeout=timeout)
        output = result.stdout
        combined_output = output + "\n" + result.stderr
        num_pattern = r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"
        obj = None
        match = re.search(rf"Optimal value:?\s*({num_pattern})", output, re.IGNORECASE)
        if not match:
            match = re.search(rf"Best objective\s+:?\s*({num_pattern})", combined_output, re.IGNORECASE)
        if match:
            try: obj = float(match.group(1))
            except: obj = None
        return (result.returncode == 0), obj, combined_output
    except Exception as e:
        return False, None, str(e)
    finally:
        if os.path.exists(path): os.remove(path)

# --- Reflexion 循环 ---

def process_reflexion(source_index, row, client, args, q_key, a_key):
    question = str(row[q_key]).strip()
    gt_obj = parse_numeric(row.get(a_key))
    history = []
    memory = []
    
    current_prompt = get_initial_prompt(question)
    
    for t in range(args.max_trials):
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        if t > 0:
            for m in memory:
                messages.append({"role": "assistant", "content": m['code']})
                messages.append({"role": "user", "content": f"[Reflection]: {m['reflection']}\nPlease fix."})
        messages.append({"role": "user", "content": current_prompt if t==0 else "Try again with the fix."})
        
        resp = client.chat(messages, temperature=args.temperature, max_tokens=args.max_tokens)
        if not resp: break
        
        code = extract_python_code(resp)
        success, obj, feedback = execute_code(code, timeout=args.exec_timeout) if code else (False, None, "No code block found.")
        feedback = feedback[:4000]
        hit = is_close(obj, gt_obj) if obj is not None else False
        history.append({"trial": t, "success": success, "obj": obj, "hit": hit, "feedback": feedback})

        if success and obj is not None: break
        
        # Reflection step
        refl_msg = [{"role": "system", "content": SYSTEM_PROMPT}, 
                    {"role": "user", "content": get_reflection_prompt(code if code else resp, feedback)}]
        reflection = client.chat(refl_msg, temperature=0.3, max_tokens=1024)
        if not reflection: break
        
        memory.append({"code": resp, "reflection": reflection})
        current_prompt = get_retry_prompt(question, reflection)

    return {
        "source_index": source_index,
        "hit": any(h["hit"] for h in history),
        "history": history,
        "gt_obj": gt_obj
    }

# --- 辅助函数 ---

def parse_numeric(v):
    if v is None or isinstance(v, (int, float)): return v
    matches = re.findall(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", str(v).replace(",", ""))
    return float(matches[-1]) if matches else None

def extract_python_code(text):
    match = re.search(r"<python>(.*?)</python>", text, re.DOTALL)
    return match.group(1).strip() if match else None

def is_close(a, b, rel_tol=1e-4, abs_tol=1e-6):
    if a is None or b is None: return False
    return abs(float(a) - float(b)) <= abs_tol + rel_tol * max(abs(float(a)), abs(float(b)), 1.0)

# --- 主程序 ---

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--max-trials", type=int, default=3)
    parser.add_argument("--parallel", type=int, default=5)
    parser.add_argument("--exec-timeout", type=int, default=30)
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--log-dir", default="outputs/reflexion_logs")
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()

    input_path = Path(args.input)
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # 定义输出文件路径
    log_path = log_dir / f"reflexion_results_{input_path.stem}.jsonl"
    summary_path = log_dir / f"summary_{input_path.stem}.json"
    
    data = [json.loads(line) for line in open(input_path, "r", encoding="utf-8") if line.strip()]
    q_key = next((k for k in ["en_question", "input", "prompt"] if k in data[0]), "question")
    a_key = next((k for k in ["en_answer", "output", "gt", "target"] if k in data[0]), "answer")

    print(f"\n>>> Running: {input_path.name} | Total: {len(data)}")

    hits = 0
    total_processed = 0
    results = []

    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        futures = [pool.submit(process_reflexion, i, row, OpenAICompatClient(model=args.model, api_key=args.api_key, base_url=args.base_url, timeout=args.timeout), args, q_key, a_key) 
                   for i, row in enumerate(data)]
        
        for f in as_completed(futures):
            res = f.result()
            total_processed += 1
            if res["hit"]: hits += 1
            
            with open(log_path, "a", encoding="utf-8") as fw:
                fw.write(json.dumps(res, ensure_ascii=False) + "\n")
            
            print(f"Progress: {total_processed}/{len(data)} | Acc: {hits/total_processed:.2%}", end="\r")

    # --- 保存汇总信息 ---
    accuracy = hits / total_processed if total_processed > 0 else 0
    summary_data = {
        "dataset": input_path.name,
        "model": args.model,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_samples": total_processed,
        "total_hits": hits,
        "accuracy": accuracy,
        "config": vars(args)
    }

    # 1. 写入独立的 summary.json
    with open(summary_path, "w", encoding="utf-8") as sw:
        json.dump(summary_data, sw, indent=4, ensure_ascii=False)

    # 2. 追加一行到 jsonl 末尾 (带 summary 类型标识)
    with open(log_path, "a", encoding="utf-8") as fw:
        fw.write(json.dumps({"type": "summary", "data": summary_data}, ensure_ascii=False) + "\n")

    print(f"\n{'='*40}")
    print(f"Final Accuracy: {accuracy:.2%}")
    print(f"Summary saved to: {summary_path}")
    print(f"{'='*40}\n")

if __name__ == "__main__":
    main()
