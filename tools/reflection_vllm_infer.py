import argparse
import json
import re
import subprocess
import tempfile
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Optional, Tuple
import urllib.request
import urllib.error
import socket
import time

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
                # 禁用系统代理以防止本地连接失败
                with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(req, timeout=self.timeout) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                    return body["choices"][0]["message"]["content"]
            except Exception as e:
                if attempt == 2: raise
                time.sleep(2 ** attempt)
        return ""

# --- 提示词模板 ---

SYSTEM_PROMPT = "You are an expert Operations Research engineer and Gurobi optimizer."

def get_initial_prompt(question):
    return f"""Answer the following mathematical modeling question:
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

def get_reflection_prompt(code, error):
    return f"""Your previous Gurobi code failed. 
[Previous Code]:
{code}

[Error Message]:
{error}

Analyze why the code failed (e.g., incorrect constraints, syntax error, or infeasibility). 
Provide a concise reflection on how to fix it."""

def get_retry_prompt(question, reflection):
    return f"""Question: {question}

[Reflection on previous attempt]:
{reflection}

Based on the reflection, provide the corrected Gurobi code in <python> tags.
"""

# --- 执行引擎 (Evaluator) ---

def execute_code(code: str, timeout: int = 30) -> Tuple[bool, Optional[float], str]:
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tf:
        tf.write(code)
        path = tf.name
    try:
        result = subprocess.run(["python", path], capture_output=True, text=True, timeout=timeout)
        output = result.stdout
        stderr = result.stderr
        combined_output = output + "\n" + stderr
        
        num_pattern = r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"
        obj = None
        match = re.search(rf"Optimal value:?\s*({num_pattern})", output, re.IGNORECASE)
        
        if not match:
            match = re.search(rf"Best objective\s+:?\s*({num_pattern})", combined_output, re.IGNORECASE)
            
        if match:
            try:
                obj = float(match.group(1))
            except ValueError:
                obj = None
        
        success = (result.returncode == 0)
        return success, obj, combined_output
        
    except subprocess.TimeoutExpired:
        return False, None, "Execution Timed Out"
    except Exception as e:
        return False, None, str(e)
    finally:
        if os.path.exists(path): os.remove(path)

# --- Reflexion 核心循环 ---

def process_reflexion(source_index, row, client, args, q_key, a_key):
    question = str(row[q_key]).strip()
    gt_obj = parse_numeric(row.get(a_key))
    
    memory = [] 
    history = [] 
    
    current_question_prompt = get_initial_prompt(question)
    
    for t in range(args.max_trials):
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        if t > 0:
            for m in memory:
                messages.append({"role": "assistant", "content": m['code']})
                messages.append({"role": "user", "content": f"[Reflection]: {m['reflection']}\nPlease fix the code."})
        
        messages.append({"role": "user", "content": current_question_prompt if t==0 else "Try again with the fix."})
        
        # 传入 max_tokens 参数
        response_text = client.chat(messages, temperature=args.temperature, max_tokens=args.max_tokens)
        code = extract_python_code(response_text)
        
        if not code:
            history.append({"trial": t, "status": "no_code", "response": response_text})
            # 如果没生成代码，直接进行反思尝试
            feedback = "No python code block found in your response."
            success = False
            obj = None
        else:
            success, obj, feedback = execute_code(code, timeout=args.exec_timeout)
        
        is_hit = False
        if obj is not None and gt_obj is not None:
            is_hit = is_close(obj, gt_obj, rel_tol=1e-4, abs_tol=1e-6)

        history.append({
            "trial": t,
            "code": code,
            "success": success,
            "obj": obj,
            "feedback": feedback if not success else "Success",
            "hit": is_hit
        })

        if success and obj is not None:
            break
            
        # Self-Reflection
        reflection_msg = [{"role": "system", "content": SYSTEM_PROMPT},
                          {"role": "user", "content": get_reflection_prompt(code if code else response_text, feedback)}]
        reflection = client.chat(reflection_msg, temperature=0.3, max_tokens=args.max_tokens)
        
        memory.append({"code": response_text, "reflection": reflection})
        current_question_prompt = get_retry_prompt(question, reflection)

    final_res = history[-1]
    return {
        "source_index": source_index,
        "status": "ok",
        "trials_taken": len(history),
        "final_obj": final_res.get("obj"),
        "gt_obj": gt_obj,
        "hit": any(h.get("hit") for h in history),
        "history": history
    }

# --- 辅助函数 ---
def parse_numeric(v: Any) -> Optional[float]:
    if v is None: return None
    if isinstance(v, (int, float)): return float(v)
    text = str(v).strip().replace(",", "")
    num_pattern = r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"
    matches = re.findall(num_pattern, text)
    if matches:
        try:
            return float(matches[-1])
        except:
            return None
    return None

def extract_python_code(text):
    match = re.search(r"<python>(.*?)</python>", text, re.DOTALL)
    return match.group(1).strip() if match else None

def is_close(a, b, rel_tol=1e-4, abs_tol=1e-6):
    if a is None or b is None: return False
    try:
        return abs(float(a) - float(b)) <= abs_tol + rel_tol * max(abs(float(a)), abs(float(b)), 1.0)
    except:
        return False

# --- 主程序逻辑 ---

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--max-trials", type=int, default=3)
    parser.add_argument("--parallel", type=int, default=5)
    parser.add_argument("--exec-timeout", type=int, default=30)
    parser.add_argument("--temperature", type=float, default=0.5)
    
    # --- 新增适配启动脚本的参数 ---
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--log-dir", default="outputs/reflexion_logs")
    parser.add_argument("--timeout", type=int, default=120, help="API request timeout")
    
    args = parser.parse_args()

    input_path = Path(args.input)
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"reflexion_results_{input_path.stem}.jsonl"
    
    data = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))

    if not data:
        print("No data found.")
        return

    sample = data[0]
    q_key = next((k for k in ["question","en_question", "input", "prompt"] if k in sample), "question")
    a_key = next((k for k in ["answer","en_answer", "output", "gt", "target"] if k in sample), "answer")

    print(f"Starting Reflexion: {len(data)} tasks, Max Trials: {args.max_trials}")

    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        futures = []
        for i, row in enumerate(data):
            # 初始化 Client 时传入自定义 timeout
            client = OpenAICompatClient(
                model=args.model, 
                api_key=args.api_key, 
                base_url=args.base_url, 
                timeout=args.timeout
            )
            futures.append(pool.submit(process_reflexion, i, row, client, args, q_key, a_key))
        
        hits = 0
        total = 0
        for f in as_completed(futures):
            res = f.result()
            total += 1
            if res.get("hit"): hits += 1
            
            with open(log_path, "a", encoding="utf-8") as fw:
                fw.write(json.dumps(res, ensure_ascii=False) + "\n")
            
            print(f"Progress: {total}/{len(data)} | Current Accuracy: {hits/total:.2%}")

if __name__ == "__main__":
    main()
