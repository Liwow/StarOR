from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from ttrl_or.types import ExecutionResult


_RUNNER_CODE = """
import importlib.util
import json
import sys
import traceback


def _to_jsonable(value):
    try:
        json.dumps(value)
        return value
    except TypeError:
        return {"repr": repr(value)}


def main():
    target = sys.argv[1]
    payload = json.loads(sys.argv[2])
    try:
        spec = importlib.util.spec_from_file_location("candidate_solution", target)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if not hasattr(module, "solve"):
            raise AttributeError("Generated code must define solve(instance: dict)")
        result = module.solve(payload)
        print(json.dumps({"ok": True, "result": _to_jsonable(result)}, sort_keys=True))
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        print(json.dumps({"ok": False, "error": str(exc), "type": type(exc).__name__}, sort_keys=True))
        raise


if __name__ == "__main__":
    main()
""".strip()


class PythonCodeExecutor:
    def __init__(self, timeout_sec: int = 6) -> None:
        self.timeout_sec = timeout_sec

    def run(self, code: str, instance: dict[str, Any]) -> ExecutionResult:
        start = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix="ttrl_or_") as td:
            temp_dir = Path(td)
            code_file = temp_dir / "solution.py"
            runner_file = temp_dir / "runner.py"
            code_file.write_text(code, encoding="utf-8")
            runner_file.write_text(_RUNNER_CODE, encoding="utf-8")

            try:
                proc = subprocess.run(
                    [sys.executable, str(runner_file), str(code_file), json.dumps(instance)],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_sec,
                )
            except subprocess.TimeoutExpired as exc:
                elapsed = time.perf_counter() - start
                return ExecutionResult(
                    success=False,
                    output={"ok": False, "error": "Execution timeout", "type": "Timeout"},
                    stdout=str(exc.stdout or ""),
                    stderr=str(exc.stderr or ""),
                    error_type="Timeout",
                    signature="EXEC_ERROR",
                    elapsed_sec=elapsed,
                )

            elapsed = time.perf_counter() - start
            parsed = self._parse_stdout(proc.stdout)
            if proc.returncode == 0 and parsed.get("ok") is True:
                output = parsed.get("result")
                return ExecutionResult(
                    success=True,
                    output=output,
                    stdout=proc.stdout,
                    stderr=proc.stderr,
                    signature=self._signature(output),
                    elapsed_sec=elapsed,
                )

            return ExecutionResult(
                success=False,
                output=parsed,
                stdout=proc.stdout,
                stderr=proc.stderr,
                error_type=parsed.get("type") if isinstance(parsed, dict) else None,
                signature="EXEC_ERROR",
                elapsed_sec=elapsed,
            )

    @staticmethod
    def _parse_stdout(stdout: str) -> dict[str, Any]:
        lines = [line.strip() for line in stdout.splitlines() if line.strip()]
        if not lines:
            return {"ok": False, "error": "No output from candidate code.", "type": "NoOutput"}

        last = lines[-1]
        try:
            parsed = json.loads(last)
            if isinstance(parsed, dict):
                return parsed
            return {"ok": False, "error": "Last output is not a JSON object.", "type": "InvalidJSON"}
        except json.JSONDecodeError:
            return {"ok": False, "error": "Invalid JSON output.", "type": "InvalidJSON"}

    @staticmethod
    def _signature(output: Any) -> str:
        if isinstance(output, dict) and "objective" in output:
            objective = output.get("objective")
            status = output.get("status", "")
            if isinstance(objective, (int, float)):
                objective = round(float(objective), 6)
            return json.dumps({"objective": objective, "status": status}, sort_keys=True)

        try:
            return json.dumps(output, sort_keys=True)
        except TypeError:
            return repr(output)
