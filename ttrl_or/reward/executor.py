from __future__ import annotations

import glob
import json
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any
import re
from ttrl_or.types import ExecutionResult, ModelInfo


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


def _pick_script_result(module):
    for key in ("result", "output", "objective", "optimal", "obj"):
        if hasattr(module, key):
            return getattr(module, key)
    return None


def main():
    target = sys.argv[1]
    payload = json.loads(sys.argv[2])
    try:
        spec = importlib.util.spec_from_file_location("candidate_solution_runtime", target)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        if hasattr(module, "solve"):
            result = module.solve(payload)
            print(json.dumps({"ok": True, "mode": "solve", "result": _to_jsonable(result)}, sort_keys=True))
            return

        # Script-style fallback: module top-level already executed on import.
        result = _pick_script_result(module)
        print(json.dumps({"ok": True, "mode": "script", "result": _to_jsonable(result)}, sort_keys=True))

    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        print(json.dumps({"ok": False, "error": str(exc), "type": type(exc).__name__}, sort_keys=True))
        raise


if __name__ == "__main__":
    main()
""".strip()


class PythonCodeExecutor:
    """
    Code executor with two modes:
    - subprocess: one temporary directory per run (strong isolation, slower startup).
    - sandbox: one persistent temp workspace reused across runs (faster I/O setup).
    """

    def __init__(self, timeout_sec: int = 6, mode: str = "subprocess") -> None:
        self.timeout_sec = timeout_sec
        self.mode = mode

        self._sandbox_dir: Path | None = None
        self._runner_file: Path | None = None
        self._thread_state = threading.local()

        if self.mode == "sandbox":
            self._init_sandbox()
        elif self.mode != "subprocess":
            raise ValueError(f"Unsupported code executor mode: {self.mode}")

    def run(self, code: str, instance: dict[str, Any]) -> ExecutionResult:
        if self.mode == "sandbox":
            return self._run_in_sandbox(code=code, instance=instance)
        return self._run_in_subprocess_mode(code=code, instance=instance)

    def close(self) -> None:
        if self._sandbox_dir is None:
            return
        try:
            shutil.rmtree(self._sandbox_dir, ignore_errors=True)
        finally:
            self._sandbox_dir = None
            self._runner_file = None

    def __del__(self):  # pragma: no cover
        self.close()

    def _init_sandbox(self) -> None:
        sandbox_path = Path(tempfile.mkdtemp(prefix="ttrl_or_sandbox_"))
        runner_file = sandbox_path / "runner.py"

        runner_file.write_text(_RUNNER_CODE, encoding="utf-8")

        self._sandbox_dir = sandbox_path
        self._runner_file = runner_file

    def _run_in_sandbox(self, code: str, instance: dict[str, Any]) -> ExecutionResult:
        worker_dir, runner_file, run_index = self._get_worker_sandbox()

        start = time.perf_counter()
        solution_file = worker_dir / f"solution_{run_index}.py"
        instrumented_code, lp_injection_applied = self._inject_lp_dump_before_optimize(code)
        solution_file.write_text(instrumented_code, encoding="utf-8")

        return self._invoke_runner(
            runner_path=runner_file,
            solution_path=solution_file,
            instance=instance,
            cwd=worker_dir,
            start=start,
            lp_injection_applied=lp_injection_applied,
        )

    def _run_in_subprocess_mode(self, code: str, instance: dict[str, Any]) -> ExecutionResult:
        start = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix="ttrl_or_") as td:
            temp_dir = Path(td)
            code_file = temp_dir / "solution.py"
            runner_file = temp_dir / "runner.py"
            instrumented_code, lp_injection_applied = self._inject_lp_dump_before_optimize(code)
            code_file.write_text(instrumented_code, encoding="utf-8")
            runner_file.write_text(_RUNNER_CODE, encoding="utf-8")

            return self._invoke_runner(
                runner_path=runner_file,
                solution_path=code_file,
                instance=instance,
                cwd=temp_dir,
                start=start,
                lp_injection_applied=lp_injection_applied,
            )

    def _get_worker_sandbox(self) -> tuple[Path, Path, int]:
        assert self._sandbox_dir is not None

        worker_dir = getattr(self._thread_state, "worker_dir", None)
        runner_file = getattr(self._thread_state, "runner_file", None)
        run_index = int(getattr(self._thread_state, "run_index", 0)) + 1

        if worker_dir is None or runner_file is None:
            worker_name = f"worker_{threading.get_ident()}"
            worker_dir = self._sandbox_dir / worker_name
            worker_dir.mkdir(parents=True, exist_ok=True)
            runner_file = worker_dir / "runner.py"
            if not runner_file.exists():
                runner_file.write_text(_RUNNER_CODE, encoding="utf-8")
            self._thread_state.worker_dir = worker_dir
            self._thread_state.runner_file = runner_file

        self._thread_state.run_index = run_index
        return worker_dir, runner_file, run_index

    @staticmethod
    def _inject_lp_dump_before_optimize(code: str) -> tuple[str, bool]:
        raw = str(code or "")
        if not raw.strip():
            return raw, False

        lowered = raw.lower()
        if '.write(' in lowered and '.lp' in lowered:
            return raw, False

        lines = raw.splitlines()
        optimize_re = re.compile(r'^(?P<indent>\s*)(?P<var>[A-Za-z_][A-Za-z0-9_]*)\.optimize\s*\(')
        write_re_template = r'^(?P<indent>\s*){var}\.write\s*\('
        inserted = False

        def previous_meaningful_line(idx: int) -> str:
            j = idx - 1
            while j >= 0:
                candidate = lines[j].strip()
                if candidate and not candidate.startswith('#'):
                    return candidate
                j -= 1
            return ''

        new_lines: list[str] = []
        for idx, line in enumerate(lines):
            match = optimize_re.match(line)
            if match is not None:
                indent = match.group('indent')
                var_name = match.group('var')
                previous = previous_meaningful_line(idx)
                write_re = re.compile(write_re_template.format(var=re.escape(var_name)))
                if not write_re.match(previous):
                    new_lines.append(f'{indent}# Auto-added for structural reward extraction')
                    new_lines.append(f'{indent}{var_name}.write("ttrl_model.lp")')
                    inserted = True
            new_lines.append(line)

        if not inserted:
            return raw, False
        suffix = "\n" if raw.endswith("\n") else ""
        return "\n".join(new_lines) + suffix, True

    def _invoke_runner(
        self,
        runner_path: Path,
        solution_path: Path,
        instance: dict[str, Any],
        cwd: Path,
        start: float,
        lp_injection_applied: bool = False,
    ) -> ExecutionResult:
        try:
            proc = subprocess.run(
                [sys.executable, str(runner_path), str(solution_path), json.dumps(instance)],
                capture_output=True,
                text=True,
                timeout=self.timeout_sec,
                cwd=str(cwd),
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
                model_info=None,
                lp_injection_applied=bool(lp_injection_applied),
            )

        elapsed = time.perf_counter() - start
        parsed = self._parse_stdout(proc.stdout)

        # Extract model info from .lp file if exists
        model_info = self._extract_model_info_from_lp(cwd)

        if proc.returncode == 0 and parsed.get("ok") is True:
            output = parsed.get("result")
            return ExecutionResult(
                success=True,
                output=output,
                stdout=proc.stdout,
                stderr=proc.stderr,
                signature=self._signature(output),
                elapsed_sec=elapsed,
                model_info=model_info,
                lp_injection_applied=bool(lp_injection_applied),
            )

        return ExecutionResult(
            success=False,
            output=parsed,
            stdout=proc.stdout,
            stderr=proc.stderr,
            error_type=parsed.get("type") if isinstance(parsed, dict) else None,
            signature="EXEC_ERROR",
            elapsed_sec=elapsed,
            model_info=model_info,
            lp_injection_applied=bool(lp_injection_applied),
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

    @staticmethod
    def _extract_model_info_from_lp(cwd: Path) -> ModelInfo | None:
        """Extract Gurobi model structural info from .lp file in working directory."""
        lp_files = list(glob.glob(str(cwd / "*.lp")))
        if not lp_files:
            return None

        # Use the most recently modified .lp file
        lp_path = max(lp_files, key=lambda f: Path(f).stat().st_mtime)

        try:
            import gurobipy as gp
        except ImportError:
            # Gurobi not available, fallback to text parsing
            return PythonCodeExecutor._parse_lp_file_text(lp_path)

        try:
            model = gp.read(lp_path)
            model_sense = int(model.ModelSense)
            num_vars = int(model.NumVars)
            num_constrs = int(model.NumConstrs)
            model_info = ModelInfo(
                model_sense=model_sense,  # 1=min, -1=max
                num_vars=num_vars,
                num_bin_vars=int(model.NumBinVars),
                num_int_vars=int(model.NumIntVars),
                num_constrs=num_constrs,
                has_objective=bool(model_sense in (-1, 1)),
                has_constraints=bool(num_constrs > 0),
                has_variables=bool(num_vars > 0),
                extracted=True,
            )
            model.dispose()
            return model_info
        except Exception:
            return PythonCodeExecutor._parse_lp_file_text(lp_path)

    @staticmethod
    def _parse_lp_file_text(lp_path: str) -> ModelInfo | None:
        """Fallback: parse .lp file as text to extract basic structure info."""
        try:
            with open(lp_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read().lower()

            # Detect model sense
            model_sense = 0
            if "minimize" in content:
                model_sense = 1
            elif "maximize" in content:
                model_sense = -1

            # Count variables (rough estimation from Bounds/Binary/General sections)
            import re
            # Count lines in Bounds section
            bounds_match = re.search(r"bounds\s*\n([\s\S]*?)(?:binary|general|end|$)", content)
            num_vars = 0
            if bounds_match:
                bounds_lines = [l.strip() for l in bounds_match.group(1).strip().split("\n") if l.strip()]
                num_vars = len(bounds_lines)

            # Count binary variables
            binary_match = re.search(r"binary\s*\n([\s\S]*?)(?:general|end|$)", content)
            num_bin_vars = 0
            if binary_match:
                bin_lines = [l.strip() for l in binary_match.group(1).strip().split("\n") if l.strip()]
                num_bin_vars = len(bin_lines)

            # Count integer (general) variables
            general_match = re.search(r"general\s*\n([\s\S]*?)(?:end|$)", content)
            num_int_vars = 0
            if general_match:
                gen_lines = [l.strip() for l in general_match.group(1).strip().split("\n") if l.strip()]
                num_int_vars = len(gen_lines)

            # Count constraints (Subject To section)
            st_match = re.search(r"subject to\s*\n([\s\S]*?)(?:bounds|binary|general|end|$)", content)
            num_constrs = 0
            if st_match:
                constr_lines = [l.strip() for l in st_match.group(1).strip().split("\n") if l.strip() and ":" in l]
                num_constrs = len(constr_lines)

            total_vars = max(num_vars, num_bin_vars + num_int_vars)
            return ModelInfo(
                model_sense=model_sense,
                num_vars=total_vars,
                num_bin_vars=num_bin_vars,
                num_int_vars=num_int_vars,
                num_constrs=num_constrs,
                has_objective=bool(model_sense in (-1, 1)),
                has_constraints=bool(num_constrs > 0),
                has_variables=bool(total_vars > 0),
                extracted=True,
            )
        except Exception:
            return None

