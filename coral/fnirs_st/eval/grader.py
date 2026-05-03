"""
fNIRS ST-GNN temporal module search grader.

Runs solution.run(data_dir) using the project venv and returns the
5-fold CV mean F1 score for the current temporal module configuration.
"""
from __future__ import annotations

import json
import os
import subprocess
import textwrap

from coral.grader import TaskGrader
from coral.types import ScoreBundle


class Grader(TaskGrader):
    def evaluate(self) -> float | ScoreBundle:
        data_dir = self.args.get("data_dir", "")
        python_exe = self.args.get(
            "python_executable",
            "/home/user/jeffrymahbuubi/PROJECTS/2-fNIRS-Graph-Base-Method/src/.venv/bin/python",
        )
        program_file = self.args.get("program_file", "solution.py")

        program_path = os.path.join(self.codebase_path, program_file)

        if not os.path.exists(program_path):
            return self.fail(f"solution file not found: {program_path}")
        if not data_dir or not os.path.isdir(data_dir):
            return self.fail(f"data_dir not found: {data_dir!r}")

        codebase = os.path.abspath(self.codebase_path)

        script = textwrap.dedent(f"""\
            import json, sys, os, traceback
            sys.path.insert(0, {codebase!r})
            try:
                import solution
                f1 = solution.run({data_dir!r})
                print(json.dumps({{"f1_score": float(f1), "status": "ok"}}))
            except Exception as exc:
                print(json.dumps({{"error": str(exc), "traceback": traceback.format_exc()[-2000:]}}))
        """)

        try:
            proc = subprocess.run(
                [python_exe, "-c", script],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=codebase,
            )
        except subprocess.TimeoutExpired:
            return self.fail(f"Timed out after {self.timeout}s")
        except Exception as exc:
            return self.fail(f"Failed to launch solution: {exc}")

        if proc.returncode != 0:
            return self.fail(
                f"Exited with code {proc.returncode}.\n"
                f"stderr: {proc.stderr.strip()[-2000:]}"
            )

        stdout = proc.stdout.strip()
        parsed = None
        for line in reversed(stdout.splitlines()):
            line = line.strip()
            if line.startswith("{"):
                try:
                    parsed = json.loads(line)
                    break
                except json.JSONDecodeError:
                    continue

        if parsed is None:
            return self.fail(
                f"No JSON in stdout.\n"
                f"stdout: {stdout[-500:]}\n"
                f"stderr: {proc.stderr.strip()[-500:]}"
            )

        if "error" in parsed:
            return self.fail(
                f"Solution raised an error: {parsed['error']}\n"
                f"{parsed.get('traceback', '')}"
            )

        f1 = parsed.get("f1_score")
        if f1 is None:
            return self.fail(f"Missing f1_score in output: {parsed}")

        explanation = f"5-fold CV mean F1 = {f1:.4f}  (HBO mt4, 62 subjects, processed-new-mc)"
        return self.score(float(f1), explanation)
