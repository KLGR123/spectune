"""Code interpreter with SandboxFusion and opt-in local backends."""

from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from spectune.config import CodeInterpreterConfig

from .base import JsonDict, Tool, ToolResult
from .utils import extract_last_json


class CodeInterpreterTool(Tool):
    name = "code_interpreter"
    description = "Execute Python code in a configured sandbox for deterministic computation and data processing."
    parameters: JsonDict = {
        "type": "object",
        "properties": {
            "code": {"type": "string", "description": "Python code to execute."},
            "stdin": {"type": "string", "description": "Optional standard input."},
            "timeout": {"type": "integer", "minimum": 1, "description": "Execution timeout in seconds."},
            "language": {"type": "string", "default": "python"},
        },
        "required": ["code"],
        "additionalProperties": False,
    }

    def __init__(self, config: CodeInterpreterConfig | None = None) -> None:
        self.config = config or CodeInterpreterConfig()

    async def execute(self, arguments: Mapping[str, Any]) -> ToolResult:
        code = str(arguments.get("code") or "")
        if not code.strip():
            return ToolResult(completion="failure", status="error", warnings=["code is required"])

        timeout = max(1, int(arguments.get("timeout") or self.config.timeout_s))
        language = str(arguments.get("language") or self.config.language)
        stdin = str(arguments.get("stdin") or "")

        if self.config.backend == "local":
            if not self.config.allow_local_execution:
                return ToolResult(
                    completion="failure",
                    status="unavailable",
                    warnings=["local execution is disabled; set allow_local_execution=True explicitly"],
                )
            return await asyncio.to_thread(self._execute_local, code, stdin, timeout, language)

        if not self.config.sandbox_url:
            return ToolResult(
                completion="failure",
                status="unavailable",
                warnings=["SANDBOX_FUSION_URL is not configured"],
            )
        return await asyncio.to_thread(self._execute_sandbox, code, stdin, timeout, language)

    def _execute_sandbox(self, code: str, stdin: str, timeout: int, language: str) -> ToolResult:
        payload = {
            "compile_timeout": timeout,
            "run_timeout": timeout,
            "code": code,
            "stdin": stdin,
            "memory_limit_MB": self.config.memory_limit_mb,
            "language": language,
            "files": {},
            "fetch_files": [],
        }
        request = urllib.request.Request(
            self.config.sandbox_url,
            data=json.dumps(payload, ensure_ascii=False).encode(),
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout + 10) as response:
                raw = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            return ToolResult(completion="failure", status="error", warnings=[f"HTTP {exc.code}: {detail}"])
        except Exception as exc:
            return ToolResult(
                completion="failure",
                status="error",
                warnings=[f"{type(exc).__name__}: {exc}"],
            )

        run = raw.get("run_result") if isinstance(raw, dict) else {}
        run = run if isinstance(run, dict) else {}
        stdout = self._truncate(str(run.get("stdout") or ""))
        stderr = self._truncate(str(run.get("stderr") or ""))
        success = raw.get("status") == "Success" and int(run.get("return_code") or 0) == 0
        return ToolResult(
            completion="success" if success else "failure",
            status="ok" if success else "error",
            data={
                "backend": "sandbox",
                "stdout": stdout,
                "stderr": stderr,
                "return_code": run.get("return_code"),
                "execution_time": run.get("execution_time"),
                "result_json": extract_last_json(stdout),
            },
        )

    def _execute_local(self, code: str, stdin: str, timeout: int, language: str) -> ToolResult:
        if language not in {"python", "python3"}:
            return ToolResult(
                completion="failure",
                status="error",
                warnings=[f"local backend does not support language: {language}"],
            )

        started = time.monotonic()
        with tempfile.TemporaryDirectory(prefix="spectune_code_") as directory:
            script = Path(directory) / "main.py"
            script.write_text(code + ("" if code.endswith("\n") else "\n"), encoding="utf-8")
            process = subprocess.Popen(
                [self.config.python_executable, str(script)],
                cwd=directory,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
                env=os.environ.copy(),
            )
            try:
                stdout, stderr = process.communicate(stdin, timeout=timeout)
                return_code = int(process.returncode or 0)
                timed_out = False
            except subprocess.TimeoutExpired:
                timed_out = True
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                stdout, stderr = process.communicate()
                return_code = -9

        stdout = self._truncate(stdout)
        stderr = self._truncate(stderr)
        warnings = [f"execution exceeded {timeout}s"] if timed_out else []
        return ToolResult(
            completion="success" if return_code == 0 else "failure",
            status="ok" if return_code == 0 else "error",
            data={
                "backend": "local",
                "stdout": stdout,
                "stderr": stderr,
                "return_code": return_code,
                "execution_time": time.monotonic() - started,
                "result_json": extract_last_json(stdout),
            },
            warnings=warnings,
        )

    def _truncate(self, value: str) -> str:
        limit = max(0, self.config.max_output_chars)
        if not limit or len(value) <= limit:
            return value
        return value[:limit] + "\n[spectune output truncated]\n"
