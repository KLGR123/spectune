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

from .base import JsonDict, Tool, ToolResult
from .config import CodeInterpreterConfig
from .help import HelpTool
from .utils import extract_last_json


def _wrap_python_code(code: str) -> str:
    """Run Python code with numerical libraries limited to one thread."""
    return (
        "import os as __spectune_os\n"
        '__spectune_os.environ["OPENBLAS_NUM_THREADS"] = "1"\n'
        '__spectune_os.environ["OMP_NUM_THREADS"] = "1"\n'
        '__spectune_os.environ["MKL_NUM_THREADS"] = "1"\n'
        '__spectune_os.environ["NUMEXPR_NUM_THREADS"] = "1"\n'
        '__spectune_os.environ["BLIS_NUM_THREADS"] = "1"\n'
        '__spectune_os.environ["VECLIB_MAXIMUM_THREADS"] = "1"\n'
        f'exec(compile({code!r}, "<code_interpreter>", "exec"))\n'
    )


def _strip_markdown_fence(code: str) -> str:
    """Remove an enclosing markdown code fence (`````lang ... `````) if present."""
    stripped = code.lstrip()
    if not stripped.startswith("```"):
        return code
    lines = stripped.rstrip().splitlines()
    body = lines[1:] if lines else []
    if body and body[-1].strip() == "```":
        body = body[:-1]
    return "\n".join(body)


class CodeInterpreterTool(Tool):
    name = "code_interpreter"
    description = (
        "代码运行工具，用于计算、数据处理、使用 RDKit 等等。"
    )
    parameters: JsonDict = {
        "type": "object",
        "properties": {
            "code": {"type": "string", "description": "要执行的代码"},
            "stdin": {"type": "string", "description": "（可选）标准输入"},
            # "timeout": {"type": "integer", "minimum": 1, "description": "超时（秒）"},
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
        if language in {"python", "python3"}:
            code = _strip_markdown_fence(code)
            code = _wrap_python_code(code)

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
        data = {
            "backend": "sandbox",
            "stdout": stdout,
            "return_code": run.get("return_code"),
            "execution_time": run.get("execution_time"),
            "result_json": extract_last_json(stdout),
        }
        if not success and stderr:
            data["stderr"] = stderr
        return ToolResult(
            completion="success" if success else "failure", status="ok" if success else "error", data=data
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
        data = {
            "backend": "local",
            "stdout": stdout,
            "return_code": return_code,
            "execution_time": time.monotonic() - started,
            "result_json": extract_last_json(stdout),
        }
        if return_code != 0 or stderr:
            data["stderr"] = stderr
        return ToolResult(
            completion="success" if return_code == 0 else "failure",
            status="ok" if return_code == 0 else "error",
            data=data,
            warnings=warnings,
        )

    def _truncate(self, value: str) -> str:
        limit = max(0, self.config.max_output_chars)
        if not limit or len(value) <= limit:
            return value
        return value[:limit] + "\n[spectune output truncated]\n"


CODE_INTERPRETER_GUIDE = """# code_interpreter 工具使用说明

可以自由组合、扩展、多次调用，一次写更长、更复杂的代码来完成完整分析和处理，不必局限于逐个调用 API。
变量必须用 print，不 print 就拿不到任何输出，包括中间过程检查也需 print。
如下是有关 RDKit 的一些常用 API 及探索方法，供参考。

## 如何查有哪些 API

不确定名字时先探索，不要凭印象猜。

```
import pkgutil
import rdkit.Chem
submodules = sorted(name for _, name, _ in pkgutil.iter_modules(rdkit.Chem.__path__))
print("num submodules:", len(submodules))
print(submodules[:20])

from rdkit.Chem import rdMolDescriptors
api_names = sorted(n for n in dir(rdMolDescriptors) if not n.startswith("_"))
print("rdMolDescriptors api count:", len(api_names))
print([n for n in api_names if "Formula" in n or "Mol" in n][:20])
print(rdMolDescriptors.CalcMolFormula.__doc__)
print("has CalcMolFormula:", hasattr(rdMolDescriptors, "CalcMolFormula"))
```

## 常用 API 一览

按需 import 后调用，签名、用法不确定时用上面的方法查。

- `rdkit.Chem.MolFromSmiles` / `MolToSmiles` / `MolFromSmarts`：SMILES/SMARTS 与 Mol 对象互转
- `rdkit.Chem.Descriptors.MolWt` / `ExactMolWt`：分子量 / 精确质量
- `rdkit.Chem.Descriptors.CalcMolDescriptors`：批量计算全部描述符（TPSA、MolLogP、NumHDonors、NumHAcceptors、NumRotatableBonds、NumAromaticRings 等，返回 dict）
- `rdkit.Chem.rdMolDescriptors.CalcMolFormula`：分子式
- `rdkit.Chem.rdMolDescriptors.CalcNumAromaticRings`：芳香环数
- `mol.GetAtoms()` / `atom.GetIdx()` / `GetSymbol()` / `GetTotalNumHs()` / `IsInRing()`：遍历原子
- `mol.GetBonds()` / `bond.GetBeginAtomIdx()` / `GetEndAtomIdx()` / `GetBondType()` / `GetBondTypeAsDouble()`：遍历键
- `mol.GetRingInfo()` / `ring_info.NumRings()` / `AtomRings()`：环信息
- DBE/不饱和度无现成 API，需按分子式 (C, H, N, 卤素) 手算：`C - H/2 - X/2 + N/2 + 1`
- `rdkit.Chem.Fragments.fr_*`：官能团计数（如 `fr_ester`、`fr_ketone`、`fr_amide`、`fr_phenol`、`fr_Ar_OH`、`fr_methoxy`、`fr_nitro`、`fr_halogen`、`fr_benzene`，完整列表用上面的探索方法查 `dir(Fragments)`）
- `mol.HasSubstructMatch` / `GetSubstructMatches`：配合 `MolFromSmarts` 做自定义子结构匹配
- `rdkit.Chem.AllChem.GetMorganFingerprintAsBitVect`：指纹
- `rdkit.DataStructs.TanimotoSimilarity`：指纹相似度
- 去重：对 SMILES 先 `MolFromSmiles` 再 `MolToSmiles` 取标准化形式后放入 set
- `rdkit.Chem.MolStandardize.rdMolStandardize.Normalizer().normalize` / `FragmentParent` / `StandardizeSmiles`：标准化、脱盐、互变异构体归一化
"""


class CodeInterpreterGuideTool(HelpTool):
    name = "read_code_interpreter_guide"
    description = (
        "查阅 code_interpreter 工具的 RDKit 用法示例。"
        "可选调用，写代码前不确定用哪个 API 时可以先查一下。无需参数。"
    )
    guide = CODE_INTERPRETER_GUIDE
