"""Spectune format contract ``v1`` for tool use and final answers.

This is the training/evaluation source of truth. verl hermes rollout and
``RewardEvaluator`` (strict mode) both assume this contract.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

FORMAT_SPEC_VERSION = "v1"

# Hermes-compatible tool calls (matches verl multi_turn.format=hermes).
TOOL_CALL_START = "<tool_call>"
TOOL_CALL_END = "</tool_call>"
TOOL_RESPONSE_START = "<tool_response>"
TOOL_RESPONSE_END = "</tool_response>"

_TOOL_CALL_RE = re.compile(
    rf"{re.escape(TOOL_CALL_START)}\s*(.*?)\s*{re.escape(TOOL_CALL_END)}",
    re.DOTALL | re.IGNORECASE,
)
_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_SMILES_OBJECT_RE = re.compile(r"\{[^{}]*\"smiles\"\s*:\s*\[[\s\S]*?\]\s*\}")

SYSTEM_PROMPT = """\
你是一名核磁共振谱图解析助手。你可以进行多轮思考、推理和决策，并调用提供的工具收集证据、计算和验证结果等，最终给出有序的候选分子结构。

## 思考
在每次调用工具或给出最终答案之前，先用 <think></think> 包裹你的思考过程：
<think>
（在这里思考）
</think>
调用工具前请做出尽可能仔细的思考，包括总结、推测、反思、质疑等；工具调用后请在新的 <think></think> 中给出尽可能多的观察、分析、总结、规划等，以便说明下一步依据。思考内容请输出中文。

## 工具调用
需要调用工具时，使用如下格式（同一轮可输出多个）：
<tool_call>
{"name": "工具名", "arguments": { ... }}
</tool_call>
`name` 与 `arguments` 必须是合法 JSON，且符合对应工具的参数 schema。

在调用代码工具时，需注意：
- 不要加 ```py 或 ```python 等代码块标记，直接写代码即可；
- 如果要输出结果，请使用 print() 函数打印该变量；
- 注释应该精简。

## 最终答案
完成推理后，输出一个 JSON 对象，格式固定为：
{"smiles": ["候选1", "候选2", "..."]}
`smiles` 按可能性从高到低排序，元素为有效的 SMILES 字符串。
"""

ANSWER_EXAMPLE = {"smiles": ["CCO", "COC"]}


def format_final_answer(smiles: Sequence[str]) -> str:
    """Serialize ranked SMILES candidates into the v1 final-answer JSON."""
    return json.dumps({"smiles": [str(item) for item in smiles]}, ensure_ascii=False)


def strip_tool_calls(text: str) -> str:
    """Remove hermes tool-call blocks so answer parsing sees only the reply."""
    return _TOOL_CALL_RE.sub("", text or "").strip()


def extract_tool_calls(text: str) -> list[dict[str, Any]]:
    """Parse hermes ``<tool_call>`` blocks from a model response.

    Returns ``[{"name": ..., "arguments": {...}}]``.  Blocks that are not
    valid JSON objects (or lack a ``name``) are skipped, mirroring verl's
    ``HermesToolParser``.  ``arguments`` is passed through as-is (mapping or
    JSON string) so callers can hand it straight to ``ToolManager.invoke``.
    """
    calls: list[dict[str, Any]] = []
    for match in _TOOL_CALL_RE.finditer(text or ""):
        try:
            parsed = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        if not isinstance(parsed, dict):
            continue
        name = parsed.get("name")
        if not isinstance(name, str) or not name.strip():
            continue
        arguments = parsed.get("arguments", {})
        if not isinstance(arguments, (dict, str)):
            arguments = {}
        calls.append({"name": name.strip(), "arguments": arguments})
    return calls


def format_tools_block(tool_schemas: Sequence[Mapping[str, Any]]) -> str:
    """Render the hermes tools block injected into the system prompt.

    Matches the text Qwen3's chat template emits for ``tools=[...]`` (and what
    verl GRPO therefore shows the model), so offline rollout sees the same
    tool surface as training.
    """
    rendered = "\n".join(json.dumps(schema, ensure_ascii=False) for schema in tool_schemas)
    return (
        "# Tools\n\n"
        "You may call one or more functions to assist with the user query.\n\n"
        "You are provided with function signatures within <tools></tools> XML tags:\n"
        f"<tools>\n{rendered}\n</tools>\n\n"
        "For each function call, return a json object with function name and arguments "
        "within <tool_call></tool_call> XML tags:\n"
        f'{TOOL_CALL_START}\n{{"name": <function-name>, "arguments": <args-json-object>}}\n{TOOL_CALL_END}'
    )


def _looks_like_tool_result(text: str) -> bool:
    """Return True when ``text`` looks like a Spectune ``ToolResult`` JSON blob."""
    stripped = (text or "").strip()
    if not stripped.startswith("{"):
        return False
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        return False
    return isinstance(value, dict) and "completion" in value and "status" in value


def _split_trailing_tool_response(text: str) -> tuple[str, str]:
    """Split ``tool_response + final_answer`` after the last tool call.

    Decoded verl multi-turn strings include tool-response tokens (mask=0) in
    the same flat string as assistant tokens. Prefer the last v1
    ``{"smiles": [...]}`` object as the answer; anything before it is treated
    as tool payload / chatter.
    """
    text = (text or "").strip()
    if not text:
        return "", ""

    smiles_matches = list(_SMILES_OBJECT_RE.finditer(text))
    if smiles_matches:
        last = smiles_matches[-1]
        before = text[: last.start()].strip()
        answer = text[last.start() :].strip()
        return before, answer

    # Fenced answer block without a bare smiles object match.
    fences = list(_JSON_FENCE_RE.finditer(text))
    for fence in reversed(fences):
        if _smiles_from_json_text(fence.group(1).strip()) is not None:
            return text[: fence.start()].strip(), text[fence.start() :].strip()

    if _looks_like_tool_result(text):
        return text, ""
    return "", text


def final_answer_region(text: str) -> str:
    """Return the v1 final-answer span: text after the last ``</tool_call>``.

    Tool-response JSON that verl concatenates into the decoded response is
    stripped when a later ``{"smiles": [...]}`` answer is present.
    """
    text = text or ""
    matches = list(_TOOL_CALL_RE.finditer(text))
    region = text[matches[-1].end() :].strip() if matches else text.strip()
    _, answer = _split_trailing_tool_response(region)
    return answer


def messages_from_decoded_hermes(text: str) -> list[dict[str, str]]:
    """Rebuild a message list from a decoded hermes / multi-turn response string.

    Each ``<tool_call>...</tool_call>`` becomes its own assistant turn. Text
    between calls (tool responses that verl glued into the decoded string)
    becomes ``role=tool``. Text after the last tool call is split into an
    optional tool response and the final assistant answer turn used for GT.
    """
    text = text or ""
    matches = list(_TOOL_CALL_RE.finditer(text))
    if not matches:
        return [{"role": "assistant", "content": text}]

    messages: list[dict[str, str]] = []
    cursor = 0
    for match in matches:
        between = text[cursor : match.start()].strip()
        if between and messages:
            messages.append({"role": "tool", "content": between})
        messages.append({"role": "assistant", "content": match.group(0)})
        cursor = match.end()

    trailing = text[cursor:].strip()
    if trailing:
        tool_part, answer_part = _split_trailing_tool_response(trailing)
        if tool_part:
            messages.append({"role": "tool", "content": tool_part})
        if answer_part:
            messages.append({"role": "assistant", "content": answer_part})
    return messages or [{"role": "assistant", "content": text}]


def extract_smiles_candidates(text: str) -> list[str]:
    """Extract the ranked SMILES list from a v1 final answer.

    Accepted shapes:
    - bare JSON: ``{"smiles": ["CCO", ...]}``
    - fenced JSON block containing the same object
    - the last JSON object in the text that has a ``smiles`` array

    When ``text`` still contains hermes tool calls, only the region after the
    last tool call is considered (see :func:`final_answer_region`).
    """
    cleaned = final_answer_region(text)
    if not cleaned:
        return []

    for candidate_text in _candidate_json_blobs(cleaned):
        smiles = _smiles_from_json_text(candidate_text)
        if smiles is not None:
            return smiles
    return []


def _candidate_json_blobs(text: str) -> list[str]:
    blobs: list[str] = [text.strip()]
    blobs.extend(fence.strip() for fence in _JSON_FENCE_RE.findall(text) if fence.strip())
    blobs.extend(match.group(0) for match in _SMILES_OBJECT_RE.finditer(text))
    # Prefer later blobs (often the final answer after tool chatter).
    return list(dict.fromkeys(reversed(blobs)))


def _smiles_from_json_text(text: str) -> list[str] | None:
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return None
    return _smiles_from_value(value)


def _smiles_from_value(value: Any) -> list[str] | None:
    if not isinstance(value, dict) or "smiles" not in value:
        return None
    raw = value["smiles"]
    if isinstance(raw, str):
        cleaned = raw.strip()
        return [cleaned] if cleaned else []
    if isinstance(raw, list):
        out: list[str] = []
        for item in raw:
            if isinstance(item, str) and item.strip():
                out.append(item.strip())
            elif isinstance(item, dict):
                for key in ("smiles", "canonical_smiles"):
                    if isinstance(item.get(key), str) and item[key].strip():
                        out.append(item[key].strip())
                        break
        return out
    return None


__all__ = [
    "ANSWER_EXAMPLE",
    "FORMAT_SPEC_VERSION",
    "SYSTEM_PROMPT",
    "TOOL_CALL_END",
    "TOOL_CALL_START",
    "TOOL_RESPONSE_END",
    "TOOL_RESPONSE_START",
    "extract_smiles_candidates",
    "extract_tool_calls",
    "final_answer_region",
    "format_final_answer",
    "format_tools_block",
    "messages_from_decoded_hermes",
    "strip_tool_calls",
]
