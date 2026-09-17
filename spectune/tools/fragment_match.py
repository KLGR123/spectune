"""Verify a ranked list of candidate structures against explicit fragment evidence."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any

from .base import JsonDict, Tool, ToolResult
from .config import FragmentMatchConfig
from .utils import as_list, canonical_smiles, post_json

_FRAGMENT_FORMATS = {"text", "smiles", "smarts"}
_MAX_RERANK_ITEMS = 1000
_MAX_FRAGMENT_ITEMS = 64


class FragmentMatchTool(Tool):
    name = "fragment_match"
    description = (
        "用显式的碎片证据核验一份候选结构排序列表，只有当碎片约束要求时才修改排名第一的候选。"
        "如果有任何碎片无法解析，或没有候选能满足所有已解析的碎片，则保留原本排名第一的候选。"
    )

    def __init__(self, config: FragmentMatchConfig | None = None) -> None:
        self.config = config or FragmentMatchConfig()

    @property
    def parameters(self) -> JsonDict:
        return {
            "type": "object",
            "properties": {
                "rerank": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": _MAX_RERANK_ITEMS,
                    "description": (
                        "完整的排序候选列表，按原始最优在前的顺序原样复制。"
                        "每一项必须只包含 rank 和 smiles。"
                    ),
                    "items": {
                        "type": "object",
                        "properties": {
                            "rank": {
                                "type": "integer",
                                "minimum": 1,
                                "description": (
                                    "从 1 开始的候选排名，必须等于该候选在此数组中的位置。"
                                ),
                            },
                            "smiles": {
                                "type": "string",
                                "minLength": 1,
                                "description": "该重排候选的有效 SMILES。",
                            },
                        },
                        "required": ["rank", "smiles"],
                        "additionalProperties": False,
                    },
                },
                "fragments": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": _MAX_FRAGMENT_ITEMS,
                    "description": (
                        "从用户查询中提取的显式碎片约束，按原始出现顺序排列；多条约束会被联合应用。"
                        "只收录明确、可信的约束（非英文名称需翻译为英文）——"
                        "切勿把 'may contain' 这类推测性措辞当作硬性约束。"
                    ),
                    "items": {
                        "type": "object",
                        "properties": {
                            "format": {
                                "type": "string",
                                "enum": sorted(_FRAGMENT_FORMATS),
                                "description": (
                                    "碎片的表示类型：text 表示简洁的英文化学名称，"
                                    "smiles 表示具体的碎片结构，"
                                    "smarts 表示结构模式。"
                                ),
                            },
                            "value": {
                                "type": "string",
                                "minLength": 1,
                                "maxLength": 4096,
                                "description": (
                                    "碎片值。当 format 为 text 时，只提供简洁的英文核心碎片名称，"
                                    "例如 '4-methoxyphenyl'，而不是完整的用户原句。"
                                    "当 format 为 smiles 或 smarts 时，提供符合所声明格式的有效表达式。"
                                ),
                            },
                        },
                        "required": ["format", "value"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["rerank", "fragments"],
            "additionalProperties": False,
        }

    async def execute(self, arguments: Mapping[str, Any]) -> ToolResult:
        if not self.config.api_url:
            return ToolResult(
                completion="failure",
                status="unavailable",
                warnings=["FRAGMENT_MATCH_API_URL is not configured"],
            )

        args = dict(arguments or {})
        warnings: list[str] = []

        rerank, rerank_warnings = _normalize_rerank(args.get("rerank"))
        warnings.extend(rerank_warnings)
        if not rerank:
            return ToolResult(
                completion="failure",
                status="error",
                warnings=[*warnings, "fragment_match requires a non-empty rerank list of {rank, smiles} items"],
            )

        fragments, fragment_warnings = _normalize_fragments(args.get("fragments"))
        warnings.extend(fragment_warnings)
        if not fragments:
            return ToolResult(
                completion="failure",
                status="error",
                warnings=[
                    *warnings,
                    "fragment_match requires at least one resolved fragment (format text/smiles/smarts + value)",
                ],
            )

        payload = {"rerank": rerank, "fragments": fragments}
        try:
            raw = await asyncio.to_thread(post_json, self.config.api_url, payload, timeout=self.config.timeout_s)
        except Exception as exc:
            return ToolResult(
                completion="failure",
                status="error",
                data={"request": payload},
                warnings=[*warnings, f"{type(exc).__name__}: {exc}"],
            )

        recommended_raw = raw.get("recommended")
        if not isinstance(recommended_raw, dict) or not str(recommended_raw.get("smiles") or "").strip():
            return ToolResult(
                completion="failure",
                status="error",
                data={"request": payload, "response": raw},
                warnings=[*warnings, "fragment_match response is missing a valid recommended candidate"],
            )

        recommended_smiles = canonical_smiles(str(recommended_raw.get("smiles")))
        try:
            recommended_rank = int(recommended_raw.get("rank"))
        except (TypeError, ValueError):
            recommended_rank = None
        recommended = {"rank": recommended_rank, "smiles": recommended_smiles, "canonical_smiles": recommended_smiles}

        status = str(raw.get("status") or "").strip() or "ok"
        return ToolResult(
            completion="success",
            status=status,
            data={
                "request": payload,
                "recommended": recommended,
                "changed_from_rerank_top1": bool(raw.get("changed_from_rerank_top1")),
                "decision_reason": raw.get("decision_reason"),
                "resolved_fragment_count": raw.get("resolved_fragment_count"),
                "matched_candidate_count": raw.get("matched_candidate_count"),
            },
            warnings=warnings,
        )


def _normalize_rerank(value: Any) -> tuple[list[JsonDict], list[str]]:
    items = as_list(value)
    warnings: list[str] = []
    if len(items) > _MAX_RERANK_ITEMS:
        warnings.append(f"rerank truncated to the first {_MAX_RERANK_ITEMS} of {len(items)} items")
        items = items[:_MAX_RERANK_ITEMS]

    normalized: list[JsonDict] = []
    for item in items:
        if not isinstance(item, Mapping):
            continue
        smiles = canonical_smiles(str(item.get("smiles") or ""))
        if not smiles:
            continue
        normalized.append({"rank": len(normalized) + 1, "smiles": smiles})
    if len(normalized) != len(items):
        warnings.append(f"dropped {len(items) - len(normalized)} rerank item(s) with a missing/blank smiles")
    return normalized, warnings


def _normalize_fragments(value: Any) -> tuple[list[JsonDict], list[str]]:
    items = as_list(value)
    warnings: list[str] = []
    if len(items) > _MAX_FRAGMENT_ITEMS:
        warnings.append(f"fragments truncated to the first {_MAX_FRAGMENT_ITEMS} of {len(items)} items")
        items = items[:_MAX_FRAGMENT_ITEMS]

    normalized: list[JsonDict] = []
    dropped = 0
    for item in items:
        if not isinstance(item, Mapping):
            dropped += 1
            continue
        fmt = str(item.get("format") or "").strip().lower()
        text_value = str(item.get("value") or "").strip()
        if fmt not in _FRAGMENT_FORMATS or not text_value:
            dropped += 1
            continue
        normalized.append({"format": fmt, "value": text_value[:4096]})
    if dropped:
        warnings.append(f"dropped {dropped} fragment(s) with an unresolved format or blank value")
    return normalized, warnings
