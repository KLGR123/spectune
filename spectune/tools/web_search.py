"""Web search tool backed by the Volcengine search API."""

from __future__ import annotations

import asyncio
import json
import urllib.error
import urllib.request
from collections.abc import Iterable, Mapping
from typing import Any

from .base import JsonDict, Tool, ToolResult
from .config import WebSearchConfig
from .help import HelpTool


class WebSearchTool(Tool):
    name = "web_search"
    description = (
        "搜索参考信息，如分子名称、SMILES、SMARTS、性质，相关文献。"
    )

    def __init__(self, config: WebSearchConfig | None = None) -> None:
        self.config = config or WebSearchConfig()

    @property
    def parameters(self) -> JsonDict:
        # Schema defaults mirror ``self.config`` so callers see the actual
        # configured behavior instead of literals baked into the class body.
        config = self.config
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "查询内容"},
                "search_type": {"type": "string", "default": config.default_search_type},
                "count": {"type": "integer", "minimum": 1, "default": config.default_count},
                "need_content": {"type": "boolean", "default": config.default_need_content},
                "need_url": {"type": "boolean", "default": config.default_need_url},
                "need_summary": {"type": "boolean", "default": config.default_need_summary},
                "sites": {"type": "string", "default": config.default_sites},
                "block_hosts": {"type": "string", "default": config.default_block_hosts},
                "auth_info_level": {"type": "integer", "minimum": 0, "default": config.default_auth_info_level},
                "time_range": {"type": "string", "default": config.default_time_range},
                "query_rewrite": {"type": "boolean", "default": config.default_query_rewrite},
            },
            "required": ["query"],
            "additionalProperties": False,
        }

    async def execute(self, arguments: Mapping[str, Any]) -> ToolResult:
        query = str(arguments.get("query") or "").strip()
        if not query:
            return ToolResult(completion="failure", status="error", warnings=["query is required"])
        if not self.config.api_key:
            return ToolResult(
                completion="failure",
                status="unavailable",
                data={"query": query, "source": "volcengine_websearch"},
                warnings=["VOLCENGINE_WEBSEARCH_API_KEY is not configured"],
            )

        body = self._request_body(query, arguments)
        try:
            raw_events = await asyncio.to_thread(self._request, body)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            return ToolResult(
                completion="failure",
                status="error",
                data={"query": query},
                warnings=[f"HTTP {exc.code}: {detail}"],
            )
        except Exception as exc:
            return ToolResult(
                completion="failure",
                status="error",
                data={"query": query},
                warnings=[f"{type(exc).__name__}: {exc}"],
            )

        hits = _parse_hits(raw_events)
        return ToolResult(
            completion="success",
            status="ok" if hits or raw_events else "no_hits",
            data={
                "query": query,
                "source": "volcengine_websearch",
                "raw_events": raw_events,
                "parsed_hits": hits,
            },
        )

    def _request_body(self, query: str, arguments: Mapping[str, Any]) -> JsonDict:
        config = self.config
        count = max(1, int(arguments.get("count") or config.default_count))
        return {
            "Query": query,
            "SearchType": str(arguments.get("search_type") or config.default_search_type),
            "Count": count,
            "Filter": {
                "NeedContent": bool(arguments.get("need_content", config.default_need_content)),
                "NeedUrl": bool(arguments.get("need_url", config.default_need_url)),
                "Sites": str(arguments.get("sites") or config.default_sites),
                "BlockHosts": str(arguments.get("block_hosts") or config.default_block_hosts),
                "AuthInfoLevel": max(0, int(arguments.get("auth_info_level") or config.default_auth_info_level)),
            },
            "NeedSummary": bool(arguments.get("need_summary", config.default_need_summary)),
            "TimeRange": str(arguments.get("time_range") or config.default_time_range),
            "QueryControl": {"QueryRewrite": bool(arguments.get("query_rewrite", config.default_query_rewrite))},
        }

    def _request(self, body: JsonDict) -> list[str]:
        request = urllib.request.Request(
            self.config.api_url,
            data=json.dumps(body, ensure_ascii=False).encode(),
            headers={
                "Content-Type": "application/json",
                "Accept": "text/event-stream, application/json",
                "Authorization": f"Bearer {self.config.api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.config.timeout_s) as response:
            events: list[str] = []
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line or line.startswith(":"):
                    continue
                if line.startswith("data:"):
                    line = line[5:].strip()
                if line and line != "[DONE]":
                    events.append(line)
            return events


WEB_SEARCH_GUIDE = """# web_search 工具使用说明

调用外部通用网页搜索引擎，用于查找超出自身训练知识、或需要外部核实的参考信息。它不是化学结构数据库，也不是波谱数据库。

## 适用场景

- 具名反应的一般机理与典型条件查询，例如常用催化剂、碱、溶剂、温度等背景信息。这类查询命中率高，返回内容通常真实可用。例如 Suzuki coupling arylboronic acid Pd(PPh3)4 K2CO3 typical conditions temperature solvent 或 reductive amination aldehyde secondary amine NaBH(OAc)3 DCE conditions；
- 特定、已被文献报道的环化或官能团转化机理查询，用词需具体，包含反应物类别和转化类型。例如 2-amino-5-methylbenzenethiol and 4-bromophenyl isothiocyanate reaction mechanism；
- 常见化合物名称到基本背景资料的一般性查询（非核心结构确认用途）。

## 不适用场景

不要用于通过 NMR 化学位移数值反查结构、把反应物 SMILES 拼进查询文本，或直接搜索已拥有的 SMILES 本身。查询应尽量使用能被搜索引擎理解为自然语言的表达，例如反应类型、官能团描述、化合物通用名，而不是数值型或结构化标识符。
"""


class WebSearchGuideTool(HelpTool):
    name = "read_web_search_guide"
    description = (
        "查阅 web_search 工具的使用说明，哪些查询方式更有效。"
        "当不确定 web_search 能否帮上忙，或某次结果不对时可确认。无需参数。"
    )
    guide = WEB_SEARCH_GUIDE


def _walk_mappings(value: Any) -> Iterable[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        yield value
        for child in value.values():
            yield from _walk_mappings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_mappings(child)


def _parse_hits(events: list[str]) -> list[JsonDict]:
    hits: list[JsonDict] = []
    seen: set[tuple[str, str, str]] = set()
    for event in events:
        try:
            value = json.loads(event)
        except json.JSONDecodeError:
            continue
        for item in _walk_mappings(value):
            hit_keys = ("title", "Title", "url", "Url", "link", "summary", "snippet", "content")
            if not any(key in item for key in hit_keys):
                continue
            hit = {
                "title": str(item.get("title") or item.get("Title") or ""),
                "url": str(item.get("url") or item.get("Url") or item.get("link") or ""),
                "snippet": str(item.get("snippet") or item.get("summary") or item.get("Summary") or ""),
                "content": str(item.get("content") or item.get("Content") or ""),
                "rank": len(hits) + 1,
            }
            key = (hit["title"], hit["url"], hit["snippet"] or hit["content"])
            if key not in seen:
                seen.add(key)
                hits.append(hit)
    return hits
