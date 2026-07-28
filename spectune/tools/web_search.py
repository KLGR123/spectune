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


class WebSearchTool(Tool):
    name = "web_search"
    description = (
        "Search the open web for reference information, e.g. a molecule's names, "
        "SMILES/SMARTS, properties, or related literature. Hits come from an external "
        "search API (may be slow) and are a useful reference to cross-check; it does not "
        "infer structures or final answers."
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
                "query": {"type": "string", "description": "Search query."},
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
