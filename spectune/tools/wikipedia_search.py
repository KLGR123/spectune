"""Free-text Wikipedia search, with optional per-hit summaries."""

from __future__ import annotations

import asyncio
import json
import re
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping
from typing import Any

from spectune.config import WikipediaSearchConfig

from .base import JsonDict, Tool, ToolResult

_TAG_RE = re.compile(r"<[^>]+>")


class WikipediaSearchTool(Tool):
    name = "wikipedia_search"
    description = (
        "Search Wikipedia (free text) for encyclopedic reference on compounds, reactions, "
        "or concepts, returning page titles, snippets, and optional per-hit summaries. "
        "Backed by the live Wikipedia API (may be slow); a generally reliable reference, "
        "though content can be outdated."
    )

    def __init__(self, config: WikipediaSearchConfig | None = None) -> None:
        self.config = config or WikipediaSearchConfig()

    @property
    def parameters(self) -> JsonDict:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Free-text search query."},
                "max_results": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 50,
                    "default": self.config.default_max_results,
                },
                "fetch_summaries": {"type": "boolean", "default": self.config.fetch_summaries},
                "language": {
                    "type": "string",
                    "default": self.config.language,
                    "description": "Wikipedia language code.",
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        }

    async def execute(self, arguments: Mapping[str, Any]) -> ToolResult:
        query = str(arguments.get("query") or "").strip()
        if not query:
            return ToolResult(completion="failure", status="error", warnings=["query is required"])

        max_results = max(1, min(int(arguments.get("max_results") or self.config.default_max_results), 50))
        fetch_summaries = bool(arguments.get("fetch_summaries", self.config.fetch_summaries))
        language = str(arguments.get("language") or self.config.language).strip() or "en"

        try:
            hits = await asyncio.to_thread(self._search_and_summarize, query, max_results, fetch_summaries, language)
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

        return ToolResult(
            completion="success",
            status="ok" if hits else "no_hits",
            data={"query": query, "source": "wikipedia_search", "language": language, "hits": hits},
        )

    def _search_and_summarize(
        self, query: str, max_results: int, fetch_summaries: bool, language: str
    ) -> list[JsonDict]:
        hits = self._search(query, max_results, language)
        if fetch_summaries:
            for hit in hits:
                summary = self._summary(hit["title"], language)
                if summary:
                    hit["summary"] = summary.get("summary", "")
                    hit["url"] = summary.get("url", hit.get("url", ""))
        return hits

    def _search(self, query: str, max_results: int, language: str) -> list[JsonDict]:
        params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": str(max_results),
            "format": "json",
        }
        base = self.config.search_base_url or f"https://{language}.wikipedia.org/w/api.php"
        url = f"{base}?{urllib.parse.urlencode(params)}"
        request = urllib.request.Request(url, headers={"Accept": "application/json"}, method="GET")
        with urllib.request.urlopen(request, timeout=self.config.timeout_s) as response:
            payload = json.loads(response.read().decode("utf-8"))

        results = ((payload.get("query") or {}).get("search")) or []
        hits: list[JsonDict] = []
        for item in results:
            if not isinstance(item, dict):
                continue
            title = str(item.get("title") or "")
            if not title:
                continue
            hits.append(
                {
                    "rank": len(hits) + 1,
                    "title": title,
                    "pageid": item.get("pageid"),
                    "snippet": _TAG_RE.sub("", str(item.get("snippet") or "")),
                    "url": f"https://{language}.wikipedia.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}",
                }
            )
        return hits

    def _summary(self, title: str, language: str) -> JsonDict | None:
        encoded = urllib.parse.quote(title.replace(" ", "_"))
        base = self.config.summary_base_url or f"https://{language}.wikipedia.org/api/rest_v1/page/summary"
        url = f"{base.rstrip('/')}/{encoded}"
        request = urllib.request.Request(url, headers={"Accept": "application/json"}, method="GET")
        try:
            with urllib.request.urlopen(request, timeout=self.config.timeout_s) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            # A missing/slow summary for one hit should not fail the whole search.
            return None
        extract = str(payload.get("extract") or "").strip()
        if not extract:
            return None
        page_url = ((payload.get("content_urls") or {}).get("desktop") or {}).get("page", "")
        return {"summary": extract, "url": page_url}
