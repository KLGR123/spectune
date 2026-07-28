"""Literature search via the Semantic Scholar Graph API."""

from __future__ import annotations

import asyncio
import json
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping
from typing import Any

from .base import JsonDict, Tool, ToolResult
from .config import SemanticScholarSearchConfig


class SemanticScholarSearchTool(Tool):
    name = "semantic_scholar_search"
    description = (
        "Search academic papers via the external Semantic Scholar API, returning metadata "
        "(title, year, venue, authors, abstract); no full text. The remote call may be slow; "
        "a reliable metadata reference, though coverage can be incomplete."
    )

    def __init__(self, config: SemanticScholarSearchConfig | None = None) -> None:
        self.config = config or SemanticScholarSearchConfig()

    @property
    def parameters(self) -> JsonDict:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query."},
                "max_results": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 100,
                    "default": self.config.default_max_results,
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        }

    async def execute(self, arguments: Mapping[str, Any]) -> ToolResult:
        query = str(arguments.get("query") or "").strip()
        if not query:
            return ToolResult(completion="failure", status="error", warnings=["query is required"])

        max_results = max(1, min(int(arguments.get("max_results") or self.config.default_max_results), 100))
        try:
            payload = await asyncio.to_thread(self._request, query, max_results)
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

        hits = _parse_hits(payload)
        return ToolResult(
            completion="success",
            status="ok" if hits else "no_hits",
            data={"query": query, "source": "semantic_scholar", "hits": hits},
        )

    def _request(self, query: str, max_results: int) -> JsonDict:
        fields = "title,year,authors,venue,url,abstract"
        url = f"{self.config.api_url}?query={urllib.parse.quote_plus(query)}&limit={max_results}&fields={fields}"
        headers = {"Accept": "application/json"}
        if self.config.api_key:
            headers["x-api-key"] = self.config.api_key
        request = urllib.request.Request(url, headers=headers, method="GET")
        with urllib.request.urlopen(request, timeout=self.config.timeout_s) as response:
            return json.loads(response.read().decode("utf-8"))


def _parse_hits(payload: JsonDict) -> list[JsonDict]:
    hits: list[JsonDict] = []
    for item in payload.get("data") or []:
        if not isinstance(item, dict):
            continue
        authors = item.get("authors") or []
        hits.append(
            {
                "rank": len(hits) + 1,
                "title": str(item.get("title") or ""),
                "year": item.get("year"),
                "venue": str(item.get("venue") or ""),
                "url": str(item.get("url") or ""),
                "authors": [str(author.get("name") or "") for author in authors[:5] if isinstance(author, dict)],
                "abstract": str(item.get("abstract") or "")[:700],
            }
        )
    return hits
