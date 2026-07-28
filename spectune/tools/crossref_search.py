"""Literature/DOI metadata search via the Crossref REST API."""

from __future__ import annotations

import asyncio
import json
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping
from typing import Any

from .base import JsonDict, Tool, ToolResult
from .config import CrossrefSearchConfig


class CrossrefSearchTool(Tool):
    name = "crossref_search"
    description = (
        "Look up scholarly work metadata (title, DOI, venue, year) via the external "
        "Crossref API; no full text. The remote call may be slow; a reliable metadata "
        "reference, though coverage can be incomplete."
    )

    def __init__(self, config: CrossrefSearchConfig | None = None) -> None:
        self.config = config or CrossrefSearchConfig()

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
            data={"query": query, "source": "crossref", "hits": hits},
        )

    def _request(self, query: str, max_results: int) -> JsonDict:
        params = {"query": query, "rows": str(max_results)}
        if self.config.mailto:
            params["mailto"] = self.config.mailto
        url = f"{self.config.api_url}?{urllib.parse.urlencode(params)}"
        headers = {"Accept": "application/json", "User-Agent": f"spectune/0.1 ({self.config.mailto or 'no-contact'})"}
        request = urllib.request.Request(url, headers=headers, method="GET")
        with urllib.request.urlopen(request, timeout=self.config.timeout_s) as response:
            return json.loads(response.read().decode("utf-8"))


def _parse_hits(payload: JsonDict) -> list[JsonDict]:
    hits: list[JsonDict] = []
    items = ((payload.get("message") or {}).get("items")) or []
    for item in items:
        if not isinstance(item, dict):
            continue
        year_parts = (item.get("issued") or {}).get("date-parts") or []
        year = year_parts[0][0] if year_parts and year_parts[0] else None
        hits.append(
            {
                "rank": len(hits) + 1,
                "title": " ".join(item.get("title") or []),
                "year": year,
                "publisher": str(item.get("publisher") or ""),
                "container_title": " ".join(item.get("container-title") or []),
                "doi": str(item.get("DOI") or ""),
                "url": str(item.get("URL") or ""),
            }
        )
    return hits
