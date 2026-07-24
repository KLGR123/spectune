"""Tool registry and dispatch."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from typing import Any

from spectune.config import SpectuneConfig

from .askcos_reaction_forward_predict import AskcosReactionForwardPredictTool
from .base import JsonDict, Tool, ToolResult
from .code_interpreter import CodeInterpreterTool
from .crossref_search import CrossrefSearchTool
from .nmr_forward_predict import NmrForwardPredictTool
from .nmr_generate import NmrGenerateTool
from .nmr_repair import NmrRepairTool
from .nmr_rerank import NmrRerankTool
from .nmrexp_search import NmrExpSearchTool
from .reaction_local_index_search import ReactionLocalIndexSearchTool
from .semantic_scholar_search import SemanticScholarSearchTool
from .unimol3_reaction_forward_predict import Unimol3ReactionForwardPredictTool
from .web_search import WebSearchTool
from .wikipedia_search import WikipediaSearchTool


class ToolManager:
    """Own tool instances and expose a single schema/dispatch interface."""

    def __init__(self, tools: Iterable[Tool] = ()) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools:
            self.register(tool)

    @classmethod
    def from_config(cls, config: SpectuneConfig | None = None) -> ToolManager:
        config = config or SpectuneConfig()
        return cls(
            [
                WebSearchTool(config.web_search),
                CodeInterpreterTool(config.code_interpreter),
                NmrGenerateTool(config.nmr_generate),
                NmrRepairTool(config.nmr_repair),
                NmrRerankTool(config.nmr_rerank),
                NmrForwardPredictTool(config.nmr_forward_predict),
                NmrExpSearchTool(config.nmrexp_search),
                ReactionLocalIndexSearchTool(config.reaction_local_index_search),
                AskcosReactionForwardPredictTool(config.askcos_reaction_forward_predict),
                Unimol3ReactionForwardPredictTool(config.unimol3_reaction_forward_predict),
                SemanticScholarSearchTool(config.semantic_scholar_search),
                CrossrefSearchTool(config.crossref_search),
                WikipediaSearchTool(config.wikipedia_search),
            ]
        )

    def register(self, tool: Tool, *, replace: bool = False) -> None:
        if tool.name in self._tools and not replace:
            raise ValueError(f"tool already registered: {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise KeyError(f"unknown tool: {name}") from exc

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._tools)

    @property
    def schemas(self) -> list[JsonDict]:
        return [tool.schema for tool in self._tools.values()]

    async def invoke(self, name: str, arguments: Mapping[str, Any] | str | None = None) -> ToolResult:
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(completion="failure", status="error", warnings=[f"unknown tool: {name}"])

        if isinstance(arguments, str):
            try:
                parsed = json.loads(arguments)
            except json.JSONDecodeError as exc:
                return ToolResult(
                    completion="failure",
                    status="error",
                    warnings=[f"invalid JSON arguments: {exc}"],
                )
            if not isinstance(parsed, dict):
                return ToolResult(
                    completion="failure",
                    status="error",
                    warnings=["tool arguments must be a JSON object"],
                )
            arguments = parsed

        try:
            return await tool.execute(arguments or {})
        except Exception as exc:
            return ToolResult(
                completion="failure",
                status="error",
                warnings=[f"{type(exc).__name__}: {exc}"],
            )
