"""Framework-agnostic RL tool catalog and schema helpers.

These utilities belong to Spectune's core surface (artifact compile, reward
judging). Backend adapters (e.g. ``spectune.tools.verl``) consume them; they
must not import torch / ray / verl.
"""

from __future__ import annotations

import copy
from collections.abc import Sequence
from functools import lru_cache
from typing import Any

from spectune.tools.manager import ToolManager

JsonDict = dict[str, Any]

# Curated default set for structure-elucidation RL. Override via CLI / YAML.
DEFAULT_RL_TOOL_NAMES: tuple[str, ...] = (
    "nmr_generate",
    "nmr_repair",
    "nmr_forward_predict",
    "nmr_rerank",
    # "reaction_local_index_search",
    "code_interpreter",
    "web_search",
)


@lru_cache(maxsize=1)
def _shared_manager() -> ToolManager:
    return ToolManager.from_config()


def reset_shared_manager() -> None:
    """Drop the cached manager (mainly for tests)."""
    _shared_manager.cache_clear()


def shared_manager() -> ToolManager:
    """Return the process-wide :class:`ToolManager` used by RL adapters."""
    return _shared_manager()


def resolve_tool_names(
    tool_names: Sequence[str] | None = None,
    *,
    manager: ToolManager | None = None,
) -> tuple[str, ...]:
    """Validate and return tool names that should be exposed to trainers."""
    tool_manager = manager or _shared_manager()
    names = tuple(tool_names) if tool_names is not None else DEFAULT_RL_TOOL_NAMES
    unknown = [name for name in names if name not in tool_manager.names]
    if unknown:
        raise KeyError(f"unknown Spectune tools: {unknown}; known={list(tool_manager.names)}")
    return names


def openai_schema_for(tool_name: str, manager: ToolManager | None = None) -> JsonDict:
    """Return a deep-copied OpenAI function schema for one Spectune tool."""
    tool_manager = manager or _shared_manager()
    return copy.deepcopy(tool_manager.get(tool_name).schema)


def schemas_for_names(
    tool_names: Sequence[str],
    *,
    manager: ToolManager | None = None,
) -> list[JsonDict]:
    """Return full OpenAI schemas for the given tool names (order preserved)."""
    names = resolve_tool_names(tool_names, manager=manager)
    tool_manager = manager or _shared_manager()
    return [openai_schema_for(name, tool_manager) for name in names]


__all__ = [
    "DEFAULT_RL_TOOL_NAMES",
    "openai_schema_for",
    "reset_shared_manager",
    "resolve_tool_names",
    "schemas_for_names",
    "shared_manager",
]
