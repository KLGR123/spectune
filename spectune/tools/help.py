"""Base class for on-demand documentation ("skill") tools.

A HelpTool takes no meaningful arguments and performs no external action --
calling it just returns static guidance text for the model to read at its
own discretion. Subclassing HelpTool (instead of Tool directly) is both the
implementation shortcut (shared parameters/execute) and an isinstance()-
checkable marker for other code (e.g. catalog helpers, trajectory tooling)
that needs to tell "this schema entry is documentation, not an action."
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .base import JsonDict, Tool, ToolResult


class HelpTool(Tool):
    """A Tool whose only effect is returning static guidance text."""

    guide: str = ""

    parameters: JsonDict = {"type": "object", "properties": {}, "additionalProperties": False}

    async def execute(self, arguments: Mapping[str, Any]) -> ToolResult:
        return ToolResult(completion="success", status="ok", data={"guide": self.guide})
