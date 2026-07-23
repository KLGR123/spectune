"""Common contracts shared by all Spectune tools."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

JsonDict = dict[str, Any]
ToolCompletion = Literal["success", "partial", "failure"]


@dataclass(slots=True)
class ToolResult:
    """Normalized result returned by every tool.

    ``completion`` describes whether the call completed successfully, while
    ``status`` is a tool-specific outcome such as ``ok`` or ``no_hits``.
    """

    completion: ToolCompletion
    status: str
    data: JsonDict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> JsonDict:
        return asdict(self)


class Tool(ABC):
    """An executable tool with an OpenAI-compatible function schema."""

    name: str
    description: str
    parameters: JsonDict

    @property
    def schema(self) -> JsonDict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
                "strict": False,
            },
        }

    @abstractmethod
    async def execute(self, arguments: Mapping[str, Any]) -> ToolResult:
        """Execute one validated tool call."""
