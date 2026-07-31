"""Common result types for Spectune reward functions."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

JsonDict = dict[str, Any]


@dataclass(slots=True)
class RewardResult:
    """A scalar reward together with inspectable component scores."""

    score: float
    components: dict[str, float] = field(default_factory=dict)
    details: JsonDict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> JsonDict:
        return asdict(self)


__all__ = ["RewardResult"]
