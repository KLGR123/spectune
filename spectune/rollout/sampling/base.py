"""Abstract base class for rollout samplers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseSampler(ABC):
    """Decide whether a candidate LLM response should be accepted."""

    @abstractmethod
    def accept(self, response: str, sample: dict[str, Any]) -> bool:
        """Return ``True`` when ``response`` is acceptable for ``sample``."""
        ...


__all__ = ["BaseSampler"]
