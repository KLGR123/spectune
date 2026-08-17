"""Structural interface shared by all chat-completion clients.

:class:`~spectune.llm.llm.LlmClient` (plain HTTP) and
:class:`~spectune.llm.litellm.LitellmClient` (routed through litellm) already
expose the identical async surface below. Callers such as
:class:`~spectune.rollout.rollout.Rollout` and
:class:`~spectune.augmentor.augmentor.Augmentor` should depend on this
protocol rather than a concrete class, so either backend can be injected at
construction time without touching call sites.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

JsonDict = dict[str, Any]


@runtime_checkable
class LlmClientProtocol(Protocol):
    """Anything with this shape can stand in for :class:`LlmClient`."""

    stats: JsonDict
    last_error: str | None

    @property
    def available(self) -> bool: ...

    async def complete(self, system_prompt: str, user_prompt: str) -> str: ...

    async def complete_messages(self, messages: list[JsonDict]) -> str: ...

    async def complete_many(self, prompts: Sequence[tuple[str, str]]) -> list[str]: ...


__all__ = ["LlmClientProtocol"]
