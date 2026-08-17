"""Async litellm client with the same interface as :class:`LlmClient`."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

import litellm

from .config import LitellmConfig

litellm.drop_params = True

JsonDict = dict[str, Any]


class LitellmClient:
    """Async, concurrency-limited litellm client with retries and failure counters."""

    def __init__(self, config: LitellmConfig | None = None) -> None:
        self.config = config or LitellmConfig()
        self.stats: JsonDict = {"requests": 0, "failures": 0, "retries": 0}
        self._semaphore: asyncio.Semaphore | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self.last_error: str | None = None

    @property
    def available(self) -> bool:
        return self.config.available

    async def complete(self, system_prompt: str, user_prompt: str) -> str:
        """Return the assistant message, or ``""`` if the call could not be made."""
        return await self.complete_messages(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )

    async def complete_messages(self, messages: list[JsonDict]) -> str:
        """Return the assistant message for a full message list, or ``""`` on failure."""
        if not self.available:
            return ""
        async with self._limiter():
            self.stats["requests"] += 1
            for attempt in range(self.config.max_retries + 1):
                try:
                    response = await asyncio.to_thread(self._call, messages)
                    return str(response.choices[0].message.content or "").strip()
                except Exception as exc:
                    self.last_error = f"{type(exc).__name__}: {exc}"
                    if attempt >= self.config.max_retries:
                        self.stats["failures"] += 1
                        return ""
                    self.stats["retries"] += 1
                    await asyncio.sleep(1.5 * (attempt + 1))
        return ""

    async def complete_many(self, prompts: Sequence[tuple[str, str]]) -> list[str]:
        """Run many ``(system, user)`` prompts concurrently, preserving order."""
        if not prompts:
            return []
        return list(await asyncio.gather(*(self.complete(system, user) for system, user in prompts)))

    def _limiter(self) -> asyncio.Semaphore:
        loop = asyncio.get_running_loop()
        if self._semaphore is None or self._loop is not loop:
            self._semaphore = asyncio.Semaphore(self.config.max_concurrency)
            self._loop = loop
        return self._semaphore

    def _call(self, messages: list[JsonDict]) -> Any:
        return litellm.completion(
            model=self.config.model,
            api_key=self.config.api_key,
            api_base=self.config.api_base.rstrip("/"),
            messages=messages,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            max_tokens=self.config.max_tokens,
            timeout=self.config.timeout_s,
        )


__all__ = ["LitellmClient"]
