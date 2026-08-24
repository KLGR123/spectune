"""Async litellm client with the same interface as :class:`LlmClient`."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from .config import LitellmConfig

if TYPE_CHECKING:
    import litellm

JsonDict = dict[str, Any]


def _litellm() -> "litellm":
    """Import litellm lazily so modules that never call the LLM avoid its startup cost."""
    import litellm

    litellm.drop_params = True
    return litellm


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
            drop_top_p = False
            attempt = 0
            while True:
                try:
                    response = await asyncio.to_thread(self._call, messages, drop_top_p=drop_top_p)
                    return str(response.choices[0].message.content or "").strip()
                except Exception as exc:
                    self.last_error = f"{type(exc).__name__}: {exc}"
                    # Some Bedrock-hosted models (e.g. Claude) reject requests that
                    # set both `temperature` and `top_p`; drop `top_p` and retry
                    # right away, without counting against max_retries.
                    if not drop_top_p and "cannot both be specified" in str(exc):
                        drop_top_p = True
                        continue
                    if attempt >= self.config.max_retries:
                        self.stats["failures"] += 1
                        return ""
                    attempt += 1
                    self.stats["retries"] += 1
                    await asyncio.sleep(1.5 * attempt)

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

    def _call(self, messages: list[JsonDict], *, drop_top_p: bool = False) -> Any:
        kwargs: JsonDict = dict(
            model=self.config.model,
            api_key=self.config.api_key,
            api_base=self.config.api_base.rstrip("/"),
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            timeout=self.config.timeout_s,
        )
        if not drop_top_p:
            kwargs["top_p"] = self.config.top_p
        return _litellm().completion(**kwargs)


__all__ = ["LitellmClient"]
