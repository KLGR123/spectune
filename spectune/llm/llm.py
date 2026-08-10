"""Minimal async client for an OpenAI-compatible ``/chat/completions`` endpoint.

Written against :mod:`urllib` in worker threads so the package keeps its core
dependency list empty. One client supports both local model servers and hosted
APIs; they differ only in ``base_url``.

Failures are absorbed rather than raised so callers can decide how to degrade.
Every failure is counted in :attr:`LlmClient.stats`.
"""

from __future__ import annotations

import asyncio
import json
import urllib.error
import urllib.request
from collections.abc import Sequence
from typing import Any

from .config import LlmConfig

JsonDict = dict[str, Any]


class LlmClient:
    """Async, concurrency-limited chat client with retries and failure counters."""

    def __init__(self, config: LlmConfig | None = None) -> None:
        self.config = config or LlmConfig()
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
        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "max_tokens": self.config.max_tokens,
        }
        async with self._limiter():
            self.stats["requests"] += 1
            for attempt in range(self.config.max_retries + 1):
                try:
                    response = await asyncio.to_thread(self._post, payload)
                    return _first_message(response)
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
        # Semaphores bind to the loop that created them, and a client instance
        # may outlive one asyncio.run() call, so rebind when the loop changes.
        loop = asyncio.get_running_loop()
        if self._semaphore is None or self._loop is not loop:
            self._semaphore = asyncio.Semaphore(self.config.max_concurrency)
            self._loop = loop
        return self._semaphore

    def _post(self, payload: JsonDict) -> JsonDict:
        url = self.config.base_url.rstrip("/") + "/chat/completions"
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        request = urllib.request.Request(
            url,
            data=json.dumps(payload, ensure_ascii=False).encode(),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.config.timeout_s) as response:
                body = response.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code}: {detail[:300]}") from exc
        parsed = json.loads(body) if body else {}
        return parsed if isinstance(parsed, dict) else {"data": parsed}


def _first_message(response: JsonDict) -> str:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    if isinstance(message, dict):
        return str(message.get("content") or "").strip()
    return str(choices[0].get("text") or "").strip() if isinstance(choices[0], dict) else ""


__all__ = ["LlmClient"]
