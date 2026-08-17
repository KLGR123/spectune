"""Backend selection: build the configured chat-completion client.

Set ``SPECTUNE_LLM_BACKEND`` in the environment (typically via
``secrets.env``) to switch between backends without touching call sites:

- ``"http"`` (default) -- :class:`~spectune.llm.llm.LlmClient` against
  ``SPECTUNE_LLM_BASE_URL`` / ``SPECTUNE_LLM_MODEL``.
- ``"litellm"`` -- :class:`~spectune.llm.litellm.LitellmClient` against
  ``LITELLM_API_BASE`` / ``LITELLM_MODEL``.

:class:`~spectune.rollout.rollout.Rollout` and
:class:`~spectune.augmentor.augmentor.Augmentor` both accept a pre-built
client via constructor injection, so ``create_llm_client()`` is the one place
a user needs to touch (an env var, or the ``backend=`` argument) to move the
whole pipeline onto a different provider.
"""

from __future__ import annotations

import dataclasses
import os

from .base import LlmClientProtocol
from .config import LitellmConfig, LlmConfig
from .litellm import LitellmClient
from .llm import LlmClient

BACKENDS = ("http", "litellm")


def create_llm_client(backend: str | None = None, **overrides: object) -> LlmClientProtocol:
    """Build the client for ``backend`` (or ``SPECTUNE_LLM_BACKEND``, default ``"http"``).

    ``overrides`` are applied to the backend's config dataclass via
    :func:`dataclasses.replace` -- e.g. ``temperature=0.2, max_concurrency=4``.
    Passing a field the selected config doesn't have raises ``TypeError``,
    same as constructing the dataclass directly.
    """
    resolved = (backend or os.getenv("SPECTUNE_LLM_BACKEND") or "http").strip().lower()
    if resolved == "litellm":
        config = dataclasses.replace(LitellmConfig(), **overrides) if overrides else LitellmConfig()
        return LitellmClient(config)
    if resolved == "http":
        config = dataclasses.replace(LlmConfig(), **overrides) if overrides else LlmConfig()
        return LlmClient(config)
    raise ValueError(f"unknown SPECTUNE_LLM_BACKEND {resolved!r}; expected one of {BACKENDS}")


__all__ = ["BACKENDS", "create_llm_client"]
