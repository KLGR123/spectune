"""Configuration for shared language-model clients."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class LlmConfig:
    """Connection settings for an OpenAI-compatible chat endpoint.

    ``base_url`` points at the ``/v1`` root (for example,
    ``http://127.0.0.1:8000/v1`` for a local vLLM server). When ``base_url`` or
    ``model`` is unset, the client reports itself unavailable.
    """

    base_url: str = field(default_factory=lambda: os.getenv("SPECTUNE_LLM_BASE_URL", ""))
    api_key: str = field(default_factory=lambda: os.getenv("SPECTUNE_LLM_API_KEY", ""))
    model: str = field(default_factory=lambda: os.getenv("SPECTUNE_LLM_MODEL", ""))
    timeout_s: float = 120.0
    max_concurrency: int = 8
    max_retries: int = 2
    temperature: float = 0.9
    top_p: float = 0.95
    max_tokens: int = 512

    @property
    def available(self) -> bool:
        return bool(self.base_url and self.model)

    def __post_init__(self) -> None:
        if self.max_concurrency < 1:
            raise ValueError("LlmConfig.max_concurrency must be at least 1")
        if self.max_retries < 0:
            raise ValueError("LlmConfig.max_retries must be non-negative")


@dataclass(frozen=True, slots=True)
class LitellmConfig:
    """Connection settings for a litellm-routed chat endpoint.

    Reads ``LITELLM_API_KEY``, ``LITELLM_API_BASE``, and ``LITELLM_MODEL``
    from the environment (typically sourced from ``secrets.env``).
    """

    api_key: str = field(default_factory=lambda: os.getenv("LITELLM_API_KEY", ""))
    api_base: str = field(default_factory=lambda: os.getenv("LITELLM_API_BASE", ""))
    model: str = field(default_factory=lambda: os.getenv("LITELLM_MODEL", ""))
    timeout_s: float = 120.0
    max_concurrency: int = 8
    max_retries: int = 2
    temperature: float = 0.9
    top_p: float = 0.95
    max_tokens: int = 512

    @property
    def available(self) -> bool:
        return bool(self.api_key and self.api_base and self.model)

    def __post_init__(self) -> None:
        if self.max_concurrency < 1:
            raise ValueError("LitellmConfig.max_concurrency must be at least 1")
        if self.max_retries < 0:
            raise ValueError("LitellmConfig.max_retries must be non-negative")


__all__ = ["LlmConfig", "LitellmConfig"]
