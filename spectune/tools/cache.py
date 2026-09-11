"""Disk-backed per-tool result cache with LRU eviction."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .base import ToolResult

if TYPE_CHECKING:
    from .manager import ToolManager


@dataclass(frozen=True, slots=True)
class ToolCacheConfig:
    """Cache location and capacity settings, read from environment variables."""

    cache_dir: str = field(
        default_factory=lambda: os.getenv(
            "SPECTUNE_TOOL_CACHE_DIR",
            str(Path(__file__).parents[3] / "cache"),
        )
    )
    max_size_per_tool: int = field(
        default_factory=lambda: int(os.getenv("SPECTUNE_TOOL_CACHE_MAX_SIZE", "50000"))
    )


class ToolCache:
    """Disk-backed, per-tool LRU cache.

    Layout: ``{cache_dir}/{tool_name}/{sha256_of_args}.json``

    Each entry file contains ``{"args": {...}, "result": {...}}``.
    LRU ordering is approximated via file mtime: every hit calls ``touch()``,
    and ``_evict`` removes the oldest files once the per-tool limit is exceeded.

    Only non-failure results (completion == "success" or "partial") are stored.
    """

    def __init__(self, config: ToolCacheConfig | None = None) -> None:
        self._cfg = config or ToolCacheConfig()

    def _tool_dir(self, tool_name: str) -> Path:
        return Path(self._cfg.cache_dir) / tool_name

    @staticmethod
    def _cache_key(arguments: dict[str, Any]) -> str:
        canonical = json.dumps(arguments, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(canonical.encode()).hexdigest()

    def lookup(self, tool_name: str, arguments: dict[str, Any]) -> ToolResult | None:
        """Return the cached ToolResult for (tool_name, arguments), or None on miss."""
        path = self._tool_dir(tool_name) / f"{self._cache_key(arguments)}.json"
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            path.touch()  # bump mtime so this entry appears recent for eviction
            return ToolResult(**data["result"])
        except Exception:
            return None

    def store(self, tool_name: str, arguments: dict[str, Any], result: ToolResult) -> None:
        """Persist result to disk; no-op for failure results."""
        if result.completion == "failure":
            return
        tool_dir = self._tool_dir(tool_name)
        try:
            tool_dir.mkdir(parents=True, exist_ok=True)
            path = tool_dir / f"{self._cache_key(arguments)}.json"
            payload = {"args": arguments, "result": result.to_dict()}
            path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            self._evict(tool_dir)
        except OSError:
            # Cache is best-effort: a failed write must never break the tool call.
            return

    def _evict(self, tool_dir: Path) -> None:
        entries = list(tool_dir.glob("*.json"))
        excess = len(entries) - self._cfg.max_size_per_tool
        if excess <= 0:
            return

        def _mtime(p: Path) -> float:
            try:
                return p.stat().st_mtime
            except OSError:
                return 0.0

        entries.sort(key=_mtime)
        for old in entries[:excess]:
            old.unlink(missing_ok=True)


class CachedToolManager:
    """Wraps a ToolManager with transparent disk caching on every invoke() call.

    Usage::

        manager = ToolManager.from_config()
        cached = CachedToolManager(manager)
        result = await cached.invoke("nmr_forward_predict", {"smiles": "CCO"})

    The first call hits the real tool and writes the result to disk.  Subsequent
    identical calls return the cached ToolResult without making any network request.
    """

    def __init__(self, manager: ToolManager, cache: ToolCache | None = None) -> None:
        self._manager = manager
        self._cache = cache or ToolCache()

    # --- proxy read-only attributes so callers can treat this like ToolManager ---

    @property
    def names(self) -> tuple[str, ...]:
        return self._manager.names

    @property
    def schemas(self) -> list[dict[str, Any]]:
        return self._manager.schemas

    def get(self, name: str):  # noqa: ANN201
        return self._manager.get(name)

    def register(self, tool: Any, *, replace: bool = False) -> None:
        self._manager.register(tool, replace=replace)

    # --- main entry point ---

    async def invoke(self, name: str, arguments: Mapping[str, Any] | str | None = None) -> ToolResult:
        """Invoke a tool, serving from cache when available.

        The cache key is (tool_name, canonicalized_arguments_dict).  Only
        successful and partial results are cached; failures are passed through
        without being stored so transient errors never poison the cache.
        """
        args_dict = self._normalize_args(arguments)

        hit = await asyncio.to_thread(self._cache.lookup, name, args_dict)
        if hit is not None:
            return hit

        result = await self._manager.invoke(name, arguments)
        try:
            await asyncio.to_thread(self._cache.store, name, args_dict, result)
        except OSError:
            # Cache write is best-effort; never let it surface as a tool failure.
            pass
        return result

    @staticmethod
    def _normalize_args(arguments: Mapping[str, Any] | str | None) -> dict[str, Any]:
        if arguments is None:
            return {}
        if isinstance(arguments, str):
            try:
                parsed = json.loads(arguments)
                return parsed if isinstance(parsed, dict) else {}
            except json.JSONDecodeError:
                return {}
        return dict(arguments)
