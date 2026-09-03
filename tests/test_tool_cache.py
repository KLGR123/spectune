"""Tests for spectune.tools.cache (ToolCache and CachedToolManager)."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from spectune.tools.base import ToolResult
from spectune.tools.cache import CachedToolManager, ToolCache, ToolCacheConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _CountingTool:
    """Minimal tool stand-in that records how many times it has been called."""

    name = "counting_tool"

    def __init__(self, result: ToolResult) -> None:
        self._result = result
        self.calls: list[dict] = []

    async def execute(self, arguments: Mapping[str, Any]) -> ToolResult:
        self.calls.append(dict(arguments))
        return self._result


class _FakeManager:
    """Minimal ToolManager-alike backed by a single _CountingTool."""

    def __init__(self, tool: _CountingTool) -> None:
        self._tool = tool
        self.names = (tool.name,)
        self.schemas: list[dict] = []

    def get(self, name: str) -> _CountingTool:
        return self._tool

    def register(self, tool: Any, *, replace: bool = False) -> None:  # noqa: ARG002
        pass

    async def invoke(self, name: str, arguments: Mapping[str, Any] | str | None = None) -> ToolResult:
        args: dict = {}
        if isinstance(arguments, str):
            args = json.loads(arguments)
        elif arguments:
            args = dict(arguments)
        return await self._tool.execute(args)


def _make_cache(tmp_path: Path, max_size: int = 100) -> ToolCache:
    return ToolCache(ToolCacheConfig(cache_dir=str(tmp_path), max_size_per_tool=max_size))


# ---------------------------------------------------------------------------
# ToolCache unit tests
# ---------------------------------------------------------------------------


def test_lookup_miss_on_empty_cache(tmp_path: Path) -> None:
    cache = _make_cache(tmp_path)
    assert cache.lookup("my_tool", {"query": "hello"}) is None


def test_store_and_lookup_roundtrip(tmp_path: Path) -> None:
    cache = _make_cache(tmp_path)
    result = ToolResult(completion="success", status="ok", data={"answer": 42})

    cache.store("my_tool", {"query": "hello"}, result)
    hit = cache.lookup("my_tool", {"query": "hello"})

    assert hit is not None
    assert hit.completion == "success"
    assert hit.data["answer"] == 42


def test_cache_key_is_argument_order_insensitive(tmp_path: Path) -> None:
    cache = _make_cache(tmp_path)
    result = ToolResult(completion="success", status="ok", data={"x": 1})

    cache.store("t", {"b": 2, "a": 1}, result)
    hit = cache.lookup("t", {"a": 1, "b": 2})

    assert hit is not None


def test_failure_results_are_not_stored(tmp_path: Path) -> None:
    cache = _make_cache(tmp_path)
    failure = ToolResult(completion="failure", status="error")

    cache.store("t", {"q": "x"}, failure)
    assert cache.lookup("t", {"q": "x"}) is None


def test_partial_results_are_cached(tmp_path: Path) -> None:
    cache = _make_cache(tmp_path)
    partial = ToolResult(completion="partial", status="ok", data={"items": [1, 2]})

    cache.store("t", {"q": "x"}, partial)
    hit = cache.lookup("t", {"q": "x"})

    assert hit is not None
    assert hit.completion == "partial"


def test_eviction_keeps_most_recently_used(tmp_path: Path) -> None:
    cache = _make_cache(tmp_path, max_size=3)
    ok = ToolResult(completion="success", status="ok")

    for i in range(3):
        cache.store("t", {"i": i}, ok)

    # Access entry 0 to bump its mtime, making it the most recent.
    cache.lookup("t", {"i": 0})

    # Storing a 4th entry triggers eviction of the least recently used (entry 1).
    cache.store("t", {"i": 3}, ok)

    tool_dir = tmp_path / "t"
    assert len(list(tool_dir.glob("*.json"))) == 3
    assert cache.lookup("t", {"i": 0}) is not None  # was touched, survived
    assert cache.lookup("t", {"i": 3}) is not None  # just written, survived


def test_different_tools_use_separate_directories(tmp_path: Path) -> None:
    cache = _make_cache(tmp_path)
    result = ToolResult(completion="success", status="ok", data={"v": 1})

    cache.store("tool_a", {"q": "x"}, result)
    cache.store("tool_b", {"q": "x"}, ToolResult(completion="success", status="ok", data={"v": 2}))

    assert (tmp_path / "tool_a").is_dir()
    assert (tmp_path / "tool_b").is_dir()
    assert cache.lookup("tool_a", {"q": "x"}).data["v"] == 1
    assert cache.lookup("tool_b", {"q": "x"}).data["v"] == 2


def test_corrupted_entry_returns_none(tmp_path: Path) -> None:
    cache = _make_cache(tmp_path)
    (tmp_path / "t").mkdir()
    key = cache._cache_key({"q": "x"})
    (tmp_path / "t" / f"{key}.json").write_text("not valid json", encoding="utf-8")

    assert cache.lookup("t", {"q": "x"}) is None


# ---------------------------------------------------------------------------
# CachedToolManager integration tests
# ---------------------------------------------------------------------------


def test_cached_manager_calls_tool_on_first_invoke(tmp_path: Path) -> None:
    ok = ToolResult(completion="success", status="ok", data={"n": 1})
    tool = _CountingTool(ok)
    cache = _make_cache(tmp_path)
    manager = CachedToolManager(_FakeManager(tool), cache)

    result = asyncio.run(manager.invoke("counting_tool", {"q": "hello"}))

    assert result.data["n"] == 1
    assert len(tool.calls) == 1


def test_cached_manager_serves_second_call_from_cache(tmp_path: Path) -> None:
    ok = ToolResult(completion="success", status="ok", data={"n": 1})
    tool = _CountingTool(ok)
    cache = _make_cache(tmp_path)
    manager = CachedToolManager(_FakeManager(tool), cache)

    asyncio.run(manager.invoke("counting_tool", {"q": "hello"}))
    asyncio.run(manager.invoke("counting_tool", {"q": "hello"}))

    assert len(tool.calls) == 1, "second call should have been served from cache"


def test_cached_manager_does_not_cache_failures(tmp_path: Path) -> None:
    failure = ToolResult(completion="failure", status="error")
    tool = _CountingTool(failure)
    cache = _make_cache(tmp_path)
    manager = CachedToolManager(_FakeManager(tool), cache)

    asyncio.run(manager.invoke("counting_tool", {"q": "x"}))
    asyncio.run(manager.invoke("counting_tool", {"q": "x"}))

    assert len(tool.calls) == 2, "failure results must not be cached"


def test_cached_manager_accepts_json_string_arguments(tmp_path: Path) -> None:
    ok = ToolResult(completion="success", status="ok", data={"v": 7})
    tool = _CountingTool(ok)
    cache = _make_cache(tmp_path)
    manager = CachedToolManager(_FakeManager(tool), cache)

    asyncio.run(manager.invoke("counting_tool", '{"q": "hello"}'))
    result = asyncio.run(manager.invoke("counting_tool", '{"q": "hello"}'))

    assert result.data["v"] == 7
    assert len(tool.calls) == 1


def test_cached_manager_proxies_names_and_schemas(tmp_path: Path) -> None:
    tool = _CountingTool(ToolResult(completion="success", status="ok"))
    cache = _make_cache(tmp_path)
    manager = CachedToolManager(_FakeManager(tool), cache)

    assert "counting_tool" in manager.names
    assert manager.schemas == []
