import asyncio
import threading
import time
from unittest.mock import AsyncMock, MagicMock, patch

from spectune.llm import LitellmClient, LitellmConfig, LlmClient, LlmClientProtocol, LlmConfig, create_llm_client


def _client(**kwargs) -> LlmClient:
    return LlmClient(LlmConfig(base_url="http://127.0.0.1:9/v1", model="test-model", **kwargs))


def _litellm_client(**kwargs) -> LitellmClient:
    return LitellmClient(LitellmConfig(api_key="test-key", api_base="http://127.0.0.1:9", model="test/model", **kwargs))


class TestLlmClientRetries:
    def test_retries_transient_failures_then_succeeds(self, monkeypatch):
        monkeypatch.setattr(asyncio, "sleep", AsyncMock())
        client = _client(max_retries=2)
        calls = {"n": 0}

        def flaky(payload):
            calls["n"] += 1
            if calls["n"] < 3:
                raise RuntimeError("HTTP 500: boom")
            return {"choices": [{"message": {"content": "ok"}}]}

        with patch.object(client, "_post", side_effect=flaky):
            reply = asyncio.run(client.complete_messages([{"role": "user", "content": "hi"}]))

        assert reply == "ok"
        assert client.stats == {"requests": 1, "failures": 0, "retries": 2}

    def test_returns_empty_string_and_counts_failure_after_retries_exhausted(self, monkeypatch):
        monkeypatch.setattr(asyncio, "sleep", AsyncMock())
        client = _client(max_retries=1)

        with patch.object(client, "_post", side_effect=RuntimeError("boom")):
            reply = asyncio.run(client.complete_messages([{"role": "user", "content": "hi"}]))

        assert reply == ""
        assert client.stats == {"requests": 1, "failures": 1, "retries": 1}
        assert "boom" in client.last_error

    def test_no_retries_when_max_retries_is_zero(self, monkeypatch):
        monkeypatch.setattr(asyncio, "sleep", AsyncMock())
        client = _client(max_retries=0)

        with patch.object(client, "_post", side_effect=RuntimeError("boom")) as post:
            reply = asyncio.run(client.complete_messages([{"role": "user", "content": "hi"}]))

        assert reply == ""
        assert post.call_count == 1
        assert client.stats == {"requests": 1, "failures": 1, "retries": 0}


class TestLlmClientConcurrency:
    def test_complete_messages_respects_max_concurrency(self):
        client = _client(max_concurrency=2, max_retries=0)
        active = {"n": 0, "max": 0}
        lock = threading.Lock()

        def slow_post(payload):
            with lock:
                active["n"] += 1
                active["max"] = max(active["max"], active["n"])
            time.sleep(0.05)
            with lock:
                active["n"] -= 1
            return {"choices": [{"message": {"content": "ok"}}]}

        async def run():
            with patch.object(client, "_post", side_effect=slow_post):
                return await asyncio.gather(
                    *(client.complete_messages([{"role": "user", "content": "x"}]) for _ in range(6))
                )

        replies = asyncio.run(run())

        assert replies == ["ok"] * 6
        assert active["max"] <= 2


class TestLitellmClientRetries:
    def test_retries_transient_failures_then_succeeds(self, monkeypatch):
        monkeypatch.setattr(asyncio, "sleep", AsyncMock())
        client = _litellm_client(max_retries=2)
        calls = {"n": 0}

        def flaky(_messages, **_kwargs):
            calls["n"] += 1
            if calls["n"] < 3:
                raise RuntimeError("API error")
            mock_resp = MagicMock()
            mock_resp.choices = [MagicMock()]
            mock_resp.choices[0].message.content = "ok"
            return mock_resp

        with patch.object(client, "_call", side_effect=flaky):
            reply = asyncio.run(client.complete_messages([{"role": "user", "content": "hi"}]))

        assert reply == "ok"
        assert client.stats == {"requests": 1, "failures": 0, "retries": 2}

    def test_returns_empty_string_and_counts_failure_after_retries_exhausted(self, monkeypatch):
        monkeypatch.setattr(asyncio, "sleep", AsyncMock())
        client = _litellm_client(max_retries=1)

        with patch.object(client, "_call", side_effect=RuntimeError("boom")):
            reply = asyncio.run(client.complete_messages([{"role": "user", "content": "hi"}]))

        assert reply == ""
        assert client.stats == {"requests": 1, "failures": 1, "retries": 1}
        assert "boom" in client.last_error

    def test_no_retries_when_max_retries_is_zero(self, monkeypatch):
        monkeypatch.setattr(asyncio, "sleep", AsyncMock())
        client = _litellm_client(max_retries=0)

        with patch.object(client, "_call", side_effect=RuntimeError("boom")) as call_mock:
            reply = asyncio.run(client.complete_messages([{"role": "user", "content": "hi"}]))

        assert reply == ""
        assert call_mock.call_count == 1
        assert client.stats == {"requests": 1, "failures": 1, "retries": 0}


class TestLitellmClientConcurrency:
    def test_complete_messages_respects_max_concurrency(self):
        client = _litellm_client(max_concurrency=2, max_retries=0)
        active = {"n": 0, "max": 0}
        lock = threading.Lock()

        def slow_call(_messages, **_kwargs):
            with lock:
                active["n"] += 1
                active["max"] = max(active["max"], active["n"])
            time.sleep(0.05)
            with lock:
                active["n"] -= 1
            mock_resp = MagicMock()
            mock_resp.choices = [MagicMock()]
            mock_resp.choices[0].message.content = "ok"
            return mock_resp

        async def run():
            with patch.object(client, "_call", side_effect=slow_call):
                return await asyncio.gather(
                    *(client.complete_messages([{"role": "user", "content": "x"}]) for _ in range(6))
                )

        replies = asyncio.run(run())

        assert replies == ["ok"] * 6
        assert active["max"] <= 2


class TestProtocolConformance:
    def test_llm_client_implements_protocol(self):
        client = _client()
        assert isinstance(client, LlmClientProtocol)

    def test_litellm_client_implements_protocol(self):
        client = _litellm_client()
        assert isinstance(client, LlmClientProtocol)

    def test_protocol_has_required_attrs(self):
        client = _client()
        assert hasattr(client, "stats")
        assert hasattr(client, "last_error")
        assert hasattr(client, "available")
        assert hasattr(client, "complete")
        assert hasattr(client, "complete_messages")
        assert hasattr(client, "complete_many")


class TestCreateLlmClient:
    def test_backend_http_returns_llm_client(self):
        client = create_llm_client("http", base_url="http://test", model="test")
        assert isinstance(client, LlmClient)
        assert client.config.base_url == "http://test"
        assert client.config.model == "test"

    def test_backend_litellm_returns_litellm_client(self):
        client = create_llm_client("litellm", api_key="k", api_base="http://test", model="test")
        assert isinstance(client, LitellmClient)
        assert client.config.api_key == "k"
        assert client.config.api_base == "http://test"
        assert client.config.model == "test"

    def test_backend_defaults_to_http(self, monkeypatch):
        monkeypatch.delenv("SPECTUNE_LLM_BACKEND", raising=False)
        client = create_llm_client(base_url="http://test", model="test")
        assert isinstance(client, LlmClient)

    def test_backend_reads_from_env(self, monkeypatch):
        monkeypatch.setenv("SPECTUNE_LLM_BACKEND", "litellm")
        client = create_llm_client(api_key="k", api_base="http://test", model="test")
        assert isinstance(client, LitellmClient)

    def test_overrides_are_applied(self):
        client = create_llm_client("http", base_url="http://test", model="test", temperature=0.1, max_concurrency=4)
        assert client.config.temperature == 0.1
        assert client.config.max_concurrency == 4

    def test_backend_local_raises_with_helpful_message(self):
        try:
            create_llm_client("local")
            assert False, "should have raised"
        except ValueError as exc:
            assert "local" in str(exc).lower()
            assert "cli" in str(exc).lower() or "vllm" in str(exc).lower()

    def test_invalid_backend_raises(self):
        try:
            create_llm_client("unknown")
            assert False, "should have raised"
        except ValueError as exc:
            assert "unknown" in str(exc).lower()
            assert "http" in str(exc)
            assert "litellm" in str(exc)
