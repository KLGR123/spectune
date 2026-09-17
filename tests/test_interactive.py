import asyncio

from spectune.rollout import InteractiveAgent, RolloutConfig, split_response_segments
from spectune.tools.base import ToolResult


class _FakeLlm:
    """Minimal stand-in exposing ``complete_messages_with_reasoning``."""

    def __init__(self, responses, reasonings=None):
        self._responses = list(responses)
        self._reasonings = list(reasonings) if reasonings is not None else [None] * len(responses)
        self.last_error = None
        self.stats = {"requests": 0, "failures": 0, "retries": 0}
        self.available = True

    async def complete_messages_with_reasoning(self, messages):
        return self._responses.pop(0), self._reasonings.pop(0)


def _agent(llm, **config_kwargs) -> InteractiveAgent:
    config = RolloutConfig(show_progress=False, **config_kwargs)
    return InteractiveAgent(config, llm=llm)


def _collect(agen):
    async def run():
        return [event async for event in agen]

    return asyncio.run(run())


class TestSplitResponseSegments:
    def test_plain_text_only(self):
        assert split_response_segments("just an answer") == [{"type": "text", "content": "just an answer"}]

    def test_think_block(self):
        segments = split_response_segments("<think>reasoning here</think>")
        assert segments == [{"type": "think", "content": "reasoning here"}]

    def test_tool_call_pretty_printed(self):
        segments = split_response_segments('<tool_call>\n{"name": "nmr_generate", "arguments": {}}\n</tool_call>')
        assert len(segments) == 1
        assert segments[0]["type"] == "tool_call"
        assert '"name": "nmr_generate"' in segments[0]["content"]

    def test_tool_call_invalid_json_falls_back_to_raw(self):
        segments = split_response_segments("<tool_call>not json</tool_call>")
        assert segments == [{"type": "tool_call", "content": "not json"}]

    def test_mixed_sequence_preserves_order(self):
        text = '<think>plan</think><tool_call>{"name": "x", "arguments": {}}</tool_call>trailing text'
        segments = split_response_segments(text)
        assert [s["type"] for s in segments] == ["think", "tool_call", "text"]
        assert segments[-1]["content"] == "trailing text"


class TestStreamQuery:
    def test_stops_when_no_tool_call(self):
        llm = _FakeLlm(['{"smiles": ["CCO"]}'])
        agent = _agent(llm, max_assistant_turns=5)

        events = _collect(agent.stream_query("hello"))

        types = [e["type"] for e in events]
        assert types == ["user", "text", "done"]
        assert events[-1]["smiles"] == ["CCO"]
        assert events[-1]["messages"][-1] == {"role": "assistant", "content": '{"smiles": ["CCO"]}'}

    def test_emits_reasoning_as_think_event(self):
        llm = _FakeLlm(['{"smiles": ["CCO"]}'], reasonings=["chain of thought"])
        agent = _agent(llm, max_assistant_turns=5)

        events = _collect(agent.stream_query("hello"))

        assert events[1] == {"type": "think", "content": "chain of thought"}

    def test_tool_call_then_final_answer(self):
        llm = _FakeLlm(
            [
                '<tool_call>\n{"name": "nmr_generate", "arguments": {}}\n</tool_call>',
                '{"smiles": ["CCO"]}',
            ]
        )
        agent = _agent(llm, max_assistant_turns=5)
        agent.tool_manager.invoke = _async_invoke(ToolResult(completion="ok", status="ok", data={"candidates": []}))

        events = _collect(agent.stream_query("hello"))

        types = [e["type"] for e in events]
        assert types == ["user", "tool_call", "tool_response", "text", "done"]
        assert events[2]["name"] == "nmr_generate"
        assert events[-1]["smiles"] == ["CCO"]

    def test_returns_error_event_when_llm_call_fails(self):
        llm = _FakeLlm([""])
        agent = _agent(llm)

        events = _collect(agent.stream_query("hello"))

        assert [e["type"] for e in events] == ["user", "error"]

    def test_continues_an_existing_conversation(self):
        llm = _FakeLlm(['{"smiles": ["CCN"]}'])
        agent = _agent(llm)
        prior_messages = [
            {"role": "system", "content": agent.system_prompt},
            {"role": "user", "content": "first question"},
            {"role": "assistant", "content": '{"smiles": ["CCO"]}'},
        ]

        events = _collect(agent.stream_query("follow up", messages=prior_messages))

        done = events[-1]
        assert done["messages"] is prior_messages
        assert done["messages"][-2] == {"role": "user", "content": "follow up"}


def _async_invoke(result):
    async def invoke(name, arguments):
        return result

    return invoke
