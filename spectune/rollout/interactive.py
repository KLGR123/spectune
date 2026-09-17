"""Streaming agent loop for a live, single-query interactive session.

:class:`InteractiveAgent` reuses all of :class:`~spectune.rollout.rollout.Rollout`'s
setup (tool manager, system prompt, LLM client) and its ``_run_agent_loop``
generate -> tool-call -> tool-response cycle; the only difference is that the
loop body is reshaped into an async generator that yields one event per
LLM turn / tool call / tool result, so a caller (e.g. ``spectune.rollout.webui``)
can push each step to a browser as it happens instead of waiting for the whole
trajectory to finish.
"""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import AsyncIterator
from typing import Any

from spectune.format.v1 import (
    TOOL_RESPONSE_END,
    TOOL_RESPONSE_START,
    extract_smiles_candidates,
    extract_tool_calls,
)
from spectune.tools import compact_tool_payload

from .rollout import Rollout

JsonDict = dict[str, Any]

# Matches the same taxonomy examples/debug/eval_rollouts.py uses to render
# offline trajectories, so the live viewer and the offline viewer look alike.
_SEGMENT_RE = re.compile(r"(<think>.*?</think>)|(<tool_call>.*?</tool_call>)", re.DOTALL)


def split_response_segments(text: str) -> list[JsonDict]:
    """Split one raw assistant turn into ordered ``{"type", "content"}`` segments.

    ``type`` is one of ``think`` / ``tool_call`` / ``text``. ``tool_call``
    segments are the raw hermes block, pretty-printed when it parses as JSON.
    """
    segments: list[JsonDict] = []
    last = 0
    for m in _SEGMENT_RE.finditer(text):
        head = text[last : m.start()].strip()
        if head:
            segments.append({"type": "text", "content": head})

        full = m.group(0)
        if full.startswith("<think>"):
            inner = full[len("<think>") : -len("</think>")].strip()
            segments.append({"type": "think", "content": inner})
        else:
            inner = full[len("<tool_call>") : -len("</tool_call>")].strip()
            try:
                pretty = json.dumps(json.loads(inner), ensure_ascii=False, indent=2)
            except json.JSONDecodeError:
                pretty = inner
            segments.append({"type": "tool_call", "content": pretty})
        last = m.end()

    tail = text[last:].strip()
    if tail:
        segments.append({"type": "text", "content": tail})
    return segments


class InteractiveAgent(Rollout):
    """:class:`Rollout` variant that streams events instead of returning a record.

    Construction is identical to ``Rollout`` (same config/llm_config/llm
    arguments); only :meth:`stream_query` is added.
    """

    async def stream_query(
        self, user_query: str, messages: list[JsonDict] | None = None
    ) -> AsyncIterator[JsonDict]:
        """Run one user turn of the tool-agent loop, yielding events as they occur.

        ``messages`` lets a caller continue a previous conversation (a list
        already containing the system message and any prior turns, mutated in
        place); when omitted a fresh conversation is started with
        ``self.system_prompt``.

        Yields dicts of the form ``{"type": ..., ...}`` where ``type`` is one
        of ``user``, ``think``, ``tool_call``, ``tool_response``, ``text``,
        ``error``, or ``done`` (the last event, carrying the final ``smiles``
        candidates and the full ``messages`` list).
        """
        if messages is None:
            messages = [{"role": "system", "content": self.system_prompt}]
        messages.append({"role": "user", "content": user_query})
        yield {"type": "user", "content": user_query}

        complete_fn = getattr(self.llm, "complete_messages_with_reasoning", None)
        final_response = ""
        for _turn_idx in range(self.config.max_assistant_turns):
            if complete_fn is not None:
                response, reasoning = await complete_fn(messages)
            else:
                response, reasoning = await self.llm.complete_messages(messages), ""
            if not response:
                yield {"type": "error", "content": getattr(self.llm, "last_error", None) or "LLM call failed"}
                return
            messages.append({"role": "assistant", "content": response})
            final_response = response

            if reasoning:
                yield {"type": "think", "content": reasoning}
            for segment in split_response_segments(response):
                yield segment

            tool_calls = extract_tool_calls(response)
            if not tool_calls:
                break

            results = await asyncio.gather(
                *(self.tool_manager.invoke(call["name"], call["arguments"]) for call in tool_calls)
            )
            for call, result in zip(tool_calls, results, strict=True):
                payload = compact_tool_payload(result)
                messages.append(
                    {
                        "role": "user",
                        "content": f"{TOOL_RESPONSE_START}\n{json.dumps(payload, ensure_ascii=False)}\n{TOOL_RESPONSE_END}",
                    }
                )
                yield {"type": "tool_response", "name": call["name"], "content": payload}

        yield {"type": "done", "smiles": extract_smiles_candidates(final_response), "messages": messages}


__all__ = ["InteractiveAgent", "split_response_segments"]
