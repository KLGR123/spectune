"""ScriptedFollowupInteraction: deliver pre-scripted user turns one at a time.

Registered as ``scripted_followup`` in the interaction config that
``python -m spectune.artifacts compile --interaction-config <path>`` writes.

Verl's ToolAgentLoop enters INTERACTING state after each tool-call-free
assistant turn.  The queue is consumed one entry per call:

  followup_turns=[]          → call 1: terminate=True   (single-turn sample)
  followup_turns=["u2"]      → call 1: (False, "u2")
                               call 2: terminate=True
  followup_turns=["u2","u3"] → call 1: (False, "u2")
                               call 2: (False, "u3")
                               call 3: terminate=True
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from uuid import uuid4

if TYPE_CHECKING:
    pass


class ScriptedFollowupInteraction:  # implements verl BaseInteraction protocol
    """Injects pre-scripted user turns in order, then terminates."""

    def __init__(self, config: dict[str, Any]) -> None:
        self._state: dict[str, list[str]] = {}

    async def start_interaction(
        self,
        instance_id: str | None = None,
        followup_turns: list[str] | None = None,
        **kwargs: Any,
    ) -> str:
        if instance_id is None:
            instance_id = str(uuid4())
        self._state[instance_id] = [t for t in (followup_turns or []) if t]
        return instance_id

    async def generate_response(
        self,
        instance_id: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> tuple[bool, str, float, dict[str, Any]]:
        queue = self._state.get(instance_id)
        if queue:
            return False, queue.pop(0), 0.0, {}
        return True, "", 0.0, {}

    async def finalize_interaction(self, instance_id: str, **kwargs: Any) -> None:
        self._state.pop(instance_id, None)
