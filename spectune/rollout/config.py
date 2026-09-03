"""Configuration for offline LLM rollout and rejection sampling."""

from __future__ import annotations

from dataclasses import dataclass, field

from spectune.format.v1 import SYSTEM_PROMPT
from spectune.tools.config import NMR_GENERATE_MAX_TOPK

DEFAULT_OUTPUT_DIR = "outputs/datasets/verl"

# Match verl's hermes agent loop defaults (actor_rollout_ref.rollout.multi_turn.*).
DEFAULT_MAX_ASSISTANT_TURNS = 16


@dataclass(frozen=True, slots=True)
class RolloutConfig:
    """Settings for parallel LLM sampling with optional rejection-based filtering.

    Each sample runs a tool-agent loop: the LLM generates a response, any
    hermes ``<tool_call>`` blocks are executed via :class:`ToolManager`, the
    results are fed back, and generation repeats until the model stops calling
    tools (final answer) or ``max_assistant_turns`` is exhausted.

    When ``max_rounds`` is ``None`` the sampler is bypassed: each sample is
    run once and the response is always accepted.

    When ``max_rounds`` is set, the pipeline retries up to that many times and
    accepts the first response that passes the sampler.  If no round passes,
    the last successful response is kept as a fallback so no sample is silently
    dropped (unless every LLM call for that sample fails outright).
    """

    max_rounds: int | None = None
    temperature: float = 0.8
    top_p: float = 0.95
    max_tokens: int = 4096
    system_prompt: str = field(default_factory=lambda: SYSTEM_PROMPT)
    output_dir: str = DEFAULT_OUTPUT_DIR
    show_progress: bool = True
    # Default topk for nmr_generate candidate generation, mirroring the verl
    # tool-config surface (spectune.tools write-config --nmr-gen-topk).
    nmr_gen_topk: int = 10
    # Tools exposed to the model during the agent loop; empty means
    # DEFAULT_RL_TOOL_NAMES.
    tool_names: tuple[str, ...] = ()
    max_assistant_turns: int = DEFAULT_MAX_ASSISTANT_TURNS
    max_concurrency: int = 8
    # Optional skill files (default: .md) whose contents are appended to the
    # end of the system prompt, one per line separator. Empty by default (off).
    skills: tuple[str, ...] = ()
    # When True (default), wrap the ToolManager with CachedToolManager so every
    # tool call checks the disk cache before hitting the remote service.
    cache_tool_results: bool = True

    def __post_init__(self) -> None:
        if self.max_rounds is not None and self.max_rounds < 1:
            raise ValueError("max_rounds must be at least 1 when set")
        if not 0.0 < self.temperature <= 2.0:
            raise ValueError("temperature must be in (0, 2]")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        if not 1 <= self.nmr_gen_topk <= NMR_GENERATE_MAX_TOPK:
            raise ValueError(f"nmr_gen_topk must be in [1, {NMR_GENERATE_MAX_TOPK}]")
        if self.max_assistant_turns < 1:
            raise ValueError("max_assistant_turns must be at least 1")
        if self.max_concurrency < 1:
            raise ValueError("max_concurrency must be at least 1")


__all__ = ["DEFAULT_MAX_ASSISTANT_TURNS", "DEFAULT_OUTPUT_DIR", "RolloutConfig"]
