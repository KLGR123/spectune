"""Versioned interaction formats for Spectune training and evaluation."""

from .v1 import (
    ANSWER_EXAMPLE,
    FORMAT_SPEC_VERSION,
    SYSTEM_PROMPT,
    TOOL_CALL_END,
    TOOL_CALL_START,
    extract_smiles_candidates,
    final_answer_region,
    format_final_answer,
    messages_from_decoded_hermes,
    strip_tool_calls,
)

__all__ = [
    "ANSWER_EXAMPLE",
    "FORMAT_SPEC_VERSION",
    "SYSTEM_PROMPT",
    "TOOL_CALL_END",
    "TOOL_CALL_START",
    "extract_smiles_candidates",
    "final_answer_region",
    "format_final_answer",
    "messages_from_decoded_hermes",
    "strip_tool_calls",
]
