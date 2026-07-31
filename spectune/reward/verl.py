"""Optional verl adapter matching custom reward-function interfaces.

Configured as::

    reward.custom_reward_function.path=pkg://spectune.reward.verl
    reward.custom_reward_function.name=compute_score

Hard ``verl`` imports are never required here — only Spectune scoring logic.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import fields
from typing import Any

from spectune.tools.catalog import DEFAULT_RL_TOOL_NAMES, schemas_for_names

from .config import RewardConfig
from .evaluator import RewardEvaluator

_CONFIG_FIELDS = {field.name for field in fields(RewardConfig)}


def _as_str_list(value: Any) -> list[str] | None:
    if isinstance(value, str | bytes) or not isinstance(value, Sequence):
        return None
    names = [str(item).strip() for item in value if str(item).strip()]
    return names or None


def _resolve_tool_schemas(extra_info: Mapping[str, Any]) -> Sequence[Mapping[str, Any]]:
    """Resolve the tool surface the reward judge should enforce.

    Priority:
    1. ``extra_info["tool_schemas"]`` (explicit, possibly from artifacts)
    2. ``extra_info["tool_names"]`` → live schemas from ``ToolManager``
    3. ``extra_info["tools_kwargs"]`` keys → live schemas
    4. ``DEFAULT_RL_TOOL_NAMES``
    """
    configured = extra_info.get("tool_schemas")
    if isinstance(configured, Sequence) and not isinstance(configured, str | bytes) and configured:
        return list(configured)

    tool_names = _as_str_list(extra_info.get("tool_names"))
    if tool_names:
        return schemas_for_names(tool_names)

    tools_kwargs = extra_info.get("tools_kwargs")
    if isinstance(tools_kwargs, Mapping) and tools_kwargs:
        return schemas_for_names(tuple(tools_kwargs.keys()))
    return schemas_for_names(DEFAULT_RL_TOOL_NAMES)


def _resolve_rollout(solution_str: str, extra_info: Mapping[str, Any]) -> Any:
    """Prefer structured messages when present; else the decoded response string."""
    rollout = extra_info.get("rollout_messages")
    if isinstance(rollout, Sequence) and not isinstance(rollout, str | bytes):
        return rollout
    return solution_str


def compute_score(
    solution_str: str,
    ground_truth: Any,
    *,
    extra_info: Mapping[str, Any] | None = None,
    data_source: Any = None,
    **kwargs: Any,
) -> float:
    """Compute one scalar score using the signature expected by verl.

    Reward settings can be passed as keyword arguments or under
    ``extra_info["reward_config"]``. If ``extra_info["rollout_messages"]`` is
    present it is used instead of the decoded response string. Otherwise the
    string is rebuilt into hermes / tool turns via
    :func:`~spectune.format.v1.messages_from_decoded_hermes`.
    """
    del data_source  # Provided by verl reward managers; unused by Spectune scoring.
    extra_info = extra_info if isinstance(extra_info, Mapping) else {}
    config_values: dict[str, Any] = {}
    nested_config = extra_info.get("reward_config")
    if isinstance(nested_config, Mapping):
        config_values.update({key: value for key, value in nested_config.items() if key in _CONFIG_FIELDS})
    config_values.update({key: value for key, value in kwargs.items() if key in _CONFIG_FIELDS})

    tool_schemas = _resolve_tool_schemas(extra_info)
    evaluator = RewardEvaluator(RewardConfig(**config_values), tool_schemas=tool_schemas)
    return evaluator.evaluate(_resolve_rollout(solution_str, extra_info), ground_truth).score


def compute_score_batched(
    solution_strs: Sequence[str],
    ground_truths: Sequence[Any],
    *,
    extra_infos: Sequence[Mapping[str, Any] | None] | None = None,
    **kwargs: Any,
) -> list[float]:
    """Batch wrapper for verl reward managers that call one Python function."""
    if len(solution_strs) != len(ground_truths):
        raise ValueError("solution_strs and ground_truths must have the same length")
    extras = list(extra_infos) if extra_infos is not None else [None] * len(solution_strs)
    if len(extras) != len(solution_strs):
        raise ValueError("extra_infos and solution_strs must have the same length")
    return [
        compute_score(solution, truth, extra_info=extra, **kwargs)
        for solution, truth, extra in zip(solution_strs, ground_truths, extras, strict=True)
    ]


__all__ = ["compute_score", "compute_score_batched"]
