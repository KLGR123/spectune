"""Optional verl adapter matching custom reward-function interfaces.

Configured as::

    custom_reward_function.path=/abs/path/to/spectune/reward/verl.py
    custom_reward_function.name=compute_score

Stock verl loads this file via ``spec_from_file_location`` as a standalone
module (not as ``spectune.reward``), so imports here must be absolute
``spectune.*`` paths — relative imports would fail with
"attempted relative import with no known parent package".

Hard ``verl`` imports are never required here — only Spectune scoring logic.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import fields
from typing import Any

from spectune.reward.config import RewardConfig
from spectune.reward.evaluator import RewardEvaluator
from spectune.tools.catalog import DEFAULT_RL_TOOL_NAMES, schemas_for_names

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
) -> dict[str, float]:
    """Compute reward and per-component scores using the signature expected by verl.

    Returns a dict with ``score`` (total reward, including the unweighted
    ``nmr_diversity_bonus``) plus individual component keys (``gt_smiles``,
    ``tool_call_format``, ``smiles_validity``, ``tool_call_count``,
    ``nmr_diversity_bonus``) so verl logs them as separate TensorBoard curves.

    Reward settings can be passed as keyword arguments or under
    ``extra_info["reward_config"]``. If ``extra_info["rollout_messages"]`` is
    present it is used instead of the decoded response string. Otherwise the
    string is rebuilt into hermes / tool turns via
    :func:`~spectune.format.v1.messages_from_decoded_hermes`.
    """
    del data_source  # Provided by verl reward managers; unused by Spectune scoring.
    extra_info = extra_info if isinstance(extra_info, Mapping) else {}

    if os.getenv("VERL_DEBUG"):
        print("\n[DEBUG] === compute_score entry ===")
        print(f"[DEBUG] ground_truth = {ground_truth!r}")
        rollout_msgs = extra_info.get("rollout_messages")
        if rollout_msgs:
            print(f"[DEBUG] rollout_messages ({len(rollout_msgs)} turns):")
            for i, m in enumerate(rollout_msgs):
                role = m.get("role", "?") if isinstance(m, Mapping) else "?"
                content = str(m.get("content", ""))[:200] if isinstance(m, Mapping) else str(m)[:200]
                print(f"[DEBUG]   [{i}] {role}: {content!r}")
        else:
            print(f"[DEBUG] solution_str = {solution_str[:500]!r}")
        breakpoint()
    config_values: dict[str, Any] = {}
    nested_config = extra_info.get("reward_config")
    if isinstance(nested_config, Mapping):
        config_values.update({key: value for key, value in nested_config.items() if key in _CONFIG_FIELDS})
    config_values.update({key: value for key, value in kwargs.items() if key in _CONFIG_FIELDS})

    tool_schemas = _resolve_tool_schemas(extra_info)
    evaluator = RewardEvaluator(RewardConfig(**config_values), tool_schemas=tool_schemas)
    result = evaluator.evaluate(_resolve_rollout(solution_str, extra_info), ground_truth)
    # ``nmr_diversity_bonus`` is added to ``score`` outside the weighted
    # component sum (see RewardConfig), so it isn't in ``result.components``.
    # Surface it explicitly so verl's NaiveRewardManager/DAPORewardManager
    # pick it up into ``reward_extra_info`` and it gets its own
    # ``reward_components/nmr_diversity_bonus/mean`` TensorBoard curve.
    return {
        "score": result.score,
        **result.components,
        "nmr_diversity_bonus": result.details["nmr_diversity_bonus"],
    }


def compute_score_batched(
    solution_strs: Sequence[str],
    ground_truths: Sequence[Any],
    *,
    extra_infos: Sequence[Mapping[str, Any] | None] | None = None,
    **kwargs: Any,
) -> list[dict[str, float]]:
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
