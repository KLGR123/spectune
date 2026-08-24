"""Configuration for structure-elucidation rewards."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

COMPONENT_KEYS = (
    "gt_smiles",
    "tool_call_format",
    "smiles_validity",
    "tool_call_count",
)

_WEIGHT_SUM_TOLERANCE = 1e-6


def _default_component_weights() -> dict[str, float]:
    # Prefer a strong GT learning signal; format / validity are light shaping.
    return {
        "gt_smiles": 0.7,
        "tool_call_format": 0.1,
        "smiles_validity": 0.1,
        "tool_call_count": 0.1,
    }


@dataclass(frozen=True, slots=True)
class RewardConfig:
    """Weights and limits used by :class:`~spectune.reward.RewardEvaluator`.

    A ground-truth hit at one-based rank ``r`` receives
    ``gt_match_reward * rank_discount ** (r - 1)``. Penalty values are kept
    negative so each raw component can be inspected before weighting.

    The final score is ``sum(component_weights[k] * components[k])``. Weights
    must be non-negative and sum to ``1``.
    """

    gt_match_reward: float = 1.0
    rank_discount: float = 0.8
    invalid_tool_call_penalty: float = -1.0
    invalid_smiles_penalty: float = -1.0
    # Align with typical verl multi_turn.max_assistant_turns for tool agents.
    max_tool_calls: int | None = 16
    excess_tool_call_penalty: float = -0.1
    # When True (default), only format-spec v1 answers are scored:
    # ``{"smiles": ["...", ...]}``. Set False only for offline legacy audits.
    strict_answer_format: bool = True
    component_weights: Mapping[str, float] = field(default_factory=_default_component_weights)
    # Extra, unweighted bonus awarded when the trajectory called ``nmr_generate``,
    # got full (rank-1) GT credit on the weighted ``gt_smiles`` component --
    # i.e. ``weighted_components["gt_smiles"] == gt_match_reward *
    # component_weights["gt_smiles"]`` (0.7 under the defaults above) -- and its
    # top answer candidate differs from every ``nmr_generate`` call's own top
    # candidate, so the model isn't just parroting the tool. Added directly to
    # the final score, outside the ``component_weights`` sum-to-1 system.
    nmr_diversity_bonus: float = 0.5

    def __post_init__(self) -> None:
        if self.gt_match_reward < 0:
            raise ValueError("gt_match_reward must be non-negative")
        if not 0 <= self.rank_discount <= 1:
            raise ValueError("rank_discount must be between 0 and 1")
        if self.invalid_tool_call_penalty > 0:
            raise ValueError("invalid_tool_call_penalty must be non-positive")
        if self.invalid_smiles_penalty > 0:
            raise ValueError("invalid_smiles_penalty must be non-positive")
        if self.max_tool_calls is not None and self.max_tool_calls < 0:
            raise ValueError("max_tool_calls must be non-negative or None")
        if self.excess_tool_call_penalty > 0:
            raise ValueError("excess_tool_call_penalty must be non-positive")
        if self.nmr_diversity_bonus < 0:
            raise ValueError("nmr_diversity_bonus must be non-negative")

        if not isinstance(self.component_weights, Mapping):
            raise TypeError("component_weights must be a mapping")
        weights = {str(key): float(value) for key, value in self.component_weights.items()}
        missing = [key for key in COMPONENT_KEYS if key not in weights]
        unknown = [key for key in weights if key not in COMPONENT_KEYS]
        if missing or unknown:
            raise ValueError(
                f"component_weights must contain exactly {list(COMPONENT_KEYS)}; missing={missing}, unknown={unknown}"
            )
        if any(value < 0 for value in weights.values()):
            raise ValueError("component_weights values must be non-negative")
        weight_sum = sum(weights[key] for key in COMPONENT_KEYS)
        if abs(weight_sum - 1.0) > _WEIGHT_SUM_TOLERANCE:
            raise ValueError(f"component_weights must sum to 1, got {weight_sum}")
        object.__setattr__(
            self,
            "component_weights",
            MappingProxyType({key: weights[key] for key in COMPONENT_KEYS}),
        )


__all__ = ["COMPONENT_KEYS", "RewardConfig"]
