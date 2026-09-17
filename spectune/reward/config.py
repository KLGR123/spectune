"""Configuration for structure-elucidation rewards."""

from __future__ import annotations

from dataclasses import dataclass

COMPONENT_KEYS = (
    "gt_smiles",
    "tool_call_format",
    "smiles_validity",
    "tool_call_count",
)


@dataclass(frozen=True, slots=True)
class RewardConfig:
    """Reward magnitudes used by :class:`~spectune.reward.RewardEvaluator`.

    Every field below is the literal point value added to (or subtracted
    from) the final score -- there is no separate weighting/normalization
    step. A ground-truth hit at one-based rank ``r`` receives
    ``gt_match_reward * rank_discount ** (r - 1)``.

    The final score is the sum of the four core components (``gt_smiles``,
    ``tool_call_format``, ``smiles_validity``, ``tool_call_count``) plus the
    situational bonuses/penalties below (``nmr_diversity_bonus``,
    ``gt_loose_match_reward``, ``code_interpreter_missing_print_penalty``,
    ``code_interpreter_error_penalty``).
    """

    gt_match_reward: float = 0.7
    rank_discount: float = 0.8
    invalid_tool_call_penalty: float = -0.1
    invalid_smiles_penalty: float = -0.1
    # Align with typical verl multi_turn.max_assistant_turns for tool agents.
    max_tool_calls: int | None = 16
    excess_tool_call_penalty: float = -0.01
    # When True (default), only format-spec v1 answers are scored:
    # ``{"smiles": ["...", ...]}``. Set False only for offline legacy audits.
    strict_answer_format: bool = True
    # Bonus awarded when the trajectory called ``nmr_generate``, got full
    # (rank-1) credit on ``gt_smiles``, and its top answer candidate differs
    # from every ``nmr_generate`` call's own top candidate, so the model
    # isn't just parroting the tool.
    nmr_diversity_bonus: float = 0.5
    # Penalties applied per ``code_interpreter`` call and summed across the
    # trajectory. Charged when a call's ``code`` argument never invokes
    # ``print(...)``, so its stdout is silently empty.
    code_interpreter_missing_print_penalty: float = -0.02
    # Charged when a call's tool response reports ``status == "error"`` (e.g.
    # a bad RDKit API usage that raised an exception).
    code_interpreter_error_penalty: float = -0.05
    # Fallback reward for a *lenient* GT match (see
    # ``spectune.reward.skeleton_match``) that ignores stereochemistry noise
    # (unspecified E/Z bonds, a lone stereocenter's chirality, whole-molecule
    # enantiomer inversion). Only applied when the exact ``gt_smiles`` match
    # misses entirely (``gt_rank is None``) but a lenient match exists;
    # discounted by ``rank_discount`` the same way as the exact match. Kept
    # below ``gt_match_reward`` so a rank-1 exact hit always outscores a
    # rank-1 lenient-only hit.
    gt_loose_match_reward: float = 0.35

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
        if self.code_interpreter_missing_print_penalty > 0:
            raise ValueError("code_interpreter_missing_print_penalty must be non-positive")
        if self.code_interpreter_error_penalty > 0:
            raise ValueError("code_interpreter_error_penalty must be non-positive")
        if self.gt_loose_match_reward < 0:
            raise ValueError("gt_loose_match_reward must be non-negative")


__all__ = ["COMPONENT_KEYS", "RewardConfig"]
