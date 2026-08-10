"""Rejection sampler driven by ground-truth SMILES reward."""

from __future__ import annotations

from typing import Any

from spectune.reward import RewardConfig, RewardEvaluator

from .base import BaseSampler


class GtRejectionSampler(BaseSampler):
    """Accept a response only when the GT SMILES appears in the final answer.

    Uses :class:`~spectune.reward.RewardEvaluator` with the v1 format contract.
    A response is accepted when ``details["gt_rank"]`` is not ``None``, meaning
    the model placed the ground-truth molecule at any rank in its answer.
    """

    def __init__(self, config: RewardConfig | None = None) -> None:
        self.evaluator = RewardEvaluator(config)

    def accept(self, response: str, sample: dict[str, Any]) -> bool:
        gt_smiles = sample.get("gt_smiles", "")
        if not gt_smiles:
            return False
        result = self.evaluator.evaluate(response, gt_smiles)
        return result.details.get("gt_rank") is not None


__all__ = ["GtRejectionSampler"]
