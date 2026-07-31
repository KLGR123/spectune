"""Reward functions for molecular-structure reinforcement learning.

Core scoring lives here. Optional trainer adapters (e.g. ``spectune.reward.verl``)
are imported explicitly by training entrypoints, not as part of the core API.
"""

from .base import RewardResult
from .config import COMPONENT_KEYS, RewardConfig
from .evaluator import RewardEvaluator

__all__ = [
    "COMPONENT_KEYS",
    "RewardConfig",
    "RewardEvaluator",
    "RewardResult",
]
