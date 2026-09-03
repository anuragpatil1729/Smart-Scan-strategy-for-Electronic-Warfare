"""Reinforcement Learning Subsystem for Smart Scan Strategy in Electronic Warfare."""

import sys

# Prevent broken Anaconda base TensorFlow / Keras build from segfaulting when
# PyTorch / TensorBoard is loaded in Python 3.13 on macOS ARM64.
if "tensorflow" not in sys.modules:
    sys.modules["tensorflow"] = None

from reinforcement_learning.environment.ew_environment import EWEnvironment
from reinforcement_learning.state.state_space import RLStateBuilder, EmitterFeatures, GlobalSensorFeatures
from reinforcement_learning.state.upstream_interface import (
    UpstreamEmitterRecord,
    UpstreamScanContext,
    UpstreamStateAdapter,
)
from reinforcement_learning.action.action_space import RLActionSpace
from reinforcement_learning.reward.reward_function import EWRewardFunction, RewardWeights

__all__ = [
    "EWEnvironment",
    "RLStateBuilder",
    "EmitterFeatures",
    "GlobalSensorFeatures",
    "UpstreamEmitterRecord",
    "UpstreamScanContext",
    "UpstreamStateAdapter",
    "RLActionSpace",
    "EWRewardFunction",
    "RewardWeights",
]
