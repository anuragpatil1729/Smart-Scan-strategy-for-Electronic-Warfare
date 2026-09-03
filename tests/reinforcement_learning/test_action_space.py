"""Unit tests for Action Space."""

import sys
if "tensorflow" not in sys.modules:
    sys.modules["tensorflow"] = None

import pytest
from reinforcement_learning.action.action_space import RLActionSpace


def test_action_space_properties():
    action_space = RLActionSpace(num_emitters=8)
    assert action_space.n == 8
    assert action_space.is_valid(0) is True
    assert action_space.is_valid(7) is True
    assert action_space.is_valid(-1) is False
    assert action_space.is_valid(8) is False


def test_action_space_sample_with_mask():
    action_space = RLActionSpace(num_emitters=8)
    mask = [0, 0, 1, 0, 0, 0, 0, 0]  # Only action 2 valid
    for _ in range(10):
        sampled = action_space.sample(mask=mask)
        assert sampled == 2
