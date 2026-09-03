"""Unit tests for Reward Function, Track Logic, and Info Gain Separation."""

import sys
if "tensorflow" not in sys.modules:
    sys.modules["tensorflow"] = None

import pytest
from reinforcement_learning.reward.reward_function import (
    EWRewardFunction,
    RewardWeights,
    StepObservationResult,
)


def test_track_maintenance_reward_within_window():
    """Verify that servicing a confirmed threat within revisit window yields track maintenance reward."""
    weights = RewardWeights(w_track=2.0, w_drop=3.5, w_threat=0.0, w_detect=0.0, w_info=0.0)
    rf = EWRewardFunction(weights=weights)

    res = StepObservationResult(
        action=0,
        detected=True,
        threat_level=0.9,
        pre_uncertainty=0.5,
        post_uncertainty=0.2,
        is_track_confirmed=True,
        time_since_last_visit=3.0,     # Within revisit window (10.0)
        max_revisit_window=10.0,
        dropped_threat_sum=0.0,
        dwell_cost=1.0,
        initial_budget=100.0,
    )
    tel = rf.compute(res)
    assert tel.track_reward > 1.0
    assert tel.drop_penalty == 0.0


def test_track_maintenance_degrades_when_overdue():
    """Verify that track maintenance reward is lower when revisit was delayed beyond window."""
    weights = RewardWeights(w_track=2.0, w_drop=0.0)
    rf = EWRewardFunction(weights=weights)

    # Prompt revisit (dt = 1)
    prompt_res = StepObservationResult(
        action=0,
        detected=True,
        threat_level=0.9,
        pre_uncertainty=0.5,
        post_uncertainty=0.2,
        is_track_confirmed=True,
        time_since_last_visit=1.0,
        max_revisit_window=10.0,
        dropped_threat_sum=0.0,
        dwell_cost=1.0,
        initial_budget=100.0,
    )
    prompt_tel = rf.compute(prompt_res)

    # Delayed revisit (dt = 15 > 10)
    delayed_res = StepObservationResult(
        action=0,
        detected=True,
        threat_level=0.9,
        pre_uncertainty=0.5,
        post_uncertainty=0.2,
        is_track_confirmed=True,
        time_since_last_visit=15.0,
        max_revisit_window=10.0,
        dropped_threat_sum=0.0,
        dwell_cost=1.0,
        initial_budget=100.0,
    )
    delayed_tel = rf.compute(delayed_res)

    assert prompt_tel.track_reward > delayed_tel.track_reward


def test_track_drop_penalty():
    """Verify that allowing an active threat track to drop triggers drop penalty."""
    weights = RewardWeights(w_drop=3.5)
    rf = EWRewardFunction(weights=weights)

    res = StepObservationResult(
        action=1,
        detected=False,
        threat_level=0.2,
        pre_uncertainty=0.8,
        post_uncertainty=0.8,
        is_track_confirmed=False,
        time_since_last_visit=0.0,
        max_revisit_window=10.0,
        dropped_threat_sum=0.95,  # Critical threat dropped!
        dwell_cost=1.0,
        initial_budget=100.0,
    )
    tel = rf.compute(res)
    assert pytest.approx(tel.drop_penalty) == 3.5 * 0.95


def test_actual_info_gain_ex_post_realization():
    """Verify that actual info gain in reward is the realized reduction in uncertainty."""
    weights = RewardWeights(w_info=2.0)
    rf = EWRewardFunction(weights=weights)

    # Big reduction in uncertainty
    res_detect = StepObservationResult(
        action=0,
        detected=True,
        threat_level=0.5,
        pre_uncertainty=0.8,
        post_uncertainty=0.2,
        is_track_confirmed=False,
        time_since_last_visit=0.0,
        max_revisit_window=10.0,
        dropped_threat_sum=0.0,
        dwell_cost=1.0,
        initial_budget=100.0,
    )
    tel_detect = rf.compute(res_detect)
    # Expected gain = 0.8 - 0.2 = 0.6 * weight 2.0 = 1.2
    assert pytest.approx(tel_detect.actual_info_gain) == 1.2

    # Zero reduction (missed / silence)
    res_miss = StepObservationResult(
        action=0,
        detected=False,
        threat_level=0.5,
        pre_uncertainty=0.8,
        post_uncertainty=0.8,
        is_track_confirmed=False,
        time_since_last_visit=0.0,
        max_revisit_window=10.0,
        dropped_threat_sum=0.0,
        dwell_cost=1.0,
        initial_budget=100.0,
    )
    tel_miss = rf.compute(res_miss)
    assert tel_miss.actual_info_gain == 0.0


def test_redundant_scan_penalty():
    """Verify that scanning an already-saturated emitter (uncertainty < 0.15) incurs a penalty."""
    weights = RewardWeights(w_redundant=1.0)
    rf = EWRewardFunction(weights=weights)

    res = StepObservationResult(
        action=0,
        detected=True,
        threat_level=0.5,
        pre_uncertainty=0.05,  # Very low uncertainty already!
        post_uncertainty=0.02,
        is_track_confirmed=False,
        time_since_last_visit=0.0,
        max_revisit_window=10.0,
        dropped_threat_sum=0.0,
        dwell_cost=1.0,
        initial_budget=100.0,
    )
    tel = rf.compute(res)
    assert tel.redundant_penalty > 0.0
