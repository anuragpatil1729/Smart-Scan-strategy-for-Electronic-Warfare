"""Unit tests for Heuristic Baselines and Objective Metrics."""

import sys
if "tensorflow" not in sys.modules:
    sys.modules["tensorflow"] = None

import pytest
import numpy as np
from reinforcement_learning.environment.ew_environment import EWEnvironment
from reinforcement_learning.training.evaluate import (
    RandomScanner,
    RoundRobinScanner,
    HighestThreatGreedy,
    HighestUncertaintyGreedy,
    MostStaleGreedy,
    evaluate_policy,
)
from evaluation.rl_metrics.metrics import (
    EpisodeTrace,
    threat_intercept_rate,
    critical_miss_rate,
    scan_diversity_entropy,
    compute_rl_summary_metrics,
)


def test_heuristic_scanners_valid_actions():
    env = EWEnvironment(num_emitters=8, max_steps=10)
    obs, _ = env.reset(seed=42)

    scanners = [
        RandomScanner(8),
        RoundRobinScanner(8),
        HighestThreatGreedy(8),
        HighestUncertaintyGreedy(8),
        MostStaleGreedy(8),
    ]

    for s in scanners:
        action = s.predict(obs)
        assert 0 <= action < 8


def test_round_robin_cycles():
    rr = RoundRobinScanner(8)
    obs = np.zeros(84, dtype=np.float32)
    actions = [rr.predict(obs) for _ in range(16)]
    assert actions == [0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7]


def test_scan_diversity_entropy():
    # Uniform across 8 actions -> entropy close to 1.0
    uniform_actions = list(range(8)) * 100
    ent_uniform = scan_diversity_entropy(uniform_actions, num_actions=8)
    assert pytest.approx(ent_uniform, abs=1e-3) == 1.0

    # Collapsed to single action -> entropy 0.0
    collapsed_actions = [3] * 100
    ent_collapsed = scan_diversity_entropy(collapsed_actions, num_actions=8)
    assert pytest.approx(ent_collapsed, abs=1e-3) == 0.0


def test_objective_summary_metrics():
    trace = EpisodeTrace(
        rewards=[1.0, 2.0, 3.0],
        actions=[0, 1, 2],
        detections=[True, False, True],
        track_drops=1,
        total_dwell_cost=3.0,
        fleet_uncertainties=[0.4, 0.35, 0.3],
        active_threat_misses=1,
        active_threat_intercepts=3,
    )
    summary = compute_rl_summary_metrics([trace], num_actions=8)

    assert pytest.approx(summary["mean_return"]) == 6.0
    assert pytest.approx(summary["threat_intercept_rate"]) == 0.75
    assert pytest.approx(summary["critical_miss_rate"]) == 0.25
    assert pytest.approx(summary["observation_efficiency"]) == 2.0 / 3.0
    assert pytest.approx(summary["mean_track_drops"]) == 1.0
