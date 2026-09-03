"""Objective Reinforcement Learning Evaluation Metrics for Electronic Warfare Scan Strategies.

Evaluates scan policies objectively across:
- Cumulative Return
- Threat Intercept Rate (fraction of active high-threat transmissions intercepted)
- Critical Miss Rate (fraction of active high-threat transmissions unobserved)
- Track Drop Rate (lost tracks per episode due to revisit timeout)
- Observation Efficiency (detections per unit dwell cost)
- Mean Fleet Uncertainty (average uncertainty across all emitters)
- Scan Diversity Entropy (Shannon entropy of action distribution)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence
import numpy as np


@dataclass
class EpisodeTrace:
    """Telemetry recorded over a single evaluation episode."""

    rewards: List[float] = field(default_factory=list)
    actions: List[int] = field(default_factory=list)
    detections: List[bool] = field(default_factory=list)
    threat_levels: List[float] = field(default_factory=list)
    track_drops: int = 0
    total_dwell_cost: float = 0.0
    fleet_uncertainties: List[float] = field(default_factory=list)
    active_threat_misses: int = 0
    active_threat_intercepts: int = 0


def threat_intercept_rate(trace: EpisodeTrace) -> float:
    """Fraction of active high-threat bursts that were intercepted."""
    total = trace.active_threat_intercepts + trace.active_threat_misses
    if total == 0:
        return 1.0
    return float(trace.active_threat_intercepts / total)


def critical_miss_rate(trace: EpisodeTrace) -> float:
    """Fraction of active high-threat bursts that occurred unobserved."""
    total = trace.active_threat_intercepts + trace.active_threat_misses
    if total == 0:
        return 0.0
    return float(trace.active_threat_misses / total)


def observation_efficiency(trace: EpisodeTrace) -> float:
    """Successful detections per unit dwell energy/cost."""
    if trace.total_dwell_cost <= 0.0:
        return 0.0
    num_detections = sum(1 for d in trace.detections if d)
    return float(num_detections / trace.total_dwell_cost)


def scan_diversity_entropy(actions: Sequence[int], num_actions: int = 8) -> float:
    """Shannon entropy of the action distribution, normalized to [0, 1].

    Entropy = 0 means policy collapsed to repeatedly scanning one emitter.
    Entropy = 1 means policy distributes dwells uniformly across all emitters.
    """
    if len(actions) == 0:
        return 0.0
    counts = np.bincount(actions, minlength=num_actions)
    probs = counts / float(len(actions))
    # Exclude zero probabilities from log
    nonzero_probs = probs[probs > 0]
    entropy = -np.sum(nonzero_probs * np.log2(nonzero_probs))
    max_entropy = np.log2(num_actions)
    return float(entropy / max_entropy) if max_entropy > 0 else 0.0


def compute_rl_summary_metrics(
    traces: Sequence[EpisodeTrace],
    num_actions: int = 8,
) -> Dict[str, float]:
    """Aggregate comprehensive performance metrics across an evaluation batch."""
    if len(traces) == 0:
        return {}

    returns = [sum(t.rewards) for t in traces]
    intercept_rates = [threat_intercept_rate(t) for t in traces]
    miss_rates = [critical_miss_rate(t) for t in traces]
    drops = [float(t.track_drops) for t in traces]
    efficiencies = [observation_efficiency(t) for t in traces]
    
    # Flatten all actions across episodes for global diversity entropy
    all_actions = [a for t in traces for a in t.actions]
    diversity = scan_diversity_entropy(all_actions, num_actions=num_actions)

    mean_uncertainties = [
        np.mean(t.fleet_uncertainties) if len(t.fleet_uncertainties) > 0 else 0.5
        for t in traces
    ]

    return {
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "threat_intercept_rate": float(np.mean(intercept_rates)),
        "critical_miss_rate": float(np.mean(miss_rates)),
        "mean_track_drops": float(np.mean(drops)),
        "observation_efficiency": float(np.mean(efficiencies)),
        "mean_fleet_uncertainty": float(np.mean(mean_uncertainties)),
        "scan_diversity_entropy": float(diversity),
    }
