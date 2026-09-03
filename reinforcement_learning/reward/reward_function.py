"""Reward Function Formulation for Electronic Warfare Smart Scan RL.

Features:
- Fixed Recency / Track Maintenance logic:
  - Track maintenance reward (R_track) rewards servicing confirmed active threats
    within their allowable revisit window.
  - Track drop penalty (R_drop) explicitly penalizes letting active confirmed tracks
    drop due to starvation / neglect.
  - Emitter track age (in state) is purely an observational urgency signal, NOT
    perversely rewarded as an incentive to delay observations.
- Strict separation of Potential Info Gain (state feature, ex-ante) from
  Actual Info Gain (reward term, ex-post realized reduction).
- Comprehensive multi-objective formulation with configurable weights.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence
import numpy as np


@dataclass
class RewardWeights:
    """Configurable weights for multi-objective reward function."""

    w_threat: float = 2.5       # Intercepting high-threat active transmissions
    w_detect: float = 1.0       # Any successful pulse interception
    w_info: float = 2.0         # Realized (ex-post) parameter uncertainty reduction
    w_track: float = 2.0        # Maintaining active threat tracks within revisit window
    w_drop: float = 3.5         # Penalty for dropping confirmed tracks
    w_cost: float = 0.2         # Sensor dwell energy/resource expenditure penalty
    w_miss: float = 1.5         # Penalty for concurrent active threats left unobserved
    w_redundant: float = 1.0    # Penalty for repeatedly scanning already-saturated emitters


@dataclass
class StepObservationResult:
    """Physical results of sensor dwell for computing rewards."""

    action: int                                 # Selected emitter index
    detected: bool                              # True if pulses intercepted in dwell
    threat_level: float                         # Threat lethality of selected emitter [0, 1]
    pre_uncertainty: float                      # Emitter uncertainty prior to dwell [0, 1]
    post_uncertainty: float                     # Emitter uncertainty after dwell [0, 1]
    is_track_confirmed: bool                    # Emitter is an active confirmed track
    time_since_last_visit: float                # Dwell intervals elapsed since last observation
    max_revisit_window: float                   # Window before track degradation begins
    dropped_threat_sum: float                   # Sum of threat scores for tracks dropped this step
    dwell_cost: float                           # Resource cost of current dwell
    initial_budget: float                       # Total mission budget
    unobserved_active_threats: List[float] = field(default_factory=list)  # Active threat scores missed


@dataclass
class RewardTelemetry:
    """Detailed breakdown of individual reward terms for logging and analysis."""

    total_reward: float = 0.0
    threat_reward: float = 0.0
    detect_reward: float = 0.0
    actual_info_gain: float = 0.0
    track_reward: float = 0.0
    drop_penalty: float = 0.0
    cost_penalty: float = 0.0
    miss_penalty: float = 0.0
    redundant_penalty: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "total_reward": self.total_reward,
            "threat_reward": self.threat_reward,
            "detect_reward": self.detect_reward,
            "actual_info_gain": self.actual_info_gain,
            "track_reward": self.track_reward,
            "drop_penalty": self.drop_penalty,
            "cost_penalty": self.cost_penalty,
            "miss_penalty": self.miss_penalty,
            "redundant_penalty": self.redundant_penalty,
        }


class EWRewardFunction:
    """Calculates multi-objective scan strategy rewards with correct track & info logic."""

    def __init__(self, weights: Optional[RewardWeights] = None) -> None:
        self.weights = weights if weights is not None else RewardWeights()

    def compute(self, result: StepObservationResult) -> RewardTelemetry:
        """Compute the composite reward and breakdown for a single scheduling step."""
        # 1. Detection Base Reward: 1.0 if intercepted pulses, else 0.0
        detect_term = 1.0 if result.detected else 0.0

        # 2. Threat-Weighted Detection Reward
        threat_term = float(result.threat_level) * detect_term

        # 3. ACTUAL Information Gain (Ex-Post Realized Reduction):
        # Strictly separated from potential info gain (which is an ex-ante state feature).
        actual_info_gain = max(0.0, float(result.pre_uncertainty) - float(result.post_uncertainty))

        # 4. Corrected Track Maintenance Reward:
        # Rewarding timely revisits for active confirmed threats within revisit window.
        # If the emitter was visited within allowable window (dt <= max_revisit_window),
        # award track maintenance continuity. If overdue (dt > max_revisit_window),
        # track is degrading so track reward decays.
        track_term = 0.0
        if result.is_track_confirmed and result.threat_level >= 0.3:
            revisit_ratio = result.time_since_last_visit / max(1.0, result.max_revisit_window)
            if revisit_ratio <= 1.0:
                # Within valid tracking window -> full track maintenance reward
                track_term = float(result.threat_level) * (1.0 - 0.3 * revisit_ratio)
            else:
                # Overdue revisit -> degraded reward
                track_term = float(result.threat_level) * max(0.0, 0.7 * (2.0 - revisit_ratio))

        # 5. Track Drop Penalty:
        # Penalizes any confirmed track dropped at this step due to revisit timeout.
        drop_term = float(result.dropped_threat_sum)

        # 6. Dwell Resource Cost Penalty
        cost_term = float(result.dwell_cost) / max(1.0, float(result.initial_budget))

        # 7. Unobserved Active Miss Penalty:
        # Penalizes leaving other active threats unobserved concurrently.
        miss_term = float(sum(result.unobserved_active_threats))

        # 8. Redundant Scan Penalty:
        # Penalizes scanning an emitter that is already near-zero uncertainty (< 0.15),
        # discouraging wasteful over-sampling when other emitters require attention.
        redundant_term = 0.0
        if result.detected and result.pre_uncertainty < 0.15:
            redundant_term = (0.15 - result.pre_uncertainty) / 0.15

        # Composite multi-objective scalar reward
        total_reward = (
            self.weights.w_threat * threat_term
            + self.weights.w_detect * detect_term
            + self.weights.w_info * actual_info_gain
            + self.weights.w_track * track_term
            - self.weights.w_drop * drop_term
            - self.weights.w_cost * cost_term
            - self.weights.w_miss * miss_term
            - self.weights.w_redundant * redundant_term
        )

        return RewardTelemetry(
            total_reward=float(total_reward),
            threat_reward=float(self.weights.w_threat * threat_term),
            detect_reward=float(self.weights.w_detect * detect_term),
            actual_info_gain=float(self.weights.w_info * actual_info_gain),
            track_reward=float(self.weights.w_track * track_term),
            drop_penalty=float(self.weights.w_drop * drop_term),
            cost_penalty=float(self.weights.w_cost * cost_term),
            miss_penalty=float(self.weights.w_miss * miss_term),
            redundant_penalty=float(self.weights.w_redundant * redundant_term),
        )
