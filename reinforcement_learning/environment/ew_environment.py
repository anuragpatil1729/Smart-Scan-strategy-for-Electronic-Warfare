"""Gymnasium-compliant Electronic Warfare Smart Scan Environment.

Simulates a dynamic radar/electronic emitter environment with:
- Multiple synthetic emitters with varying rotation periods, scan patterns,
  threat levels, and operational modes.
- Realistic pulse interception geometry (mainlobe beam sweeps).
- Track lifecycle: confirmation, revisit latency tracking, track degradation,
  and track drops upon revisit timeout.
- Bayesian-style parameter uncertainty drift (when unobserved) and reduction (when intercepted).
- 84-dimensional observation state (8 emitters x 10 estimations + 4 global features).
- Discrete beam dwell action space.
- Corrected track maintenance and track drop penalties.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from reinforcement_learning.action.action_space import RLActionSpace
from reinforcement_learning.reward.reward_function import (
    EWRewardFunction,
    RewardWeights,
    StepObservationResult,
)
from reinforcement_learning.state.state_space import (
    EmitterFeatures,
    GlobalSensorFeatures,
    RLStateBuilder,
)


@dataclass
class SyntheticEmitter:
    """Internal simulation model for a synthetic radar emitter."""

    emitter_id: int
    threat_level: float                 # Inherent threat level [0, 1]
    scan_period: float                  # Steps per full 360-degree beam rotation
    beamwidth_deg: float                # Antenna beamwidth in degrees
    azimuth_deg: float = 0.0            # Current antenna pointing angle relative to EW receiver
    operational_mode: float = 0.25      # 0.0=idle, 0.25=search, 0.5=acquisition, 0.75=track, 1.0=guidance
    burst_probability: float = 0.85     # Probability of emitting pulses when mainlobe points at receiver
    uncertainty: float = 0.80           # Current parameter uncertainty [0, 1]
    identity_confidence: float = 0.20   # Emitter classification confidence [0, 1]
    novelty: float = 0.10               # Novelty distance score [0, 1]
    dwell_cost: float = 1.0             # Cost in budget units per dwell
    
    # Track state
    is_confirmed: bool = False          # Track confirmed by receiver
    time_since_last_intercept: int = 0  # Scheduling steps since last successful pulse intercept
    is_dropped: bool = False            # True if track was dropped due to starvation

    def step_rotation(self) -> None:
        """Advance antenna scan beam rotation angle."""
        deg_per_step = 360.0 / max(1.0, self.scan_period)
        self.azimuth_deg = (self.azimuth_deg + deg_per_step) % 360.0

    def is_mainlobe_pointing_at_receiver(self) -> bool:
        """True if mainlobe beam illuminates the EW receiver at angle 0 deg."""
        diff = abs((self.azimuth_deg + 180.0) % 360.0 - 180.0)
        return diff <= (self.beamwidth_deg / 2.0)

    def is_transmitting_towards_receiver(self, rng: np.random.Generator) -> bool:
        """Simulate whether receiver intercepts active pulses during this step."""
        if self.operational_mode == 0.0:
            return False
        # Mainlobe illumination
        if self.is_mainlobe_pointing_at_receiver():
            return bool(rng.random() < self.burst_probability)
        # Sidelobe leakage (rare, 3% chance if in track/guidance)
        if self.operational_mode >= 0.75:
            return bool(rng.random() < 0.03)
        return False


class EWEnvironment(gym.Env):
    """Gymnasium Environment for Smart Scan Strategy in Electronic Warfare."""

    metadata = {"render_modes": ["human"], "render_fps": 10}

    def __init__(
        self,
        num_emitters: int = 8,
        max_steps: int = 100,
        initial_budget: float = 100.0,
        dwell_cost: float = 1.0,
        max_revisit_window: int = 10,
        drop_timeout: int = 20,
        uncertainty_drift_rate: float = 0.04,
        uncertainty_reduction_rate: float = 0.65,
        reward_weights: Optional[RewardWeights] = None,
        scenario_preset: str = "dense_fleet",
    ) -> None:
        super().__init__()

        self.num_emitters = num_emitters
        self.max_steps = max_steps
        self.initial_budget = initial_budget
        self.dwell_cost = dwell_cost
        self.max_revisit_window = max_revisit_window
        self.drop_timeout = drop_timeout
        self.uncertainty_drift_rate = uncertainty_drift_rate
        self.uncertainty_reduction_rate = uncertainty_reduction_rate
        self.scenario_preset = scenario_preset

        # State and action definitions
        self.state_builder = RLStateBuilder(num_emitters=self.num_emitters)
        self.observation_space = self.state_builder.observation_space
        self.action_space = spaces.Discrete(self.num_emitters)

        # Reward engine
        self.reward_fn = EWRewardFunction(weights=reward_weights)

        # Runtime state
        self.emitters: List[SyntheticEmitter] = []
        self.current_step = 0
        self.remaining_budget = self.initial_budget
        self.total_detections = 0
        self.total_drops = 0
        self.rng = np.random.default_rng()

    def _create_synthetic_emitters(self) -> List[SyntheticEmitter]:
        """Generate diverse radar emitter profiles based on scenario presets."""
        emitters: List[SyntheticEmitter] = []

        # Presets: diverse mix of early warning, acquisition, tracking, and agile radars
        profiles = [
            # id, threat, scan_period, beamwidth, mode, burst_prob, unc, conf
            (0, 0.95, 12.0, 15.0, 0.75, 0.90, 0.70, 0.30),  # High-threat fire control / track radar
            (1, 0.80, 8.0,  20.0, 0.50, 0.85, 0.75, 0.25),  # High-threat missile acquisition
            (2, 0.60, 24.0, 30.0, 0.25, 0.80, 0.80, 0.20),  # Medium-threat sector search radar
            (3, 0.45, 18.0, 25.0, 0.25, 0.75, 0.85, 0.15),  # Medium surveillance radar
            (4, 0.30, 30.0, 40.0, 0.25, 0.70, 0.90, 0.10),  # Low-threat long-range early warning
            (5, 0.85, 10.0, 18.0, 0.75, 0.92, 0.65, 0.35),  # High-threat agile tracking radar
            (6, 0.50, 15.0, 22.0, 0.50, 0.80, 0.75, 0.20),  # Medium naval acquisition
            (7, 0.20, 36.0, 45.0, 0.25, 0.65, 0.95, 0.10),  # Low navigation / civil radar
        ]

        if self.scenario_preset == "high_threat_surge":
            # Scenario with severe concentration of high/critical threats
            profiles[2] = (2, 0.90, 9.0, 16.0, 0.75, 0.92, 0.65, 0.35)
            profiles[3] = (3, 0.85, 11.0, 18.0, 0.50, 0.88, 0.70, 0.30)
        elif self.scenario_preset == "sparse_agile":
            # Scenario with low burst probability agile radars
            profiles[0] = (0, 0.95, 16.0, 10.0, 0.50, 0.40, 0.85, 0.15)
            profiles[1] = (1, 0.80, 14.0, 12.0, 0.50, 0.45, 0.80, 0.20)

        for i in range(self.num_emitters):
            p = profiles[i % len(profiles)]
            initial_azimuth = float(self.rng.uniform(0.0, 360.0))
            e = SyntheticEmitter(
                emitter_id=i,
                threat_level=float(p[1]),
                scan_period=float(p[2]),
                beamwidth_deg=float(p[3]),
                azimuth_deg=initial_azimuth,
                operational_mode=float(p[4]),
                burst_probability=float(p[5]),
                uncertainty=float(p[6]),
                identity_confidence=float(p[7]),
                novelty=float(self.rng.uniform(0.05, 0.25)),
                dwell_cost=self.dwell_cost,
                is_confirmed=False,
                time_since_last_intercept=0,
                is_dropped=False,
            )
            emitters.append(e)

        return emitters

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment to an initial state."""
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        if options and "scenario_preset" in options:
            self.scenario_preset = str(options["scenario_preset"])

        self.emitters = self._create_synthetic_emitters()
        self.current_step = 0
        self.remaining_budget = self.initial_budget
        self.total_detections = 0
        self.total_drops = 0

        obs = self._get_observation()
        info: Dict[str, Any] = {
            "current_step": 0,
            "remaining_budget": self.remaining_budget,
            "total_detections": 0,
            "total_drops": 0,
        }
        return obs, info

    def _get_observation(self) -> np.ndarray:
        """Construct the 84-dimensional state vector."""
        emitter_features: List[EmitterFeatures] = []

        for e in self.emitters:
            # 1. Activity probability estimate
            # Approximated by mainlobe proximity and operational mode
            beam_diff = abs((e.azimuth_deg + 180.0) % 360.0 - 180.0)
            angle_factor = max(0.0, 1.0 - (beam_diff / max(1.0, e.beamwidth_deg * 2.0)))
            activity_prob = float(np.clip(angle_factor * e.burst_probability, 0.0, 1.0))

            # 2. Threat level
            threat = float(e.threat_level)

            # 3. Identity confidence
            conf = float(e.identity_confidence)

            # 4. Uncertainty
            unc = float(e.uncertainty)

            # 5. Novelty
            nov = float(e.novelty)

            # 6. Track age / revisit urgency (informational state feature, not directly rewarded)
            track_age = float(np.clip(e.time_since_last_intercept / max(1.0, self.drop_timeout), 0.0, 1.0))

            # 7. Mode
            mode = float(e.operational_mode)

            # 8. Potential Information Gain (EX-ANTE EXPECTED REDUCTION in state, NOT reward):
            # Prior expectation of uncertainty reduction if dwell allocated
            potential_info = float(np.clip(unc * (0.5 + 0.5 * activity_prob), 0.0, 1.0))

            # 9. Observation cost
            obs_cost = float(np.clip(e.dwell_cost / max(1.0, self.initial_budget), 0.0, 1.0))

            # 10. Miss risk
            miss_risk = float(np.clip(threat * activity_prob * unc, 0.0, 1.0))

            feat = EmitterFeatures(
                activity_prob=activity_prob,
                threat_level=threat,
                identity_confidence=conf,
                uncertainty=unc,
                novelty=nov,
                track_age=track_age,
                mode=mode,
                potential_info_gain=potential_info,
                observation_cost=obs_cost,
                miss_risk=miss_risk,
            )
            emitter_features.append(feat)

        # Global features
        utilization = float(self.current_step / max(1, self.max_steps))
        budget_fraction = float(np.clip(self.remaining_budget / max(1.0, self.initial_budget), 0.0, 1.0))
        normalized_step = utilization
        high_threat_active = sum(
            1 for e in self.emitters if e.threat_level >= 0.7 and e.is_mainlobe_pointing_at_receiver()
        )
        active_threat_ratio = float(high_threat_active / max(1, self.num_emitters))

        global_features = GlobalSensorFeatures(
            sensor_utilization=utilization,
            remaining_budget_fraction=budget_fraction,
            normalized_step=normalized_step,
            active_threat_ratio=active_threat_ratio,
        )

        return self.state_builder.build_state_vector(emitter_features, global_features)

    def step(
        self,
        action: int,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one dwell scheduling action."""
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action} for Discrete({self.num_emitters})")

        action_int = int(action)
        self.current_step += 1
        target = self.emitters[action_int]

        # Pre-dwell state
        pre_uncertainty = float(target.uncertainty)
        time_since_last_visit = float(target.time_since_last_intercept)
        is_track_confirmed = bool(target.is_confirmed)

        # Deduct dwell cost from budget
        self.remaining_budget = max(0.0, self.remaining_budget - target.dwell_cost)

        # Check if target emitter is transmitting towards receiver during this dwell
        detected = target.is_transmitting_towards_receiver(self.rng)
        if detected:
            self.total_detections += 1
            target.is_confirmed = True
            target.time_since_last_intercept = 0
            # Significant uncertainty reduction upon pulse interception
            target.uncertainty = float(np.clip(
                target.uncertainty * (1.0 - self.uncertainty_reduction_rate),
                0.01,
                1.0,
            ))
            # Boost identification confidence
            target.identity_confidence = float(np.clip(target.identity_confidence + 0.15, 0.0, 1.0))
        else:
            # Dwell executed but no pulses intercepted (pointing away / silent)
            # Minor negative info gain (knowing it was not transmitting in mainlobe)
            target.uncertainty = float(np.clip(target.uncertainty - 0.02, 0.01, 1.0))

        post_uncertainty = float(target.uncertainty)

        # Advance emitter rotations and unobserved track dynamics
        dropped_threat_sum = 0.0
        unobserved_active_threats: List[float] = []

        for i, e in enumerate(self.emitters):
            # Advance scan beam rotation
            e.step_rotation()

            if i != action_int:
                # Emitter was not observed in this dwell
                e.time_since_last_intercept += 1

                # Parameter uncertainty drifts upward due to elapsed time
                e.uncertainty = float(np.clip(
                    e.uncertainty + self.uncertainty_drift_rate,
                    0.0,
                    1.0,
                ))

                # Check if this unobserved emitter was actively transmitting (missed burst)
                if e.is_transmitting_towards_receiver(self.rng):
                    unobserved_active_threats.append(e.threat_level)

                # Check for track drop timeout on confirmed tracks
                if e.is_confirmed and not e.is_dropped and e.time_since_last_intercept > self.drop_timeout:
                    e.is_dropped = True
                    self.total_drops += 1
                    dropped_threat_sum += float(e.threat_level)

        # Compute multi-objective reward with corrected track & info logic
        step_result = StepObservationResult(
            action=action_int,
            detected=detected,
            threat_level=float(target.threat_level),
            pre_uncertainty=pre_uncertainty,
            post_uncertainty=post_uncertainty,
            is_track_confirmed=is_track_confirmed,
            time_since_last_visit=time_since_last_visit,
            max_revisit_window=float(self.max_revisit_window),
            dropped_threat_sum=dropped_threat_sum,
            dwell_cost=target.dwell_cost,
            initial_budget=self.initial_budget,
            unobserved_active_threats=unobserved_active_threats,
        )
        telemetry = self.reward_fn.compute(step_result)

        # Episode termination / truncation
        terminated = False
        truncated = False

        if self.current_step >= self.max_steps:
            truncated = True
        elif self.remaining_budget <= 0.0:
            terminated = True

        obs = self._get_observation()
        info: Dict[str, Any] = {
            "current_step": self.current_step,
            "action": action_int,
            "detected": detected,
            "remaining_budget": self.remaining_budget,
            "total_detections": self.total_detections,
            "total_drops": self.total_drops,
            "telemetry": telemetry.to_dict(),
        }

        return obs, telemetry.total_reward, terminated, truncated, info
