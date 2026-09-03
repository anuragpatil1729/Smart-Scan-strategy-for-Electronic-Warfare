"""State Space Representation for Electronic Warfare Smart Scan RL.

Defines the 84-dimensional observation space:
- 8 emitters x 10 estimation features = 80 dimensions
- 4 global sensor/system features = 4 dimensions
Total = 84 dimensions.

The 10 estimations per emitter are input features provided to the policy
(e.g., from upstream signal processing, deinterleaving, and temporal models),
NOT ten separate neural network models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence, Tuple
import numpy as np
from gymnasium import spaces


@dataclass
class EmitterFeatures:
    """10 scalar estimation features representing one emitter track."""

    activity_prob: float = 0.0          # Predicted probability of active transmission [0, 1]
    threat_level: float = 0.0           # Lethality / priority score [0, 1]
    identity_confidence: float = 0.0    # Emitter classification confidence [0, 1]
    uncertainty: float = 1.0            # Parameter uncertainty / entropy [0, 1]
    novelty: float = 0.0                # Anomaly / library distance score [0, 1]
    track_age: float = 0.0              # Normalized time since last observation [0, 1]
    mode: float = 0.0                   # Operational mode index [0, 1]
    potential_info_gain: float = 0.0    # Ex-ante expected uncertainty reduction [0, 1]
    observation_cost: float = 0.1       # Dwell resource / time cost fraction [0, 1]
    miss_risk: float = 0.0              # Threat * activity_prob * uncertainty [0, 1]

    def to_array(self) -> np.ndarray:
        """Export as a 10-dimensional numpy float32 vector clipped to [0, 1]."""
        vec = np.array([
            self.activity_prob,
            self.threat_level,
            self.identity_confidence,
            self.uncertainty,
            self.novelty,
            self.track_age,
            self.mode,
            self.potential_info_gain,
            self.observation_cost,
            self.miss_risk,
        ], dtype=np.float32)
        return np.clip(vec, 0.0, 1.0)

    @classmethod
    def from_array(cls, arr: Sequence[float]) -> EmitterFeatures:
        """Create instance from a 10-element sequence."""
        if len(arr) != 10:
            raise ValueError(f"Expected 10 features for emitter, got {len(arr)}")
        return cls(
            activity_prob=float(arr[0]),
            threat_level=float(arr[1]),
            identity_confidence=float(arr[2]),
            uncertainty=float(arr[3]),
            novelty=float(arr[4]),
            track_age=float(arr[5]),
            mode=float(arr[6]),
            potential_info_gain=float(arr[7]),
            observation_cost=float(arr[8]),
            miss_risk=float(arr[9]),
        )


@dataclass
class GlobalSensorFeatures:
    """4 global receiver / mission state features."""

    sensor_utilization: float = 0.0         # Fraction of receiver capacity utilized [0, 1]
    remaining_budget_fraction: float = 1.0  # Remaining dwell resource fraction [0, 1]
    normalized_step: float = 0.0            # Episode step / max_steps [0, 1]
    active_threat_ratio: float = 0.0        # Ratio of high-threat emitters active [0, 1]

    def to_array(self) -> np.ndarray:
        """Export as a 4-dimensional numpy float32 vector clipped to [0, 1]."""
        vec = np.array([
            self.sensor_utilization,
            self.remaining_budget_fraction,
            self.normalized_step,
            self.active_threat_ratio,
        ], dtype=np.float32)
        return np.clip(vec, 0.0, 1.0)

    @classmethod
    def from_array(cls, arr: Sequence[float]) -> GlobalSensorFeatures:
        if len(arr) != 4:
            raise ValueError(f"Expected 4 global features, got {len(arr)}")
        return cls(
            sensor_utilization=float(arr[0]),
            remaining_budget_fraction=float(arr[1]),
            normalized_step=float(arr[2]),
            active_threat_ratio=float(arr[3]),
        )


class RLStateBuilder:
    """Constructs, validates, and parses the 84-dimensional RL state vector."""

    FEATURES_PER_EMITTER: int = 10
    GLOBAL_FEATURES_DIM: int = 4

    def __init__(self, num_emitters: int = 8) -> None:
        self.num_emitters = num_emitters
        self.total_dim = self.num_emitters * self.FEATURES_PER_EMITTER + self.GLOBAL_FEATURES_DIM

        # Standard Gymnasium Box space bounded in [0.0, 1.0]
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.total_dim,),
            dtype=np.float32,
        )

    def build_state_vector(
        self,
        emitters: Sequence[EmitterFeatures],
        global_features: GlobalSensorFeatures,
    ) -> np.ndarray:
        """Assemble the 84-dimensional observation vector."""
        if len(emitters) != self.num_emitters:
            raise ValueError(
                f"Expected {self.num_emitters} emitter states, got {len(emitters)}"
            )

        emitter_vecs = [e.to_array() for e in emitters]
        flattened_emitters = np.concatenate(emitter_vecs, dtype=np.float32)
        global_vec = global_features.to_array()

        obs = np.concatenate([flattened_emitters, global_vec], dtype=np.float32)
        return np.clip(obs, 0.0, 1.0)

    def parse_state_vector(
        self,
        obs: np.ndarray,
    ) -> Tuple[List[EmitterFeatures], GlobalSensorFeatures]:
        """Unpack an 84-dimensional vector into structured dataclasses."""
        if obs.shape != (self.total_dim,):
            raise ValueError(f"Expected shape ({self.total_dim},), got {obs.shape}")

        emitters: List[EmitterFeatures] = []
        emitter_end = self.num_emitters * self.FEATURES_PER_EMITTER

        for i in range(self.num_emitters):
            start = i * self.FEATURES_PER_EMITTER
            end = start + self.FEATURES_PER_EMITTER
            emitters.append(EmitterFeatures.from_array(obs[start:end]))

        global_features = GlobalSensorFeatures.from_array(obs[emitter_end:])
        return emitters, global_features

    def to_dict(self, obs: np.ndarray) -> Dict[str, Any]:
        """Convert state vector into a JSON/telemetry friendly dictionary."""
        emitters, global_feat = self.parse_state_vector(obs)
        return {
            "emitters": [
                {
                    "id": i,
                    "activity_prob": float(e.activity_prob),
                    "threat_level": float(e.threat_level),
                    "identity_confidence": float(e.identity_confidence),
                    "uncertainty": float(e.uncertainty),
                    "novelty": float(e.novelty),
                    "track_age": float(e.track_age),
                    "mode": float(e.mode),
                    "potential_info_gain": float(e.potential_info_gain),
                    "observation_cost": float(e.observation_cost),
                    "miss_risk": float(e.miss_risk),
                }
                for i, e in enumerate(emitters)
            ],
            "global": {
                "sensor_utilization": float(global_feat.sensor_utilization),
                "remaining_budget_fraction": float(global_feat.remaining_budget_fraction),
                "normalized_step": float(global_feat.normalized_step),
                "active_threat_ratio": float(global_feat.active_threat_ratio),
            },
        }
