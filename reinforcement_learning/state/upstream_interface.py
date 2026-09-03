"""Upstream Interface for Real Emitter Data Ingestion.

Provides a clean, strongly-typed interface for ingesting emitter representations
produced by upstream signal processing, PDW deinterleaving, and temporal models,
and mapping them into the standardized 84-dimensional RL state vector.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Union
import numpy as np

from reinforcement_learning.state.state_space import (
    EmitterFeatures,
    GlobalSensorFeatures,
    RLStateBuilder,
)


@dataclass
class UpstreamEmitterRecord:
    """Represents an emitter track received from upstream signal processing.

    Attributes:
        emitter_id: Unique identifier from deinterleaving / emitter database.
        threat_level: Lethality / priority score in [0.0, 1.0].
        activity_prob: Upstream temporal model predicted probability of emission in [0.0, 1.0].
        identity_confidence: Classifier confidence in emitter type in [0.0, 1.0].
        uncertainty: Parameter entropy or normalized covariance trace in [0.0, 1.0].
        novelty_score: Distance metric from known threat library in [0.0, 1.0].
        last_intercept_timestamp: Wall clock or simulation time of last pulse intercept (seconds).
        operational_mode: Mode string ('idle', 'search', 'acquisition', 'track', 'guidance') or float in [0.0, 1.0].
        dwell_cost: Normalized dwell duration/energy cost in [0.0, 1.0].
        carrier_freq_mhz: Optional estimated carrier frequency (MHz).
        pri_us: Optional estimated pulse repetition interval (microseconds).
        snr_db: Optional signal-to-noise ratio (dB) from last dwell.
    """

    emitter_id: Union[str, int]
    threat_level: float = 0.5
    activity_prob: float = 0.5
    identity_confidence: float = 0.5
    uncertainty: float = 0.5
    novelty_score: float = 0.0
    last_intercept_timestamp: float = 0.0
    operational_mode: Union[str, float] = 0.25
    dwell_cost: float = 0.1
    carrier_freq_mhz: Optional[float] = None
    pri_us: Optional[float] = None
    snr_db: Optional[float] = None

    def get_normalized_mode(self) -> float:
        """Map categorical or float mode to [0.0, 1.0]."""
        if isinstance(self.operational_mode, (int, float)):
            return float(np.clip(self.operational_mode, 0.0, 1.0))

        mode_str = str(self.operational_mode).strip().lower()
        mode_mapping = {
            "idle": 0.0,
            "search": 0.25,
            "acquisition": 0.50,
            "track": 0.75,
            "guidance": 1.0,
            "lock": 1.0,
        }
        return mode_mapping.get(mode_str, 0.25)


@dataclass
class UpstreamScanContext:
    """Current sensor platform status and global mission context."""

    current_timestamp: float = 0.0          # Current simulation/mission time (seconds)
    current_step: int = 0                   # Current discrete scheduling step
    max_steps: int = 100                    # Total episode horizon
    sensor_utilization: float = 0.0         # Sensor load fraction in [0.0, 1.0]
    remaining_budget_fraction: float = 1.0  # Remaining energy/dwell budget in [0.0, 1.0]
    drop_timeout_seconds: float = 10.0      # Revisit latency after which track drops


class UpstreamStateAdapter:
    """Adapter bridging real upstream emitter tracks to the 84-dimensional RL state space."""

    def __init__(self, num_emitters: int = 8, default_drop_timeout: float = 10.0) -> None:
        self.num_emitters = num_emitters
        self.default_drop_timeout = max(1e-3, default_drop_timeout)
        self.state_builder = RLStateBuilder(num_emitters=num_emitters)

    def convert_to_observation(
        self,
        upstream_records: Sequence[UpstreamEmitterRecord],
        context: Optional[UpstreamScanContext] = None,
    ) -> np.ndarray:
        """Ingest upstream emitter tracks and return the validated 84-dimensional state vector.

        If fewer than `num_emitters` tracks are provided, pads with inactive/zeroed slots.
        If more than `num_emitters` tracks are provided, selects the top tracks prioritized by
        composite risk: threat_level * (activity_prob + uncertainty).
        """
        if context is None:
            context = UpstreamScanContext()

        # Prioritize tracks if there are more than num_emitters
        records = list(upstream_records)
        if len(records) > self.num_emitters:
            records.sort(
                key=lambda r: float(r.threat_level) * (float(r.activity_prob) + float(r.uncertainty)),
                reverse=True,
            )
            records = records[: self.num_emitters]

        emitter_features_list: List[EmitterFeatures] = []

        for record in records:
            # 1. Activity probability [0, 1]
            act_prob = float(np.clip(record.activity_prob, 0.0, 1.0))

            # 2. Threat level [0, 1]
            threat = float(np.clip(record.threat_level, 0.0, 1.0))

            # 3. Identity confidence [0, 1]
            conf = float(np.clip(record.identity_confidence, 0.0, 1.0))

            # 4. Uncertainty [0, 1]
            unc = float(np.clip(record.uncertainty, 0.0, 1.0))

            # 5. Novelty score [0, 1]
            nov = float(np.clip(record.novelty_score, 0.0, 1.0))

            # 6. Track age / revisit urgency [0, 1]:
            # Informational state feature based on time elapsed since last observation
            dt = max(0.0, context.current_timestamp - record.last_intercept_timestamp)
            drop_timeout = max(1e-3, context.drop_timeout_seconds)
            track_age = float(np.clip(dt / drop_timeout, 0.0, 1.0))

            # 7. Normalized mode [0, 1]
            mode = record.get_normalized_mode()

            # 8. Potential Information Gain (EX-ANTE ESTIMATE in state, NOT actual reward):
            # Prior expectation of uncertainty reduction if dwell is allocated.
            # Scales with current uncertainty and estimated SNR if available.
            snr_factor = 1.0
            if record.snr_db is not None:
                snr_factor = float(np.clip(record.snr_db / 20.0, 0.2, 1.0))
            potential_info_gain = float(np.clip(unc * snr_factor, 0.0, 1.0))

            # 9. Observation dwell cost [0, 1]
            obs_cost = float(np.clip(record.dwell_cost, 0.0, 1.0))

            # 10. Miss risk [0, 1]
            miss_risk = float(np.clip(threat * act_prob * unc, 0.0, 1.0))

            features = EmitterFeatures(
                activity_prob=act_prob,
                threat_level=threat,
                identity_confidence=conf,
                uncertainty=unc,
                novelty=nov,
                track_age=track_age,
                mode=mode,
                potential_info_gain=potential_info_gain,
                observation_cost=obs_cost,
                miss_risk=miss_risk,
            )
            emitter_features_list.append(features)

        # Pad with inactive dummy slots if fewer than num_emitters
        while len(emitter_features_list) < self.num_emitters:
            emitter_features_list.append(
                EmitterFeatures(
                    activity_prob=0.0,
                    threat_level=0.0,
                    identity_confidence=0.0,
                    uncertainty=0.0,
                    novelty=0.0,
                    track_age=0.0,
                    mode=0.0,
                    potential_info_gain=0.0,
                    observation_cost=0.0,
                    miss_risk=0.0,
                )
            )

        # Global features
        step_frac = float(np.clip(context.current_step / max(1, context.max_steps), 0.0, 1.0))
        high_threats = sum(1 for e in emitter_features_list if e.threat_level >= 0.7 and e.activity_prob >= 0.5)
        active_threat_ratio = float(high_threats / max(1, self.num_emitters))

        global_features = GlobalSensorFeatures(
            sensor_utilization=float(np.clip(context.sensor_utilization, 0.0, 1.0)),
            remaining_budget_fraction=float(np.clip(context.remaining_budget_fraction, 0.0, 1.0)),
            normalized_step=step_frac,
            active_threat_ratio=float(np.clip(active_threat_ratio, 0.0, 1.0)),
        )

        obs = self.state_builder.build_state_vector(emitter_features_list, global_features)
        return obs

    def validate_observation(self, obs: np.ndarray) -> bool:
        """Validate shape, type, and value bounds of the observation vector."""
        if not isinstance(obs, np.ndarray):
            return False
        if obs.shape != (self.state_builder.total_dim,):
            return False
        if np.isnan(obs).any() or np.isinf(obs).any():
            return False
        if (obs < 0.0).any() or (obs > 1.0).any():
            return False
        return True

    def parse_observation(self, obs: np.ndarray) -> Dict[str, Any]:
        """Convert an 84-dimensional vector into an interpretable diagnostic dictionary."""
        return self.state_builder.to_dict(obs)
