"""Emitter Tracker and Bridge to RL Upstream Interface.

Converts deinterleaved pulse clusters into tracked emitter profiles and maps them
directly into `UpstreamEmitterRecord` objects for the RL Smart Scan subsystem.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np

from deinterleaving.dbscan.clustering import SpatialSpectralDBSCAN, ClusterSummary
from deinterleaving.sedcam.pri_transform import PRITransformer, PRIAnalysisResult
from reinforcement_learning.state.upstream_interface import UpstreamEmitterRecord


@dataclass
class EmitterTrack:
    """Estimated parameters for a tracked emitter derived from deinterleaved pulses."""

    track_id: str
    num_pulses: int
    mean_freq_mhz: float
    std_freq_mhz: float
    mean_pw_us: float
    std_pw_us: float
    mean_aoa_deg: float
    std_aoa_deg: float
    mean_amplitude_db: float
    estimated_pri_us: float
    pri_type: str
    pri_confidence: float
    last_intercept_us: float
    threat_level: float = 0.5
    activity_prob: float = 0.5
    uncertainty: float = 0.5
    identity_confidence: float = 0.5

    def to_upstream_record(self) -> UpstreamEmitterRecord:
        """Map estimated emitter parameters into RL UpstreamEmitterRecord."""
        # Map operational mode heuristic
        if self.estimated_pri_us < 50.0 or self.mean_pw_us < 1.0:
            mode = "track" if self.threat_level >= 0.7 else "acquisition"
        else:
            mode = "search"

        return UpstreamEmitterRecord(
            emitter_id=self.track_id,
            threat_level=float(np.clip(self.threat_level, 0.0, 1.0)),
            activity_prob=float(np.clip(self.activity_prob, 0.0, 1.0)),
            identity_confidence=float(np.clip(self.identity_confidence, 0.0, 1.0)),
            uncertainty=float(np.clip(self.uncertainty, 0.0, 1.0)),
            novelty_score=0.10,
            last_intercept_timestamp=self.last_intercept_us * 1e-6,  # convert to seconds
            operational_mode=mode,
            dwell_cost=0.10,
            carrier_freq_mhz=self.mean_freq_mhz,
            pri_us=self.estimated_pri_us,
            snr_db=self.mean_amplitude_db,
        )


class EmitterTracker:
    """End-to-end deinterleaver and emitter tracking engine."""

    def __init__(
        self,
        clusterer: Optional[SpatialSpectralDBSCAN] = None,
        pri_analyzer: Optional[PRITransformer] = None,
    ) -> None:
        self.clusterer = clusterer or SpatialSpectralDBSCAN()
        self.pri_analyzer = pri_analyzer or PRITransformer()

    def process_pulse_train(
        self,
        pdws: np.ndarray,
        return_tracks: bool = False,
    ) -> List[UpstreamEmitterRecord]:
        """Run full deinterleaving pipeline on raw 5-D PDW stream.

        Args:
            pdws: Array of shape (N, 5) [ToA, Frequency, PulseWidth, AoA, Amplitude]
            return_tracks: If True, returns internal EmitterTrack objects instead

        Returns:
            List of UpstreamEmitterRecord instances ready for the RL State Adapter.
        """
        if len(pdws) == 0:
            return []

        # 1. Coarse Spatial-Spectral Clustering
        labels = self.clusterer.fit_predict(pdws)
        cluster_summaries = self.clusterer.summarize_clusters(pdws, labels=labels)

        tracks: List[EmitterTrack] = []
        duration_us = float(pdws[-1, 0] - pdws[0, 0]) if len(pdws) > 1 else 1e6

        for summary in cluster_summaries:
            cluster_pdws = pdws[summary.indices]

            # 2. Fine Temporal PRI Analysis
            pri_result = self.pri_analyzer.analyze_sequence(cluster_pdws[:, 0])

            # 3. Estimate Threat Level (radar heuristic):
            # X-band (8-12 GHz) with short PW / fast PRI is typical missile/fire-control (High Threat)
            # S/C band (2-6 GHz) with medium PRI is typical acquisition/surveillance (Medium Threat)
            # UHF/L band (< 2 GHz) long range early warning (Low/Medium Threat)
            threat = 0.5
            freq_ghz = summary.mean_freq_mhz / 1000.0
            if freq_ghz >= 8.0:
                threat = 0.85 if summary.mean_pw_us < 2.0 else 0.70
            elif freq_ghz >= 4.0:
                threat = 0.60
            else:
                threat = 0.35

            # Agile / staggered modulations elevate threat
            if pri_result.pri_type in ("staggered", "agile"):
                threat = min(1.0, threat + 0.15)

            # Activity probability estimated from burst pulse density
            cluster_duration = float(cluster_pdws[-1, 0] - cluster_pdws[0, 0]) if len(cluster_pdws) > 1 else 1.0
            duty_cycle = (summary.num_pulses * summary.mean_pw_us) / max(1.0, cluster_duration)
            activity_prob = float(np.clip(duty_cycle * 100.0 + 0.3, 0.1, 0.95))

            # Parameter uncertainty derived from frequency and PRI jitter
            rel_freq_std = summary.std_freq_mhz / max(1.0, summary.mean_freq_mhz)
            rel_pri_std = pri_result.jitter_percentage / 100.0
            uncertainty = float(np.clip(0.5 * rel_freq_std * 10.0 + 0.5 * rel_pri_std + (1.0 - pri_result.confidence) * 0.5, 0.05, 0.95))

            track = EmitterTrack(
                track_id=f"track_{summary.cluster_id}",
                num_pulses=summary.num_pulses,
                mean_freq_mhz=summary.mean_freq_mhz,
                std_freq_mhz=summary.std_freq_mhz,
                mean_pw_us=summary.mean_pw_us,
                std_pw_us=summary.std_pw_us,
                mean_aoa_deg=summary.mean_aoa_deg,
                std_aoa_deg=summary.std_aoa_deg,
                mean_amplitude_db=summary.mean_amplitude_db,
                estimated_pri_us=pri_result.estimated_pri_us,
                pri_type=pri_result.pri_type,
                pri_confidence=pri_result.confidence,
                last_intercept_us=float(cluster_pdws[-1, 0]),
                threat_level=threat,
                activity_prob=activity_prob,
                uncertainty=uncertainty,
                identity_confidence=pri_result.confidence,
            )
            tracks.append(track)

        if return_tracks:
            return tracks  # type: ignore[return-value]

        # Convert to UpstreamEmitterRecord
        return [t.to_upstream_record() for t in tracks]
