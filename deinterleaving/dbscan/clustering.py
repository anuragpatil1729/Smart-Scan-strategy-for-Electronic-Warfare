"""Spatial-Spectral Pulse Deinterleaving via DBSCAN.

Clusters interleaved radar pulses based on spatial Angle of Arrival (AoA),
carrier frequency, and pulse width to perform coarse emitter separation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
from sklearn.cluster import DBSCAN

from pdw.features.pdw_features import PDWFeatureScaler


@dataclass
class ClusterSummary:
    """Summary statistics for a deinterleaved pulse cluster."""

    cluster_id: int
    num_pulses: int
    mean_aoa_deg: float
    std_aoa_deg: float
    mean_freq_mhz: float
    std_freq_mhz: float
    mean_pw_us: float
    std_pw_us: float
    mean_amplitude_db: float
    indices: np.ndarray


class SpatialSpectralDBSCAN:
    """Multi-parameter DBSCAN for coarse radar pulse deinterleaving."""

    def __init__(
        self,
        eps: float = 0.08,
        min_samples: int = 5,
        feature_weights: Optional[Dict[str, float]] = None,
        scaler: Optional[PDWFeatureScaler] = None,
    ) -> None:
        self.eps = eps
        self.min_samples = min_samples
        self.feature_weights = feature_weights or {"aoa": 1.5, "frequency": 1.0, "pulse_width": 1.0}
        self.scaler = scaler or PDWFeatureScaler()

        self.model = DBSCAN(eps=self.eps, min_samples=self.min_samples, metric="euclidean", n_jobs=-1)
        self.labels_: Optional[np.ndarray] = None

    def fit_predict(self, pdws: np.ndarray) -> np.ndarray:
        """Cluster pulses based on normalized [AoA, Frequency, PulseWidth].

        Args:
            pdws: Array of shape (N, 5) [ToA, Frequency, PulseWidth, AoA, Amplitude]

        Returns:
            Cluster labels of shape (N,), where -1 represents unclustered noise.
        """
        features = self.scaler.extract_spatial_spectral(pdws, weights=self.feature_weights)
        self.labels_ = self.model.fit_predict(features)
        return self.labels_

    def summarize_clusters(self, pdws: np.ndarray, labels: Optional[np.ndarray] = None) -> List[ClusterSummary]:
        """Compute statistical summary for each identified cluster (excluding noise)."""
        lbls = labels if labels is not None else self.labels_
        if lbls is None:
            raise ValueError("Model has not been fitted yet.")

        unique_clusters = [c for c in np.unique(lbls) if c != -1]
        summaries: List[ClusterSummary] = []

        for c in unique_clusters:
            idx = np.where(lbls == c)[0]
            cluster_pdws = pdws[idx]

            summary = ClusterSummary(
                cluster_id=int(c),
                num_pulses=int(len(idx)),
                mean_aoa_deg=float(np.mean(cluster_pdws[:, 3])),
                std_aoa_deg=float(np.std(cluster_pdws[:, 3])),
                mean_freq_mhz=float(np.mean(cluster_pdws[:, 1])),
                std_freq_mhz=float(np.std(cluster_pdws[:, 1])),
                mean_pw_us=float(np.mean(cluster_pdws[:, 2])),
                std_pw_us=float(np.std(cluster_pdws[:, 2])),
                mean_amplitude_db=float(np.mean(cluster_pdws[:, 4])),
                indices=idx,
            )
            summaries.append(summary)

        # Sort clusters by pulse count descending
        summaries.sort(key=lambda s: s.num_pulses, reverse=True)
        return summaries
