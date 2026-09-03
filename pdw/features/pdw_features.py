"""PDW Feature Extraction, Scaling, and Differential Features."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np


def compute_delta_toa(toa: np.ndarray) -> np.ndarray:
    """Compute first differences of arrival times (Delta-ToA).

    Delta_ToA[k] = ToA[k] - ToA[k-1] for k >= 1.
    First element is set to 0.0.
    """
    if len(toa) <= 1:
        return np.zeros_like(toa)

    delta = np.empty_like(toa)
    delta[0] = 0.0
    delta[1:] = np.diff(toa)
    # Clip negative values if any pulses are out of order
    return np.maximum(0.0, delta)


@dataclass
class PDWFeatureScaler:
    """Scales 5-D PDWs into normalized [0, 1] ranges for distance-based clustering."""

    max_freq_mhz: float = 18000.0   # 18 GHz max RF
    max_pw_us: float = 100.0        # 100 us max pulse width
    max_aoa_deg: float = 360.0      # 360 degrees
    min_amp_db: float = -100.0      # -100 dB minimum sensitivity
    max_amp_db: float = 20.0        # +20 dB maximum saturation

    def extract_spatial_spectral(
        self,
        pdws: np.ndarray,
        weights: Optional[Dict[str, float]] = None,
    ) -> np.ndarray:
        """Extract and normalize [AoA, Frequency, PulseWidth] for clustering.

        Args:
            pdws: Array of shape (N, 5) [ToA, Frequency, PulseWidth, AoA, Amplitude]
            weights: Optional feature weight dictionary
        """
        w = weights or {"aoa": 1.5, "frequency": 1.0, "pulse_width": 1.0}

        freq_norm = np.clip(pdws[:, 1] / self.max_freq_mhz, 0.0, 1.0) * w.get("frequency", 1.0)
        pw_norm = np.clip(pdws[:, 2] / self.max_pw_us, 0.0, 1.0) * w.get("pulse_width", 1.0)
        aoa_norm = np.clip(pdws[:, 3] / self.max_aoa_deg, 0.0, 1.0) * w.get("aoa", 1.5)

        # Stack into shape (N, 3)
        return np.column_stack([aoa_norm, freq_norm, pw_norm]).astype(np.float32)

    def extract_all_normalized(self, pdws: np.ndarray) -> np.ndarray:
        """Normalize all 5 PDW channels."""
        toa_span = max(1.0, float(pdws[-1, 0] - pdws[0, 0])) if len(pdws) > 1 else 1.0
        toa_norm = (pdws[:, 0] - pdws[0, 0]) / toa_span
        freq_norm = np.clip(pdws[:, 1] / self.max_freq_mhz, 0.0, 1.0)
        pw_norm = np.clip(pdws[:, 2] / self.max_pw_us, 0.0, 1.0)
        aoa_norm = np.clip(pdws[:, 3] / self.max_aoa_deg, 0.0, 1.0)
        amp_span = max(1.0, self.max_amp_db - self.min_amp_db)
        amp_norm = np.clip((pdws[:, 4] - self.min_amp_db) / amp_span, 0.0, 1.0)

        return np.column_stack([toa_norm, freq_norm, pw_norm, aoa_norm, amp_norm]).astype(np.float32)


def compute_pulse_stream_stats(pdws: np.ndarray) -> Dict[str, float]:
    """Compute summary statistics for a pulse train stream."""
    if len(pdws) == 0:
        return {}

    duration_us = float(pdws[-1, 0] - pdws[0, 0]) if len(pdws) > 1 else 0.0
    pulse_rate_pps = float(len(pdws) / (duration_us * 1e-6)) if duration_us > 0 else 0.0

    return {
        "num_pulses": int(len(pdws)),
        "duration_sec": duration_us * 1e-6,
        "mean_pulse_rate_pps": pulse_rate_pps,
        "mean_freq_mhz": float(np.mean(pdws[:, 1])),
        "std_freq_mhz": float(np.std(pdws[:, 1])),
        "mean_pw_us": float(np.mean(pdws[:, 2])),
        "mean_amplitude_db": float(np.mean(pdws[:, 4])),
    }
