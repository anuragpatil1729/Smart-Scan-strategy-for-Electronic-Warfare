"""Pulse Repetition Interval (PRI) Analysis and Transformation.

Analyzes arrival times (ToA) of deinterleaved pulse sequences to extract
fundamental PRIs, detect stagger/jitter modulations, and validate periodic pulse trains.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np
from scipy.signal import find_peaks


@dataclass
class PRIAnalysisResult:
    """Results of temporal PRI analysis on a pulse sequence."""

    estimated_pri_us: float                     # Fundamental estimated PRI (microseconds)
    candidate_pris: List[float] = field(default_factory=list)  # Secondary or staggered PRIs
    pri_type: str = "unknown"                   # "fixed", "staggered", "jittered", "agile"
    confidence: float = 0.0                     # Confidence in [0.0, 1.0]
    jitter_percentage: float = 0.0              # Estimated percentage jitter


class PRITransformer:
    """Estimates Pulse Repetition Intervals from pulse arrival times."""

    def __init__(
        self,
        min_pri_us: float = 10.0,
        max_pri_us: float = 5000.0,
        num_bins: int = 500,
        peak_prominence: float = 0.1,
    ) -> None:
        self.min_pri_us = min_pri_us
        self.max_pri_us = max_pri_us
        self.num_bins = num_bins
        self.peak_prominence = peak_prominence

    def analyze_sequence(self, toa: np.ndarray) -> PRIAnalysisResult:
        """Estimate PRI and modulation characteristics from a sequence of ToA values."""
        if len(toa) < 4:
            return PRIAnalysisResult(estimated_pri_us=0.0, pri_type="insufficient_pulses", confidence=0.0)

        # Sort ToA chronologically
        sorted_toa = np.sort(toa)

        # Compute consecutive first-order delta-ToA
        first_diffs = np.diff(sorted_toa)
        valid_diffs = first_diffs[(first_diffs >= self.min_pri_us) & (first_diffs <= self.max_pri_us)]

        if len(valid_diffs) < 3:
            # Fallback to median difference
            med = float(np.median(first_diffs)) if len(first_diffs) > 0 else 0.0
            return PRIAnalysisResult(estimated_pri_us=med, pri_type="irregular", confidence=0.1)

        # Build histogram of delta-ToA
        counts, bin_edges = np.histogram(valid_diffs, bins=self.num_bins, range=(self.min_pri_us, self.max_pri_us))
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        # Normalize histogram to [0, 1]
        max_count = np.max(counts)
        norm_counts = counts / max_count if max_count > 0 else counts

        # Detect peaks
        peaks, properties = find_peaks(norm_counts, height=0.15, prominence=self.peak_prominence, distance=5)

        if len(peaks) == 0:
            # No clear peaks; estimate via median
            med_pri = float(np.median(valid_diffs))
            return PRIAnalysisResult(estimated_pri_us=med_pri, pri_type="jittered", confidence=0.35)

        # Peak PRIs sorted by prominence/height descending
        peak_heights = norm_counts[peaks]
        sorted_peak_indices = np.argsort(peak_heights)[::-1]
        sorted_peaks = peaks[sorted_peak_indices]

        candidate_pris = [float(bin_centers[p]) for p in sorted_peaks]
        primary_pri = candidate_pris[0]

        # Determine PRI modulation type
        if len(sorted_peaks) == 1:
            # Single narrow peak -> Fixed PRI
            # Check spread around peak
            peak_bin = sorted_peaks[0]
            local_std = np.std(valid_diffs[np.abs(valid_diffs - primary_pri) < primary_pri * 0.1])
            jitter_pct = float(local_std / primary_pri * 100.0) if primary_pri > 0 else 0.0
            pri_type = "fixed" if jitter_pct < 3.0 else "jittered"
            confidence = 0.90 if pri_type == "fixed" else 0.75
        elif len(sorted_peaks) in (2, 3, 4):
            # Multiple distinct peaks -> Staggered PRI
            pri_type = "staggered"
            jitter_pct = 0.0
            confidence = 0.85
        else:
            pri_type = "agile"
            jitter_pct = 15.0
            confidence = 0.60

        return PRIAnalysisResult(
            estimated_pri_us=primary_pri,
            candidate_pris=candidate_pris,
            pri_type=pri_type,
            confidence=confidence,
            jitter_percentage=jitter_pct,
        )
