"""Unit tests for deinterleaving algorithms, PRI analysis, and evaluation metrics."""

import sys
from pathlib import Path
import numpy as np
import pytest

# Ensure project root is on sys.path
PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from deinterleaving.dbscan.clustering import SpatialSpectralDBSCAN
from deinterleaving.sedcam.pri_transform import PRITransformer, PRIAnalysisResult
from deinterleaving.emitter_tracker import EmitterTracker
from evaluation.deinterleaving_metrics.metrics import (
    evaluate_deinterleaving,
    pairwise_f1_and_mcc,
)


def test_spatial_spectral_dbscan():
    # Construct synthetic 2-emitter interleaved stream
    # Emitter 0: AoA=45 deg, RF=9400 MHz, PW=1.0 us
    # Emitter 1: AoA=180 deg, RF=3200 MHz, PW=5.0 us
    rng = np.random.default_rng(42)
    n0, n1 = 50, 60

    e0 = np.column_stack([
        np.sort(rng.uniform(0, 10000, n0)),
        rng.normal(9400.0, 5.0, n0),
        rng.normal(1.0, 0.05, n0),
        rng.normal(45.0, 1.0, n0),
        rng.normal(-30.0, 2.0, n0),
    ])

    e1 = np.column_stack([
        np.sort(rng.uniform(0, 10000, n1)),
        rng.normal(3200.0, 5.0, n1),
        rng.normal(5.0, 0.05, n1),
        rng.normal(180.0, 1.0, n1),
        rng.normal(-25.0, 2.0, n1),
    ])

    all_pdws = np.vstack([e0, e1])
    # Sort chronologically by ToA
    all_pdws = all_pdws[np.argsort(all_pdws[:, 0])]

    clusterer = SpatialSpectralDBSCAN(eps=0.10, min_samples=10)
    labels = clusterer.fit_predict(all_pdws)

    unique_clusters = set(labels) - {-1}
    # Should identify 2 distinct clusters
    assert len(unique_clusters) == 2

    summaries = clusterer.summarize_clusters(all_pdws, labels=labels)
    assert len(summaries) == 2
    # Verify frequencies match the two synthetic emitters
    freqs = sorted([s.mean_freq_mhz for s in summaries])
    assert freqs[0] < 5000.0
    assert freqs[1] > 8000.0


def test_pri_transformer_fixed_pri():
    # Periodic pulse train with fixed PRI = 250 us
    pri = 250.0
    toa = np.arange(0, 50) * pri + np.random.normal(0, 0.5, 50)

    analyzer = PRITransformer(min_pri_us=50.0, max_pri_us=1000.0)
    result = analyzer.analyze_sequence(toa)

    assert pytest.approx(result.estimated_pri_us, rel=0.05) == 250.0
    assert result.pri_type in ("fixed", "jittered")
    assert result.confidence > 0.5


def test_emitter_tracker_pipeline():
    # Interleaved pulses
    pdws = np.zeros((100, 5), dtype=np.float32)
    pdws[:50, 0] = np.arange(50) * 100.0   # PRI 100 us
    pdws[:50, 1] = 9400.0                  # 9.4 GHz
    pdws[:50, 2] = 1.0                     # 1 us
    pdws[:50, 3] = 45.0                    # AoA 45 deg
    pdws[:50, 4] = -30.0

    pdws[50:, 0] = np.arange(50) * 200.0   # PRI 200 us
    pdws[50:, 1] = 3000.0                  # 3 GHz
    pdws[50:, 2] = 5.0                     # 5 us
    pdws[50:, 3] = 180.0                   # AoA 180 deg
    pdws[50:, 4] = -20.0

    # Sort by ToA
    pdws = pdws[np.argsort(pdws[:, 0])]

    tracker = EmitterTracker()
    upstream_records = tracker.process_pulse_train(pdws)

    assert len(upstream_records) >= 2
    for r in upstream_records:
        assert r.threat_level > 0.0
        assert r.carrier_freq_mhz is not None
        assert r.pri_us is not None


def test_evaluate_deinterleaving():
    true_labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    pred_labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])

    scores = evaluate_deinterleaving(true_labels, pred_labels)
    assert pytest.approx(scores["adjusted_rand_index"]) == 1.0
    assert pytest.approx(scores["v_measure"]) == 1.0
    assert pytest.approx(scores["pairwise_f1"]) == 1.0
    assert pytest.approx(scores["pairwise_mcc"]) == 1.0
