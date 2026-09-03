"""Unit tests for PDW extraction, dataset reader, and feature scaling."""

import sys
from pathlib import Path
import numpy as np
import pytest

# Ensure project root is on sys.path
PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from pdw.extraction.dataset_reader import TSRDDatasetReader, PulseTrainSample
from pdw.features.pdw_features import (
    PDWFeatureScaler,
    compute_delta_toa,
    compute_pulse_stream_stats,
)

DATASET_PATH = Path("datasets/synthetic/turing_radar_data")


@pytest.mark.skipif(not DATASET_PATH.exists(), reason="TSRD dataset directory not found")
def test_tsrd_reader_list_files():
    reader = TSRDDatasetReader(root_path=DATASET_PATH)
    files = reader.list_files("train_scan")
    assert len(files) > 0
    assert files[0].name.endswith(".h5")


@pytest.mark.skipif(not DATASET_PATH.exists(), reason="TSRD dataset directory not found")
def test_tsrd_reader_load_sample():
    reader = TSRDDatasetReader(root_path=DATASET_PATH)
    sample = reader.load_sample(split="train_scan", index_or_filename=0, max_pulses=500)

    assert isinstance(sample, PulseTrainSample)
    assert sample.pdws.shape == (500, 5)
    assert sample.labels.shape == (500,)
    assert sample.num_pulses == 500
    assert sample.num_emitters > 0

    # Test channel accessors
    assert len(sample.toa) == 500
    assert len(sample.frequency) == 500
    assert len(sample.pulse_width) == 500
    assert len(sample.aoa) == 500
    assert len(sample.amplitude) == 500


def test_compute_delta_toa():
    toa = np.array([10.0, 25.0, 45.0, 70.0], dtype=np.float32)
    delta = compute_delta_toa(toa)
    np.testing.assert_allclose(delta, [0.0, 15.0, 20.0, 25.0])


def test_pdw_feature_scaler():
    scaler = PDWFeatureScaler()
    mock_pdws = np.array([
        [100.0, 9400.0, 1.5, 45.0, -40.0],
        [150.0, 3200.0, 5.0, 180.0, -20.0],
    ], dtype=np.float32)

    features = scaler.extract_spatial_spectral(mock_pdws)
    assert features.shape == (2, 3)
    assert (features >= 0.0).all() and (features <= 2.0).all()

    all_norm = scaler.extract_all_normalized(mock_pdws)
    assert all_norm.shape == (2, 5)
    assert (all_norm >= 0.0).all() and (all_norm <= 1.0).all()


def test_pulse_stream_stats():
    mock_pdws = np.array([
        [0.0, 9400.0, 1.0, 30.0, -30.0],
        [1000.0, 9400.0, 1.0, 30.0, -30.0],
        [2000.0, 9400.0, 1.0, 30.0, -30.0],
    ], dtype=np.float32)

    stats = compute_pulse_stream_stats(mock_pdws)
    assert stats["num_pulses"] == 3
    assert pytest.approx(stats["mean_freq_mhz"]) == 9400.0
