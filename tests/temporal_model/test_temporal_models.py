"""Unit tests for Temporal Pulse Sequence Models (LSTM / GRU) and Encoder."""

import sys
from pathlib import Path
import numpy as np
import pytest
import torch

# Ensure project root is on sys.path
PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if "tensorflow" not in sys.modules:
    sys.modules["tensorflow"] = None

from temporal_model.lstm.model import PulseSequenceLSTM
from temporal_model.gru.model import PulseSequenceGRU
from temporal_model.emitter_representation.encoder import TemporalEmitterEncoder
from deinterleaving.emitter_tracker import EmitterTracker


def test_pulse_sequence_lstm_forward():
    model = PulseSequenceLSTM(input_dim=5, hidden_dim=32, num_layers=2, embedding_dim=16)
    batch_size = 4
    seq_len = 20

    dummy_input = torch.randn(batch_size, seq_len, 5)
    out = model(dummy_input)

    assert "embedding" in out
    assert out["embedding"].shape == (batch_size, 16)
    assert "next_delta_toa" in out
    assert out["next_delta_toa"].shape == (batch_size, 1)
    assert "activity_prob" in out
    assert out["activity_prob"].shape == (batch_size, 1)
    # Check bounds
    assert (out["activity_prob"] >= 0.0).all() and (out["activity_prob"] <= 1.0).all()
    assert (out["next_delta_toa"] >= 0.0).all()


def test_pulse_sequence_gru_forward():
    model = PulseSequenceGRU(input_dim=5, hidden_dim=32, num_layers=2, embedding_dim=16)
    batch_size = 3
    seq_len = 15

    dummy_input = torch.randn(batch_size, seq_len, 5)
    out = model(dummy_input)

    assert out["embedding"].shape == (batch_size, 16)
    assert out["activity_prob"].shape == (batch_size, 1)


def test_temporal_emitter_encoder():
    encoder = TemporalEmitterEncoder()

    # Create dummy 5D pulse stream (20 pulses)
    toas = np.cumsum(np.random.uniform(100, 300, 20))
    pdws = np.column_stack([
        toas,
        np.random.normal(9400.0, 5.0, 20),
        np.random.normal(1.0, 0.05, 20),
        np.random.normal(45.0, 1.0, 20),
        np.random.normal(-30.0, 2.0, 20),
    ])

    feats = encoder.encode_cluster(pdws)

    assert "activity_prob" in feats
    assert 0.05 <= feats["activity_prob"] <= 0.95
    assert "predicted_next_delta_toa_us" in feats
    assert feats["predicted_next_delta_toa_us"] > 0.0
    assert "temporal_uncertainty" in feats
    assert 0.05 <= feats["temporal_uncertainty"] <= 0.95
    assert feats["embedding"].shape == (32,)


def test_emitter_tracker_with_lstm():
    tracker = EmitterTracker()

    pdws = np.zeros((40, 5), dtype=np.float32)
    pdws[:, 0] = np.cumsum(np.random.uniform(80, 120, 40))
    pdws[:, 1] = 9400.0
    pdws[:, 2] = 1.0
    pdws[:, 3] = 45.0
    pdws[:, 4] = -30.0

    records = tracker.process_pulse_train(pdws)
    assert len(records) >= 1
    assert 0.0 <= records[0].activity_prob <= 1.0
    assert 0.0 <= records[0].uncertainty <= 1.0
