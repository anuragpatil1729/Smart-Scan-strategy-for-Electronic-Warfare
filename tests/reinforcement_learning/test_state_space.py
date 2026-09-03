"""Unit tests for the 84-dimensional State Space and State Builder."""

import sys
if "tensorflow" not in sys.modules:
    sys.modules["tensorflow"] = None

import numpy as np
import pytest
from reinforcement_learning.state.state_space import (
    EmitterFeatures,
    GlobalSensorFeatures,
    RLStateBuilder,
)


def test_emitter_features_to_array():
    feat = EmitterFeatures(
        activity_prob=0.8,
        threat_level=0.9,
        identity_confidence=0.7,
        uncertainty=0.4,
        novelty=0.1,
        track_age=0.3,
        mode=0.5,
        potential_info_gain=0.6,
        observation_cost=0.1,
        miss_risk=0.288,
    )
    arr = feat.to_array()
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (10,)
    assert arr.dtype == np.float32
    assert (arr >= 0.0).all() and (arr <= 1.0).all()

    # Test reconstruct
    reconstructed = EmitterFeatures.from_array(arr)
    assert pytest.approx(reconstructed.threat_level) == 0.9
    assert pytest.approx(reconstructed.uncertainty) == 0.4


def test_global_sensor_features_to_array():
    gf = GlobalSensorFeatures(
        sensor_utilization=0.45,
        remaining_budget_fraction=0.85,
        normalized_step=0.45,
        active_threat_ratio=0.25,
    )
    arr = gf.to_array()
    assert arr.shape == (4,)
    assert (arr >= 0.0).all() and (arr <= 1.0).all()

    reconstructed = GlobalSensorFeatures.from_array(arr)
    assert pytest.approx(reconstructed.remaining_budget_fraction) == 0.85


def test_state_builder_dimension_84():
    builder = RLStateBuilder(num_emitters=8)
    assert builder.total_dim == 84
    assert builder.observation_space.shape == (84,)

    emitters = [EmitterFeatures() for _ in range(8)]
    gf = GlobalSensorFeatures()
    obs = builder.build_state_vector(emitters, gf)

    assert obs.shape == (84,)
    assert obs.dtype == np.float32
    assert (obs >= 0.0).all() and (obs <= 1.0).all()


def test_state_builder_lossless_parse():
    builder = RLStateBuilder(num_emitters=8)
    emitters_in = [
        EmitterFeatures(
            activity_prob=i / 10.0,
            threat_level=(8 - i) / 10.0,
            identity_confidence=0.5,
            uncertainty=0.3,
            novelty=0.05,
            track_age=i * 0.1,
            mode=0.25,
            potential_info_gain=0.2,
            observation_cost=0.1,
            miss_risk=0.15,
        )
        for i in range(8)
    ]
    gf_in = GlobalSensorFeatures(
        sensor_utilization=0.6,
        remaining_budget_fraction=0.75,
        normalized_step=0.6,
        active_threat_ratio=0.375,
    )

    obs = builder.build_state_vector(emitters_in, gf_in)
    emitters_out, gf_out = builder.parse_state_vector(obs)

    assert len(emitters_out) == 8
    for i in range(8):
        assert pytest.approx(emitters_out[i].activity_prob, abs=1e-5) == i / 10.0
        assert pytest.approx(emitters_out[i].threat_level, abs=1e-5) == (8 - i) / 10.0

    assert pytest.approx(gf_out.sensor_utilization, abs=1e-5) == 0.6
    assert pytest.approx(gf_out.remaining_budget_fraction, abs=1e-5) == 0.75


def test_state_builder_dict():
    builder = RLStateBuilder(num_emitters=8)
    emitters = [EmitterFeatures() for _ in range(8)]
    gf = GlobalSensorFeatures()
    obs = builder.build_state_vector(emitters, gf)
    d = builder.to_dict(obs)

    assert "emitters" in d
    assert "global" in d
    assert len(d["emitters"]) == 8
    assert "activity_prob" in d["emitters"][0]
