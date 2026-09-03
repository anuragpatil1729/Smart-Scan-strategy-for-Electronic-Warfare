"""Unit tests for Upstream Emitter Interface and Adapter."""

import sys
if "tensorflow" not in sys.modules:
    sys.modules["tensorflow"] = None

import numpy as np
import pytest
from reinforcement_learning.state.upstream_interface import (
    UpstreamEmitterRecord,
    UpstreamScanContext,
    UpstreamStateAdapter,
)


def test_upstream_adapter_shape_and_validation():
    adapter = UpstreamStateAdapter(num_emitters=8)
    records = [
        UpstreamEmitterRecord(
            emitter_id=f"radar_{i}",
            threat_level=0.5 + 0.05 * i,
            activity_prob=0.8,
            identity_confidence=0.6,
            uncertainty=0.4,
            novelty_score=0.1,
            last_intercept_timestamp=5.0,
            operational_mode="track",
            dwell_cost=0.1,
            carrier_freq_mhz=9400.0,
            pri_us=125.0,
            snr_db=18.0,
        )
        for i in range(8)
    ]
    context = UpstreamScanContext(
        current_timestamp=8.0,
        current_step=15,
        max_steps=100,
        sensor_utilization=0.15,
        remaining_budget_fraction=0.85,
        drop_timeout_seconds=10.0,
    )

    obs = adapter.convert_to_observation(records, context)
    assert obs.shape == (84,)
    assert adapter.validate_observation(obs) is True


def test_upstream_adapter_padding_fewer_emitters():
    """Verify that fewer than 8 emitters are safely zero-padded to 84 dimensions."""
    adapter = UpstreamStateAdapter(num_emitters=8)
    records = [
        UpstreamEmitterRecord(emitter_id="target_alpha", threat_level=0.9),
        UpstreamEmitterRecord(emitter_id="target_bravo", threat_level=0.7),
    ]
    obs = adapter.convert_to_observation(records)
    assert obs.shape == (84,)
    assert adapter.validate_observation(obs) is True

    # Check that padded slots have 0 threat and 0 activity
    diag = adapter.parse_observation(obs)
    assert len(diag["emitters"]) == 8
    assert pytest.approx(diag["emitters"][0]["threat_level"], abs=1e-5) == 0.9
    assert pytest.approx(diag["emitters"][1]["threat_level"], abs=1e-5) == 0.7
    assert diag["emitters"][2]["threat_level"] == 0.0
    assert diag["emitters"][7]["threat_level"] == 0.0


def test_upstream_adapter_prioritization_more_emitters():
    """Verify that more than 8 emitters are prioritized down to top 8 by composite urgency."""
    adapter = UpstreamStateAdapter(num_emitters=8)
    records = [
        UpstreamEmitterRecord(
            emitter_id=f"emit_{i}",
            threat_level=i / 15.0,
            activity_prob=0.8,
            uncertainty=0.5,
        )
        for i in range(15)
    ]
    obs = adapter.convert_to_observation(records)
    assert obs.shape == (84,)
    assert adapter.validate_observation(obs) is True

    diag = adapter.parse_observation(obs)
    # The highest threat emitter (threat ~ 14/15 = 0.933) should be included
    threats = [e["threat_level"] for e in diag["emitters"]]
    assert max(threats) == pytest.approx(14 / 15.0, abs=1e-4)


def test_upstream_adapter_potential_info_gain_is_ex_ante():
    """Verify that potential_info_gain is an ex-ante state feature."""
    adapter = UpstreamStateAdapter(num_emitters=8)
    rec = UpstreamEmitterRecord(
        emitter_id="test_emit",
        uncertainty=0.8,
        snr_db=20.0,
    )
    obs = adapter.convert_to_observation([rec])
    diag = adapter.parse_observation(obs)
    e0 = diag["emitters"][0]
    # Expect potential_info_gain to be non-zero and derived from uncertainty & SNR
    assert e0["potential_info_gain"] > 0.5
    assert e0["potential_info_gain"] <= 1.0
