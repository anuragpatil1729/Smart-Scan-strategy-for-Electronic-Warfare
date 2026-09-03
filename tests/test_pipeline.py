"""End-to-End System Pipeline Integration Test.

Connects:
1. Raw TSRD radar dataset HDF5 pulse stream
2. Spatial-spectral deinterleaving and PRI estimation
3. Emitter track extraction
4. Upstream interface conversion to 84-dimensional RL observation
5. Trained RL agent action prediction (PPO & Double DQN)
6. EW Environment execution
"""

import sys
from pathlib import Path
import numpy as np
import pytest

# Ensure project root is on sys.path
PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if "tensorflow" not in sys.modules:
    sys.modules["tensorflow"] = None

from pdw.extraction.dataset_reader import TSRDDatasetReader
from deinterleaving.dbscan.clustering import SpatialSpectralDBSCAN
from deinterleaving.emitter_tracker import EmitterTracker
from evaluation.deinterleaving_metrics.metrics import evaluate_deinterleaving
from reinforcement_learning.state.upstream_interface import (
    UpstreamScanContext,
    UpstreamStateAdapter,
)
from reinforcement_learning.ppo.agent import PPOAgent
from reinforcement_learning.double_dqn.agent import DoubleDQNAgent
from reinforcement_learning.environment.ew_environment import EWEnvironment

DATASET_PATH = Path("datasets/synthetic/turing_radar_data")


@pytest.mark.skipif(not DATASET_PATH.exists(), reason="TSRD dataset not available")
def test_full_pipeline_from_dataset_to_rl_decision():
    """Verify end-to-end integration: TSRD HDF5 -> Deinterleaving -> RL State -> Action."""
    # 1. Load real radar pulses from TSRD
    reader = TSRDDatasetReader(root_path=DATASET_PATH)
    sample = reader.load_sample(split="train_scan", index_or_filename=0, max_pulses=2000)

    assert sample.pdws.shape == (2000, 5)
    assert len(sample.labels) == 2000

    # 2. Deinterleave pulses using spatial-spectral clustering
    clusterer = SpatialSpectralDBSCAN(eps=0.10, min_samples=8)
    predicted_labels = clusterer.fit_predict(sample.pdws)

    # Evaluate deinterleaving quality on real data
    metrics = evaluate_deinterleaving(sample.labels, predicted_labels)
    assert "v_measure" in metrics
    assert "adjusted_rand_index" in metrics
    print(f"\n[Deinterleaving Quality on TSRD sample]: V-Measure={metrics['v_measure']:.3f}, ARI={metrics['adjusted_rand_index']:.3f}")

    # 3. Extract emitter tracks and map to UpstreamEmitterRecord
    tracker = EmitterTracker(clusterer=clusterer)
    upstream_records = tracker.process_pulse_train(sample.pdws)
    assert len(upstream_records) > 0

    # 4. Ingest upstream records into RL state adapter
    adapter = UpstreamStateAdapter(num_emitters=8)
    context = UpstreamScanContext(
        current_timestamp=float(sample.pdws[-1, 0] * 1e-6),
        current_step=10,
        max_steps=100,
        sensor_utilization=0.10,
        remaining_budget_fraction=0.90,
    )
    obs = adapter.convert_to_observation(upstream_records, context=context)

    # 5. Verify the 84-dimensional state representation
    assert obs.shape == (84,)
    assert adapter.validate_observation(obs) is True

    # 6. Feed observation into trained RL models to decide directional dwell
    models_dir = Path("models/rl")
    ppo_path = models_dir / "ppo_agent.zip"
    ddqn_path = models_dir / "double_dqn_agent.zip"

    env = EWEnvironment(num_emitters=8, max_steps=100)
    env.reset(seed=42)

    if ppo_path.exists():
        ppo = PPOAgent.load(ppo_path, env=env)
        action_ppo, _ = ppo.predict(obs, deterministic=True)
        assert 0 <= action_ppo < 8
        print(f"[RL Decision] PPO scheduled dwell on Emitter Index: {action_ppo}")

    if ddqn_path.exists():
        ddqn = DoubleDQNAgent.load(ddqn_path, env=env)
        action_ddqn, _ = ddqn.predict(obs, deterministic=True)
        assert 0 <= action_ddqn < 8
        print(f"[RL Decision] Authentic Double DQN scheduled dwell on Emitter Index: {action_ddqn}")

    # 7. Step environment with chosen action
    next_obs, reward, term, trunc, info = env.step(action_ppo if ppo_path.exists() else 0)
    assert next_obs.shape == (84,)
    assert "telemetry" in info
