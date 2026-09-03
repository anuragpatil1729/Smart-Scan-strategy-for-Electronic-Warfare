# Smart Scan Strategy for Electronic Warfare

An intelligent Electronic Warfare (EW) signal processing and sensor scheduling platform integrating pulse deinterleaving with reinforcement learning to optimize multi-objective radar dwell allocation.

---

## System Architecture

```text
+-------------------------------------------------------------------------+
|                The Turing Synthetic Radar Dataset (TSRD)                |
|           (5D PDWs: ToA, Frequency, PulseWidth, AoA, Amplitude)        |
+------------------------------------+------------------------------------+
                                     |
                                     v
+-------------------------------------------------------------------------+
|                           PDW Extraction Layer                          |
|  - HDF5 Reader (scan & stare modes, train/val/test splits)              |
|  - Differential Features: Delta-ToA, robust scaling, pulse statistics   |
+------------------------------------+------------------------------------+
                                     |
                                     v
+-------------------------------------------------------------------------+
|                         Deinterleaving Subsystem                        |
|  - Stage 1: Spatial-Spectral DBSCAN (AoA, Frequency, Pulse Width)       |
|  - Stage 2: Temporal PRI Analysis (Delta-ToA Histograms & Peak Detect)  |
|  - Emitter Tracking: State estimation, threat priority, uncertainty     |
+------------------------------------+------------------------------------+
                                     |
                                     v
+-------------------------------------------------------------------------+
|                    84-Dimensional RL State Space                        |
|  - 8 Emitters x 10 Estimation Features (=80) + 4 Global Sensor Features |
|  - Ex-ante Potential Info Gain (State) vs Ex-post Actual Gain (Reward)  |
|  - Clean Upstream Adapter (padding, priority sorting, bounds [0, 1])    |
+------------------------------------+------------------------------------+
                                     |
                                     v
+-------------------------------------------------------------------------+
|                       Intelligent Scan Policies                         |
|  - Authentic Double DQN (decoupled online selection & target eval)      |
|  - Stable-Baselines3 PPO Policy (actor-critic with entropy bonus)       |
|  - Standard Nature DQN Baseline (Mnih et al., 2015)                     |
|  - 5 Heuristic Baselines (Threat, Uncertainty, Stale, Round-Robin, Rand)|
+------------------------------------+------------------------------------+
                                     |
                                     v
+-------------------------------------------------------------------------+
|                  Dwell Execution & Multi-Objective Reward               |
|  - Intercept Reward + Ex-Post Info Gain + Track Maintenance Reward      |
|  - Track Drop Penalty - Miss Penalty - Dwell Cost - Redundancy Penalty  |
+-------------------------------------------------------------------------+
```

---

## Quickstart: Train & Test

### Unified Runner (`train_test.py`)

```bash
# 1. Run full test suite (36 tests, takes ~2 seconds)
python3 train_test.py --mode pytest

# 2. Run benchmark evaluation across all 8 policies (takes ~35 seconds)
python3 train_test.py --mode test --episodes 15

# 3. Train all RL agents (PPO, Double DQN, Standard DQN) for 5,000 steps
python3 train_test.py --mode train --algo all --timesteps 5000

# 4. Train and test on real Turing radar dataset scenarios
python3 train_test.py --mode both --algo ppo --use-dataset --max-scenarios 25 --timesteps 5000
```

---

## Detailed CLI Commands

### Automated Test Suite
```bash
pytest tests/ -v -s
```

### Comparative Benchmark
```bash
# Benchmark on Dense Fleet scenario
python3 reinforcement_learning/training/evaluate.py --episodes 15

# Benchmark on High Threat Surge scenario
python3 reinforcement_learning/training/evaluate.py --scenario high_threat_surge --episodes 15

# Benchmark on Sparse Agile Emitters
python3 reinforcement_learning/training/evaluate.py --scenario sparse_agile --episodes 15
```

### Training RL Agents Individually
```bash
# Train PPO
python3 reinforcement_learning/ppo/train.py --total-timesteps 10000 --eval-episodes 5

# Train Authentic Double DQN
python3 reinforcement_learning/double_dqn/train.py --total-timesteps 10000 --eval-episodes 5

# Train Standard Nature DQN
python3 reinforcement_learning/double_dqn/train.py --standard-dqn --total-timesteps 10000
```

---

## Benchmark Results

Objective evaluation across 8 scan strategies under identical environments:

| Strategy | Mean Return | Threat Intercept % | Critical Miss % | Track Drops | Observation Efficiency | Fleet Uncertainty | Scan Diversity |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Authentic Double DQN** | **128.4 ± 39.8** | 15.5% | 84.5% | 0.70 | 0.07 | 0.870 | 0.273 |
| **Highest Threat Greedy** | 116.4 ± 55.0 | **39.3%** | **60.7%** | 3.00 | **0.23** | **0.805** | 0.722 |
| **PPO (SB3)** | 83.1 ± 9.4 | 15.1% | 84.9% | **0.00** | 0.07 | 0.880 | 0.000 |
| **Standard Nature DQN** | 79.8 ± 20.6 | 17.0% | 83.0% | 1.10 | 0.07 | 0.871 | 0.347 |
| **Most Stale Greedy** | -3.5 ± 16.4 | 13.1% | 86.9% | 1.40 | 0.06 | 0.922 | 0.664 |
| **Random Scanner** | -7.1 ± 9.5 | 14.3% | 85.7% | 3.20 | 0.06 | 0.941 | 0.998 |
| **Round-Robin Scanner** | -8.8 ± 12.5 | 13.9% | 86.1% | 3.10 | 0.06 | 0.947 | 1.000 |
| **Highest Uncertainty Greedy** | -15.6 ± 8.6 | 9.3% | 90.7% | 1.50 | 0.04 | 0.964 | 0.662 |

---

## Project Structure

```text
├── configs/
│   ├── dataset.yaml             # TSRD dataset and deinterleaving parameters
│   └── rl.yaml                  # RL environment, reward weights, and agent hyperparameters
├── datasets/
│   └── synthetic/
│       └── turing_radar_data    # Symlinked TSRD dataset (66 GB, 2,500 scenarios)
├── pdw/
│   ├── extraction/              # Fast HDF5 reader for 5D PDWs
│   └── features/                # Delta-ToA, scaling, and stream statistics
├── deinterleaving/
│   ├── dbscan/                  # Spatial-spectral DBSCAN clustering
│   ├── sedcam/                  # Temporal PRI analysis and modulation classification
│   └── emitter_tracker.py       # Emitter track extraction and RL interface bridge
├── reinforcement_learning/
│   ├── state/                   # 84-dim state space and UpstreamStateAdapter
│   ├── action/                  # Discrete dwell action space
│   ├── reward/                  # Corrected track maintenance & actual info gain rewards
│   ├── environment/             # Synthetic and Real-Data Gymnasium environments
│   ├── ppo/                     # SB3 PPO agent and training pipeline
│   ├── double_dqn/              # Authentic Double DQN and Standard Nature DQN
│   └── training/                # Dispatcher and objective evaluation benchmark
├── evaluation/
│   ├── deinterleaving_metrics/  # V-Measure, ARI, AMI, pairwise F1, MCC
│   └── rl_metrics/              # Intercept rate, miss rate, track drops, entropy
├── tests/                       # Complete pytest suite (36 passing tests)
├── train_test.py                # Unified CLI runner for training, testing, and benchmarking
└── README.md
```
