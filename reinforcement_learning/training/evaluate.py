"""Objective Comparative Evaluation Harness for EW Scan Strategies.

Evaluates trained RL agents (PPO, Double DQN, Standard DQN) against 5 heuristic baselines:
1. Random Scanner: Uniform stochastic dwell allocation.
2. Round-Robin Scanner: Deterministic sequential cyclic scan.
3. Highest Threat Greedy: Greedy dwell on maximum threat * predicted activity.
4. Highest Uncertainty Greedy: Greedy dwell on maximum parameter uncertainty.
5. Most Stale Greedy: Greedy dwell on longest unvisited track to prevent track drops.

Evaluates all strategies objectively without hardcoded preference, reporting comprehensive
metrics: Cumulative Return, Threat Intercept Rate, Critical Miss Rate, Track Drops,
Observation Efficiency, Fleet Uncertainty, and Scan Diversity Entropy.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
import numpy as np
import yaml

# Ensure project root is on sys.path
PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Safety guard for macOS Anaconda TensorFlow conflict
if "tensorflow" not in sys.modules:
    sys.modules["tensorflow"] = None

from reinforcement_learning.environment.ew_environment import EWEnvironment
from reinforcement_learning.state.state_space import RLStateBuilder
from reinforcement_learning.reward.reward_function import RewardWeights
from evaluation.rl_metrics.metrics import (
    EpisodeTrace,
    compute_rl_summary_metrics,
)


class BaseHeuristicScanner:
    """Base interface for heuristic EW scan strategies."""

    def __init__(self, num_emitters: int = 8) -> None:
        self.num_emitters = num_emitters
        self.state_builder = RLStateBuilder(num_emitters=num_emitters)

    def reset(self) -> None:
        pass

    def predict(self, observation: np.ndarray) -> int:
        raise NotImplementedError


class RandomScanner(BaseHeuristicScanner):
    """Uniform stochastic beam allocation."""

    def predict(self, observation: np.ndarray) -> int:
        return int(np.random.randint(0, self.num_emitters))


class RoundRobinScanner(BaseHeuristicScanner):
    """Deterministic cyclical sequential scan."""

    def __init__(self, num_emitters: int = 8) -> None:
        super().__init__(num_emitters=num_emitters)
        self.current_idx = 0

    def reset(self) -> None:
        self.current_idx = 0

    def predict(self, observation: np.ndarray) -> int:
        action = self.current_idx
        self.current_idx = (self.current_idx + 1) % self.num_emitters
        return action


class HighestThreatGreedy(BaseHeuristicScanner):
    """Greedy dwell on emitter with highest expected threat activity."""

    def predict(self, observation: np.ndarray) -> int:
        emitters, _ = self.state_builder.parse_state_vector(observation)
        scores = [float(e.threat_level) * (0.3 + 0.7 * float(e.activity_prob)) for e in emitters]
        return int(np.argmax(scores))


class HighestUncertaintyGreedy(BaseHeuristicScanner):
    """Greedy dwell on emitter with highest parameter uncertainty."""

    def predict(self, observation: np.ndarray) -> int:
        emitters, _ = self.state_builder.parse_state_vector(observation)
        scores = [float(e.uncertainty) for e in emitters]
        return int(np.argmax(scores))


class MostStaleGreedy(BaseHeuristicScanner):
    """Greedy dwell on emitter with highest track age (urgency to prevent track drop)."""

    def predict(self, observation: np.ndarray) -> int:
        emitters, _ = self.state_builder.parse_state_vector(observation)
        scores = [float(e.track_age) * (0.5 + 0.5 * float(e.threat_level)) for e in emitters]
        return int(np.argmax(scores))


def evaluate_policy(
    policy_fn: Callable[[np.ndarray], int],
    env: EWEnvironment,
    num_episodes: int = 20,
    base_seed: int = 1000,
    reset_policy_fn: Optional[Callable[[], None]] = None,
) -> List[EpisodeTrace]:
    """Run an evaluation batch and collect detailed telemetry traces."""
    traces: List[EpisodeTrace] = []

    for ep in range(num_episodes):
        if reset_policy_fn is not None:
            reset_policy_fn()

        obs, info = env.reset(seed=base_seed + ep)
        done = False
        trace = EpisodeTrace()

        while not done:
            action = int(policy_fn(obs))
            obs, reward, terminated, truncated, step_info = env.step(action)
            done = terminated or truncated

            # Record step telemetry
            trace.rewards.append(float(reward))
            trace.actions.append(action)
            detected = bool(step_info.get("detected", False))
            trace.detections.append(detected)
            trace.total_dwell_cost += 1.0

            # Record active threat intercept vs miss
            telemetry = step_info.get("telemetry", {})
            threat_reward = float(telemetry.get("threat_reward", 0.0))
            miss_penalty = float(telemetry.get("miss_penalty", 0.0))

            if detected and threat_reward > 0.0:
                trace.active_threat_intercepts += 1
            if miss_penalty > 0.0:
                trace.active_threat_misses += 1

            # Track fleet mean uncertainty from internal emitters
            mean_unc = float(np.mean([e.uncertainty for e in env.emitters]))
            trace.fleet_uncertainties.append(mean_unc)

        trace.track_drops = int(step_info.get("total_drops", 0))
        traces.append(trace)

    return traces


def print_comparison_table(results: Dict[str, Dict[str, float]]) -> None:
    """Print an objective ASCII benchmark comparison table."""
    headers = [
        "Strategy",
        "Mean Return",
        "Intercept %",
        "Miss %",
        "Track Drops",
        "Efficiency",
        "Fleet Unc",
        "Diversity",
    ]
    row_fmt = "{:<26} | {:>11} | {:>11} | {:>9} | {:>11} | {:>10} | {:>9} | {:>9}"
    divider = "-" * 115

    print("\n" + divider)
    print("OBJECTIVE ELECTRONIC WARFARE SCAN STRATEGY BENCHMARK RESULTS")
    print(divider)
    print(row_fmt.format(*headers))
    print(divider)

    for name, m in results.items():
        print(row_fmt.format(
            name,
            f"{m.get('mean_return', 0.0):.1f}±{m.get('std_return', 0.0):.1f}",
            f"{m.get('threat_intercept_rate', 0.0)*100.0:.1f}%",
            f"{m.get('critical_miss_rate', 0.0)*100.0:.1f}%",
            f"{m.get('mean_track_drops', 0.0):.2f}",
            f"{m.get('observation_efficiency', 0.0):.2f}",
            f"{m.get('mean_fleet_uncertainty', 0.0):.3f}",
            f"{m.get('scan_diversity_entropy', 0.0):.3f}",
        ))

    print(divider + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Objective Evaluation of EW Scan Strategies")
    parser.add_argument("--config", type=str, default="configs/rl.yaml", help="Path to rl.yaml")
    parser.add_argument("--episodes", type=int, default=15, help="Number of test episodes per strategy")
    parser.add_argument("--ppo-path", type=str, default="models/rl/ppo_agent.zip", help="PPO checkpoint")
    parser.add_argument("--double-dqn-path", type=str, default="models/rl/double_dqn_agent.zip", help="Double DQN checkpoint")
    parser.add_argument("--standard-dqn-path", type=str, default="models/rl/standard_dqn_agent.zip", help="Standard DQN checkpoint")
    parser.add_argument("--scenario", type=str, default="dense_fleet", help="Scenario preset (dense_fleet, high_threat_surge, sparse_agile)")
    parser.add_argument("--output-json", type=str, default=None, help="Optional path to save JSON results")
    parser.add_argument("--seed", type=int, default=1000, help="Evaluation base seed")
    args = parser.parse_args()

    # Load configuration
    env_cfg = {}
    reward_cfg = {}
    if os.path.exists(args.config):
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f)
            env_cfg = cfg.get("environment", {})
            reward_cfg = cfg.get("reward_weights", {})

    weights = RewardWeights(
        w_threat=float(reward_cfg.get("w_threat", 2.5)),
        w_detect=float(reward_cfg.get("w_detect", 1.0)),
        w_info=float(reward_cfg.get("w_info", 2.0)),
        w_track=float(reward_cfg.get("w_track", 2.0)),
        w_drop=float(reward_cfg.get("w_drop", 3.5)),
        w_cost=float(reward_cfg.get("w_cost", 0.2)),
        w_miss=float(reward_cfg.get("w_miss", 1.5)),
        w_redundant=float(reward_cfg.get("w_redundant", 1.0)),
    )

    num_emitters = int(env_cfg.get("num_emitters", 8))
    env = EWEnvironment(
        num_emitters=num_emitters,
        max_steps=int(env_cfg.get("max_steps_per_episode", 100)),
        initial_budget=float(env_cfg.get("initial_resource_budget", 100.0)),
        dwell_cost=float(env_cfg.get("dwell_cost", 1.0)),
        max_revisit_window=int(env_cfg.get("max_revisit_window", 10)),
        drop_timeout=int(env_cfg.get("drop_timeout", 20)),
        reward_weights=weights,
        scenario_preset=args.scenario,
    )

    results: Dict[str, Dict[str, float]] = {}

    # 1. Evaluate Heuristic Baselines
    heuristics = [
        ("Random Scanner", RandomScanner(num_emitters)),
        ("Round-Robin Scanner", RoundRobinScanner(num_emitters)),
        ("Highest Threat Greedy", HighestThreatGreedy(num_emitters)),
        ("Highest Uncertainty Greedy", HighestUncertaintyGreedy(num_emitters)),
        ("Most Stale Greedy", MostStaleGreedy(num_emitters)),
    ]

    for name, scanner in heuristics:
        traces = evaluate_policy(
            policy_fn=scanner.predict,
            env=env,
            num_episodes=args.episodes,
            base_seed=args.seed,
            reset_policy_fn=scanner.reset,
        )
        results[name] = compute_rl_summary_metrics(traces, num_actions=num_emitters)

    # 2. Evaluate PPO if checkpoint exists
    if os.path.exists(args.ppo_path):
        from reinforcement_learning.ppo.agent import PPOAgent
        ppo_agent = PPOAgent.load(args.ppo_path, env=env)
        traces = evaluate_policy(
            policy_fn=lambda o: ppo_agent.predict(o, deterministic=True)[0],
            env=env,
            num_episodes=args.episodes,
            base_seed=args.seed,
        )
        results["PPO (SB3)"] = compute_rl_summary_metrics(traces, num_actions=num_emitters)

    # 3. Evaluate Double DQN if checkpoint exists
    if os.path.exists(args.double_dqn_path):
        from reinforcement_learning.double_dqn.agent import DoubleDQNAgent
        ddqn_agent = DoubleDQNAgent.load(args.double_dqn_path, env=env)
        traces = evaluate_policy(
            policy_fn=lambda o: ddqn_agent.predict(o, deterministic=True)[0],
            env=env,
            num_episodes=args.episodes,
            base_seed=args.seed,
        )
        results["Authentic Double DQN"] = compute_rl_summary_metrics(traces, num_actions=num_emitters)

    # 4. Evaluate Standard DQN if checkpoint exists
    if os.path.exists(args.standard_dqn_path):
        from reinforcement_learning.double_dqn.agent import StandardDQNAgent
        sdqn_agent = StandardDQNAgent.load(args.standard_dqn_path, env=env)
        traces = evaluate_policy(
            policy_fn=lambda o: sdqn_agent.predict(o, deterministic=True)[0],
            env=env,
            num_episodes=args.episodes,
            base_seed=args.seed,
        )
        results["Standard Nature DQN"] = compute_rl_summary_metrics(traces, num_actions=num_emitters)

    # Display comparison
    print_comparison_table(results)

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output_json}")


if __name__ == "__main__":
    main()
