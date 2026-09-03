"""Unified Training Dispatcher for EW Scan Strategy RL.

Supports training PPO, Authentic Double DQN, and Standard DQN with a single CLI.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
import yaml

# Ensure project root is on sys.path
PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Safety guard for macOS Anaconda TensorFlow conflict
if "tensorflow" not in sys.modules:
    sys.modules["tensorflow"] = None

from reinforcement_learning.environment.ew_environment import EWEnvironment
from reinforcement_learning.reward.reward_function import RewardWeights
from reinforcement_learning.ppo.agent import PPOAgent
from reinforcement_learning.double_dqn.agent import DoubleDQNAgent, StandardDQNAgent


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def train_single_agent(algo: str, config: dict, timesteps: int, seed: int) -> None:
    env_cfg = config.get("environment", {})
    reward_cfg = config.get("reward_weights", {})

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

    if config.get("use_dataset", False):
        from reinforcement_learning.environment.real_data_environment import RealDataEWEnvironment
        print(f">>> Initializing RealDataEWEnvironment ({config.get('split', 'train_scan')}, max {config.get('max_scenarios', 'all')} scenarios)...")
        env = RealDataEWEnvironment(
            dataset_path=config.get("dataset_path", "datasets/synthetic/turing_radar_data"),
            split=config.get("split", "train_scan"),
            num_emitters=int(env_cfg.get("num_emitters", 8)),
            max_steps=int(env_cfg.get("max_steps_per_episode", 100)),
            initial_budget=float(env_cfg.get("initial_resource_budget", 100.0)),
            dwell_cost=float(env_cfg.get("dwell_cost", 1.0)),
            max_revisit_window=int(env_cfg.get("max_revisit_window", 10)),
            drop_timeout=int(env_cfg.get("drop_timeout", 20)),
            max_scenarios=config.get("max_scenarios"),
            reward_weights=weights,
        )
    else:
        env = EWEnvironment(
            num_emitters=int(env_cfg.get("num_emitters", 8)),
            max_steps=int(env_cfg.get("max_steps_per_episode", 100)),
            initial_budget=float(env_cfg.get("initial_resource_budget", 100.0)),
            dwell_cost=float(env_cfg.get("dwell_cost", 1.0)),
            max_revisit_window=int(env_cfg.get("max_revisit_window", 10)),
            drop_timeout=int(env_cfg.get("drop_timeout", 20)),
            reward_weights=weights,
        )

    models_dir = Path("models/rl")
    models_dir.mkdir(parents=True, exist_ok=True)

    if algo == "ppo":
        print(f"\n>>> Training PPO Agent for {timesteps} steps...")
        agent = PPOAgent(env=env, config=config.get("ppo", {}), seed=seed)
        agent.learn(total_timesteps=timesteps)
        save_path = models_dir / "ppo_agent.zip"
        agent.save(save_path)
        print(f">>> PPO saved to {save_path}")

    elif algo == "double_dqn":
        print(f"\n>>> Training Authentic Double DQN Agent for {timesteps} steps...")
        agent = DoubleDQNAgent(env=env, config=config.get("double_dqn", {}), seed=seed)
        agent.learn(total_timesteps=timesteps)
        save_path = models_dir / "double_dqn_agent.zip"
        agent.save(save_path)
        print(f">>> Authentic Double DQN saved to {save_path}")

    elif algo == "standard_dqn":
        print(f"\n>>> Training Standard Nature DQN Agent for {timesteps} steps...")
        agent = StandardDQNAgent(env=env, config=config.get("standard_dqn", {}), seed=seed)
        agent.learn(total_timesteps=timesteps)
        save_path = models_dir / "standard_dqn_agent.zip"
        agent.save(save_path)
        print(f">>> Standard DQN saved to {save_path}")

    else:
        raise ValueError(f"Unknown algorithm: {algo}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified Training for EW Scan RL")
    parser.add_argument("--config", type=str, default="configs/rl.yaml", help="Path to config file")
    parser.add_argument("--algo", type=str, default="ppo", choices=["ppo", "double_dqn", "standard_dqn", "all"], help="Algorithm to train")
    parser.add_argument("--total-timesteps", type=int, default=5000, help="Total training steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--use-dataset", action="store_true", help="Train on real TSRD radar dataset files")
    parser.add_argument("--max-scenarios", type=int, default=None, help="Max scenarios to load (default: all 2500)")
    parser.add_argument("--dataset-path", type=str, default="datasets/synthetic/turing_radar_data", help="TSRD dataset path")
    parser.add_argument("--split", type=str, default="train_scan", help="Dataset split (train_scan, val_scan, train_stare)")
    args = parser.parse_args()

    config = load_config(args.config)
    config["use_dataset"] = args.use_dataset
    config["max_scenarios"] = args.max_scenarios
    config["dataset_path"] = args.dataset_path
    config["split"] = args.split

    if args.algo == "all":
        for a in ["ppo", "double_dqn", "standard_dqn"]:
            train_single_agent(a, config, args.total_timesteps, args.seed)
    else:
        train_single_agent(args.algo, config, args.total_timesteps, args.seed)


if __name__ == "__main__":
    main()
