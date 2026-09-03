"""Training script for Double DQN (and optional Standard DQN) Agent."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
import yaml
import numpy as np

# Ensure project root is on sys.path
PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Safety guard for macOS Anaconda TensorFlow conflict
if "tensorflow" not in sys.modules:
    sys.modules["tensorflow"] = None

from reinforcement_learning.environment.ew_environment import EWEnvironment
from reinforcement_learning.double_dqn.agent import DoubleDQNAgent, StandardDQNAgent
from reinforcement_learning.reward.reward_function import RewardWeights


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Double DQN Agent for EW Scan Strategy")
    parser.add_argument("--config", type=str, default="configs/rl.yaml", help="Path to rl.yaml")
    parser.add_argument("--total-timesteps", type=int, default=5000, help="Total training timesteps")
    parser.add_argument("--eval-episodes", type=int, default=5, help="Number of evaluation episodes")
    parser.add_argument("--save-path", type=str, default="models/rl/double_dqn_agent.zip", help="Save path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--standard-dqn", action="store_true", help="Train Standard Nature DQN instead of Double DQN")
    args = parser.parse_args()

    config = load_config(args.config)
    env_cfg = config.get("environment", {})
    reward_cfg = config.get("reward_weights", {})
    dqn_cfg = config.get("double_dqn", {})

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

    train_env = EWEnvironment(
        num_emitters=int(env_cfg.get("num_emitters", 8)),
        max_steps=int(env_cfg.get("max_steps_per_episode", 100)),
        initial_budget=float(env_cfg.get("initial_resource_budget", 100.0)),
        dwell_cost=float(env_cfg.get("dwell_cost", 1.0)),
        max_revisit_window=int(env_cfg.get("max_revisit_window", 10)),
        drop_timeout=int(env_cfg.get("drop_timeout", 20)),
        reward_weights=weights,
    )

    algo_name = "Standard DQN (Nature DQN)" if args.standard_dqn else "Authentic Double DQN"
    print(f"=== Starting {algo_name} Training ({args.total_timesteps} steps) ===")

    if args.standard_dqn:
        agent = StandardDQNAgent(env=train_env, config=dqn_cfg, seed=args.seed)
    else:
        agent = DoubleDQNAgent(env=train_env, config=dqn_cfg, seed=args.seed)

    agent.learn(total_timesteps=args.total_timesteps)
    agent.save(args.save_path)
    print(f"Model successfully saved to {args.save_path}")

    # Evaluation
    eval_env = EWEnvironment(
        num_emitters=int(env_cfg.get("num_emitters", 8)),
        max_steps=int(env_cfg.get("max_steps_per_episode", 100)),
        reward_weights=weights,
    )

    returns: list[float] = []
    detections: list[int] = []
    drops: list[int] = []

    for ep in range(args.eval_episodes):
        obs, _ = eval_env.reset(seed=args.seed + ep + 100)
        done = False
        ep_ret = 0.0
        while not done:
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            ep_ret += float(reward)
            done = terminated or truncated

        returns.append(ep_ret)
        detections.append(int(info.get("total_detections", 0)))
        drops.append(int(info.get("total_drops", 0)))

    print(f"=== Evaluation Results ({args.eval_episodes} episodes) ===")
    print(f"Mean Return:     {np.mean(returns):.2f} +/- {np.std(returns):.2f}")
    print(f"Mean Detections: {np.mean(detections):.2f}")
    print(f"Mean Drops:      {np.mean(drops):.2f}")


if __name__ == "__main__":
    main()
