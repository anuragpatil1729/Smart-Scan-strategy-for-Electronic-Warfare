"""Unified Train & Test CLI Runner for Smart Scan Strategy in Electronic Warfare.

Provides a unified workflow command to train RL agents (PPO, Authentic Double DQN,
Standard DQN), run comparative benchmarking against 5 heuristic baselines, and execute
the automated test suite.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = str(Path(__file__).resolve().parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if "tensorflow" not in sys.modules:
    sys.modules["tensorflow"] = None


def run_command(cmd_args: list[str], description: str) -> int:
    """Execute a subprocess command and stream output."""
    print(f"\n{'='*70}\n[RUNNING] {description}\n{'='*70}")
    print(f"$ {' '.join(cmd_args)}\n")
    proc = subprocess.run(cmd_args)
    if proc.returncode != 0:
        print(f"\n[ERROR] Command failed with exit code {proc.returncode}: {description}")
    return proc.returncode


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unified Train & Test Runner for EW Smart Scan Strategy",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="both",
        choices=["train", "test", "both", "pytest"],
        help="Execution mode: train agents, run benchmark test, both, or run pytest suite",
    )
    parser.add_argument(
        "--algo",
        type=str,
        default="all",
        choices=["ppo", "double_dqn", "standard_dqn", "all"],
        help="RL algorithm to train",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=5000,
        help="Training timesteps per algorithm",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=15,
        help="Evaluation episodes per policy",
    )
    parser.add_argument(
        "--use-dataset",
        action="store_true",
        help="Train / test on real Turing Synthetic Radar Dataset scenarios",
    )
    parser.add_argument(
        "--max-scenarios",
        type=int,
        default=None,
        help="Limit number of TSRD dataset scenarios (e.g. 5, 25, 250, or None for all)",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="dense_fleet",
        choices=["dense_fleet", "high_threat_surge", "sparse_agile"],
        help="Scenario preset for synthetic evaluation",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="evaluation/results/train_test_results.json",
        help="Path to save evaluation benchmark metrics",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed",
    )
    args = parser.parse_args()

    python_bin = sys.executable

    # Mode: pytest only
    if args.mode == "pytest":
        code = run_command([python_bin, "-m", "pytest", "tests/", "-v", "-s"], "Full Repository Test Suite")
        sys.exit(code)

    # Mode: train or both
    if args.mode in ("train", "both"):
        train_cmd = [
            python_bin,
            "reinforcement_learning/training/train.py",
            "--algo", args.algo,
            "--total-timesteps", str(args.timesteps),
            "--seed", str(args.seed),
        ]
        if args.use_dataset:
            train_cmd.append("--use-dataset")
            if args.max_scenarios is not None:
                train_cmd.extend(["--max-scenarios", str(args.max_scenarios)])

        desc = f"Training RL Agents ({args.algo}) for {args.timesteps} timesteps"
        if args.use_dataset:
            desc += f" on real TSRD scenarios (max: {args.max_scenarios or 'all'})"

        code = run_command(train_cmd, desc)
        if code != 0 and args.mode == "train":
            sys.exit(code)

    # Mode: test or both
    if args.mode in ("test", "both"):
        eval_cmd = [
            python_bin,
            "reinforcement_learning/training/evaluate.py",
            "--episodes", str(args.episodes),
            "--scenario", args.scenario,
            "--output-json", args.output_json,
            "--seed", str(args.seed + 1000),
        ]
        code = run_command(eval_cmd, f"Objective Benchmark Evaluation ({args.episodes} episodes on {args.scenario})")
        if code != 0:
            sys.exit(code)

    print(f"\n{'='*70}\n[SUCCESS] Train-Test execution completed successfully!\n{'='*70}\n")


if __name__ == "__main__":
    main()
