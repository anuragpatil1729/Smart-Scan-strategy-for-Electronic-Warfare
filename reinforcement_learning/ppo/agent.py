"""PPO Agent implementation for Electronic Warfare Smart Scan.

Wraps Stable-Baselines3 PPO with custom policy configuration,
automatic device selection, and model persistence.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import torch as th
import gymnasium as gym

# Safety guard for macOS Anaconda TensorFlow conflict
if "tensorflow" not in sys.modules:
    sys.modules["tensorflow"] = None

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback


def get_optimal_device() -> str:
    """Select appropriate compute device: MPS (Apple Silicon), CUDA, or CPU."""
    if th.cuda.is_available():
        return "cuda"
    # SB3 categorical distribution sampling on MPS can occasionally throw or be slower on small MLPs;
    # CPU is fast and ultra-stable for 84-dim MLP vectors on Apple Silicon
    return "cpu"


class PPOAgent:
    """Configurable PPO Agent for discrete scan strategy decisions."""

    def __init__(
        self,
        env: gym.Env,
        config: Optional[Dict[str, Any]] = None,
        device: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.env = env
        self.config = config or {}
        self.device = device or get_optimal_device()
        self.seed = seed

        learning_rate = float(self.config.get("learning_rate", 3e-4))
        n_steps = int(self.config.get("n_steps", 256))
        batch_size = int(self.config.get("batch_size", 64))
        n_epochs = int(self.config.get("n_epochs", 10))
        gamma = float(self.config.get("gamma", 0.99))
        gae_lambda = float(self.config.get("gae_lambda", 0.95))
        clip_range = float(self.config.get("clip_range", 0.2))
        ent_coef = float(self.config.get("ent_coef", 0.01))
        vf_coef = float(self.config.get("vf_coef", 0.5))
        max_grad_norm = float(self.config.get("max_grad_norm", 0.5))

        net_arch_cfg = self.config.get("net_arch", {"pi": [128, 128], "vf": [128, 128]})
        policy_kwargs = {"net_arch": net_arch_cfg}

        self.model = PPO(
            policy="MlpPolicy",
            env=self.env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            policy_kwargs=policy_kwargs,
            device=self.device,
            seed=self.seed,
            verbose=0,
        )

    def learn(
        self,
        total_timesteps: int,
        callback: Optional[BaseCallback] = None,
    ) -> PPOAgent:
        """Train the PPO model."""
        self.model.learn(total_timesteps=total_timesteps, callback=callback)
        return self

    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = True,
    ) -> Tuple[int, Any]:
        """Predict action for given observation."""
        action, state = self.model.predict(observation, deterministic=deterministic)
        return int(action), state

    def save(self, path: Union[str, Path]) -> None:
        """Save model checkpoint to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(path))

    @classmethod
    def load(
        cls,
        path: Union[str, Path],
        env: Optional[gym.Env] = None,
        device: Optional[str] = None,
    ) -> PPOAgent:
        """Load trained agent from checkpoint."""
        dev = device or get_optimal_device()
        model = PPO.load(str(path), env=env, device=dev)
        agent = cls.__new__(cls)
        agent.env = env
        agent.model = model
        agent.device = dev
        agent.config = {}
        agent.seed = None
        return agent
