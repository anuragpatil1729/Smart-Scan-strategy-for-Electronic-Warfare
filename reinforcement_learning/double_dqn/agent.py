"""Authentic Double DQN and Standard DQN Agent implementations.

Important Distinction:
- Standard SB3 `DQN` implements Mnih et al. (2015) Nature DQN, where the target
  network performs BOTH action selection and action evaluation:
  Y = r + gamma * max_a' Q_target(s', a').
- True Double DQN (Van Hasselt et al., 2016) decouples selection from evaluation:
  a* = argmax_a' Q_online(s', a')
  Y = r + gamma * Q_target(s', a*).

This module implements:
1. `DoubleDQN`: Subclasses `stable_baselines3.DQN` and overrides `train()` to execute
   true Double Q-learning.
2. `DoubleDQNAgent`: Clean wrapper class for the Double DQN model.
3. `StandardDQNAgent`: Wrapper around standard Nature DQN for direct ablation.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import torch as th
import torch.nn.functional as F
import gymnasium as gym

# Safety guard for macOS Anaconda TensorFlow conflict
if "tensorflow" not in sys.modules:
    sys.modules["tensorflow"] = None

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback


class DoubleDQN(DQN):
    """Authentic Double DQN (Van Hasselt et al., 2016).

    Decouples action selection (via online q_net) from action evaluation
    (via target q_net_target) to mitigate positive overestimation bias.
    """

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Switch to train mode
        self.policy.set_training_mode(True)
        # Update learning rate according to schedule
        self._update_learning_rate(self.policy.optimizer)

        losses = []
        for _ in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            discounts = replay_data.discounts if replay_data.discounts is not None else self.gamma

            with th.no_grad():
                # -------------------------------------------------------------
                # TRUE DOUBLE Q-LEARNING STEP:
                # 1. Action selection using the ONLINE network (self.q_net):
                #    a* = argmax_a' Q_online(s', a')
                next_state_actions = self.q_net(replay_data.next_observations).argmax(dim=1, keepdim=True)

                # 2. Action evaluation using the TARGET network (self.q_net_target):
                #    Q_target(s', a*)
                next_q_values = th.gather(
                    self.q_net_target(replay_data.next_observations),
                    dim=1,
                    index=next_state_actions,
                )

                # 1-step Bellman TD target
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * discounts * next_q_values
                # -------------------------------------------------------------

            # Get current Q-values estimates from online network for sampled actions
            current_q_values = self.q_net(replay_data.observations)
            current_q_values = th.gather(current_q_values, dim=1, index=replay_data.actions.long())

            # Huber loss
            loss = F.smooth_l1_loss(current_q_values, target_q_values)
            losses.append(loss.item())

            # Optimize online policy
            self.policy.optimizer.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        # Increase update counter
        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))


class DoubleDQNAgent:
    """Wrapper managing the authentic Double DQN model."""

    def __init__(
        self,
        env: gym.Env,
        config: Optional[Dict[str, Any]] = None,
        device: str = "cpu",
        seed: Optional[int] = None,
    ) -> None:
        self.env = env
        self.config = config or {}
        self.device = device
        self.seed = seed

        lr = float(self.config.get("learning_rate", 5e-4))
        buffer_size = int(self.config.get("buffer_size", 50000))
        learning_starts = int(self.config.get("learning_starts", 200))
        batch_size = int(self.config.get("batch_size", 64))
        gamma = float(self.config.get("gamma", 0.99))
        train_freq = int(self.config.get("train_freq", 4))
        gradient_steps = int(self.config.get("gradient_steps", 1))
        target_update_interval = int(self.config.get("target_update_interval", 250))
        exploration_initial_eps = float(self.config.get("exploration_initial_eps", 1.0))
        exploration_final_eps = float(self.config.get("exploration_final_eps", 0.05))
        exploration_fraction = float(self.config.get("exploration_fraction", 0.3))
        net_arch = self.config.get("net_arch", [128, 128])

        policy_kwargs = {"net_arch": net_arch}

        self.model = DoubleDQN(
            policy="MlpPolicy",
            env=self.env,
            learning_rate=lr,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            gamma=gamma,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            target_update_interval=target_update_interval,
            exploration_initial_eps=exploration_initial_eps,
            exploration_final_eps=exploration_final_eps,
            exploration_fraction=exploration_fraction,
            policy_kwargs=policy_kwargs,
            device=self.device,
            seed=self.seed,
            verbose=0,
        )

    def learn(
        self,
        total_timesteps: int,
        callback: Optional[BaseCallback] = None,
    ) -> DoubleDQNAgent:
        self.model.learn(total_timesteps=total_timesteps, callback=callback)
        return self

    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = True,
    ) -> Tuple[int, Any]:
        action, state = self.model.predict(observation, deterministic=deterministic)
        return int(action), state

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(path))

    @classmethod
    def load(
        cls,
        path: Union[str, Path],
        env: Optional[gym.Env] = None,
        device: str = "cpu",
    ) -> DoubleDQNAgent:
        model = DoubleDQN.load(str(path), env=env, device=device)
        agent = cls.__new__(cls)
        agent.env = env
        agent.model = model
        agent.device = device
        agent.config = {}
        agent.seed = None
        return agent


class StandardDQNAgent:
    """Wrapper around standard SB3 Nature DQN for comparative ablation."""

    def __init__(
        self,
        env: gym.Env,
        config: Optional[Dict[str, Any]] = None,
        device: str = "cpu",
        seed: Optional[int] = None,
    ) -> None:
        self.env = env
        self.config = config or {}
        self.device = device
        self.seed = seed

        lr = float(self.config.get("learning_rate", 5e-4))
        buffer_size = int(self.config.get("buffer_size", 50000))
        learning_starts = int(self.config.get("learning_starts", 200))
        batch_size = int(self.config.get("batch_size", 64))
        gamma = float(self.config.get("gamma", 0.99))
        train_freq = int(self.config.get("train_freq", 4))
        gradient_steps = int(self.config.get("gradient_steps", 1))
        target_update_interval = int(self.config.get("target_update_interval", 250))
        exploration_initial_eps = float(self.config.get("exploration_initial_eps", 1.0))
        exploration_final_eps = float(self.config.get("exploration_final_eps", 0.05))
        exploration_fraction = float(self.config.get("exploration_fraction", 0.3))
        net_arch = self.config.get("net_arch", [128, 128])

        policy_kwargs = {"net_arch": net_arch}

        self.model = DQN(
            policy="MlpPolicy",
            env=self.env,
            learning_rate=lr,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            gamma=gamma,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            target_update_interval=target_update_interval,
            exploration_initial_eps=exploration_initial_eps,
            exploration_final_eps=exploration_final_eps,
            exploration_fraction=exploration_fraction,
            policy_kwargs=policy_kwargs,
            device=self.device,
            seed=self.seed,
            verbose=0,
        )

    def learn(
        self,
        total_timesteps: int,
        callback: Optional[BaseCallback] = None,
    ) -> StandardDQNAgent:
        self.model.learn(total_timesteps=total_timesteps, callback=callback)
        return self

    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = True,
    ) -> Tuple[int, Any]:
        action, state = self.model.predict(observation, deterministic=deterministic)
        return int(action), state

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(path))

    @classmethod
    def load(
        cls,
        path: Union[str, Path],
        env: Optional[gym.Env] = None,
        device: str = "cpu",
    ) -> StandardDQNAgent:
        model = DQN.load(str(path), env=env, device=device)
        agent = cls.__new__(cls)
        agent.env = env
        agent.model = model
        agent.device = device
        agent.config = {}
        agent.seed = None
        return agent
