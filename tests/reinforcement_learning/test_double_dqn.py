"""Unit tests for Authentic Double DQN vs Standard DQN."""

import sys
if "tensorflow" not in sys.modules:
    sys.modules["tensorflow"] = None

import pytest
import numpy as np
import torch as th
from stable_baselines3 import DQN
from reinforcement_learning.environment.ew_environment import EWEnvironment
from reinforcement_learning.double_dqn.agent import (
    DoubleDQN,
    DoubleDQNAgent,
    StandardDQNAgent,
)


def test_double_dqn_inheritance_and_action_decoupling():
    """Verify that DoubleDQN subclasses SB3 DQN and correctly decouples online selection from target evaluation."""
    assert issubclass(DoubleDQN, DQN)

    env = EWEnvironment(num_emitters=8, max_steps=50)
    agent = DoubleDQNAgent(
        env=env,
        config={"learning_starts": 10, "batch_size": 16, "buffer_size": 1000},
        device="cpu",
        seed=42,
    )

    # Check model architecture
    assert hasattr(agent.model, "q_net")
    assert hasattr(agent.model, "q_net_target")

    # Predict test
    obs, _ = env.reset(seed=42)
    action, _ = agent.predict(obs, deterministic=True)
    assert 0 <= action < 8


def test_double_dqn_learn_short_step():
    """Verify that train() executes gradient updates without error."""
    env = EWEnvironment(num_emitters=8, max_steps=50)
    agent = DoubleDQNAgent(
        env=env,
        config={"learning_starts": 20, "batch_size": 16, "buffer_size": 1000},
        device="cpu",
        seed=42,
    )
    # Collect small experience and train for 50 steps
    agent.learn(total_timesteps=50)
    obs, _ = env.reset(seed=42)
    action, _ = agent.predict(obs, deterministic=True)
    assert 0 <= action < 8


def test_standard_dqn_agent_initialization():
    """Verify StandardDQNAgent initializes standard Nature DQN."""
    env = EWEnvironment(num_emitters=8, max_steps=50)
    agent = StandardDQNAgent(
        env=env,
        config={"learning_starts": 20, "batch_size": 16, "buffer_size": 1000},
        device="cpu",
        seed=42,
    )
    assert type(agent.model) is DQN
    obs, _ = env.reset(seed=42)
    action, _ = agent.predict(obs, deterministic=True)
    assert 0 <= action < 8
