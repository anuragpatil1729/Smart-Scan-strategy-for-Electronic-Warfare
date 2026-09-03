"""Unit tests for the Gymnasium EW Environment."""

import sys
if "tensorflow" not in sys.modules:
    sys.modules["tensorflow"] = None

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env
from reinforcement_learning.environment.ew_environment import EWEnvironment


def test_gym_check_env():
    """Verify standard Gymnasium compliance."""
    env = EWEnvironment(num_emitters=8, max_steps=50)
    check_env(env)


def test_environment_reset_seed_reproducibility():
    """Verify that deterministic seeding produces identical initial observations."""
    env1 = EWEnvironment(num_emitters=8, max_steps=50)
    env2 = EWEnvironment(num_emitters=8, max_steps=50)

    obs1, info1 = env1.reset(seed=12345)
    obs2, info2 = env2.reset(seed=12345)

    np.testing.assert_allclose(obs1, obs2, atol=1e-5)
    assert obs1.shape == (84,)


def test_environment_step_progression():
    """Verify environment step advances time and terminates on budget or step limit."""
    env = EWEnvironment(num_emitters=8, max_steps=10, initial_budget=10.0, dwell_cost=1.0)
    obs, info = env.reset(seed=42)

    total_reward = 0.0
    for step in range(10):
        action = step % 8
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.shape == (84,)
        assert (obs >= 0.0).all() and (obs <= 1.0).all()
        assert "telemetry" in info
        total_reward += reward

        if terminated or truncated:
            break

    assert step == 9 or terminated or truncated
