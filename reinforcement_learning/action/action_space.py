"""Action Space Definition for Electronic Warfare Smart Scan RL.

Defines the discrete dwell scheduling action space:
Action a in {0, ..., num_emitters - 1} allocates the receiver's directional
beam to dwell upon emitter a for the current decision window.
"""

from __future__ import annotations

from typing import Optional, Sequence
import numpy as np
from gymnasium import spaces


class RLActionSpace:
    """Manages the action space and validation for sensor dwell allocation."""

    def __init__(self, num_emitters: int = 8) -> None:
        if num_emitters < 1:
            raise ValueError(f"num_emitters must be at least 1, got {num_emitters}")
        self.num_emitters = num_emitters
        self.space = spaces.Discrete(num_emitters)

    def sample(self, mask: Optional[Sequence[int]] = None) -> int:
        """Sample an action, optionally restricted by a binary availability mask."""
        if mask is not None:
            mask_arr = np.array(mask, dtype=bool)
            if not mask_arr.any():
                return int(self.space.sample())
            valid_indices = np.where(mask_arr)[0]
            return int(np.random.choice(valid_indices))
        return int(self.space.sample())

    def is_valid(self, action: int) -> bool:
        """Check if an action is within valid discrete bounds."""
        return 0 <= action < self.num_emitters

    @property
    def n(self) -> int:
        """Return the number of discrete actions."""
        return self.num_emitters
