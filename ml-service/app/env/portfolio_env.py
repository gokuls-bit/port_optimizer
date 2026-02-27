"""Custom Gym environment for portfolio optimization."""

import gym
from gym import spaces
import numpy as np


class PortfolioEnvironment(gym.Env):
    """Custom environment for portfolio optimization."""

    def __init__(self, data, initial_balance=100000):
        """Initialize the environment."""
        self.data = data
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.current_step = 0

        # Action space: portfolio weights for each asset
        self.action_space = spaces.Box(low=0, high=1, shape=(10,), dtype=np.float32)

        # Observation space: market data features
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(50,), dtype=np.float32)

    def reset(self):
        """Reset environment."""
        self.current_balance = self.initial_balance
        self.current_step = 0
        return self._get_observation()

    def step(self, action):
        """Execute one step in the environment."""
        self.current_step += 1
        reward = 0
        done = self.current_step >= len(self.data)
        return self._get_observation(), reward, done, {}

    def _get_observation(self):
        """Get current observation."""
        return np.zeros(50, dtype=np.float32)
