"""Training script for RL agents."""

from app.models.ppo_agent import PPOAgent
from app.models.dqn_agent import DQNAgent
from app.env.portfolio_env import PortfolioEnvironment
from app.utils.data_loader import DataLoader
import logging

logger = logging.getLogger(__name__)


def train_ppo(config):
    """Train PPO agent."""
    logger.info("Starting PPO training...")
    # Implementation here
    pass


def train_dqn(config):
    """Train DQN agent."""
    logger.info("Starting DQN training...")
    # Implementation here
    pass
