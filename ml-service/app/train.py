"""Training script for RL agents."""

import logging
from app.models.ppo_agent import PPOAgent
from app.models.dqn_agent import DQNAgent
from app.env.portfolio_env import PortfolioEnvironment
from app.utils.data_loader import DataLoader

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# ---------------------------
# PPO Training
# ---------------------------

def train_ppo(config):
    """Train PPO agent."""
    logger.info("Starting PPO training...")

    # Load data
    data_loader = DataLoader(config.DATA_DIR)
    data = data_loader.load_data()

    # Initialize environment
    env = PortfolioEnvironment(
        data=data,
        portfolio_size=config.PORTFOLIO_SIZE,
        max_steps=config.MAX_STEPS
    )

    # Initialize agent
    agent = PPOAgent(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        lr=config.LEARNING_RATE
    )

    # Training loop
    for episode in range(config.EPOCHS):
        state = env.reset()
        total_reward = 0

        done = False
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)

            agent.store_transition(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

        agent.update()

        logger.info(f"PPO Episode {episode+1}/{config.EPOCHS} | Reward: {total_reward:.2f}")

    # Save model
    agent.save(f"{config.MODEL_DIR}/ppo_model.pth")
    logger.info("PPO training completed and model saved.")


# ---------------------------
# DQN Training
# ---------------------------

def train_dqn(config):
    """Train DQN agent."""
    logger.info("Starting DQN training...")

    # Load data
    data_loader = DataLoader(config.DATA_DIR)
    data = data_loader.load_data()

    # Initialize environment
    env = PortfolioEnvironment(
        data=data,
        portfolio_size=config.PORTFOLIO_SIZE,
        max_steps=config.MAX_STEPS
    )

    # Initialize agent
    agent = DQNAgent(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        lr=config.LEARNING_RATE
    )

    # Training loop
    for episode in range(config.EPOCHS):
        state = env.reset()
        total_reward = 0

        done = False
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)

            agent.store_transition(state, action, reward, next_state, done)
            agent.update()  # DQN updates more frequently

            state = next_state
            total_reward += reward

        logger.info(f"DQN Episode {episode+1}/{config.EPOCHS} | Reward: {total_reward:.2f}")

    # Save model
    agent.save(f"{config.MODEL_DIR}/dqn_model.pth")
    logger.info("DQN training completed and model saved.")
