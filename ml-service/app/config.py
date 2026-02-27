"""Configuration settings for the ML service."""

import os
from dataclasses import dataclass


@dataclass
class Config:
    """Application configuration."""

    # Training
    EPOCHS: int = 100
    BATCH_SIZE: int = 32
    LEARNING_RATE: float = 0.001

    # Environment
    PORTFOLIO_SIZE: int = 10
    MAX_STEPS: int = 252  # Trading days

    # Model paths
    MODEL_DIR: str = os.getenv("MODEL_DIR", "./models")
    DATA_DIR: str = os.getenv("DATA_DIR", "./data")

    # API
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", 8000))


config = Config()
