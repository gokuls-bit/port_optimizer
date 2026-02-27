"""Data loading utilities."""

import logging
import pandas as pd

logger = logging.getLogger(__name__)


class DataLoader:
    """Load and preprocess data for training."""

    def __init__(self, data_path: str):
        """Initialize data loader."""
        self.data_path = data_path
        self.data = None

    def load_data(self) -> pd.DataFrame:
        """Load data from file."""
        logger.info(f"Loading data from {self.data_path}")
        self.data = pd.read_csv(self.data_path)
        return self.data

    def preprocess(self) -> pd.DataFrame:
        """Preprocess data."""
        logger.info("Preprocessing data...")
        # Implementation here
        return self.data
