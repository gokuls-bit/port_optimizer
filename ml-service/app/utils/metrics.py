"""Performance metrics calculation."""

import numpy as np
from typing import Dict


def calculate_metrics(portfolio_values: np.ndarray, returns: np.ndarray) -> Dict:
    """Calculate comprehensive performance metrics."""
    metrics = {
        "total_return": (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0],
        "annual_return": np.mean(returns) * 252,
        "annual_volatility": np.std(returns) * np.sqrt(252),
        "sharpe_ratio": (np.mean(returns) * 252) / (np.std(returns) * np.sqrt(252) + 1e-8),
    }
    return metrics


def evaluate_model(predictions, actual_values) -> Dict:
    """Evaluate model predictions."""
    mse = np.mean((predictions - actual_values) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - actual_values))

    return {"mse": mse, "rmse": rmse, "mae": mae}
