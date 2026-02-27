"""Reward function definitions."""

import numpy as np


def calculate_portfolio_return(portfolio_values: np.ndarray) -> float:
    """Calculate portfolio return."""
    if len(portfolio_values) < 2:
        return 0.0
    return (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]


def calculate_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
    """Calculate Sharpe ratio."""
    if len(returns) == 0:
        return 0.0
    excess_returns = returns - risk_free_rate
    return np.mean(excess_returns) / (np.std(excess_returns) + 1e-8)


def calculate_max_drawdown(portfolio_values: np.ndarray) -> float:
    """Calculate maximum drawdown."""
    if len(portfolio_values) == 0:
        return 0.0
    cummax = np.maximum.accumulate(portfolio_values)
    drawdown = (portfolio_values - cummax) / cummax
    return np.min(drawdown)
