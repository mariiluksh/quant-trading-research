"""Performance-metric interfaces for backtest evaluation."""

from dataclasses import dataclass


@dataclass(frozen=True)
class PerformanceSnapshot:
    """Container for summary performance statistics."""

    cumulative_return: float
    annualized_volatility: float
    sharpe_ratio: float
