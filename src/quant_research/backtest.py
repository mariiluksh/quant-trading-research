"""Backtesting primitives kept separate from strategy logic."""

from dataclasses import dataclass


@dataclass(frozen=True)
class TradingAssumptions:
    """Minimal container for explicit trading assumptions."""

    initial_capital: float
    rebalance_frequency: str
