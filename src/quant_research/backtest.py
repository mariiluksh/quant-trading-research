"""Simple vectorized backtesting engine.

Timing convention:
- target positions or signals observed at time `t` are assumed to be decided
  after the return for period `t` is known,
- therefore they are lagged by one period before they are allowed to earn
  returns,
- position `p_t = signal_{t-1}` earns return `r_t`.

This convention makes the execution assumption explicit and prevents the common
vectorized look-ahead error of multiplying same-period signals by same-period
returns.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quant_research.metrics import (
    annualized_return,
    annualized_volatility,
    cumulative_return,
    hit_rate,
    maximum_drawdown,
    running_equity_curve,
    sharpe_ratio,
    sortino_ratio,
)


@dataclass(frozen=True)
class TradingAssumptions:
    """Execution and transaction-cost assumptions for the backtest."""

    initial_capital: float = 1.0
    periods_per_year: int = 252
    risk_free_rate: float = 0.0
    transaction_cost_bps_per_unit_turnover: float = 0.0
    position_lag: int = 1


@dataclass(frozen=True)
class BacktestResult:
    """Structured result returned by the backtesting engine."""

    positions: pd.Series
    gross_returns: pd.Series
    turnover: pd.Series
    costs: pd.Series
    net_returns: pd.Series
    equity_curve: pd.Series
    summary_metrics: dict[str, float]


def run_vectorized_backtest(
    asset_returns: pd.Series,
    target_positions: pd.Series,
    *,
    assumptions: TradingAssumptions | None = None,
) -> BacktestResult:
    """Run a one-asset vectorized backtest from returns and target positions.

    Parameters
    ----------
    asset_returns:
        Simple asset returns indexed by timestamp.
    target_positions:
        Desired target exposures in the range [-1, 1]. A value of +1 is fully
        long, 0 is flat, and -1 is fully short.
    assumptions:
        Explicit execution and transaction-cost assumptions.
    """

    config = assumptions or TradingAssumptions()
    _validate_assumptions(config)

    clean_returns = _clean_series(asset_returns, "asset_returns")
    clean_targets = _clean_series(target_positions, "target_positions")

    if clean_returns.empty:
        raise ValueError("asset_returns must contain at least one non-missing observation.")
    if clean_targets.empty:
        raise ValueError("target_positions must contain at least one non-missing observation.")

    aligned = pd.concat(
        [clean_returns.rename("asset_return"), clean_targets.rename("target_position")],
        axis=1,
        join="outer",
    ).sort_index()

    if aligned["asset_return"].isna().any():
        raise ValueError("asset_returns and target_positions must share the same index coverage.")
    if aligned["target_position"].isna().any():
        raise ValueError("target_positions must be defined for every asset return timestamp.")

    if (aligned["asset_return"] <= -1.0).any():
        raise ValueError("asset_returns must be strictly greater than -100%.")
    if (aligned["target_position"].abs() > 1.0).any():
        raise ValueError("target_positions must stay within [-1, 1].")

    positions = aligned["target_position"].shift(config.position_lag).fillna(0.0)
    positions.name = "position"

    gross_returns = positions * aligned["asset_return"]
    gross_returns.name = "gross_return"

    turnover = positions.diff().abs().fillna(0.0)
    turnover.name = "turnover"

    cost_rate = config.transaction_cost_bps_per_unit_turnover / 10_000.0
    costs = turnover * cost_rate
    costs.name = "cost"

    net_returns = gross_returns - costs
    net_returns.name = "net_return"

    equity_curve = running_equity_curve(
        net_returns,
        starting_value=config.initial_capital,
    )
    equity_curve.name = "equity"

    summary = _build_summary_metrics(
        net_returns=net_returns,
        turnover=turnover,
        assumptions=config,
    )

    return BacktestResult(
        positions=positions,
        gross_returns=gross_returns,
        turnover=turnover,
        costs=costs,
        net_returns=net_returns,
        equity_curve=equity_curve,
        summary_metrics=summary,
    )


def _build_summary_metrics(
    *,
    net_returns: pd.Series,
    turnover: pd.Series,
    assumptions: TradingAssumptions,
) -> dict[str, float]:
    """Build a compact summary dictionary from net returns."""

    return {
        "cumulative_return": cumulative_return(net_returns),
        "annualized_return": annualized_return(
            net_returns,
            periods_per_year=assumptions.periods_per_year,
        ),
        "annualized_volatility": annualized_volatility(
            net_returns,
            periods_per_year=assumptions.periods_per_year,
        ),
        "sharpe_ratio": sharpe_ratio(
            net_returns,
            periods_per_year=assumptions.periods_per_year,
            risk_free_rate=assumptions.risk_free_rate,
        ),
        "sortino_ratio": sortino_ratio(
            net_returns,
            periods_per_year=assumptions.periods_per_year,
            risk_free_rate=assumptions.risk_free_rate,
        ),
        "maximum_drawdown": maximum_drawdown(
            net_returns,
            starting_value=assumptions.initial_capital,
        ),
        "hit_rate": hit_rate(net_returns),
        "average_turnover": float(turnover.mean()),
    }


def _clean_series(values: pd.Series, name: str) -> pd.Series:
    """Normalize a numeric series while preserving the original index."""

    if not isinstance(values, pd.Series):
        raise TypeError(f"{name} must be a pandas Series.")

    numeric = pd.to_numeric(values, errors="coerce")
    return numeric.dropna().astype(float)


def _validate_assumptions(assumptions: TradingAssumptions) -> None:
    """Reject invalid execution or cost assumptions."""

    if assumptions.initial_capital <= 0.0:
        raise ValueError("initial_capital must be positive.")
    if assumptions.periods_per_year <= 0:
        raise ValueError("periods_per_year must be a positive integer.")
    if assumptions.transaction_cost_bps_per_unit_turnover < 0.0:
        raise ValueError("transaction_cost_bps_per_unit_turnover must be non-negative.")
    if assumptions.position_lag < 1:
        raise ValueError("position_lag must be at least 1 to prevent look-ahead bias.")
