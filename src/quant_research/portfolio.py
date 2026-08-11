"""Multi-asset portfolio construction and aggregation utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quant_research.metrics import (
    annualized_return,
    annualized_volatility,
    maximum_drawdown,
    running_equity_curve,
    sharpe_ratio,
)


@dataclass(frozen=True)
class Position:
    """Single-asset portfolio position."""

    symbol: str
    weight: float


@dataclass(frozen=True)
class PortfolioBacktestResult:
    """Structured result for a multi-asset portfolio backtest."""

    target_weights: pd.DataFrame
    executed_weights: pd.DataFrame
    asset_gross_contributions: pd.DataFrame
    gross_returns: pd.Series
    turnover: pd.Series
    costs: pd.Series
    net_returns: pd.Series
    equity_curve: pd.Series
    asset_return_contribution: pd.Series
    asset_volatility_contribution: pd.Series
    summary_metrics: dict[str, float]


def equal_weight_portfolio(
    target_positions: pd.DataFrame,
    *,
    max_gross_exposure: float = 1.0,
    allow_short: bool = True,
) -> pd.DataFrame:
    """Build equal-absolute-weight portfolio weights from target positions."""

    cleaned = _clean_target_positions(target_positions, allow_short=allow_short)
    signs = np.sign(cleaned)
    active = signs.ne(0.0)
    counts = active.sum(axis=1)

    base = pd.DataFrame(0.0, index=cleaned.index, columns=cleaned.columns)
    valid_rows = counts > 0
    if valid_rows.any():
        base.loc[valid_rows] = signs.loc[valid_rows].div(counts.loc[valid_rows], axis=0)

    scaled = base * max_gross_exposure
    return apply_gross_exposure_constraint(scaled, max_gross_exposure=max_gross_exposure)


def volatility_scaled_portfolio(
    target_positions: pd.DataFrame,
    asset_returns: pd.DataFrame,
    *,
    vol_lookback: int = 20,
    allow_short: bool = True,
    max_gross_exposure: float = 1.0,
) -> pd.DataFrame:
    """Build inverse-volatility-scaled portfolio weights.

    Rolling volatilities are lagged by one period so only previously observed
    return variability influences the weight assigned to the next period.
    """

    _validate_positive_int(vol_lookback, "vol_lookback")
    cleaned = _clean_target_positions(target_positions, allow_short=allow_short)
    returns = _clean_returns_frame(asset_returns)
    _validate_same_shape(cleaned, returns, "target_positions", "asset_returns")

    rolling_vol = returns.rolling(window=vol_lookback).std(ddof=1).shift(1)
    inverse_vol = 1.0 / rolling_vol.replace(0.0, np.nan)
    raw = cleaned * inverse_vol
    gross = raw.abs().sum(axis=1).replace(0.0, np.nan)
    normalized = raw.div(gross, axis=0).fillna(0.0) * max_gross_exposure
    return apply_gross_exposure_constraint(normalized, max_gross_exposure=max_gross_exposure)


def apply_gross_exposure_constraint(
    weights: pd.DataFrame,
    *,
    max_gross_exposure: float = 1.0,
) -> pd.DataFrame:
    """Scale portfolio rows so gross exposure never exceeds the specified maximum."""

    _validate_positive_float(max_gross_exposure, "max_gross_exposure")
    numeric = _clean_weight_frame(weights, "weights")
    gross = numeric.abs().sum(axis=1)
    scale = pd.Series(1.0, index=numeric.index)
    over_limit = gross > max_gross_exposure
    scale.loc[over_limit] = max_gross_exposure / gross.loc[over_limit]
    return numeric.mul(scale, axis=0)


def portfolio_turnover(weights: pd.DataFrame) -> pd.Series:
    """Compute multi-asset portfolio turnover as gross absolute weight change."""

    numeric = _clean_weight_frame(weights, "weights")
    return numeric.diff().abs().sum(axis=1).fillna(0.0).rename("turnover")


def aggregate_portfolio_returns(
    asset_returns: pd.DataFrame,
    target_weights: pd.DataFrame,
    *,
    transaction_cost_bps_per_unit_turnover: float = 0.0,
    periods_per_year: int = 252,
    risk_free_rate: float = 0.0,
    initial_capital: float = 1.0,
    rebalance_lag: int = 1,
) -> PortfolioBacktestResult:
    """Aggregate multi-asset returns into a lagged, cost-aware portfolio backtest."""

    returns = _clean_returns_frame(asset_returns)
    weights = _clean_weight_frame(target_weights, "target_weights")
    _validate_same_shape(weights, returns, "target_weights", "asset_returns")
    _validate_non_negative(transaction_cost_bps_per_unit_turnover, "transaction_cost_bps_per_unit_turnover")
    _validate_positive_int(periods_per_year, "periods_per_year")
    _validate_positive_float(initial_capital, "initial_capital")
    _validate_positive_int(rebalance_lag, "rebalance_lag")

    if (returns <= -1.0).any().any():
        raise ValueError("asset_returns must be strictly greater than -100%.")

    executed_weights = weights.shift(rebalance_lag).fillna(0.0)
    executed_weights = apply_gross_exposure_constraint(
        executed_weights,
        max_gross_exposure=float(weights.abs().sum(axis=1).max()) if not weights.empty else 1.0,
    )
    executed_weights = executed_weights.rename(columns=str)

    asset_gross_contributions = executed_weights * returns
    gross_returns = asset_gross_contributions.sum(axis=1).rename("gross_return")
    turnover = portfolio_turnover(executed_weights)
    costs = (turnover * (transaction_cost_bps_per_unit_turnover / 10_000.0)).rename("cost")
    net_returns = (gross_returns - costs).rename("net_return")
    equity_curve = running_equity_curve(net_returns, starting_value=initial_capital).rename("equity")

    asset_return_contribution = asset_gross_contributions.sum(axis=0).rename("return_contribution")
    asset_volatility_contribution = _asset_volatility_contribution(executed_weights, returns)

    summary_metrics = {
        "annual_return": annualized_return(net_returns, periods_per_year=periods_per_year),
        "annual_volatility": annualized_volatility(net_returns, periods_per_year=periods_per_year),
        "sharpe_ratio": sharpe_ratio(
            net_returns,
            periods_per_year=periods_per_year,
            risk_free_rate=risk_free_rate,
        ),
        "max_drawdown": maximum_drawdown(net_returns, starting_value=initial_capital),
        "total_turnover": float(turnover.sum()),
    }

    return PortfolioBacktestResult(
        target_weights=weights,
        executed_weights=executed_weights,
        asset_gross_contributions=asset_gross_contributions,
        gross_returns=gross_returns,
        turnover=turnover,
        costs=costs,
        net_returns=net_returns,
        equity_curve=equity_curve,
        asset_return_contribution=asset_return_contribution,
        asset_volatility_contribution=asset_volatility_contribution,
        summary_metrics=summary_metrics,
    )


def _asset_volatility_contribution(
    executed_weights: pd.DataFrame,
    asset_returns: pd.DataFrame,
) -> pd.Series:
    """Estimate ex-post asset contribution to portfolio volatility.

    This uses the sample covariance matrix of realized asset returns and the
    average executed weight vector over the sample. It is descriptive rather
    than a predictive risk model.
    """

    if executed_weights.empty or asset_returns.empty:
        return pd.Series(dtype=float, name="volatility_contribution")

    common = executed_weights.index.intersection(asset_returns.index)
    weights = executed_weights.loc[common]
    returns = asset_returns.loc[common]
    if common.empty:
        return pd.Series(dtype=float, name="volatility_contribution")

    average_weights = weights.mean(axis=0)
    covariance = returns.cov()
    portfolio_variance = float(average_weights.T @ covariance @ average_weights)
    if np.isclose(portfolio_variance, 0.0):
        return pd.Series(np.nan, index=returns.columns, name="volatility_contribution")

    portfolio_vol = np.sqrt(portfolio_variance)
    marginal = covariance @ average_weights
    contribution = average_weights * marginal / portfolio_vol
    contribution.name = "volatility_contribution"
    return contribution


def _clean_target_positions(target_positions: pd.DataFrame, *, allow_short: bool) -> pd.DataFrame:
    """Validate and normalize target-position inputs."""

    cleaned = _clean_weight_frame(target_positions, "target_positions")
    if not allow_short and (cleaned < 0.0).any().any():
        cleaned = cleaned.clip(lower=0.0)
    if (cleaned.abs() > 1.0).any().any():
        raise ValueError("target_positions must stay within [-1, 1].")
    return cleaned


def _clean_returns_frame(asset_returns: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize an asset-return matrix."""

    returns = _clean_weight_frame(asset_returns, "asset_returns")
    return returns


def _clean_weight_frame(values: pd.DataFrame, name: str) -> pd.DataFrame:
    """Validate and normalize aligned two-dimensional numeric inputs."""

    if not isinstance(values, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame.")
    if values.empty:
        raise ValueError(f"{name} must not be empty.")
    if not values.index.is_monotonic_increasing:
        raise ValueError(f"{name} index must be sorted in chronological order.")
    if values.index.has_duplicates:
        raise ValueError(f"{name} index must not contain duplicate timestamps.")
    if values.columns.has_duplicates:
        raise ValueError(f"{name} columns must not contain duplicate symbols.")
    return values.astype(float).fillna(0.0)


def _validate_same_shape(
    left: pd.DataFrame,
    right: pd.DataFrame,
    left_name: str,
    right_name: str,
) -> None:
    """Require identical index and column alignment."""

    if not left.index.equals(right.index):
        raise ValueError(f"{left_name} and {right_name} must share the same index.")
    if list(left.columns) != list(right.columns):
        raise ValueError(f"{left_name} and {right_name} must share the same columns in the same order.")


def _validate_positive_int(value: int, name: str) -> None:
    """Reject non-positive integer parameters."""

    if value <= 0:
        raise ValueError(f"{name} must be a positive integer.")


def _validate_positive_float(value: float, name: str) -> None:
    """Reject non-positive floating-point parameters."""

    if value <= 0.0:
        raise ValueError(f"{name} must be positive.")


def _validate_non_negative(value: float, name: str) -> None:
    """Reject negative parameters."""

    if value < 0.0:
        raise ValueError(f"{name} must be non-negative.")
