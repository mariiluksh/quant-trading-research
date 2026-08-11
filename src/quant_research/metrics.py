"""Performance metrics computed from return series.

The functions in this module operate on generic return series rather than on a
specific strategy object. Unless noted otherwise, returns are assumed to be
simple period returns, for example daily returns for `periods_per_year=252`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PerformanceSnapshot:
    """Container for a small set of summary performance statistics."""

    cumulative_return: float
    annualized_volatility: float
    sharpe_ratio: float


def cumulative_return(returns: pd.Series) -> float:
    """Compute cumulative simple return.

    Formula:
    `(Π_t (1 + r_t)) - 1`
    """

    clean = _clean_return_series(returns)
    if clean.empty:
        return np.nan
    return float((1.0 + clean).prod() - 1.0)


def annualized_return(
    returns: pd.Series,
    *,
    periods_per_year: int = 252,
) -> float:
    """Compute annualized geometric return.

    Formula:
    `(Π_t (1 + r_t)) ** (periods_per_year / n) - 1`
    """

    clean = _clean_return_series(returns)
    _validate_periods_per_year(periods_per_year)

    if clean.empty:
        return np.nan

    compounded = (1.0 + clean).prod()
    if compounded <= 0:
        raise ValueError(
            "Annualized return is undefined when compounded wealth is non-positive."
        )
    return float(compounded ** (periods_per_year / clean.size) - 1.0)


def annualized_volatility(
    returns: pd.Series,
    *,
    periods_per_year: int = 252,
    ddof: int = 1,
) -> float:
    """Compute annualized volatility from simple returns.

    Formula:
    `std(r) * sqrt(periods_per_year)`
    """

    clean = _clean_return_series(returns)
    _validate_periods_per_year(periods_per_year)

    if clean.size < 2:
        return np.nan

    vol = clean.std(ddof=ddof) * np.sqrt(periods_per_year)
    return float(vol)


def downside_deviation(
    returns: pd.Series,
    *,
    periods_per_year: int = 252,
    risk_free_rate: float = 0.0,
) -> float:
    """Compute annualized downside deviation relative to the per-period hurdle.

    Formula:
    `sqrt(periods_per_year * mean(min(r_t - h, 0)^2))`

    where `h` is the per-period risk-free hurdle implied by the annual
    risk-free rate.
    """

    clean = _clean_return_series(returns)
    _validate_periods_per_year(periods_per_year)

    if clean.empty:
        return np.nan

    hurdle = _per_period_rate(risk_free_rate, periods_per_year)
    downside = np.minimum(clean - hurdle, 0.0)
    value = np.sqrt(periods_per_year * np.mean(np.square(downside)))
    return float(value)


def sharpe_ratio(
    returns: pd.Series,
    *,
    periods_per_year: int = 252,
    risk_free_rate: float = 0.0,
) -> float:
    """Compute the annualized Sharpe ratio from simple returns.

    Formula:
    `sqrt(periods_per_year) * mean(r_t - h) / std(r_t - h)`

    where `h` is the per-period risk-free hurdle implied by the annual
    risk-free rate.
    """

    clean = _clean_return_series(returns)
    _validate_periods_per_year(periods_per_year)

    if clean.size < 2:
        return np.nan

    excess = clean - _per_period_rate(risk_free_rate, periods_per_year)
    volatility = excess.std(ddof=1)
    if np.isclose(volatility, 0.0):
        return np.nan
    return float(np.sqrt(periods_per_year) * excess.mean() / volatility)


def sortino_ratio(
    returns: pd.Series,
    *,
    periods_per_year: int = 252,
    risk_free_rate: float = 0.0,
) -> float:
    """Compute the annualized Sortino ratio from simple returns.

    Formula:
    `(periods_per_year * mean(r_t - h)) / downside_deviation`

    where `h` is the per-period risk-free hurdle implied by the annual
    risk-free rate.
    """

    clean = _clean_return_series(returns)
    _validate_periods_per_year(periods_per_year)

    if clean.empty:
        return np.nan

    excess = clean - _per_period_rate(risk_free_rate, periods_per_year)
    downside = downside_deviation(
        clean,
        periods_per_year=periods_per_year,
        risk_free_rate=risk_free_rate,
    )
    if np.isclose(downside, 0.0):
        return np.nan
    return float((periods_per_year * excess.mean()) / downside)


def running_equity_curve(
    returns: pd.Series,
    *,
    starting_value: float = 1.0,
) -> pd.Series:
    """Convert a return series into a running equity curve.

    Formula:
    `equity_t = starting_value * Π_{i<=t} (1 + r_i)`
    """

    clean = _clean_return_series(returns)
    if clean.empty:
        return clean
    return starting_value * (1.0 + clean).cumprod()


def drawdown_series(
    returns: pd.Series,
    *,
    starting_value: float = 1.0,
) -> pd.Series:
    """Compute the running drawdown series from returns.

    Formula:
    `drawdown_t = equity_t / max_{i<=t}(equity_i) - 1`
    """

    equity = running_equity_curve(returns, starting_value=starting_value)
    if equity.empty:
        return equity
    peak = equity.cummax()
    return equity / peak - 1.0


def maximum_drawdown(
    returns: pd.Series,
    *,
    starting_value: float = 1.0,
) -> float:
    """Compute the worst drawdown observed in the return path."""

    drawdowns = drawdown_series(returns, starting_value=starting_value)
    if drawdowns.empty:
        return np.nan
    return float(drawdowns.min())


def rolling_volatility(
    returns: pd.Series,
    *,
    window: int,
    periods_per_year: int = 252,
    ddof: int = 1,
) -> pd.Series:
    """Compute rolling annualized volatility over a fixed window."""

    clean = _clean_return_series(returns)
    _validate_periods_per_year(periods_per_year)
    _validate_window(window)

    return clean.rolling(window=window).std(ddof=ddof) * np.sqrt(periods_per_year)


def rolling_sharpe_ratio(
    returns: pd.Series,
    *,
    window: int,
    periods_per_year: int = 252,
    risk_free_rate: float = 0.0,
) -> pd.Series:
    """Compute the rolling annualized Sharpe ratio over a fixed window."""

    clean = _clean_return_series(returns)
    _validate_periods_per_year(periods_per_year)
    _validate_window(window)

    hurdle = _per_period_rate(risk_free_rate, periods_per_year)
    excess = clean - hurdle
    rolling_mean = excess.rolling(window=window).mean()
    rolling_std = excess.rolling(window=window).std(ddof=1)
    ratio = np.sqrt(periods_per_year) * rolling_mean / rolling_std
    return ratio.where(~np.isclose(rolling_std, 0.0), np.nan)


def rolling_cumulative_return(
    returns: pd.Series,
    *,
    window: int,
) -> pd.Series:
    """Compute rolling cumulative simple return over a fixed window.

    Formula:
    `(Π_{i=t-window+1}^t (1 + r_i)) - 1`
    """

    clean = _clean_return_series(returns)
    _validate_window(window)
    return (1.0 + clean).rolling(window=window).apply(np.prod, raw=True) - 1.0


def rolling_correlation(
    left: pd.Series,
    right: pd.Series,
    *,
    window: int,
) -> pd.Series:
    """Compute rolling correlation between two aligned return series."""

    _validate_window(window)
    if not isinstance(left, pd.Series) or not isinstance(right, pd.Series):
        raise TypeError("rolling_correlation expects two pandas Series inputs.")

    paired = pd.concat(
        [
            pd.to_numeric(left, errors="coerce").rename("left"),
            pd.to_numeric(right, errors="coerce").rename("right"),
        ],
        axis=1,
        join="inner",
    ).dropna()
    if paired.empty:
        return pd.Series(dtype=float)
    return paired["left"].rolling(window=window).corr(paired["right"])


def hit_rate(returns: pd.Series) -> float:
    """Compute the fraction of non-zero observations that are positive."""

    clean = _clean_return_series(returns)
    if clean.empty:
        return np.nan

    non_zero = clean[clean != 0.0]
    if non_zero.empty:
        return np.nan
    return float((non_zero > 0.0).mean())


def turnover(weights: pd.DataFrame) -> pd.Series:
    """Compute one-way portfolio turnover from weight history.

    Formula:
    `0.5 * Σ_i |w_{t,i} - w_{t-1,i}|`

    Missing asset weights are treated as zero, which is appropriate for assets
    entering or leaving the portfolio universe.
    """

    if not isinstance(weights, pd.DataFrame):
        raise TypeError("Turnover expects a pandas DataFrame of asset weights.")
    if weights.empty:
        return pd.Series(dtype=float)

    numeric = weights.astype(float).fillna(0.0)
    changes = numeric.diff().abs().sum(axis=1) * 0.5
    if not changes.empty:
        changes.iloc[0] = np.nan
    return changes


def _clean_return_series(returns: pd.Series) -> pd.Series:
    """Validate and normalize return input."""

    if not isinstance(returns, pd.Series):
        raise TypeError("Expected a pandas Series of returns.")

    clean = pd.to_numeric(returns, errors="coerce").dropna()
    if (clean <= -1.0).any():
        raise ValueError("Returns must be strictly greater than -100%.")
    return clean.astype(float)


def _per_period_rate(risk_free_rate: float, periods_per_year: int) -> float:
    """Convert an annual risk-free rate into a per-period hurdle."""

    return float((1.0 + risk_free_rate) ** (1.0 / periods_per_year) - 1.0)


def _validate_periods_per_year(periods_per_year: int) -> None:
    """Reject non-positive annualization scales."""

    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be a positive integer.")


def _validate_window(window: int) -> None:
    """Reject invalid rolling-window lengths."""

    if window <= 0:
        raise ValueError("window must be a positive integer.")
