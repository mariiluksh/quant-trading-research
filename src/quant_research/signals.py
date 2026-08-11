"""Transparent signal-generation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd


class SignalGenerator(Protocol):
    """Protocol for transparent signal-generation components."""

    def name(self) -> str:
        """Return a human-readable strategy name."""


@dataclass(frozen=True)
class MomentumSignalResult:
    """Container for time-series momentum signal components."""

    trailing_returns: pd.Series
    raw_signals: pd.Series
    target_positions: pd.Series


@dataclass(frozen=True)
class MeanReversionSignalResult:
    """Container for mean-reversion signal components."""

    z_scores: pd.Series
    raw_signals: pd.Series
    target_positions: pd.Series


def trailing_return(prices: pd.Series, *, lookback: int) -> pd.Series:
    """Compute trailing simple return over a fixed lookback window.

    Formula:
    `price_t / price_{t-lookback} - 1`
    """

    clean = _clean_price_series(prices)
    _validate_lookback(lookback)
    return clean / clean.shift(lookback) - 1.0


def time_series_momentum_signal(
    prices: pd.Series,
    *,
    lookback: int,
    neutral_band: float = 0.0,
) -> MomentumSignalResult:
    """Generate a transparent time-series momentum signal from a price series.

    The raw signal is the trailing return itself. The target position is:
    - `+1` if trailing return is above `neutral_band`,
    - `-1` if trailing return is below `-neutral_band`,
    - `0` otherwise.

    The function does not lag positions. Execution lag belongs in the
    backtester so the timing convention is explicit in one place.
    """

    _validate_neutral_band(neutral_band)
    trailing = trailing_return(prices, lookback=lookback)
    raw_signals = trailing.rename("raw_signal")
    epsilon = np.finfo(float).eps * 10.0

    target_positions = pd.Series(0.0, index=trailing.index, name="target_position")
    target_positions = target_positions.mask(trailing > neutral_band + epsilon, 1.0)
    target_positions = target_positions.mask(trailing < -neutral_band - epsilon, -1.0)
    target_positions = target_positions.where(~trailing.isna(), np.nan)

    return MomentumSignalResult(
        trailing_returns=trailing.rename("trailing_return"),
        raw_signals=raw_signals,
        target_positions=target_positions,
    )


def rolling_zscore(values: pd.Series, *, lookback: int) -> pd.Series:
    """Compute a rolling z-score using information available up to each time.

    Formula:
    `(x_t - mean_t) / std_t`

    where `mean_t` and `std_t` are computed over the trailing `lookback`
    observations ending at `t`.
    """

    clean = _clean_numeric_series(values, "values")
    _validate_lookback(lookback)

    rolling_mean = clean.rolling(window=lookback).mean()
    rolling_std = clean.rolling(window=lookback).std(ddof=0)
    zscore = (clean - rolling_mean) / rolling_std
    return zscore.where(~np.isclose(rolling_std, 0.0), np.nan)


def mean_reversion_signal(
    prices: pd.Series,
    *,
    lookback: int,
    entry_threshold: float,
    exit_threshold: float,
    signal_input: str = "price",
) -> MeanReversionSignalResult:
    """Generate a simple mean-reversion signal from rolling z-scores.

    Parameters
    ----------
    prices:
        Price series used directly when `signal_input="price"`.
    lookback:
        Rolling z-score window length.
    entry_threshold:
        Enter long when z-score <= `-entry_threshold`, enter short when
        z-score >= `entry_threshold`.
    exit_threshold:
        Exit a long when z-score >= `-exit_threshold`, exit a short when
        z-score <= `exit_threshold`.
    signal_input:
        Either `"price"` or `"return"`. With `"return"`, the z-score is
        computed on simple returns rather than price levels.
    """

    _validate_thresholds(entry_threshold=entry_threshold, exit_threshold=exit_threshold)
    base_series = _signal_input_series(prices, signal_input=signal_input)
    z_scores = rolling_zscore(base_series, lookback=lookback).rename("z_score")
    raw_signals = z_scores.rename("raw_signal")
    target_positions = _zscore_positions(
        z_scores,
        entry_threshold=entry_threshold,
        exit_threshold=exit_threshold,
    )

    return MeanReversionSignalResult(
        z_scores=z_scores,
        raw_signals=raw_signals,
        target_positions=target_positions,
    )


def _clean_price_series(prices: pd.Series) -> pd.Series:
    """Validate and normalize a price series."""

    if not isinstance(prices, pd.Series):
        raise TypeError("Expected a pandas Series of prices.")

    clean = pd.to_numeric(prices, errors="coerce").astype(float)
    if (clean.dropna() <= 0.0).any():
        raise ValueError("Prices must be strictly positive.")
    return clean


def _clean_numeric_series(values: pd.Series, name: str) -> pd.Series:
    """Validate and normalize a numeric series."""

    if not isinstance(values, pd.Series):
        raise TypeError(f"Expected a pandas Series for {name}.")
    return pd.to_numeric(values, errors="coerce").astype(float)


def _signal_input_series(prices: pd.Series, *, signal_input: str) -> pd.Series:
    """Choose the series used to build the mean-reversion z-score."""

    clean_prices = _clean_price_series(prices)
    if signal_input == "price":
        return clean_prices
    if signal_input == "return":
        return clean_prices.pct_change(fill_method=None)
    raise ValueError("signal_input must be either 'price' or 'return'.")


def _zscore_positions(
    z_scores: pd.Series,
    *,
    entry_threshold: float,
    exit_threshold: float,
) -> pd.Series:
    """Map z-scores into long, short, and flat target positions."""

    positions = pd.Series(np.nan, index=z_scores.index, name="target_position")
    current_position = 0.0

    for timestamp, z_value in z_scores.items():
        if pd.isna(z_value):
            positions.loc[timestamp] = np.nan
            continue

        if current_position == 0.0:
            if z_value <= -entry_threshold:
                current_position = 1.0
            elif z_value >= entry_threshold:
                current_position = -1.0
        elif current_position > 0.0:
            if z_value >= -exit_threshold:
                current_position = 0.0
        else:
            if z_value <= exit_threshold:
                current_position = 0.0

        positions.loc[timestamp] = current_position

    return positions


def _validate_lookback(lookback: int) -> None:
    """Reject invalid trailing-window lengths."""

    if lookback <= 0:
        raise ValueError("lookback must be a positive integer.")


def _validate_neutral_band(neutral_band: float) -> None:
    """Reject negative neutral bands."""

    if neutral_band < 0.0:
        raise ValueError("neutral_band must be non-negative.")


def _validate_thresholds(*, entry_threshold: float, exit_threshold: float) -> None:
    """Reject inconsistent mean-reversion threshold settings."""

    if entry_threshold <= 0.0:
        raise ValueError("entry_threshold must be strictly positive.")
    if exit_threshold < 0.0:
        raise ValueError("exit_threshold must be non-negative.")
    if exit_threshold > entry_threshold:
        raise ValueError("exit_threshold must be less than or equal to entry_threshold.")
