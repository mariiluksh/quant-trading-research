"""Tests for signal-generation utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_research.signals import (
    mean_reversion_signal,
    rolling_zscore,
    time_series_momentum_signal,
    trailing_return,
)


def test_trailing_return_matches_hand_calculation() -> None:
    """Trailing returns should match simple price-ratio arithmetic."""

    prices = pd.Series(
        [100.0, 105.0, 110.0, 121.0],
        index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]),
    )

    result = trailing_return(prices, lookback=2)

    assert np.isnan(result.iloc[0])
    assert np.isnan(result.iloc[1])
    assert result.iloc[2] == pytest.approx(0.10)
    assert result.iloc[3] == pytest.approx((121.0 / 105.0) - 1.0)


def test_time_series_momentum_signal_generates_long_short_and_neutral() -> None:
    """Momentum signals should map trailing returns into {-1, 0, 1} positions."""

    prices = pd.Series(
        [100.0, 101.0, 100.4, 98.0, 98.2],
        index=pd.to_datetime(
            ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08"]
        ),
    )

    result = time_series_momentum_signal(prices, lookback=1, neutral_band=0.01)

    assert np.isnan(result.trailing_returns.iloc[0])
    assert result.target_positions.iloc[1] == 0.0
    assert result.target_positions.iloc[2] == 0.0
    assert result.target_positions.iloc[3] == -1.0
    assert result.target_positions.iloc[4] == 0.0
    pd.testing.assert_series_equal(result.raw_signals, result.trailing_returns.rename("raw_signal"))


def test_future_prices_do_not_change_past_signals() -> None:
    """Changing future observations must not alter already-formed signals."""

    index = pd.to_datetime(
        ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08"]
    )
    base_prices = pd.Series([100.0, 101.0, 102.0, 103.0, 104.0], index=index)
    modified_prices = base_prices.copy()
    modified_prices.iloc[-1] = 500.0

    base = time_series_momentum_signal(base_prices, lookback=2)
    modified = time_series_momentum_signal(modified_prices, lookback=2)

    pd.testing.assert_series_equal(
        base.target_positions.iloc[:-1],
        modified.target_positions.iloc[:-1],
    )
    pd.testing.assert_series_equal(
        base.trailing_returns.iloc[:-1],
        modified.trailing_returns.iloc[:-1],
    )


def test_invalid_signal_inputs_raise_errors() -> None:
    """Bad prices, lookbacks, and neutral bands should fail explicitly."""

    prices = pd.Series([100.0, 101.0])

    with pytest.raises(ValueError, match="lookback must be a positive integer"):
        trailing_return(prices, lookback=0)

    with pytest.raises(ValueError, match="neutral_band must be non-negative"):
        time_series_momentum_signal(prices, lookback=1, neutral_band=-0.01)

    with pytest.raises(ValueError, match="Prices must be strictly positive"):
        trailing_return(pd.Series([100.0, 0.0]), lookback=1)


def test_rolling_zscore_matches_hand_calculation() -> None:
    """Rolling z-score should match a simple trailing-window example."""

    values = pd.Series([1.0, 2.0, 3.0, 4.0])
    result = rolling_zscore(values, lookback=3)

    expected_last = (4.0 - 3.0) / np.sqrt(2.0 / 3.0)
    assert np.isnan(result.iloc[0])
    assert np.isnan(result.iloc[1])
    assert result.iloc[2] == pytest.approx((3.0 - 2.0) / np.sqrt(2.0 / 3.0))
    assert result.iloc[3] == pytest.approx(expected_last)


def test_mean_reversion_signal_enters_and_exits_with_hysteresis() -> None:
    """Mean reversion should enter on extremes and exit as the z-score normalizes."""

    prices = pd.Series(
        [100.0, 100.0, 100.0, 90.0, 95.0, 105.0, 101.0],
        index=pd.to_datetime(
            ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08", "2024-01-09", "2024-01-10"]
        ),
    )

    result = mean_reversion_signal(
        prices,
        lookback=3,
        entry_threshold=1.0,
        exit_threshold=0.25,
        signal_input="price",
    )

    assert np.isnan(result.target_positions.iloc[0])
    assert np.isnan(result.target_positions.iloc[1])
    assert np.isnan(result.target_positions.iloc[2])
    assert result.target_positions.iloc[3] == 1.0
    assert result.target_positions.iloc[4] == 0.0
    assert result.target_positions.iloc[5] == -1.0
    assert result.target_positions.iloc[6] == 0.0


def test_mean_reversion_signal_can_use_returns() -> None:
    """Mean reversion should also support return-based z-scores."""

    prices = pd.Series(
        [100.0, 101.0, 100.0, 99.0, 100.0, 100.5],
        index=pd.to_datetime(
            ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08", "2024-01-09"]
        ),
    )

    result = mean_reversion_signal(
        prices,
        lookback=3,
        entry_threshold=0.5,
        exit_threshold=0.1,
        signal_input="return",
    )

    assert result.raw_signals.name == "raw_signal"
    assert result.z_scores.name == "z_score"
    assert set(result.target_positions.dropna().unique()).issubset({-1.0, 0.0, 1.0})


def test_future_prices_do_not_change_past_mean_reversion_signals() -> None:
    """Changing only the future price should not change already-formed z-scores."""

    index = pd.to_datetime(
        ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08", "2024-01-09"]
    )
    base_prices = pd.Series([100.0, 101.0, 102.0, 103.0, 104.0, 105.0], index=index)
    modified_prices = base_prices.copy()
    modified_prices.iloc[-1] = 50.0

    base = mean_reversion_signal(
        base_prices,
        lookback=3,
        entry_threshold=1.0,
        exit_threshold=0.25,
    )
    modified = mean_reversion_signal(
        modified_prices,
        lookback=3,
        entry_threshold=1.0,
        exit_threshold=0.25,
    )

    pd.testing.assert_series_equal(base.z_scores.iloc[:-1], modified.z_scores.iloc[:-1])
    pd.testing.assert_series_equal(
        base.target_positions.iloc[:-1],
        modified.target_positions.iloc[:-1],
    )


def test_invalid_mean_reversion_inputs_raise_errors() -> None:
    """Bad threshold settings and signal inputs should fail explicitly."""

    prices = pd.Series([100.0, 101.0, 102.0])

    with pytest.raises(ValueError, match="entry_threshold must be strictly positive"):
        mean_reversion_signal(prices, lookback=2, entry_threshold=0.0, exit_threshold=0.0)

    with pytest.raises(ValueError, match="exit_threshold must be less than or equal"):
        mean_reversion_signal(prices, lookback=2, entry_threshold=1.0, exit_threshold=1.5)

    with pytest.raises(ValueError, match="signal_input must be either 'price' or 'return'"):
        mean_reversion_signal(
            prices,
            lookback=2,
            entry_threshold=1.0,
            exit_threshold=0.5,
            signal_input="bad",
        )
