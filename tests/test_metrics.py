"""Tests for performance metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_research.metrics import (
    annualized_return,
    annualized_volatility,
    cumulative_return,
    downside_deviation,
    drawdown_series,
    hit_rate,
    maximum_drawdown,
    rolling_correlation,
    rolling_cumulative_return,
    rolling_sharpe_ratio,
    rolling_volatility,
    running_equity_curve,
    sharpe_ratio,
    sortino_ratio,
    turnover,
)


def test_cumulative_and_annualized_return_match_hand_calculation() -> None:
    """Compounded and annualized returns should match simple examples."""

    returns = pd.Series([0.10, -0.05, 0.02])

    expected_cumulative = (1.10 * 0.95 * 1.02) - 1.0
    expected_annualized = (1.10 * 0.95 * 1.02) ** (12 / 3) - 1.0

    assert cumulative_return(returns) == pytest.approx(expected_cumulative)
    assert annualized_return(returns, periods_per_year=12) == pytest.approx(expected_annualized)


def test_annualized_volatility_and_sharpe_ratio_use_standard_formulas() -> None:
    """Volatility and Sharpe should use sample standard deviation and annualization."""

    returns = pd.Series([0.01, 0.02, -0.01, 0.03])
    mean_return = returns.mean()
    std_return = returns.std(ddof=1)

    expected_vol = std_return * np.sqrt(12)
    expected_sharpe = np.sqrt(12) * mean_return / std_return

    assert annualized_volatility(returns, periods_per_year=12) == pytest.approx(expected_vol)
    assert sharpe_ratio(returns, periods_per_year=12) == pytest.approx(expected_sharpe)


def test_sharpe_and_sortino_apply_risk_free_hurdle() -> None:
    """Risk-free rate should be converted to a per-period hurdle."""

    returns = pd.Series([0.01, 0.02, 0.00, -0.01])
    periods_per_year = 12
    annual_rf = 0.12
    hurdle = (1.0 + annual_rf) ** (1.0 / periods_per_year) - 1.0
    excess = returns - hurdle
    downside = np.minimum(excess, 0.0)

    expected_sharpe = np.sqrt(periods_per_year) * excess.mean() / excess.std(ddof=1)
    expected_downside = np.sqrt(periods_per_year * np.mean(np.square(downside)))
    expected_sortino = (periods_per_year * excess.mean()) / expected_downside

    assert sharpe_ratio(
        returns,
        periods_per_year=periods_per_year,
        risk_free_rate=annual_rf,
    ) == pytest.approx(expected_sharpe)
    assert downside_deviation(
        returns,
        periods_per_year=periods_per_year,
        risk_free_rate=annual_rf,
    ) == pytest.approx(expected_downside)
    assert sortino_ratio(
        returns,
        periods_per_year=periods_per_year,
        risk_free_rate=annual_rf,
    ) == pytest.approx(expected_sortino)


def test_running_equity_curve_drawdowns_and_maximum_drawdown() -> None:
    """Equity and drawdown paths should be hand-verifiable."""

    returns = pd.Series(
        [0.10, -0.20, 0.05],
        index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
    )

    equity = running_equity_curve(returns)
    expected_equity = pd.Series(
        [1.10, 0.88, 0.924],
        index=returns.index,
    )
    expected_drawdowns = pd.Series(
        [0.0, -0.20, -0.16],
        index=returns.index,
    )

    pd.testing.assert_series_equal(equity, expected_equity)
    pd.testing.assert_series_equal(drawdown_series(returns), expected_drawdowns)
    assert maximum_drawdown(returns) == pytest.approx(-0.20)


def test_rolling_volatility_and_sharpe_ratio() -> None:
    """Rolling metrics should only appear once a full window exists."""

    returns = pd.Series([0.01, 0.02, -0.01, 0.03])
    window = 3

    rolling_vol = rolling_volatility(returns, window=window, periods_per_year=12)
    rolling_sharpe = rolling_sharpe_ratio(returns, window=window, periods_per_year=12)

    window_one = returns.iloc[:3]
    expected_vol = window_one.std(ddof=1) * np.sqrt(12)
    expected_sharpe = np.sqrt(12) * window_one.mean() / window_one.std(ddof=1)

    assert np.isnan(rolling_vol.iloc[0])
    assert np.isnan(rolling_vol.iloc[1])
    assert rolling_vol.iloc[2] == pytest.approx(expected_vol)
    assert rolling_sharpe.iloc[2] == pytest.approx(expected_sharpe)


def test_rolling_cumulative_return_and_correlation() -> None:
    """Rolling cumulative return and correlation should match simple examples."""

    left = pd.Series([0.10, 0.00, -0.10, 0.20])
    right = pd.Series([0.05, 0.00, -0.05, 0.10])

    cumulative = rolling_cumulative_return(left, window=2)
    correlation = rolling_correlation(left, right, window=3)

    assert np.isnan(cumulative.iloc[0])
    assert cumulative.iloc[1] == pytest.approx((1.10 * 1.00) - 1.0)
    assert cumulative.iloc[2] == pytest.approx((1.00 * 0.90) - 1.0)

    assert np.isnan(correlation.iloc[0])
    assert np.isnan(correlation.iloc[1])
    assert correlation.iloc[2] == pytest.approx(1.0)
    assert correlation.iloc[3] == pytest.approx(1.0)


def test_hit_rate_ignores_zeros_and_nans() -> None:
    """Hit rate should use non-zero observed returns only."""

    returns = pd.Series([0.02, 0.0, -0.01, np.nan, 0.03])
    assert hit_rate(returns) == pytest.approx(2 / 3)


def test_turnover_uses_one_way_weight_changes() -> None:
    """Turnover should reflect half the gross absolute weight change."""

    weights = pd.DataFrame(
        {
            "AAPL": [0.60, 0.50, 0.50],
            "MSFT": [0.40, 0.30, np.nan],
            "CASH": [0.00, 0.20, 0.50],
        },
        index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
    )

    result = turnover(weights)

    assert np.isnan(result.iloc[0])
    assert result.iloc[1] == pytest.approx(0.20)
    assert result.iloc[2] == pytest.approx(0.30)


def test_metrics_handle_nan_and_zero_volatility_conservatively() -> None:
    """Undefined ratios should produce NaN rather than misleading infinities."""

    constant = pd.Series([0.01, 0.01, 0.01])
    with_nans = pd.Series([0.02, np.nan, -0.01])

    assert np.isnan(sharpe_ratio(constant))
    assert np.isnan(sortino_ratio(constant))
    assert cumulative_return(with_nans) == pytest.approx((1.02 * 0.99) - 1.0)


def test_invalid_inputs_raise_informative_errors() -> None:
    """Bad return inputs and invalid rolling settings should fail explicitly."""

    with pytest.raises(ValueError, match="strictly greater than -100%"):
        cumulative_return(pd.Series([0.1, -1.0]))

    with pytest.raises(ValueError, match="periods_per_year must be a positive integer"):
        annualized_return(pd.Series([0.1]), periods_per_year=0)

    with pytest.raises(ValueError, match="window must be a positive integer"):
        rolling_volatility(pd.Series([0.1, 0.2]), window=0)
