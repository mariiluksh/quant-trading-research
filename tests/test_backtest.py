"""Tests for the vectorized backtesting engine."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_research.backtest import TradingAssumptions, run_vectorized_backtest


def test_future_information_cannot_affect_past_returns() -> None:
    """A same-day signal spike must not earn the same day's return."""

    index = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    asset_returns = pd.Series([0.50, 0.00, 0.00], index=index)
    target_positions = pd.Series([1.0, 0.0, 0.0], index=index)

    result = run_vectorized_backtest(asset_returns, target_positions)

    assert result.positions.iloc[0] == 0.0
    assert result.gross_returns.iloc[0] == 0.0
    assert result.positions.iloc[1] == 1.0
    assert result.gross_returns.iloc[1] == 0.0


def test_transaction_costs_reduce_performance_correctly() -> None:
    """Turnover-based costs should subtract directly from gross returns."""

    index = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    asset_returns = pd.Series([0.00, 0.10, 0.10], index=index)
    target_positions = pd.Series([0.0, 1.0, 1.0], index=index)
    assumptions = TradingAssumptions(transaction_cost_bps_per_unit_turnover=100.0)

    result = run_vectorized_backtest(asset_returns, target_positions, assumptions=assumptions)

    assert result.turnover.iloc[0] == 0.0
    assert result.turnover.iloc[1] == 0.0
    assert result.turnover.iloc[2] == 1.0
    assert result.costs.iloc[2] == pytest.approx(0.01)
    assert result.gross_returns.iloc[2] == pytest.approx(0.10)
    assert result.net_returns.iloc[2] == pytest.approx(0.09)


def test_no_trade_strategy_produces_zero_turnover() -> None:
    """A flat strategy should have no turnover, no costs, and flat equity."""

    index = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    asset_returns = pd.Series([0.02, -0.03, 0.01], index=index)
    target_positions = pd.Series([0.0, 0.0, 0.0], index=index)

    result = run_vectorized_backtest(asset_returns, target_positions)

    assert (result.turnover == 0.0).all()
    assert (result.costs == 0.0).all()
    assert (result.net_returns == 0.0).all()
    assert (result.equity_curve == 1.0).all()


def test_long_short_sign_handling_is_correct() -> None:
    """Short positions should profit from negative returns and lose on positive returns."""

    index = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"])
    asset_returns = pd.Series([0.00, -0.02, 0.03, -0.01], index=index)
    target_positions = pd.Series([1.0, -1.0, 0.5, -0.5], index=index)

    result = run_vectorized_backtest(asset_returns, target_positions)

    expected_positions = pd.Series([0.0, 1.0, -1.0, 0.5], index=index, name="position")
    expected_gross = pd.Series([0.0, -0.02, -0.03, -0.005], index=index, name="gross_return")

    pd.testing.assert_series_equal(result.positions, expected_positions)
    pd.testing.assert_series_equal(result.gross_returns, expected_gross)


def test_summary_metrics_and_equity_curve_use_net_returns() -> None:
    """Summary metrics should be built from net returns rather than gross returns."""

    index = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    asset_returns = pd.Series([0.00, 0.10, 0.10], index=index)
    target_positions = pd.Series([0.0, 1.0, 1.0], index=index)
    assumptions = TradingAssumptions(transaction_cost_bps_per_unit_turnover=100.0)

    result = run_vectorized_backtest(asset_returns, target_positions, assumptions=assumptions)

    expected_equity_last = 1.0 * (1.0 + 0.0) * (1.0 + 0.0) * (1.0 + 0.09)
    assert result.equity_curve.iloc[-1] == pytest.approx(expected_equity_last)
    assert result.summary_metrics["cumulative_return"] == pytest.approx(0.09)


def test_invalid_positions_and_missing_coverage_raise_errors() -> None:
    """Invalid exposure ranges and missing timestamps should fail explicitly."""

    index = pd.to_datetime(["2024-01-02", "2024-01-03"])
    returns = pd.Series([0.01, 0.02], index=index)

    with pytest.raises(ValueError, match="within \\[-1, 1\\]"):
        run_vectorized_backtest(returns, pd.Series([0.0, 1.5], index=index))

    with pytest.raises(ValueError, match="defined for every asset return timestamp"):
        run_vectorized_backtest(returns, pd.Series([0.0], index=index[:1]))


def test_position_lag_must_remain_positive() -> None:
    """Disabling lag should be rejected rather than allowed silently."""

    assumptions = TradingAssumptions(position_lag=0)
    returns = pd.Series([0.01], index=pd.to_datetime(["2024-01-02"]))
    signals = pd.Series([0.0], index=pd.to_datetime(["2024-01-02"]))

    with pytest.raises(ValueError, match="position_lag must be at least 1"):
        run_vectorized_backtest(returns, signals, assumptions=assumptions)
