"""Tests for multi-asset portfolio construction and aggregation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_research.portfolio import (
    aggregate_portfolio_returns,
    apply_gross_exposure_constraint,
    equal_weight_portfolio,
    volatility_scaled_portfolio,
)


def test_equal_weight_portfolio_sums_to_gross_limit() -> None:
    """Equal-weight construction should allocate equal absolute weight across active assets."""

    index = pd.to_datetime(["2024-01-02", "2024-01-03"])
    positions = pd.DataFrame(
        {
            "SPY": [1.0, 1.0],
            "QQQ": [1.0, -1.0],
            "TLT": [0.0, 0.0],
        },
        index=index,
    )

    weights = equal_weight_portfolio(positions, max_gross_exposure=1.0, allow_short=True)

    assert weights.loc[index[0], "SPY"] == pytest.approx(0.5)
    assert weights.loc[index[0], "QQQ"] == pytest.approx(0.5)
    assert weights.loc[index[1], "SPY"] == pytest.approx(0.5)
    assert weights.loc[index[1], "QQQ"] == pytest.approx(-0.5)
    assert weights.abs().sum(axis=1).tolist() == pytest.approx([1.0, 1.0])


def test_gross_exposure_constraint_scales_rows_down() -> None:
    """Exposure constraint should scale gross weight to the requested cap."""

    weights = pd.DataFrame(
        {"SPY": [0.8], "QQQ": [0.7], "TLT": [-0.5]},
        index=pd.to_datetime(["2024-01-02"]),
    )

    constrained = apply_gross_exposure_constraint(weights, max_gross_exposure=1.0)

    assert constrained.abs().sum(axis=1).iloc[0] == pytest.approx(1.0)
    scale = 1.0 / (0.8 + 0.7 + 0.5)
    assert constrained.iloc[0]["SPY"] == pytest.approx(0.8 * scale)


def test_volatility_scaled_portfolio_respects_exposure_limit() -> None:
    """Volatility-scaled weights should still satisfy the gross exposure cap."""

    index = pd.date_range("2024-01-01", periods=5, freq="D")
    positions = pd.DataFrame(
        {
            "SPY": [1.0, 1.0, 1.0, 1.0, 1.0],
            "QQQ": [1.0, 1.0, 1.0, 1.0, 1.0],
        },
        index=index,
    )
    returns = pd.DataFrame(
        {
            "SPY": [0.01, 0.011, 0.009, 0.010, 0.011],
            "QQQ": [0.02, -0.02, 0.02, -0.02, 0.02],
        },
        index=index,
    )

    weights = volatility_scaled_portfolio(
        positions,
        returns,
        vol_lookback=2,
        max_gross_exposure=1.0,
    )

    assert (weights.abs().sum(axis=1) <= 1.0 + 1e-12).all()
    assert weights.iloc[-1]["SPY"] > weights.iloc[-1]["QQQ"]


def test_portfolio_return_aggregation_uses_lagged_weights() -> None:
    """Portfolio return aggregation should lag target weights by one period."""

    index = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    asset_returns = pd.DataFrame(
        {
            "SPY": [0.10, 0.00, 0.04],
            "QQQ": [0.00, 0.20, -0.02],
        },
        index=index,
    )
    target_weights = pd.DataFrame(
        {
            "SPY": [0.50, 0.50, 0.00],
            "QQQ": [0.50, 0.50, 1.00],
        },
        index=index,
    )

    result = aggregate_portfolio_returns(
        asset_returns,
        target_weights,
        transaction_cost_bps_per_unit_turnover=0.0,
    )

    assert result.executed_weights.iloc[0].tolist() == pytest.approx([0.0, 0.0])
    assert result.executed_weights.iloc[1].tolist() == pytest.approx([0.5, 0.5])
    assert result.gross_returns.iloc[0] == pytest.approx(0.0)
    assert result.gross_returns.iloc[1] == pytest.approx(0.10)
    assert result.gross_returns.iloc[2] == pytest.approx(0.01)
    assert result.turnover.iloc[2] == pytest.approx(0.0)


def test_long_only_option_clips_negative_targets() -> None:
    """Long-only construction should remove short exposure."""

    positions = pd.DataFrame(
        {"SPY": [1.0], "QQQ": [-1.0]},
        index=pd.to_datetime(["2024-01-02"]),
    )

    weights = equal_weight_portfolio(positions, allow_short=False)

    assert weights.iloc[0]["SPY"] == pytest.approx(1.0)
    assert weights.iloc[0]["QQQ"] == pytest.approx(0.0)
