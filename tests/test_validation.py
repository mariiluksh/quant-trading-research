"""Tests for chronology-preserving validation helpers."""

from __future__ import annotations

import pandas as pd
import pytest

from quant_research.validation import (
    classify_volatility_regimes,
    chronological_train_test_split,
    chronological_train_validation_test_split,
    evaluate_parameter_grid,
    performance_by_regime,
    stability_analysis,
    transaction_cost_sensitivity,
    walk_forward_splits,
)


def test_chronological_train_test_split_preserves_order_and_boundaries() -> None:
    """Train/test split should be chronological and non-overlapping."""

    index = pd.date_range("2024-01-01", periods=10, freq="D")
    series = pd.Series(range(10), index=index)

    split = chronological_train_test_split(series, train_size=0.6)

    assert split.train.index.tolist() == index[:6].tolist()
    assert split.test.index.tolist() == index[6:].tolist()
    assert split.train.index.max() < split.test.index.min()


def test_train_validation_test_split_creates_three_ordered_segments() -> None:
    """Train, validation, and test partitions should be clearly separated."""

    index = pd.date_range("2024-01-01", periods=10, freq="D")
    frame = pd.DataFrame({"value": range(10)}, index=index)

    split = chronological_train_validation_test_split(
        frame,
        train_size=0.5,
        validation_size=0.2,
    )

    assert split.train.index.tolist() == index[:5].tolist()
    assert split.validation.index.tolist() == index[5:7].tolist()
    assert split.test.index.tolist() == index[7:].tolist()
    assert split.train.index.max() < split.validation.index.min()
    assert split.validation.index.max() < split.test.index.min()


def test_walk_forward_splits_prevent_future_leakage() -> None:
    """Every walk-forward train window must end before its test window starts."""

    index = pd.date_range("2024-01-01", periods=12, freq="D")
    series = pd.Series(range(12), index=index)

    windows = walk_forward_splits(series, train_size=5, test_size=2, step_size=2, expanding=True)

    assert len(windows) == 3
    assert windows[0].train.index.tolist() == index[:5].tolist()
    assert windows[0].test.index.tolist() == index[5:7].tolist()
    assert windows[1].train.index.tolist() == index[:7].tolist()
    assert windows[1].test.index.tolist() == index[7:9].tolist()
    for window in windows:
        assert window.train.index.max() < window.test.index.min()


def test_parameter_grid_evaluation_records_results_without_reordering() -> None:
    """Grid evaluation should record every parameter choice in supplied order."""

    parameter_grid = {"lookback": [63, 21, 126]}

    def evaluator(params: dict[str, int]) -> dict[str, float]:
        return {"in_sample_sharpe": params["lookback"] / 100.0}

    results = evaluate_parameter_grid(parameter_grid, evaluator)

    assert results["lookback"].tolist() == [63, 21, 126]
    assert results["in_sample_sharpe"].tolist() == [0.63, 0.21, 1.26]


def test_stability_analysis_uses_neighboring_parameter_values() -> None:
    """Stability analysis should compare each point with immediate neighbors."""

    results = pd.DataFrame(
        {
            "lookback": [21, 63, 126],
            "out_of_sample_sharpe": [0.2, 0.8, 0.3],
        }
    )

    stability = stability_analysis(
        results,
        parameter_column="lookback",
        metric_column="out_of_sample_sharpe",
    )

    middle = stability.loc[stability["lookback"] == 63].iloc[0]
    assert middle["neighborhood_mean"] == pytest.approx(0.25)
    assert middle["stability_gap"] == pytest.approx(0.55)


def test_unsorted_or_duplicate_indices_raise_errors() -> None:
    """Validation helpers should reject unsorted or duplicate time indices."""

    unsorted = pd.Series(
        [1, 2],
        index=pd.to_datetime(["2024-01-02", "2024-01-01"]),
    )
    duplicate = pd.Series(
        [1, 2],
        index=pd.to_datetime(["2024-01-01", "2024-01-01"]),
    )

    with pytest.raises(ValueError, match="sorted in chronological order"):
        chronological_train_test_split(unsorted)

    with pytest.raises(ValueError, match="must not contain duplicate timestamps"):
        walk_forward_splits(duplicate, train_size=1, test_size=1)


def test_transaction_cost_sensitivity_records_each_cost_without_reordering() -> None:
    """Cost sensitivity should preserve explicit assumptions and metric outputs."""

    results = transaction_cost_sensitivity(
        [0.0, 5.0, 1.0],
        lambda cost_bps: {"sharpe": 1.0 - cost_bps / 10.0},
    )

    assert results["transaction_cost_bps"].tolist() == [0.0, 5.0, 1.0]
    assert results["sharpe"].tolist() == [1.0, 0.5, 0.9]


def test_transaction_cost_sensitivity_rejects_negative_costs() -> None:
    """Negative cost assumptions should fail explicitly."""

    with pytest.raises(ValueError, match="must be non-negative"):
        transaction_cost_sensitivity([-1.0], lambda _: {"sharpe": 1.0})


def test_classify_volatility_regimes_labels_quantile_buckets() -> None:
    """Volatility regimes should map into low, medium, and high buckets."""

    volatility = pd.Series(
        [0.10, 0.20, 0.30, 0.40, 0.50],
        index=pd.date_range("2024-01-01", periods=5, freq="D"),
    )

    regimes = classify_volatility_regimes(volatility, low_quantile=0.33, high_quantile=0.67)

    assert regimes.iloc[0] == "low"
    assert regimes.iloc[1] == "low"
    assert regimes.iloc[2] == "medium"
    assert regimes.iloc[3] == "high"
    assert regimes.iloc[4] == "high"


def test_performance_by_regime_summarizes_each_bucket() -> None:
    """Performance summary should report metrics separately by descriptive regime."""

    index = pd.date_range("2024-01-01", periods=6, freq="D")
    returns = pd.Series([0.01, 0.02, -0.01, 0.03, -0.02, 0.01], index=index)
    regimes = pd.Series(["low", "low", "medium", "medium", "high", "high"], index=index)

    summary = performance_by_regime(returns, regimes, periods_per_year=252)

    assert summary["regime"].tolist() == ["low", "medium", "high"]
    assert summary["observations"].tolist() == [2, 2, 2]
    low_row = summary.loc[summary["regime"] == "low"].iloc[0]
    assert low_row["hit_rate"] == pytest.approx(1.0)
