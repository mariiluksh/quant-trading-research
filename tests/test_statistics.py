"""Tests for statistical diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_research.statistics import (
    bootstrap_sharpe_confidence_interval,
    distribution_diagnostics,
    mean_return_confidence_interval,
    mean_return_t_statistic,
    return_autocorrelation,
)


def test_mean_return_t_statistic_matches_hand_calculation() -> None:
    """Mean t-statistic should match the standard sample formula."""

    returns = pd.Series([0.01, 0.02, 0.00, -0.01])
    expected = returns.mean() / (returns.std(ddof=1) / np.sqrt(len(returns)))

    assert mean_return_t_statistic(returns) == pytest.approx(expected)


def test_mean_return_confidence_interval_contains_mean() -> None:
    """The mean-return interval should bracket the sample mean in a regular case."""

    returns = pd.Series([0.01, 0.02, 0.00, -0.01, 0.03])
    interval = mean_return_confidence_interval(returns, confidence_level=0.95)

    assert interval.lower <= returns.mean() <= interval.upper


def test_bootstrap_sharpe_confidence_interval_is_reproducible() -> None:
    """Bootstrap Sharpe interval should be reproducible with a fixed seed."""

    returns = pd.Series([0.01, 0.02, -0.01, 0.015, 0.0, 0.01])
    interval = bootstrap_sharpe_confidence_interval(
        returns,
        periods_per_year=12,
        confidence_level=0.90,
        n_bootstrap=200,
        random_seed=7,
    )

    assert interval.lower <= interval.bootstrap_mean <= interval.upper
    assert interval.lower < interval.upper


def test_return_autocorrelation_reports_requested_lags() -> None:
    """Autocorrelation output should preserve the requested lag keys."""

    returns = pd.Series([0.01, 0.02, 0.01, 0.02, 0.01, 0.02])
    result = return_autocorrelation(returns, lags=(1, 2))

    assert result.index.tolist() == [1, 2]
    assert np.isfinite(result.iloc[0])


def test_distribution_diagnostics_summarize_basic_shape() -> None:
    """Distribution diagnostics should expose descriptive sample properties."""

    returns = pd.Series([0.01, -0.02, 0.03, 0.00, 0.02])
    diagnostics = distribution_diagnostics(returns)

    assert diagnostics.observations == 5
    assert diagnostics.mean == pytest.approx(returns.mean())
    assert diagnostics.minimum == pytest.approx(-0.02)
    assert diagnostics.maximum == pytest.approx(0.03)
    assert diagnostics.positive_fraction == pytest.approx(3 / 5)


def test_invalid_statistical_inputs_raise_errors() -> None:
    """Invalid confidence levels and impossible returns should fail explicitly."""

    with pytest.raises(ValueError, match="confidence_level must lie strictly between 0 and 1"):
        mean_return_confidence_interval(pd.Series([0.01, 0.02]), confidence_level=1.0)

    with pytest.raises(ValueError, match="strictly greater than -100%"):
        mean_return_t_statistic(pd.Series([0.01, -1.0]))
