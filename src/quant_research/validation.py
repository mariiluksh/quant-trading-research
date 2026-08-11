"""Chronology-preserving validation helpers for financial research."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

from quant_research.metrics import annualized_return, hit_rate, maximum_drawdown, sharpe_ratio


@dataclass(frozen=True)
class TrainTestSplit:
    """Structured chronological train/test split."""

    train: pd.Series | pd.DataFrame
    test: pd.Series | pd.DataFrame


@dataclass(frozen=True)
class TrainValidationTestSplit:
    """Structured chronological train/validation/test split."""

    train: pd.Series | pd.DataFrame
    validation: pd.Series | pd.DataFrame
    test: pd.Series | pd.DataFrame


@dataclass(frozen=True)
class WalkForwardWindow:
    """One walk-forward training and testing window."""

    window_number: int
    train: pd.Series | pd.DataFrame
    test: pd.Series | pd.DataFrame


def require_non_empty(name: str, size: int) -> None:
    """Raise a clear error when an expected collection is empty."""

    if size <= 0:
        raise ValueError(f"{name} must not be empty.")


def chronological_train_test_split(
    data: pd.Series | pd.DataFrame,
    *,
    train_size: float = 0.7,
) -> TrainTestSplit:
    """Split time-series data into chronological train and test partitions."""

    validated = _validate_time_series(data, "data")
    split_point = _fractional_split_index(len(validated), train_size, "train_size")
    return TrainTestSplit(
        train=validated.iloc[:split_point].copy(),
        test=validated.iloc[split_point:].copy(),
    )


def chronological_train_validation_test_split(
    data: pd.Series | pd.DataFrame,
    *,
    train_size: float = 0.6,
    validation_size: float = 0.2,
    test_size: float | None = None,
) -> TrainValidationTestSplit:
    """Split time-series data into chronological train, validation, and test sets."""

    validated = _validate_time_series(data, "data")
    total = train_size + validation_size + (test_size if test_size is not None else 0.0)

    if test_size is None:
        remaining = 1.0 - train_size - validation_size
        if remaining <= 0.0:
            raise ValueError("train_size and validation_size must leave room for a non-empty test set.")
        test_size = remaining
    elif not np.isclose(total, 1.0):
        raise ValueError("train_size, validation_size, and test_size must sum to 1.0.")

    _validate_fraction(train_size, "train_size")
    _validate_fraction(validation_size, "validation_size")
    _validate_fraction(test_size, "test_size")

    n_obs = len(validated)
    train_end = int(np.floor(n_obs * train_size))
    validation_end = train_end + int(np.floor(n_obs * validation_size))

    if train_end <= 0 or validation_end <= train_end or validation_end >= n_obs:
        raise ValueError("Split sizes produce an empty train, validation, or test partition.")

    return TrainValidationTestSplit(
        train=validated.iloc[:train_end].copy(),
        validation=validated.iloc[train_end:validation_end].copy(),
        test=validated.iloc[validation_end:].copy(),
    )


def walk_forward_splits(
    data: pd.Series | pd.DataFrame,
    *,
    train_size: int,
    test_size: int,
    step_size: int | None = None,
    expanding: bool = True,
) -> list[WalkForwardWindow]:
    """Construct chronology-preserving walk-forward windows.

    Each test window occurs strictly after its corresponding training window.
    No random shuffling is used, and future observations never enter the train
    segment for a given window.
    """

    validated = _validate_time_series(data, "data")
    _validate_positive_int(train_size, "train_size")
    _validate_positive_int(test_size, "test_size")
    step = test_size if step_size is None else step_size
    _validate_positive_int(step, "step_size")

    windows: list[WalkForwardWindow] = []
    n_obs = len(validated)
    start = 0
    train_end = train_size
    window_number = 1

    while train_end + test_size <= n_obs:
        if expanding:
            train = validated.iloc[:train_end].copy()
        else:
            train = validated.iloc[start:train_end].copy()
        test = validated.iloc[train_end : train_end + test_size].copy()
        windows.append(
            WalkForwardWindow(
                window_number=window_number,
                train=train,
                test=test,
            )
        )
        window_number += 1
        train_end += step
        if not expanding:
            start += step

    if not windows:
        raise ValueError("Not enough observations for even one walk-forward window.")
    return windows


def evaluate_parameter_grid(
    parameter_grid: dict[str, Sequence[Any]],
    evaluator: Callable[[dict[str, Any]], dict[str, Any]],
) -> pd.DataFrame:
    """Evaluate a parameter grid and record results without selecting a winner."""

    if not parameter_grid:
        raise ValueError("parameter_grid must not be empty.")

    keys = list(parameter_grid)
    values = [list(parameter_grid[key]) for key in keys]
    if any(len(items) == 0 for items in values):
        raise ValueError("Each parameter grid entry must contain at least one value.")

    records: list[dict[str, Any]] = []
    for combination in product(*values):
        params = dict(zip(keys, combination, strict=True))
        evaluation = evaluator(params)
        records.append({**params, **evaluation})
    return pd.DataFrame(records)


def stability_analysis(
    results: pd.DataFrame,
    *,
    parameter_column: str,
    metric_column: str,
) -> pd.DataFrame:
    """Measure local parameter stability across neighboring values.

    The result reports the target metric alongside the average of immediate
    neighboring parameter values. Large deviations from the neighborhood mean
    indicate fragile or isolated parameter performance.
    """

    if parameter_column not in results.columns:
        raise ValueError(f"parameter_column {parameter_column!r} not found in results.")
    if metric_column not in results.columns:
        raise ValueError(f"metric_column {metric_column!r} not found in results.")

    ordered = results.sort_values(parameter_column).reset_index(drop=True).copy()
    left_metric = ordered[metric_column].shift(1)
    right_metric = ordered[metric_column].shift(-1)

    neighborhood_mean: list[float] = []
    neighborhood_std: list[float] = []
    neighborhood_count: list[int] = []

    for left, right in zip(left_metric, right_metric, strict=True):
        neighbors = [value for value in (left, right) if pd.notna(value)]
        if neighbors:
            neighborhood_mean.append(float(np.mean(neighbors)))
            neighborhood_std.append(float(np.std(neighbors, ddof=0)))
            neighborhood_count.append(len(neighbors))
        else:
            neighborhood_mean.append(np.nan)
            neighborhood_std.append(np.nan)
            neighborhood_count.append(0)

    ordered["neighborhood_mean"] = neighborhood_mean
    ordered["neighborhood_std"] = neighborhood_std
    ordered["neighborhood_count"] = neighborhood_count
    ordered["stability_gap"] = (ordered[metric_column] - ordered["neighborhood_mean"]).abs()
    return ordered


def transaction_cost_sensitivity(
    cost_bps_values: Sequence[float],
    evaluator: Callable[[float], dict[str, Any]],
) -> pd.DataFrame:
    """Record strategy performance across explicit transaction-cost assumptions.

    This helper preserves the supplied transaction-cost order and does not rank,
    optimize, or select a preferred cost assumption.
    """

    values = list(cost_bps_values)
    require_non_empty("cost_bps_values", len(values))
    if any(value < 0.0 for value in values):
        raise ValueError("Transaction-cost assumptions must be non-negative.")

    records: list[dict[str, Any]] = []
    for cost_bps in values:
        evaluation = evaluator(cost_bps)
        records.append({"transaction_cost_bps": cost_bps, **evaluation})
    return pd.DataFrame(records)


def classify_volatility_regimes(
    volatility: pd.Series,
    *,
    low_quantile: float = 1.0 / 3.0,
    high_quantile: float = 2.0 / 3.0,
) -> pd.Series:
    """Classify historical volatility observations into low, medium, and high.

    This is descriptive only. It labels each observation using quantiles of the
    observed historical volatility series and does not imply a predictive regime
    model.
    """

    series = _validate_time_series(volatility, "volatility")
    if not isinstance(series, pd.Series):
        raise TypeError("volatility must be a pandas Series.")
    _validate_quantile(low_quantile, "low_quantile")
    _validate_quantile(high_quantile, "high_quantile")
    if low_quantile >= high_quantile:
        raise ValueError("low_quantile must be strictly less than high_quantile.")

    clean = pd.to_numeric(series, errors="coerce")
    low_cutoff = clean.quantile(low_quantile)
    high_cutoff = clean.quantile(high_quantile)

    regimes = pd.Series(pd.NA, index=clean.index, dtype="object")
    regimes.loc[clean <= low_cutoff] = "low"
    regimes.loc[(clean > low_cutoff) & (clean < high_cutoff)] = "medium"
    regimes.loc[clean >= high_cutoff] = "high"
    regimes.loc[clean.isna()] = pd.NA
    return regimes.rename("volatility_regime")


def performance_by_regime(
    returns: pd.Series,
    regimes: pd.Series,
    *,
    periods_per_year: int = 252,
    risk_free_rate: float = 0.0,
) -> pd.DataFrame:
    """Summarize realized strategy performance within descriptive volatility regimes."""

    if not isinstance(returns, pd.Series) or not isinstance(regimes, pd.Series):
        raise TypeError("returns and regimes must both be pandas Series.")

    paired = pd.concat(
        [
            pd.to_numeric(returns, errors="coerce").rename("returns"),
            regimes.rename("regime"),
        ],
        axis=1,
        join="inner",
    ).dropna()
    if paired.empty:
        raise ValueError("returns and regimes do not share any non-missing observations.")

    records: list[dict[str, Any]] = []
    for regime_name in ["low", "medium", "high"]:
        regime_returns = paired.loc[paired["regime"] == regime_name, "returns"]
        if regime_returns.empty:
            records.append(
                {
                    "regime": regime_name,
                    "observations": 0,
                    "annual_return": np.nan,
                    "sharpe_ratio": np.nan,
                    "max_drawdown": np.nan,
                    "hit_rate": np.nan,
                }
            )
            continue

        records.append(
            {
                "regime": regime_name,
                "observations": int(regime_returns.size),
                "annual_return": annualized_return(
                    regime_returns,
                    periods_per_year=periods_per_year,
                ),
                "sharpe_ratio": sharpe_ratio(
                    regime_returns,
                    periods_per_year=periods_per_year,
                    risk_free_rate=risk_free_rate,
                ),
                "max_drawdown": maximum_drawdown(regime_returns),
                "hit_rate": hit_rate(regime_returns),
            }
        )

    return pd.DataFrame(records)


def _validate_time_series(
    data: pd.Series | pd.DataFrame,
    name: str,
) -> pd.Series | pd.DataFrame:
    """Validate common time-series assumptions used by this module."""

    if not isinstance(data, (pd.Series, pd.DataFrame)):
        raise TypeError(f"{name} must be a pandas Series or DataFrame.")
    require_non_empty(name, len(data))
    if not data.index.is_monotonic_increasing:
        raise ValueError(f"{name} index must be sorted in chronological order.")
    if data.index.has_duplicates:
        raise ValueError(f"{name} index must not contain duplicate timestamps.")
    return data


def _fractional_split_index(n_obs: int, fraction: float, name: str) -> int:
    """Convert a fractional split size into a safe boundary index."""

    _validate_fraction(fraction, name)
    split_point = int(np.floor(n_obs * fraction))
    if split_point <= 0 or split_point >= n_obs:
        raise ValueError(f"{name} produces an empty train or test partition.")
    return split_point


def _validate_fraction(value: float, name: str) -> None:
    """Reject invalid fractional split sizes."""

    if value <= 0.0 or value >= 1.0:
        raise ValueError(f"{name} must lie strictly between 0 and 1.")


def _validate_positive_int(value: int, name: str) -> None:
    """Reject non-positive integer window sizes."""

    if value <= 0:
        raise ValueError(f"{name} must be a positive integer.")


def _validate_quantile(value: float, name: str) -> None:
    """Reject invalid quantile inputs."""

    if value <= 0.0 or value >= 1.0:
        raise ValueError(f"{name} must lie strictly between 0 and 1.")
