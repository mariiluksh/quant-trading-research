"""Statistical diagnostics for strategy return series.

These tools are intended for cautious descriptive research rather than strong
claims of statistical significance. Financial return series often violate the
IID assumptions behind textbook inference because they can exhibit
autocorrelation, volatility clustering, fat tails, and regime shifts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import stats

from quant_research.metrics import sharpe_ratio


@dataclass(frozen=True)
class MeanConfidenceInterval:
    """Confidence interval for the sample mean return."""

    confidence_level: float
    lower: float
    upper: float


@dataclass(frozen=True)
class SharpeBootstrapInterval:
    """Bootstrap confidence interval for the Sharpe ratio."""

    confidence_level: float
    lower: float
    upper: float
    bootstrap_mean: float


@dataclass(frozen=True)
class DistributionDiagnostics:
    """Simple descriptive diagnostics for a return distribution."""

    observations: int
    mean: float
    standard_deviation: float
    skewness: float
    excess_kurtosis: float
    minimum: float
    maximum: float
    positive_fraction: float


def mean_return_t_statistic(returns: pd.Series) -> float:
    """Compute the t-statistic for the sample mean return.

    Formula:
    `t = mean(r) / (std(r) / sqrt(n))`

    Limitation:
    This statistic assumes the sample mean is well-approximated by a standard
    t-test setting. Financial returns can violate that through serial
    dependence, time-varying volatility, and fat tails, so the result should be
    treated as a rough diagnostic rather than proof of significance.
    """

    clean = _clean_return_series(returns)
    if clean.size < 2:
        return np.nan

    sample_std = clean.std(ddof=1)
    if np.isclose(sample_std, 0.0):
        return np.nan
    standard_error = sample_std / np.sqrt(clean.size)
    return float(clean.mean() / standard_error)


def mean_return_confidence_interval(
    returns: pd.Series,
    *,
    confidence_level: float = 0.95,
) -> MeanConfidenceInterval:
    """Compute a t-based confidence interval for the sample mean return.

    Limitation:
    This interval inherits the same IID-style assumptions as the mean
    t-statistic. In financial data, autocorrelation and heteroskedasticity can
    make the interval too narrow or otherwise misleading.
    """

    clean = _clean_return_series(returns)
    _validate_confidence_level(confidence_level)

    if clean.size < 2:
        return MeanConfidenceInterval(confidence_level, np.nan, np.nan)

    sample_std = clean.std(ddof=1)
    if np.isclose(sample_std, 0.0):
        mean = float(clean.mean())
        return MeanConfidenceInterval(confidence_level, mean, mean)

    standard_error = sample_std / np.sqrt(clean.size)
    alpha = 1.0 - confidence_level
    critical_value = stats.t.ppf(1.0 - alpha / 2.0, df=clean.size - 1)
    mean = clean.mean()
    margin = critical_value * standard_error
    return MeanConfidenceInterval(
        confidence_level=confidence_level,
        lower=float(mean - margin),
        upper=float(mean + margin),
    )


def bootstrap_sharpe_confidence_interval(
    returns: pd.Series,
    *,
    periods_per_year: int = 252,
    risk_free_rate: float = 0.0,
    confidence_level: float = 0.95,
    n_bootstrap: int = 1000,
    random_seed: int | None = 0,
) -> SharpeBootstrapInterval:
    """Compute an IID bootstrap confidence interval for the Sharpe ratio.

    Limitation:
    This uses simple resampling with replacement of individual return
    observations. It ignores time dependence and volatility clustering, so it
    should not be presented as a definitive confidence interval when return
    dynamics are serially dependent.
    """

    clean = _clean_return_series(returns)
    _validate_confidence_level(confidence_level)
    _validate_positive_int(n_bootstrap, "n_bootstrap")
    _validate_positive_int(periods_per_year, "periods_per_year")

    if clean.size < 2:
        return SharpeBootstrapInterval(confidence_level, np.nan, np.nan, np.nan)

    rng = np.random.default_rng(random_seed)
    bootstrapped: list[float] = []

    for _ in range(n_bootstrap):
        sample = rng.choice(clean.to_numpy(), size=clean.size, replace=True)
        sample_series = pd.Series(sample)
        value = sharpe_ratio(
            sample_series,
            periods_per_year=periods_per_year,
            risk_free_rate=risk_free_rate,
        )
        if not np.isnan(value):
            bootstrapped.append(float(value))

    if not bootstrapped:
        return SharpeBootstrapInterval(confidence_level, np.nan, np.nan, np.nan)

    alpha = 1.0 - confidence_level
    lower = np.quantile(bootstrapped, alpha / 2.0)
    upper = np.quantile(bootstrapped, 1.0 - alpha / 2.0)
    return SharpeBootstrapInterval(
        confidence_level=confidence_level,
        lower=float(lower),
        upper=float(upper),
        bootstrap_mean=float(np.mean(bootstrapped)),
    )


def return_autocorrelation(
    returns: pd.Series,
    *,
    lags: Sequence[int] = (1, 5, 20),
) -> pd.Series:
    """Compute return autocorrelation at selected lags.

    Limitation:
    Sample autocorrelation is descriptive. Apparent autocorrelation in
    financial returns can be unstable across subperiods and affected by market
    microstructure, sampling frequency, and volatility regimes.
    """

    clean = _clean_return_series(returns)
    if clean.empty:
        return pd.Series(dtype=float)

    values: dict[int, float] = {}
    for lag in lags:
        _validate_positive_int(lag, "lag")
        values[lag] = float(clean.autocorr(lag=lag))
    return pd.Series(values, name="autocorrelation")


def distribution_diagnostics(returns: pd.Series) -> DistributionDiagnostics:
    """Compute simple descriptive diagnostics for a return distribution.

    Limitation:
    These summaries describe the realized sample only. Skewness and kurtosis
    are noisy in finite samples, and distribution shape can change materially
    across regimes.
    """

    clean = _clean_return_series(returns)
    if clean.empty:
        return DistributionDiagnostics(
            observations=0,
            mean=np.nan,
            standard_deviation=np.nan,
            skewness=np.nan,
            excess_kurtosis=np.nan,
            minimum=np.nan,
            maximum=np.nan,
            positive_fraction=np.nan,
        )

    return DistributionDiagnostics(
        observations=int(clean.size),
        mean=float(clean.mean()),
        standard_deviation=float(clean.std(ddof=1)) if clean.size > 1 else np.nan,
        skewness=float(clean.skew()) if clean.size > 2 else np.nan,
        excess_kurtosis=float(clean.kurt()) if clean.size > 3 else np.nan,
        minimum=float(clean.min()),
        maximum=float(clean.max()),
        positive_fraction=float((clean > 0.0).mean()),
    )


def _clean_return_series(returns: pd.Series) -> pd.Series:
    """Validate and normalize return input."""

    if not isinstance(returns, pd.Series):
        raise TypeError("Expected a pandas Series of returns.")

    clean = pd.to_numeric(returns, errors="coerce").dropna().astype(float)
    if (clean <= -1.0).any():
        raise ValueError("Returns must be strictly greater than -100%.")
    return clean


def _validate_confidence_level(confidence_level: float) -> None:
    """Reject invalid confidence levels."""

    if confidence_level <= 0.0 or confidence_level >= 1.0:
        raise ValueError("confidence_level must lie strictly between 0 and 1.")


def _validate_positive_int(value: int, name: str) -> None:
    """Reject non-positive integer parameters."""

    if value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
