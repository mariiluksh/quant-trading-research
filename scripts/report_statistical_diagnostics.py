#!/usr/bin/env python3
"""Generate a cautious statistical diagnostics report for strategy returns."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from quant_research.backtest import TradingAssumptions, run_vectorized_backtest
from quant_research.data import download_daily_ohlcv, simple_returns
from quant_research.signals import mean_reversion_signal, time_series_momentum_signal
from quant_research.statistics import (
    bootstrap_sharpe_confidence_interval,
    distribution_diagnostics,
    mean_return_confidence_interval,
    mean_return_t_statistic,
    return_autocorrelation,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", nargs="+", default=["SPY", "TLT", "GLD"])
    parser.add_argument("--start", default="2010-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--momentum-lookback", type=int, default=126)
    parser.add_argument("--momentum-neutral-band", type=float, default=0.0)
    parser.add_argument("--mean-reversion-lookback", type=int, default=20)
    parser.add_argument("--entry-threshold", type=float, default=1.5)
    parser.add_argument("--exit-threshold", type=float, default=0.5)
    parser.add_argument("--mean-reversion-input", choices=["price", "return"], default="price")
    parser.add_argument("--transaction-cost-bps", type=float, default=5.0)
    parser.add_argument("--periods-per-year", type=int, default=252)
    parser.add_argument("--risk-free-rate", type=float, default=0.0)
    parser.add_argument("--initial-capital", type=float, default=1.0)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--use-cache", action="store_true")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--report-name", default="statistical_diagnostics_report.md")
    parser.add_argument("--distribution-plot-name", default="strategy_return_distributions.png")
    return parser.parse_args()


def build_price_panel(
    symbols: list[str],
    *,
    start: str,
    end: str | None,
    use_cache: bool,
) -> pd.DataFrame:
    """Load adjusted-close prices and align the asset panel."""

    frames = download_daily_ohlcv(
        symbols,
        start=start,
        end=end,
        use_cache=use_cache,
        cache_format="csv",
    )
    adjusted_close = {
        symbol: frame["adjusted_close"].rename(symbol)
        for symbol, frame in frames.items()
    }
    panel = pd.concat(adjusted_close.values(), axis=1, join="inner").sort_index()
    if panel.empty:
        raise ValueError("No overlapping adjusted-close history was available for the selected symbols.")
    return panel


def strategy_portfolio_returns(
    prices: pd.DataFrame,
    *,
    assumptions: TradingAssumptions,
    strategy_name: str,
    momentum_lookback: int,
    momentum_neutral_band: float,
    mean_reversion_lookback: int,
    entry_threshold: float,
    exit_threshold: float,
    mean_reversion_input: str,
) -> pd.Series:
    """Build equal-weight portfolio net returns for the selected strategy."""

    asset_returns = simple_returns(prices).dropna(how="any")
    if asset_returns.empty:
        raise ValueError("Asset return panel is empty after return calculation.")

    strategy_net_by_asset: dict[str, pd.Series] = {}
    for symbol in prices.columns:
        if strategy_name == "momentum":
            signal = time_series_momentum_signal(
                prices[symbol],
                lookback=momentum_lookback,
                neutral_band=momentum_neutral_band,
            )
        elif strategy_name == "mean_reversion":
            signal = mean_reversion_signal(
                prices[symbol],
                lookback=mean_reversion_lookback,
                entry_threshold=entry_threshold,
                exit_threshold=exit_threshold,
                signal_input=mean_reversion_input,
            )
        else:
            raise ValueError(f"Unsupported strategy_name: {strategy_name}.")

        target_positions = signal.target_positions.reindex(asset_returns.index).fillna(0.0)
        result = run_vectorized_backtest(
            asset_returns[symbol],
            target_positions,
            assumptions=assumptions,
        )
        strategy_net_by_asset[symbol] = result.net_returns

    return pd.DataFrame(strategy_net_by_asset).mean(axis=1)


def summarize_statistics(
    returns: pd.Series,
    *,
    periods_per_year: int,
    risk_free_rate: float,
    bootstrap_samples: int,
) -> dict[str, object]:
    """Compute the requested statistical diagnostics for one return series."""

    return {
        "t_stat": mean_return_t_statistic(returns),
        "mean_ci": mean_return_confidence_interval(returns),
        "sharpe_ci": bootstrap_sharpe_confidence_interval(
            returns,
            periods_per_year=periods_per_year,
            risk_free_rate=risk_free_rate,
            n_bootstrap=bootstrap_samples,
        ),
        "autocorrelation": return_autocorrelation(returns, lags=(1, 5, 20)),
        "distribution": distribution_diagnostics(returns),
    }


def render_strategy_section(name: str, diagnostics: dict[str, object]) -> str:
    """Render one strategy diagnostics section as markdown."""

    mean_ci = diagnostics["mean_ci"]
    sharpe_ci = diagnostics["sharpe_ci"]
    autocorr = diagnostics["autocorrelation"]
    distribution = diagnostics["distribution"]

    autocorr_lines = "\n".join(
        [f"- lag {lag}: {value:.4f}" for lag, value in autocorr.items()]
    )

    return f"""## {name}

- Mean return t-statistic: {diagnostics["t_stat"]:.4f}
- Mean return {mean_ci.confidence_level:.0%} confidence interval: [{mean_ci.lower:.6f}, {mean_ci.upper:.6f}]
- Sharpe bootstrap {sharpe_ci.confidence_level:.0%} confidence interval: [{sharpe_ci.lower:.4f}, {sharpe_ci.upper:.4f}]
- Bootstrap Sharpe mean: {sharpe_ci.bootstrap_mean:.4f}

Autocorrelation:
{autocorr_lines}

Distribution diagnostics:
- observations: {distribution.observations}
- mean: {distribution.mean:.6f}
- standard deviation: {distribution.standard_deviation:.6f}
- skewness: {distribution.skewness:.4f}
- excess kurtosis: {distribution.excess_kurtosis:.4f}
- minimum: {distribution.minimum:.6f}
- maximum: {distribution.maximum:.6f}
- positive fraction: {distribution.positive_fraction:.2%}
"""


def limitation_section() -> str:
    """Return the cautionary limitations section for the report."""

    return """## Limitations

- Mean return t-statistic:
  This is a classical IID-style diagnostic. Financial returns often exhibit autocorrelation,
  volatility clustering, and non-normal tails, so the statistic should not be treated as
  definitive proof of significance.
- Confidence interval for mean returns:
  The usual t-based interval can be too narrow when returns are heteroskedastic or serially
  dependent.
- Bootstrap confidence interval for Sharpe ratio:
  The simple bootstrap here resamples individual observations and ignores time dependence,
  so it is not a full solution when return dynamics are path dependent.
- Return autocorrelation:
  Sample autocorrelation is descriptive and can be unstable across subsamples, frequencies,
  and market regimes.
- Distribution diagnostics:
  Skewness and kurtosis are noisy in finite samples and can shift materially across time.

These diagnostics are useful for disciplined comparison, but they are not a license to make
strong claims of statistical significance when financial time-series assumptions are clearly
violated.
"""


def interpretation_section() -> str:
    """Return a balanced comparison of the two strategy families."""

    return """## Interpretation

Momentum and mean reversion are exposed to different market behaviors. Momentum implicitly
benefits from persistence and regime continuation. Mean reversion implicitly benefits from
temporary dislocations and partial reversals.

This report should therefore be read as a diagnostic comparison, not as proof that one
strategy is universally superior. A strategy can look favorable on one sample yet be highly
sensitive to turnover, regime shifts, or implementation assumptions.
"""


def save_distribution_plot(
    momentum_returns: pd.Series,
    mean_reversion_returns: pd.Series,
    *,
    output_dir: Path,
    plot_name: str,
) -> None:
    """Save a simple distribution comparison plot for the two strategies."""

    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hist(momentum_returns.dropna(), bins=40, alpha=0.7, label="Momentum")
    axes[0].hist(mean_reversion_returns.dropna(), bins=40, alpha=0.7, label="Mean Reversion")
    axes[0].set_title("Return Distributions")
    axes[0].set_xlabel("Daily Return")
    axes[0].set_ylabel("Frequency")
    axes[0].legend()

    pd.DataFrame(
        {
            "Momentum": momentum_returns.dropna().sort_values().reset_index(drop=True),
            "Mean Reversion": mean_reversion_returns.dropna().sort_values().reset_index(drop=True),
        }
    ).plot(ax=axes[1], title="Sorted Return Profiles")
    axes[1].set_ylabel("Daily Return")

    fig.tight_layout()
    fig.savefig(output_dir / plot_name, dpi=150)
    plt.close(fig)


def main() -> None:
    """Generate the statistical diagnostics report."""

    args = parse_args()
    assumptions = TradingAssumptions(
        initial_capital=args.initial_capital,
        periods_per_year=args.periods_per_year,
        risk_free_rate=args.risk_free_rate,
        transaction_cost_bps_per_unit_turnover=args.transaction_cost_bps,
        position_lag=1,
    )

    prices = build_price_panel(
        args.symbols,
        start=args.start,
        end=args.end,
        use_cache=args.use_cache,
    )
    momentum_returns = strategy_portfolio_returns(
        prices,
        assumptions=assumptions,
        strategy_name="momentum",
        momentum_lookback=args.momentum_lookback,
        momentum_neutral_band=args.momentum_neutral_band,
        mean_reversion_lookback=args.mean_reversion_lookback,
        entry_threshold=args.entry_threshold,
        exit_threshold=args.exit_threshold,
        mean_reversion_input=args.mean_reversion_input,
    )
    mean_reversion_returns = strategy_portfolio_returns(
        prices,
        assumptions=assumptions,
        strategy_name="mean_reversion",
        momentum_lookback=args.momentum_lookback,
        momentum_neutral_band=args.momentum_neutral_band,
        mean_reversion_lookback=args.mean_reversion_lookback,
        entry_threshold=args.entry_threshold,
        exit_threshold=args.exit_threshold,
        mean_reversion_input=args.mean_reversion_input,
    )

    momentum_stats = summarize_statistics(
        momentum_returns,
        periods_per_year=args.periods_per_year,
        risk_free_rate=args.risk_free_rate,
        bootstrap_samples=args.bootstrap_samples,
    )
    mean_reversion_stats = summarize_statistics(
        mean_reversion_returns,
        periods_per_year=args.periods_per_year,
        risk_free_rate=args.risk_free_rate,
        bootstrap_samples=args.bootstrap_samples,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / args.report_name
    report_text = "\n\n".join(
        [
            "# Statistical Diagnostics Report",
            render_strategy_section("Momentum", momentum_stats),
            render_strategy_section("Mean Reversion", mean_reversion_stats),
            limitation_section(),
            interpretation_section(),
        ]
    )
    report_path.write_text(report_text)
    print(report_text)

    save_distribution_plot(
        momentum_returns,
        mean_reversion_returns,
        output_dir=output_dir,
        plot_name=args.distribution_plot_name,
    )


if __name__ == "__main__":
    main()
