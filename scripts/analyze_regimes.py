#!/usr/bin/env python3
"""Run descriptive rolling and volatility-regime diagnostics for both strategies."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from quant_research.backtest import TradingAssumptions, run_vectorized_backtest
from quant_research.data import download_daily_ohlcv, simple_returns
from quant_research.metrics import (
    drawdown_series,
    rolling_correlation,
    rolling_cumulative_return,
    rolling_sharpe_ratio,
    rolling_volatility,
)
from quant_research.signals import mean_reversion_signal, time_series_momentum_signal
from quant_research.validation import classify_volatility_regimes, performance_by_regime


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
    parser.add_argument(
        "--regime-vol-window",
        type=int,
        default=60,
        help="Rolling volatility window used to define descriptive regimes.",
    )
    parser.add_argument(
        "--rolling-sharpe-window",
        type=int,
        default=60,
        help="Rolling window used for Sharpe diagnostics.",
    )
    parser.add_argument(
        "--rolling-correlation-window",
        type=int,
        default=60,
        help="Rolling window used for correlation diagnostics.",
    )
    parser.add_argument(
        "--rolling-return-window",
        type=int,
        default=20,
        help="Rolling window used for cumulative-return diagnostics.",
    )
    parser.add_argument("--use-cache", action="store_true")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--plot-name", default="rolling_regime_diagnostics.png")
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


def portfolio_returns_for_strategy(
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
    """Build equal-weight portfolio net returns for one strategy."""

    asset_returns = simple_returns(prices).dropna(how="any")
    if asset_returns.empty:
        raise ValueError("Asset return panel is empty after return calculation.")

    net_by_asset: dict[str, pd.Series] = {}
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
        net_by_asset[symbol] = result.net_returns

    return pd.DataFrame(net_by_asset).mean(axis=1)


def print_regime_table(strategy_name: str, regime_summary: pd.DataFrame) -> None:
    """Print descriptive performance by volatility regime."""

    print(f"\n{strategy_name.title()} By Volatility Regime")
    print("Regime    Observations    Annual Return    Sharpe Ratio    Max Drawdown    Hit Rate")
    print("-----------------------------------------------------------------------------------")
    for _, row in regime_summary.iterrows():
        observations = int(row["observations"])
        annual_return = row["annual_return"]
        sharpe = row["sharpe_ratio"]
        drawdown = row["max_drawdown"]
        hit = row["hit_rate"]
        print(
            f"{row['regime']:<8}  {observations:>12}    "
            f"{annual_return:>12.2%}    {sharpe:>12.3f}    {drawdown:>12.2%}    {hit:>8.2%}"
        )


def save_diagnostics_plot(
    *,
    benchmark_returns: pd.Series,
    momentum_returns: pd.Series,
    mean_reversion_returns: pd.Series,
    periods_per_year: int,
    risk_free_rate: float,
    rolling_sharpe_window: int,
    rolling_correlation_window: int,
    rolling_return_window: int,
    output_dir: Path,
    plot_name: str,
) -> None:
    """Save a multi-panel descriptive diagnostics plot."""

    output_dir.mkdir(parents=True, exist_ok=True)

    benchmark_vol_20 = rolling_volatility(benchmark_returns, window=20, periods_per_year=periods_per_year)
    benchmark_vol_60 = rolling_volatility(benchmark_returns, window=60, periods_per_year=periods_per_year)
    benchmark_vol_120 = rolling_volatility(benchmark_returns, window=120, periods_per_year=periods_per_year)

    momentum_sharpe = rolling_sharpe_ratio(
        momentum_returns,
        window=rolling_sharpe_window,
        periods_per_year=periods_per_year,
        risk_free_rate=risk_free_rate,
    )
    mean_reversion_sharpe = rolling_sharpe_ratio(
        mean_reversion_returns,
        window=rolling_sharpe_window,
        periods_per_year=periods_per_year,
        risk_free_rate=risk_free_rate,
    )

    momentum_buyhold_corr = rolling_correlation(
        momentum_returns,
        benchmark_returns,
        window=rolling_correlation_window,
    )
    meanrev_buyhold_corr = rolling_correlation(
        mean_reversion_returns,
        benchmark_returns,
        window=rolling_correlation_window,
    )
    cross_strategy_corr = rolling_correlation(
        momentum_returns,
        mean_reversion_returns,
        window=rolling_correlation_window,
    )

    momentum_roll = rolling_cumulative_return(momentum_returns, window=rolling_return_window)
    meanrev_roll = rolling_cumulative_return(mean_reversion_returns, window=rolling_return_window)
    benchmark_roll = rolling_cumulative_return(benchmark_returns, window=rolling_return_window)

    momentum_drawdown = drawdown_series(momentum_returns)
    meanrev_drawdown = drawdown_series(mean_reversion_returns)
    benchmark_drawdown = drawdown_series(benchmark_returns)

    fig, axes = plt.subplots(5, 1, figsize=(12, 22), sharex=True)

    axes[0].plot(benchmark_vol_20.index, benchmark_vol_20, label="20-day Vol")
    axes[0].plot(benchmark_vol_60.index, benchmark_vol_60, label="60-day Vol")
    axes[0].plot(benchmark_vol_120.index, benchmark_vol_120, label="120-day Vol")
    axes[0].set_title("Rolling Benchmark Volatility")
    axes[0].set_ylabel("Annualized Volatility")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(momentum_sharpe.index, momentum_sharpe, label="Momentum")
    axes[1].plot(mean_reversion_sharpe.index, mean_reversion_sharpe, label="Mean Reversion")
    axes[1].set_title(f"Rolling Sharpe ({rolling_sharpe_window}-Day Window)")
    axes[1].set_ylabel("Sharpe Ratio")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(momentum_buyhold_corr.index, momentum_buyhold_corr, label="Momentum vs Buy-and-Hold")
    axes[2].plot(meanrev_buyhold_corr.index, meanrev_buyhold_corr, label="Mean Reversion vs Buy-and-Hold")
    axes[2].plot(cross_strategy_corr.index, cross_strategy_corr, label="Momentum vs Mean Reversion")
    axes[2].set_title(f"Rolling Correlations ({rolling_correlation_window}-Day Window)")
    axes[2].set_ylabel("Correlation")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(momentum_roll.index, momentum_roll, label="Momentum")
    axes[3].plot(meanrev_roll.index, meanrev_roll, label="Mean Reversion")
    axes[3].plot(benchmark_roll.index, benchmark_roll, label="Buy-and-Hold")
    axes[3].set_title(f"Rolling Strategy Returns ({rolling_return_window}-Day Cumulative)")
    axes[3].set_ylabel("Cumulative Return")
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)

    axes[4].plot(momentum_drawdown.index, momentum_drawdown, label="Momentum")
    axes[4].plot(meanrev_drawdown.index, meanrev_drawdown, label="Mean Reversion")
    axes[4].plot(benchmark_drawdown.index, benchmark_drawdown, label="Buy-and-Hold")
    axes[4].set_title("Drawdowns")
    axes[4].set_ylabel("Drawdown")
    axes[4].legend()
    axes[4].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / plot_name, dpi=150)
    plt.close(fig)


def main() -> None:
    """Run descriptive rolling and volatility-regime diagnostics."""

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
    benchmark_returns = simple_returns(prices).dropna(how="any").mean(axis=1)
    momentum_returns = portfolio_returns_for_strategy(
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
    mean_reversion_returns = portfolio_returns_for_strategy(
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

    benchmark_regime_vol = rolling_volatility(
        benchmark_returns,
        window=args.regime_vol_window,
        periods_per_year=args.periods_per_year,
    )
    regimes = classify_volatility_regimes(benchmark_regime_vol)

    momentum_regimes = performance_by_regime(
        momentum_returns,
        regimes,
        periods_per_year=args.periods_per_year,
        risk_free_rate=args.risk_free_rate,
    )
    mean_reversion_regimes = performance_by_regime(
        mean_reversion_returns,
        regimes,
        periods_per_year=args.periods_per_year,
        risk_free_rate=args.risk_free_rate,
    )
    benchmark_regimes = performance_by_regime(
        benchmark_returns,
        regimes,
        periods_per_year=args.periods_per_year,
        risk_free_rate=args.risk_free_rate,
    )

    print("Volatility regimes are descriptive labels derived from realized rolling volatility quantiles.")
    print("They are not a predictive regime model.\n")
    print_regime_table("momentum", momentum_regimes)
    print_regime_table("mean reversion", mean_reversion_regimes)
    print_regime_table("buy and hold", benchmark_regimes)

    save_diagnostics_plot(
        benchmark_returns=benchmark_returns,
        momentum_returns=momentum_returns,
        mean_reversion_returns=mean_reversion_returns,
        periods_per_year=args.periods_per_year,
        risk_free_rate=args.risk_free_rate,
        rolling_sharpe_window=args.rolling_sharpe_window,
        rolling_correlation_window=args.rolling_correlation_window,
        rolling_return_window=args.rolling_return_window,
        output_dir=Path(args.output_dir),
        plot_name=args.plot_name,
    )


if __name__ == "__main__":
    main()
