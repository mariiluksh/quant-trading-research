#!/usr/bin/env python3
"""Run a configurable multi-asset strategy experiment on a diversified universe."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from quant_research.data import download_daily_ohlcv, simple_returns
from quant_research.metrics import running_equity_curve
from quant_research.portfolio import (
    aggregate_portfolio_returns,
    equal_weight_portfolio,
    volatility_scaled_portfolio,
)
from quant_research.signals import mean_reversion_signal, time_series_momentum_signal


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", nargs="+", default=["SPY", "QQQ", "TLT", "GLD"])
    parser.add_argument("--start", default="2010-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--strategy", choices=["momentum", "mean_reversion"], default="momentum")
    parser.add_argument("--weighting", choices=["equal_weight", "volatility_scaled"], default="volatility_scaled")
    parser.add_argument("--max-gross-exposure", type=float, default=1.0)
    parser.add_argument("--allow-short", action="store_true")
    parser.add_argument("--momentum-lookback", type=int, default=126)
    parser.add_argument("--momentum-neutral-band", type=float, default=0.0)
    parser.add_argument("--mean-reversion-lookback", type=int, default=20)
    parser.add_argument("--entry-threshold", type=float, default=1.5)
    parser.add_argument("--exit-threshold", type=float, default=0.5)
    parser.add_argument("--mean-reversion-input", choices=["price", "return"], default="price")
    parser.add_argument("--vol-lookback", type=int, default=20)
    parser.add_argument("--transaction-cost-bps", type=float, default=5.0)
    parser.add_argument("--periods-per-year", type=int, default=252)
    parser.add_argument("--risk-free-rate", type=float, default=0.0)
    parser.add_argument("--initial-capital", type=float, default=1.0)
    parser.add_argument("--use-cache", action="store_true")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--equity-chart-name", default="multi_asset_portfolio_equity_curve.png")
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


def build_target_positions(
    prices: pd.DataFrame,
    *,
    strategy: str,
    momentum_lookback: int,
    momentum_neutral_band: float,
    mean_reversion_lookback: int,
    entry_threshold: float,
    exit_threshold: float,
    mean_reversion_input: str,
) -> pd.DataFrame:
    """Generate per-asset target positions from the selected strategy family."""

    positions: dict[str, pd.Series] = {}
    for symbol in prices.columns:
        if strategy == "momentum":
            signal = time_series_momentum_signal(
                prices[symbol],
                lookback=momentum_lookback,
                neutral_band=momentum_neutral_band,
            )
        elif strategy == "mean_reversion":
            signal = mean_reversion_signal(
                prices[symbol],
                lookback=mean_reversion_lookback,
                entry_threshold=entry_threshold,
                exit_threshold=exit_threshold,
                signal_input=mean_reversion_input,
            )
        else:
            raise ValueError(f"Unsupported strategy: {strategy}.")
        positions[symbol] = signal.target_positions.rename(symbol)

    return pd.concat(positions.values(), axis=1).fillna(0.0)


def construct_weights(
    target_positions: pd.DataFrame,
    asset_returns: pd.DataFrame,
    *,
    weighting: str,
    allow_short: bool,
    max_gross_exposure: float,
    vol_lookback: int,
) -> pd.DataFrame:
    """Construct portfolio weights from target positions."""

    if weighting == "equal_weight":
        return equal_weight_portfolio(
            target_positions,
            max_gross_exposure=max_gross_exposure,
            allow_short=allow_short,
        )
    if weighting == "volatility_scaled":
        return volatility_scaled_portfolio(
            target_positions,
            asset_returns,
            vol_lookback=vol_lookback,
            allow_short=allow_short,
            max_gross_exposure=max_gross_exposure,
        )
    raise ValueError(f"Unsupported weighting scheme: {weighting}.")


def print_summary(result_name: str, summary_metrics: dict[str, float]) -> None:
    """Print high-level portfolio metrics."""

    print(f"\n{result_name}")
    print(f"Annual return:    {summary_metrics['annual_return']:.2%}")
    print(f"Annual volatility:{summary_metrics['annual_volatility']:.2%}")
    print(f"Sharpe ratio:     {summary_metrics['sharpe_ratio']:.3f}")
    print(f"Max drawdown:     {summary_metrics['max_drawdown']:.2%}")
    print(f"Total turnover:   {summary_metrics['total_turnover']:.3f}")


def print_contributions(title: str, series: pd.Series) -> None:
    """Print asset-level contribution summaries."""

    print(f"\n{title}")
    for symbol, value in series.items():
        if pd.isna(value):
            display = "nan"
        else:
            display = f"{value:.6f}"
        print(f"{symbol}: {display}")


def save_equity_chart(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    *,
    initial_capital: float,
    output_dir: Path,
    chart_name: str,
) -> None:
    """Save an equity-curve comparison chart."""

    output_dir.mkdir(parents=True, exist_ok=True)
    portfolio_equity = running_equity_curve(portfolio_returns, starting_value=initial_capital)
    benchmark_equity = running_equity_curve(benchmark_returns, starting_value=initial_capital)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(portfolio_equity.index, portfolio_equity, label="Strategy Portfolio")
    ax.plot(benchmark_equity.index, benchmark_equity, label="Equal-Weight Buy-and-Hold")
    ax.set_title("Multi-Asset Portfolio Equity Curve")
    ax.set_ylabel("Portfolio Value")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / chart_name, dpi=150)
    plt.close(fig)


def main() -> None:
    """Run the multi-asset portfolio experiment."""

    args = parse_args()
    prices = build_price_panel(
        args.symbols,
        start=args.start,
        end=args.end,
        use_cache=args.use_cache,
    )
    asset_returns = simple_returns(prices).dropna(how="any")
    if asset_returns.empty:
        raise ValueError("Asset return panel is empty after return calculation.")

    target_positions = build_target_positions(
        prices.reindex(asset_returns.index),
        strategy=args.strategy,
        momentum_lookback=args.momentum_lookback,
        momentum_neutral_band=args.momentum_neutral_band,
        mean_reversion_lookback=args.mean_reversion_lookback,
        entry_threshold=args.entry_threshold,
        exit_threshold=args.exit_threshold,
        mean_reversion_input=args.mean_reversion_input,
    ).reindex(asset_returns.index).fillna(0.0)

    weights = construct_weights(
        target_positions,
        asset_returns,
        weighting=args.weighting,
        allow_short=args.allow_short,
        max_gross_exposure=args.max_gross_exposure,
        vol_lookback=args.vol_lookback,
    )

    result = aggregate_portfolio_returns(
        asset_returns,
        weights,
        transaction_cost_bps_per_unit_turnover=args.transaction_cost_bps,
        periods_per_year=args.periods_per_year,
        risk_free_rate=args.risk_free_rate,
        initial_capital=args.initial_capital,
    )

    benchmark_weights = equal_weight_portfolio(
        pd.DataFrame(1.0, index=asset_returns.index, columns=asset_returns.columns),
        allow_short=False,
        max_gross_exposure=1.0,
    )
    benchmark = aggregate_portfolio_returns(
        asset_returns,
        benchmark_weights,
        transaction_cost_bps_per_unit_turnover=0.0,
        periods_per_year=args.periods_per_year,
        risk_free_rate=args.risk_free_rate,
        initial_capital=args.initial_capital,
    )

    print_summary("Strategy Portfolio", result.summary_metrics)
    print_summary("Equal-Weight Buy-and-Hold", benchmark.summary_metrics)
    print_contributions("Asset Contribution to Return", result.asset_return_contribution)
    print_contributions("Asset Contribution to Volatility", result.asset_volatility_contribution)

    save_equity_chart(
        result.net_returns,
        benchmark.net_returns,
        initial_capital=args.initial_capital,
        output_dir=Path(args.output_dir),
        chart_name=args.equity_chart_name,
    )


if __name__ == "__main__":
    main()
