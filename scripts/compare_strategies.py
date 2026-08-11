#!/usr/bin/env python3
"""Compare momentum and mean-reversion on the same assets and time period."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from quant_research.backtest import TradingAssumptions, run_vectorized_backtest
from quant_research.data import download_daily_ohlcv, simple_returns
from quant_research.metrics import (
    annualized_return,
    annualized_volatility,
    maximum_drawdown,
    running_equity_curve,
    sharpe_ratio,
)
from quant_research.signals import mean_reversion_signal, time_series_momentum_signal


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
    parser.add_argument("--use-cache", action="store_true")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--equity-chart-name", default="strategy_comparison_equity_curve.png")
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


def summarize_returns(
    returns: pd.Series,
    *,
    periods_per_year: int,
    risk_free_rate: float,
    average_turnover: float,
) -> dict[str, float]:
    """Build a compact metrics dictionary for reporting."""

    return {
        "annual_return": annualized_return(returns, periods_per_year=periods_per_year),
        "annual_volatility": annualized_volatility(returns, periods_per_year=periods_per_year),
        "sharpe": sharpe_ratio(
            returns,
            periods_per_year=periods_per_year,
            risk_free_rate=risk_free_rate,
        ),
        "max_drawdown": maximum_drawdown(returns),
        "turnover": average_turnover,
    }


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
) -> tuple[pd.Series, pd.Series]:
    """Build equal-weight portfolio returns for the selected strategy."""

    asset_returns = simple_returns(prices).dropna(how="any")
    if asset_returns.empty:
        raise ValueError("Asset return panel is empty after return calculation.")

    strategy_net_by_asset: dict[str, pd.Series] = {}
    turnover_by_asset: dict[str, pd.Series] = {}

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
        turnover_by_asset[symbol] = result.turnover

    strategy_net = pd.DataFrame(strategy_net_by_asset).mean(axis=1)
    turnover = pd.DataFrame(turnover_by_asset).mean(axis=1)
    return strategy_net, turnover


def print_comparison_table(
    *,
    momentum_summary: dict[str, float],
    mean_reversion_summary: dict[str, float],
    buy_hold_summary: dict[str, float],
) -> None:
    """Print strategy metrics side by side."""

    print("Metric                 Momentum        Mean Reversion     Buy-and-Hold")
    print("---------------------------------------------------------------------")
    print(f"Annual return          {momentum_summary['annual_return']:>10.2%}   {mean_reversion_summary['annual_return']:>14.2%}   {buy_hold_summary['annual_return']:>12.2%}")
    print(f"Annual volatility      {momentum_summary['annual_volatility']:>10.2%}   {mean_reversion_summary['annual_volatility']:>14.2%}   {buy_hold_summary['annual_volatility']:>12.2%}")
    print(f"Sharpe                 {momentum_summary['sharpe']:>10.3f}   {mean_reversion_summary['sharpe']:>14.3f}   {buy_hold_summary['sharpe']:>12.3f}")
    print(f"Max drawdown           {momentum_summary['max_drawdown']:>10.2%}   {mean_reversion_summary['max_drawdown']:>14.2%}   {buy_hold_summary['max_drawdown']:>12.2%}")
    print(f"Average turnover       {momentum_summary['turnover']:>10.3f}   {mean_reversion_summary['turnover']:>14.3f}   {buy_hold_summary['turnover']:>12.3f}")


def print_behavior_discussion() -> None:
    """Explain the market-behavior assumptions behind each strategy."""

    print("\nInterpretation")
    print("Momentum implicitly assumes trend persistence: assets that have been rising may keep rising,")
    print("and assets that have been falling may keep falling over the next horizon.")
    print("Mean reversion implicitly assumes temporary dislocations: large moves away from a recent mean")
    print("may partially reverse as prices normalize.")
    print("Neither strategy is universally superior. Their relative behavior depends on market regime,")
    print("trend strength, volatility clustering, reversals, and transaction-cost drag.")


def save_equity_chart(
    *,
    momentum_returns: pd.Series,
    mean_reversion_returns: pd.Series,
    buy_hold_returns: pd.Series,
    initial_capital: float,
    output_dir: Path,
    chart_name: str,
) -> None:
    """Save an equity-curve comparison chart."""

    output_dir.mkdir(parents=True, exist_ok=True)

    momentum_equity = running_equity_curve(momentum_returns, starting_value=initial_capital)
    mean_reversion_equity = running_equity_curve(mean_reversion_returns, starting_value=initial_capital)
    buy_hold_equity = running_equity_curve(buy_hold_returns, starting_value=initial_capital)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(momentum_equity.index, momentum_equity, label="Momentum")
    ax.plot(mean_reversion_equity.index, mean_reversion_equity, label="Mean Reversion")
    ax.plot(buy_hold_equity.index, buy_hold_equity, label="Buy and Hold")
    ax.set_title("Strategy Comparison Equity Curves")
    ax.set_ylabel("Portfolio Value")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / chart_name, dpi=150)
    plt.close(fig)


def main() -> None:
    """Run the strategy comparison."""

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
    buy_hold_returns = simple_returns(prices).dropna(how="any").mean(axis=1)

    momentum_returns, momentum_turnover = strategy_portfolio_returns(
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
    mean_reversion_returns, mean_reversion_turnover = strategy_portfolio_returns(
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

    momentum_summary = summarize_returns(
        momentum_returns,
        periods_per_year=args.periods_per_year,
        risk_free_rate=args.risk_free_rate,
        average_turnover=float(momentum_turnover.mean()),
    )
    mean_reversion_summary = summarize_returns(
        mean_reversion_returns,
        periods_per_year=args.periods_per_year,
        risk_free_rate=args.risk_free_rate,
        average_turnover=float(mean_reversion_turnover.mean()),
    )
    buy_hold_summary = summarize_returns(
        buy_hold_returns,
        periods_per_year=args.periods_per_year,
        risk_free_rate=args.risk_free_rate,
        average_turnover=0.0,
    )

    print_comparison_table(
        momentum_summary=momentum_summary,
        mean_reversion_summary=mean_reversion_summary,
        buy_hold_summary=buy_hold_summary,
    )
    print_behavior_discussion()

    save_equity_chart(
        momentum_returns=momentum_returns,
        mean_reversion_returns=mean_reversion_returns,
        buy_hold_returns=buy_hold_returns,
        initial_capital=args.initial_capital,
        output_dir=Path(args.output_dir),
        chart_name=args.equity_chart_name,
    )


if __name__ == "__main__":
    main()
