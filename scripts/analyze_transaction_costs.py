#!/usr/bin/env python3
"""Evaluate transaction-cost sensitivity for momentum and mean reversion."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from quant_research.backtest import TradingAssumptions, run_vectorized_backtest
from quant_research.data import download_daily_ohlcv, simple_returns
from quant_research.metrics import annualized_return, maximum_drawdown, sharpe_ratio
from quant_research.signals import mean_reversion_signal, time_series_momentum_signal
from quant_research.validation import transaction_cost_sensitivity


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", nargs="+", default=["SPY", "TLT", "GLD"])
    parser.add_argument("--start", default="2010-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--cost-bps", nargs="+", type=float, default=[0.0, 1.0, 5.0, 10.0, 20.0])
    parser.add_argument("--momentum-lookback", type=int, default=126)
    parser.add_argument("--momentum-neutral-band", type=float, default=0.0)
    parser.add_argument("--mean-reversion-lookback", type=int, default=20)
    parser.add_argument("--entry-threshold", type=float, default=1.5)
    parser.add_argument("--exit-threshold", type=float, default=0.5)
    parser.add_argument("--mean-reversion-input", choices=["price", "return"], default="price")
    parser.add_argument("--periods-per-year", type=int, default=252)
    parser.add_argument("--risk-free-rate", type=float, default=0.0)
    parser.add_argument("--initial-capital", type=float, default=1.0)
    parser.add_argument("--use-cache", action="store_true")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--sharpe-chart-name", default="transaction_cost_sensitivity.png")
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


def portfolio_backtest(
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
    """Build equal-weight portfolio net returns and turnover for one strategy."""

    asset_returns = simple_returns(prices).dropna(how="any")
    if asset_returns.empty:
        raise ValueError("Asset return panel is empty after return calculation.")

    net_by_asset: dict[str, pd.Series] = {}
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
        net_by_asset[symbol] = result.net_returns
        turnover_by_asset[symbol] = result.turnover

    portfolio_returns = pd.DataFrame(net_by_asset).mean(axis=1)
    portfolio_turnover = pd.DataFrame(turnover_by_asset).mean(axis=1)
    return portfolio_returns, portfolio_turnover


def summarize_cost_case(
    returns: pd.Series,
    turnover: pd.Series,
    *,
    assumptions: TradingAssumptions,
) -> dict[str, float]:
    """Build the cost-sensitivity metrics for one strategy and cost level."""

    return {
        "net_annual_return": annualized_return(
            returns,
            periods_per_year=assumptions.periods_per_year,
        ),
        "sharpe_ratio": sharpe_ratio(
            returns,
            periods_per_year=assumptions.periods_per_year,
            risk_free_rate=assumptions.risk_free_rate,
        ),
        "max_drawdown": maximum_drawdown(returns),
        "total_turnover": float(turnover.sum()),
    }


def evaluate_strategy_cost_sensitivity(
    prices: pd.DataFrame,
    *,
    strategy_name: str,
    cost_bps_values: list[float],
    periods_per_year: int,
    risk_free_rate: float,
    initial_capital: float,
    momentum_lookback: int,
    momentum_neutral_band: float,
    mean_reversion_lookback: int,
    entry_threshold: float,
    exit_threshold: float,
    mean_reversion_input: str,
) -> pd.DataFrame:
    """Evaluate one strategy under multiple transaction-cost assumptions."""

    def evaluator(cost_bps: float) -> dict[str, float]:
        assumptions = TradingAssumptions(
            initial_capital=initial_capital,
            periods_per_year=periods_per_year,
            risk_free_rate=risk_free_rate,
            transaction_cost_bps_per_unit_turnover=cost_bps,
            position_lag=1,
        )
        returns, turnover = portfolio_backtest(
            prices,
            assumptions=assumptions,
            strategy_name=strategy_name,
            momentum_lookback=momentum_lookback,
            momentum_neutral_band=momentum_neutral_band,
            mean_reversion_lookback=mean_reversion_lookback,
            entry_threshold=entry_threshold,
            exit_threshold=exit_threshold,
            mean_reversion_input=mean_reversion_input,
        )
        return summarize_cost_case(returns, turnover, assumptions=assumptions)

    return transaction_cost_sensitivity(cost_bps_values, evaluator)


def print_cost_table(strategy_name: str, results: pd.DataFrame) -> None:
    """Print the requested cost-sensitivity metrics."""

    print(f"\n{strategy_name.title()} Transaction-Cost Sensitivity")
    print("Cost (bps)    Net Annual Return    Sharpe Ratio    Max Drawdown    Total Turnover")
    print("-------------------------------------------------------------------------------")
    for _, row in results.iterrows():
        print(
            f"{row['transaction_cost_bps']:>10.0f}    "
            f"{row['net_annual_return']:>17.2%}    "
            f"{row['sharpe_ratio']:>12.3f}    "
            f"{row['max_drawdown']:>12.2%}    "
            f"{row['total_turnover']:>14.3f}"
        )


def save_sharpe_cost_chart(
    momentum_results: pd.DataFrame,
    mean_reversion_results: pd.DataFrame,
    *,
    output_dir: Path,
    chart_name: str,
) -> None:
    """Save Sharpe ratio versus transaction cost for both strategies."""

    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        momentum_results["transaction_cost_bps"],
        momentum_results["sharpe_ratio"],
        marker="o",
        label="Momentum",
    )
    ax.plot(
        mean_reversion_results["transaction_cost_bps"],
        mean_reversion_results["sharpe_ratio"],
        marker="o",
        label="Mean Reversion",
    )
    ax.set_title("Sharpe Ratio vs. Transaction Cost")
    ax.set_xlabel("Transaction Cost (bps per unit turnover)")
    ax.set_ylabel("Sharpe Ratio")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / chart_name, dpi=150)
    plt.close(fig)


def main() -> None:
    """Run transaction-cost sensitivity analysis for both strategies."""

    args = parse_args()
    prices = build_price_panel(
        args.symbols,
        start=args.start,
        end=args.end,
        use_cache=args.use_cache,
    )

    momentum_results = evaluate_strategy_cost_sensitivity(
        prices,
        strategy_name="momentum",
        cost_bps_values=args.cost_bps,
        periods_per_year=args.periods_per_year,
        risk_free_rate=args.risk_free_rate,
        initial_capital=args.initial_capital,
        momentum_lookback=args.momentum_lookback,
        momentum_neutral_band=args.momentum_neutral_band,
        mean_reversion_lookback=args.mean_reversion_lookback,
        entry_threshold=args.entry_threshold,
        exit_threshold=args.exit_threshold,
        mean_reversion_input=args.mean_reversion_input,
    )
    mean_reversion_results = evaluate_strategy_cost_sensitivity(
        prices,
        strategy_name="mean_reversion",
        cost_bps_values=args.cost_bps,
        periods_per_year=args.periods_per_year,
        risk_free_rate=args.risk_free_rate,
        initial_capital=args.initial_capital,
        momentum_lookback=args.momentum_lookback,
        momentum_neutral_band=args.momentum_neutral_band,
        mean_reversion_lookback=args.mean_reversion_lookback,
        entry_threshold=args.entry_threshold,
        exit_threshold=args.exit_threshold,
        mean_reversion_input=args.mean_reversion_input,
    )

    print_cost_table("momentum", momentum_results)
    print_cost_table("mean reversion", mean_reversion_results)

    save_sharpe_cost_chart(
        momentum_results,
        mean_reversion_results,
        output_dir=Path(args.output_dir),
        chart_name=args.sharpe_chart_name,
    )


if __name__ == "__main__":
    main()
