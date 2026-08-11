#!/usr/bin/env python3
"""Run a transparent time-series momentum backtest on liquid ETFs."""

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
    drawdown_series,
    maximum_drawdown,
    running_equity_curve,
    sharpe_ratio,
)
from quant_research.signals import time_series_momentum_signal


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["SPY", "TLT", "GLD"],
        help="Liquid assets to include in the equal-weight portfolio.",
    )
    parser.add_argument(
        "--start",
        default="2010-01-01",
        help="Inclusive start date for historical data.",
    )
    parser.add_argument(
        "--end",
        default=None,
        help="Exclusive end date for historical data.",
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=126,
        help="Momentum lookback window in trading days.",
    )
    parser.add_argument(
        "--neutral-band",
        type=float,
        default=0.0,
        help="Neutral band around zero trailing return before taking a position.",
    )
    parser.add_argument(
        "--transaction-cost-bps",
        type=float,
        default=5.0,
        help="Transaction cost in basis points per unit turnover.",
    )
    parser.add_argument(
        "--periods-per-year",
        type=int,
        default=252,
        help="Annualization factor for daily data.",
    )
    parser.add_argument(
        "--risk-free-rate",
        type=float,
        default=0.0,
        help="Annual risk-free rate used in Sharpe calculations.",
    )
    parser.add_argument(
        "--initial-capital",
        type=float,
        default=1.0,
        help="Starting portfolio equity.",
    )
    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="Load and save processed data under data/processed.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed",
        help="Directory for saved charts.",
    )
    parser.add_argument(
        "--equity-chart-name",
        default="momentum_equity_curve.png",
        help="Filename for the equity-curve chart.",
    )
    parser.add_argument(
        "--drawdown-chart-name",
        default="momentum_drawdown.png",
        help="Filename for the drawdown chart.",
    )
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


def build_portfolio_returns(
    prices: pd.DataFrame,
    *,
    lookback: int,
    neutral_band: float,
    assumptions: TradingAssumptions,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Backtest per asset and combine into equal-weight portfolio returns."""

    asset_returns = simple_returns(prices).dropna(how="any")
    if asset_returns.empty:
        raise ValueError("Asset return panel is empty after return calculation.")

    strategy_net_by_asset: dict[str, pd.Series] = {}
    turnover_by_asset: dict[str, pd.Series] = {}

    for symbol in prices.columns:
        signal = time_series_momentum_signal(
            prices[symbol],
            lookback=lookback,
            neutral_band=neutral_band,
        )
        target_positions = signal.target_positions.reindex(asset_returns.index).fillna(0.0)
        result = run_vectorized_backtest(
            asset_returns[symbol],
            target_positions,
            assumptions=assumptions,
        )
        strategy_net_by_asset[symbol] = result.net_returns
        turnover_by_asset[symbol] = result.turnover

    strategy_net = pd.DataFrame(strategy_net_by_asset).mean(axis=1)
    buy_and_hold = asset_returns.mean(axis=1)
    turnover = pd.DataFrame(turnover_by_asset).mean(axis=1)
    return strategy_net, buy_and_hold, turnover


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


def print_summary(
    *,
    strategy_summary: dict[str, float],
    buy_hold_summary: dict[str, float],
) -> None:
    """Print a simple comparison table."""

    print("Metric                 Strategy        Buy-and-Hold")
    print("---------------------------------------------------")
    print(f"Annual return          {strategy_summary['annual_return']:>10.2%}   {buy_hold_summary['annual_return']:>12.2%}")
    print(f"Annual volatility      {strategy_summary['annual_volatility']:>10.2%}   {buy_hold_summary['annual_volatility']:>12.2%}")
    print(f"Sharpe                 {strategy_summary['sharpe']:>10.3f}   {buy_hold_summary['sharpe']:>12.3f}")
    print(f"Max drawdown           {strategy_summary['max_drawdown']:>10.2%}   {buy_hold_summary['max_drawdown']:>12.2%}")
    print(f"Average turnover       {strategy_summary['turnover']:>10.3f}   {buy_hold_summary['turnover']:>12.3f}")


def save_charts(
    *,
    strategy_returns: pd.Series,
    buy_hold_returns: pd.Series,
    initial_capital: float,
    output_dir: Path,
    equity_chart_name: str,
    drawdown_chart_name: str,
) -> None:
    """Save equity and drawdown comparison charts."""

    output_dir.mkdir(parents=True, exist_ok=True)

    strategy_equity = running_equity_curve(strategy_returns, starting_value=initial_capital)
    buy_hold_equity = running_equity_curve(buy_hold_returns, starting_value=initial_capital)
    strategy_drawdown = drawdown_series(strategy_returns, starting_value=initial_capital)
    buy_hold_drawdown = drawdown_series(buy_hold_returns, starting_value=initial_capital)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(strategy_equity.index, strategy_equity, label="Momentum Strategy")
    ax.plot(buy_hold_equity.index, buy_hold_equity, label="Buy and Hold")
    ax.set_title("Equity Curve Comparison")
    ax.set_ylabel("Portfolio Value")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / equity_chart_name, dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(strategy_drawdown.index, strategy_drawdown, label="Momentum Strategy")
    ax.plot(buy_hold_drawdown.index, buy_hold_drawdown, label="Buy and Hold")
    ax.set_title("Drawdown Comparison")
    ax.set_ylabel("Drawdown")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / drawdown_chart_name, dpi=150)
    plt.close(fig)


def main() -> None:
    """Run the end-to-end momentum research workflow."""

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
    strategy_returns, buy_hold_returns, strategy_turnover = build_portfolio_returns(
        prices,
        lookback=args.lookback,
        neutral_band=args.neutral_band,
        assumptions=assumptions,
    )

    strategy_summary = summarize_returns(
        strategy_returns,
        periods_per_year=args.periods_per_year,
        risk_free_rate=args.risk_free_rate,
        average_turnover=float(strategy_turnover.mean()),
    )
    buy_hold_summary = summarize_returns(
        buy_hold_returns,
        periods_per_year=args.periods_per_year,
        risk_free_rate=args.risk_free_rate,
        average_turnover=0.0,
    )

    print_summary(
        strategy_summary=strategy_summary,
        buy_hold_summary=buy_hold_summary,
    )

    save_charts(
        strategy_returns=strategy_returns,
        buy_hold_returns=buy_hold_returns,
        initial_capital=args.initial_capital,
        output_dir=Path(args.output_dir),
        equity_chart_name=args.equity_chart_name,
        drawdown_chart_name=args.drawdown_chart_name,
    )


if __name__ == "__main__":
    main()
