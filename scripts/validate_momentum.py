#!/usr/bin/env python3
"""Run out-of-sample validation diagnostics for the momentum strategy."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from quant_research.backtest import TradingAssumptions, run_vectorized_backtest
from quant_research.data import download_daily_ohlcv, simple_returns
from quant_research.metrics import sharpe_ratio
from quant_research.signals import time_series_momentum_signal
from quant_research.validation import (
    chronological_train_test_split,
    chronological_train_validation_test_split,
    evaluate_parameter_grid,
    stability_analysis,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", nargs="+", default=["SPY", "TLT", "GLD"])
    parser.add_argument("--start", default="2010-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument(
        "--lookbacks",
        nargs="+",
        type=int,
        default=[21, 42, 63, 126, 189, 252],
        help="Reasonable momentum lookbacks in trading days.",
    )
    parser.add_argument("--neutral-band", type=float, default=0.0)
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.0,
        help="Optional validation fraction. Use 0.0 to skip validation.",
    )
    parser.add_argument("--transaction-cost-bps", type=float, default=5.0)
    parser.add_argument("--periods-per-year", type=int, default=252)
    parser.add_argument("--risk-free-rate", type=float, default=0.0)
    parser.add_argument("--initial-capital", type=float, default=1.0)
    parser.add_argument("--use-cache", action="store_true")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument(
        "--stability-chart-name",
        default="momentum_parameter_stability.png",
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


def build_momentum_portfolio_returns(
    prices: pd.DataFrame,
    *,
    lookback: int,
    neutral_band: float,
    assumptions: TradingAssumptions,
) -> pd.Series:
    """Build equal-weight momentum portfolio returns for a single lookback."""

    asset_returns = simple_returns(prices).dropna(how="any")
    if asset_returns.empty:
        raise ValueError("Asset return panel is empty after return calculation.")

    strategy_net_by_asset: dict[str, pd.Series] = {}
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

    return pd.DataFrame(strategy_net_by_asset).mean(axis=1)


def evaluate_lookback_grid(
    prices: pd.DataFrame,
    *,
    lookbacks: list[int],
    neutral_band: float,
    assumptions: TradingAssumptions,
    train_fraction: float,
    validation_fraction: float,
) -> pd.DataFrame:
    """Evaluate multiple lookbacks without selecting a historical winner."""

    def evaluator(params: dict[str, int]) -> dict[str, float]:
        returns = build_momentum_portfolio_returns(
            prices,
            lookback=params["lookback"],
            neutral_band=neutral_band,
            assumptions=assumptions,
        )
        in_sample, validation, out_of_sample = split_returns(
            returns,
            train_fraction=train_fraction,
            validation_fraction=validation_fraction,
        )
        result = {
            "in_sample_sharpe": sharpe_ratio(
                in_sample,
                periods_per_year=assumptions.periods_per_year,
                risk_free_rate=assumptions.risk_free_rate,
            ),
            "out_of_sample_sharpe": sharpe_ratio(
                out_of_sample,
                periods_per_year=assumptions.periods_per_year,
                risk_free_rate=assumptions.risk_free_rate,
            ),
            "in_sample_count": float(len(in_sample)),
            "out_of_sample_count": float(len(out_of_sample)),
        }
        if validation is not None:
            result["validation_sharpe"] = sharpe_ratio(
                validation,
                periods_per_year=assumptions.periods_per_year,
                risk_free_rate=assumptions.risk_free_rate,
            )
            result["validation_count"] = float(len(validation))
        return result

    return evaluate_parameter_grid({"lookback": lookbacks}, evaluator)


def split_returns(
    returns: pd.Series,
    *,
    train_fraction: float,
    validation_fraction: float,
) -> tuple[pd.Series, pd.Series | None, pd.Series]:
    """Split return series into explicitly labeled sample segments."""

    if validation_fraction > 0.0:
        split = chronological_train_validation_test_split(
            returns,
            train_size=train_fraction,
            validation_size=validation_fraction,
        )
        return split.train, split.validation, split.test

    split = chronological_train_test_split(returns, train_size=train_fraction)
    return split.train, None, split.test


def print_results_table(results: pd.DataFrame, *, include_validation: bool) -> None:
    """Print in-sample and out-of-sample validation results."""

    if include_validation:
        print("Lookback    In-Sample Sharpe    Validation Sharpe    Out-of-Sample Sharpe")
        print("--------------------------------------------------------------------------")
        for _, row in results.iterrows():
            print(
                f"{int(row['lookback']):>8}    "
                f"{row['in_sample_sharpe']:>16.3f}    "
                f"{row['validation_sharpe']:>17.3f}    "
                f"{row['out_of_sample_sharpe']:>21.3f}"
            )
        return

    print("Lookback    In-Sample Sharpe    Out-of-Sample Sharpe")
    print("----------------------------------------------------")
    for _, row in results.iterrows():
        print(
            f"{int(row['lookback']):>8}    "
            f"{row['in_sample_sharpe']:>16.3f}    "
            f"{row['out_of_sample_sharpe']:>21.3f}"
        )


def save_stability_chart(
    results: pd.DataFrame,
    *,
    output_dir: Path,
    chart_name: str,
    include_validation: bool,
) -> None:
    """Save a parameter-stability chart for momentum lookbacks."""

    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(results["lookback"], results["in_sample_sharpe"], marker="o", label="In-Sample Sharpe")
    ax.plot(results["lookback"], results["out_of_sample_sharpe"], marker="o", label="Out-of-Sample Sharpe")
    if include_validation:
        ax.plot(results["lookback"], results["validation_sharpe"], marker="o", label="Validation Sharpe")
    ax.set_title("Momentum Parameter Stability")
    ax.set_xlabel("Lookback Window")
    ax.set_ylabel("Sharpe Ratio")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / chart_name, dpi=150)
    plt.close(fig)


def assess_robustness(results: pd.DataFrame) -> str:
    """Summarize whether the sampled parameter behavior looks robust or fragile."""

    stability = stability_analysis(
        results,
        parameter_column="lookback",
        metric_column="out_of_sample_sharpe",
    )
    valid_gaps = stability["stability_gap"].dropna()
    median_gap = float(valid_gaps.median()) if not valid_gaps.empty else np.nan
    positive_fraction = float((results["out_of_sample_sharpe"] > 0.0).mean())
    in_out_gap = float((results["in_sample_sharpe"] - results["out_of_sample_sharpe"]).abs().median())

    if positive_fraction >= 0.6 and (np.isnan(median_gap) or median_gap <= 0.25) and in_out_gap <= 0.5:
        return (
            "Within this sample, momentum performance looks relatively robust: "
            "out-of-sample Sharpe is positive for most neighboring lookbacks and "
            "does not collapse sharply outside a single parameter choice."
        )
    return (
        "Within this sample, momentum performance looks fragile: "
        "out-of-sample Sharpe is highly sensitive to neighboring lookbacks, "
        "or the in-sample to out-of-sample drop is large."
    )


def main() -> None:
    """Run the momentum overfitting and out-of-sample validation experiment."""

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
    results = evaluate_lookback_grid(
        prices,
        lookbacks=args.lookbacks,
        neutral_band=args.neutral_band,
        assumptions=assumptions,
        train_fraction=args.train_fraction,
        validation_fraction=args.validation_fraction,
    )

    include_validation = args.validation_fraction > 0.0
    print_results_table(results, include_validation=include_validation)
    print()
    print(assess_robustness(results))

    save_stability_chart(
        results,
        output_dir=Path(args.output_dir),
        chart_name=args.stability_chart_name,
        include_validation=include_validation,
    )


if __name__ == "__main__":
    main()
