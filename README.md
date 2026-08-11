# quant-trading-research

Portfolio-quality quantitative trading research project aimed at quantitative trading and quantitative research internships.

## Status

Research package with:

- clean Python package layout under `src/quant_research`
- research guardrails documented in `AGENTS.md`
- historical-data layer
- transparent metrics layer
- simple vectorized backtester
- baseline time-series momentum strategy
- no machine learning, live trading, broker integration, or application framework

## Repository Layout

```text
quant-trading-research/
├── AGENTS.md
├── README.md
├── pyproject.toml
├── .gitignore
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
├── scripts/
├── src/
│   └── quant_research/
└── tests/
```

## Getting Started

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .[dev]
python -m pytest
```

## Development Workflow

1. Keep research code in `src/quant_research`, not in notebooks.
2. Use notebooks only for exploration and result communication.
3. Make one logical change per commit.
4. Run tests before each commit.
5. Push `main` only when the baseline is stable.

## Momentum Hypothesis

The baseline strategy in this repository is time-series momentum. The hypothesis is simple:
assets with positive trailing returns may continue trending over the next period, while
assets with negative trailing returns may continue underperforming. The implementation is
intentionally transparent: it computes a trailing return over a fixed lookback window,
maps that to `+1`, `0`, or `-1`, and then lags execution in the backtester so the signal
cannot trade on the same return that created it.

This is only a research hypothesis, not evidence of a durable edge. A backtest does not
prove future profitability because:

- the future may not resemble the historical sample,
- transaction costs, market impact, and financing assumptions may be understated,
- parameter choices can overfit even when they look reasonable,
- data quality issues and regime shifts can make apparent edges disappear,
- statistical significance is not the same as economic robustness.

The purpose of this project is to build and defend a careful research process, not to
claim that one historical strategy result is predictive.

## Momentum Vs. Mean Reversion

The repository now includes two simple strategy families:

- time-series momentum, which assumes some degree of trend persistence,
- mean reversion, which assumes sufficiently extreme moves may partially reverse.

These strategies are not rivals in a universal sense. They embed different views of how
prices behave:

- momentum is more naturally aligned with persistent directional moves, slower-moving
  information diffusion, and regime continuation,
- mean reversion is more naturally aligned with temporary dislocations, overshooting,
  liquidity shocks, and short-horizon normalization.

Comparing them on the same assets and period is useful for understanding regime
dependence and implementation tradeoffs, but it does not prove one is structurally better
than the other.

## Overfitting and Out-of-Sample Validation

This repository treats overfitting as a primary research risk. A strategy can look
excellent in-sample for reasons that do not survive new data:

- parameters may adapt too closely to one historical regime,
- repeated experimentation can turn noise into a convincing story,
- a small set of apparently strong backtests may hide instability across nearby choices,
- even sensible economic ideas can fail once transaction costs and regime shifts matter.

To reduce that risk, the validation workflow in this project follows a few explicit rules:

- never randomly shuffle financial time-series data,
- keep training periods strictly earlier than validation or test periods,
- label in-sample and out-of-sample results separately,
- evaluate reasonable parameter grids without automatically declaring the historical best
  parameter to be the “winner,”
- inspect stability across neighboring parameter values rather than trusting one isolated
  peak.

The momentum validation experiment applies those ideas directly by comparing multiple
lookback windows, reporting in-sample versus out-of-sample Sharpe, and plotting parameter
stability. The goal is not to prove a parameter is correct, but to judge whether observed
performance appears robust or fragile within the sampled history.

## Transaction Costs and Sensitivity

This repository includes explicit transaction-cost sensitivity analysis because execution
costs are often the difference between an interesting gross signal and an implementable
net strategy.

High-turnover strategies are more sensitive to execution costs because they repeatedly pay
 the spread, commissions, and other trading frictions each time they rebalance. A strategy
that changes position frequently can look attractive before costs and deteriorate quickly
once even modest basis-point assumptions are applied.

Ignoring transaction costs can make a backtest misleading because:

- gross returns may overstate what could actually be captured in practice,
- Sharpe ratios can look artificially strong when frequent small trades are treated as free,
- parameter choices that trade more often may appear best only because costs were omitted,
- performance comparisons across strategies become distorted when turnover is ignored.

This project deliberately starts with simple basis-points-per-unit-turnover assumptions and
does not yet include more sophisticated market-impact models. The goal at this stage is to
measure basic cost sensitivity transparently before adding more complex execution modeling.

## Rolling Diagnostics and Volatility Regimes

The repository also includes descriptive rolling diagnostics such as rolling volatility,
rolling Sharpe, rolling correlations, rolling cumulative returns, and drawdowns. These are
useful for understanding when a strategy appears to be working, when it is unstable, and
how its behavior changes across the sample.

Volatility-regime labels in this project are descriptive only. They are created by taking
historical realized volatility and bucketing it into low, medium, and high regimes using
quantiles. This is not a predictive regime model and should not be interpreted as a claim
that future volatility states can be known ahead of time.

## Near-Term Plan

1. Extend strategy coverage beyond the single baseline momentum rule.
2. Add richer portfolio construction and validation tooling.
3. Add notebook-based analysis built on the tested package code.
