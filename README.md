# quant-trading-research

Portfolio-quality quantitative trading research project aimed at quantitative trading and quantitative research internships.

## Status

Phase 0 repository scaffold only.

- Clean Python package layout under `src/quant_research`
- Research guardrails documented in `AGENTS.md`
- No strategies yet
- No machine learning, live trading, broker integration, or application framework

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

## Near-Term Plan

1. Implement data ingestion and normalization.
2. Add explicit market and transaction-cost assumptions.
3. Add first transparent baseline strategy and backtest tests.
