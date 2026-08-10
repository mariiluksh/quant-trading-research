# AGENTS.md

This repository is an educational quantitative-research platform.

Every implementation in this repository must:

1. avoid look-ahead bias,
2. avoid survivorship bias and data leakage where relevant,
3. make trading assumptions explicit,
4. include tests,
5. favour simple transparent implementations over unnecessary abstraction,
6. keep strategy logic separate from backtesting logic,
7. explain any financial assumptions in comments or documentation.

Additional expectations:

- Keep package code under `src/quant_research`.
- Use type hints throughout.
- Prefer concise docstrings over verbose commentary.
- Do not add machine learning, live trading, broker integration, or UI frameworks at this stage.
- Treat notebooks as exploratory artifacts, not as the primary implementation surface.
