"""Baseline tests for repository scaffolding."""

import importlib


MODULES = [
    "quant_research",
    "quant_research.backtest",
    "quant_research.data",
    "quant_research.metrics",
    "quant_research.plotting",
    "quant_research.portfolio",
    "quant_research.signals",
    "quant_research.validation",
]


def test_modules_import() -> None:
    """Ensure the package scaffold imports cleanly."""

    for module_name in MODULES:
        assert importlib.import_module(module_name) is not None
