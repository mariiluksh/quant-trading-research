"""Plotting interfaces for research diagnostics and reporting."""

from dataclasses import dataclass


@dataclass(frozen=True)
class PlotStyle:
    """Lightweight plotting configuration."""

    title: str
    figsize: tuple[float, float] = (10.0, 6.0)
