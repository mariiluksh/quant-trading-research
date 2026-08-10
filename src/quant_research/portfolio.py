"""Portfolio construction interfaces and shared structures."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Position:
    """Single-asset portfolio position."""

    symbol: str
    weight: float
