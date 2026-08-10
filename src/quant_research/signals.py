"""Signal-generation interfaces for future strategy implementations."""

from typing import Protocol


class SignalGenerator(Protocol):
    """Protocol for transparent signal-generation components."""

    def name(self) -> str:
        """Return a human-readable strategy name."""
