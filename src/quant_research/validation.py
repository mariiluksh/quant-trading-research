"""Validation helpers for research and backtest inputs."""


def require_non_empty(name: str, size: int) -> None:
    """Raise a clear error when an expected collection is empty."""

    if size <= 0:
        msg = f"{name} must not be empty."
        raise ValueError(msg)
