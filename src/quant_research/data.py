"""Data access interfaces and shared data structures."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataPaths:
    """Repository paths for raw and processed datasets."""

    raw: Path
    processed: Path
