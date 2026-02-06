"""
Validation utilities for filesystem and inputs.

These helpers reduce duplicated checks across scripts.
"""

# Importing libraries and packages
from __future__ import annotations

from pathlib import Path


def ensure_dir(path: Path) -> None:
    """Create a directory (and parents) if it does not exist."""
    path.mkdir(parents=True, exist_ok=True)


def require_file(path: Path, hint: str) -> None:
    """
    Require a file to exist; otherwise raise FileNotFoundError with a hint.

    Parameters
    ----------
    path:
        File path that must exist.
    hint:
        Helpful message for the user (e.g., how to generate the file).
    """
    if not path.exists():
        raise FileNotFoundError(f"No existe {path}. {hint}")


def require_non_empty(condition: bool, message: str) -> None:
    """Raise ValueError if a required condition is not met."""
    if not condition:
        raise ValueError(message)
