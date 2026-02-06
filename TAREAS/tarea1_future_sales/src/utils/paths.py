"""
Utilities for repository paths.

This module centralizes path logic such as resolving the repository root.
"""

from __future__ import annotations

from pathlib import Path


def get_repo_root(current_file: str) -> Path:
    """
    Return the repository root path (one level above /src).

    Parameters
    ----------
    current_file:
        The __file__ value from a module located in src/.

    Returns
    -------
    Path
        The repository root directory.
    """
    return Path(current_file).resolve().parents[1]
