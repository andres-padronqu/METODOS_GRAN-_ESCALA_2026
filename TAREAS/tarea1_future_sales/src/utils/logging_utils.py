"""
Logging utilities.

Creates a per-run logger that writes to a timestamped file and stdout.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from src.utils.validation import ensure_dir


def _to_relpath(project_root: Path, path: Path) -> str:
    """Return a repo-relative path for logging (avoid absolute system paths)."""
    try:
        return str(path.relative_to(project_root))
    except ValueError:
        return str(path)


def setup_logger(project_root: Path, script_name: str) -> logging.Logger:
    """
    Configure and return a logger (file + console) for a script run.

    Parameters
    ----------
    project_root:
        Repo root path (used to place logs in artifacts/logs and log relative paths).
    script_name:
        Name of the script (e.g., "prep", "train", "inference").

    Returns
    -------
    logging.Logger
        Configured logger writing to stdout and artifacts/logs/<script>_TIMESTAMP.log
    """
    logs_dir = project_root / "artifacts" / "logs"
    ensure_dir(logs_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = logs_dir / f"{script_name}_{timestamp}.log"

    logger = logging.getLogger(script_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # Avoid duplicate handlers if re-run in same Python process.
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(logging.INFO)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    logger.info("Logger initialized. log_path=%s", _to_relpath(project_root, log_path))
    return logger

