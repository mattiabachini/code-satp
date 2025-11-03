"""
Shared file I/O utilities for count extraction experiments.
Follows the same conventions as classification-models.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import pandas as pd


TASK_ALIASES = {
    "death-counts": "death-counts",
    "injury-counts": "injury-counts",
    "count-extraction": "count-extraction",
}


def _detect_base_results_dir() -> Path:
    """Detect base results directory based on environment."""
    env_dir = os.getenv("SATP_RESULTS_DIR")
    if env_dir:
        return Path(env_dir).expanduser().resolve()

    colab_drive = Path("/content/drive/MyDrive/colab/satp-results")
    if colab_drive.exists():
        return colab_drive

    colab_local = Path("/content/satp-results")
    if colab_local.exists() or Path("/content").exists():
        return colab_local

    return Path.cwd() / "results"


def get_base_results_dir(create: bool = True) -> Path:
    """Return the base results directory, creating it if requested."""
    base = _detect_base_results_dir()
    if create:
        base.mkdir(parents=True, exist_ok=True)
    return base


def normalize_task_name(task_name: Optional[str]) -> Optional[str]:
    """Normalize task names to standard format."""
    if task_name is None:
        return None
    key = str(task_name).strip().lower()
    return TASK_ALIASES.get(key, key)


def get_task_results_dir(task_name: Optional[str], create: bool = True) -> Path:
    """Return the directory for a given task."""
    base = get_base_results_dir(create=create)
    if not task_name:
        return base
    normalized = normalize_task_name(task_name)
    path = base / normalized
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def save_dataframe_csv(df: pd.DataFrame, filename: str, task_name: Optional[str] = None, index: bool = False) -> Path:
    """Save a DataFrame to CSV under the base/task directory and return the path."""
    target_dir = get_task_results_dir(task_name, create=True)
    safe_filename = filename if filename.lower().endswith(".csv") else f"{filename}.csv"
    target_path = target_dir / safe_filename
    df.to_csv(target_path, index=index)
    return target_path


def load_dataframe_csv(filename: str, task_name: Optional[str] = None, **read_csv_kwargs) -> pd.DataFrame:
    """Load a CSV from the base/task directory."""
    target_dir = get_task_results_dir(task_name, create=False)
    safe_filename = filename if filename.lower().endswith(".csv") else f"{filename}.csv"
    csv_path = target_dir / safe_filename
    return pd.read_csv(csv_path, **read_csv_kwargs)


__all__ = [
    "get_base_results_dir",
    "get_task_results_dir",
    "normalize_task_name",
    "save_dataframe_csv",
    "load_dataframe_csv",
]

