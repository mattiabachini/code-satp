"""
Shared file I/O utilities to standardize saving/loading artifacts across notebooks.

Conventions:
- Base results directory is detected in this order:
  1) Environment variable SATP_RESULTS_DIR (if set)
  2) Google Drive path on Colab: /content/drive/MyDrive/colab/satp-results
  3) Colab local: /content/satp-results
  4) Local workspace: ./results (relative to current working directory)

- Task subdirectory names use hyphenated forms (e.g., "action-type", "target-type").
- Helper functions create directories if they do not exist.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import pandas as pd


HYphenated_TASK_ALIASES = {
    "actiontype": "action-type",
    "action-type": "action-type",
    "targettype": "target-type",
    "target-type": "target-type",
    "perpetrator": "perpetrator",
}


def _detect_base_results_dir() -> Path:
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
    if task_name is None:
        return None
    key = str(task_name).strip().lower()
    return HYphenated_TASK_ALIASES.get(key, key)


def get_task_results_dir(task_name: Optional[str], create: bool = True) -> Path:
    """Return the directory for a given task (e.g., "action-type")."""
    base = get_base_results_dir(create=create)
    if not task_name:
        return base
    normalized = normalize_task_name(task_name)
    path = base / normalized
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _ensure_csv_suffix(filename: str) -> str:
    return filename if filename.lower().endswith(".csv") else f"{filename}.csv"


def save_dataframe_csv(df: pd.DataFrame, filename: str, task_name: Optional[str] = None, index: bool = False) -> Path:
    """Save a DataFrame to CSV under the base/task directory and return the path."""
    target_dir = get_task_results_dir(task_name, create=True)
    safe_filename = _ensure_csv_suffix(filename)
    target_path = target_dir / safe_filename
    df.to_csv(target_path, index=index)
    return target_path


def load_dataframe_csv(filename: str, task_name: Optional[str] = None, **read_csv_kwargs) -> pd.DataFrame:
    """Load a CSV from the base/task directory."""
    target_dir = get_task_results_dir(task_name, create=False)
    safe_filename = _ensure_csv_suffix(filename)
    csv_path = target_dir / safe_filename
    return pd.read_csv(csv_path, **read_csv_kwargs)


def build_filename(prefix: str, suffix: Optional[str] = None, add_timestamp: bool = False, extension: str = ".csv") -> str:
    """Construct a standardized filename like: prefix[_suffix][_YYYYmmdd-HHMMSS].ext"""
    parts = [prefix]
    if suffix:
        parts.append(suffix)
    if add_timestamp:
        from datetime import datetime

        parts.append(datetime.now().strftime("%Y%m%d-%H%M%S"))
    stem = "_".join(filter(None, parts))
    if not extension.startswith("."):
        extension = f".{extension}"
    return f"{stem}{extension}"


__all__ = [
    "get_base_results_dir",
    "get_task_results_dir",
    "normalize_task_name",
    "save_dataframe_csv",
    "load_dataframe_csv",
    "build_filename",
    "ensure_directory",
]


