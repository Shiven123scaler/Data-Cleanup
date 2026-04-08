"""
Grader Easy — Exact Match

Compares the agent's cleaned DataFrame against the ground-truth
`clean_easy.csv`.  Returns 1.0 for an exact match, 0.0 otherwise.

Matching logic:
  1. Both DataFrames are sorted by ALL columns to ignore row ordering.
  2. Indexes are reset so positional differences don't matter.
  3. Column order is normalised.
  4. dtypes are aligned (values compared as strings when necessary).
"""

from __future__ import annotations

import pandas as pd
from pathlib import Path

_BASE_DIR = Path(__file__).resolve().parents[2]
CLEAN_DATA_PATH = _BASE_DIR / "data" / "clean_easy.csv"


def _load_clean() -> pd.DataFrame:
    """Load the ground-truth clean dataset."""
    try:
        return pd.read_csv(CLEAN_DATA_PATH, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(CLEAN_DATA_PATH, encoding="latin-1")


def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Sort rows & columns, reset index, cast to string for safe compare."""
    df = df.copy()
    cols = sorted(df.columns.tolist())
    df = df[cols]
    # Fill NaN with a sentinel so sort is deterministic
    df = df.fillna("__NAN__")
    # Coerce float columns that are really ints (e.g. 44.0 → 44)
    # This handles the pandas NaN-induced float promotion issue
    for col in df.columns:
        if df[col].dtype in ("float64", "float32"):
            try:
                as_int = df[col].astype(int)
                # Only convert if lossless (no fractional parts)
                if (df[col] == as_int.astype(float)).all():
                    df[col] = as_int
            except (ValueError, TypeError):
                pass
    # Cast everything to str for dtype-agnostic comparison
    df = df.astype(str)
    df = df.sort_values(by=cols).reset_index(drop=True)
    return df


def grade(agent_df: pd.DataFrame, clean_df: pd.DataFrame | None = None) -> float:
    """
    Grade the agent's output against the ground-truth dataset.

    Parameters
    ----------
    agent_df : pd.DataFrame
        The DataFrame produced by the agent after cleaning.
    clean_df : pd.DataFrame | None
        Optional ground-truth. If None, loaded from disk.

    Returns
    -------
    float
        1.0 if the DataFrames match exactly, 0.0 otherwise.
    """
    if clean_df is None:
        clean_df = _load_clean()

    # Quick shape check
    if agent_df.shape != clean_df.shape:
        return 0.0

    # Quick column check
    if set(agent_df.columns) != set(clean_df.columns):
        return 0.0

    # Normalise and compare
    try:
        norm_agent = _normalise(agent_df)
        norm_clean = _normalise(clean_df)
        return 1.0 if norm_agent.equals(norm_clean) else 0.0
    except Exception:
        return 0.0
