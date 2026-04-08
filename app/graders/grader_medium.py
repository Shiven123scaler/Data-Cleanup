"""
Grader Medium — Partial Scoring (Deduplication + Date Normalisation)

Scoring breakdown:
  • Deduplication factor   (weight 0.5):  1.0 if zero duplicates remain,
    else  1 - (remaining_dupes / original_dupes).
  • Date normalisation factor (weight 0.5):  fraction of `join_date`
    values that match YYYY-MM-DD format.

Final score = 0.5 * dedup_factor + 0.5 * date_factor   → [0.0, 1.0]
"""

from __future__ import annotations

import re
import pandas as pd
from pathlib import Path

_BASE_DIR = Path(__file__).resolve().parents[2]
CLEAN_DATA_PATH = _BASE_DIR / "data" / "clean_medium.csv"
RAW_DATA_PATH = _BASE_DIR / "data" / "raw_medium.csv"

# ISO date regex: exactly YYYY-MM-DD
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _load_clean() -> pd.DataFrame:
    try:
        return pd.read_csv(CLEAN_DATA_PATH, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(CLEAN_DATA_PATH, encoding="latin-1")


def _load_raw() -> pd.DataFrame:
    try:
        return pd.read_csv(RAW_DATA_PATH, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(RAW_DATA_PATH, encoding="latin-1")


def _dedup_score(agent_df: pd.DataFrame, raw_df: pd.DataFrame) -> float:
    """
    Score deduplication quality.

    Returns 1.0 when all duplicates have been removed,
    proportionally less if some remain.
    """
    original_dupes = int(raw_df.duplicated().sum())
    if original_dupes == 0:
        # Nothing to deduplicate → full marks
        return 1.0

    remaining_dupes = int(agent_df.duplicated().sum())
    return max(0.0, 1.0 - (remaining_dupes / original_dupes))


def _date_score(agent_df: pd.DataFrame, date_col: str = "join_date") -> float:
    """
    Score date normalisation quality.

    Returns the fraction of non-null values in `date_col` that
    match YYYY-MM-DD format.
    """
    if date_col not in agent_df.columns:
        return 0.0

    values = agent_df[date_col].dropna().astype(str)
    if len(values) == 0:
        return 0.0

    matches = values.apply(lambda v: bool(_DATE_RE.match(v.strip())))
    return float(matches.mean())


def grade(
    agent_df: pd.DataFrame,
    clean_df: pd.DataFrame | None = None,
    raw_df: pd.DataFrame | None = None,
) -> float:
    """
    Grade the agent's output for the medium task.

    Parameters
    ----------
    agent_df : pd.DataFrame
        The DataFrame produced by the agent after cleaning.
    clean_df : pd.DataFrame | None
        Ground-truth (unused directly, kept for API consistency).
    raw_df : pd.DataFrame | None
        Original raw data (used to compute duplicate baseline).

    Returns
    -------
    float
        Combined score in [0.0, 1.0].
    """
    if raw_df is None:
        raw_df = _load_raw()

    dedup = _dedup_score(agent_df, raw_df)
    date = _date_score(agent_df)

    score = 0.5 * dedup + 0.5 * date
    return round(min(max(score, 0.0), 1.0), 4)
