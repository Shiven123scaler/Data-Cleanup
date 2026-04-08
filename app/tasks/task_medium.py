"""
Task Medium — Date Normalisation + Deduplication

Objective:
  1. Normalise all date columns to YYYY-MM-DD format.
  2. Remove duplicate rows.

The agent receives `raw_medium.csv` (mixed date formats + ~15% dupes)
and must produce a DataFrame matching `clean_medium.csv`.

Grading: Partial scoring — 0.5 for dedup + 0.5 for date normalisation.
"""

from __future__ import annotations

import pandas as pd
from pathlib import Path

# ── paths ────────────────────────────────────────────────────────────────
_BASE_DIR = Path(__file__).resolve().parents[2]          # project root
RAW_DATA_PATH = _BASE_DIR / "data" / "raw_medium.csv"
CLEAN_DATA_PATH = _BASE_DIR / "data" / "clean_medium.csv"

# ── task metadata ────────────────────────────────────────────────────────
TASK_ID = "medium"
MAX_STEPS = 15
DESCRIPTION = (
    "Clean the dataset by normalising all date columns to YYYY-MM-DD "
    "format and removing duplicate rows."
)

INSTRUCTIONS = """
You are given a CSV dataset with three columns: id, name, join_date.
The `join_date` column contains dates in mixed formats:
  - DD-MM-YYYY   (e.g., 09-26-2022)
  - YYYY.MM.DD   (e.g., 2018.05.15)
  - DD/MM/YYYY   (e.g., 13/09/2022)
  - ISO 8601     (e.g., 2011-12-26)
There are also duplicate rows scattered throughout the dataset.

Your goal:
  1. Normalise ALL dates in `join_date` to YYYY-MM-DD format.
  2. Remove all duplicate rows.
  3. The resulting DataFrame should match the ground-truth clean dataset.

Available actions:
  - normalize_dates (column="join_date")
  - drop_duplicates (column=None)
  - no_op

Hint: Use `normalize_dates` on `join_date`, then `drop_duplicates`.
""".strip()


def load_raw_data() -> pd.DataFrame:
    """Load the raw (messy) medium dataset."""
    try:
        return pd.read_csv(RAW_DATA_PATH, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(RAW_DATA_PATH, encoding="latin-1")


def load_clean_data() -> pd.DataFrame:
    """Load the ground-truth clean dataset for grading."""
    try:
        return pd.read_csv(CLEAN_DATA_PATH, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(CLEAN_DATA_PATH, encoding="latin-1")


def get_task_config() -> dict:
    """Return the configuration dict consumed by the environment."""
    return {
        "task_id": TASK_ID,
        "max_steps": MAX_STEPS,
        "raw_data_path": str(RAW_DATA_PATH),
        "clean_data_path": str(CLEAN_DATA_PATH),
        "description": DESCRIPTION,
        "instructions": INSTRUCTIONS,
    }
