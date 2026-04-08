"""
Task Easy — Null Row Removal

Objective: Remove all rows containing missing / NaN values.
The agent receives `raw_easy.csv` (which has ~10-20% null-injected rows)
and must produce a DataFrame matching `clean_easy.csv`.

Grading: Exact match → 1.0, otherwise 0.0.
"""

from __future__ import annotations

import os
import pandas as pd
from pathlib import Path

# ── paths ────────────────────────────────────────────────────────────────
_BASE_DIR = Path(__file__).resolve().parents[2]          # project root
RAW_DATA_PATH = _BASE_DIR / "data" / "raw_easy.csv"
CLEAN_DATA_PATH = _BASE_DIR / "data" / "clean_easy.csv"

# ── task metadata ────────────────────────────────────────────────────────
TASK_ID = "easy"
MAX_STEPS = 5
DESCRIPTION = (
    "Clean the dataset by removing every row that contains at least one "
    "missing or NaN value.  The target is an exact match with the ground-truth "
    "cleaned dataset."
)

INSTRUCTIONS = """
You are given a CSV dataset with three columns: name, email, age.
Some rows have missing values (NaN / empty cells).

Your goal:
  1. Identify rows that contain ANY null or missing value.
  2. Remove those rows entirely.
  3. The resulting DataFrame should exactly match the ground-truth clean dataset.

Available actions:
  - drop_nulls (column=None → drop rows with ANY null; column="X" → drop rows where column X is null)
  - no_op (do nothing)

Hint: A single `drop_nulls` with column=None should handle this task.
""".strip()


def load_raw_data() -> pd.DataFrame:
    """Load the raw (messy) easy dataset."""
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
