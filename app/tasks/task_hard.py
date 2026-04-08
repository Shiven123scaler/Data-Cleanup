"""
Task Hard — Full Dataset Cleaning

Objective:
  1. Validate email addresses (keep only RFC-valid emails).
  2. Normalise phone numbers to a consistent format (XXX-XXX-XXXX).
  3. Capitalise status values consistently (e.g. "ACTIVE" → "Active").

The agent receives `raw_hard.csv` and must produce a DataFrame
matching `clean_hard.csv`.

Grading: Multi-factor — email ratio + phone normalisation + status capitalisation.
"""

from __future__ import annotations

import pandas as pd
from pathlib import Path

# ── paths ────────────────────────────────────────────────────────────────
_BASE_DIR = Path(__file__).resolve().parents[2]          # project root
RAW_DATA_PATH = _BASE_DIR / "data" / "raw_hard.csv"
CLEAN_DATA_PATH = _BASE_DIR / "data" / "clean_hard.csv"

# ── task metadata ────────────────────────────────────────────────────────
TASK_ID = "hard"
MAX_STEPS = 30
DESCRIPTION = (
    "Perform end-to-end cleaning: validate emails, normalise phone numbers "
    "to XXX-XXX-XXXX format, and capitalise status values consistently "
    "(Active, Inactive, Pending)."
)

INSTRUCTIONS = """
You are given a CSV dataset with five columns:
  user_id, full_name, email, phone, status

Issues present:
  • Some emails may be invalid (malformed).
  • Phone numbers are in mixed formats:
      (XXX) XXX-XXXX, XXX.XXX.XXXX, XXX-XXX-XXXX, etc.
  • Status values have inconsistent casing:
      "ACTIVE", "active", "Pending", "pending", "INACTIVE", etc.

Your goal:
  1. Validate emails — keep only rows with valid email addresses.
  2. Normalise ALL phone numbers to XXX-XXX-XXXX format.
  3. Capitalise statuses to title case: "Active", "Inactive", "Pending".

Available actions:
  - validate_emails  (column="email")
  - lowercase_column (column="status") — then further normalisation
  - drop_nulls       (column=None or specific column)
  - no_op

Hint: Use `validate_emails` on the email column, then work on
phone normalisation and status capitalisation through the environment
step actions.
""".strip()


def load_raw_data() -> pd.DataFrame:
    """Load the raw (messy) hard dataset."""
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
