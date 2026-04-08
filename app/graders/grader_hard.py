"""
Grader Hard — Multi-Factor Scoring

Scoring breakdown:
  • Email factor   (weight 0.35):  Ratio of valid email addresses
    present in the agent's output compared to ground-truth.
  • Phone factor   (weight 0.35):  Fraction of phone numbers that
    match the normalised XXX-XXX-XXXX format.
  • Status factor  (weight 0.30):  Fraction of status values that
    are correctly title-cased ("Active", "Inactive", "Pending").

Final score = weighted average  → [0.0, 1.0]
"""

from __future__ import annotations

import re
import pandas as pd
from pathlib import Path

_BASE_DIR = Path(__file__).resolve().parents[2]
CLEAN_DATA_PATH = _BASE_DIR / "data" / "clean_hard.csv"
RAW_DATA_PATH = _BASE_DIR / "data" / "raw_hard.csv"

# ── patterns ─────────────────────────────────────────────────────────────
_EMAIL_RE = re.compile(r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$")
_PHONE_RE = re.compile(r"^\d{3}-\d{3}-\d{4}$")
_VALID_STATUSES = {"Active", "Inactive", "Pending"}


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


def _email_score(agent_df: pd.DataFrame, clean_df: pd.DataFrame) -> float:
    """
    Ratio of valid emails in the agent output vs ground truth.

    Checks two things:
      1. All emails in agent_df are RFC-valid.
      2. The count of valid emails approaches the ground-truth count.
    """
    email_col = "email"
    if email_col not in agent_df.columns:
        return 0.0

    agent_emails = agent_df[email_col].dropna().astype(str)
    clean_emails = clean_df[email_col].dropna().astype(str)

    if len(clean_emails) == 0:
        return 1.0

    # Fraction of agent emails that are valid
    valid_count = agent_emails.apply(
        lambda e: bool(_EMAIL_RE.match(e.strip()))
    ).sum()

    # Score: ratio of valid emails vs expected count
    return min(1.0, float(valid_count) / len(clean_emails))


def _phone_score(agent_df: pd.DataFrame) -> float:
    """Fraction of phone numbers matching XXX-XXX-XXXX format."""
    phone_col = "phone"
    if phone_col not in agent_df.columns:
        return 0.0

    phones = agent_df[phone_col].dropna().astype(str)
    if len(phones) == 0:
        return 0.0

    matches = phones.apply(lambda p: bool(_PHONE_RE.match(p.strip())))
    return float(matches.mean())


def _status_score(agent_df: pd.DataFrame) -> float:
    """Fraction of status values that are correctly title-cased."""
    status_col = "status"
    if status_col not in agent_df.columns:
        return 0.0

    statuses = agent_df[status_col].dropna().astype(str)
    if len(statuses) == 0:
        return 0.0

    correct = statuses.apply(lambda s: s.strip() in _VALID_STATUSES)
    return float(correct.mean())


def grade(
    agent_df: pd.DataFrame,
    clean_df: pd.DataFrame | None = None,
    raw_df: pd.DataFrame | None = None,
) -> float:
    """
    Grade the agent's output for the hard task.

    Parameters
    ----------
    agent_df : pd.DataFrame
        The DataFrame produced by the agent after cleaning.
    clean_df : pd.DataFrame | None
        Ground-truth clean dataset.
    raw_df : pd.DataFrame | None
        Original raw data (unused here, kept for API consistency).

    Returns
    -------
    float
        Weighted multi-factor score in [0.0, 1.0].
    """
    if clean_df is None:
        clean_df = _load_clean()

    email = _email_score(agent_df, clean_df)
    phone = _phone_score(agent_df)
    status = _status_score(agent_df)

    score = 0.35 * email + 0.35 * phone + 0.30 * status
    return round(min(max(score, 0.0), 1.0), 4)
