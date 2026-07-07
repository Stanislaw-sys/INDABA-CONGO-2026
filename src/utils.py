"""Text normalization shared across the pipeline.

The datasets are French with accents, inconsistent casing, and the literal
sentinel "Non déclaré" for missing categorical values. Everything that feeds
the vectorizers goes through :func:`norm` so that "Génie Logistique" and
"genie logistique" collapse to the same tokens.
"""
from __future__ import annotations

import re
import unicodedata

import pandas as pd

# Values that mean "missing" in the categorical columns.
MISSING_SENTINELS = {"non declare", "non déclaré", "non declaré", "nan", "", "n/a", "na"}


def strip_accents(text: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn"
    )


def norm(value) -> str:
    """Lowercase, strip accents, drop punctuation, and blank out missing sentinels."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = strip_accents(str(value).lower()).strip()
    if text in MISSING_SENTINELS:
        return ""
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def is_missing(value) -> bool:
    """True when a categorical value should be treated as absent."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return True
    return strip_accents(str(value).lower()).strip() in MISSING_SENTINELS


def tokens(value) -> set[str]:
    """Token set of a normalized value, excluding very short tokens."""
    return {t for t in norm(value).split() if len(t) > 2}
