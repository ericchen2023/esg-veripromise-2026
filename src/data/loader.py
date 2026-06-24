"""Data loading utilities for the VeriPromise ESG 2026 dataset.

Loads either the official CSV or JSON file, validates schema, and
returns a list of dict samples plus a pandas DataFrame.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

REQUIRED_FIELDS = (
    "data",
    "promise_status",
    "verification_timeline",
    "evidence_status",
    "evidence_quality",
)

OPTIONAL_FIELDS = (
    "id",
    "esg_type",
    "promise_string",
    "evidence_string",
    "company",
    "ticker",
    "page_number",
    "pdf_url",
    "company_source",
)

LABEL_DOMAINS: dict[str, list[str]] = {
    "promise_status": ["Yes", "No"],
    "verification_timeline": [
        "already",
        "within_2_years",
        "between_2_and_5_years",
        "longer_than_5_years",
        "N/A",
    ],
    "evidence_status": ["Yes", "No", "N/A"],
    "evidence_quality": ["Clear", "Not Clear", "Misleading", "N/A"],
}


def load_dataset(path: str | Path) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    """Load dataset from CSV or JSON.

    Returns:
        records: list of per-sample dicts.
        df: pandas DataFrame view (no transformation).
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset not found: {p}")

    suffix = p.suffix.lower()
    if suffix == ".json":
        with p.open("r", encoding="utf-8") as f:
            records = json.load(f)
        df = pd.DataFrame(records)
    elif suffix == ".csv":
        df = pd.read_csv(p, encoding="utf-8")
        records = df.to_dict(orient="records")
    else:
        raise ValueError(f"Unsupported extension: {suffix}")

    df = _normalize_labels(df)
    records = df.to_dict(orient="records")
    _validate_schema(df)
    return records, df


_LABEL_ALIASES: dict[str, dict[str, str]] = {
    # The official validation set (released 2026-06-03) uses 'more_than_5_years'
    # for verification_timeline; the training set uses 'longer_than_5_years'.
    # Normalise both to the canonical training-set label so that downstream
    # code (LABEL2ID, model training, hillclimb evaluation) stays unchanged.
    # When generating final submissions, map 'longer_than_5_years' back to
    # 'more_than_5_years' if the test set / scoring system expects that form.
    "verification_timeline": {
        "more_than_5_years": "longer_than_5_years",
    },
}


def _normalize_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize NaN/None/empty in label fields to the string 'N/A'.

    Also applies label aliases (e.g. 'more_than_5_years' → 'longer_than_5_years')
    to harmonise data from multiple official releases.

    The official CSV/JSON encodes "not applicable" as NaN/null rather than
    the string "N/A"; downstream code expects a categorical string.
    Also normalize promise_string / evidence_string NaN -> "".
    """
    df = df.copy()
    for field in LABEL_DOMAINS:
        if field in df.columns:
            s = df[field].astype("string")
            s = s.where(~s.isna(), other="N/A")
            s = s.replace({"": "N/A", "nan": "N/A", "None": "N/A"})
            aliases = _LABEL_ALIASES.get(field, {})
            if aliases:
                s = s.replace(aliases)
            df[field] = s.astype(object)
    for field in ("promise_string", "evidence_string"):
        if field in df.columns:
            df[field] = df[field].astype("string").fillna("").astype(object)
    return df


def _validate_schema(df: pd.DataFrame) -> None:
    missing = [f for f in REQUIRED_FIELDS if f not in df.columns]
    if missing:
        raise ValueError(f"Missing required fields: {missing}")

    for field, domain in LABEL_DOMAINS.items():
        bad = set(df[field].dropna().unique()) - set(domain)
        if bad:
            raise ValueError(f"Field '{field}' contains out-of-domain values: {bad}")
