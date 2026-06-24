"""Schema, validators, and ID partitioning for Aug-Plus (AP) augmentation data.

Aug-Plus is the suite of officially-authorised data augmentation pipelines
(hand-crafted seeds + LLM-synthesised rows) added on top of Phase 36 in
response to the organiser's 2026-05-17 ruling permitting any data-augmentation
method that does not touch the test set or final predictions.

This module is the **single source of truth** for:
  * Column layout of every Aug-Plus CSV (compatible with `data/processed/u10/`).
  * Label-domain checks and the four-task hierarchical constraint.
  * ID-space partitioning (no collisions with official train, U10 pseudo, or
    future modules).

Used by:
  * scripts/ap_llm_synth.py    -- generation
  * scripts/ap_quality_gate.py -- teacher-confidence filtering + dedup
  * tests/test_aug_plus.py     -- unit tests
  * src/train_pseudo_kfold.py  -- (indirectly via shared CSV column layout)
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable

from src.data.loader import LABEL_DOMAINS

# ---------------------------------------------------------------------------
# CSV columns -- must match data/processed/u10/pseudo_labels*.csv exactly so
# train_pseudo_kfold.py consumes both pipelines through the same loader.
# ---------------------------------------------------------------------------
PSEUDO_CSV_COLUMNS: tuple[str, ...] = (
    "id",
    "data",
    "esg_type",
    "promise_status",
    "promise_string",
    "verification_timeline",
    "evidence_status",
    "evidence_string",
    "evidence_quality",
    "company",
    "ticker",
    "page_number",
    "pdf_url",
    "company_source",
    "confidence_min",
    "conf_T1",
    "conf_T2",
    "conf_T3",
    "conf_T4",
)

# ---------------------------------------------------------------------------
# ID-space partitioning -- chosen to never collide with any existing source.
# ---------------------------------------------------------------------------
ID_RANGES: dict[str, tuple[int, int]] = {
    "official":    (0,      999),       # competition train
    "u10_pseudo":  (100000, 199999),    # U10 pseudo-labels
    "reserved":    (200000, 299999),    # reserved for future modules
    "ap_handcraft": (400000, 449999),   # Aug-Plus hand-crafted seed
    "ap_llm":      (500000, 599999),    # Aug-Plus LLM-synthesised
}

AP_HANDCRAFT_BASE: int = ID_RANGES["ap_handcraft"][0]
AP_LLM_BASE: int = ID_RANGES["ap_llm"][0]


def stable_synth_id(text: str, namespace: str = "ap_llm") -> int:
    """Deterministic ID in the ``namespace`` partition derived from ``text``.

    Two callers passing the same text get the same id, enabling SimHash-free
    near-duplicate suppression on re-runs.
    """
    if namespace not in ID_RANGES:
        raise ValueError(f"unknown namespace {namespace!r}; valid={list(ID_RANGES)}")
    lo, hi = ID_RANGES[namespace]
    span = hi - lo + 1
    h = int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16)
    return lo + (h % span)


# ---------------------------------------------------------------------------
# Hierarchy + label validation (mirrors src/inference/post_process.py).
# ---------------------------------------------------------------------------
def assert_hierarchy(labels: dict[str, str]) -> None:
    """Raise ``ValueError`` if the 4-task labels violate the hierarchical
    constraint enforced at inference time.

    Rules:
      * T1 == "No"  -> T2, T3, T4 must all be "N/A"
      * T3 == "No"  -> T4 must be "N/A"
      * T1 == "Yes" -> T2 must NOT be "N/A" (a promise must have a timeline)
    """
    t1 = labels["promise_status"]
    t2 = labels["verification_timeline"]
    t3 = labels["evidence_status"]
    t4 = labels["evidence_quality"]

    if t1 == "No":
        for k, v in (("verification_timeline", t2), ("evidence_status", t3), ("evidence_quality", t4)):
            if v != "N/A":
                raise ValueError(f"hierarchy: promise_status=No requires {k}=N/A, got {v}")
        return

    # t1 == "Yes" branch
    if t2 == "N/A":
        raise ValueError("hierarchy: promise_status=Yes requires verification_timeline != N/A")
    if t3 == "No" and t4 != "N/A":
        raise ValueError(f"hierarchy: evidence_status=No requires evidence_quality=N/A, got {t4}")


def assert_labels_valid(labels: dict[str, str]) -> None:
    """Validate label domains AND hierarchical constraint."""
    for col, domain in LABEL_DOMAINS.items():
        if col not in labels:
            raise ValueError(f"missing label column: {col}")
        if labels[col] not in domain:
            raise ValueError(f"{col}={labels[col]!r} not in {domain}")
    assert_hierarchy(labels)


# ---------------------------------------------------------------------------
# SynthRow dataclass
# ---------------------------------------------------------------------------
@dataclass
class SynthRow:
    """A single synthetic (hand-crafted or LLM-generated) ESG row.

    ``confidence_min`` / ``conf_T*`` are populated by the quality gate; for
    hand-crafted seeds they are pre-set to 1.0 (human authority = certain).
    For raw LLM output they start at 0.0 and are filled after teacher scoring.
    """
    id: int
    data: str
    promise_status: str
    verification_timeline: str
    evidence_status: str
    evidence_quality: str
    promise_string: str = ""
    evidence_string: str = ""
    esg_type: str = ""
    company: str = "synthetic"
    ticker: str = ""
    page_number: int = -1
    pdf_url: str = ""
    company_source: str = "aug_plus"
    confidence_min: float = 0.0
    conf_T1: float = 0.0
    conf_T2: float = 0.0
    conf_T3: float = 0.0
    conf_T4: float = 0.0

    def validate(self) -> None:
        """Raise on schema or hierarchy violations."""
        if not self.data or not self.data.strip():
            raise ValueError(f"id={self.id}: 'data' field is empty")
        assert_labels_valid(
            {
                "promise_status": self.promise_status,
                "verification_timeline": self.verification_timeline,
                "evidence_status": self.evidence_status,
                "evidence_quality": self.evidence_quality,
            }
        )

    def to_csv_row(self) -> dict[str, object]:
        """Materialise as a dict in canonical column order."""
        return {col: getattr(self, col) for col in PSEUDO_CSV_COLUMNS}


def validate_rows(rows: Iterable[SynthRow]) -> tuple[int, list[tuple[int, str]]]:
    """Validate every row; return (n_ok, errors)."""
    n_ok = 0
    errors: list[tuple[int, str]] = []
    for r in rows:
        try:
            r.validate()
            n_ok += 1
        except ValueError as e:
            errors.append((r.id, str(e)))
    return n_ok, errors


# ---------------------------------------------------------------------------
# Target distribution -- gives the LLM synth + quality gate a concrete budget.
#
# Bottleneck analysis (train_1000):
#   T4 Misleading:           1  -> target +60  (61x lift)
#   T2 within_2_years:      13  -> target +40  (4x  lift)
#   T4 Not Clear:          124  -> target +30  (mild balance)
#   T2 longer_than_5_years: 197 -> target +30  (mild balance)
# ---------------------------------------------------------------------------
AUG_PLUS_TARGETS: dict[str, int] = {
    "T4_Misleading":         60,
    "T2_within_2_years":     40,
    "T4_Not_Clear":          30,
    "T2_longer_than_5_years": 30,
}
