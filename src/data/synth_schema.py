"""Schema + validation helpers for Phase 37 synthetic ESG data.

Shared between ``scripts/u13_synth_llm.py`` (generation), ``u13_manual_seed.py``
(template authoring) and the downstream pseudo-label trainer.

The synthetic CSV follows the same column layout as
``data/processed/u10/pseudo_labels*.csv`` so that ``train_pseudo_kfold.py``
can consume both without code changes.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Iterable

# Reuse canonical label domains (single source of truth).
from src.data.loader import LABEL_DOMAINS

# Pseudo-CSV columns, in canonical order.
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

# Hierarchical constraint enforced post-prediction in src/inference/post_process.py.
# We assert the same invariants on every synthetic row so we never inject
# logically-impossible labels into the training pool.
def assert_hierarchy(labels: dict[str, str]) -> None:
    t1 = labels["promise_status"]
    t2 = labels["verification_timeline"]
    t3 = labels["evidence_status"]
    t4 = labels["evidence_quality"]
    if t1 == "No":
        for k, v in (("verification_timeline", t2), ("evidence_status", t3), ("evidence_quality", t4)):
            if v != "N/A":
                raise ValueError(f"hierarchy: promise_status=No requires {k}=N/A, got {v}")
    if t3 == "No" and t4 != "N/A":
        raise ValueError(f"hierarchy: evidence_status=No requires evidence_quality=N/A, got {t4}")
    # T2 N/A is required iff T1 == No. (Cannot have T1=Yes and T2=N/A.)
    if t1 == "Yes" and t2 == "N/A":
        raise ValueError("hierarchy: promise_status=Yes requires verification_timeline != N/A")


def assert_labels_valid(labels: dict[str, str]) -> None:
    for col, domain in LABEL_DOMAINS.items():
        if col not in labels:
            raise ValueError(f"missing label column: {col}")
        if labels[col] not in domain:
            raise ValueError(f"{col}={labels[col]!r} not in {domain}")
    assert_hierarchy(labels)


# Synthetic ID range. Chosen to never collide with:
#   - Official competition IDs (0 ~ 999)
#   - U10 pseudo IDs (100000 ~ 199999)
#   - Reserved U11/U12 (200000 ~ 299999)
# Phase 37 (u13) owns the 300000+ range.
SYNTH_ID_BASE: int = 300000


def stable_synth_id(text: str, salt: str = "u13") -> int:
    """Deterministic 7-digit ID derived from text + salt.

    Allows the pipeline to be idempotent: re-generating the same prompt will
    produce the same ID, so promotion to CSV does not introduce duplicates.
    """
    h = hashlib.blake2b(f"{salt}::{text}".encode("utf-8"), digest_size=4).hexdigest()
    return SYNTH_ID_BASE + (int(h, 16) % 9_000_000)


@dataclass
class SynthRow:
    """One synthetic training example (LLM-generated or hand-crafted)."""

    text: str
    labels: dict[str, str]
    source: str  # e.g. "u13_synth_llm:openai_gpt4o", "u13_manual"
    generator_meta: dict[str, object] = field(default_factory=dict)
    confidence: float = 1.0  # asserted by author/LLM. 1.0 = full trust.

    def __post_init__(self) -> None:
        if not isinstance(self.text, str) or len(self.text.strip()) == 0:
            raise ValueError("text must be non-empty string")
        assert_labels_valid(self.labels)
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError("confidence must be in [0, 1]")

    def to_csv_dict(self) -> dict:
        sid = stable_synth_id(self.text)
        return {
            "id": sid,
            "data": self.text,
            "esg_type": "",
            "promise_status": self.labels["promise_status"],
            "promise_string": self.labels.get("promise_string", ""),
            "verification_timeline": self.labels["verification_timeline"],
            "evidence_status": self.labels["evidence_status"],
            "evidence_string": self.labels.get("evidence_string", ""),
            "evidence_quality": self.labels["evidence_quality"],
            "company": self.generator_meta.get("company", ""),
            "ticker": "",
            "page_number": "",
            "pdf_url": "",
            "company_source": self.source,
            "confidence_min": self.confidence,
            "conf_T1": self.confidence,
            "conf_T2": self.confidence,
            "conf_T3": self.confidence,
            "conf_T4": self.confidence,
        }


# Target counts for Phase 37 first batch. Tuned to lift the two weakest classes
# (T2 within_2y has 13 real samples; T4 Misleading has 1) to a learnable
# regime (~10% of total dataset) without overwhelming the real-data prior.
PHASE37_TARGET_COUNTS: dict[tuple[str, str], int] = {
    # (verification_timeline, evidence_quality) → n samples
    # Focus on combos that are rare AND logically valid under hierarchy.
    ("within_2_years", "Clear"): 60,
    ("within_2_years", "Not Clear"): 40,
    ("within_2_years", "Misleading"): 30,
    ("longer_than_5_years", "Misleading"): 40,
    ("longer_than_5_years", "Not Clear"): 30,
    ("between_2_and_5_years", "Misleading"): 30,
    ("already", "Misleading"): 30,
    # Pure-T4 reinforcement for Clear/Not Clear baselines
    ("already", "Not Clear"): 40,
    # T1=No / T3=No anchors (under-represented hierarchies)
    ("N/A", "N/A"): 30,  # T1=No
}


def iter_target_specs() -> Iterable[dict]:
    """Yield generation specs for each Phase 37 target bucket."""
    for (t2, t4), n in PHASE37_TARGET_COUNTS.items():
        if t2 == "N/A":
            # T1=No → all downstream tasks are N/A
            yield {
                "n": n,
                "labels": {
                    "promise_status": "No",
                    "verification_timeline": "N/A",
                    "evidence_status": "N/A",
                    "evidence_quality": "N/A",
                },
                "bucket": "T1=No",
            }
        else:
            # T1=Yes; T3 must be Yes if T4 != N/A
            yield {
                "n": n,
                "labels": {
                    "promise_status": "Yes",
                    "verification_timeline": t2,
                    "evidence_status": "Yes",
                    "evidence_quality": t4,
                },
                "bucket": f"T2={t2} / T4={t4}",
            }
