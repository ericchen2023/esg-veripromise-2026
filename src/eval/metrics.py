"""Competition scoring utilities.

Final score:
    S = 0.20 * F1(T1, positive='Yes')
      + 0.15 * MacroF1(T2)
      + 0.30 * F1(T3, positive='Yes')
      + 0.35 * MacroF1(T4)
"""
from __future__ import annotations

from typing import Sequence

from sklearn.metrics import f1_score

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

FIELD_WEIGHTS: dict[str, float] = {
    "promise_status": 0.20,
    "verification_timeline": 0.15,
    "evidence_status": 0.30,
    "evidence_quality": 0.35,
}

# T1 / T3 use binary F1 with 'Yes' as the positive class;
# T2 / T4 use macro F1 over the full label domain.
BINARY_POSITIVE: dict[str, str] = {
    "promise_status": "Yes",
    "evidence_status": "Yes",
}


def task_score(field: str, y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    """Per-task F1 according to competition rules."""
    if len(y_true) != len(y_pred):
        raise ValueError(f"length mismatch: {len(y_true)} vs {len(y_pred)}")

    if field in BINARY_POSITIVE:
        pos = BINARY_POSITIVE[field]
        # Target may be multiclass (e.g. evidence_status has Yes/No/N/A);
        # compute per-class F1 then take the positive class only. This is
        # equivalent to binary F1 for the positive label.
        scores = f1_score(
            y_true,
            y_pred,
            labels=[pos],
            average=None,
            zero_division=0,
        )
        return float(scores[0])

    labels = LABEL_DOMAINS[field]
    return float(
        f1_score(
            y_true,
            y_pred,
            labels=labels,
            average="macro",
            zero_division=0,
        )
    )


def weighted_score(
    y_true_dict: dict[str, Sequence[str]],
    y_pred_dict: dict[str, Sequence[str]],
) -> dict[str, float]:
    """Compute per-task scores and the final weighted total."""
    out: dict[str, float] = {}
    total = 0.0
    for field, weight in FIELD_WEIGHTS.items():
        s = task_score(field, y_true_dict[field], y_pred_dict[field])
        out[field] = s
        total += weight * s
    out["final_weighted_score"] = total
    return out
