"""Unit tests for src.eval.metrics."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval.metrics import FIELD_WEIGHTS, task_score, weighted_score


def test_perfect_predictions():
    # Cover every label in every task domain so Macro-F1 = 1.0
    y_true = {
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
    # Pad each task to the same length so weighted_score length-checks pass
    n = max(len(v) for v in y_true.values())
    for k, v in list(y_true.items()):
        while len(v) < n:
            v.append(v[-1])
        y_true[k] = v
    out = weighted_score(y_true, y_true)
    for field in FIELD_WEIGHTS:
        assert out[field] == pytest.approx(1.0)
    assert out["final_weighted_score"] == pytest.approx(1.0)


def test_weight_sum_is_one():
    assert sum(FIELD_WEIGHTS.values()) == pytest.approx(1.0)


def test_promise_status_uses_binary_yes_positive():
    # Predict all 'No' -> binary F1 with positive='Yes' must be 0
    y_true = ["Yes", "Yes", "No"]
    y_pred = ["No", "No", "No"]
    s = task_score("promise_status", y_true, y_pred)
    assert s == pytest.approx(0.0)


def test_evidence_status_binary_yes_positive():
    # 1 TP, 1 FN, 1 TN -> precision=1, recall=0.5, F1=2/3
    y_true = ["Yes", "Yes", "No"]
    y_pred = ["Yes", "No", "No"]
    s = task_score("evidence_status", y_true, y_pred)
    assert s == pytest.approx(2 / 3, rel=1e-6)


def test_macro_f1_for_t4():
    # All predictions 'Clear' -> only Clear class scores; others = 0
    y_true = ["Clear", "Not Clear", "Misleading", "N/A"]
    y_pred = ["Clear", "Clear", "Clear", "Clear"]
    s = task_score("evidence_quality", y_true, y_pred)
    # Clear: precision=1/4, recall=1, F1=0.4; others=0; macro = 0.4/4 = 0.1
    assert s == pytest.approx(0.1, rel=1e-6)


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        task_score("promise_status", ["Yes"], ["Yes", "No"])
