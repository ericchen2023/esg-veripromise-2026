"""Unit tests for hierarchical post-processing constraints."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.inference.post_process import apply_constraints, apply_constraints_batch


def test_promise_no_overrides_all_downstream():
    r = {
        "promise_status": "No",
        "verification_timeline": "already",       # invalid; must be wiped
        "evidence_status": "Yes",                  # invalid; must be wiped
        "evidence_quality": "Clear",               # invalid; must be wiped
        "promise_string": "we will reduce by 30%",
        "evidence_string": "verified by EY",
    }
    out = apply_constraints(r)
    assert out["promise_status"] == "No"
    assert out["verification_timeline"] == "N/A"
    assert out["evidence_status"] == "N/A"
    assert out["evidence_quality"] == "N/A"
    assert out["promise_string"] == ""
    assert out["evidence_string"] == ""


def test_evidence_no_overrides_quality_and_string():
    r = {
        "promise_status": "Yes",
        "verification_timeline": "already",
        "evidence_status": "No",
        "evidence_quality": "Clear",                 # invalid; must be wiped
        "promise_string": "promise text",
        "evidence_string": "old evidence",          # must be wiped
    }
    out = apply_constraints(r)
    assert out["evidence_status"] == "No"
    assert out["evidence_quality"] == "N/A"
    assert out["evidence_string"] == ""
    # Promise side untouched
    assert out["promise_status"] == "Yes"
    assert out["verification_timeline"] == "already"
    assert out["promise_string"] == "promise text"


def test_valid_record_unchanged():
    r = {
        "promise_status": "Yes",
        "verification_timeline": "within_2_years",
        "evidence_status": "Yes",
        "evidence_quality": "Clear",
        "promise_string": "p",
        "evidence_string": "e",
    }
    assert apply_constraints(r) == r


def test_does_not_mutate_input():
    r = {
        "promise_status": "No",
        "verification_timeline": "already",
        "evidence_status": "Yes",
        "evidence_quality": "Clear",
    }
    snapshot = dict(r)
    _ = apply_constraints(r)
    assert r == snapshot


def test_batch_processing():
    rs = [
        {"promise_status": "No", "verification_timeline": "already",
         "evidence_status": "Yes", "evidence_quality": "Clear"},
        {"promise_status": "Yes", "verification_timeline": "already",
         "evidence_status": "No", "evidence_quality": "Clear"},
    ]
    outs = apply_constraints_batch(rs)
    assert outs[0]["evidence_status"] == "N/A"
    assert outs[1]["evidence_quality"] == "N/A"
