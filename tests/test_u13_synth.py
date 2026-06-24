"""Tests for Phase 37 U13 synthetic-data pipeline.

Run with:  pytest tests/test_u13_synth.py -v
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.synth_schema import (
    PHASE37_TARGET_COUNTS,
    PSEUDO_CSV_COLUMNS,
    SYNTH_ID_BASE,
    SynthRow,
    assert_hierarchy,
    assert_labels_valid,
    iter_target_specs,
    stable_synth_id,
)


# --- Schema tests -----------------------------------------------------------

def test_pseudo_columns_match_u10():
    """U13 schema must exactly mirror U10 pseudo CSV column order."""
    u10 = pd.read_csv(ROOT / "data/processed/u10/pseudo_labels.csv", nrows=1)
    assert list(u10.columns) == list(PSEUDO_CSV_COLUMNS)


def test_hierarchy_valid_combos():
    # T1=No → all N/A
    assert_hierarchy({
        "promise_status": "No",
        "verification_timeline": "N/A",
        "evidence_status": "N/A",
        "evidence_quality": "N/A",
    })
    # T1=Yes T3=Yes T4=anything but N/A
    assert_hierarchy({
        "promise_status": "Yes",
        "verification_timeline": "within_2_years",
        "evidence_status": "Yes",
        "evidence_quality": "Clear",
    })
    # T3=No → T4 must be N/A
    assert_hierarchy({
        "promise_status": "Yes",
        "verification_timeline": "already",
        "evidence_status": "No",
        "evidence_quality": "N/A",
    })


@pytest.mark.parametrize("bad", [
    {"promise_status": "No", "verification_timeline": "already", "evidence_status": "N/A", "evidence_quality": "N/A"},
    {"promise_status": "Yes", "verification_timeline": "N/A", "evidence_status": "Yes", "evidence_quality": "Clear"},
    {"promise_status": "Yes", "verification_timeline": "already", "evidence_status": "No", "evidence_quality": "Clear"},
])
def test_hierarchy_rejects_invalid(bad):
    with pytest.raises(ValueError):
        assert_hierarchy(bad)


def test_assert_labels_rejects_out_of_domain():
    bad = {
        "promise_status": "Maybe",
        "verification_timeline": "already",
        "evidence_status": "Yes",
        "evidence_quality": "Clear",
    }
    with pytest.raises(ValueError):
        assert_labels_valid(bad)


# --- ID determinism ---------------------------------------------------------

def test_stable_id_deterministic():
    a = stable_synth_id("台積電承諾 2030 年完成減碳 40%。")
    b = stable_synth_id("台積電承諾 2030 年完成減碳 40%。")
    assert a == b
    assert SYNTH_ID_BASE <= a < SYNTH_ID_BASE + 10_000_000


def test_stable_id_changes_with_text():
    a = stable_synth_id("台積電承諾 2030 年完成減碳 40%。")
    b = stable_synth_id("台積電承諾 2030 年完成減碳 41%。")
    assert a != b


# --- SynthRow → CSV dict ----------------------------------------------------

def test_synthrow_round_trip():
    sr = SynthRow(
        text="本公司承諾於 2026 年完成範疇一減量 30%。",
        labels={
            "promise_status": "Yes",
            "verification_timeline": "within_2_years",
            "evidence_status": "Yes",
            "evidence_quality": "Clear",
        },
        source="u13_test",
        confidence=0.95,
    )
    d = sr.to_csv_dict()
    assert set(d.keys()) == set(PSEUDO_CSV_COLUMNS)
    assert d["promise_status"] == "Yes"
    assert d["confidence_min"] == 0.95
    assert SYNTH_ID_BASE <= d["id"] < SYNTH_ID_BASE + 10_000_000


# --- Target spec coverage ---------------------------------------------------

def test_target_specs_all_hierarchy_valid():
    for spec in iter_target_specs():
        assert_labels_valid(spec["labels"])


def test_target_counts_focus_on_minorities():
    """Sanity check: 60+ synth rows targeting T4=Misleading (real corpus has 1)."""
    n_misleading = sum(n for (_, t4), n in PHASE37_TARGET_COUNTS.items() if t4 == "Misleading")
    n_within_2y = sum(n for (t2, _), n in PHASE37_TARGET_COUNTS.items() if t2 == "within_2_years")
    assert n_misleading >= 100, "should generate ≥100 Misleading examples"
    assert n_within_2y >= 100, "should generate ≥100 within_2_years examples"


# --- CLI smoke test (mock provider, end-to-end) ----------------------------

def test_cli_pipeline_with_mock(tmp_path):
    raw = tmp_path / "raw.jsonl"
    flt = tmp_path / "filtered.jsonl"
    csv = tmp_path / "pseudo.csv"
    script = ROOT / "scripts" / "u13_synth_llm.py"

    # Generate
    r = subprocess.run([sys.executable, str(script), "generate", "--provider", "mock", "--output", str(raw)],
                       capture_output=True, text=True, cwd=ROOT, timeout=120)
    assert r.returncode == 0, r.stderr
    assert raw.exists()
    n_lines = sum(1 for _ in raw.open(encoding="utf-8"))
    assert n_lines == sum(s["n"] for s in iter_target_specs())

    # Validate
    r = subprocess.run([sys.executable, str(script), "validate", "--input", str(raw), "--output", str(flt),
                        "--official", str(ROOT / "vpesg4k_train_1000 V1.csv"),
                        "--min-len", "30", "--max-len", "600"],
                       capture_output=True, text=True, cwd=ROOT, timeout=60)
    assert r.returncode == 0, r.stderr
    assert flt.exists()

    # Promote
    r = subprocess.run([sys.executable, str(script), "promote", "--input", str(flt), "--output", str(csv),
                        "--confidence", "0.95"],
                       capture_output=True, text=True, cwd=ROOT, timeout=60)
    assert r.returncode == 0, r.stderr
    df = pd.read_csv(csv)
    assert len(df) > 100, "should promote > 100 rows even after dedup"
    assert list(df.columns) == list(PSEUDO_CSV_COLUMNS)
    # All synth IDs in correct range
    assert df["id"].min() >= SYNTH_ID_BASE
    # All hierarchy-valid. NOTE: pandas treats "N/A" as NaN on read (codebase
    # convention — see data/processed/u10/pseudo_labels_v3.csv with 1413 NaN
    # rows in verification_timeline). Re-fill before validating.
    df = df.fillna("N/A")
    for _, row in df.iterrows():
        assert_hierarchy({
            "promise_status": row["promise_status"],
            "verification_timeline": row["verification_timeline"],
            "evidence_status": row["evidence_status"],
            "evidence_quality": row["evidence_quality"],
        })
