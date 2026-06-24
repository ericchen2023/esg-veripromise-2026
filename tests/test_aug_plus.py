"""Unit tests for the Aug-Plus augmentation suite.

Covers:
  * Schema column layout and the ID-space partition contract.
  * Hierarchy validator catches every disallowed combination.
  * Hand-crafted seed CSV builds, passes validation, and respects the
    minority-class budget.
  * LLM-synth CLI dry-run (Mock provider) is deterministic and produces
    well-formed JSONL.
  * Quality gate SimHash + length filter behaves as expected.
"""
from __future__ import annotations

import csv
import importlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.aug_schema import (  # noqa: E402
    AP_HANDCRAFT_BASE,
    AP_LLM_BASE,
    ID_RANGES,
    PSEUDO_CSV_COLUMNS,
    SynthRow,
    assert_hierarchy,
    assert_labels_valid,
    stable_synth_id,
    validate_rows,
)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
def test_pseudo_csv_columns_match_u10_layout():
    expected_subset = (
        "id", "data", "promise_status", "verification_timeline",
        "evidence_status", "evidence_quality", "conf_T1", "conf_T2",
        "conf_T3", "conf_T4", "confidence_min",
    )
    for col in expected_subset:
        assert col in PSEUDO_CSV_COLUMNS


def test_id_ranges_do_not_overlap():
    spans = sorted(ID_RANGES.values())
    for (a_lo, a_hi), (b_lo, b_hi) in zip(spans, spans[1:]):
        assert a_hi < b_lo, f"overlap between {(a_lo, a_hi)} and {(b_lo, b_hi)}"


def test_stable_synth_id_deterministic_and_in_partition():
    a = stable_synth_id("foo", namespace="ap_llm")
    b = stable_synth_id("foo", namespace="ap_llm")
    assert a == b
    lo, hi = ID_RANGES["ap_llm"]
    assert lo <= a <= hi


# ---------------------------------------------------------------------------
# Hierarchy
# ---------------------------------------------------------------------------
def test_hierarchy_t1_no_requires_all_na():
    with pytest.raises(ValueError, match="promise_status=No"):
        assert_hierarchy({
            "promise_status": "No",
            "verification_timeline": "already",
            "evidence_status": "N/A",
            "evidence_quality": "N/A",
        })


def test_hierarchy_t1_yes_requires_t2_not_na():
    with pytest.raises(ValueError, match="verification_timeline != N/A"):
        assert_hierarchy({
            "promise_status": "Yes",
            "verification_timeline": "N/A",
            "evidence_status": "Yes",
            "evidence_quality": "Clear",
        })


def test_hierarchy_t3_no_requires_t4_na():
    with pytest.raises(ValueError, match="evidence_status=No"):
        assert_hierarchy({
            "promise_status": "Yes",
            "verification_timeline": "already",
            "evidence_status": "No",
            "evidence_quality": "Clear",
        })


def test_hierarchy_valid_combination_passes():
    assert_hierarchy({
        "promise_status": "Yes",
        "verification_timeline": "within_2_years",
        "evidence_status": "Yes",
        "evidence_quality": "Misleading",
    })


def test_assert_labels_valid_rejects_unknown_domain():
    with pytest.raises(ValueError, match="not in"):
        assert_labels_valid({
            "promise_status": "Maybe",  # invalid
            "verification_timeline": "within_2_years",
            "evidence_status": "Yes",
            "evidence_quality": "Clear",
        })


# ---------------------------------------------------------------------------
# Hand-crafted seed
# ---------------------------------------------------------------------------
def test_handcrafted_seed_csv_exists_and_validates():
    csv_path = ROOT / "assets" / "aug_plus" / "handcrafted_v1.csv"
    assert csv_path.exists(), "handcrafted seed CSV not built; run build_handcrafted_v1.py"
    with csv_path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) >= 40, f"expected >= 40 seed rows, got {len(rows)}"
    # Validate all label tuples
    for r in rows:
        assert_labels_valid({
            "promise_status": r["promise_status"],
            "verification_timeline": r["verification_timeline"],
            "evidence_status": r["evidence_status"],
            "evidence_quality": r["evidence_quality"],
        })
    # Minority-class coverage
    t4_misleading = sum(1 for r in rows if r["evidence_quality"] == "Misleading")
    t2_within2 = sum(1 for r in rows if r["verification_timeline"] == "within_2_years")
    assert t4_misleading >= 15, f"T4=Misleading coverage too low: {t4_misleading}"
    assert t2_within2 >= 15, f"T2=within_2_years coverage too low: {t2_within2}"
    # IDs in handcraft partition
    lo, hi = ID_RANGES["ap_handcraft"]
    for r in rows:
        assert lo <= int(r["id"]) <= hi


# ---------------------------------------------------------------------------
# LLM-synth CLI (Mock provider, deterministic)
# ---------------------------------------------------------------------------
def test_mock_provider_deterministic_with_seed(tmp_path, monkeypatch):
    monkeypatch.chdir(ROOT)  # ensure relative paths resolve
    ap_llm_synth = importlib.import_module("scripts.ap_llm_synth")
    spec = ap_llm_synth.load_prompt("misleading")
    rows_a = ap_llm_synth._mock_provider(spec, n=10, seed=123)
    rows_b = ap_llm_synth._mock_provider(spec, n=10, seed=123)
    assert rows_a == rows_b
    assert len(rows_a) == 10
    for r in rows_a:
        assert "data" in r and r["data"].strip()
        assert "verification_timeline" in r


def test_synth_row_validate_round_trip():
    r = SynthRow(
        id=AP_LLM_BASE + 1,
        data="本公司承諾2030年達淨零，公司高度重視永續發展。",
        promise_status="Yes",
        verification_timeline="between_2_and_5_years",
        evidence_status="Yes",
        evidence_quality="Misleading",
    )
    r.validate()
    row_dict = r.to_csv_row()
    assert set(row_dict.keys()) == set(PSEUDO_CSV_COLUMNS)


def test_validate_rows_aggregates_errors():
    good = SynthRow(
        id=AP_LLM_BASE + 2, data="ok text",
        promise_status="No",
        verification_timeline="N/A",
        evidence_status="N/A",
        evidence_quality="N/A",
    )
    bad = SynthRow(
        id=AP_LLM_BASE + 3, data="bad",
        promise_status="Yes",
        verification_timeline="N/A",  # hierarchy violation
        evidence_status="Yes",
        evidence_quality="Clear",
    )
    n_ok, errors = validate_rows([good, bad])
    assert n_ok == 1 and len(errors) == 1
    assert errors[0][0] == bad.id


# ---------------------------------------------------------------------------
# Quality gate
# ---------------------------------------------------------------------------
def test_quality_gate_dedup_and_length(tmp_path):
    ap_gate = importlib.import_module("scripts.ap_quality_gate")
    a = ap_gate.simhash("本公司承諾2030年達成淨零碳排目標並提升永續經營績效")
    b = ap_gate.simhash("本公司承諾2030年達成淨零碳排目標並提升永續經營績效")
    assert a == b
    assert ap_gate.hamming(a, b) == 0
    c = ap_gate.simhash("這是完全無關的另一句話用來測試 simhash 區辨能力")
    assert ap_gate.hamming(a, c) > 10


def test_quality_gate_cli_smoke(tmp_path):
    # Build a tiny merged CSV: 1 ok row + 1 too-short row + 1 schema-bad row
    merged = tmp_path / "merged.csv"
    fieldnames = list(PSEUDO_CSV_COLUMNS)
    with merged.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)
        w.writeheader()
        def row(**kw):
            base = {k: "" for k in fieldnames}
            base.update(kw)
            return base
        w.writerow(row(
            id=500001,
            data="本公司承諾於2026年底前完成總部太陽能板建置並降低用電成本約三成。",
            promise_status="Yes", verification_timeline="within_2_years",
            evidence_status="No", evidence_quality="N/A",
            confidence_min="1.0",
        ))
        w.writerow(row(
            id=500002, data="太短",
            promise_status="No", verification_timeline="N/A",
            evidence_status="N/A", evidence_quality="N/A",
            confidence_min="1.0",
        ))
        w.writerow(row(
            id=500003,
            data="這是一段不算太短但 label 違反階層的句子用以測試 schema drop。",
            promise_status="Yes", verification_timeline="N/A",  # invalid
            evidence_status="Yes", evidence_quality="Clear",
            confidence_min="1.0",
        ))
    out_csv = tmp_path / "gated.csv"
    rc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "ap_quality_gate.py"),
         "--in", str(merged),
         "--out", str(out_csv),
         "--report", str(tmp_path / "report.json"),
         "--train-csv", "",
         "--skip-teacher"],
        cwd=str(ROOT),
        capture_output=True, text=True,
    )
    assert rc.returncode == 0, rc.stderr
    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert report["kept"] == 1
    assert report["dropped_by_reason"].get("length") == 1
    assert report["dropped_by_reason"].get("schema") == 1
