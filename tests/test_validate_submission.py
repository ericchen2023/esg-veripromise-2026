from __future__ import annotations

import pandas as pd

from src.tools.validate_submission import validate_submission_file, validate_submission_frame


def _valid_submission_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "id": 1,
                "promise_status": "Yes",
                "verification_timeline": "already",
                "evidence_status": "Yes",
                "evidence_quality": "Clear",
                "promise_string": "2024 年已完成再生能源採購目標",
                "evidence_string": "第三方查證達成率 100%",
            },
            {
                "id": 2,
                "promise_status": "No",
                "verification_timeline": "N/A",
                "evidence_status": "N/A",
                "evidence_quality": "N/A",
                "promise_string": "",
                "evidence_string": "",
            },
            {
                "id": 3,
                "promise_status": "Yes",
                "verification_timeline": "within_2_years",
                "evidence_status": "No",
                "evidence_quality": "N/A",
                "promise_string": "2026 年底前完成節水改善",
                "evidence_string": "",
            },
        ]
    )


def test_valid_submission_passes() -> None:
    report = validate_submission_frame(_valid_submission_frame())
    assert report.ok
    assert report.errors == ()


def test_missing_string_columns_fail_in_submission_mode() -> None:
    frame = _valid_submission_frame().drop(columns=["promise_string", "evidence_string"])
    report = validate_submission_frame(frame, mode="submission")
    assert not report.ok
    assert "missing required columns" in report.errors[0]


def test_internal_preds_mode_allows_label_only_outputs_with_warning() -> None:
    frame = _valid_submission_frame().drop(columns=["promise_string", "evidence_string"])
    report = validate_submission_frame(frame, mode="preds")
    assert report.ok
    assert any("final submission" in warning for warning in report.warnings)


def test_promise_no_requires_downstream_na_and_blank_strings() -> None:
    frame = _valid_submission_frame()
    frame.loc[1, "verification_timeline"] = "already"
    frame.loc[1, "promise_string"] = "invalid promise"
    report = validate_submission_frame(frame)
    assert not report.ok
    assert any("promise_status=No requires verification_timeline=N/A" in error for error in report.errors)
    assert any("promise_status=No requires promise_string blank" in error for error in report.errors)


def test_evidence_no_requires_quality_na_and_blank_evidence() -> None:
    frame = _valid_submission_frame()
    frame.loc[2, "evidence_quality"] = "Clear"
    frame.loc[2, "evidence_string"] = "invalid evidence"
    report = validate_submission_frame(frame)
    assert not report.ok
    assert any("evidence_status=No requires evidence_quality=N/A" in error for error in report.errors)
    assert any("evidence_status=No requires evidence_string blank" in error for error in report.errors)


def test_promise_yes_timeline_na_fails_by_default() -> None:
    frame = _valid_submission_frame()
    frame.loc[0, "verification_timeline"] = "N/A"
    report = validate_submission_frame(frame)
    assert not report.ok
    assert any("promise_status=Yes requires verification_timeline != N/A" in error for error in report.errors)


def test_can_relax_promise_yes_timeline_guard_for_diagnostics() -> None:
    frame = _valid_submission_frame()
    frame.loc[0, "verification_timeline"] = "N/A"
    report = validate_submission_frame(frame, strict_timeline=False)
    assert report.ok


def test_duplicate_ids_fail() -> None:
    frame = _valid_submission_frame()
    frame.loc[2, "id"] = 1
    report = validate_submission_frame(frame)
    assert not report.ok
    assert any("duplicate ids" in error for error in report.errors)


def test_out_of_domain_label_fails() -> None:
    frame = _valid_submission_frame()
    frame.loc[0, "evidence_quality"] = "Unclear"
    report = validate_submission_frame(frame)
    assert not report.ok
    assert any("out-of-domain" in error for error in report.errors)


def test_file_validation_preserves_literal_na(tmp_path) -> None:
    path = tmp_path / "submission.csv"
    _valid_submission_frame().to_csv(path, index=False)
    report = validate_submission_file(path)
    assert report.ok