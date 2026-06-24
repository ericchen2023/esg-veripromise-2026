"""Validate VeriPromise ESG prediction/submission CSV files.

The ensemble tools write internal ``*_preds.csv`` files with ``id`` plus the
four task labels. Final competition submissions may additionally require
``promise_string`` and ``evidence_string``. This module validates both modes so
the same guard can be used before local analysis and before final upload.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.dataset import TASKS
from src.data.loader import LABEL_DOMAINS

TEXT_COLUMNS = ("promise_string", "evidence_string")
ID_COLUMN = "id"


@dataclass(frozen=True)
class ValidationReport:
    rows: int
    mode: str
    errors: tuple[str, ...]
    warnings: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.errors


def _is_blank(value: Any) -> bool:
    if pd.isna(value):
        return True
    return str(value).strip() == ""


def _required_columns(mode: str) -> tuple[str, ...]:
    if mode == "preds":
        return (ID_COLUMN, *TASKS)
    if mode == "submission":
        return (ID_COLUMN, *TASKS, *TEXT_COLUMNS)
    raise ValueError(f"unknown validation mode: {mode!r}")


def validate_submission_frame(
    frame: pd.DataFrame,
    *,
    mode: str = "submission",
    strict_timeline: bool = True,
) -> ValidationReport:
    """Return validation errors/warnings for a prediction frame.

    Args:
        frame: DataFrame loaded from a prediction/submission CSV.
        mode: ``submission`` requires text columns; ``preds`` only requires
            ``id`` and the four task labels.
        strict_timeline: When true, ``promise_status=Yes`` cannot have
            ``verification_timeline=N/A``. This matches synthetic-data schema
            and is safer for final submission. Disable only for diagnostic OOF
            predictions if needed.
    """
    required = _required_columns(mode)
    errors: list[str] = []
    warnings: list[str] = []

    missing = [column for column in required if column not in frame.columns]
    if missing:
        errors.append(f"missing required columns: {missing}")

    if ID_COLUMN in frame.columns:
        if frame[ID_COLUMN].isna().any():
            errors.append("id contains empty values")
        duplicated = frame[frame[ID_COLUMN].duplicated()][ID_COLUMN].tolist()
        if duplicated:
            preview = duplicated[:10]
            errors.append(f"duplicate ids: {preview}{' ...' if len(duplicated) > 10 else ''}")

    for task in TASKS:
        if task not in frame.columns:
            continue
        bad_values = sorted(set(frame[task].dropna().astype(str)) - set(LABEL_DOMAINS[task]))
        if bad_values:
            errors.append(f"{task} has out-of-domain labels: {bad_values}")

    if errors:
        return ValidationReport(rows=len(frame), mode=mode, errors=tuple(errors), warnings=tuple(warnings))

    for row_index, row in frame.iterrows():
        row_id = row.get(ID_COLUMN, row_index)
        promise_status = str(row["promise_status"])
        verification_timeline = str(row["verification_timeline"])
        evidence_status = str(row["evidence_status"])
        evidence_quality = str(row["evidence_quality"])

        prefix = f"row id={row_id!r}"
        if promise_status == "No":
            for column, actual in (
                ("verification_timeline", verification_timeline),
                ("evidence_status", evidence_status),
                ("evidence_quality", evidence_quality),
            ):
                if actual != "N/A":
                    errors.append(f"{prefix}: promise_status=No requires {column}=N/A, got {actual!r}")
            if "promise_string" in frame.columns and not _is_blank(row["promise_string"]):
                errors.append(f"{prefix}: promise_status=No requires promise_string blank")
            if "evidence_string" in frame.columns and not _is_blank(row["evidence_string"]):
                errors.append(f"{prefix}: promise_status=No requires evidence_string blank")

        if promise_status == "Yes" and strict_timeline and verification_timeline == "N/A":
            errors.append(f"{prefix}: promise_status=Yes requires verification_timeline != N/A")

        if evidence_status == "No":
            if evidence_quality != "N/A":
                errors.append(f"{prefix}: evidence_status=No requires evidence_quality=N/A, got {evidence_quality!r}")
            if "evidence_string" in frame.columns and not _is_blank(row["evidence_string"]):
                errors.append(f"{prefix}: evidence_status=No requires evidence_string blank")

        if mode == "preds":
            for text_column in TEXT_COLUMNS:
                if text_column not in frame.columns:
                    warnings.append(f"{text_column} is absent in preds mode; final submission must add/validate it")

    warnings = sorted(set(warnings))
    return ValidationReport(rows=len(frame), mode=mode, errors=tuple(errors), warnings=tuple(warnings))


def validate_submission_file(
    path: str | Path,
    *,
    mode: str = "submission",
    strict_timeline: bool = True,
) -> ValidationReport:
    csv_path = Path(path)
    frame = pd.read_csv(csv_path, encoding="utf-8", keep_default_na=False)
    return validate_submission_frame(frame, mode=mode, strict_timeline=strict_timeline)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate ESG VeriPromise prediction/submission CSV files.")
    parser.add_argument("path", help="CSV file to validate.")
    parser.add_argument(
        "--mode",
        choices=("submission", "preds"),
        default="submission",
        help="submission requires promise/evidence string columns; preds validates internal label-only outputs.",
    )
    parser.add_argument(
        "--allow-promise-yes-na-timeline",
        action="store_true",
        help="Disable the strict promise_status=Yes -> timeline != N/A guard.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    report = validate_submission_file(
        args.path,
        mode=args.mode,
        strict_timeline=not args.allow_promise_yes_na_timeline,
    )
    if report.ok:
        print(f"[validate-submission] OK rows={report.rows} mode={report.mode}")
        for warning in report.warnings:
            print(f"[validate-submission] WARN {warning}")
        return

    print(f"[validate-submission] FAILED rows={report.rows} mode={report.mode}")
    for error in report.errors[:50]:
        print(f"[validate-submission] ERROR {error}")
    if len(report.errors) > 50:
        print(f"[validate-submission] ERROR ... {len(report.errors) - 50} more")
    raise SystemExit(1)


if __name__ == "__main__":
    main()