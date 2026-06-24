"""Unit tests for src.data.loader — label normalisation and schema validation."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.loader import load_dataset, LABEL_DOMAINS

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_COMMON_COLS = [
    "id", "data", "promise_status", "verification_timeline",
    "evidence_status", "evidence_quality",
]

def _make_df(**kwargs) -> pd.DataFrame:
    """Minimal valid row with all task fields, overridden by kwargs."""
    base = {
        "id": 1,
        "data": "企業承諾 2025 年達成淨零目標。",
        "promise_status": "Yes",
        "verification_timeline": "already",
        "evidence_status": "Yes",
        "evidence_quality": "Clear",
    }
    base.update(kwargs)
    return pd.DataFrame([base])


def _write_and_load(df: pd.DataFrame, tmp_path: Path) -> tuple[list, pd.DataFrame]:
    p = tmp_path / "tmp.csv"
    df.to_csv(p, index=False)
    return load_dataset(str(p))


# ---------------------------------------------------------------------------
# Label alias normalisation
# ---------------------------------------------------------------------------

class TestLabelAliasNormalisation:
    """Verify _LABEL_ALIASES in loader.py: more_than_5_years → longer_than_5_years."""

    def test_more_than_5_years_normalized_to_longer(self, tmp_path) -> None:
        """Val-set label 'more_than_5_years' must be normalised to 'longer_than_5_years'."""
        df = _make_df(verification_timeline="more_than_5_years")
        records, out_df = _write_and_load(df, tmp_path)
        assert out_df["verification_timeline"].iloc[0] == "longer_than_5_years"
        assert records[0]["verification_timeline"] == "longer_than_5_years"

    def test_longer_than_5_years_unchanged(self, tmp_path) -> None:
        """Train-set canonical label 'longer_than_5_years' must pass through unchanged."""
        df = _make_df(verification_timeline="longer_than_5_years")
        records, out_df = _write_and_load(df, tmp_path)
        assert out_df["verification_timeline"].iloc[0] == "longer_than_5_years"

    def test_already_unchanged(self, tmp_path) -> None:
        df = _make_df(verification_timeline="already")
        records, out_df = _write_and_load(df, tmp_path)
        assert out_df["verification_timeline"].iloc[0] == "already"

    def test_nan_timeline_normalised_to_na_string(self, tmp_path) -> None:
        """NaN verification_timeline (No-promise rows) must become 'N/A'."""
        import numpy as np
        df = _make_df(verification_timeline=float("nan"))
        records, out_df = _write_and_load(df, tmp_path)
        assert out_df["verification_timeline"].iloc[0] == "N/A"

    def test_val_batch_normalisation(self, tmp_path) -> None:
        """A batch containing both alias variants must all normalise correctly."""
        rows = [
            _make_df(verification_timeline="more_than_5_years").iloc[0].to_dict(),
            _make_df(verification_timeline="longer_than_5_years").iloc[0].to_dict(),
            _make_df(verification_timeline="already").iloc[0].to_dict(),
        ]
        df = pd.DataFrame(rows)
        p = tmp_path / "batch.csv"
        df.to_csv(p, index=False)
        records, out_df = load_dataset(str(p))
        values = out_df["verification_timeline"].tolist()
        assert values[0] == "longer_than_5_years", "alias not resolved"
        assert values[1] == "longer_than_5_years", "canonical changed unexpectedly"
        assert values[2] == "already", "unrelated label changed"

    def test_no_alias_in_other_tasks(self, tmp_path) -> None:
        """Alias mapping must NOT affect other task columns."""
        df = _make_df(promise_status="Yes", evidence_status="Yes", evidence_quality="Clear")
        records, out_df = _write_and_load(df, tmp_path)
        assert out_df["promise_status"].iloc[0] == "Yes"
        assert out_df["evidence_status"].iloc[0] == "Yes"
        assert out_df["evidence_quality"].iloc[0] == "Clear"


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------

class TestSchemaValidation:
    def test_valid_train_row_loads(self, tmp_path) -> None:
        """Standard train-set row (longer_than_5_years) loads without error."""
        df = _make_df(verification_timeline="longer_than_5_years")
        records, out_df = _write_and_load(df, tmp_path)
        assert len(records) == 1

    def test_valid_val_row_loads(self, tmp_path) -> None:
        """Val-set row (more_than_5_years) loads without error after normalisation."""
        df = _make_df(verification_timeline="more_than_5_years")
        records, out_df = _write_and_load(df, tmp_path)
        assert len(records) == 1

    def test_unknown_label_raises(self, tmp_path) -> None:
        """Completely unknown label must still raise ValueError."""
        df = _make_df(verification_timeline="invalid_label_xyz")
        with pytest.raises(ValueError, match="invalid_label_xyz"):
            _write_and_load(df, tmp_path)

    def test_label_domains_contain_longer_than(self) -> None:
        """LABEL_DOMAINS in loader.py must still list 'longer_than_5_years' as canonical."""
        assert "longer_than_5_years" in LABEL_DOMAINS["verification_timeline"]
        assert "more_than_5_years" not in LABEL_DOMAINS["verification_timeline"]
