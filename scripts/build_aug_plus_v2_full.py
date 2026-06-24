"""Phase 38 build-step audit remediation (2026-05-21).

Background
----------
Phase 38 (§55 in MASTER_PLAN_AND_PROGRESS.md) intended to merge
``data/aug_plus/llm_synth_misleading.jsonl`` (80 rows) +
``data/aug_plus/llm_synth_within_2_years.jsonl`` (60 rows) into the v1
hand-crafted aug-plus pool to produce ``aug_plus_v2_with_u10v2.csv``. A
post-hoc audit (2026-05-21) discovered that the local historical
``data/processed/aug_plus/aug_plus_v2_with_u10v2.csv`` is byte-identical to
``aug_plus_v1_with_u10v2.csv`` (md5 = 2246df4f387ef21c866c148f15650121). The
LLM-synth jsonl files were generated correctly by ``ap_llm_synth.py``, but
the merge step that should have appended them to v1 was never executed (or
was executed against an empty queue). Stem #8
(``p2_combo_best_aug_plus_v2``) was therefore trained on the same data as
stem #7 (``p2_combo_best_aug_plus``); the ensemble gain AP-D3 → AP-D4
(+0.00244) came from non-deterministic CUDA re-run variance acting as an
extra ensemble degree of freedom rather than from the alleged LLM samples.

Why this script does NOT overwrite the historical v2 CSV
--------------------------------------------------------
The historical local/processed
``data/processed/aug_plus/aug_plus_v2_with_u10v2.csv`` is the *exact* file
that produced the stored stem #8 OOF probabilities and therefore the
SOTA ensemble score 0.71608. Overwriting it would silently invalidate the
stored OOF artifacts for any reviewer who re-ran the ensemble step. We
instead write the *true* LLM-merged corpus to a sibling path
``data/processed/aug_plus/aug_plus_v2_full_with_u10v2.csv`` and leave the
historical artifact untouched. Future phases can train a corrected stem
#8b on the new file and re-evaluate.

Usage
-----
    py -3.13 scripts/build_aug_plus_v2_full.py

Inputs (read-only):
    data/processed/aug_plus/aug_plus_v1_with_u10v2.csv
    data/aug_plus/llm_synth_misleading.jsonl
    data/aug_plus/llm_synth_within_2_years.jsonl

Outputs:
    data/processed/aug_plus/aug_plus_v2_full_with_u10v2.csv
    data/processed/aug_plus/aug_plus_v2_full_build_report.json

The script is deterministic given fixed inputs (no RNG, no network).
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd


REPO = Path(__file__).resolve().parent.parent
V1_CSV = REPO / "data" / "processed" / "aug_plus" / "aug_plus_v1_with_u10v2.csv"
LLM_MISLEADING = REPO / "data" / "aug_plus" / "llm_synth_misleading.jsonl"
LLM_WITHIN = REPO / "data" / "aug_plus" / "llm_synth_within_2_years.jsonl"
OUT_CSV = REPO / "data" / "processed" / "aug_plus" / "aug_plus_v2_full_with_u10v2.csv"
OUT_REPORT = REPO / "data" / "processed" / "aug_plus" / "aug_plus_v2_full_build_report.json"
HISTORICAL_V2 = REPO / "data" / "processed" / "aug_plus" / "aug_plus_v2_with_u10v2.csv"


def _md5(p: Path) -> str:
    h = hashlib.md5()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_jsonl(p: Path) -> list[dict]:
    rows: list[dict] = []
    with p.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise SystemExit(f"{p}:{ln} invalid JSON: {e}") from e
    return rows


def _normalize_llm_row(row: dict, kind: str, idx: int) -> dict:
    """Map an Ollama-emitted LLM-synth row onto the v1 CSV schema.

    Schema columns (from aug_plus_v1_with_u10v2.csv):
        id, data, esg_type, promise_status, promise_string,
        verification_timeline, evidence_status, evidence_string,
        evidence_quality, ...

    LLM rows always have ``data``, ``promise_string``, ``evidence_string``.
    'misleading' rows additionally specify T4=Misleading.
    'within_2_years' rows additionally specify T2=within_2_years and T4
    label (verified by ap_quality_gate during generation).
    """
    out: dict = {
        "id": f"llm_{kind}_{idx:04d}",
        "data": row["data"],
        "esg_type": row.get("esg_type", "promise_with_evidence"),
        "promise_status": row.get("promise_status", "Yes"),
        "promise_string": row.get("promise_string", ""),
        # ``llm_synth_within_2_years.jsonl`` was generated without the
        # ``verification_timeline`` key (audit 2026-05-21). When the file kind
        # is ``within_2_years`` we force T2='within_2_years' to match the
        # generation intent declared by the file name; otherwise we honor the
        # explicit value in the row (e.g., misleading-batch rows specify
        # mixed T2 labels).
        "verification_timeline": (
            "within_2_years"
            if kind == "within_2_years"
            else row.get("verification_timeline", "N/A")
        ),
        "evidence_status": row.get("evidence_status", "Yes"),
        "evidence_string": row.get("evidence_string", ""),
        "evidence_quality": row.get(
            "evidence_quality", "Misleading" if kind == "misleading" else "Clear"
        ),
    }
    # Audit trail columns
    out["_source_id"] = -1  # not derived from any train sample → never injected via _source_id filter
    out["_origin"] = f"llm_synth_{kind}"
    return out


def main() -> int:
    for p in (V1_CSV, LLM_MISLEADING, LLM_WITHIN):
        if not p.exists():
            print(f"[err] missing input: {p}", file=sys.stderr)
            return 2

    v1 = pd.read_csv(V1_CSV)
    llm_mis = _load_jsonl(LLM_MISLEADING)
    llm_win = _load_jsonl(LLM_WITHIN)

    add_rows: list[dict] = []
    for i, r in enumerate(llm_mis):
        add_rows.append(_normalize_llm_row(r, "misleading", i))
    for i, r in enumerate(llm_win):
        add_rows.append(_normalize_llm_row(r, "within_2_years", i))

    add_df = pd.DataFrame(add_rows)
    # Align columns with v1 (extra audit columns kept on the right)
    for col in v1.columns:
        if col not in add_df.columns:
            add_df[col] = pd.NA
    extra_cols = [c for c in add_df.columns if c not in v1.columns]
    add_df = add_df[list(v1.columns) + extra_cols]

    full = pd.concat([v1.assign(_source_id=v1.get("_source_id", -1), _origin=v1.get("_origin", "v1_handcraft+u10v2_pseudo")), add_df], ignore_index=True)
    full.to_csv(OUT_CSV, index=False, encoding="utf-8")

    report = {
        "created": "2026-05-21",
        "purpose": "Phase 38 build-step audit remediation (true LLM-synth merge).",
        "inputs": {
            "v1_csv": {"path": str(V1_CSV.relative_to(REPO)).replace("\\", "/"), "rows": len(v1), "md5": _md5(V1_CSV)},
            "llm_misleading_jsonl": {"path": str(LLM_MISLEADING.relative_to(REPO)).replace("\\", "/"), "rows": len(llm_mis)},
            "llm_within_2_years_jsonl": {"path": str(LLM_WITHIN.relative_to(REPO)).replace("\\", "/"), "rows": len(llm_win)},
        },
        "output": {
            "path": str(OUT_CSV.relative_to(REPO)).replace("\\", "/"),
            "rows": len(full),
            "md5": _md5(OUT_CSV),
        },
        "historical_v2_csv": {
            "path": str(HISTORICAL_V2.relative_to(REPO)).replace("\\", "/"),
            "md5": _md5(HISTORICAL_V2) if HISTORICAL_V2.exists() else None,
            "rows": int(pd.read_csv(HISTORICAL_V2).shape[0]) if HISTORICAL_V2.exists() else None,
            "note": "Untouched. md5 equals v1 (build-step bug). Stem #8 OOF was trained on this file; AP-D4 = 0.71608 reproducible from stored OOF. To consume the corrected merge, retrain a new stem #8b against the OUT_CSV emitted by this script.",
        },
        "label_distribution_full": {
            "verification_timeline": full["verification_timeline"].fillna("NaN").value_counts().to_dict(),
            "evidence_quality": full["evidence_quality"].fillna("NaN").value_counts().to_dict(),
        },
    }
    OUT_REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[ok] wrote {OUT_CSV.relative_to(REPO)} rows={len(full)} md5={_md5(OUT_CSV)}")
    print(f"[ok] wrote {OUT_REPORT.relative_to(REPO)}")
    print("[note] historical aug_plus_v2_with_u10v2.csv left untouched to preserve stem #8 OOF reproducibility.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
