"""Apples-to-apples comparison: Phase 37 (Aug-Plus) vs Phase 36 stem #6 baseline.

Both runs share identical recipe (classw + Focal-T4 + U6-pro back-translation,
seed=42, 5-fold stratified, stage A 4 ep + stage B 3 ep). The only difference
is the pseudo CSV:
  - Phase 36 stem #6 : data/processed/u10/pseudo_labels_v2.csv (3,904 rows)
  - Phase 37         : data/processed/aug_plus/aug_plus_v1_with_u10v2.csv
                        (50 hand-crafted AP + 3,904 U10v2 = 3,954 rows;
                         trainer caps pseudo to 2,000 per fold).
"""

import json
from pathlib import Path

KEYS = [
    "weighted_score",
    "f1_promise_status",
    "f1_verification_timeline",
    "f1_evidence_status",
    "f1_evidence_quality",
]

ROOT = Path(__file__).resolve().parents[1]


def agg(rows, seed=42):
    rs = sorted([r for r in rows if r["seed"] == seed], key=lambda r: r["fold"])
    means = {k: sum(r[k] for r in rs) / len(rs) for k in KEYS}
    return rs, means


def main() -> None:
    ap = json.loads((ROOT / "reports/experiments/p2_combo_best_aug_plus/score_summary.json").read_text(encoding="utf-8"))
    bp = json.loads((ROOT / "reports/experiments/p2_combo_best_classw_focal_u6pro/score_summary.json").read_text(encoding="utf-8"))

    ap_rows, ap_m = agg(ap)
    bp_rows, bp_m = agg(bp)

    print("=== Phase 37 (Aug-Plus, seed=42, 5-fold, best-epoch) ===")
    for r in ap_rows:
        print(
            f"  fold {r['fold']}: w={r['weighted_score']:.5f}  "
            f"T1={r['f1_promise_status']:.4f} T2={r['f1_verification_timeline']:.4f} "
            f"T3={r['f1_evidence_status']:.4f} T4={r['f1_evidence_quality']:.4f}  "
            f"best_ep={r['best_epoch']}"
        )
    print(
        f"  MEAN: w={ap_m['weighted_score']:.5f}  "
        f"T1={ap_m['f1_promise_status']:.4f} T2={ap_m['f1_verification_timeline']:.4f} "
        f"T3={ap_m['f1_evidence_status']:.4f} T4={ap_m['f1_evidence_quality']:.4f}"
    )
    print()
    print("=== Phase 36 baseline (classw_focal_u6pro, seed=42, 5-fold, best-epoch) ===")
    for r in bp_rows:
        print(
            f"  fold {r['fold']}: w={r['weighted_score']:.5f}  "
            f"T1={r['f1_promise_status']:.4f} T2={r['f1_verification_timeline']:.4f} "
            f"T3={r['f1_evidence_status']:.4f} T4={r['f1_evidence_quality']:.4f}"
        )
    print(
        f"  MEAN: w={bp_m['weighted_score']:.5f}  "
        f"T1={bp_m['f1_promise_status']:.4f} T2={bp_m['f1_verification_timeline']:.4f} "
        f"T3={bp_m['f1_evidence_status']:.4f} T4={bp_m['f1_evidence_quality']:.4f}"
    )
    print()
    print("=== Δ Phase 37 - Phase 36 (positive = AP improved) ===")
    for k in KEYS:
        print(f"  {k:30s}: {ap_m[k] - bp_m[k]:+.5f}")


if __name__ == "__main__":
    main()
