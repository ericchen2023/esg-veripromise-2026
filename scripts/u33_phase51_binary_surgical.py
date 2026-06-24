"""U33 — Phase 51: surgical binary-only TAPT probes after LB feedback.

LB feedback on Phase 50:
  c3/equal8 baseline: T1=0.7864, T2=~0.6060, T3=0.6750, T4=~0.4370, weighted=0.6037
  equal9          : T1=0.7864, T2=0.5915, T3=0.6721, T4=0.4249, weighted=0.5963
  tapt_only       : T1=0.7967, T2=0.5117, T3=0.6804, T4=0.4288, weighted=0.5903

Interpretation:
  TAPT *does* improve the binary tasks on the real LB (+0.0103 T1, +0.0054 T3),
  but destroys timeline macro F1 (-0.0943) and slightly hurts quality (-0.0082).
    Therefore the next upload should not be equal9/tapt_only. It should use TAPT
    surgically for T1/T3 only while keeping the c3 macro-a0.3 base for T2/T4.

This script builds legal constrained submissions from cached test probabilities:
    - phase51_tapt_t1only: TAPT promise_status only; all other task probs from c3 base.
    - phase51_tapt_t3only: TAPT evidence_status only; promise/timeline/quality from c3 base.
    - phase51_tapt_binary: TAPT promise_status + evidence_status; macro tasks from c3 base.
    - phase51_tapt_binary_soft50: 50/50 binary blend TAPT+c3 base; macro tasks from c3 base.
    - phase51_tapt_binary_soft75: 75/25 binary blend TAPT+c3 base; macro tasks from c3 base.

All candidates use the same constrained decode and submission writer as c3.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data.dataset import TASKS
from src.tools.validate_submission import validate_submission_frame
from scripts.u18_decoding_experiments import train_priors
from scripts.u17_phase42_test_inference import (
    SUBMISSION_COLUMNS,
    load_test_records,
    probs_to_records,
    write_submission,
)
from scripts.u20_binary_prior_correction import prior_correct_per_task

OUT_DIR = Path("outputs/submissions")
TV_PROBS = OUT_DIR / "phase43_test_probs.npz"
TAPT_PROBS = OUT_DIR / "phase50_tapt_test_probs.npz"
C3 = OUT_DIR / "FROZEN_BEST_0.6037_c3.csv"

TV_STEMS = (
    "p2_combo_best_tv",
    "p2_combo_best_u10_pseudo_tv",
    "p2_combo_best_u10_pseudo_v2_tv",
    "p2_combo_best_u10_pseudo_v2_classw_focal_t4_g3_tv",
    "p2_combo_best_u10_pseudo_v3_classw_focal_t4_g3_tv",
    "p2_combo_best_classw_focal_u6pro_tv",
    "p2_combo_best_aug_plus_tv",
    "p2_combo_best_aug_plus_v2_tv",
)

BINARY_TASKS = {"promise_status", "evidence_status"}


def load_equal8() -> dict[str, np.ndarray]:
    z = np.load(TV_PROBS)
    out: dict[str, np.ndarray] = {}
    for task in TASKS:
        acc = None
        for stem in TV_STEMS:
            arr = z[f"{stem}__{task}"].astype(np.float64)
            acc = arr.copy() if acc is None else acc + arr
        out[task] = acc / len(TV_STEMS)
    return out


def load_tapt() -> dict[str, np.ndarray]:
    z = np.load(TAPT_PROBS)
    return {task: z[task].astype(np.float64) for task in TASKS}


def build(base: dict[str, np.ndarray], tapt: dict[str, np.ndarray], *,
          t1_alpha: float, t3_alpha: float) -> dict[str, np.ndarray]:
    """Blend binary tasks with TAPT alpha, keep macro tasks at the c3 base."""
    mixed = {task: base[task].copy() for task in TASKS}
    mixed["promise_status"] = (1.0 - t1_alpha) * base["promise_status"] + t1_alpha * tapt["promise_status"]
    mixed["evidence_status"] = (1.0 - t3_alpha) * base["evidence_status"] + t3_alpha * tapt["evidence_status"]
    return mixed


def main() -> None:
    records = load_test_records("vpesg4k_test_2000.csv")
    c3 = pd.read_csv(C3, keep_default_na=False).sort_values("id").reset_index(drop=True)
    equal8_plain = load_equal8()
    tapt = load_tapt()
    priors = train_priors()
    c3_base = prior_correct_per_task(
        equal8_plain,
        priors,
        {"verification_timeline": 0.3, "evidence_quality": 0.3},
    )

    rebuilt = write_submission(
        probs_to_records(c3_base, records),
        OUT_DIR / "phase51_rebuilt_c3_a03_submission.csv",
    )
    rebuild_diffs = {
        col: int((rebuilt[col].values != c3[col].values).sum())
        for col in SUBMISSION_COLUMNS[1:]
    }
    print(f"[rebuild c3 macro-a03] diffs vs frozen c3: {rebuild_diffs}")

    candidates = {
        "phase51_tapt_t1only": (1.00, 0.00),
        "phase51_tapt_t3only": (0.00, 1.00),
        "phase51_tapt_binary": (1.00, 1.00),
        "phase51_tapt_binary_soft50": (0.50, 0.50),
        "phase51_tapt_binary_soft75": (0.75, 0.75),
    }

    print("[baseline c3 distributions]")
    print("  T1", c3["promise_status"].value_counts().to_dict())
    print("  T3", c3["evidence_status"].value_counts().to_dict())

    for tag, (t1_alpha, t3_alpha) in candidates.items():
        mixed = build(c3_base, tapt, t1_alpha=t1_alpha, t3_alpha=t3_alpha)
        constrained = probs_to_records(mixed, records)
        df = write_submission(constrained, OUT_DIR / f"{tag}_submission.csv")

        chk = df.copy()
        chk["verification_timeline"] = chk["verification_timeline"].replace(
            {"more_than_5_years": "longer_than_5_years"}
        )
        rep = validate_submission_frame(chk[SUBMISSION_COLUMNS], mode="preds")
        if not rep.ok:
            raise RuntimeError(f"{tag} invalid: {rep.errors[:5]}")

        diffs = {
            col: int((df[col].values != c3[col].values).sum())
            for col in SUBMISSION_COLUMNS[1:]
        }
        identical_rows = int((df[SUBMISSION_COLUMNS[1:]].values == c3[SUBMISSION_COLUMNS[1:]].values).all(axis=1).sum())
        print(f"\n[wrote] {tag}_submission.csv valid={rep.ok} t1_alpha={t1_alpha} t3_alpha={t3_alpha}")
        print(f"  diffs vs c3: {diffs}; identical_rows={identical_rows}/2000")
        print(f"  T1 {df['promise_status'].value_counts().to_dict()}")
        print(f"  T3 {df['evidence_status'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
