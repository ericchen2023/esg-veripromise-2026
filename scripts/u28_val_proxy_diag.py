"""U28 — Phase 48 validation-proxy diagnostic (the foundation for a stronger backbone).

THE PROBLEM we keep hitting: 5-fold CV-OOF over the 2,000-row combined set reads
0.69759 (equal-weight 8-stem), but the leaderboard (test) reads 0.6037. That
-0.094 gap lives almost entirely in the BINARY tasks (T1 OOF 0.94 -> LB 0.786,
T3 OOF 0.86 -> LB 0.675). Phase 45 adversarial validation proved NO input domain
shift (AUC 0.51), and Phase 46/47 proved post-processing is exhausted (3 uploads
0.6037 -> 0.6029 -> 0.6026). So the only remaining lever is a genuinely better
TRAINED model — but we MUST first establish a LOCAL metric that predicts the LB,
or we will overfit the CV-OOF again.

KEY INSIGHT: the 8 TV stems were 5-fold-CV-trained on the COMBINED 2,000 rows
(train id 10001-11000 + val id 11001-12000). Therefore the OOF predictions for
the VAL rows (combined indices 1000-1999) are genuine held-out predictions. The
official val set was released separately/later (2026-06-03) and is the most
"test-like" labelled data we own. If the equal-weight ensemble's VAL-ONLY OOF
score is close to the LB 0.6037 (and shows the same binary craters), then VAL is
a trustworthy LB proxy and we can do model selection on it WITHOUT uploading.

This script slices the existing OOF (no training, no inference) and reports the
equal-weight 8-stem score on: all 2000 / train-only / val-only, per task. It then
contrasts val-only against the banked LB 0.6037 to decide whether val is a usable
proxy for the Phase 48 stronger-backbone experiment.

Usage:
    python -m scripts.u28_val_proxy_diag
"""
from __future__ import annotations

import numpy as np

from scripts.u16_tv_oof_ensemble import TV_STEMS, _mix, _reconstruct_oof, _score
from src.data.dataset import TASKS
from src.data.loader import load_dataset

COMBINED_CSV = "data/processed/train_val_combined.csv"
# combined order: rows 0..999 = train (id 10001-11000), 1000..1999 = val (11001-12000)
VAL_START = 1000

# banked leaderboard per-task (c3 equal-weight + macro corr 0.3 = 0.6037)
LB = {
    "promise_status": 0.7864,
    "verification_timeline": 0.606,
    "evidence_status": 0.675,
    "evidence_quality": 0.437,
    "final_weighted_score": 0.6037,
}


def _score_subset(per_stem, records, idx):
    sub_stem = {s: {t: per_stem[s][t][idx] for t in TASKS} for s in TV_STEMS}
    sub_rec = [records[i] for i in idx]
    equal = {t: tuple([1.0] * len(TV_STEMS)) for t in TASKS}
    return _score(_mix(sub_stem, equal), sub_rec)


def main() -> None:
    records, df = load_dataset(COMBINED_CSV)
    n = len(records)
    assert n == 2000, n
    # sanity: confirm val block is id 11001-12000
    ids = list(df["id"])
    assert ids[VAL_START] == 11001 and ids[-1] == 12000, (ids[VAL_START], ids[-1])

    per_stem = {s: _reconstruct_oof(s, n) for s in TV_STEMS}
    all_idx = np.arange(n)
    tr_idx = np.arange(0, VAL_START)
    va_idx = np.arange(VAL_START, n)

    s_all = _score_subset(per_stem, records, all_idx)
    s_tr = _score_subset(per_stem, records, tr_idx)
    s_va = _score_subset(per_stem, records, va_idx)

    cols = [*TASKS, "final_weighted_score"]
    print("\nequal-weight 8-stem OOF score by data slice")
    print(f"{'task':24s} {'ALL(2000)':>10s} {'train(1000)':>12s} {'val(1000)':>10s} "
          f"{'LB(test)':>9s} {'val-LB':>8s}")
    for t in cols:
        gap = s_va[t] - LB[t]
        print(f"{t:24s} {s_all[t]:10.4f} {s_tr[t]:12.4f} {s_va[t]:10.4f} "
              f"{LB[t]:9.4f} {gap:+8.4f}")

    print("\nINTERPRETATION:")
    vfinal, lbfinal = s_va["final_weighted_score"], LB["final_weighted_score"]
    print(f"  val-OOF final = {vfinal:.4f}  vs  LB = {lbfinal:.4f}  (gap {vfinal-lbfinal:+.4f})")
    if abs(vfinal - lbfinal) <= 0.012:
        print("  => VAL-OOF ~ LB. Val is a TRUSTWORTHY LB proxy. Do model selection on")
        print("     val-only OOF (no uploads needed) for the Phase 48 stronger backbone.")
    elif vfinal - lbfinal > 0.03:
        print("  => VAL-OOF still optimistic like CV. Val is NOT a clean proxy; the gap is")
        print("     OOF leakage (aug/pseudo touching val rows). Must train a NEW backbone on")
        print("     train-only (id 10001-11000) and hold out val (11001-12000) UNSEEN.")
    else:
        print("  => Partial proxy. Use val-only as a directional check, calibrate with care.")


if __name__ == "__main__":
    main()
