"""U38 — Phase 54: macro-F1-optimal T4 (and T2) decision rule on trustworthy OOF.

Phase 53 closed the binary axis (T1/T3 saturated). The only high-weight task
whose OOF transfers faithfully to the LB is T4 evidence_quality (w=0.35, LB
0.446; val-slice T4 == LB T4 within 0.003, Phase 48). T4 is macro-F1 over
{Clear, Not Clear, Misleading, N/A}; gold support on the 2000-row CV set is
Clear 1118 / Not Clear 225 / Misleading 2 / N/A 655. With Misleading support=2
its F1 is structurally ~0, so T4 macro is gated by **Not Clear**, which plain
argmax UNDER-predicts (Clear dominates). Macro-F1 is not maximised by argmax:
boosting the minority-class decision raises its recall and thus the macro mean.

This script searches a tiny per-class additive offset on the evidence_quality
probs (Clear fixed at 0; search Not Clear + Misleading) to maximise the official
T4 macro-F1 on the equal-8 OOF, then PROVES it transfers by tuning on the
train-slice (rows 0-999) and scoring the held-out val-slice (1000-1999), and
vice-versa. Only an offset that improves BOTH held-out slices is trustworthy
(guards against the Phase 47 OOF-overfit trap). No test file is written here;
that is u39 once transfer is confirmed.

Usage:
    python -m scripts.u38_macro_decision_opt           # T4
    python -m scripts.u38_macro_decision_opt --task t2 # T2 (verification_timeline)
"""
from __future__ import annotations

import argparse
import numpy as np
import pandas as pd

from src.data.dataset import LABEL2ID
from src.eval.metrics import task_score
from scripts.u17_phase42_test_inference import probs_to_records
from scripts.u16_tv_oof_ensemble import _reconstruct_oof, TV_STEMS

N = 2000
TASKS = ("promise_status", "verification_timeline", "evidence_status", "evidence_quality")
COMBINED = "data/processed/train_val_combined.csv"


def equal8_oof() -> dict[str, np.ndarray]:
    """Mean stored-view OOF probs over the 8 TV stems -> {task: [2000, C]}."""
    acc = {t: np.zeros((N, {"promise_status": 2, "verification_timeline": 5,
                            "evidence_status": 3, "evidence_quality": 4}[t])) for t in TASKS}
    for stem in TV_STEMS:
        o = _reconstruct_oof(stem, N)
        for t in TASKS:
            acc[t] += o[t]
    for t in TASKS:
        acc[t] /= len(TV_STEMS)
    return acc


def gold(col: str) -> np.ndarray:
    df = pd.read_csv(COMBINED, keep_default_na=False)
    return df[col].astype(str).to_numpy()


def decode_task(probs: dict[str, np.ndarray], col: str) -> list[str]:
    recs = [{"id": i} for i in range(N)]
    out = probs_to_records(probs, recs)
    return [r[col] for r in out]


def score_slice(probs, gold_arr, col, idx) -> float:
    pred = np.array(decode_task(probs, col))
    return task_score(col, list(gold_arr[idx]), list(pred[idx]))


def apply_offset(base: dict[str, np.ndarray], col: str, offsets: dict[str, float]) -> dict[str, np.ndarray]:
    """Return a copy of base with additive per-class offsets on `col` probs."""
    out = {t: base[t].copy() for t in TASKS}
    p = out[col].copy()
    for lab, v in offsets.items():
        p[:, LABEL2ID[col][lab]] += v
    out[col] = p
    return out


def search(base, gold_arr, col, classes, grids, idx) -> tuple[dict, float]:
    """Grid-search additive offsets for `classes` to maximise macro-F1 on idx."""
    best_off, best = {c: 0.0 for c in classes}, score_slice(base, gold_arr, col, idx)
    base_score = best
    # coordinate grid over the (1 or 2) minority classes
    import itertools
    keys = list(classes)
    for combo in itertools.product(*[grids[k] for k in keys]):
        off = dict(zip(keys, combo))
        s = score_slice(apply_offset(base, col, off), gold_arr, col, idx)
        if s > best + 1e-9:
            best, best_off = s, off
    return best_off, best, base_score


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["t4", "t2"], default="t4")
    args = ap.parse_args()
    col = "evidence_quality" if args.task == "t4" else "verification_timeline"
    if args.task == "t4":
        classes = ["Not Clear", "Misleading"]
        grids = {"Not Clear": np.round(np.arange(0.0, 0.55, 0.025), 3),
                 "Misleading": np.round(np.arange(0.0, 0.95, 0.05), 3)}
    else:
        classes = ["within_2_years", "longer_than_5_years"]
        grids = {"within_2_years": np.round(np.arange(0.0, 0.55, 0.025), 3),
                 "longer_than_5_years": np.round(np.arange(0.0, 0.35, 0.025), 3)}

    base = equal8_oof()
    g = gold(col)
    full = np.arange(N)
    tr = np.arange(0, 1000)     # original train
    va = np.arange(1000, 2000)  # official val (faithful LB proxy)

    print(f"[task] {col}  | gold dist:", dict(pd.Series(g).value_counts()))
    base_full = score_slice(base, g, col, full)
    base_va = score_slice(base, g, col, va)
    print(f"[baseline] equal8 argmax  full={base_full:.4f}  val-slice={base_va:.4f}")

    # 1) tune on TRAIN slice, evaluate on held-out VAL slice (transfer test)
    off_tr, fit_tr, _ = search(base, g, col, classes, grids, tr)
    va_with_tr = score_slice(apply_offset(base, col, off_tr), g, col, va)
    print(f"\n[transfer A] tune on train-slice -> {off_tr}")
    print(f"    train-slice {score_slice(base,g,col,tr):.4f} -> {fit_tr:.4f}")
    print(f"    HELD-OUT val-slice {base_va:.4f} -> {va_with_tr:.4f}  (delta {va_with_tr-base_va:+.4f})")

    # 2) tune on VAL slice, evaluate on held-out TRAIN slice (symmetric)
    off_va, fit_va, _ = search(base, g, col, classes, grids, va)
    tr_with_va = score_slice(apply_offset(base, col, off_va), g, col, tr)
    base_tr = score_slice(base, g, col, tr)
    print(f"\n[transfer B] tune on val-slice -> {off_va}")
    print(f"    val-slice {base_va:.4f} -> {fit_va:.4f}")
    print(f"    HELD-OUT train-slice {base_tr:.4f} -> {tr_with_va:.4f}  (delta {tr_with_va-base_tr:+.4f})")

    # 3) full-set optimum (only to be applied to test IF transfer holds)
    off_full, fit_full, _ = search(base, g, col, classes, grids, full)
    print(f"\n[full] optimum {off_full}: full {base_full:.4f} -> {fit_full:.4f}  (delta {fit_full-base_full:+.4f})")

    transfers = (va_with_tr > base_va + 1e-4) and (tr_with_va > base_tr + 1e-4)
    print(f"\n[VERDICT] transfers on BOTH held-out slices: {transfers}")
    if transfers:
        print(f"  -> trustworthy. Apply offset {off_full} to test T4 (u39).")
        print(f"  -> expected LB {col} gain ~ held-out mean delta "
              f"{((va_with_tr-base_va)+(tr_with_va-base_tr))/2:+.4f} "
              f"-> weighted ~ {0.35 if args.task=='t4' else 0.15:.2f} x that")
    else:
        print("  -> does NOT transfer; this is OOF-overfit. Do not apply to test.")


if __name__ == "__main__":
    main()
