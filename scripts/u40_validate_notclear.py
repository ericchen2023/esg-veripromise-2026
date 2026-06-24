"""U40 — Phase 55: validate the Not Clear augmentation on the faithful val-slice.

T4 macro OOF transfers to the LB (val-slice T4 == LB T4 within 0.003, Phase 48),
and the Phase 55 synth was injected leak-free (per-fold via _source_id), so the
val-slice OOF (rows 1000-1999) is an honest held-out test of whether the
augmentation actually improves the T4 head.

Compares, on the val-slice, the T4 macro-F1 and the Not Clear class F1 for the
SAME recipe with vs without the Not Clear synth, plus what happens when the new
stem's T4 is added to / swapped into the equal-8 pool. To isolate the T4-head
effect, every variant is decoded under the SAME (equal-8) binary cascade so only
the evidence_quality probs differ.

Run AFTER `python -m src.train_kfold --config configs/exp_p56_combo_best_notclear.yaml`.

Usage:
    python -m scripts.u40_validate_notclear
"""
from __future__ import annotations

import glob
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from scripts.u16_tv_oof_ensemble import _reconstruct_oof, TV_STEMS
from scripts.u17_phase42_test_inference import probs_to_records

N = 2000
TASKS = ("promise_status", "verification_timeline", "evidence_status", "evidence_quality")
LABS = ["Clear", "Not Clear", "Misleading", "N/A"]
VA = np.arange(1000, 2000)
FULL = np.arange(N)


def reconstruct_t4(stem: str) -> np.ndarray | None:
    out = np.zeros((N, 4)); seen = np.zeros(N, bool)
    for f in glob.glob(f"outputs/checkpoints/{stem}/**/oof_probs.npz", recursive=True):
        if "stage_a" in f:
            continue
        z = np.load(f); idx = z["indices"].astype(int)
        out[idx] = z["probs_evidence_quality"].astype(float); seen[idx] = True
    return out if seen.all() else None


def equal8() -> dict[str, np.ndarray]:
    acc = {t: np.zeros((N, {"promise_status": 2, "verification_timeline": 5,
                            "evidence_status": 3, "evidence_quality": 4}[t])) for t in TASKS}
    for s in TV_STEMS:
        o = _reconstruct_oof(s, N)
        for t in TASKS:
            acc[t] += o[t]
    for t in TASKS:
        acc[t] /= len(TV_STEMS)
    return acc


def t4_scores(base: dict, t4probs: np.ndarray, idx) -> tuple[float, float]:
    """Decode with equal8 cascade but supplied T4 probs; return (macro, NotClear F1)."""
    p = {k: base[k].copy() for k in TASKS}
    p["evidence_quality"] = t4probs
    pred = np.array([r["evidence_quality"] for r in probs_to_records(p, [{"id": i} for i in range(N)])])
    g = pd.read_csv("data/processed/train_val_combined.csv", keep_default_na=False)["evidence_quality"].astype(str).to_numpy()
    macro = f1_score(g[idx], pred[idx], labels=LABS, average="macro", zero_division=0)
    nc = f1_score(g[idx] == "Not Clear", pred[idx] == "Not Clear", zero_division=0)
    return macro, nc


def main() -> None:
    base = equal8()
    base_t4 = base["evidence_quality"]
    combo = reconstruct_t4("p2_combo_best_tv")          # baseline single stem (no synth)
    p56 = reconstruct_t4("p56_combo_best_notclear")     # same recipe + Not Clear synth
    if p56 is None:
        raise SystemExit("p56 OOF not found — train exp_p56_combo_best_notclear first")

    print(f"{'variant':40s} {'val macro':>10} {'val NotClr':>11} {'full macro':>11}")
    print("-" * 76)
    rows = [
        ("equal8 (current T4 source)", base_t4),
        ("baseline combo_best single stem", combo),
        ("p56 combo_best + NotClear synth", p56),
        ("equal8 + p56 (mean)", (base_t4 * len(TV_STEMS) + p56) / (len(TV_STEMS) + 1)),
        ("equal8 + p56 (T4 weight 2x p56)", (base_t4 * len(TV_STEMS) + 2 * p56) / (len(TV_STEMS) + 2)),
    ]
    for name, t4 in rows:
        if t4 is None:
            print(f"{name:40s}   (missing)")
            continue
        vm, vnc = t4_scores(base, t4, VA)
        fm, _ = t4_scores(base, t4, FULL)
        print(f"{name:40s} {vm:10.4f} {vnc:11.4f} {fm:11.4f}")

    # decisive single-recipe comparison
    bm, bnc = t4_scores(base, combo, VA)
    pm, pnc = t4_scores(base, p56, VA)
    print(f"\n[single-recipe delta on val-slice]  T4 macro {pm-bm:+.4f}   Not Clear F1 {pnc-bnc:+.4f}")
    print("VERDICT:", "synth HELPS (build candidate)" if (pnc > bnc + 1e-3 and pm >= bm - 1e-4)
          else "synth does NOT help on the faithful val-slice (do not upload)")


if __name__ == "__main__":
    main()
