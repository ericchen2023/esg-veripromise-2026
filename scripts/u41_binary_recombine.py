"""U41 — Phase 56: per-task binary recombination LB-probes.

Binary OOF is untrustworthy (leaky), so binary/cascade changes can ONLY be
validated on the real leaderboard. With upload budget to spend, this builds a
principled set of probes from cached test probs (no retraining), keeping the c3
macro base (equal8 + prior-corr 0.3) for T2/T4 and varying only the T1/T3 source.

Measured per-task LB so far:
  binary  0.6113: T1 0.7967 (tapt s42) / T3 0.6804
  pool    0.6120: T1 0.7922 (5-pool)   / T3 0.6829 (best T3) / T4 0.4494 (best)
So best-T1 (tapt s42) + best-T3 (5-pool) is a genuinely untested recombination.

Binary sources (T1=promise_status, T3=evidence_status):
  tapt42      single macbert-TAPT seed42        (best measured T1)
  pool5       mean of the 5 TAPT-pool models     (best measured T3; == pool_binary)
  pool4       pool5 minus the weakest (electra)
  robertalarge / electralarge  individual large-TAPT (untested, distinct No-rates)

Usage:
    python -m scripts.u41_binary_recombine
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from src.data.dataset import TASKS
from src.tools.validate_submission import validate_submission_frame
from scripts.u18_decoding_experiments import train_priors
from scripts.u20_binary_prior_correction import prior_correct_per_task
from scripts.u17_phase42_test_inference import (
    SUBMISSION_COLUMNS, load_test_records, probs_to_records, write_submission,
)

OUT = Path("outputs/submissions")
TV = OUT / "phase43_test_probs.npz"
TV_STEMS = ("p2_combo_best_tv", "p2_combo_best_u10_pseudo_tv", "p2_combo_best_u10_pseudo_v2_tv",
            "p2_combo_best_u10_pseudo_v2_classw_focal_t4_g3_tv",
            "p2_combo_best_u10_pseudo_v3_classw_focal_t4_g3_tv",
            "p2_combo_best_classw_focal_u6pro_tv", "p2_combo_best_aug_plus_tv",
            "p2_combo_best_aug_plus_v2_tv")
POOL = {
    "tapt42": OUT / "phase50_tapt_test_probs.npz",
    "s2024": OUT / "p53_tapt_macbert_s2024_test_probs.npz",
    "s20260417": OUT / "p53_tapt_macbert_s20260417_test_probs.npz",
    "robertalarge": OUT / "p54_tapt_robertalarge_test_probs.npz",
    "electralarge": OUT / "p55_tapt_electralarge_test_probs.npz",
}
# extra macbert-TAPT seeds for the enlarged T1 ensemble (Phase 56 cascade lever)
EXTRA = {
    "s7": OUT / "p53_tapt_macbert_s7_test_probs.npz",
    "s777": OUT / "p53_tapt_macbert_s777_test_probs.npz",
    "s11": OUT / "p53_tapt_macbert_s11_test_probs.npz",
    "s22": OUT / "p53_tapt_macbert_s22_test_probs.npz",
}
MACBERT3 = ["tapt42", "s2024", "s20260417"]
MACBERT5 = ["tapt42", "s2024", "s20260417", "s7", "s777"]
MACBERT7 = ["tapt42", "s2024", "s20260417", "s7", "s777", "s11", "s22"]
POOL7 = list(POOL) + ["s7", "s777"]              # winning 7-model T3 (= 0.6210)
POOL9 = list(POOL) + ["s7", "s777", "s11", "s22"]  # 9-model T3 (more diversity)


def load_npz(p):
    z = np.load(p)
    return {t: z[t].astype(np.float64) for t in TASKS}


def equal8():
    z = np.load(TV)
    out = {}
    for t in TASKS:
        out[t] = np.mean([z[f"{s}__{t}"].astype(np.float64) for s in TV_STEMS], axis=0)
    return out


def main():
    records = load_test_records("vpesg4k_test_2000.csv")
    base = prior_correct_per_task(equal8(), train_priors(),
                                  {"verification_timeline": 0.3, "evidence_quality": 0.3})
    src = {k: load_npz(p) for k, p in {**POOL, **EXTRA}.items()}
    # binary sources: per-task promise/evidence probs
    def mean_bin(keys):
        return {t: np.mean([src[k][t] for k in keys], axis=0) for t in ("promise_status", "evidence_status")}
    binsrc = {
        "tapt42": {t: src["tapt42"][t] for t in ("promise_status", "evidence_status")},
        "pool5": mean_bin(list(POOL)),
        "pool4": mean_bin([k for k in POOL if k != "electralarge"]),
        "robertalarge": {t: src["robertalarge"][t] for t in ("promise_status", "evidence_status")},
        "electralarge": {t: src["electralarge"][t] for t in ("promise_status", "evidence_status")},
    }
    # tapt42 T1 WON (0.6161): best T1 + cascade compounding. Keep T1=tapt42 fixed
    # and search the T3 source (the lever that, via cascade, also moves T3/T4).
    binsrc["pool_macbert5"] = mean_bin(MACBERT5)  # 5-seed T1
    binsrc["pool_macbert7"] = mean_bin(MACBERT7)  # 7-seed T1 (enlarged)
    binsrc["pool7T3"] = mean_bin(POOL7)  # winning 7-model T3 (= 0.6210)
    binsrc["pool9T3"] = mean_bin(POOL9)  # 9-model T3 (more diversity; electra-drop HURT so keep all)
    # WINNER = (macbert5 T1, pool7 T3) = 0.6210. Both ensembles benefit from MORE models
    # (electra-drop hurt -> diversity helps). Enlarge both T1 (5->7 seeds) and T3 (7->9).
    cands = {
        "u41_macbert5T1_pool7T3": ("pool_macbert5", "pool7T3"),  # 0.6210 current best (lock anchor)
        "u41_macbert7T1_pool7T3": ("pool_macbert7", "pool7T3"),  # 7-seed T1 + winning T3 (top probe)
        "u41_macbert7T1_pool9T3": ("pool_macbert7", "pool9T3"),  # both ensembles maxed
        "u41_macbert5T1_pool9T3": ("pool_macbert5", "pool9T3"),  # 5-seed T1 + bigger T3
    }
    # compare diffs against the current LB best (macbert5T1_pool5T3 = 0.6195)
    ref_path = OUT / "u41_macbert5T1_pool5T3_submission.csv"
    pool_ref = pd.read_csv(ref_path, keep_default_na=False).sort_values("id").reset_index(drop=True)
    print(f"{'candidate':30s} {'valid':>5}  diffs_vs_0.6195(T1/T2/T3/T4)   T1 No / T3 No")
    for tag, (t1s, t3s) in cands.items():
        mixed = {t: base[t].copy() for t in TASKS}
        mixed["promise_status"] = binsrc[t1s]["promise_status"]
        mixed["evidence_status"] = binsrc[t3s]["evidence_status"]
        df = write_submission(probs_to_records(mixed, records), OUT / f"{tag}_submission.csv")
        chk = df.copy(); chk["verification_timeline"] = chk["verification_timeline"].replace({"more_than_5_years": "longer_than_5_years"})
        rep = validate_submission_frame(chk[SUBMISSION_COLUMNS], mode="preds")
        if not rep.ok:
            raise RuntimeError(f"{tag} invalid: {rep.errors[:3]}")
        diffs = {c: int((df[c].values != pool_ref[c].values).sum()) for c in SUBMISSION_COLUMNS[1:]}
        t1no = int((df["promise_status"] == "No").sum()); t3no = int((df["evidence_status"] == "No").sum())
        print(f"{tag:26s} {str(rep.ok):>5}  {diffs}   {t1no} / {t3no}")


if __name__ == "__main__":
    main()
