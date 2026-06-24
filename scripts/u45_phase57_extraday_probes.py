"""U45 - Phase 57: extra-day LB probes (organizer added one more upload day).

The validated lever is per-task binary recombination + cascade; binary OOF is
leaky, so these can only be confirmed on the real leaderboard. The winning T1
(macbert7T1_pool7T3 = 0.6218) is macbert-only -- it has never included the two
large-TAPT architectures (roberta-large, electra-large). Since architectural
diversity demonstrably helped the T3 pool (dropping electra HURT), the strongest
untested bet is adding that same diversity to the T1 cascade driver.

Two principled probes, both keeping the proven c3 macro base (T2/T4) and the
winning pool7 T3 -- only the T1 source changes:
  macbert9T1   = macbert7 + roberta-large + electra-large  (max diversity)
  macbert8rT1  = macbert7 + roberta-large                  (roberta-only hedge)
Plus the 0.6218 anchor, re-emitted for the end-of-day "last upload wins" rule.

Usage:
    python -m scripts.u45_phase57_extraday_probes
"""
from __future__ import annotations

import hashlib
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
from scripts.u41_binary_recombine import (
    OUT, POOL, EXTRA, MACBERT7, POOL7, load_npz, equal8,
)

BIN = ("promise_status", "evidence_status")
MACBERT9 = MACBERT7 + ["robertalarge", "electralarge"]   # +both large architectures
MACBERT8R = MACBERT7 + ["robertalarge"]                  # +roberta-large only
WINNER = OUT / "u41_macbert7T1_pool7T3_submission.csv"    # 0.6218 (md5 d3745d70...)


def md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def main():
    records = load_test_records("vpesg4k_test_2000.csv")
    base = prior_correct_per_task(equal8(), train_priors(),
                                  {"verification_timeline": 0.3, "evidence_quality": 0.3})
    src = {k: load_npz(p) for k, p in {**POOL, **EXTRA}.items()}

    def mean_bin(keys):
        return {t: np.mean([src[k][t] for k in keys], axis=0) for t in BIN}

    t1 = {
        "macbert7": mean_bin(MACBERT7),     # 0.6218 anchor T1
        "macbert9": mean_bin(MACBERT9),     # probe 1: +roberta +electra
        "macbert8r": mean_bin(MACBERT8R),   # probe 2: +roberta only
    }
    t3_pool7 = mean_bin(POOL7)

    cands = {
        "u41_macbert7T1_pool7T3": "macbert7",    # re-emit 0.6218 anchor (end-of-day lock)
        "u45_macbert9T1_pool7T3": "macbert9",    # PROBE 1 (primary: max T1 diversity)
        "u45_macbert8rT1_pool7T3": "macbert8r",  # PROBE 2 (hedge: roberta-only)
    }

    win = pd.read_csv(WINNER, keep_default_na=False).sort_values("id").reset_index(drop=True)
    print(f"{'candidate':28s} {'valid':>5} {'md5':>10}  diffs_vs_0.6218(T1/T2/T3/T4)  T1No/T3No")
    for tag, t1key in cands.items():
        mixed = {t: base[t].copy() for t in TASKS}
        mixed["promise_status"] = t1[t1key]["promise_status"]
        mixed["evidence_status"] = t3_pool7["evidence_status"]
        out = OUT / f"{tag}_submission.csv"
        df = write_submission(probs_to_records(mixed, records), out)
        chk = df.copy()
        chk["verification_timeline"] = chk["verification_timeline"].replace(
            {"more_than_5_years": "longer_than_5_years"})
        rep = validate_submission_frame(chk[SUBMISSION_COLUMNS], mode="preds")
        if not rep.ok:
            raise RuntimeError(f"{tag} invalid: {rep.errors[:3]}")
        diffs = {c: int((df[c].values != win[c].values).sum()) for c in SUBMISSION_COLUMNS[1:]}
        t1no = int((df["promise_status"] == "No").sum())
        t3no = int((df["evidence_status"] == "No").sum())
        assert len(df) == 2000, f"{tag} row count {len(df)} != 2000"
        print(f"{tag:28s} {str(rep.ok):>5} {md5(out)[:8]:>10}  {diffs}  {t1no}/{t3no}")


if __name__ == "__main__":
    main()
