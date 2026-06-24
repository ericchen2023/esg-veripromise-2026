"""U46 — extra-day binary recombination probes (built on the 0.6218 winner).

The extra contest day adds upload budget. Binary OOF is leaky, so T1/T3 changes
can only be confirmed on the real leaderboard; this script builds principled,
schema-valid probes from cached test probs (no retraining for the instant ones)
and reports No-rates + diff-vs-0.6218 so a probe's risk is visible before upload.

Lever (validated): T1 is the cascade master — a better promise=No boundary lifts
T2/T3/T4 via the N/A rule. Trajectory was monotonic in T1 size (3->5->7 =>
0.6176->0.6195->0.6218); growing T3 past pool7 (pool9) HURT. So new headroom is
on T1, not T3.

Candidates:
  t1div9      INSTANT. T1 = 7 macbert-TAPT seeds + roberta-large + electra-large
              (backbone diversity on the master lever; diversity helped T3, so
              test it on T1). T3 = pool7 (winner). Macro = c3 base.
  macbert9    NEEDS 2 new seeds (s33, s2025). T1 = 9 macbert-TAPT seeds; extends
              the proven monotonic seed-depth trend. Built only once the npz exist.

Both keep the banked c3 macro base for T2/T4 and the winning pool7 T3, varying
only T1 -- so any LB delta is attributable to the T1 change.

    python -m scripts.u46_extraday_probes
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
from scripts.u41_binary_recombine import (
    TV, TV_STEMS, POOL, EXTRA, MACBERT7, POOL7, load_npz, equal8,
)

OUT = Path("outputs/submissions")
BINARY = ("promise_status", "evidence_status")
# new seeds for the macbert9 depth probe (disjoint from the existing 7)
NEW_SEEDS = {
    "s33":   OUT / "p53_tapt_macbert_s33_test_probs.npz",
    "s2025": OUT / "p53_tapt_macbert_s2025_test_probs.npz",
}
LARGE = ["robertalarge", "electralarge"]
BEST = OUT / "u41_macbert7T1_pool7T3_submission.csv"   # banked 0.6218 (md5 d3745d70...)


def main():
    records = load_test_records("vpesg4k_test_2000.csv")
    base = prior_correct_per_task(equal8(), train_priors(),
                                  {"verification_timeline": 0.3, "evidence_quality": 0.3})

    available = {k: p for k, p in {**POOL, **EXTRA, **NEW_SEEDS}.items() if p.exists()}
    src = {k: load_npz(p) for k, p in available.items()}

    def mean_bin(keys):
        return {t: np.mean([src[k][t] for k in keys], axis=0) for t in BINARY}

    # T3 is fixed at the winning pool7 for every probe (isolate the T1 change).
    t3_pool7 = mean_bin(POOL7)["evidence_status"]

    cands = {}
    # Candidate 1 (INSTANT): backbone-diverse T1 = macbert7 + roberta + electra.
    cands["u46_t1div9_pool7T3"] = mean_bin(MACBERT7 + LARGE)["promise_status"]
    # Candidate 2 (needs training): depth T1 = 9 macbert-TAPT seeds.
    macbert9 = MACBERT7 + [k for k in ("s33", "s2025") if k in src]
    if len(macbert9) == 9:
        cands["u46_macbert9T1_pool7T3"] = mean_bin(macbert9)["promise_status"]
    else:
        missing = [s for s in ("s33", "s2025") if s not in src]
        print(f"[skip] macbert9 probe: new seeds not trained yet -> {missing}")

    best = pd.read_csv(BEST, keep_default_na=False).sort_values("id").reset_index(drop=True)
    print(f"{'candidate':30s} {'valid':>5}  diffs_vs_0.6218(T1/T2/T3/T4)        T1_No / T3_No  (best: "
          f"{int((best['promise_status']=='No').sum())} / {int((best['evidence_status']=='No').sum())})")
    for tag, t1 in cands.items():
        mixed = {t: base[t].copy() for t in TASKS}
        mixed["promise_status"] = t1
        mixed["evidence_status"] = t3_pool7
        df = write_submission(probs_to_records(mixed, records), OUT / f"{tag}_submission.csv")
        chk = df.copy()
        chk["verification_timeline"] = chk["verification_timeline"].replace(
            {"more_than_5_years": "longer_than_5_years"})
        rep = validate_submission_frame(chk[SUBMISSION_COLUMNS], mode="preds")
        if not rep.ok:
            raise RuntimeError(f"{tag} invalid: {rep.errors[:3]}")
        diffs = {c: int((df[c].values != best[c].values).sum()) for c in SUBMISSION_COLUMNS[1:]}
        t1no = int((df["promise_status"] == "No").sum())
        t3no = int((df["evidence_status"] == "No").sum())
        print(f"{tag:30s} {str(rep.ok):>5}  {diffs}   {t1no} / {t3no}")


if __name__ == "__main__":
    main()
