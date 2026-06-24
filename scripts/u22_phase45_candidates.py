"""U22 — Phase 45 information-efficient candidate sweep.

The Phase 45 diagnostic (u21) returned two decisive local signals:

  * adversarial validation train-vs-test ROC-AUC = 0.51  -> NO input domain shift
    (the test paragraphs are the same distribution as train);
  * binary P(Yes) confidence is near-identical OOF vs test (mean delta ~ -0.01).

Implication: the binary-task F1 collapse (T1 0.94->0.79, T3 0.86->0.68) is NOT a
prior/threshold shift the model can "see". It is most consistent with a stricter
test labeling convention (more genuine "No") OR mild OOF optimism. Therefore:

  * aggressive toward-No correction (d5 a0.8 / d6 a1.2) is NOT supported locally
    and likely overshoots;
  * only a GENTLE toward-No probe is justified, AND we must first isolate what
    actually produced the +0.0075 of upload-2 vs upload-1.

Open unknowns this sweep is designed to resolve with minimal leaderboard quota:

  U-A  Did macro prior-correction help, or was the gain purely from equal-
       weighting?  -> `p45_c1_equal` (equal weights, NO correction at all) is the
       control. c1 vs the 0.6037 c3 result isolates the macro-correction value.
  U-B  Did macro a0.3 OVERSHOOT (Not Clear 7%->17.7% vs train 11%)? 
       -> `p45_macro_a02` (gentler macro a0.2, no binary corr).
  U-C  Is the T1->T3 constraint cascade the real lever? (promise=Yes forces an
       evidence_status onto rows that should be N/A) 
       -> `p45_t1only_a04` (correct ONLY T1; watch whether T3 improves for free).
  U-D  Direct T3 lever (w=0.30, biggest recoverable). 
       -> `p45_t3only_a04`.
  U-E  Gentle combined toward-No matched to the small confidence shift. 
       -> `p45_gentle_a02`.

All candidates are built on the proven base = equal-weight 8-stem blend, decoded
with the same constrained cascade, validated (mode="preds"), and written as
preds + submission CSVs. Predicted Yes-rates + marginal tables are printed.

Usage:
    python -m scripts.u22_phase45_candidates
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch

from scripts.u17_phase42_test_inference import (
    META_PATH,
    OUT_DIR,
    TV_STEMS,
    load_test_records,
    mix,
    probs_to_records,
    write_submission,
)
from scripts.u18_decoding_experiments import (
    load_cached_probs,
    marginal_table,
    train_priors,
)
from scripts.u20_binary_prior_correction import prior_correct_per_task
from src.data.dataset import TASKS
from src.tools.validate_submission import validate_submission_frame

MACRO = ("verification_timeline", "evidence_quality")
BINARY = ("promise_status", "evidence_status")


def emit(name: str, mixed: dict[str, np.ndarray], records: list[dict],
         priors: dict[str, np.ndarray]) -> pd.DataFrame:
    """Constrained-decode + validate + write preds/submission CSVs for one candidate."""
    constrained = probs_to_records(mixed, records)
    preds_df = (pd.DataFrame(constrained)[["id", *TASKS]]
                .sort_values("id").reset_index(drop=True))
    rep = validate_submission_frame(preds_df, mode="preds")
    if not rep.ok:
        raise RuntimeError(f"[{name}] preds validation failed: {list(rep.errors)[:10]}")

    preds_df.to_csv(OUT_DIR / f"phase45_{name}_preds.csv", index=False, encoding="utf-8")
    sub_df = write_submission(constrained, OUT_DIR / f"phase45_{name}_submission.csv")

    yes_t1 = (sub_df["promise_status"] == "Yes").mean()
    yes_t3 = (sub_df["evidence_status"] == "Yes").mean()
    print(f"\n=== {name} ===  (validated ok={rep.ok}, rows={rep.rows})")
    print(f"  T1 promise Yes-rate  = {yes_t1:6.1%}  (train 81.3%)")
    print(f"  T3 evidence Yes-rate = {yes_t3:6.1%}  (train 67.2%)")
    print(marginal_table(sub_df, priors))
    return sub_df


def main() -> None:
    device = torch.device("cpu")  # cached probs -> no GPU
    records = load_test_records("vpesg4k_test_2000.csv")
    ids = [r["id"] for r in records]
    print(f"[data] {len(records)} test records (id {min(ids)}..{max(ids)})")

    json.loads(META_PATH.read_text(encoding="utf-8"))  # presence check
    equal_w = {t: [1.0] * len(TV_STEMS) for t in TASKS}
    per_stem = load_cached_probs(records, device, use_cache=True)
    priors = train_priors()
    mixed_eq = mix(per_stem, equal_w)

    # U-A control: equal weights, NO correction at all.
    emit("c1_equal",
         prior_correct_per_task(mixed_eq, priors, {}),
         records, priors)

    # U-B: gentler macro (a0.2) only — test whether macro a0.3 overshot.
    emit("macro_a02",
         prior_correct_per_task(mixed_eq, priors, {t: 0.2 for t in MACRO}),
         records, priors)

    # U-C: correct ONLY T1 promise (cascade test) on top of proven macro a0.3.
    emit("t1only_a04",
         prior_correct_per_task(mixed_eq, priors,
                                {**{t: 0.3 for t in MACRO}, "promise_status": 0.4}),
         records, priors)

    # U-D: correct ONLY T3 evidence_status (biggest direct lever, w=0.30).
    emit("t3only_a04",
         prior_correct_per_task(mixed_eq, priors,
                                {**{t: 0.3 for t in MACRO}, "evidence_status": 0.4}),
         records, priors)

    # U-E: gentle combined toward-No, matched to the ~0.01 confidence shift.
    emit("gentle_a02",
         prior_correct_per_task(mixed_eq, priors,
                                {**{t: 0.3 for t in MACRO}, **{t: 0.2 for t in BINARY}}),
         records, priors)

    print("\n[done] phase45 candidates written to", OUT_DIR)


if __name__ == "__main__":
    main()
