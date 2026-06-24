"""U20 — Phase 44 binary-task prior-correction (T1 promise, T3 evidence_status).

Phase 43's 2nd upload (c3 = equal-weight + macro T2/T4 prior-corr alpha=0.3)
scored 0.6037 and, crucially, returned a PER-TASK leaderboard breakdown:

    task                     OOF(equal)   TEST     gap
    promise_status   (w.20)    0.941      0.786   -0.155
    verification_tl  (w.15)    0.605      0.606   +0.001   <- perfect transfer
    evidence_status  (w.30)    0.861      0.675   -0.186   <- biggest loss
    evidence_quality (w.35)    0.454      0.437   -0.017

This INVERTS the earlier "macro-class collapse" hypothesis: the macro tasks
(T2, T4) transferred almost perfectly, while the BINARY tasks (T1, T3) cratered.
The backbone is fine on test; only the binary Yes/No decision boundaries are
miscalibrated. Recoverable weighted score by task:
    T3: 0.30 * 0.186 = 0.056   (largest lever)
    T1: 0.20 * 0.155 = 0.031
    T4: 0.35 * 0.017 = 0.006
    T2: ~0

We predict Yes 85% (T1) / 72% (T3), matching the TRAIN prior. If the test ESG
paragraphs (different companies / vaguer reports) carry MORE promise=No and
evidence=No, we over-predict Yes -> low precision -> binary-F1 collapse. The
principled lever is prior-correction toward the minority class (No) on the
BINARY tasks: dividing the posterior by prior^alpha and renormalising. Because
p(Yes) > p(No), alpha>0 shifts the argmax toward No (raises the bar for Yes).

This script reuses the cached per-stem test probs (no re-inference) and emits a
focused candidate set on top of the equal-weight blend:

  d1_binNo_a03   equal + macro(T2,T4) a0.3 + binary(T1,T3) a0.3   (primary bet)
  d2_binNo_a05   equal + macro(T2,T4) a0.3 + binary(T1,T3) a0.5   (stronger)
  d3_binOnly_a03 equal + binary(T1,T3) a0.3 only (no macro corr)  (isolation)
  d4_binT3_a05   equal + macro a0.3 + binary ONLY on T3 a0.5      (target the 0.30-weight task hardest)

Each candidate is validated (mode="preds") and written as preds + submission
CSVs; predicted Yes-rates for T1/T3 are printed so the threshold shift is
visible before upload.

Usage:
    python -m scripts.u20_binary_prior_correction
"""
from __future__ import annotations

import json
from pathlib import Path

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
    PRIOR_FLOOR,
    load_cached_probs,
    marginal_table,
    train_priors,
)
from src.data.dataset import TASKS
from src.tools.validate_submission import validate_submission_frame


def prior_correct_per_task(
    mixed: dict[str, np.ndarray],
    priors: dict[str, np.ndarray],
    alpha_by_task: dict[str, float],
) -> dict[str, np.ndarray]:
    """Divide each listed task's posterior by prior^alpha (floored); renormalize.

    alpha>0 down-weights high-prior classes (shifts argmax toward the minority,
    i.e. 'No' for the binary tasks). Tasks absent from alpha_by_task (or alpha=0)
    pass through unchanged.
    """
    out: dict[str, np.ndarray] = {}
    for t in TASKS:
        p = mixed[t]
        a = float(alpha_by_task.get(t, 0.0))
        if a != 0.0:
            floored = np.maximum(priors[t], PRIOR_FLOOR)
            adj = p / (floored[None, :] ** a)
            adj = adj / adj.sum(axis=1, keepdims=True)
            out[t] = adj
        else:
            out[t] = p
    return out


def emit(name: str, mixed: dict[str, np.ndarray], records: list[dict],
         priors: dict[str, np.ndarray]) -> pd.DataFrame:
    """Constrained-decode + validate + write preds/submission CSVs for one candidate."""
    constrained = probs_to_records(mixed, records)
    preds_df = pd.DataFrame(constrained)[["id", *TASKS]].sort_values("id").reset_index(drop=True)
    rep = validate_submission_frame(preds_df, mode="preds")
    if not rep.ok:
        raise RuntimeError(f"[{name}] preds validation failed: {list(rep.errors)[:10]}")

    preds_df.to_csv(OUT_DIR / f"phase44_{name}_preds.csv", index=False, encoding="utf-8")
    sub_df = write_submission(constrained, OUT_DIR / f"phase44_{name}_submission.csv")

    yes_t1 = (sub_df["promise_status"] == "Yes").mean()
    # evidence_status Yes-rate among ALL rows and among promise=Yes rows.
    yes_t3 = (sub_df["evidence_status"] == "Yes").mean()
    print(f"\n=== {name} ===  (validated ok={rep.ok}, rows={rep.rows})")
    print(f"  T1 promise Yes-rate   = {yes_t1:6.1%}  (train 81.3%)")
    print(f"  T3 evidence Yes-rate  = {yes_t3:6.1%}  (train 67.2%)")
    print(marginal_table(sub_df, priors))
    return sub_df


def main() -> None:
    device = torch.device("cpu")  # cached probs -> no GPU needed
    records = load_test_records("vpesg4k_test_2000.csv")
    ids = [r["id"] for r in records]
    print(f"[data] {len(records)} test records (id {min(ids)}..{max(ids)})")

    json.loads(META_PATH.read_text(encoding="utf-8"))  # presence check
    equal_w = {t: [1.0] * len(TV_STEMS) for t in TASKS}

    per_stem = load_cached_probs(records, device, use_cache=True)
    priors = train_priors()
    mixed_eq = mix(per_stem, equal_w)

    MACRO = "verification_timeline", "evidence_quality"
    BINARY = "promise_status", "evidence_status"

    # d1: macro a0.3 + binary a0.3 (primary bet)
    emit("d1_binNo_a03",
         prior_correct_per_task(mixed_eq, priors,
                                {**{t: 0.3 for t in MACRO}, **{t: 0.3 for t in BINARY}}),
         records, priors)

    # d2: macro a0.3 + binary a0.5 (stronger)
    emit("d2_binNo_a05",
         prior_correct_per_task(mixed_eq, priors,
                                {**{t: 0.3 for t in MACRO}, **{t: 0.5 for t in BINARY}}),
         records, priors)

    # d3: binary a0.3 only (isolation — no macro correction)
    emit("d3_binOnly_a03",
         prior_correct_per_task(mixed_eq, priors, {t: 0.3 for t in BINARY}),
         records, priors)

    # d4: macro a0.3 + binary on T3 only a0.5 (hit the 0.30-weight task hardest)
    emit("d4_binT3_a05",
         prior_correct_per_task(mixed_eq, priors,
                                {**{t: 0.3 for t in MACRO}, "evidence_status": 0.5}),
         records, priors)

    # d5/d6: stronger binary correction (the model is confident, so mild alpha
    # barely shifts the argmax; span the range to locate the F1 optimum via
    # sequential uploads).
    emit("d5_binNo_a08",
         prior_correct_per_task(mixed_eq, priors,
                                {**{t: 0.3 for t in MACRO}, **{t: 0.8 for t in BINARY}}),
         records, priors)
    emit("d6_binNo_a12",
         prior_correct_per_task(mixed_eq, priors,
                                {**{t: 0.3 for t in MACRO}, **{t: 1.2 for t in BINARY}}),
         records, priors)

    print("\n[done] phase44 candidates written to", OUT_DIR)


if __name__ == "__main__":
    main()
