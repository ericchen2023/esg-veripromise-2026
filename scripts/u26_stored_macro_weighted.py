"""U26 — Phase 46 stored-view macro-weighted candidates (lever A isolation).

u23 found that re-weighting ONLY the macro-task (T2/T4) stem mixes — while
freezing the binary tasks at equal weight — lifts the OOF by +0.0045
(0.69759 -> 0.70208), entirely via T2 (0.6053 -> 0.6286). Phase 45 proved the
macro tasks transfer faithfully to test (no domain shift), so this OOF gain is
expected to carry to the leaderboard.

This script applies the learned macro-weights to the STORED-view cached test
probs (phase43_test_probs.npz) — NO TTA — so the leaderboard delta vs the banked
c3=0.6037 (equal + macro corr 0.3) isolates the macro-weighting lever alone,
separate from the multi-window TTA lever (u24/u25). Candidates:

  mw_a02   macro-weighted stems + macro prior-corr 0.2
  mw_a03   macro-weighted stems + macro prior-corr 0.3   (matches c3's corr)
  mw_none  macro-weighted stems, NO prior-corr           (pure weighting effect)

Usage:
    python -m scripts.u26_stored_macro_weighted
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scripts.u17_phase42_test_inference import (
    OUT_DIR,
    TV_STEMS,
    load_test_records,
    mix,
    probs_to_records,
    write_submission,
)
from scripts.u18_decoding_experiments import load_cached_probs, marginal_table, train_priors
from scripts.u20_binary_prior_correction import prior_correct_per_task
from src.data.dataset import TASKS
from src.tools.validate_submission import validate_submission_frame

MACRO_META = Path("reports/analysis/_ensemble/macro_weighting_meta.json")
MACRO = ("verification_timeline", "evidence_quality")


def emit(name: str, mixed: dict[str, np.ndarray], records: list[dict],
         priors: dict[str, np.ndarray]) -> pd.DataFrame:
    constrained = probs_to_records(mixed, records)
    preds_df = (pd.DataFrame(constrained)[["id", *TASKS]]
                .sort_values("id").reset_index(drop=True))
    rep = validate_submission_frame(preds_df, mode="preds")
    if not rep.ok:
        raise RuntimeError(f"[{name}] validation failed: {list(rep.errors)[:10]}")
    preds_df.to_csv(OUT_DIR / f"phase46_{name}_preds.csv", index=False, encoding="utf-8")
    sub_df = write_submission(constrained, OUT_DIR / f"phase46_{name}_submission.csv")
    print(f"\n=== {name} ===  (ok={rep.ok}, rows={rep.rows})")
    print(marginal_table(sub_df, priors))
    return sub_df


def main() -> None:
    if not MACRO_META.exists():
        raise SystemExit("run `python -m scripts.u23_macro_weighting` first")
    meta = json.loads(MACRO_META.read_text(encoding="utf-8"))
    sw = meta["stem_weights_per_task"]
    print(f"[meta] macro-weighting OOF gain {meta['oof_gain']:+.6f} "
          f"(equal {meta['equal_score']:.5f} -> {meta['macro_opt_score']:.5f})")

    device = torch.device("cpu")
    records = load_test_records("vpesg4k_test_2000.csv")
    per_stem = load_cached_probs(records, device, use_cache=True)  # stored view
    priors = train_priors()

    mixed_w = mix(per_stem, {t: sw[t] for t in TASKS})  # macro-weighted stems

    emit("mw_none", prior_correct_per_task(mixed_w, priors, {}), records, priors)
    emit("mw_a02", prior_correct_per_task(mixed_w, priors, {t: 0.2 for t in MACRO}),
         records, priors)
    emit("mw_a03", prior_correct_per_task(mixed_w, priors, {t: 0.3 for t in MACRO}),
         records, priors)

    print("\n[done] stored macro-weighted candidates written to", OUT_DIR)


if __name__ == "__main__":
    main()
