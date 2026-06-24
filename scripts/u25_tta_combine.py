"""U25 — Phase 46 TTA combiner: blend stored+middle+tail and emit candidates.

Consumes the cached per-view per-stem test probs:
    outputs/submissions/phase43_test_probs.npz        (stored / head view, Phase 43)
    outputs/submissions/phase46_tta_middle_probs.npz  (Phase 46, u24)
    outputs/submissions/phase46_tta_tail_probs.npz     (Phase 46, u24)

For each stem we average the available views (equal weight — the TV stems have
no middle/tail OOF, so a view-blend weight cannot be tuned offline; equal is the
standard robust TTA choice and is what lifted the saturated AP-D4 ensemble in
the project's U1 TTA experiment). The view-averaged stems are then ensembled and
decoded exactly like the production path.

Candidates (all built on the equal-weight 8-stem blend of the TTA-averaged
stems, constrained-decoded, validated mode="preds"):

  tta_plain        TTA blend, NO prior-correction          (isolate raw TTA lift)
  tta_macro_a02    TTA blend + macro(T2,T4) prior-corr 0.2 (gentle, matches Ph45)
  tta_macro_a03    TTA blend + macro(T2,T4) prior-corr 0.3 (the c3=0.6037 setting)
  tta_macrow_a02   TTA blend + macro-OOF stem weights (u23) + macro corr 0.2
                    (only emitted if reports/.../macro_weighting_meta.json exists)

Usage:
    python -m scripts.u25_tta_combine
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
from scripts.u18_decoding_experiments import marginal_table, train_priors
from scripts.u20_binary_prior_correction import prior_correct_per_task
from src.data.dataset import TASKS
from src.tools.validate_submission import validate_submission_frame

STORED_NPZ = OUT_DIR / "phase43_test_probs.npz"
MIDDLE_NPZ = OUT_DIR / "phase46_tta_middle_probs.npz"
TAIL_NPZ = OUT_DIR / "phase46_tta_tail_probs.npz"
MACRO_META = Path("reports/analysis/_ensemble/macro_weighting_meta.json")
MACRO = ("verification_timeline", "evidence_quality")


def _load_view(path: Path) -> dict[str, dict[str, np.ndarray]]:
    data = np.load(path)
    return {stem: {t: data[f"{stem}__{t}"] for t in TASKS} for stem in TV_STEMS}


def _blend_views(views: list[dict[str, dict[str, np.ndarray]]]
                 ) -> dict[str, dict[str, np.ndarray]]:
    """Equal-average the available views per stem/task."""
    out: dict[str, dict[str, np.ndarray]] = {}
    for stem in TV_STEMS:
        out[stem] = {}
        for t in TASKS:
            stacked = np.stack([v[stem][t] for v in views], axis=0)
            out[stem][t] = stacked.mean(axis=0)
    return out


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
    yes_t1 = (sub_df["promise_status"] == "Yes").mean()
    yes_t3 = (sub_df["evidence_status"] == "Yes").mean()
    print(f"\n=== {name} ===  (ok={rep.ok}, rows={rep.rows})")
    print(f"  T1 Yes={yes_t1:6.1%}  T3 Yes={yes_t3:6.1%}")
    print(marginal_table(sub_df, priors))
    return sub_df


def main() -> None:
    records = load_test_records("vpesg4k_test_2000.csv")
    print(f"[data] {len(records)} test records")

    views = [_load_view(STORED_NPZ)]
    tags = ["stored"]
    for path, tag in ((MIDDLE_NPZ, "middle"), (TAIL_NPZ, "tail")):
        if path.exists():
            views.append(_load_view(path))
            tags.append(tag)
        else:
            print(f"[warn] missing {path.name}; run u24_tta_inference first")
    print(f"[views] blending: {tags}")
    if len(views) == 1:
        raise SystemExit("No TTA views found — run `python -m scripts.u24_tta_inference` first.")

    per_stem_tta = _blend_views(views)
    priors = train_priors()
    equal_w = {t: [1.0] * len(TV_STEMS) for t in TASKS}
    mixed_eq = mix(per_stem_tta, equal_w)

    # raw TTA, no correction
    emit("tta_plain", prior_correct_per_task(mixed_eq, priors, {}), records, priors)
    # TTA + gentle macro corr
    emit("tta_macro_a02",
         prior_correct_per_task(mixed_eq, priors, {t: 0.2 for t in MACRO}), records, priors)
    # TTA + macro corr 0.3 (the proven c3 setting, now on TTA blend)
    emit("tta_macro_a03",
         prior_correct_per_task(mixed_eq, priors, {t: 0.3 for t in MACRO}), records, priors)

    # TTA + macro-OOF stem weights (u23) + gentle macro corr
    if MACRO_META.exists():
        meta = json.loads(MACRO_META.read_text(encoding="utf-8"))
        sw = meta["stem_weights_per_task"]
        gain = meta.get("oof_gain", 0.0)
        print(f"\n[macro-weighting] OOF gain {gain:+.6f}; applying learned T2/T4 weights")
        mixed_w = mix(per_stem_tta, {t: sw[t] for t in TASKS})
        emit("tta_macrow_a02",
             prior_correct_per_task(mixed_w, priors, {t: 0.2 for t in MACRO}), records, priors)
    else:
        print(f"[info] {MACRO_META.name} not found; skipping macro-weighted TTA candidate")

    print("\n[done] phase46 TTA candidates written to", OUT_DIR)


if __name__ == "__main__":
    main()
