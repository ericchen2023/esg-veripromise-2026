"""U18 — Phase 43 decoding / weighting experiments on the test set.

Phase 42's primary submission (per-task OOF hillclimb weights + constrained
argmax) scored **0.5961966** on the real leaderboard versus an OOF proxy of
0.71033 — a ~0.114 collapse. Diagnostics ruled out an alignment/marginal bug:
the predicted marginals match the TRAIN distribution almost perfectly. The gap
is driven by

  1. OOF hillclimb OVERFIT (32 weights x 12000 iters tuned to the exact OOF set);
  2. majority-class collapse on the macro-F1 tasks (T2 verification_timeline,
     T4 evidence_quality): the model defaults to `already` / `Clear`, so on a
     more-diverse test set the rare classes get F1=0 and macro craters;
  3. domain shift (different companies/report styles in test).

Rather than burn submission quota re-running the expensive 40-checkpoint
inference for every idea, this script runs inference ONCE, caches the per-stem
softmax probs to ``outputs/submissions/phase43_test_probs.npz``, and then cheaply
generates several candidate submissions that test concrete hypotheses:

  C1  equal-weight blend            -> tests the "hillclimb overfit" hypothesis
                                       (a flatter blend should generalise better).
  C2  prior-correction (T2/T4)      -> divides macro-task posteriors by the train
                                       prior^alpha (floored) to push minority
                                       classes, fighting majority collapse.
                                       alpha in {0.3, 0.5}.
  C3  equal-weight + prior-correct   -> combines C1 + C2.

Binary tasks (T1 promise, T3 evidence_status) are left on plain argmax — they
are not macro-scored and over-correcting their threshold without a validation
signal is risky.

Every candidate is validated (mode="preds") and written both as a canonical
preds CSV and an official submission CSV (`more_than_5_years` remap, literal
"N/A", 5-column order). A marginal-distribution table is printed so each
candidate can be sanity-checked before upload.

Usage:
    python -m scripts.u18_decoding_experiments
    python -m scripts.u18_decoding_experiments --no-cache   # force re-inference
"""
from __future__ import annotations

import argparse
import collections
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scripts.u17_phase42_test_inference import (
    META_PATH,
    OUT_DIR,
    TV_STEMS,
    infer_all_stems,
    load_test_records,
    mix,
    probs_to_records,
    write_submission,
)
from src.data.dataset import ID2LABEL, LABEL2ID, TASKS
from src.tools.validate_submission import validate_submission_frame

import json

CACHE_PATH = OUT_DIR / "phase43_test_probs.npz"
TRAIN_CSV = "data/processed/train_val_combined.csv"

# Macro-F1 tasks where majority-class collapse hurts most.
MACRO_TASKS = ("verification_timeline", "evidence_quality")
PRIOR_FLOOR = 0.05  # clamp tiny priors (e.g. Misleading=0.1%) to avoid 1000x boosts


def load_cached_probs(records: list[dict], device: torch.device, use_cache: bool
                      ) -> dict[str, dict[str, np.ndarray]]:
    """Run 8-stem x 5-fold inference once; cache to npz keyed ``stem__task``."""
    if use_cache and CACHE_PATH.exists():
        data = np.load(CACHE_PATH)
        per_stem: dict[str, dict[str, np.ndarray]] = {}
        for stem in TV_STEMS:
            per_stem[stem] = {t: data[f"{stem}__{t}"] for t in TASKS}
        print(f"[cache] loaded per-stem probs from {CACHE_PATH}")
        return per_stem

    print(f"[infer] running 8 stems x 5 folds on {len(records)} test rows")
    per_stem = infer_all_stems(records, device)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    flat = {f"{stem}__{t}": per_stem[stem][t] for stem in TV_STEMS for t in TASKS}
    np.savez_compressed(CACHE_PATH, **flat)
    print(f"[cache] saved per-stem probs -> {CACHE_PATH}")
    return per_stem


def train_priors() -> dict[str, np.ndarray]:
    """Per-task class-prior vectors (LABEL_DOMAIN order) from the train+val set."""
    from src.data.loader import load_dataset

    recs, _ = load_dataset(TRAIN_CSV)
    priors: dict[str, np.ndarray] = {}
    for t in TASKS:
        counts = collections.Counter(str(r.get(t)) for r in recs)
        total = sum(counts.values())
        vec = np.array([counts.get(ID2LABEL[t][i], 0) / total
                        for i in range(len(ID2LABEL[t]))], dtype=np.float64)
        priors[t] = vec
    return priors


def prior_correct(mixed: dict[str, np.ndarray], priors: dict[str, np.ndarray],
                  alpha: float, tasks: tuple[str, ...]) -> dict[str, np.ndarray]:
    """Divide posteriors by prior^alpha (floored) on the given tasks; renormalize.

    alpha=0 -> unchanged; alpha=1 -> full prior-nulling (target = uniform).
    Only the listed (macro) tasks are adjusted; others pass through unchanged.
    """
    out: dict[str, np.ndarray] = {}
    for t in TASKS:
        p = mixed[t]
        if t in tasks and alpha > 0.0:
            floored = np.maximum(priors[t], PRIOR_FLOOR)
            adj = p / (floored[None, :] ** alpha)
            adj = adj / adj.sum(axis=1, keepdims=True)
            out[t] = adj
        else:
            out[t] = p
    return out


def marginal_table(df: pd.DataFrame, priors: dict[str, np.ndarray]) -> str:
    """Render a predicted-vs-train marginal comparison (canonical labels)."""
    canon = df.copy()
    canon["verification_timeline"] = canon["verification_timeline"].replace(
        {"more_than_5_years": "longer_than_5_years"})
    lines = []
    for t in TASKS:
        vc = collections.Counter(canon[t])
        tot = sum(vc.values())
        lines.append(f"  [{t}]")
        for i, lab in ID2LABEL[t].items():
            lines.append(f"    {lab:24s} train={priors[t][i]:6.1%}  pred={vc.get(lab, 0) / tot:6.1%}")
    return "\n".join(lines)


def emit_candidate(name: str, mixed: dict[str, np.ndarray], records: list[dict],
                   priors: dict[str, np.ndarray]) -> None:
    """Constrained-decode, validate, and write preds + submission CSVs for one candidate."""
    constrained = probs_to_records(mixed, records)
    preds_df = pd.DataFrame(constrained)[["id", *TASKS]].sort_values("id").reset_index(drop=True)
    rep = validate_submission_frame(preds_df, mode="preds")
    if not rep.ok:
        raise RuntimeError(f"[{name}] preds validation failed: {list(rep.errors)[:10]}")

    preds_path = OUT_DIR / f"phase43_{name}_preds.csv"
    preds_df.to_csv(preds_path, index=False, encoding="utf-8")
    sub_path = OUT_DIR / f"phase43_{name}_submission.csv"
    sub_df = write_submission(constrained, sub_path)

    print(f"\n=== candidate: {name} ===")
    print(f"  validated ok={rep.ok} rows={rep.rows}")
    print(f"  wrote {sub_path}")
    print(marginal_table(sub_df, priors))


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 43 decoding/weighting experiments")
    ap.add_argument("--test-csv", default="vpesg4k_test_2000.csv")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no-cache", action="store_true", help="force re-inference (ignore npz cache)")
    args = ap.parse_args()

    device = torch.device(args.device)
    print(f"[device] {device}")

    records = load_test_records(args.test_csv)
    ids = [r["id"] for r in records]
    print(f"[data] loaded {len(records)} test records (id {min(ids)}..{max(ids)})")

    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    stem_w = meta["stem_weights_per_task"]
    equal_w = {t: [1.0] * len(TV_STEMS) for t in TASKS}

    per_stem = load_cached_probs(records, device, use_cache=not args.no_cache)
    priors = train_priors()

    mixed_hc = mix(per_stem, stem_w)        # hillclimb-weighted (Phase 42 primary)
    mixed_eq = mix(per_stem, equal_w)       # equal-weight blend

    # C1: equal-weight, plain constrained decode.
    emit_candidate("c1_equalweight", mixed_eq, records, priors)

    # C2: hillclimb weights + prior-correction on macro tasks.
    for alpha in (0.3, 0.5):
        tag = f"c2_priorcorr_a{str(alpha).replace('.', '')}"
        emit_candidate(tag, prior_correct(mixed_hc, priors, alpha, MACRO_TASKS), records, priors)

    # C3: equal-weight + prior-correction on macro tasks.
    for alpha in (0.3, 0.5):
        tag = f"c3_equal_priorcorr_a{str(alpha).replace('.', '')}"
        emit_candidate(tag, prior_correct(mixed_eq, priors, alpha, MACRO_TASKS), records, priors)

    print("\n[done] phase43 candidates written to", OUT_DIR)


if __name__ == "__main__":
    main()
