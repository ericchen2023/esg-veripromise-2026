"""U23 — Phase 46 macro-task-only OOF weighting (T2, T4) with frozen binary.

Phase 45's adversarial validation proved there is NO input domain shift
(train-vs-test ROC-AUC 0.51) and that the MACRO tasks (T2 verification_timeline,
T4 evidence_quality) transfer faithfully (OOF->test gap +0.001 / -0.017), while
the BINARY tasks (T1, T3) suffer a non-recoverable F1 drop most consistent with
OOF optimism / stricter test labels.

Consequence for weighting: because the macro tasks transfer, stem-mix weights
that are OPTIMAL on the OOF set for T2/T4 should also transfer to test. The
Phase 42 joint hillclimb (0.71044 OOF) failed on the leaderboard (0.5962) only
because it ALSO moved the BINARY weights, which overfit the OOF. So here we run
a post-constraint joint hillclimb that perturbs ONLY the T2 and T4 stem-weight
vectors and FREEZES T1/T3 at the robust equal weight.

  * objective = post-constraint weighted_score (respects the T1->T3->T4 cascade,
    so the optimizer sees the true coupled effect of moving macro weights);
  * search space = T2 and T4 simplex weights only (8 dims each);
  * warm-start = equal weights on every task.

We report the OOF gain (must clear equal-weight to be worth uploading) and the
learned T2/T4 weights, which u24 will apply to the cached test probs to build a
combined candidate (macro-weighting + gentle macro prior-correction).

Usage:
    python -m scripts.u23_macro_weighting --iters 12000 --refine-iters 6000
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

from scripts.u16_tv_oof_ensemble import (
    COMBINED_CSV,
    TV_STEMS,
    _mix,
    _move_simplex_mass,
    _reconstruct_oof,
    _score,
)
from src.data.dataset import TASKS
from src.data.loader import load_dataset

OUT_META = Path("reports/analysis/_ensemble/macro_weighting_meta.json")
MACRO_TASKS = ("verification_timeline", "evidence_quality")


def macro_only_hillclimb(
    per_stem: dict[str, dict[str, np.ndarray]],
    records: list[dict],
    *,
    init: dict[str, tuple[float, ...]],
    n_iters: int,
    step: float,
    seed: int,
) -> tuple[dict[str, tuple[float, ...]], dict[str, float], list[dict]]:
    """Joint post-constraint hillclimb that perturbs ONLY MACRO_TASKS weights."""
    rng = random.Random(seed)
    best = {t: tuple(init[t]) for t in TASKS}
    best_score = _score(_mix(per_stem, best), records)
    history: list[dict] = []
    for it in range(1, n_iters + 1):
        cand = dict(best)
        t = rng.choice(MACRO_TASKS)  # only macro dims move
        cand[t] = _move_simplex_mass(best[t], step=step, rng=rng)
        if cand[t] == best[t]:
            continue
        cand_score = _score(_mix(per_stem, cand), records)
        if cand_score["final_weighted_score"] > best_score["final_weighted_score"] + 1e-12:
            best = cand
            best_score = cand_score
            history.append({"iter": it, "task": t,
                            "score": best_score["final_weighted_score"]})
            print(f"[macro {it:5d}] {t:24s} -> {best_score['final_weighted_score']:.6f}")
    return best, best_score, history


def main() -> None:
    ap = argparse.ArgumentParser(description="Macro-only OOF weighting (T2/T4)")
    ap.add_argument("--iters", type=int, default=12000)
    ap.add_argument("--step", type=float, default=0.1)
    ap.add_argument("--refine-iters", type=int, default=6000)
    ap.add_argument("--refine-step", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    records, _ = load_dataset(COMBINED_CSV)
    n = len(records)
    print(f"[data] {n} OOF records")

    per_stem = {stem: _reconstruct_oof(stem, n) for stem in TV_STEMS}
    print(f"[oof] reconstructed {len(TV_STEMS)} stems")

    equal = {t: tuple([1.0] * len(TV_STEMS)) for t in TASKS}
    equal_score = _score(_mix(per_stem, equal), records)
    print(f"[baseline] equal-weight = {equal_score['final_weighted_score']:.6f}")
    for t in TASKS:
        print(f"    {t:24s} = {equal_score[t]:.4f}")

    print(f"\n[stage1] macro-only step={args.step} iters={args.iters}")
    w1, s1, h1 = macro_only_hillclimb(per_stem, records, init=equal,
                                      n_iters=args.iters, step=args.step, seed=args.seed)
    print(f"[stage1] best = {s1['final_weighted_score']:.6f} ({len(h1)} accepts)")

    print(f"\n[stage2] refine step={args.refine_step} iters={args.refine_iters}")
    w2, s2, h2 = macro_only_hillclimb(per_stem, records, init=w1,
                                      n_iters=args.refine_iters, step=args.refine_step,
                                      seed=args.seed + 1)
    print(f"[stage2] best = {s2['final_weighted_score']:.6f} ({len(h2)} accepts)")

    gain = s2["final_weighted_score"] - equal_score["final_weighted_score"]
    print("\n" + "=" * 60)
    print(f" macro-only weighting OOF gain = {gain:+.6f}")
    print(f"   equal      = {equal_score['final_weighted_score']:.6f}")
    print(f"   macro-opt  = {s2['final_weighted_score']:.6f}")
    print("   per-task (macro-opt):")
    for t in TASKS:
        print(f"     {t:24s} = {s2[t]:.4f}  (equal {equal_score[t]:.4f})")
    print("   T1/T3 frozen at equal:",
          all(tuple(w2[t]) == equal[t] for t in ("promise_status", "evidence_status")))

    OUT_META.parent.mkdir(parents=True, exist_ok=True)
    norm_w = {t: [round(x / sum(w2[t]), 6) for x in w2[t]] for t in TASKS}
    OUT_META.write_text(json.dumps({
        "tag": "macro_weighting",
        "phase": 46,
        "frozen_tasks": ["promise_status", "evidence_status"],
        "optimized_tasks": list(MACRO_TASKS),
        "equal_score": equal_score["final_weighted_score"],
        "macro_opt_score": s2["final_weighted_score"],
        "oof_gain": gain,
        "per_task_macro_opt": {t: s2[t] for t in TASKS},
        "stem_weights_per_task": norm_w,
        "stems": list(TV_STEMS),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[wrote] {OUT_META}")


if __name__ == "__main__":
    main()
