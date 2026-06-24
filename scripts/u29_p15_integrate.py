"""U29 — Phase 48 new-stem integration + analysis.

Reconstructs the new backbone stem's 2000-row OOF, reports its single-model
per-task score (admission-gate check on T4, the transferable high-weight task),
slices train(0-999) vs val(1000-1999) to confirm T4 transfer-faithfulness, then
runs a 9-way (8 TV stems + new stem) post-constraint joint hillclimb and reports
whether it beats the equal-weight 8-stem baseline.

Usage:
    python -m scripts.u29_p15_integrate --new-stem p15_robertalarge_t4
    python -m scripts.u29_p15_integrate --new-stem p15_robertalarge_t4 --iters 12000 --refine-iters 6000
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

from src.data.dataset import LABEL_DOMAINS, NUM_LABELS, TASKS
from src.data.loader import load_dataset
from src.eval.metrics import weighted_score
from src.inference.post_process import apply_constraints_batch
from scripts.u16_tv_oof_ensemble import (
    COMBINED_CSV,
    TV_STEMS,
    _reconstruct_oof,
    _score,
)

VAL_START = 1000  # combined idx 1000-1999 = official val (held-out within CV)
OUTPUT_DIR = Path("reports/analysis/_ensemble")


def _mix(per_stem, stems, stem_weights_per_task):
    out = {}
    for t in TASKS:
        ws = stem_weights_per_task[t]
        s = sum(ws)
        assert s > 0, f"zero stem weights for task {t}"
        arr = np.zeros_like(per_stem[stems[0]][t], dtype=np.float64)
        for stem, w in zip(stems, ws):
            if w == 0.0:
                continue
            arr += w * per_stem[stem][t]
        out[t] = arr / s
    return out


def _move(weights, *, step, rng):
    vals = list(weights)
    donors = [i for i, v in enumerate(vals) if v >= step - 1e-12]
    if not donors:
        return weights
    donor = rng.choice(donors)
    receiver = rng.choice([i for i in range(len(vals)) if i != donor])
    vals[donor] = max(0.0, vals[donor] - step)
    vals[receiver] += step
    total = sum(vals)
    if total <= 0.0:
        return weights
    return tuple(round(v / total, 10) for v in vals)


def _slice_score(per_task, records, lo, hi):
    sub = {t: per_task[t][lo:hi] for t in TASKS}
    return _score(sub, records[lo:hi])


def joint_hillclimb(per_stem, stems, records, *, init, n_iters, step, seed,
                    bias_new_idx=None):
    rng = random.Random(seed)
    best = {t: tuple(init[t]) for t in TASKS}
    best_score = _score(_mix(per_stem, stems, best), records)
    accepts = 0
    for it in range(1, n_iters + 1):
        cand = dict(best)
        t = rng.choice(TASKS)
        cand[t] = _move(best[t], step=step, rng=rng)
        if cand[t] == best[t]:
            continue
        cs = _score(_mix(per_stem, stems, cand), records)
        if cs["final_weighted_score"] > best_score["final_weighted_score"] + 1e-12:
            best, best_score = cand, cs
            accepts += 1
            print(f"[{it:5d}] {t:24s} -> {best_score['final_weighted_score']:.6f}")
    return best, best_score, accepts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--new-stem", required=True)
    ap.add_argument("--iters", type=int, default=12000)
    ap.add_argument("--step", type=float, default=0.1)
    ap.add_argument("--refine-iters", type=int, default=6000)
    ap.add_argument("--refine-step", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gate-t4", type=float, default=0.458,
                    help="admission gate: new stem single T4 OOF must exceed this")
    args = ap.parse_args()

    records, _ = load_dataset(COMBINED_CSV)
    n = len(records)
    stems = list(TV_STEMS) + [args.new_stem]
    print(f"[data] {n} records; {len(stems)} stems (new={args.new_stem})")

    per_stem = {}
    for stem in stems:
        per_stem[stem] = _reconstruct_oof(stem, n)

    # ---- Admission gate + transfer-faithfulness on the new stem ----
    new = per_stem[args.new_stem]
    full = _score(new, records)
    tr = _slice_score(new, records, 0, VAL_START)
    va = _slice_score(new, records, VAL_START, n)
    print("\n=== NEW STEM single-model OOF ===")
    print(f"  final_weighted = {full['final_weighted_score']:.5f}")
    for t in TASKS:
        print(f"  {t:24s} all={full[t]:.4f}  train={tr[t]:.4f}  val={va[t]:.4f}")
    t4 = full["evidence_quality"]
    gate_pass = t4 > args.gate_t4
    print(f"\n  ADMISSION GATE: T4={t4:.4f} vs {args.gate_t4:.4f} -> "
          f"{'PASS' if gate_pass else 'FAIL'}")

    # compare to best existing single-stem T4
    existing_t4 = {s: _score(per_stem[s], records)["evidence_quality"] for s in TV_STEMS}
    best_ex = max(existing_t4, key=existing_t4.get)
    print(f"  best existing single-stem T4 = {existing_t4[best_ex]:.4f} ({best_ex})")

    # ---- 8-stem equal baseline (banked best proxy) ----
    eq8 = {t: tuple([1.0] * len(TV_STEMS)) for t in TASKS}
    s_eq8 = _score(_mix(per_stem, list(TV_STEMS), eq8), records)
    print(f"\n[baseline] equal 8-stem = {s_eq8['final_weighted_score']:.6f}")

    # ---- 9-way joint hillclimb, warm-start = equal-9 ----
    eq9 = {t: tuple([1.0] * len(stems)) for t in TASKS}
    s_eq9 = _score(_mix(per_stem, stems, eq9), records)
    print(f"[baseline] equal 9-stem = {s_eq9['final_weighted_score']:.6f}")

    print(f"\n[hillclimb] stage1 step={args.step} iters={args.iters}")
    w1, sc1, a1 = joint_hillclimb(per_stem, stems, records, init=eq9,
                                  n_iters=args.iters, step=args.step, seed=args.seed)
    print(f"  stage1 = {sc1['final_weighted_score']:.6f} ({a1} accepts)")
    print(f"[hillclimb] stage2 step={args.refine_step} iters={args.refine_iters}")
    w2, sc2, a2 = joint_hillclimb(per_stem, stems, records, init=w1,
                                  n_iters=args.refine_iters, step=args.refine_step,
                                  seed=args.seed + 1)
    print(f"  stage2 = {sc2['final_weighted_score']:.6f} ({a2} accepts)")

    new_idx = len(stems) - 1
    print("\n" + "=" * 64)
    print(f"  equal 8-stem (banked proxy) = {s_eq8['final_weighted_score']:.6f}")
    print(f"  9-way joint hillclimb       = {sc2['final_weighted_score']:.6f}")
    print(f"  delta                       = "
          f"{sc2['final_weighted_score'] - s_eq8['final_weighted_score']:+.6f}")
    print("\n  per-task (final 9-way):")
    for t in TASKS:
        print(f"    {t:24s} = {sc2[t]:.4f}")
    print("\n  new-stem weight share per task:")
    for t in TASKS:
        ws = w2[t]
        s = sum(ws)
        print(f"    {t:24s}: new={ws[new_idx] / s:.4f}  (full={[round(w / s, 3) for w in ws]})")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    meta = {
        "tag": "u29_p15_integrate",
        "phase": 48,
        "new_stem": args.new_stem,
        "new_single_oof": full["final_weighted_score"],
        "new_per_task_all": {t: full[t] for t in TASKS},
        "new_per_task_train": {t: tr[t] for t in TASKS},
        "new_per_task_val": {t: va[t] for t in TASKS},
        "admission_gate_t4": args.gate_t4,
        "admission_pass": bool(gate_pass),
        "equal8_score": s_eq8["final_weighted_score"],
        "equal9_score": s_eq9["final_weighted_score"],
        "hillclimb9_score": sc2["final_weighted_score"],
        "delta_vs_equal8": sc2["final_weighted_score"] - s_eq8["final_weighted_score"],
        "final_per_task": {t: sc2[t] for t in TASKS},
        "stem_weights_per_task": {t: list(w2[t]) for t in TASKS},
        "stems": stems,
        "accepts": {"stage1": a1, "stage2": a2},
    }
    out = OUTPUT_DIR / f"u29_{args.new_stem}_meta.json"
    out.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[wrote] {out}")


if __name__ == "__main__":
    main()
