"""U16 — Train+Val (TV) 8-stem OOF ensemble joint hillclimb (Phase 42 prep).

Phase 41 retrained all 8 AP-D4 stems on the 2,000-row train+val combined set
(single seed 42, 5-fold). Each fold saved per-sample stored-view OOF probs to
``outputs/checkpoints/<stem>/seed42/fold*/oof_probs.npz`` keyed by the global
record index into ``load_dataset(train_val_combined.csv)`` order.

This script reconstructs the full 2,000-row OOF for each stem, then runs a
post-constraint **joint** hillclimb over per-task stem-mix weights (the
Phase 13+ SOTA search method). The result is the expected TV ensemble OOF
score — the closest proxy to test-set performance available before the
official test release, and the warm-start anchor for Phase 42.

Note: TV stems have only the stored view (no middle/tail TTA computed), so the
search is over stem-mix weights only — a faithful single-view analog of the
AP-D4 8-way per-task stack.

Usage:
    python scripts/u16_tv_oof_ensemble.py
    python scripts/u16_tv_oof_ensemble.py --iters 8000 --seed 42
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

COMBINED_CSV = "data/processed/train_val_combined.csv"
CKPT_ROOT = Path("outputs/checkpoints")
OUTPUT_DIR = Path("reports/analysis/_ensemble")
SEED_DIR = "seed42"
N_FOLDS = 5

# The 8 AP-D4 TV stems (Phase 41), ordered as in phase41_train_all_tv.py.
TV_STEMS: tuple[str, ...] = (
    "p2_combo_best_tv",
    "p2_combo_best_u10_pseudo_tv",
    "p2_combo_best_u10_pseudo_v2_tv",
    "p2_combo_best_u10_pseudo_v2_classw_focal_t4_g3_tv",
    "p2_combo_best_u10_pseudo_v3_classw_focal_t4_g3_tv",
    "p2_combo_best_classw_focal_u6pro_tv",
    "p2_combo_best_aug_plus_tv",
    "p2_combo_best_aug_plus_v2_tv",
)


def _reconstruct_oof(stem: str, n: int) -> dict[str, np.ndarray]:
    """Reconstruct full [n, C_t] stored OOF probs for one stem from 5 folds."""
    out = {t: np.full((n, NUM_LABELS[t]), np.nan, dtype=np.float64) for t in TASKS}
    seen = np.zeros(n, dtype=bool)
    for fold in range(N_FOLDS):
        npz_path = CKPT_ROOT / stem / SEED_DIR / f"fold{fold}" / "oof_probs.npz"
        if not npz_path.exists():
            raise FileNotFoundError(f"missing OOF for {stem} fold{fold}: {npz_path}")
        z = np.load(npz_path)
        idx = z["indices"].astype(int)
        for t in TASKS:
            out[t][idx] = z[f"probs_{t}"].astype(np.float64)
        seen[idx] = True
    if not seen.all():
        missing = int((~seen).sum())
        raise RuntimeError(f"{stem}: {missing} samples uncovered by OOF folds")
    for t in TASKS:
        if np.isnan(out[t]).any():
            raise RuntimeError(f"{stem}: NaN remaining in task {t}")
    return out


def _score(probs_per_task: dict[str, np.ndarray], records: list[dict]) -> dict[str, float]:
    n = len(records)
    raw = []
    for i in range(n):
        row = {"id": records[i].get("id", i)}
        for t in TASKS:
            row[t] = LABEL_DOMAINS[t][int(probs_per_task[t][i].argmax())]
        raw.append(row)
    constrained = apply_constraints_batch(raw)
    truth = {t: [r[t] for r in records] for t in TASKS}
    pred = {t: [r[t] for r in constrained] for t in TASKS}
    return weighted_score(truth, pred)


def _mix(
    per_stem: dict[str, dict[str, np.ndarray]],
    stem_weights_per_task: dict[str, tuple[float, ...]],
) -> dict[str, np.ndarray]:
    """Blend stems per task -> {task: [n, C_t]} (weights renormalized)."""
    out: dict[str, np.ndarray] = {}
    for t in TASKS:
        ws = stem_weights_per_task[t]
        s = sum(ws)
        assert s > 0, f"zero stem weights for task {t}"
        arr = np.zeros_like(per_stem[TV_STEMS[0]][t], dtype=np.float64)
        for stem, w in zip(TV_STEMS, ws):
            if w == 0.0:
                continue
            arr += w * per_stem[stem][t]
        out[t] = arr / s
    return out


def _move_simplex_mass(weights: tuple[float, ...], *, step: float, rng: random.Random) -> tuple[float, ...]:
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


def joint_hillclimb(
    per_stem: dict[str, dict[str, np.ndarray]],
    records: list[dict],
    *,
    init: dict[str, tuple[float, ...]],
    n_iters: int,
    step: float,
    seed: int,
) -> tuple[dict[str, tuple[float, ...]], dict[str, float], list[dict]]:
    rng = random.Random(seed)
    best = {t: tuple(init[t]) for t in TASKS}
    best_score = _score(_mix(per_stem, best), records)
    history: list[dict] = []
    for it in range(1, n_iters + 1):
        cand = dict(best)
        t = rng.choice(TASKS)
        cand[t] = _move_simplex_mass(best[t], step=step, rng=rng)
        if cand[t] == best[t]:
            continue
        cand_score = _score(_mix(per_stem, cand), records)
        if cand_score["final_weighted_score"] > best_score["final_weighted_score"] + 1e-12:
            best = cand
            best_score = cand_score
            history.append({"iter": it, "task": t, "score": best_score["final_weighted_score"]})
            print(f"[refine {it:5d}] {t:24s} -> {best_score['final_weighted_score']:.6f}")
    return best, best_score, history


def main() -> None:
    ap = argparse.ArgumentParser(description="TV 8-stem OOF ensemble joint hillclimb")
    ap.add_argument("--iters", type=int, default=8000)
    ap.add_argument("--step", type=float, default=0.1)
    ap.add_argument("--refine-iters", type=int, default=4000)
    ap.add_argument("--refine-step", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--csv", default=COMBINED_CSV)
    args = ap.parse_args()

    records, _ = load_dataset(args.csv)
    n = len(records)
    print(f"[data] loaded {n} records from {args.csv}")

    per_stem: dict[str, dict[str, np.ndarray]] = {}
    print(f"[oof] reconstructing 2000-row OOF for {len(TV_STEMS)} TV stems")
    single_scores: dict[str, float] = {}
    for stem in TV_STEMS:
        per_stem[stem] = _reconstruct_oof(stem, n)
        s = _score(per_stem[stem], records)
        single_scores[stem] = s["final_weighted_score"]
        print(f"  {stem:52s} single OOF={s['final_weighted_score']:.5f}")

    # --- Baseline 1: equal-weight blend across all 8 stems ---
    equal = {t: tuple([1.0] * len(TV_STEMS)) for t in TASKS}
    equal_score = _score(_mix(per_stem, equal), records)
    print(f"\n[baseline] equal-weight 8-stem blend = {equal_score['final_weighted_score']:.6f}")

    best_single = max(single_scores, key=single_scores.get)
    print(f"[baseline] best single stem = {best_single} ({single_scores[best_single]:.5f})")

    # --- Joint hillclimb, warm-start from equal weights ---
    print(f"\n[hillclimb] stage 1: step={args.step} iters={args.iters}")
    w1, s1, h1 = joint_hillclimb(
        per_stem, records, init=equal, n_iters=args.iters, step=args.step, seed=args.seed
    )
    print(f"[hillclimb] stage 1 best = {s1['final_weighted_score']:.6f} ({len(h1)} accepts)")

    print(f"\n[hillclimb] stage 2 (refine): step={args.refine_step} iters={args.refine_iters}")
    w2, s2, h2 = joint_hillclimb(
        per_stem, records, init=w1, n_iters=args.refine_iters, step=args.refine_step, seed=args.seed + 1
    )
    print(f"[hillclimb] stage 2 best = {s2['final_weighted_score']:.6f} ({len(h2)} accepts)")

    final_weights, final_score = w2, s2

    print("\n" + "=" * 64)
    print(" TV 8-stem OOF ensemble — Phase 42 prep summary")
    print("=" * 64)
    print(f"  best single stem   = {single_scores[best_single]:.6f}  ({best_single})")
    print(f"  equal-weight blend = {equal_score['final_weighted_score']:.6f}")
    print(f"  joint hillclimb    = {final_score['final_weighted_score']:.6f}")
    print("\n  per-task (final ensemble):")
    for t in TASKS:
        print(f"    {t:24s} = {final_score[t]:.4f}")
    print("\n  per-task stem weights (normalized):")
    for t in TASKS:
        ws = final_weights[t]
        s = sum(ws)
        norm = [round(w / s, 4) for w in ws]
        print(f"    {t:24s}: {norm}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    meta = {
        "tag": "tv_oof_ensemble",
        "phase": 41,
        "purpose": "Phase 42 prep: TV 8-stem OOF joint hillclimb on 2000-row combined set",
        "stems": list(TV_STEMS),
        "n_records": n,
        "single_oof_scores": single_scores,
        "equal_weight_score": equal_score["final_weighted_score"],
        "best_single_stem": best_single,
        "final_score": final_score["final_weighted_score"],
        "final_per_task": {t: final_score[t] for t in TASKS},
        "stem_weights_per_task": {t: list(final_weights[t]) for t in TASKS},
        "search": {
            "stage1": {"step": args.step, "iters": args.iters, "accepts": len(h1)},
            "stage2": {"step": args.refine_step, "iters": args.refine_iters, "accepts": len(h2)},
            "seed": args.seed,
        },
        "history": h1 + h2,
    }
    out = OUTPUT_DIR / "tv_oof_ensemble_meta.json"
    out.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[wrote] {out}")
    print("Next: when test set is released, reuse stem_weights_per_task for Phase 42 inference.")


if __name__ == "__main__":
    main()
