"""Per-task simplex hillclimb on N-way stored OOF stack.

For each (exp), build seed-averaged stored OOF probs (N x C_t per task).
For each task, search a simplex over exp weights (sum=1, step) and keep
the best per-task weights. Final ensemble = weighted sum per task.

Usage:
  python -m src.tools.u10_classw_stack_search \
      --exps p2_combo_best p2_combo_best_u10_pseudo p2_combo_best_u10_pseudo_v2 \
             p2_combo_best_u10_pseudo_v2_classw_focal_t4_g3 \
      --grid-step 0.05 --tag u10_4way_classw
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.dataset import LABEL_DOMAINS, NUM_LABELS, TASKS
from src.data.loader import load_dataset
from src.eval.metrics import weighted_score
from src.inference.post_process import apply_constraints_batch
from src.tools.oof_ensemble import _build_seed_oof


def stored_probs(exp: str, n: int) -> dict[str, np.ndarray]:
    exp_dir = Path("outputs/checkpoints") / exp
    splits_dir = Path("data/splits") / exp
    seeds = sorted({int(p.name.replace("seed", "")) for p in exp_dir.iterdir() if p.name.startswith("seed")})
    accum = {t: np.zeros((n, NUM_LABELS[t]), dtype=np.float64) for t in TASKS}
    for s in seeds:
        sp = _build_seed_oof(exp_dir, splits_dir, n, s)
        for t in TASKS:
            accum[t] += sp[t]
    for t in TASKS:
        accum[t] /= max(len(seeds), 1)
    return accum


def simplex_grid(k: int, step: float) -> list[tuple[float, ...]]:
    n = int(round(1.0 / step))
    out: list[tuple[float, ...]] = []
    def rec(prefix: list[int], remain: int, slots: int) -> None:
        if slots == 1:
            out.append(tuple((x / n) for x in prefix + [remain]))
            return
        for i in range(remain + 1):
            rec(prefix + [i], remain - i, slots - 1)
    rec([], n, k)
    return out


def score_with_weights(exp_probs: list[dict[str, np.ndarray]], task_weights: dict[str, tuple[float, ...]],
                       records: list[dict]) -> tuple[float, dict[str, float]]:
    n = len(records)
    mixed: dict[str, np.ndarray] = {}
    for t in TASKS:
        ws = task_weights[t]
        m = np.zeros((n, NUM_LABELS[t]), dtype=np.float64)
        for w, ep in zip(ws, exp_probs):
            m += w * ep[t]
        mixed[t] = m
    raw = []
    for i in range(n):
        row = {"id": records[i].get("id", i)}
        for t in TASKS:
            row[t] = LABEL_DOMAINS[t][int(mixed[t][i].argmax())]
        raw.append(row)
    constrained = apply_constraints_batch(raw)
    truth = {t: [r[t] for r in records] for t in TASKS}
    pred = {t: [r[t] for r in constrained] for t in TASKS}
    res = weighted_score(truth, pred)
    final = float(res.pop("final_weighted_score"))
    per_task = {t: float(res[t]) for t in TASKS}
    return final, per_task


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exps", nargs="+", required=True)
    ap.add_argument("--data", default="data/raw/vpesg4k_train_1000 V1.csv")
    ap.add_argument("--grid-step", type=float, default=0.05)
    ap.add_argument("--tag", default="stack_search")
    args = ap.parse_args()

    records, _ = load_dataset(args.data)
    n = len(records)
    print(f"[stack-search] N={n} exps={args.exps}")

    exp_probs = [stored_probs(e, n) for e in args.exps]
    k = len(args.exps)
    grid = simplex_grid(k, args.grid_step)
    print(f"[stack-search] simplex points = {len(grid)} (k={k}, step={args.grid_step})")

    # Equal-weight baseline ----
    eq = tuple([1.0 / k] * k)
    eq_pt = {t: eq for t in TASKS}
    eq_score, eq_per = score_with_weights(exp_probs, eq_pt, records)
    print(f"[stack-search] equal weights: weighted={eq_score:.5f}")
    print("  per-task:", {t: round(v, 5) for t, v in eq_per.items()})

    # Per-task search (other tasks fixed at equal weights) ----
    best_pt: dict[str, tuple[float, ...]] = {}
    for t in TASKS:
        best_w = eq
        best_s = eq_score
        for w in grid:
            tw = dict(eq_pt)
            tw[t] = w
            s, _per = score_with_weights(exp_probs, tw, records)
            if s > best_s:
                best_s = s
                best_w = w
        best_pt[t] = best_w
        print(f"[stack-search] task={t}: best_w={[round(x,3) for x in best_w]} score={best_s:.5f}")

    # Joint refinement (one round) ----
    final_score, final_per = score_with_weights(exp_probs, best_pt, records)
    print(f"\n[stack-search] FINAL after per-task search: weighted={final_score:.5f}")
    print("  per-task:", {t: round(v, 5) for t, v in final_per.items()})

    out_dir = Path("reports/analysis/_ensemble")
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "exps": args.exps,
        "grid_step": args.grid_step,
        "equal_score": eq_score,
        "equal_per_task": eq_per,
        "final_score": final_score,
        "final_per_task": final_per,
        "best_weights_per_task": {t: list(w) for t, w in best_pt.items()},
    }
    (out_dir / f"{args.tag}_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[wrote] {out_dir / (args.tag + '_meta.json')}")


if __name__ == "__main__":
    main()
