"""Per-task hillclimb v7b: same 11-way pool as v7 but biased random search.

v7 issue: warm-start has new members (idx 9, 10) at weight 0; random search
samples 1-3 of 11 indices uniformly, so new members are explored only ~27%
of iterations and need to land on a non-zero weight that simultaneously
improves the score. Result: 10000 iters never broke v6 SOTA.

v7b fix: every perturbation MUST include at least one of {9, 10} (the new
Sprint A members). The other 0-2 indices are sampled uniformly from {0..8}.
This guarantees the search explores the 2D subspace spanned by p4a/p4b at
every step, while still allowing concurrent tuning of legacy weights.
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.dataset import LABEL_DOMAINS, NUM_LABELS, TASKS
from src.data.loader import load_dataset
from src.eval.metrics import weighted_score
from src.inference.post_process import apply_constraints_batch
from src.tools.oof_ensemble import _build_seed_oof
from src.tools.per_task_hillclimb_v7 import EXPS, LOADERS, SEED_W, _combine, _load_member, _score_one


GRID = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
N_RANDOM = 12000
NEW_IDX = [9, 10]                 # p4a, p4b
LEGACY_IDX = list(range(9))       # 0..8
RNG = random.Random(20260502)


def main() -> None:
    records, _ = load_dataset("data/raw/vpesg4k_train_1000 V1.csv")
    n = len(records)
    truth = {t: [r[t] for r in records] for t in TASKS}

    per_exp_probs = {name: _load_member(name, n) for name in EXPS}

    default = tuple([1.0] * len(EXPS))
    accum_default = {t: np.zeros((n, NUM_LABELS[t])) for t in TASKS}
    for exp, w in zip(EXPS, default):
        for t in TASKS:
            accum_default[t] += w * per_exp_probs[exp][t]
    for t in TASKS:
        accum_default[t] /= sum(default)

    best_per_task: dict[str, tuple] = {}
    best_acc_per_task: dict[str, np.ndarray] = {}

    for t in TASKS:
        seed_w = SEED_W[t]
        seed_arr = _combine(per_exp_probs, seed_w, t, n)
        seed_f1 = _score_one(truth, seed_arr, accum_default, t, records, n)
        best_w = seed_w
        best_score = seed_f1
        best_arr = seed_arr
        print(f"[task] {t:25s}  seed_w F1={seed_f1:.5f}")
        for it in range(N_RANDOM):
            w = list(best_w)
            # force at least one new member to be perturbed
            new_pick = RNG.sample(NEW_IDX, RNG.randint(1, 2))
            extra_k = RNG.randint(0, 2)
            extra_pick = RNG.sample(LEGACY_IDX, extra_k)
            for idx in new_pick + extra_pick:
                w[idx] = RNG.choice(GRID)
            w = tuple(w)
            if sum(w) == 0:
                continue
            arr = _combine(per_exp_probs, w, t, n)
            f1 = _score_one(truth, arr, accum_default, t, records, n)
            if f1 > best_score:
                best_score = f1
                best_w = w
                best_arr = arr
        best_per_task[t] = (best_w, best_score)
        best_acc_per_task[t] = best_arr
        print(f"[task] {t:25s}  best_w={best_w}  F1={best_score:.5f}")

    preds = []
    for i in range(n):
        row = {"id": records[i].get("id", i)}
        for t in TASKS:
            row[t] = LABEL_DOMAINS[t][int(best_acc_per_task[t][i].argmax())]
        preds.append(row)
    proc = apply_constraints_batch(preds)
    pd_dict = {t: [r[t] for r in proc] for t in TASKS}
    ws = weighted_score(truth, pd_dict)
    print()
    print(f"[FINAL v7b] per-task weighted score = {ws['final_weighted_score']:.5f}")
    print(f"            T1={ws['promise_status']:.4f} T2={ws['verification_timeline']:.4f} "
          f"T3={ws['evidence_status']:.4f} T4={ws['evidence_quality']:.4f}")

    out_dir = Path("reports/analysis/_ensemble")
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(proc).to_csv(out_dir / "per_task_hillclimb_v7b_preds.csv", index=False)
    rows = []
    for t in TASKS:
        w_tuple, f1 = best_per_task[t]
        row = {"task": t, "best_f1": f1}
        for exp, ww in zip(EXPS, w_tuple):
            row[f"w_{exp}"] = ww
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_dir / "per_task_hillclimb_v7b_summary.csv", index=False)
    print(f"[wrote] {out_dir}/per_task_hillclimb_v7b_summary.csv")
    print(f"[wrote] {out_dir}/per_task_hillclimb_v7b_preds.csv")


if __name__ == "__main__":
    main()
