"""Per-task hillclimb v3: random search on fine grid around v2 winners.

Starts from the v2 winning weights and explores 5000 random perturbations
per task on grid [0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0].
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


EXPS = [
    "p2_combo_v2",
    "p2_combo_best",
    "p3_large_lr2e5",
    "p2ab_aug_mask10",
    "p2ac_aug_mix",
    "p2ad_rdrop05",
    "p2ae_msd5",
]
GRID = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
SEED_W = {
    "promise_status": (0.0, 0.0, 0.0, 2.0, 1.0, 1.0, 1.0),
    "verification_timeline": (1.0, 2.0, 1.0, 1.0, 1.0, 0.0, 2.0),
    "evidence_status": (1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0),
    "evidence_quality": (1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0),
}
N_RANDOM = 5000
RNG = random.Random(20260429)


def _score_one(truth, accum_for_one, accum_default, target, records, n):
    accum_full = dict(accum_default)
    accum_full[target] = accum_for_one
    preds = []
    for i in range(n):
        row = {"id": records[i].get("id", i)}
        for t in TASKS:
            row[t] = LABEL_DOMAINS[t][int(accum_full[t][i].argmax())]
        preds.append(row)
    proc = apply_constraints_batch(preds)
    pd_dict = {t: [r[t] for r in proc] for t in TASKS}
    return weighted_score(truth, pd_dict)[target]


def _combine(per_exp_probs, w_tuple, task, n):
    arr = np.zeros((n, NUM_LABELS[task]))
    s = sum(w_tuple)
    if s == 0:
        return arr
    for exp, ww in zip(EXPS, w_tuple):
        if ww == 0:
            continue
        arr += ww * per_exp_probs[exp][task]
    return arr / s


def main() -> None:
    records, _ = load_dataset("data/raw/vpesg4k_train_1000 V1.csv")
    n = len(records)
    truth = {t: [r[t] for r in records] for t in TASKS}

    per_exp_probs: dict[str, dict[str, np.ndarray]] = {}
    for exp in EXPS:
        exp_dir = Path("outputs/checkpoints") / exp
        splits_dir = Path("data/splits") / exp
        seeds = sorted({int(p.name.replace("seed", "")) for p in exp_dir.iterdir() if p.name.startswith("seed")})
        acc = {t: np.zeros((n, NUM_LABELS[t]), dtype=np.float64) for t in TASKS}
        for s in seeds:
            sp = _build_seed_oof(exp_dir, splits_dir, n, s)
            for t in TASKS:
                acc[t] += sp[t]
        for t in TASKS:
            acc[t] /= len(seeds)
        per_exp_probs[exp] = acc

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
        for trial in range(N_RANDOM):
            # perturb: pick 1-3 indices, change to random GRID value
            w = list(best_w)
            k = RNG.randint(1, 3)
            idxs = RNG.sample(range(len(EXPS)), k)
            for idx in idxs:
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
    print(f"[FINAL v3] per-task weighted score = {ws['final_weighted_score']:.5f}")
    print(f"           T1={ws['promise_status']:.4f} T2={ws['verification_timeline']:.4f} "
          f"T3={ws['evidence_status']:.4f} T4={ws['evidence_quality']:.4f}")

    out_dir = Path("reports/analysis/_ensemble")
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(proc).to_csv(out_dir / "per_task_hillclimb_v3_preds.csv", index=False)
    rows = []
    for t in TASKS:
        w_tuple, f1 = best_per_task[t]
        row = {"task": t, "best_f1": f1}
        for exp, ww in zip(EXPS, w_tuple):
            row[f"w_{exp}"] = ww
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_dir / "per_task_hillclimb_v3_summary.csv", index=False)
    print(f"[wrote] {out_dir}/per_task_hillclimb_v3_summary.csv")
    print(f"[wrote] {out_dir}/per_task_hillclimb_v3_preds.csv")


if __name__ == "__main__":
    main()
