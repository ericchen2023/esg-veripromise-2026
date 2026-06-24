"""Per-task hillclimb v2: 7-exp pool, coarse grid [0,1,2].

Pool: combo_v2, combo_best, p3_large_lr2e5, p2ab_aug_mask10, p2ac_aug_mix,
p2ad_rdrop05, p2ae_msd5. Wave C members add diversity even when individually
weaker than combo_best.

3^7 = 2187 candidates per task × 4 tasks ≈ 9k evaluations.
"""
from __future__ import annotations

import itertools
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
GRID = [0.0, 1.0, 2.0]


def _score_one_task(truth, accum_for_one_task, accum_other_default, target_task, records, n):
    accum_full = dict(accum_other_default)
    accum_full[target_task] = accum_for_one_task
    preds = []
    for i in range(n):
        row = {"id": records[i].get("id", i)}
        for t in TASKS:
            row[t] = LABEL_DOMAINS[t][int(accum_full[t][i].argmax())]
        preds.append(row)
    proc = apply_constraints_batch(preds)
    pd_dict = {t: [r[t] for r in proc] for t in TASKS}
    ws = weighted_score(truth, pd_dict)
    return ws[target_task]


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
        print(f"[load] {exp}  seeds={seeds}")

    default = tuple([1.0] * len(EXPS))
    accum_default = {t: np.zeros((n, NUM_LABELS[t])) for t in TASKS}
    for exp, w in zip(EXPS, default):
        for t in TASKS:
            accum_default[t] += w * per_exp_probs[exp][t]
    for t in TASKS:
        accum_default[t] /= sum(default)

    best_per_task: dict[str, tuple] = {}
    best_acc_per_task: dict[str, np.ndarray] = {}
    total_combos = len(GRID) ** len(EXPS)
    for t in TASKS:
        best_score = -1.0
        best_w = default
        best_arr = accum_default[t]
        seen = 0
        for w in itertools.product(GRID, repeat=len(EXPS)):
            if sum(w) == 0:
                continue
            arr = np.zeros((n, NUM_LABELS[t]))
            for exp, ww in zip(EXPS, w):
                if ww == 0:
                    continue
                arr += ww * per_exp_probs[exp][t]
            arr /= sum(w)
            f1 = _score_one_task(truth, arr, accum_default, t, records, n)
            if f1 > best_score:
                best_score = f1
                best_w = w
                best_arr = arr
            seen += 1
        best_per_task[t] = (best_w, best_score)
        best_acc_per_task[t] = best_arr
        print(f"[task] {t:25s}  best_w={best_w}  F1={best_score:.5f}  ({seen} combos)")

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
    print(f"[FINAL v2] per-task weighted score = {ws['final_weighted_score']:.5f}")
    print(f"           T1={ws['promise_status']:.4f} T2={ws['verification_timeline']:.4f} "
          f"T3={ws['evidence_status']:.4f} T4={ws['evidence_quality']:.4f}")

    out_dir = Path("reports/analysis/_ensemble")
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(proc).to_csv(out_dir / "per_task_hillclimb_v2_preds.csv", index=False)
    rows = []
    for t in TASKS:
        w_tuple, f1 = best_per_task[t]
        row = {"task": t, "best_f1": f1}
        for exp, ww in zip(EXPS, w_tuple):
            row[f"w_{exp}"] = ww
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_dir / "per_task_hillclimb_v2_summary.csv", index=False)
    print(f"[wrote] {out_dir}/per_task_hillclimb_v2_summary.csv")
    print(f"[wrote] {out_dir}/per_task_hillclimb_v2_preds.csv")


if __name__ == "__main__":
    main()
