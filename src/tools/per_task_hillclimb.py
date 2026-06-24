"""Per-task hillclimb on weights for OOF ensemble.

For each task independently, search a discrete weight grid over the EXPS pool
to maximize the per-task F1; then combine and report the final weighted score.

This relaxes the assumption that one weight tuple is best for all four tasks.
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


EXPS = ["p2_combo_v2", "p2_combo_best", "p3_large_lr2e5"]
GRID = [0.0, 0.5, 1.0, 1.5, 2.0]


def _score_one_task_with_full_pipeline(
    truth, accum_for_one_task, accum_other_default, target_task, records, n
):
    """Build full preds replacing one task; return per-task F1 for target_task."""
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
    return ws[target_task], ws["final_weighted_score"]


def main() -> None:
    records, _ = load_dataset("data/raw/vpesg4k_train_1000 V1.csv")
    n = len(records)
    truth = {t: [r[t] for r in records] for t in TASKS}

    # per-exp seed-averaged probs
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

    # default per-task weights = (1,1,1)
    default = (1.0, 1.0, 1.0)
    accum_default = {t: np.zeros((n, NUM_LABELS[t])) for t in TASKS}
    for exp, w in zip(EXPS, default):
        for t in TASKS:
            accum_default[t] += w * per_exp_probs[exp][t]
    for t in TASKS:
        accum_default[t] /= sum(default)

    # Per-task best weights
    best_per_task: dict[str, tuple] = {}
    best_acc_per_task: dict[str, np.ndarray] = {}
    for t in TASKS:
        best_score = -1.0
        best_w = default
        best_arr = accum_default[t]
        for w in itertools.product(GRID, repeat=len(EXPS)):
            if sum(w) == 0:
                continue
            arr = np.zeros((n, NUM_LABELS[t]))
            for exp, ww in zip(EXPS, w):
                if ww == 0:
                    continue
                arr += ww * per_exp_probs[exp][t]
            arr /= sum(w)
            f1, _ = _score_one_task_with_full_pipeline(
                truth, arr, accum_default, t, records, n
            )
            if f1 > best_score:
                best_score = f1
                best_w = w
                best_arr = arr
        best_per_task[t] = (best_w, best_score)
        best_acc_per_task[t] = best_arr
        print(f"[task] {t:25s}  best_w={best_w}  F1={best_score:.5f}")

    # Final combined preds
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
    print(f"[FINAL] per-task weighted score = {ws['final_weighted_score']:.5f}")
    print(f"        T1={ws['promise_status']:.4f} T2={ws['verification_timeline']:.4f} "
          f"T3={ws['evidence_status']:.4f} T4={ws['evidence_quality']:.4f}")

    # Save preds + summary
    out_dir = Path("reports/analysis/_ensemble")
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(proc).to_csv(out_dir / "per_task_hillclimb_preds.csv", index=False)
    summary = {
        "task": list(TASKS),
        "best_w_v2": [best_per_task[t][0][0] for t in TASKS],
        "best_w_best": [best_per_task[t][0][1] for t in TASKS],
        "best_w_large": [best_per_task[t][0][2] for t in TASKS],
        "best_f1": [best_per_task[t][1] for t in TASKS],
    }
    pd.DataFrame(summary).to_csv(out_dir / "per_task_hillclimb_summary.csv", index=False)
    print(f"[wrote] {out_dir}/per_task_hillclimb_summary.csv")
    print(f"[wrote] {out_dir}/per_task_hillclimb_preds.csv")


if __name__ == "__main__":
    main()
