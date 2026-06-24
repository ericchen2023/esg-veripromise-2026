"""Quick in-process weight scan for 3-way OOF ensemble.

Avoids the python startup overhead of repeated CLI calls and produces a
single comparison CSV across many weight configurations.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data.dataset import LABEL_DOMAINS, NUM_LABELS, TASKS
from src.data.loader import load_dataset
from src.eval.metrics import weighted_score
from src.inference.post_process import apply_constraints_batch
from src.tools.oof_ensemble import _build_seed_oof


EXPS = ["p2_combo_v2", "p2_combo_best", "p3_large_lr2e5"]
WEIGHT_COMBOS = [
    (1.0, 1.0, 0.0),    # combo_v2 + combo_best (no large)
    (1.0, 0.0, 0.0),    # combo_v2 only
    (1.5, 1.0, 0.5),
    (1.5, 1.0, 0.7),
    (1.0, 1.0, 0.5),
    (1.0, 1.0, 0.7),
    (1.0, 1.0, 1.0),    # equal
    (1.5, 0.5, 0.5),
    (2.0, 1.0, 0.5),
    (2.0, 1.0, 1.0),
    (1.5, 1.5, 1.0),
    (2.0, 1.5, 1.0),
    (2.0, 0.5, 0.5),
    (3.0, 1.0, 1.0),
    (1.0, 1.5, 1.0),    # combo_best heaviest
    (1.0, 0.5, 0.0),    # no large
]


def main() -> None:
    records, _ = load_dataset("data/raw/vpesg4k_train_1000 V1.csv")
    n = len(records)
    truth = {t: [r[t] for r in records] for t in TASKS}

    # Pre-compute per-exp seed-averaged probs ONCE
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
        # standalone score
        preds = []
        for i in range(n):
            row = {"id": records[i].get("id", i)}
            for t in TASKS:
                row[t] = LABEL_DOMAINS[t][int(acc[t][i].argmax())]
            preds.append(row)
        proc = apply_constraints_batch(preds)
        pd_dict = {t: [r[t] for r in proc] for t in TASKS}
        ws = weighted_score(truth, pd_dict)
        print(f"[solo] {exp:20s}  score={ws['final_weighted_score']:.5f}  "
              f"T1={ws['promise_status']:.4f} T2={ws['verification_timeline']:.4f} "
              f"T3={ws['evidence_status']:.4f} T4={ws['evidence_quality']:.4f}")

    # Weight scan
    rows = []
    for weights in WEIGHT_COMBOS:
        if sum(weights) == 0:
            continue
        accum = {t: np.zeros((n, NUM_LABELS[t]), dtype=np.float64) for t in TASKS}
        for exp, w in zip(EXPS, weights):
            if w == 0:
                continue
            for t in TASKS:
                accum[t] += w * per_exp_probs[exp][t]
        total_w = sum(weights)
        for t in TASKS:
            accum[t] /= total_w
        preds = []
        for i in range(n):
            row = {"id": records[i].get("id", i)}
            for t in TASKS:
                row[t] = LABEL_DOMAINS[t][int(accum[t][i].argmax())]
            preds.append(row)
        proc = apply_constraints_batch(preds)
        pd_dict = {t: [r[t] for r in proc] for t in TASKS}
        ws = weighted_score(truth, pd_dict)
        tag = f"{weights[0]}_{weights[1]}_{weights[2]}"
        rows.append({
            "weights_v2_best_large": tag,
            "T1": ws["promise_status"],
            "T2": ws["verification_timeline"],
            "T3": ws["evidence_status"],
            "T4": ws["evidence_quality"],
            "score": ws["final_weighted_score"],
        })

    df = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    out = Path("reports/analysis/_ensemble") / "weight_scan_v2_best_large.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print()
    print(df.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print(f"\n[wrote] {out}")


if __name__ == "__main__":
    main()
