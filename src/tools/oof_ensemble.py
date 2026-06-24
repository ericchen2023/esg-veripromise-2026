"""Probability-level ensemble across multiple experiments.

Reads oof_probs.npz from each (exp, seed, fold), maps local indices to global,
averages probabilities per task across all (exp, seed) pairs (folds combined to
form full N), then post-processes and reports weighted_score.

Usage:
    python -m src.tools.oof_ensemble --exps p2_combo_best p3_large_lr2e5 --weights 1.0 1.0
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.dataset import LABEL_DOMAINS, NUM_LABELS, TASKS
from src.data.loader import load_dataset
from src.eval.metrics import weighted_score
from src.inference.post_process import apply_constraints_batch


def _load_split_map(splits_dir: Path, seed: int) -> dict[int, list[int]]:
    p = splits_dir / f"seed{seed}.json"
    obj = json.loads(p.read_text(encoding="utf-8"))
    return {int(f["fold"]): list(map(int, f.get("val_idx", f.get("val", [])))) for f in obj["folds"]}


def _build_seed_oof(exp_dir: Path, splits_dir: Path, n: int, seed: int) -> dict[str, np.ndarray]:
    seed_dir = exp_dir / f"seed{seed}"
    fold_dirs = sorted([p for p in seed_dir.iterdir() if p.is_dir()])
    probs = {t: np.zeros((n, NUM_LABELS[t]), dtype=np.float64) for t in TASKS}
    seen = np.zeros(n, dtype=bool)
    split_map = _load_split_map(splits_dir, seed)
    for fdir in fold_dirs:
        npz = fdir / "oof_probs.npz"
        if not npz.exists():
            continue
        fold = int(fdir.name.replace("fold", ""))
        z = np.load(npz)
        idx = z["indices"].astype(int)
        if idx.max() < len(split_map.get(fold, [])):
            idx = np.array(split_map[fold], dtype=int)
        for t in TASKS:
            probs[t][idx] = z[f"probs_{t}"]
        seen[idx] = True
    if not seen.all():
        miss = int((~seen).sum())
        print(f"[ens] {exp_dir.name}/seed{seed}: missing {miss} OOF samples")
    return probs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exps", nargs="+", required=True)
    ap.add_argument("--weights", nargs="+", type=float, default=None)
    ap.add_argument("--data", default="data/raw/vpesg4k_train_1000 V1.csv")
    ap.add_argument("--out", default="reports/analysis/_ensemble")
    args = ap.parse_args()

    weights = args.weights or [1.0] * len(args.exps)
    assert len(weights) == len(args.exps)

    records, _ = load_dataset(args.data)
    n = len(records)
    print(f"[ens] N={n} exps={args.exps} weights={weights}")

    # accumulate per-exp seed-averaged probs
    accum = {t: np.zeros((n, NUM_LABELS[t]), dtype=np.float64) for t in TASKS}
    total_w = 0.0
    summary_rows = []
    for exp, w in zip(args.exps, weights):
        exp_dir = Path("outputs/checkpoints") / exp
        splits_dir = Path("data/splits") / exp
        seeds = sorted({int(p.name.replace("seed", "")) for p in exp_dir.iterdir() if p.name.startswith("seed")})
        per_exp = {t: np.zeros((n, NUM_LABELS[t]), dtype=np.float64) for t in TASKS}
        for s in seeds:
            sp = _build_seed_oof(exp_dir, splits_dir, n, s)
            for t in TASKS:
                per_exp[t] += sp[t]
        for t in TASKS:
            per_exp[t] /= len(seeds)
            accum[t] += w * per_exp[t]
        total_w += w

        # standalone score for this exp
        truth = {t: [r[t] for r in records] for t in TASKS}
        preds_text = []
        for i in range(n):
            row = {"id": records[i].get("id", i)}
            for t in TASKS:
                lid = int(per_exp[t][i].argmax())
                row[t] = LABEL_DOMAINS[t][lid]
            preds_text.append(row)
        proc = apply_constraints_batch(preds_text)
        preds_dict = {t: [r[t] for r in proc] for t in TASKS}
        ws = weighted_score(truth, preds_dict)
        per_task = {t: ws[t] for t in TASKS}
        summary_rows.append({"exp": exp, **per_task, "weighted_score": ws["final_weighted_score"]})

    for t in TASKS:
        accum[t] /= total_w

    # ensemble preds
    truth = {t: [r[t] for r in records] for t in TASKS}
    preds_text = []
    for i in range(n):
        row = {"id": records[i].get("id", i)}
        for t in TASKS:
            lid = int(accum[t][i].argmax())
            row[t] = LABEL_DOMAINS[t][lid]
        preds_text.append(row)
    proc = apply_constraints_batch(preds_text)
    preds_dict = {t: [r[t] for r in proc] for t in TASKS}
    ws = weighted_score(truth, preds_dict)
    summary_rows.append({"exp": "ENSEMBLE", **{t: ws[t] for t in TASKS}, "weighted_score": ws["final_weighted_score"]})

    df = pd.DataFrame(summary_rows)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = "_".join(args.exps)
    csv = out / f"{tag}.csv"
    df.to_csv(csv, index=False)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print(f"[wrote] {csv}")


if __name__ == "__main__":
    main()
