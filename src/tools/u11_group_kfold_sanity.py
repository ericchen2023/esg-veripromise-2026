"""U11 — GroupKFold sanity check (validation only, no retrain).

Compare the 5-Fold StratifiedKFold splits used by every active member with
StratifiedGroupKFold(group=company) on the same data, then quantify:

  1. Company-level concentration (how many samples per company).
  2. Per-fold leakage: % of val companies that also appear in train under
     plain StratifiedKFold (high = potential train/val leakage).
  3. Per-fold label distribution drift between the two split modes (max
     |delta| of class proportion).
  4. Per-fold size and per-task macro label coverage.

Outputs a JSON to ``reports/analysis/diagnostics/u11_group_kfold_sanity.json``
and a Markdown summary to the same folder.

This script does NOT retrain anything; it only inspects splits / data.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import load_config
from src.data.dataset import LABEL_DOMAINS, TASKS
from src.data.loader import load_dataset
from src.data.splits import make_folds


CFG = Path("configs/base.yaml")
DATA_PATH = Path("data/raw/vpesg4k_train_1000 V1.csv")
OUT_DIR = Path("reports/analysis/diagnostics")
SEEDS = [42, 2024, 20260417]
N_SPLITS = 5
STRATIFY_FIELDS = ["promise_status", "evidence_status"]
GROUP_FIELD = "company"


def _label_dist(df: pd.DataFrame, idx: np.ndarray, task: str) -> dict[str, float]:
    sub = df.iloc[idx][task].astype(str).value_counts(normalize=True).to_dict()
    domain = LABEL_DOMAINS[task]
    return {label: float(sub.get(label, 0.0)) for label in domain}


def _max_delta(d1: dict[str, float], d2: dict[str, float]) -> float:
    return max(abs(d1[k] - d2[k]) for k in d1)


def _fold_company_leakage(df: pd.DataFrame, train_idx: np.ndarray, val_idx: np.ndarray) -> dict:
    train_companies = set(df.iloc[train_idx][GROUP_FIELD].astype(str))
    val_companies = set(df.iloc[val_idx][GROUP_FIELD].astype(str))
    leaked = val_companies & train_companies
    val_rows_leaked = int(df.iloc[val_idx][GROUP_FIELD].astype(str).isin(leaked).sum())
    return {
        "n_val_companies": len(val_companies),
        "n_train_companies": len(train_companies),
        "n_leaked_companies": len(leaked),
        "leak_company_ratio": len(leaked) / max(1, len(val_companies)),
        "n_val_rows": int(len(val_idx)),
        "n_val_rows_leaked": val_rows_leaked,
        "leak_row_ratio": val_rows_leaked / max(1, len(val_idx)),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    records, df = load_dataset(DATA_PATH)
    n = len(df)
    print(f"[u11] N={n}")
    assert GROUP_FIELD in df.columns, f"missing column {GROUP_FIELD}"

    # 1) Company concentration
    company_counts = df[GROUP_FIELD].astype(str).value_counts()
    company_summary = {
        "n_unique_companies": int(company_counts.shape[0]),
        "min_count": int(company_counts.min()),
        "max_count": int(company_counts.max()),
        "median_count": float(company_counts.median()),
        "mean_count": float(company_counts.mean()),
        "top_10": company_counts.head(10).to_dict(),
        "n_singletons": int((company_counts == 1).sum()),
        "n_companies_ge5": int((company_counts >= 5).sum()),
        "rows_in_companies_ge5": int(company_counts[company_counts >= 5].sum()),
    }
    print(f"[u11] companies: {company_summary['n_unique_companies']} unique, "
          f"max={company_summary['max_count']} median={company_summary['median_count']}")

    overall = {
        "n_records": n,
        "stratify_fields": STRATIFY_FIELDS,
        "group_field": GROUP_FIELD,
        "n_splits": N_SPLITS,
        "seeds": SEEDS,
        "company_summary": company_summary,
        "per_seed": {},
    }

    for seed in SEEDS:
        print(f"\n[u11] === seed={seed} ===")
        sk_folds = make_folds(df, N_SPLITS, STRATIFY_FIELDS, seed, mode="stratified_kfold")
        gk_folds = make_folds(df, N_SPLITS, STRATIFY_FIELDS, seed, mode="stratified_group_kfold", group_field=GROUP_FIELD)

        sk_rows: list[dict] = []
        gk_rows: list[dict] = []
        max_drift_per_task: dict[str, list[float]] = {t: [] for t in TASKS}

        for fold_index in range(N_SPLITS):
            sk_tr, sk_va = sk_folds[fold_index]
            gk_tr, gk_va = gk_folds[fold_index]

            sk_leak = _fold_company_leakage(df, sk_tr, sk_va)
            gk_leak = _fold_company_leakage(df, gk_tr, gk_va)

            sk_dist = {t: _label_dist(df, sk_va, t) for t in TASKS}
            gk_dist = {t: _label_dist(df, gk_va, t) for t in TASKS}
            full_dist = {t: _label_dist(df, np.arange(n), t) for t in TASKS}

            for t in TASKS:
                drift = _max_delta(sk_dist[t], gk_dist[t])
                max_drift_per_task[t].append(drift)

            sk_rows.append({
                "fold": fold_index,
                "n_train": int(len(sk_tr)),
                "n_val": int(len(sk_va)),
                **{f"leak_{k}": v for k, v in sk_leak.items()},
                "val_label_dist": sk_dist,
            })
            gk_rows.append({
                "fold": fold_index,
                "n_train": int(len(gk_tr)),
                "n_val": int(len(gk_va)),
                **{f"leak_{k}": v for k, v in gk_leak.items()},
                "val_label_dist": gk_dist,
            })

        sk_mean_leak_row = float(np.mean([r["leak_leak_row_ratio"] for r in sk_rows]))
        gk_mean_leak_row = float(np.mean([r["leak_leak_row_ratio"] for r in gk_rows]))
        sk_mean_leak_co = float(np.mean([r["leak_leak_company_ratio"] for r in sk_rows]))
        gk_mean_leak_co = float(np.mean([r["leak_leak_company_ratio"] for r in gk_rows]))
        max_drift = {t: float(np.max(max_drift_per_task[t])) for t in TASKS}
        mean_drift = {t: float(np.mean(max_drift_per_task[t])) for t in TASKS}

        overall["per_seed"][str(seed)] = {
            "stratified_kfold": {
                "folds": sk_rows,
                "mean_val_company_leak_ratio": sk_mean_leak_co,
                "mean_val_row_leak_ratio": sk_mean_leak_row,
            },
            "stratified_group_kfold": {
                "folds": gk_rows,
                "mean_val_company_leak_ratio": gk_mean_leak_co,
                "mean_val_row_leak_ratio": gk_mean_leak_row,
            },
            "label_drift": {
                "max_per_task": max_drift,
                "mean_per_task": mean_drift,
            },
            "full_val_label_dist_reference": {t: _label_dist(df, np.arange(n), t) for t in TASKS},
        }

        print(f"[u11] seed={seed}  StratifiedKFold leakage: company={sk_mean_leak_co:.3f}  row={sk_mean_leak_row:.3f}")
        print(f"[u11] seed={seed}  GroupKFold        leakage: company={gk_mean_leak_co:.3f}  row={gk_mean_leak_row:.3f}")
        print(f"[u11] seed={seed}  max label drift between modes: {max_drift}")

    # Decision rule (per §4.2 / §6 historical doc): if max drift >= 0.02 OR row leakage >= 0.50
    # then GroupKFold should at least be considered. The leakage by itself only tells us
    # whether plain StratifiedKFold likely overestimates OOF; the drift bounds tell us whether
    # GroupKFold itself is feasible without hurting class balance.
    sk_leak_avg = float(np.mean([overall["per_seed"][str(s)]["stratified_kfold"]["mean_val_row_leak_ratio"] for s in SEEDS]))
    gk_drift_avg = {
        t: float(np.mean([overall["per_seed"][str(s)]["label_drift"]["mean_per_task"][t] for s in SEEDS]))
        for t in TASKS
    }
    overall["decision_summary"] = {
        "sk_avg_val_row_leak_ratio_across_seeds": sk_leak_avg,
        "gk_avg_label_drift_per_task_across_seeds": gk_drift_avg,
        "leak_threshold": 0.50,
        "drift_threshold": 0.02,
        "verdict": (
            "RETRAIN-WITH-GROUPKFOLD"
            if sk_leak_avg >= 0.50 and max(gk_drift_avg.values()) < 0.04
            else (
                "MONITOR — leakage observed but drift would distort label balance"
                if sk_leak_avg >= 0.50
                else "KEEP-StratifiedKFold (current splits show low company leakage; no retrain)"
            )
        ),
    }
    print(f"\n[u11] DECISION: {overall['decision_summary']['verdict']}")
    print(f"[u11] sk_avg_val_row_leak_ratio = {sk_leak_avg:.4f}")

    # Persist
    out_json = OUT_DIR / "u11_group_kfold_sanity.json"
    out_json.write_text(json.dumps(overall, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[wrote] {out_json}")

    # Compact markdown
    lines = ["# U11 — GroupKFold sanity check (read-only)\n"]
    lines.append(f"- N records = {n}\n- companies = {company_summary['n_unique_companies']} unique, "
                 f"max={company_summary['max_count']}, median={company_summary['median_count']}, "
                 f"singletons={company_summary['n_singletons']}\n")
    lines.append(f"- decision: **{overall['decision_summary']['verdict']}**\n")
    lines.append(f"- avg StratifiedKFold val-row company-leak ratio across seeds = {sk_leak_avg:.4f}\n")
    lines.append("\n## Per-seed leakage and drift\n")
    lines.append("| seed | mode | mean val company leak | mean val row leak |\n| :--: | :--: | --: | --: |\n")
    for seed in SEEDS:
        sk = overall["per_seed"][str(seed)]["stratified_kfold"]
        gk = overall["per_seed"][str(seed)]["stratified_group_kfold"]
        lines.append(f"| {seed} | StratifiedKFold | {sk['mean_val_company_leak_ratio']:.4f} | {sk['mean_val_row_leak_ratio']:.4f} |\n")
        lines.append(f"| {seed} | GroupKFold      | {gk['mean_val_company_leak_ratio']:.4f} | {gk['mean_val_row_leak_ratio']:.4f} |\n")
    lines.append("\n## Mean per-task label drift (StratifiedKFold val vs GroupKFold val)\n")
    lines.append("| seed | T1 | T2 | T3 | T4 |\n| :--: | --: | --: | --: | --: |\n")
    for seed in SEEDS:
        d = overall["per_seed"][str(seed)]["label_drift"]["mean_per_task"]
        lines.append(f"| {seed} | {d['promise_status']:.4f} | {d['verification_timeline']:.4f} | {d['evidence_status']:.4f} | {d['evidence_quality']:.4f} |\n")
    out_md = OUT_DIR / "u11_group_kfold_sanity.md"
    out_md.write_text("".join(lines), encoding="utf-8")
    print(f"[wrote] {out_md}")


if __name__ == "__main__":
    main()
