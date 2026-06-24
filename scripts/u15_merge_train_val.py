"""U15 Merge train + val datasets — Phase 41 (2026-06-05).

Concatenates the official training set (1000 rows) with the official
validation set (1000 rows, released 2026-06-03) to produce a single combined
dataset (≤2000 rows) for final-submission retraining.

The val set uses 'more_than_5_years' for verification_timeline; loader.py
normalises it to 'longer_than_5_years' automatically via _LABEL_ALIASES,
so the combined output is consistent with existing training code.

Outputs:
  data/processed/train_val_combined.csv   — merged, index-reset CSV
  (also prints per-task distribution report to stdout)

Usage:
    python -m scripts.u15_merge_train_val
    python -m scripts.u15_merge_train_val --train data/raw/vpesg4k_train_1000 V1.csv
    python -m scripts.u15_merge_train_val --val data/raw/vpesg4k_val_1000.csv
    python -m scripts.u15_merge_train_val --out data/processed/train_val_combined.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.data.dataset import TASKS
from src.data.loader import load_dataset

DEFAULT_TRAIN = Path("data/raw/vpesg4k_train_1000 V1.csv")
DEFAULT_VAL = Path("data/raw/vpesg4k_val_1000.csv")
DEFAULT_OUT = Path("data/processed/train_val_combined.csv")


def _distribution_table(df: pd.DataFrame) -> str:
    lines = []
    for task in TASKS:
        if task not in df.columns:
            continue
        counts = df[task].value_counts().sort_index()
        lines.append(f"  {task}:")
        for label, cnt in counts.items():
            lines.append(f"    {label:30s}: {cnt:5d}  ({cnt/len(df)*100:.1f}%)")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="U15 merge train+val datasets (Phase 41)")
    ap.add_argument("--train", default=str(DEFAULT_TRAIN), help="Train CSV path")
    ap.add_argument("--val", default=str(DEFAULT_VAL), help="Val CSV path")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="Output CSV path")
    args = ap.parse_args()

    train_path = Path(args.train)
    val_path = Path(args.val)
    out_path = Path(args.out)

    if not train_path.exists():
        raise FileNotFoundError(f"Training data not found: {train_path}")
    if not val_path.exists():
        raise FileNotFoundError(f"Validation data not found: {val_path}")

    print(f"[u15] loading train: {train_path}")
    _, train_df = load_dataset(str(train_path))
    print(f"[u15] loading val:   {val_path}")
    _, val_df = load_dataset(str(val_path))

    n_train = len(train_df)
    n_val = len(val_df)
    print(f"[u15] train rows: {n_train}")
    print(f"[u15] val rows:   {n_val}")

    combined = pd.concat([train_df, val_df], ignore_index=True)
    combined = combined.reset_index(drop=True)
    n_combined = len(combined)
    print(f"[u15] combined rows: {n_combined}")

    # Sanity check: no duplicate IDs
    if "id" in combined.columns:
        dup_ids = combined["id"].duplicated().sum()
        if dup_ids:
            print(f"  [warn] {dup_ids} duplicate id values in combined set")
        else:
            print("  [ok] all IDs unique in combined set")

    # Distribution report
    print("\n--- Train distribution ---")
    print(_distribution_table(train_df))
    print("\n--- Val distribution ---")
    print(_distribution_table(val_df))
    print("\n--- Combined distribution ---")
    print(_distribution_table(combined))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_path, index=False)
    print(f"\n[u15] wrote {n_combined} rows to: {out_path}")
    print("[u15] IMPORTANT: 'more_than_5_years' in val was normalised to 'longer_than_5_years'")
    print("[u15] Train kfold with any of the exp_*_tv.yaml configs will use this combined file.")


if __name__ == "__main__":
    main()
