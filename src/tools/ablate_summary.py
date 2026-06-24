"""Aggregate Phase 2 ablation results into one comparison table.

Reads reports/experiments/{exp}/score_summary.csv for each `p2*` exp, groups by
seed, and writes:
  - reports/experiments/_phase2_ablation.csv
  - reports/experiments/_phase2_ablation.md
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path("reports/experiments")
OUT_CSV = ROOT / "_phase2_ablation.csv"
OUT_MD = ROOT / "_phase2_ablation.md"

BASELINE_EXP = "p1_baseline_macbert_base"


def main() -> None:
    rows = []
    candidates = sorted([p for p in ROOT.iterdir() if p.is_dir() and (p.name.startswith("p2") or p.name == BASELINE_EXP)])
    for d in candidates:
        f = d / "score_summary.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f)
        if df.empty:
            continue
        agg = df.groupby("seed")["weighted_score"].agg(["mean", "std", "count"]).reset_index()
        for _, r in agg.iterrows():
            rows.append({
                "exp": d.name,
                "seed": int(r["seed"]),
                "mean_weighted": float(r["mean"]),
                "std": float(r["std"]) if pd.notna(r["std"]) else 0.0,
                "n_folds": int(r["count"]),
                "f1_T1": float(df[df["seed"] == r["seed"]]["f1_promise_status"].mean()),
                "f1_T2": float(df[df["seed"] == r["seed"]]["f1_verification_timeline"].mean()),
                "f1_T3": float(df[df["seed"] == r["seed"]]["f1_evidence_status"].mean()),
                "f1_T4": float(df[df["seed"] == r["seed"]]["f1_evidence_quality"].mean()),
            })
    if not rows:
        print("[ablate] no results found")
        return
    out = pd.DataFrame(rows).sort_values(["exp", "seed"]).reset_index(drop=True)
    out.to_csv(OUT_CSV, index=False)

    # Find baseline reference (seed=42) and compute delta
    baseline = out[(out["exp"] == BASELINE_EXP) & (out["seed"] == 42)]
    if not baseline.empty:
        ref = float(baseline.iloc[0]["mean_weighted"])
        out["delta_vs_p1_seed42"] = out["mean_weighted"] - ref

    OUT_MD.write_text(
        "# Phase 2 Ablation Summary\n\n"
        + (f"Baseline (p1, seed=42): **{ref:.5f}**\n\n" if not baseline.empty else "")
        + out.to_markdown(index=False, floatfmt=".5f"),
        encoding="utf-8",
    )
    print(out.to_string(index=False))
    print(f"\n[wrote] {OUT_CSV}")
    print(f"[wrote] {OUT_MD}")


if __name__ == "__main__":
    main()
