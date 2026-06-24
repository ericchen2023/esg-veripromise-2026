"""U12 — OOF cross-fold variance / gap analysis (read-only diagnostic).

For each active v12 ensemble member, parse its per-fold score_summary.csv
to compute:

  - per-fold weighted-score (using competition weights 0.20/0.15/0.30/0.35).
  - cross-fold mean / std / range of weighted-score and per-task macro-F1.
  - "fold-fragility": (max - min) / mean — high values indicate the member is
    very sensitive to the specific train/val split.
  - per-task min/max excursion vs that member's fold-mean — flags T2/T4
    instability that would survive into the valid set.

Also computes ensemble-level mean per-task drift across members (i.e., for
each task, the std of per-fold per-task macro-F1 averaged across members),
which serves as a coarse "expected OOF→valid degradation budget" — useful
to interpret the §17.20 GroupKFold leakage finding.

This script reads only `reports/experiments/<exp>/score_summary.csv` and
the v12 active member list from `joint_hillclimb_v12_meta.json`. No model
inference, no retraining.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

V12_META = Path("reports/analysis/_ensemble/joint_hillclimb_v12_meta.json")
EXP_ROOT = Path("reports/experiments")
OUT_DIR = Path("reports/analysis/diagnostics")
TASK_WEIGHTS = {
    "promise_status": 0.20,
    "verification_timeline": 0.15,
    "evidence_status": 0.30,
    "evidence_quality": 0.35,
}
TASKS = list(TASK_WEIGHTS.keys())


def _load_active_members() -> list[str]:
    meta = json.loads(V12_META.read_text(encoding="utf-8"))
    members = list(meta.get("members", []))
    best_w = meta.get("best_w") or meta.get("per_task_w") or {}
    if best_w:
        active = []
        for i, m in enumerate(members):
            non_zero = any(float(best_w[t][i]) > 0 for t in TASKS)
            if non_zero:
                active.append(m)
        return active
    return members


def _load_score_summary(exp: str) -> pd.DataFrame | None:
    p = EXP_ROOT / exp / "score_summary.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    return df


def _compute_per_fold(df: pd.DataFrame) -> pd.DataFrame:
    cols = df.columns.tolist()
    # Heuristic detection of per-task macro-F1 columns
    candidates = {
        "promise_status": ["promise_status_macro_f1", "promise_macro_f1", "macro_f1_promise_status"],
        "verification_timeline": ["verification_timeline_macro_f1", "timeline_macro_f1", "macro_f1_verification_timeline"],
        "evidence_status": ["evidence_status_macro_f1", "evi_status_macro_f1", "macro_f1_evidence_status"],
        "evidence_quality": ["evidence_quality_macro_f1", "evi_quality_macro_f1", "macro_f1_evidence_quality"],
    }
    rename = {}
    # Auto-detect from the file's columns
    for task, options in candidates.items():
        for opt in options:
            if opt in cols:
                rename[opt] = task
                break
    if len(rename) < 4:
        # Fallback: use last 4 numeric columns or known schema
        # The schema seen: exp,seed,fold,best_epoch,weighted,promise,verification,evidence,evidence_quality,...
        # 5th col onward are: weighted, t1, t2, t3, t4
        numeric_cols = [c for c in cols if df[c].dtype.kind in {"f", "i"}]
        if len(numeric_cols) >= 6:
            # take cols index 4,5,6,7,8 by header position (skipping exp,seed,fold,best_epoch)
            after_meta = [c for c in cols if c not in {"exp", "seed", "fold", "best_epoch", "best_pt", "oof", "n_train", "n_val"}]
            after_meta_numeric = [c for c in after_meta if df[c].dtype.kind in {"f", "i"}]
            if len(after_meta_numeric) >= 5:
                df = df.rename(columns={
                    after_meta_numeric[0]: "weighted",
                    after_meta_numeric[1]: "promise_status",
                    after_meta_numeric[2]: "verification_timeline",
                    after_meta_numeric[3]: "evidence_status",
                    after_meta_numeric[4]: "evidence_quality",
                })
                return df
    if rename:
        df = df.rename(columns=rename)
    # Re-derive weighted if missing
    if "weighted" not in df.columns:
        df["weighted"] = sum(
            float(TASK_WEIGHTS[t]) * df[t] for t in TASKS if t in df.columns
        )
    return df


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    members = _load_active_members()
    print(f"[u12] active v12 members: {len(members)}")

    rows: list[dict] = []
    fold_rows: list[dict] = []
    skipped: list[str] = []
    for m in members:
        df = _load_score_summary(m)
        if df is None:
            skipped.append(m)
            print(f"  [skip] {m} (no score_summary.csv)")
            continue
        df = _compute_per_fold(df)
        df = df.sort_values("fold")
        if not all(t in df.columns for t in TASKS):
            skipped.append(m)
            print(f"  [skip] {m} (missing task columns after rename: {df.columns.tolist()})")
            continue

        per_fold = df["weighted"].astype(float).to_numpy()
        rec = {
            "member": m,
            "n_folds": int(len(df)),
            "weighted_mean": float(np.mean(per_fold)),
            "weighted_std": float(np.std(per_fold, ddof=0)),
            "weighted_min": float(np.min(per_fold)),
            "weighted_max": float(np.max(per_fold)),
            "weighted_range": float(np.max(per_fold) - np.min(per_fold)),
            "fragility": float((np.max(per_fold) - np.min(per_fold)) / max(np.mean(per_fold), 1e-9)),
        }
        for t in TASKS:
            arr = df[t].astype(float).to_numpy()
            rec[f"{t}_mean"] = float(np.mean(arr))
            rec[f"{t}_std"] = float(np.std(arr, ddof=0))
            rec[f"{t}_range"] = float(np.max(arr) - np.min(arr))
        rows.append(rec)
        for _, r in df.iterrows():
            fold_rows.append({
                "member": m,
                "fold": int(r["fold"]),
                "weighted": float(r["weighted"]),
                **{t: float(r[t]) for t in TASKS},
            })

    res = pd.DataFrame(rows).sort_values("fragility", ascending=False)
    fold_df = pd.DataFrame(fold_rows)
    res_path = OUT_DIR / "u12_oof_fold_variance.csv"
    fold_path = OUT_DIR / "u12_oof_fold_perfold.csv"
    res.to_csv(res_path, index=False)
    fold_df.to_csv(fold_path, index=False)
    print(f"[wrote] {res_path}")
    print(f"[wrote] {fold_path}")

    # ---- Aggregate across active pool ----
    pool_summary = {
        "n_members_with_summary": int(len(rows)),
        "n_skipped": int(len(skipped)),
        "skipped": skipped,
        "weighted_std_mean": float(np.mean([r["weighted_std"] for r in rows])) if rows else None,
        "weighted_std_max": float(np.max([r["weighted_std"] for r in rows])) if rows else None,
        "weighted_range_mean": float(np.mean([r["weighted_range"] for r in rows])) if rows else None,
        "weighted_range_max": float(np.max([r["weighted_range"] for r in rows])) if rows else None,
        "per_task_std_mean": {t: float(np.mean([r[f"{t}_std"] for r in rows])) if rows else None for t in TASKS},
        "per_task_std_max": {t: float(np.max([r[f"{t}_std"] for r in rows])) if rows else None for t in TASKS},
        "per_task_range_mean": {t: float(np.mean([r[f"{t}_range"] for r in rows])) if rows else None for t in TASKS},
        "fragility_top5": res.head(5)[["member", "fragility", "weighted_mean", "weighted_std"]].to_dict(orient="records"),
    }

    # Decision rule: if any member's fragility >= 0.05 OR T4 std >= 0.04
    # → flag as "expect non-negligible OOF→valid drift"
    fragile = [r["member"] for r in rows if r["fragility"] >= 0.05]
    t4_unstable = [r["member"] for r in rows if r["evidence_quality_std"] >= 0.04]
    pool_summary["fragile_members_ge0_05"] = fragile
    pool_summary["t4_unstable_members_ge0_04"] = t4_unstable

    expected_drift = {
        "T1_mean_std": pool_summary["per_task_std_mean"]["promise_status"],
        "T2_mean_std": pool_summary["per_task_std_mean"]["verification_timeline"],
        "T3_mean_std": pool_summary["per_task_std_mean"]["evidence_status"],
        "T4_mean_std": pool_summary["per_task_std_mean"]["evidence_quality"],
    }
    weighted_expected = (
        0.20 * expected_drift["T1_mean_std"]
        + 0.15 * expected_drift["T2_mean_std"]
        + 0.30 * expected_drift["T3_mean_std"]
        + 0.35 * expected_drift["T4_mean_std"]
    )
    pool_summary["weighted_expected_per_member_std"] = float(weighted_expected)
    # Heuristic: ensemble averaging shrinks std by sqrt(n_members)
    n_eff = max(1, pool_summary["n_members_with_summary"])
    pool_summary["weighted_expected_ensemble_std"] = float(weighted_expected / np.sqrt(n_eff))
    pool_summary["interpretation"] = (
        "OOF→valid drift budget per single member ≈ {:.4f}; ensemble averaging shrinks "
        "to ≈ {:.4f}. If actual valid gap exceeds ~3× this budget, suspect distribution "
        "shift (likely from §17.20 row-leakage 99.93%). U1-c +0.000463 must therefore "
        "be re-validated on official valid before promotion to submission."
    ).format(weighted_expected, pool_summary["weighted_expected_ensemble_std"])

    out_json = OUT_DIR / "u12_oof_fold_variance.json"
    out_json.write_text(json.dumps(pool_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[wrote] {out_json}")
    print(f"[u12] interpretation: {pool_summary['interpretation']}")

    # Compact markdown summary
    lines = ["# U12 — OOF cross-fold variance (read-only)\n\n"]
    lines.append(f"- members analysed: {pool_summary['n_members_with_summary']} (skipped {len(skipped)})\n")
    lines.append(f"- mean per-member weighted-score std across 5 folds: **{pool_summary['weighted_std_mean']:.4f}**\n")
    lines.append(f"- max per-member weighted-score std: {pool_summary['weighted_std_max']:.4f}\n")
    lines.append(f"- weighted_expected per-member std budget: **{pool_summary['weighted_expected_per_member_std']:.4f}**\n")
    lines.append(f"- weighted_expected ensemble std budget (÷√N): **{pool_summary['weighted_expected_ensemble_std']:.4f}**\n\n")
    lines.append("## Per-task std mean (across active members)\n\n")
    lines.append("| task | mean std | max std | mean range |\n| :-- | --: | --: | --: |\n")
    for t in TASKS:
        lines.append(
            f"| {t} | {pool_summary['per_task_std_mean'][t]:.4f} | {pool_summary['per_task_std_max'][t]:.4f} | {pool_summary['per_task_range_mean'][t]:.4f} |\n"
        )
    lines.append("\n## Top-5 most fragile members (highest (max−min)/mean)\n\n")
    lines.append("| member | fragility | weighted mean | weighted std |\n| :-- | --: | --: | --: |\n")
    for r in pool_summary["fragility_top5"]:
        lines.append(f"| {r['member']} | {r['fragility']:.4f} | {r['weighted_mean']:.4f} | {r['weighted_std']:.4f} |\n")
    lines.append(f"\n## Interpretation\n\n{pool_summary['interpretation']}\n")
    out_md = OUT_DIR / "u12_oof_fold_variance.md"
    out_md.write_text("".join(lines), encoding="utf-8")
    print(f"[wrote] {out_md}")


if __name__ == "__main__":
    main()
