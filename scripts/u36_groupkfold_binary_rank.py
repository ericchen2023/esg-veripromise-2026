"""U36 — Phase 53: rank binary models by honest GroupKFold (held-out company) CV.

The standard StratifiedKFold OOF is untrustworthy for the binary tasks because
all 50 companies appear in every fold's train split (within-company leakage),
and 40+ phases have adaptively overfit it. GroupKFold holds out whole companies,
so a model that still scores well on T1/T3 there is genuinely generalising.

IMPORTANT (measured, MASTER_PLAN §69.3): the TEST set reuses the SAME 50
companies as train+val (0 new companies). Holding out a whole company is
therefore HARDER than the test condition, so GroupKFold binary F1 is a
conservative LOWER BOUND, not a test proxy. Use it to RANK models relatively
(which backbone/recipe generalises best on binary) and to avoid uploading
models that collapse on unseen companies -- not to predict the absolute LB.

Reads reports/experiments/<exp>_gkf/score_summary.json (written by
`python -m src.train_kfold --config <cfg> --split-mode stratified_group_kfold`)
and prints a per-model table of held-out-company binary F1.

Usage:
    python -m scripts.u36_groupkfold_binary_rank \
        p2_combo_best p49_tapt_combo_best p54_tapt_robertalarge
    # (names are exp_name WITHOUT the _gkf suffix; the suffix is added here)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

WEIGHTS = {"promise_status": 0.20, "verification_timeline": 0.15,
           "evidence_status": 0.30, "evidence_quality": 0.35}
BINARY = ("promise_status", "evidence_status")


def read_summary(exp: str) -> dict | None:
    p = Path("reports/experiments") / f"{exp}_gkf" / "score_summary.json"
    if not p.exists():
        return None
    rows = json.loads(p.read_text(encoding="utf-8"))
    if not rows:
        return None
    tasks = list(WEIGHTS)
    means = {t: sum(r[f"f1_{t}"] for r in rows) / len(rows) for t in tasks}
    means["weighted"] = sum(means[t] * WEIGHTS[t] for t in tasks)
    means["_n"] = len(rows)
    return means


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("exps", nargs="+", help="exp_name(s) without the _gkf suffix")
    args = ap.parse_args()

    print(f"{'model':40s} {'T1':>7} {'T3':>7} {'bin_wF1':>8} {'T2':>7} {'T4':>7} {'weighted':>9} {'folds':>6}")
    print("-" * 96)
    ranked = []
    for exp in args.exps:
        m = read_summary(exp)
        if m is None:
            print(f"{exp:40s} {'-- no _gkf score_summary.json (run GroupKFold first) --':>50}")
            continue
        bin_wf1 = (m["promise_status"] * 0.20 + m["evidence_status"] * 0.30) / 0.50
        ranked.append((bin_wf1, exp, m))
        print(f"{exp:40s} {m['promise_status']:7.4f} {m['evidence_status']:7.4f} "
              f"{bin_wf1:8.4f} {m['verification_timeline']:7.4f} {m['evidence_quality']:7.4f} "
              f"{m['weighted']:9.4f} {m['_n']:6d}")

    if ranked:
        ranked.sort(reverse=True)
        print("\n[ranking by held-out-company binary weighted-F1 (T1*0.2+T3*0.3)/0.5]")
        for i, (b, exp, _) in enumerate(ranked, 1):
            print(f"  {i}. {exp}  bin_wF1={b:.4f}")
        print("\nPick the top-1/2 binary backbones for the T1/T3 ensemble (scripts/u35).")
        print("A model that wins here is robust on unseen companies -> trustworthy LB probe.")


if __name__ == "__main__":
    main()
