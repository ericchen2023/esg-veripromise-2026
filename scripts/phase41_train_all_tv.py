"""Phase 41 batch runner — train all 8 Train+Val stems sequentially.

Runs the 8 _tv configs one by one:
  1 × train_kfold          (stem 1: p2_combo_best_tv, no pseudo)
  7 × train_pseudo_kfold   (stems 2–8, with pseudo/aug data)

After each stem completes, validates that the best.pt checkpoint exists for
every fold. Reports per-stem OOF score from score_summary.json if available.

Usage:
    python scripts/phase41_train_all_tv.py
    python scripts/phase41_train_all_tv.py --resume   # skip already-trained stems
    python scripts/phase41_train_all_tv.py --dry-run  # print commands only
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

# Ordered as per AP-D4 stem list
STEMS: list[tuple[str, str, str]] = [
    # (exp_name, config_file, train_script_module)
    (
        "p2_combo_best_tv",
        "configs/exp_p2_combo_best_tv.yaml",
        "src.train_kfold",
    ),
    (
        "p2_combo_best_u10_pseudo_tv",
        "configs/exp_p2_combo_best_u10_pseudo_tv.yaml",
        "src.train_pseudo_kfold",
    ),
    (
        "p2_combo_best_u10_pseudo_v2_tv",
        "configs/exp_p2_combo_best_u10_pseudo_v2_tv.yaml",
        "src.train_pseudo_kfold",
    ),
    (
        "p2_combo_best_u10_pseudo_v2_classw_focal_t4_g3_tv",
        "configs/exp_p2_combo_best_u10_pseudo_v2_classw_focal_t4_g3_tv.yaml",
        "src.train_pseudo_kfold",
    ),
    (
        "p2_combo_best_u10_pseudo_v3_classw_focal_t4_g3_tv",
        "configs/exp_p2_combo_best_u10_pseudo_v3_classw_focal_t4_g3_tv.yaml",
        "src.train_pseudo_kfold",
    ),
    (
        "p2_combo_best_classw_focal_u6pro_tv",
        "configs/exp_p2_combo_best_classw_focal_u6pro_tv.yaml",
        "src.train_pseudo_kfold",
    ),
    (
        "p2_combo_best_aug_plus_tv",
        "configs/exp_p2_combo_best_aug_plus_tv.yaml",
        "src.train_pseudo_kfold",
    ),
    (
        "p2_combo_best_aug_plus_v2_tv",
        "configs/exp_p2_combo_best_aug_plus_v2_tv.yaml",
        "src.train_pseudo_kfold",
    ),
]

N_FOLDS = 5
CKPT_ROOT = Path("outputs/checkpoints")
REPORT_ROOT = Path("reports/experiments")


def _all_checkpoints_present(exp_name: str) -> bool:
    """Return True if all 5 fold checkpoints exist for seed42."""
    for fold in range(N_FOLDS):
        p = CKPT_ROOT / exp_name / "seed42" / f"fold{fold}" / "best.pt"
        if not p.exists():
            return False
    return True


def _read_score_summary(exp_name: str) -> float | None:
    """Return mean OOF weighted_score, or None if summary file missing."""
    p = REPORT_ROOT / exp_name / "score_summary.json"
    if not p.exists():
        return None
    data = json.loads(p.read_text(encoding="utf-8"))
    # data is a list of per-fold dicts: [{"weighted_score": 0.681, ...}, ...]
    if isinstance(data, list) and data:
        scores = [row["weighted_score"] for row in data if "weighted_score" in row]
        return round(sum(scores) / len(scores), 5) if scores else None
    # fallback: older nested-dict format
    if isinstance(data, dict):
        return data.get("overall", {}).get("mean")
    return None


def _print_separator(title: str) -> None:
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 41 batch TV training runner")
    ap.add_argument("--resume", action="store_true",
                    help="Skip stems where all 5 fold checkpoints already exist")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print commands without executing")
    ap.add_argument("--stems", nargs="+", default=None,
                    help="Only run specific exp_names (subset of the 8)")
    args = ap.parse_args()

    stems_to_run = STEMS
    if args.stems:
        stems_to_run = [s for s in STEMS if s[0] in args.stems]
        if not stems_to_run:
            print(f"[error] none of {args.stems} found in STEMS list")
            sys.exit(1)

    print(f"[phase41] {len(stems_to_run)} stems to process")
    print(f"[phase41] mode: {'dry-run' if args.dry_run else 'resume' if args.resume else 'full'}")

    results: list[dict] = []
    total_t0 = time.time()

    for i, (exp_name, config_file, module) in enumerate(stems_to_run, 1):
        _print_separator(f"Stem {i}/{len(stems_to_run)}: {exp_name}")

        if args.resume and _all_checkpoints_present(exp_name):
            print(f"  [skip] all {N_FOLDS} checkpoints already present")
            ws = _read_score_summary(exp_name)
            if ws is not None:
                print(f"  [cached] OOF weighted_score = {ws}")
            results.append({"stem": exp_name, "status": "skipped", "score": ws})
            continue

        cmd = [sys.executable, "-m", module, "--config", config_file]
        print(f"  [cmd] {' '.join(cmd)}")

        if args.dry_run:
            results.append({"stem": exp_name, "status": "dry-run", "score": None})
            continue

        t0 = time.time()
        proc = subprocess.run(cmd, check=False)
        elapsed = time.time() - t0

        if proc.returncode != 0:
            print(f"  [ERROR] exit code {proc.returncode} — aborting batch")
            results.append({"stem": exp_name, "status": f"failed(rc={proc.returncode})", "score": None})
            break

        ckpts_ok = _all_checkpoints_present(exp_name)
        ws = _read_score_summary(exp_name)
        print(f"  [done] elapsed={elapsed:.0f}s  checkpoints_ok={ckpts_ok}  OOF={ws}")
        results.append({
            "stem": exp_name,
            "status": "ok" if ckpts_ok else "incomplete",
            "score": ws,
            "elapsed_s": round(elapsed),
        })

    total_elapsed = time.time() - total_t0
    _print_separator("Phase 41 Summary")
    print(f"Total elapsed: {total_elapsed/3600:.1f}h ({total_elapsed:.0f}s)")
    print(f"{'Stem':<55}  {'Status':<12}  Score")
    print("-" * 80)
    for r in results:
        print(f"  {r['stem']:<53}  {r['status']:<12}  {r['score']}")

    failed = [r for r in results if r["status"].startswith("failed")]
    incomplete = [r for r in results if r["status"] == "incomplete"]
    if failed or incomplete:
        print("\n[WARN] Some stems failed or have incomplete checkpoints:")
        for r in failed + incomplete:
            print(f"  {r['stem']}: {r['status']}")
        sys.exit(1)
    else:
        print("\n[Phase 41] All stems completed successfully.")
        print("Next step: run scripts/u12_val_gap.py to validate, then Phase 42 ensemble re-search.")


if __name__ == "__main__":
    main()
