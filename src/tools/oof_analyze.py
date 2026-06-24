"""Aggregate OOF predictions across all (seed, fold) of one experiment and
produce diagnostic artifacts:

  - reports/analysis/{exp}/oof_per_seed_fold.csv         per-fold scores
  - reports/analysis/{exp}/oof_aggregate_seed_avg.csv    seed-averaged probs scoring
  - reports/analysis/{exp}/confusion_{seed}_{task}.csv   per-seed confusion matrices
  - reports/analysis/{exp}/confusion_seedavg_{task}.csv  seed-averaged confusion
  - reports/analysis/{exp}/classification_report_seedavg.json
  - reports/analysis/{exp}/error_inventory.csv           per-sample errors w/ probs
  - reports/analysis/{exp}/calibration_{task}.csv        probability calibration bins
  - reports/analysis/{exp}/summary.md                    human-readable summary

Hierarchical post-processing IS applied before scoring (matches submission).

Usage:
    python -m src.tools.oof_analyze --exp p1_baseline_macbert_base
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from src.config import load_config
from src.data.dataset import ID2LABEL, LABEL_DOMAINS, NUM_LABELS, TASKS
from src.data.loader import load_dataset
from src.eval.metrics import FIELD_WEIGHTS, weighted_score
from src.inference.post_process import apply_constraints_batch


def _load_one_oof(npz_path: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    z = np.load(npz_path, allow_pickle=False)
    indices = z["indices"].astype(np.int64)
    probs = {t: z[f"probs_{t}"].astype(np.float32) for t in TASKS}
    return indices, probs


def _load_split_map(splits_dir: Path, seed: int) -> dict[int, list[int]]:
    """Return {fold_index: list of global val indices} from saved split JSON."""
    p = splits_dir / f"seed{seed}.json"
    payload = json.loads(p.read_text(encoding="utf-8"))
    return {int(f["fold"]): [int(i) for i in f["val_idx"]] for f in payload["folds"]}


def _build_seed_oof(
    exp_dir: Path, splits_dir: Path, n: int
) -> dict[int, dict[str, np.ndarray]]:
    """For each seed, assemble full N×C probability matrix from its 5 folds.

    Backward compatible: if npz `indices` look like local (max < fold-size),
    remap using saved splits/seed{S}.json.
    """
    seed_dirs = sorted([p for p in exp_dir.iterdir() if p.is_dir() and p.name.startswith("seed")])
    out: dict[int, dict[str, np.ndarray]] = {}
    for sd in seed_dirs:
        seed = int(sd.name.replace("seed", ""))
        split_map = _load_split_map(splits_dir, seed)
        full = {t: np.zeros((n, NUM_LABELS[t]), dtype=np.float32) for t in TASKS}
        seen = np.zeros(n, dtype=bool)
        for fold_dir in sorted(sd.iterdir()):
            m = fold_dir.name
            if not m.startswith("fold"):
                continue
            fold = int(m.replace("fold", ""))
            npz = fold_dir / "oof_probs.npz"
            if not npz.exists():
                continue
            idx, probs = _load_one_oof(npz)
            global_val = np.asarray(split_map[fold], dtype=np.int64)
            # Detect legacy (local) indexing: indices are 0..len(global_val)-1
            if idx.max() < len(global_val) and set(idx.tolist()) == set(range(len(global_val))):
                # Reorder probs by local idx then map to global
                order = np.argsort(idx)
                target_global = global_val  # local i corresponds to val_records[i] = records[global_val[i]]
                for t in TASKS:
                    full[t][target_global] = probs[t][order]
                seen[target_global] = True
            else:
                for t in TASKS:
                    full[t][idx] = probs[t]
                seen[idx] = True
        if not seen.all():
            missing = int((~seen).sum())
            print(f"[warn] seed={seed} missing {missing}/{n} OOF samples")
        out[seed] = full
    return out


def _argmax_to_text(probs_per_task: dict[str, np.ndarray]) -> dict[str, list[str]]:
    return {
        t: [ID2LABEL[t][int(i)] for i in probs_per_task[t].argmax(axis=1)]
        for t in TASKS
    }


def _post_processed_preds(
    base_records: list[dict], probs_per_task: dict[str, np.ndarray]
) -> dict[str, list[str]]:
    """Apply hierarchical constraints to argmax predictions and return per-task lists."""
    raw = _argmax_to_text(probs_per_task)
    recs = []
    for i, r in enumerate(base_records):
        nr = {k: r.get(k) for k in ("id", "promise_string", "evidence_string")}
        for t in TASKS:
            nr[t] = raw[t][i]
        recs.append(nr)
    recs = apply_constraints_batch(recs)
    return {t: [r[t] for r in recs] for t in TASKS}


def _save_confusion(y_true, y_pred, labels, path: Path) -> pd.DataFrame:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    df = pd.DataFrame(cm, index=[f"true_{l}" for l in labels], columns=[f"pred_{l}" for l in labels])
    df.to_csv(path, encoding="utf-8")
    return df


def _calibration(probs: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Reliability bins on top-1 confidence."""
    conf = probs.max(axis=1)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (conf >= lo) & (conf < hi if i < n_bins - 1 else conf <= hi)
        n = int(mask.sum())
        if n == 0:
            rows.append({"bin": f"[{lo:.1f},{hi:.1f})", "n": 0, "avg_conf": 0.0, "accuracy": 0.0})
        else:
            rows.append({
                "bin": f"[{lo:.1f},{hi:.1f})",
                "n": n,
                "avg_conf": float(conf[mask].mean()),
                "accuracy": float(correct[mask].mean()),
            })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", required=True)
    parser.add_argument("--config", default=None,
                        help="Optional config path; defaults to configs/exp_{exp}.yaml")
    args = parser.parse_args()

    exp = args.exp
    cfg_path = args.config or f"configs/exp_{exp.replace('p1_baseline_macbert_base','p1_baseline')}.yaml"
    if not Path(cfg_path).exists():
        cfg_path = f"configs/exp_p1_baseline.yaml"  # fallback
    cfg = load_config(cfg_path)

    csv_path = cfg["data"]["csv_path"]
    records, df = load_dataset(csv_path)
    n = len(records)
    print(f"[oof] exp={exp} N={n}")

    exp_dir = Path("outputs/checkpoints") / exp
    splits_dir = Path("data/splits") / exp
    out_dir = Path("reports/analysis") / exp
    out_dir.mkdir(parents=True, exist_ok=True)

    seed_oof = _build_seed_oof(exp_dir, splits_dir, n)
    seeds = sorted(seed_oof.keys())
    print(f"[oof] seeds={seeds}")

    # ---- Per-seed scoring -----------------------------------------------
    truth = {t: [r[t] for r in records] for t in TASKS}
    per_seed_rows = []
    seed_pred_text: dict[int, dict[str, list[str]]] = {}
    for seed in seeds:
        preds = _post_processed_preds(records, seed_oof[seed])
        seed_pred_text[seed] = preds
        sc = weighted_score(truth, preds)
        row = {"seed": seed, **{f"f1_{t}": sc[t] for t in TASKS},
               "weighted_score": sc["final_weighted_score"]}
        per_seed_rows.append(row)
        # confusion
        for t in TASKS:
            _save_confusion(truth[t], preds[t], LABEL_DOMAINS[t],
                            out_dir / f"confusion_seed{seed}_{t}.csv")
    per_seed_df = pd.DataFrame(per_seed_rows)
    per_seed_df.to_csv(out_dir / "oof_per_seed.csv", index=False)
    print("\n=== Per-seed OOF (post-processed) ===")
    print(per_seed_df.to_string(index=False))

    # ---- Seed-averaged probability ensemble -----------------------------
    avg_probs = {
        t: np.mean([seed_oof[s][t] for s in seeds], axis=0) for t in TASKS
    }
    avg_preds = _post_processed_preds(records, avg_probs)
    avg_sc = weighted_score(truth, avg_preds)
    seedavg_row = {"seed": "seed_avg", **{f"f1_{t}": avg_sc[t] for t in TASKS},
                   "weighted_score": avg_sc["final_weighted_score"]}
    print("\n=== Seed-averaged probability ensemble (post-processed) ===")
    print(json.dumps(seedavg_row, ensure_ascii=False, indent=2))

    agg_df = pd.concat([per_seed_df, pd.DataFrame([seedavg_row])], ignore_index=True)
    agg_df.to_csv(out_dir / "oof_aggregate.csv", index=False)

    # Confusions for seed-avg
    cls_reports = {}
    for t in TASKS:
        _save_confusion(truth[t], avg_preds[t], LABEL_DOMAINS[t],
                        out_dir / f"confusion_seedavg_{t}.csv")
        cr = classification_report(truth[t], avg_preds[t],
                                   labels=LABEL_DOMAINS[t],
                                   output_dict=True, zero_division=0)
        cls_reports[t] = cr
    (out_dir / "classification_report_seedavg.json").write_text(
        json.dumps(cls_reports, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # ---- Calibration ----------------------------------------------------
    for t in TASKS:
        argmax_id = avg_probs[t].argmax(axis=1)
        truth_id = np.array([LABEL_DOMAINS[t].index(v) if v in LABEL_DOMAINS[t] else -1
                             for v in truth[t]], dtype=np.int64)
        correct = (argmax_id == truth_id).astype(np.int64)
        cal = _calibration(avg_probs[t], correct)
        cal.to_csv(out_dir / f"calibration_{t}.csv", index=False)

    # ---- Per-sample error inventory (seed-avg, post-processed) ----------
    err_rows = []
    for i, r in enumerate(records):
        sample_err = {}
        any_wrong = False
        for t in TASKS:
            pred = avg_preds[t][i]
            true = truth[t][i]
            ok = pred == true
            if not ok:
                any_wrong = True
            sample_err[f"{t}_true"] = true
            sample_err[f"{t}_pred"] = pred
            sample_err[f"{t}_ok"] = bool(ok)
            sample_err[f"{t}_top_conf"] = float(avg_probs[t][i].max())
        if any_wrong:
            err_rows.append({
                "id": r.get("id"),
                "company": r.get("company"),
                "esg_type": r.get("esg_type"),
                "text_len": len(str(r.get("data") or "")),
                "text_preview": str(r.get("data") or "")[:120].replace("\n", " "),
                **sample_err,
            })
    err_df = pd.DataFrame(err_rows)
    err_df.to_csv(out_dir / "error_inventory.csv", index=False, encoding="utf-8-sig")

    # ---- Error pattern aggregation ---------------------------------------
    patterns = []
    for t in TASKS:
        pair_ctr: Counter = Counter()
        for i, r in enumerate(records):
            true = truth[t][i]
            pred = avg_preds[t][i]
            if true != pred:
                pair_ctr[(true, pred)] += 1
        for (tr, pr), c in pair_ctr.most_common():
            patterns.append({"task": t, "true": tr, "pred": pr, "count": c})
    pat_df = pd.DataFrame(patterns)
    pat_df.to_csv(out_dir / "error_patterns.csv", index=False, encoding="utf-8")

    # ---- Markdown summary ------------------------------------------------
    md = []
    md.append(f"# OOF Analysis — `{exp}`\n")
    md.append(f"- Records: **{n}** | Seeds: {seeds} | Tasks: {list(TASKS)}\n")
    md.append("\n## Per-seed (post-processed)\n")
    md.append(per_seed_df.to_markdown(index=False, floatfmt=".4f"))
    md.append("\n\n## Seed-averaged probability ensemble\n")
    md.append(pd.DataFrame([seedavg_row]).to_markdown(index=False, floatfmt=".4f"))
    md.append("\n\n## Per-class F1 (seed-avg)\n")
    rows = []
    for t in TASKS:
        for lab in LABEL_DOMAINS[t]:
            cr = cls_reports[t].get(lab, {})
            rows.append({"task": t, "label": lab,
                         "support": int(cr.get("support", 0)),
                         "precision": float(cr.get("precision", 0.0)),
                         "recall": float(cr.get("recall", 0.0)),
                         "f1": float(cr.get("f1-score", 0.0))})
    md.append(pd.DataFrame(rows).to_markdown(index=False, floatfmt=".4f"))
    md.append("\n\n## Top error transitions (seed-avg)\n")
    md.append(pat_df.head(30).to_markdown(index=False))
    md.append("\n\n## Calibration (seed-avg, top-1 confidence)\n")
    for t in TASKS:
        md.append(f"\n### {t}\n")
        md.append(pd.read_csv(out_dir / f"calibration_{t}.csv").to_markdown(
            index=False, floatfmt=".4f"))
    md.append(f"\n\n## Files\n")
    md.append("- `oof_per_seed.csv`, `oof_aggregate.csv`\n")
    md.append("- `confusion_seed{S}_{task}.csv`, `confusion_seedavg_{task}.csv`\n")
    md.append("- `classification_report_seedavg.json`\n")
    md.append("- `error_inventory.csv`, `error_patterns.csv`\n")
    md.append("- `calibration_{task}.csv`\n")

    (out_dir / "summary.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\n[wrote] {out_dir}/summary.md")
    print(f"[wrote] {out_dir}/error_inventory.csv  ({len(err_df)} samples with >=1 wrong task)")


if __name__ == "__main__":
    main()
