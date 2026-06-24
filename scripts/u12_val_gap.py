"""U12 OOF↔Val gap analysis — Phase 40 (2026-06-05).

Loads the official validation set (released 2026-06-03) and runs inference
using the AP-D4 trained checkpoints.  Applies the same per-task stem/view
weights from ap_d4_8way_3view_meta.json, then computes the validation score
and compares it against the OOF score of 0.71608.

NOTE: The val set uses 'more_than_5_years' for verification_timeline;
      loader.py normalises this to 'longer_than_5_years' automatically.

Usage:
    python -m scripts.u12_val_gap                          # default paths
    python -m scripts.u12_val_gap --data data/raw/vpesg4k_val_1000.csv
    python -m scripts.u12_val_gap --cpu                    # force CPU inference
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data.dataset import LABEL_DOMAINS, NUM_LABELS, TASKS
from src.data.loader import load_dataset
from src.eval.metrics import FIELD_WEIGHTS, task_score, weighted_score
from src.inference.post_process import apply_constraints_batch
from src.tools.tta_fast_eval import (
    apply_constraints_to_label_ids,
    encode_truth,
    score_label_ids,
)
from src.tools.u1_tta_oof import (
    TTADataset,
    create_model,
    create_tokenizer,
    predict_loader,
    tta_collate,
)
from src.tools.u1_tta_oof import config_for_stem

DEFAULT_VAL = Path("data/raw/vpesg4k_val_1000.csv")
DEFAULT_META = Path("reports/analysis/_ensemble/ap_d4_8way_3view_meta.json")
OUTPUT_DIR = Path("reports/analysis/u12_val_gap")
AP_D4_OOF_SCORE = 0.71608


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def _predict_stem_all_folds(
    stem: str,
    val_records: list[dict[str, Any]],
    *,
    device: torch.device,
    use_amp: bool,
    batch_size: int | None,
) -> dict[str, np.ndarray]:
    """Run all 5 folds (seed42) of a stem on val_records; return 5-fold avg.

    Each fold checkpoint is independent; averaging mimics OOF ensemble but
    applied to the unseen val set.  Uses the 'stored' (head-truncation) view
    to match the OOF training view.
    """
    cfg = config_for_stem(stem)
    tokenizer, text_transform, extra_tokens = create_tokenizer(cfg)
    max_length = int(cfg["model"]["max_length"])
    bs = batch_size or int(cfg["training"].get("batch_size", 8)) * 2
    n = len(val_records)

    fold_accumulator = {task: np.zeros((n, NUM_LABELS[task]), dtype=np.float64) for task in TASKS}
    n_folds_loaded = 0

    for fold_idx in range(5):
        ckpt = (
            Path("outputs/checkpoints") / stem / "seed42" / f"fold{fold_idx}" / "best.pt"
        )
        if not ckpt.exists():
            print(f"  [warn] checkpoint missing: {ckpt} — skipping fold {fold_idx}")
            continue
        dataset = TTADataset(
            val_records,
            list(range(n)),
            tokenizer,
            max_length,
            view="head",
            text_transform=text_transform,
        )
        loader = DataLoader(
            dataset, batch_size=bs, shuffle=False, collate_fn=tta_collate
        )
        model = create_model(cfg, tokenizer, extra_tokens, ckpt, device)
        result = predict_loader(model, loader, device, use_amp)
        for task in TASKS:
            fold_accumulator[task] += result[task]
        n_folds_loaded += 1
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"  [fold {fold_idx}] done", flush=True)

    if n_folds_loaded == 0:
        raise RuntimeError(f"No checkpoints found for stem '{stem}'. Train first.")
    for task in TASKS:
        fold_accumulator[task] /= n_folds_loaded
    print(f"  [{stem}] averaged {n_folds_loaded} folds")
    return fold_accumulator


def _apply_weights(
    per_stem_probs: dict[str, dict[str, np.ndarray]],
    stem_weights: dict[str, list[float]],
    view_alpha: dict[str, list[float]],
    stems: list[str],
    n: int,
) -> dict[str, np.ndarray]:
    """Combine per-stem probs using AP-D4 stem_star weights.

    For val inference we only have one view (stored/head), so we use
    alpha[task][0] as the full weight for the stored view (i.e., ignore
    middle/tail which require additional forward passes).
    """
    mixed: dict[str, np.ndarray] = {}
    for task in TASKS:
        ws = stem_weights[task]
        total = sum(ws)
        if total == 0:
            raise ValueError(f"stem_star weights for {task} sum to 0")
        arr = np.zeros((n, NUM_LABELS[task]), dtype=np.float64)
        for stem, w in zip(stems, ws):
            if w == 0.0 or stem not in per_stem_probs:
                continue
            arr += w * per_stem_probs[stem][task]
        mixed[task] = arr / total
    return mixed


# ---------------------------------------------------------------------------
# Scoring utilities
# ---------------------------------------------------------------------------

def _score_from_probs(
    probs: dict[str, np.ndarray],
    val_records: list[dict[str, Any]],
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """Argmax + constraints + compute weighted score."""
    raw_preds = []
    n = len(val_records)
    for i in range(n):
        row: dict[str, Any] = {"id": val_records[i].get("id", i)}
        for task in TASKS:
            row[task] = LABEL_DOMAINS[task][int(probs[task][i].argmax())]
        raw_preds.append(row)
    constrained = apply_constraints_batch(raw_preds)
    truth = {task: [r[task] for r in val_records] for task in TASKS}
    pred = {task: [r[task] for r in constrained] for task in TASKS}
    return weighted_score(truth, pred), constrained


def _distribution_report(records: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    dist: dict[str, dict[str, int]] = {}
    for task in TASKS:
        counts: dict[str, int] = {}
        for r in records:
            v = str(r.get(task, ""))
            counts[v] = counts.get(v, 0) + 1
        dist[task] = counts
    return dist


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="U12 OOF↔Val gap analysis (Phase 40)")
    ap.add_argument("--data", default=str(DEFAULT_VAL), help="Val CSV path")
    ap.add_argument("--meta", default=str(DEFAULT_META), help="AP-D4 meta JSON path")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--stems", nargs="+", default=None,
                    help="Override AP-D4 stem list (default: read from meta)")
    args = ap.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    use_amp = device.type == "cuda" and not args.no_amp

    meta_path = Path(args.meta)
    if not meta_path.exists():
        raise FileNotFoundError(f"AP-D4 meta not found: {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    stems: list[str] = args.stems or meta["stems"]
    stem_star: dict[str, list[float]] = meta["stem_star"]
    alpha_star: dict[str, list[float]] = meta["alpha_star"]

    print(f"[u12] loading val set from: {args.data}")
    val_records, val_df = load_dataset(args.data)
    n = len(val_records)
    print(f"[u12] N={n} device={device} stems={stems}")

    # Label distribution report
    val_dist = _distribution_report(val_records)
    print("[u12] val label distribution:")
    for task, counts in val_dist.items():
        print(f"  {task}: {counts}")

    t0 = time.time()
    per_stem_probs: dict[str, dict[str, np.ndarray]] = {}
    for stem in stems:
        print(f"[u12] inference: {stem}")
        try:
            per_stem_probs[stem] = _predict_stem_all_folds(
                stem, val_records,
                device=device, use_amp=use_amp, batch_size=args.batch_size,
            )
        except RuntimeError as e:
            print(f"  [ERROR] {e} — skipping stem")

    if not per_stem_probs:
        raise RuntimeError("No stems could be evaluated. Check that checkpoints exist.")

    # Prune stem_star to available stems
    available_stems = list(per_stem_probs.keys())
    pruned_stem_star: dict[str, list[float]] = {}
    for task in TASKS:
        pruned_ws = [
            w for stem, w in zip(stems, stem_star[task])
            if stem in per_stem_probs
        ]
        pruned_stem_star[task] = pruned_ws
        if len(pruned_ws) < len(stem_star[task]):
            print(f"  [warn] task {task}: using {len(pruned_ws)}/{len(stem_star[task])} stems")

    mixed = _apply_weights(per_stem_probs, pruned_stem_star, alpha_star,
                           available_stems, n)
    val_score, val_preds = _score_from_probs(mixed, val_records)
    elapsed = time.time() - t0

    print("\n" + "=" * 60)
    print(f"AP-D4 OOF score:  {AP_D4_OOF_SCORE:.10f}")
    print(f"Val score:        {val_score['final_weighted_score']:.10f}")
    print(f"Gap (val - OOF):  {val_score['final_weighted_score'] - AP_D4_OOF_SCORE:+.10f}")
    print("=" * 60)
    print("Per-task breakdown:")
    for task, w in FIELD_WEIGHTS.items():
        oof_t = meta["final_score"].get(task, float("nan"))
        val_t = val_score[task]
        print(f"  {task:30s}  OOF={oof_t:.6f}  Val={val_t:.6f}  Δ={val_t - oof_t:+.6f}  w={w:.2f}")
    print(f"\n[u12] inference elapsed: {elapsed:.1f}s\n")

    # Per-task equal-weight baseline for reference
    eq_probs = {
        task: np.mean(
            [per_stem_probs[s][task] for s in available_stems], axis=0
        ) for task in TASKS
    }
    eq_score, _ = _score_from_probs(eq_probs, val_records)
    print(f"[u12] equal-weight baseline val: {eq_score['final_weighted_score']:.10f}")

    # Save outputs
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    preds_df = pd.DataFrame(val_preds)
    preds_df.to_csv(OUTPUT_DIR / "val_preds.csv", index=False)

    report = {
        "oof_score": AP_D4_OOF_SCORE,
        "val_score": val_score,
        "gap": val_score["final_weighted_score"] - AP_D4_OOF_SCORE,
        "equal_weight_val_score": eq_score["final_weighted_score"],
        "per_task": {
            task: {
                "oof": meta["final_score"].get(task, None),
                "val": val_score[task],
                "gap": val_score[task] - meta["final_score"].get(task, 0.0),
                "weight": FIELD_WEIGHTS[task],
            }
            for task in TASKS
        },
        "val_distribution": val_dist,
        "stems_evaluated": available_stems,
        "stems_requested": stems,
        "n_val": n,
        "note_label_alias": (
            "Val set uses 'more_than_5_years'; loader.py normalises to "
            "'longer_than_5_years' for scoring consistency with AP-D4 OOF."
        ),
        "meta_path": str(meta_path),
        "elapsed_s": round(elapsed, 1),
    }
    report_path = OUTPUT_DIR / "u12_val_gap.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[u12] wrote {report_path}")
    print(f"[u12] wrote {OUTPUT_DIR / 'val_preds.csv'}")


if __name__ == "__main__":
    main()
