"""U3 / B1 — SWA aggregator over last-K epoch checkpoints.

Reads ``outputs/checkpoints/{exp}/seed{S}/fold{F}/swa_epoch*.pt`` for each
fold, simple-averages the model state dicts, instantiates a fresh model,
loads the averaged weights, then re-runs validation to produce
``swa.pt`` and ``oof_probs_swa.npz`` per fold. Finally aggregates
per-fold OOF probabilities into a member-level OOF view and reports
the SWA weighted_score vs the original best_score.

Usage:
    python -m src.tools.u3_swa_aggregate --config configs/exp_p2_combo_best_swa.yaml

This script never trains; it only inferences with averaged weights.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.config import load_config
from src.data.dataset import ESGDataset, ID2LABEL, NUM_LABELS, TASKS, esg_collate
from src.data.loader import load_dataset
from src.data.splits import load_folds
from src.eval.metrics import weighted_score
from src.models.multitask import MultiTaskClassifier
from src.training.trainer import _predict


def _avg_state_dicts(ckpts: list[Path]) -> dict:
    sds = []
    for c in ckpts:
        payload = torch.load(c, map_location="cpu", weights_only=False)
        sds.append(payload["model_state_dict"])
    keys = list(sds[0].keys())
    avg = {}
    for k in keys:
        if not torch.is_tensor(sds[0][k]):
            avg[k] = sds[0][k]
            continue
        if not sds[0][k].is_floating_point():
            # Non-float (e.g. position_ids, token_type_ids, integer buffers): keep first.
            avg[k] = sds[0][k]
            continue
        stacked = torch.stack([sd[k].float() for sd in sds], dim=0)
        avg[k] = stacked.mean(dim=0).to(sds[0][k].dtype)
    return avg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=None, help="Limit to a single seed (default: all in cfg)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    exp_name = cfg.get("exp_name", Path(args.config).stem)
    print(f"[u3-swa] exp={exp_name}")

    csv_path = cfg["data"]["csv_path"]
    records, df = load_dataset(csv_path)
    n_total = len(records)
    print(f"[data] N={n_total}")

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["backbone"])
    extra_tokens = cfg.get("_runtime_extra_tokens") or []
    if extra_tokens:
        tokenizer.add_tokens(list(extra_tokens), special_tokens=False)

    out_root = Path("outputs/checkpoints") / exp_name
    rep_root = Path("reports/experiments") / exp_name
    rep_root.mkdir(parents=True, exist_ok=True)
    diag_dir = Path("reports/analysis/diagnostics")
    diag_dir.mkdir(parents=True, exist_ok=True)

    seeds = [int(s) for s in (cfg.get("seeds") or [42])]
    if args.seed is not None:
        seeds = [int(args.seed)]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(cfg.get("training", {}).get("use_amp", True)) and device.type == "cuda"
    bs = int(cfg["training"]["batch_size"])
    nw = int(cfg.get("runtime", {}).get("num_workers", 0))
    pin = bool(cfg.get("runtime", {}).get("pin_memory", True))
    max_len = int(cfg["model"]["max_length"])

    all_rows: list[dict] = []
    for seed in seeds:
        split_root = Path("data/splits") / exp_name / f"seed{seed}.json"
        folds = load_folds(split_root)
        member_oof_probs = {t: np.zeros((n_total, NUM_LABELS[t]), dtype=np.float32) for t in TASKS}
        member_filled = np.zeros(n_total, dtype=bool)

        for fi, (tr_idx, va_idx) in enumerate(folds):
            fold_dir = out_root / f"seed{seed}" / f"fold{fi}"
            ckpts = sorted(fold_dir.glob("swa_epoch*.pt"))
            if not ckpts:
                print(f"  [skip] {fold_dir}: no swa_epoch*.pt")
                continue
            print(f"[u3-swa] fold={fi}: averaging {len(ckpts)} checkpoints {[c.name for c in ckpts]}")

            avg_sd = _avg_state_dicts(ckpts)

            model = MultiTaskClassifier(
                backbone=cfg["model"]["backbone"],
                num_labels=NUM_LABELS,
                pooling=cfg["model"].get("pooling", "cls_mean"),
                dropout=float(cfg["model"].get("dropout", 0.1)),
                msd_k=int(cfg["model"].get("msd_k", 1)),
            )
            if extra_tokens:
                model.encoder.resize_token_embeddings(len(tokenizer))
            missing, unexpected = model.load_state_dict(avg_sd, strict=False)
            if missing or unexpected:
                print(f"    [warn] missing={missing[:3]}... unexpected={unexpected[:3]}...")
            model = model.to(device)

            val_recs = [records[i] for i in va_idx]
            val_truth = {t: [r[t] for r in val_recs] for t in TASKS}
            va_ds = ESGDataset(val_recs, tokenizer, max_length=max_len)
            va_loader = DataLoader(
                va_ds, batch_size=bs * 2, shuffle=False, collate_fn=esg_collate,
                num_workers=nw, pin_memory=pin, drop_last=False,
            )
            probs, idx = _predict(model, va_loader, device, use_amp)
            order = np.argsort(idx)
            ordered_probs = {t: probs[t][order] for t in TASKS}
            preds_id = {t: ordered_probs[t].argmax(axis=1) for t in TASKS}
            preds_text = {t: [ID2LABEL[t][int(i)] for i in preds_id[t]] for t in TASKS}
            score = weighted_score(val_truth, preds_text)
            print(f"  [fold={fi}] SWA score = {score['final_weighted_score']:.5f}  per-task={ {t: round(score[t],4) for t in TASKS} }")

            # Save SWA checkpoint + OOF probs
            swa_ckpt_path = fold_dir / "swa.pt"
            torch.save({"model_state_dict": avg_sd, "scores": score, "fold": fi, "seed": seed, "n_avg": len(ckpts)}, swa_ckpt_path)
            local_sorted = np.asarray(idx)[order].astype(np.int32)
            global_sorted = np.asarray(va_idx, dtype=np.int32)[local_sorted]
            np.savez_compressed(
                fold_dir / "oof_probs_swa.npz",
                indices=global_sorted,
                local_indices=local_sorted,
                **{f"probs_{t}": ordered_probs[t].astype(np.float16) for t in TASKS},
                meta=json.dumps({"fold": fi, "seed": seed, "n_avg": len(ckpts), "score": float(score["final_weighted_score"]), "tasks": list(TASKS)}, ensure_ascii=False),
            )

            # Also overwrite the canonical oof_probs.npz so downstream tools
            # (joint hillclimb, u1-c TTA) pick up the SWA member as a regular
            # ensemble member of experiment `p2_combo_best_swa`. The original
            # best.pt OOF for this experiment is still represented by p2_combo_best
            # (the non-SWA experiment) — they live in different exp directories.
            np.savez_compressed(
                fold_dir / "oof_probs.npz",
                indices=global_sorted,
                local_indices=local_sorted,
                **{f"probs_{t}": ordered_probs[t].astype(np.float16) for t in TASKS},
                meta=json.dumps({"fold": fi, "seed": seed, "n_avg_swa": len(ckpts), "score": float(score["final_weighted_score"]), "tasks": list(TASKS), "source": "swa"}, ensure_ascii=False),
            )

            # Pull baseline best.pt score for comparison from existing oof_probs.npz
            base_npz = fold_dir / "oof_probs.npz"
            base_score = None
            if base_npz.exists():
                z = np.load(base_npz, allow_pickle=True)
                meta = json.loads(str(z["meta"]))
                base_score = float(meta.get("score", float("nan")))

            row = {"seed": seed, "fold": fi, "n_avg": len(ckpts),
                   "swa_score": float(score["final_weighted_score"]), "base_score": base_score,
                   "delta": (float(score["final_weighted_score"]) - base_score) if base_score is not None else None,
                   **{f"swa_{t}": float(score[t]) for t in TASKS}}
            all_rows.append(row)

            for t in TASKS:
                member_oof_probs[t][global_sorted] = ordered_probs[t].astype(np.float32)
            member_filled[global_sorted] = True

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Member-level OOF score
        if member_filled.all():
            preds_id = {t: member_oof_probs[t].argmax(axis=1) for t in TASKS}
            preds_text = {t: [ID2LABEL[t][int(i)] for i in preds_id[t]] for t in TASKS}
            truth = {t: [r[t] for r in records] for t in TASKS}
            agg_score = weighted_score(truth, preds_text)
            print(f"\n[u3-swa] seed={seed} MEMBER-LEVEL SWA OOF score = {agg_score['final_weighted_score']:.5f}")
            agg_score_summary = {"member_oof_swa_score": float(agg_score["final_weighted_score"]),
                                 **{t: float(agg_score[t]) for t in TASKS}}
        else:
            agg_score_summary = {"warning": f"{int((~member_filled).sum())} OOF rows missing"}

    out_csv = rep_root / "swa_score_summary.csv"
    import pandas as pd
    pd.DataFrame(all_rows).to_csv(out_csv, index=False)
    out_json = rep_root / "swa_score_summary.json"
    out_json.write_text(json.dumps({"rows": all_rows, "agg": agg_score_summary, "exp": exp_name, "seeds": seeds}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[wrote] {out_csv}")
    print(f"[wrote] {out_json}")

    # Diagnostic markdown
    base_mean = float(np.mean([r["base_score"] for r in all_rows if r["base_score"] is not None])) if all_rows else None
    swa_mean = float(np.mean([r["swa_score"] for r in all_rows])) if all_rows else None
    delta_mean = float(np.mean([r["delta"] for r in all_rows if r["delta"] is not None])) if all_rows else None
    md_lines = [f"# U3 — SWA aggregator: {exp_name}\n\n"]
    md_lines.append(f"- base mean per-fold score: {base_mean}\n- SWA mean per-fold score: {swa_mean}\n- delta mean: {delta_mean}\n")
    md_lines.append(f"- member-level OOF SWA: {agg_score_summary}\n")
    md_lines.append("\n## Per-fold detail\n\n| fold | n_avg | base | SWA | delta |\n| --: | --: | --: | --: | --: |\n")
    for r in all_rows:
        md_lines.append(f"| {r['fold']} | {r['n_avg']} | {r['base_score']} | {r['swa_score']:.5f} | {r['delta']} |\n")
    (diag_dir / f"u3_swa_{exp_name}.md").write_text("".join(md_lines), encoding="utf-8")
    print(f"[wrote] {diag_dir / f'u3_swa_{exp_name}.md'}")


if __name__ == "__main__":
    main()
