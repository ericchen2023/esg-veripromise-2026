"""Run two-stage K-Fold training with external pseudo labels.

Stage A trains on official-train-fold rows plus accepted pseudo rows.
Stage B reloads the best Stage A checkpoint and fine-tunes on official rows only.
Validation and OOF scoring always use official validation folds, so the reported
score remains comparable with previous experiments.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.data.dataset import NUM_LABELS, TASKS
from src.data.loader import load_dataset
from src.data.splits import make_folds, report_distribution, save_folds
from src.data.text_augment import get_added_tokens
from src.eval.metrics import FIELD_WEIGHTS
from src.models.multitask import MultiTaskClassifier
from src.seed import set_seed
from src.train_kfold import _build_loaders
from src.training.trainer import train_fold


def _load_pseudo_records(path: str, min_confidence: float, max_pseudo: int) -> list[dict]:
    pseudo_path = Path(path)
    if not pseudo_path.exists():
        raise FileNotFoundError(f"Pseudo CSV not found: {pseudo_path}")
    _, df = load_dataset(pseudo_path)
    if "confidence_min" in df.columns:
        df["confidence_min"] = pd.to_numeric(df["confidence_min"], errors="coerce").fillna(0.0)
        df = df[df["confidence_min"] >= min_confidence].copy()
        df = df.sort_values("confidence_min", ascending=False)
    if max_pseudo > 0:
        df = df.head(max_pseudo).copy()
    return df.to_dict(orient="records")


def _make_model(cfg: dict, tokenizer) -> MultiTaskClassifier:
    model = MultiTaskClassifier(
        backbone=cfg["model"]["backbone"],
        num_labels=NUM_LABELS,
        pooling=cfg["model"].get("pooling", "cls_mean"),
        dropout=float(cfg["model"].get("dropout", 0.1)),
        msd_k=int(cfg["model"].get("msd_k", 1)),
    )
    if cfg.get("_runtime_extra_tokens"):
        model.encoder.resize_token_embeddings(len(tokenizer))
    return model


def _load_stage_a_model(cfg: dict, tokenizer, ckpt_path: str) -> MultiTaskClassifier:
    model = _make_model(cfg, tokenizer)
    payload = torch.load(ckpt_path, map_location="cpu")
    state = payload.get("model_state_dict", payload)
    model.load_state_dict(state, strict=True)
    return model


def _stage_cfg(cfg: dict, epochs: int) -> dict:
    out = copy.deepcopy(cfg)
    out["training"]["epochs"] = int(epochs)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--folds", type=int, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--pseudo", default=None, help="Override data.pseudo_csv_path")
    args = parser.parse_args()

    cfg = load_config(args.config)
    exp_name = cfg.get("exp_name", Path(args.config).stem)
    if args.smoke:
        exp_name = f"{exp_name}_smoke"
    print(f"[run] exp={exp_name} mode=pseudo_two_stage")

    records, df = load_dataset(cfg["data"]["csv_path"])
    pseudo_cfg = cfg.get("pseudo", {}) or {}
    pseudo_csv = args.pseudo or cfg.get("data", {}).get("pseudo_csv_path")
    if not pseudo_csv:
        raise ValueError("Set data.pseudo_csv_path in config or pass --pseudo")

    min_conf = float(pseudo_cfg.get("min_confidence", 0.90))
    max_pseudo = int(pseudo_cfg.get("max_pseudo", 3000))
    if args.smoke:
        min_conf = 0.0
        max_pseudo = min(max_pseudo, 32) if max_pseudo > 0 else 32
    pseudo_records = _load_pseudo_records(pseudo_csv, min_conf, max_pseudo)
    print(f"[data] official={len(records)} pseudo={len(pseudo_records)} from {pseudo_csv}")

    # U6-pro / B4 — optional back-translation augmentation pool. Loaded once and
    # injected per-fold based on each augmented record's ``_source_id`` so it
    # only enters folds whose source sample is in the train index (never
    # leaks into val/OOF). Used by both stage A and stage B train pools.
    aug_records: list = []
    aug_path = (cfg.get("data") or {}).get("augment_path")
    if aug_path:
        with open(aug_path, "r", encoding="utf-8") as _f:
            aug_records = json.load(_f)
        print(f"[u6] loaded {len(aug_records)} augmented records from {aug_path}")

    if args.smoke:
        records = records[:64]
        df = df.iloc[:64].reset_index(drop=True)
        pseudo_records = pseudo_records[:32]
        cfg["split"]["n_splits"] = 2
        cfg["seeds"] = [42]
        pseudo_cfg["stage_a_epochs"] = 1
        pseudo_cfg["stage_b_epochs"] = 1
        print("[smoke] official=64 pseudo<=32, 1+1 epochs, 2 folds, 1 seed")

    if args.folds:
        cfg["split"]["n_splits"] = int(args.folds)
    if args.seeds:
        cfg["seeds"] = list(args.seeds)

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["backbone"])
    extra_tokens = get_added_tokens((cfg.get("data", {}) or {}).get("text_transform"))
    if extra_tokens:
        n_added = tokenizer.add_tokens(extra_tokens, special_tokens=False)
        print(f"[tokenizer] added {n_added} tokens for pseudo training: {extra_tokens}")
    cfg["_runtime_extra_tokens"] = extra_tokens

    stage_a_epochs = int(pseudo_cfg.get("stage_a_epochs", 2))
    stage_b_epochs = int(pseudo_cfg.get("stage_b_epochs", 3))
    if stage_a_epochs < 0 or stage_b_epochs <= 0:
        raise ValueError("Require stage_a_epochs >= 0 and stage_b_epochs > 0")

    out_root = Path("outputs/checkpoints") / exp_name
    log_root = Path("outputs/logs") / exp_name
    split_root = Path("data/splits") / exp_name
    rep_root = Path("reports/experiments") / exp_name
    rep_root.mkdir(parents=True, exist_ok=True)

    all_rows = []
    t_start = time.time()

    for seed in cfg["seeds"]:
        set_seed(int(seed))
        folds = make_folds(
            df,
            n_splits=int(cfg["split"]["n_splits"]),
            stratify_fields=cfg["split"]["stratify_fields"],
            seed=int(seed),
            mode=cfg["split"].get("type", "stratified_kfold"),
            group_field=cfg["data"].get("group_field"),
        )
        save_folds(
            folds,
            split_root,
            int(seed),
            meta={"exp": exp_name, "n_official_records": len(records), "n_pseudo_records": len(pseudo_records)},
        )
        dist = report_distribution(df, folds, list(NUM_LABELS.keys()))
        (rep_root / f"split_seed{seed}.json").write_text(
            json.dumps(dist, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        for fi, (tr_idx, va_idx) in enumerate(folds):
            print(
                f"\n=== seed={seed} fold={fi+1}/{len(folds)} "
                f"official_train={len(tr_idx)} pseudo={len(pseudo_records)} val={len(va_idx)} ==="
            )
            train_recs = [records[i] for i in tr_idx]
            val_recs = [records[i] for i in va_idx]
            fold_dir = out_root / f"seed{seed}" / f"fold{fi}"

            # U6-pro — append BT augmented samples whose source id is in the
            # train index. Never inject into val/OOF.
            aug_inject: list = []
            if aug_records:
                tr_ids = {int(records[i]["id"]) for i in tr_idx}
                aug_inject = [a for a in aug_records if int(a.get("_source_id", -1)) in tr_ids]
                if aug_inject:
                    print(f"[u6] fold={fi} injected {len(aug_inject)} BT augmented samples")

            stage_a_score = None
            if stage_a_epochs > 0 and pseudo_records:
                cfg_a = _stage_cfg(cfg, stage_a_epochs)
                train_loader_a, val_loader_a = _build_loaders(train_recs + pseudo_records + aug_inject, val_recs, tokenizer, cfg_a)
                set_seed(int(seed) + fi)
                model_a = _make_model(cfg_a, tokenizer)
                res_a = train_fold(
                    fold=fi,
                    seed=int(seed),
                    train_records=train_recs + pseudo_records + aug_inject,
                    val_records=val_recs,
                    model=model_a,
                    tokenizer=tokenizer,
                    train_loader=train_loader_a,
                    val_loader=val_loader_a,
                    cfg=cfg_a,
                    out_root=fold_dir / "stage_a",
                    log_path=log_root / f"seed{seed}.jsonl",
                    val_global_indices=[int(i) for i in va_idx],
                )
                stage_a_score = res_a.best_score
                model_b = _load_stage_a_model(cfg, tokenizer, res_a.ckpt_path)
                del model_a
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            else:
                set_seed(int(seed) + fi)
                model_b = _make_model(cfg, tokenizer)

            cfg_b = _stage_cfg(cfg, stage_b_epochs)
            train_recs_b = train_recs + aug_inject
            train_loader_b, val_loader_b = _build_loaders(train_recs_b, val_recs, tokenizer, cfg_b)
            res_b = train_fold(
                fold=fi,
                seed=int(seed),
                train_records=train_recs_b,
                val_records=val_recs,
                model=model_b,
                tokenizer=tokenizer,
                train_loader=train_loader_b,
                val_loader=val_loader_b,
                cfg=cfg_b,
                out_root=fold_dir,
                log_path=log_root / f"seed{seed}.jsonl",
                val_global_indices=[int(i) for i in va_idx],
            )
            row = {
                "exp": exp_name,
                "seed": int(seed),
                "fold": fi,
                "stage_a_best_score": stage_a_score,
                "best_epoch": res_b.best_epoch,
                "weighted_score": res_b.best_score,
                **{f"f1_{task}": res_b.per_task[task] for task in TASKS},
                "ckpt": res_b.ckpt_path,
                "oof": res_b.oof_path,
                "pseudo_rows": len(pseudo_records),
            }
            all_rows.append(row)
            del model_b
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    df_sum = pd.DataFrame(all_rows)
    csv_out = rep_root / "score_summary.csv"
    json_out = rep_root / "score_summary.json"
    df_sum.to_csv(csv_out, index=False, encoding="utf-8")
    df_sum.to_json(json_out, orient="records", force_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print(f"[exp={exp_name}] elapsed={time.time()-t_start:.1f}s")
    if not df_sum.empty:
        agg = df_sum.groupby("seed")["weighted_score"].agg(["mean", "std", "min", "max"])
        print("\nPer-seed weighted_score statistics:")
        print(agg.to_string())
        overall_mean = df_sum["weighted_score"].mean()
        overall_std = df_sum["weighted_score"].std()
        print(f"\nOverall: mean={overall_mean:.5f} std={overall_std:.5f}  (n={len(df_sum)})")
        per_task_means = {task: df_sum[f"f1_{task}"].mean() for task in TASKS}
        recon = sum(per_task_means[task] * FIELD_WEIGHTS[task] for task in TASKS)
        print("\nPer-task means:")
        for task in TASKS:
            print(f"  {task:25s} = {per_task_means[task]:.4f}  (w={FIELD_WEIGHTS[task]})")
        print(f"  weighted reconstruction = {recon:.5f}")

    print(f"\n[wrote] {csv_out}")
    print(f"[wrote] {json_out}")


if __name__ == "__main__":
    main()