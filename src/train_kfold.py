"""Run K-Fold cross-validation for one experiment config and one or more seeds.

Usage:
    python -m src.train_kfold --config configs/exp_p1_baseline.yaml [--smoke]

Outputs (per seed, per fold):
    outputs/checkpoints/{exp_name}/seed{S}/fold{F}/best.pt
    outputs/checkpoints/{exp_name}/seed{S}/fold{F}/oof_probs.npz
Logs:
    outputs/logs/{exp_name}/seed{S}.jsonl
Summary:
    reports/experiments/{exp_name}/score_summary.csv
    reports/experiments/{exp_name}/score_summary.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

# Allow `python -m src.train_kfold` and direct execution
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.data.dataset import ESGDataset, NUM_LABELS, TASKS, esg_collate
from src.data.loader import load_dataset
from src.data.splits import make_folds, report_distribution, save_folds
from src.data.text_augment import get_text_transform, get_added_tokens
from src.eval.metrics import FIELD_WEIGHTS
from src.models.multitask import MultiTaskClassifier
from src.seed import set_seed
from src.training.trainer import train_fold


def _build_loaders(
    train_records, val_records, tokenizer, cfg
) -> tuple[DataLoader, DataLoader]:
    max_len = int(cfg["model"]["max_length"])
    bs = int(cfg["training"]["batch_size"])
    nw = int(cfg.get("runtime", {}).get("num_workers", 0))
    pin = bool(cfg.get("runtime", {}).get("pin_memory", True))

    tr_aug = (cfg.get("training", {}) or {}).get("augment", {}) or {}
    text_tf_name = (cfg.get("data", {}) or {}).get("text_transform")
    text_tf = get_text_transform(text_tf_name)
    tr_ds = ESGDataset(
        train_records,
        tokenizer,
        max_length=max_len,
        aug_prob=float(tr_aug.get("aug_prob", 0.0)),
        mask_ratio=float(tr_aug.get("mask_ratio", 0.0)),
        swap_ratio=float(tr_aug.get("swap_ratio", 0.0)),
        delete_ratio=float(tr_aug.get("delete_ratio", 0.0)),
        aug_seed=int(tr_aug.get("aug_seed", 1234)),
        text_transform=text_tf,
    )
    va_ds = ESGDataset(val_records, tokenizer, max_length=max_len, text_transform=text_tf)
    # U5 / B2 — optional T4 class-balanced re-sampling.
    # When training.resample_t4 is set (true OR a float oversample exponent in (0,1]),
    # build a WeightedRandomSampler with weights = 1 / count(class)^alpha, where
    # alpha defaults to 1.0 (full inverse frequency); 0.5 uses sqrt-frequency.
    resample_cfg = (cfg.get("training") or {}).get("resample_t4")
    if resample_cfg:
        from collections import Counter
        from torch.utils.data import WeightedRandomSampler
        from src.data.dataset import LABEL_DOMAINS
        alpha = 1.0 if resample_cfg is True else float(resample_cfg)
        labels = [str(r.get("evidence_quality", "N/A")) for r in train_records]
        counts = Counter(labels)
        domain_size = len(LABEL_DOMAINS["evidence_quality"])
        weights = []
        for lab in labels:
            c = max(1, counts.get(lab, 1))
            weights.append((1.0 / c) ** alpha)
        sampler = WeightedRandomSampler(
            weights=torch.tensor(weights, dtype=torch.double),
            num_samples=len(weights),
            replacement=True,
        )
        print(f"[u5] T4 resample alpha={alpha} class_counts={dict(counts)} (n_classes={domain_size})")
        tr = DataLoader(
            tr_ds, batch_size=bs, sampler=sampler, collate_fn=esg_collate,
            num_workers=nw, pin_memory=pin, drop_last=False,
        )
    else:
        tr = DataLoader(
            tr_ds, batch_size=bs, shuffle=True, collate_fn=esg_collate,
            num_workers=nw, pin_memory=pin, drop_last=False,
        )
    va = DataLoader(
        va_ds, batch_size=bs * 2, shuffle=False, collate_fn=esg_collate,
        num_workers=nw, pin_memory=pin, drop_last=False,
    )
    return tr, va


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Smoke test: 1 seed, 2 folds, 1 epoch, 64-sample subset.",
    )
    parser.add_argument("--folds", type=int, default=None, help="Override n_splits")
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="Override seeds")
    parser.add_argument(
        "--split-mode",
        choices=["stratified_kfold", "stratified_group_kfold"],
        default=None,
        help=(
            "Override split.type. 'stratified_group_kfold' holds out whole "
            "companies (group_field) for an honest binary-task floor (Phase 53); "
            "outputs go to a distinct '<exp>_gkf' tree so stratified ckpts/OOF "
            "are never overwritten."
        ),
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    exp_name = cfg.get("exp_name", Path(args.config).stem)
    if args.split_mode:
        cfg.setdefault("split", {})["type"] = args.split_mode
        if args.split_mode == "stratified_group_kfold":
            exp_name = f"{exp_name}_gkf"
            cfg["exp_name"] = exp_name
            print(f"[split] GroupKFold by '{cfg['data'].get('group_field')}' "
                  f"-> isolated output tree exp={exp_name}")
    print(f"[run] exp={exp_name}")

    csv_path = cfg["data"]["csv_path"]
    records, df = load_dataset(csv_path)
    print(f"[data] loaded {len(records)} records from {csv_path}")

    # U6 / B4 — optional back-translation augmentation pool. Loaded once and
    # injected per-fold based on each augmented record's ``_source_id`` so it
    # only enters folds whose source sample is in the train index (never
    # leaks into val/OOF).
    aug_records: list = []
    aug_path = (cfg.get("data") or {}).get("augment_path")
    if aug_path:
        with open(aug_path, "r", encoding="utf-8") as _f:
            aug_records = json.load(_f)
        print(f"[u6] loaded {len(aug_records)} augmented records from {aug_path}")

    if args.smoke:
        records = records[:64]
        df = df.iloc[:64].reset_index(drop=True)
        cfg["training"]["epochs"] = 1
        cfg["split"]["n_splits"] = 2
        cfg["seeds"] = [42]
        print("[smoke] reduced to 64 samples, 1 epoch, 2 folds, 1 seed")

    if args.folds:
        cfg["split"]["n_splits"] = int(args.folds)
    if args.seeds:
        cfg["seeds"] = list(args.seeds)

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["backbone"])

    # T6 v2: register dedicated bucket tokens before any model build, so all
    # folds see consistent vocab and the encoder embedding can be resized.
    text_tf_name = (cfg.get("data", {}) or {}).get("text_transform")
    extra_tokens = get_added_tokens(text_tf_name)
    if extra_tokens:
        n_added = tokenizer.add_tokens(extra_tokens, special_tokens=False)
        print(f"[tokenizer] added {n_added} tokens for transform={text_tf_name!r}: {extra_tokens}")
    cfg["_runtime_extra_tokens"] = extra_tokens

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
        save_folds(folds, split_root, int(seed),
                   meta={"exp": exp_name, "n_records": len(records)})
        dist = report_distribution(df, folds, list(NUM_LABELS.keys()))
        (rep_root / f"split_seed{seed}.json").write_text(
            json.dumps(dist, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        for fi, (tr_idx, va_idx) in enumerate(folds):
            print(
                f"\n=== seed={seed} fold={fi+1}/{len(folds)} "
                f"train={len(tr_idx)} val={len(va_idx)} ==="
            )
            train_recs = [records[i] for i in tr_idx]
            val_recs = [records[i] for i in va_idx]

            # U6 — append back-translation augmented samples whose source id
            # is in the train index. Never inject into val/OOF.
            if aug_records:
                tr_ids = {int(records[i]["id"]) for i in tr_idx}
                injected = [a for a in aug_records if int(a.get("_source_id", -1)) in tr_ids]
                if injected:
                    train_recs = train_recs + injected
                    print(f"[u6] fold={fi} injected {len(injected)} BT augmented samples (train={len(train_recs)})")

            train_loader, val_loader = _build_loaders(
                train_recs, val_recs, tokenizer, cfg
            )

            set_seed(int(seed) + fi)  # different init per fold
            model = MultiTaskClassifier(
                backbone=cfg["model"]["backbone"],
                num_labels=NUM_LABELS,
                pooling=cfg["model"].get("pooling", "cls_mean"),
                dropout=float(cfg["model"].get("dropout", 0.1)),
                msd_k=int(cfg["model"].get("msd_k", 1)),
            )
            if cfg.get("_runtime_extra_tokens"):
                model.encoder.resize_token_embeddings(len(tokenizer))

            fold_dir = out_root / f"seed{seed}" / f"fold{fi}"
            res = train_fold(
                fold=fi,
                seed=int(seed),
                train_records=train_recs,
                val_records=val_recs,
                model=model,
                tokenizer=tokenizer,
                train_loader=train_loader,
                val_loader=val_loader,
                cfg=cfg,
                out_root=fold_dir,
                log_path=log_root / f"seed{seed}.jsonl",
                val_global_indices=[int(i) for i in va_idx],
            )
            row = {
                "exp": exp_name,
                "seed": int(seed),
                "fold": fi,
                "best_epoch": res.best_epoch,
                "weighted_score": res.best_score,
                **{f"f1_{t}": res.per_task[t] for t in TASKS},
                "ckpt": res.ckpt_path,
                "oof": res.oof_path,
            }
            all_rows.append(row)

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    df_sum = pd.DataFrame(all_rows)
    csv_out = rep_root / "score_summary.csv"
    json_out = rep_root / "score_summary.json"
    df_sum.to_csv(csv_out, index=False, encoding="utf-8")
    df_sum.to_json(json_out, orient="records", force_ascii=False, indent=2)

    # Aggregate
    print("\n" + "=" * 60)
    print(f"[exp={exp_name}] elapsed={time.time()-t_start:.1f}s")
    if not df_sum.empty:
        agg = df_sum.groupby("seed")["weighted_score"].agg(["mean", "std", "min", "max"])
        print("\nPer-seed weighted_score statistics:")
        print(agg.to_string())
        overall_mean = df_sum["weighted_score"].mean()
        overall_std = df_sum["weighted_score"].std()
        print(f"\nOverall: mean={overall_mean:.5f} std={overall_std:.5f}  (n={len(df_sum)})")

        per_task_means = {t: df_sum[f"f1_{t}"].mean() for t in TASKS}
        recon = sum(per_task_means[t] * FIELD_WEIGHTS[t] for t in TASKS)
        print("\nPer-task means:")
        for t in TASKS:
            print(f"  {t:25s} = {per_task_means[t]:.4f}  (w={FIELD_WEIGHTS[t]})")
        print(f"  weighted reconstruction = {recon:.5f}")

    print(f"\n[wrote] {csv_out}")
    print(f"[wrote] {json_out}")


if __name__ == "__main__":
    main()
