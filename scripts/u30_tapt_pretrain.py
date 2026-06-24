"""U30 — Task-Adaptive Pretraining (TAPT) for ESG VeriPromise 2026.

Phase 49. The labeled data is exhausted (all 2,000 rows used by the 8 _tv
submission stems) and backbone retraining (Phase 48) failed. The ONLY untried,
positive-expected-value axis is domain/task-adaptive continued MLM pretraining
on the in-domain ESG text — including the *unlabeled* official test set
(transductive TAPT, which is legitimate: we use only the raw ``data`` text,
never any label).

Pipeline:
  1. Build an MLM corpus from the ``data`` column of train(1000) + val(1000) +
     test(2000) = 4,000 ESG paragraphs (deduplicated).
  2. Continue MLM pretraining hfl/chinese-macbert-base for a few epochs with a
     small LR and standard 15% masking.
  3. Save the adapted encoder + tokenizer to a local directory so it can be
     dropped into any training config via ``model.backbone: <dir>``.

Usage:
    # macbert-base (Phase 49 default)
    python -m scripts.u30_tapt_pretrain --epochs 8 --lr 5e-5 --batch 16

    # Phase 53 — domain-adapt a LARGER backbone (8 GB-safe: small batch).
    # This is the genuinely-new axis: Phase 48 tried large WITHOUT TAPT and it
    # overfit; large + TAPT has never been tried.
    python -m scripts.u30_tapt_pretrain \
        --backbone hfl/chinese-roberta-wwm-ext-large \
        --out-dir outputs/tapt/roberta_large_esg \
        --epochs 8 --lr 3e-5 --batch 4 --max-length 256
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

BACKBONE = "hfl/chinese-macbert-base"
OUT_DIR = Path("outputs/tapt/macbert_esg")
CORPUS_FILES = [
    "vpesg4k_train_1000 V1.csv",
    "vpesg4k_val_1000.csv",
    "vpesg4k_test_2000.csv",
]


def build_corpus() -> list[str]:
    texts: list[str] = []
    for f in CORPUS_FILES:
        df = pd.read_csv(f, encoding="utf-8")
        if "data" not in df.columns:
            raise ValueError(f"{f} has no 'data' column: {list(df.columns)}")
        texts.extend(str(t) for t in df["data"].tolist())
    # Deduplicate while preserving order.
    seen: set[str] = set()
    uniq: list[str] = []
    for t in texts:
        t = t.strip()
        if t and t not in seen:
            seen.add(t)
            uniq.append(t)
    return uniq


class MLMTextDataset(Dataset):
    def __init__(self, texts: list[str], tokenizer, max_length: int):
        self.enc = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding=False,
            return_special_tokens_mask=True,
        )

    def __len__(self) -> int:
        return len(self.enc["input_ids"])

    def __getitem__(self, i: int) -> dict:
        return {k: self.enc[k][i] for k in self.enc}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", type=str, default=BACKBONE,
                    help="HF id or local dir of the encoder to continue-pretrain.")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=1,
                    help="Gradient accumulation; raise for large backbones on 8 GB.")
    ap.add_argument("--max-length", type=int, default=384)
    ap.add_argument("--mlm-prob", type=float, default=0.15)
    ap.add_argument("--warmup-ratio", type=float, default=0.06)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", type=str, default=str(OUT_DIR))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    texts = build_corpus()
    print(f"[corpus] {len(texts)} unique ESG paragraphs "
          f"(from {', '.join(CORPUS_FILES)})")

    print(f"[tapt] backbone={args.backbone} -> out_dir={out_dir}")
    tokenizer = AutoTokenizer.from_pretrained(args.backbone)
    model = AutoModelForMaskedLM.from_pretrained(args.backbone)

    ds = MLMTextDataset(texts, tokenizer, args.max_length)
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=args.mlm_prob
    )

    targs = TrainingArguments(
        output_dir=str(out_dir / "_trainer"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="linear",
        fp16=torch.cuda.is_available(),
        logging_steps=25,
        save_strategy="no",
        report_to=[],
        seed=args.seed,
        dataloader_num_workers=2,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=ds,
        data_collator=collator,
    )

    print(f"[tapt] starting MLM continued-pretraining: epochs={args.epochs} "
          f"lr={args.lr} batch={args.batch} maxlen={args.max_length} "
          f"mlm_prob={args.mlm_prob}")
    trainer.train()

    # Save adapted encoder + tokenizer so it can be loaded via from_pretrained.
    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    print(f"[tapt] saved adapted model + tokenizer to {out_dir}")


if __name__ == "__main__":
    main()
