"""U24 — Phase 46 multi-window TTA test inference (middle + tail views).

The 8 TV stems were trained/inferred with HEAD truncation at max_length=384
("stored" view). ESG paragraphs that exceed 384 tokens lose everything after the
head — and per the project's own TTA analysis (notebook §13.2) the tail of an
ESG paragraph frequently carries the commitment / evidence statements. Phase 45
established there is NO domain shift, so a genuine signal-recovery lever (rather
than reweighting) is the most promising route to beat the 0.61 leaderboard mark.

This script computes two ADDITIONAL views per stem and caches them, with NO
re-training:

  * middle : the W-2 content tokens centred at (len-budget)//2
  * tail   : the last W-2 content tokens

Each view is built by tokenising the full text WITHOUT special tokens / without
truncation, slicing the token window, then manually prepending [CLS] and
appending [SEP] and right-padding to W (transformers 5.6 tokeniser objects make
build_inputs_with_special_tokens unreliable, per project memory). For texts that
fit within the window, middle/tail collapse to the stored view (no harm).

Per-stem view probs are cached to:
    outputs/submissions/phase46_tta_<view>_probs.npz   (keyed stem__task)

A companion combiner (u25) blends stored+middle+tail. This script also prints,
for reference, how many test rows actually exceed the window (the population TTA
can help).

Usage:
    python -m scripts.u24_tta_inference                # both views
    python -m scripts.u24_tta_inference --views tail   # one view
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from scripts.u17_phase42_test_inference import (
    BACKBONE,
    BATCH_SIZE,
    CKPT_ROOT,
    DROPOUT,
    MAX_LEN,
    N_FOLDS,
    OUT_DIR,
    POOLING,
    SEED_DIR,
    TV_STEMS,
    load_test_records,
)
from src.data.dataset import NUM_LABELS, TASKS


class WindowDataset(Dataset):
    """Token-window view of each record: [CLS] + window(W-2) + [SEP], padded to W."""

    def __init__(self, records: list[dict], tokenizer, view: str, max_length: int):
        self.records = records
        self.tok = tokenizer
        self.view = view
        self.W = int(max_length)
        self.cls_id = tokenizer.cls_token_id
        self.sep_id = tokenizer.sep_token_id
        self.pad_id = tokenizer.pad_token_id or 0
        self.n_truncated = 0

    def __len__(self) -> int:
        return len(self.records)

    def _content_ids(self, text: str) -> list[int]:
        return self.tok(text, add_special_tokens=False, truncation=False)["input_ids"]

    def __getitem__(self, idx: int) -> dict:
        text = str(self.records[idx].get("data", "") or "")
        content = self._content_ids(text)
        budget = self.W - 2  # reserve CLS + SEP
        if len(content) > budget:
            self.n_truncated += 1
            if self.view == "middle":
                start = max(0, (len(content) - budget) // 2)
                window = content[start:start + budget]
            elif self.view == "tail":
                window = content[-budget:]
            else:  # head / stored
                window = content[:budget]
        else:
            window = content  # fits entirely -> all views identical

        ids = [self.cls_id] + window + [self.sep_id]
        attn = [1] * len(ids)
        pad_n = self.W - len(ids)
        if pad_n > 0:
            ids += [self.pad_id] * pad_n
            attn += [0] * pad_n
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "_index": idx,
        }


def _collate(batch: list[dict]) -> dict:
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "_index": torch.tensor([b["_index"] for b in batch], dtype=torch.long),
    }


@torch.no_grad()
def infer_stem_fold(ckpt_path: Path, loader: DataLoader, device: torch.device, n: int
                    ) -> dict[str, np.ndarray]:
    from src.models.multitask import MultiTaskClassifier

    model = MultiTaskClassifier(backbone=BACKBONE, num_labels=NUM_LABELS,
                                pooling=POOLING, dropout=DROPOUT)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    model.to(device).eval()

    probs = {t: np.zeros((n, NUM_LABELS[t]), dtype=np.float64) for t in TASKS}
    use_amp = device.type == "cuda"
    with torch.amp.autocast(device_type=device.type, enabled=use_amp):
        for batch in loader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            mask = batch["attention_mask"].to(device, non_blocking=True)
            idx = batch["_index"].cpu().numpy()
            logits = model(input_ids, mask)
            for t in TASKS:
                probs[t][idx] = torch.softmax(logits[t].float(), dim=-1).cpu().numpy()
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return probs


def infer_view(records: list[dict], view: str, device: torch.device
               ) -> dict[str, dict[str, np.ndarray]]:
    tokenizer = AutoTokenizer.from_pretrained(BACKBONE)
    ds = WindowDataset(records, tokenizer, view, MAX_LEN)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=_collate,
                        num_workers=0, pin_memory=(device.type == "cuda"))
    n = len(records)
    per_stem: dict[str, dict[str, np.ndarray]] = {}
    for stem in TV_STEMS:
        acc = {t: np.zeros((n, NUM_LABELS[t]), dtype=np.float64) for t in TASKS}
        for fold in range(N_FOLDS):
            ckpt = CKPT_ROOT / stem / SEED_DIR / f"fold{fold}" / "best.pt"
            if not ckpt.exists():
                raise FileNotFoundError(f"missing checkpoint: {ckpt}")
            fp = infer_stem_fold(ckpt, loader, device, n)
            for t in TASKS:
                acc[t] += fp[t]
        for t in TASKS:
            acc[t] /= N_FOLDS
        per_stem[stem] = acc
        print(f"  [{view}] {stem:52s} {N_FOLDS} folds")
    print(f"  [{view}] rows exceeding window (TTA-relevant): {ds.n_truncated}/{n}")
    return per_stem


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 46 multi-window TTA inference")
    ap.add_argument("--test-csv", default="vpesg4k_test_2000.csv")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--views", nargs="+", default=["middle", "tail"],
                    choices=["middle", "tail", "head"])
    args = ap.parse_args()

    device = torch.device(args.device)
    records = load_test_records(args.test_csv)
    print(f"[data] {len(records)} test records; device={device}; views={args.views}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for view in args.views:
        print(f"\n[view] {view}")
        per_stem = infer_view(records, view, device)
        flat = {f"{stem}__{t}": per_stem[stem][t] for stem in TV_STEMS for t in TASKS}
        path = OUT_DIR / f"phase46_tta_{view}_probs.npz"
        np.savez_compressed(path, **flat)
        print(f"  [cache] wrote {path}")

    print("\n[done] TTA view probs cached. Next: python -m scripts.u25_tta_combine")


if __name__ == "__main__":
    main()
