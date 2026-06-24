"""PyTorch Dataset and collate function for VeriPromise ESG 2026."""
from __future__ import annotations

import random
from typing import Any, Sequence

import torch
from torch.utils.data import Dataset

LABEL_DOMAINS: dict[str, list[str]] = {
    "promise_status": ["Yes", "No"],
    "verification_timeline": [
        "already",
        "within_2_years",
        "between_2_and_5_years",
        "longer_than_5_years",
        "N/A",
    ],
    "evidence_status": ["Yes", "No", "N/A"],
    "evidence_quality": ["Clear", "Not Clear", "Misleading", "N/A"],
}

LABEL2ID: dict[str, dict[str, int]] = {
    f: {lab: i for i, lab in enumerate(domain)} for f, domain in LABEL_DOMAINS.items()
}
ID2LABEL: dict[str, dict[int, str]] = {
    f: {i: lab for lab, i in m.items()} for f, m in LABEL2ID.items()
}
NUM_LABELS: dict[str, int] = {f: len(d) for f, d in LABEL_DOMAINS.items()}
TASKS: tuple[str, ...] = tuple(LABEL_DOMAINS.keys())


class ESGDataset(Dataset):
    """Tokenize text on the fly; emit input_ids/mask + integer task labels.

    Optional token-level augmentation (training only): with probability
    `aug_prob` per sample, randomly mask `mask_ratio` tokens with [MASK] and
    swap `swap_ratio` adjacent token pairs (special tokens / pads excluded).
    """

    def __init__(
        self,
        records: Sequence[dict[str, Any]],
        tokenizer,
        max_length: int = 256,
        text_field: str = "data",
        with_labels: bool = True,
        aug_prob: float = 0.0,
        mask_ratio: float = 0.0,
        swap_ratio: float = 0.0,
        delete_ratio: float = 0.0,
        aug_seed: int = 1234,
        text_transform=None,
    ) -> None:
        self.records = list(records)
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self.text_field = text_field
        self.with_labels = with_labels
        self.text_transform = text_transform
        self.aug_prob = float(aug_prob)
        self.mask_ratio = float(mask_ratio)
        self.swap_ratio = float(swap_ratio)
        self.delete_ratio = float(delete_ratio)
        self._rng = random.Random(aug_seed)
        # Cache special token ids that must never be perturbed
        self._mask_id = getattr(tokenizer, "mask_token_id", None)
        self._pad_id = getattr(tokenizer, "pad_token_id", 0)
        special_ids = set(tokenizer.all_special_ids or [])
        special_ids.discard(self._mask_id)  # MASK can be a target
        self._special_ids = special_ids

    def __len__(self) -> int:
        return len(self.records)

    def _augment_ids(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self._mask_id is None:
            return input_ids, attention_mask
        ids = input_ids.tolist()
        attn = attention_mask.tolist()
        # candidate positions = real tokens that are not special
        cand = [i for i, (tok, a) in enumerate(zip(ids, attn)) if a == 1 and tok not in self._special_ids]
        n = len(cand)
        if n < 4:
            return input_ids, attention_mask
        # mask
        if self.mask_ratio > 0:
            k = max(1, int(n * self.mask_ratio))
            for pos in self._rng.sample(cand, min(k, n)):
                ids[pos] = self._mask_id
        # swap adjacent pairs
        if self.swap_ratio > 0:
            k = max(1, int(n * self.swap_ratio))
            for _ in range(k):
                p = self._rng.choice(cand[:-1])
                if attn[p + 1] == 1 and ids[p + 1] not in self._special_ids:
                    ids[p], ids[p + 1] = ids[p + 1], ids[p]
        # delete (compact left, pad with [PAD] on right)
        if self.delete_ratio > 0:
            k = max(1, int(n * self.delete_ratio))
            del_set = set(self._rng.sample(cand, min(k, n)))
            new_ids: list[int] = []
            new_attn: list[int] = []
            for i, (tok, a) in enumerate(zip(ids, attn)):
                if i in del_set:
                    continue
                new_ids.append(tok)
                new_attn.append(a)
            pad_n = len(ids) - len(new_ids)
            new_ids.extend([self._pad_id] * pad_n)
            new_attn.extend([0] * pad_n)
            ids, attn = new_ids, new_attn
        return torch.tensor(ids, dtype=input_ids.dtype), torch.tensor(attn, dtype=attention_mask.dtype)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        rec = self.records[idx]
        text = str(rec.get(self.text_field, "") or "")
        if self.text_transform is not None:
            text = self.text_transform(text)
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        if (
            self.with_labels
            and self.aug_prob > 0
            and self._rng.random() < self.aug_prob
        ):
            input_ids, attention_mask = self._augment_ids(input_ids, attention_mask)
        item: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if self.with_labels:
            labels: dict[str, torch.Tensor] = {}
            for f in TASKS:
                lab = rec.get(f, "N/A")
                if lab not in LABEL2ID[f]:
                    raise ValueError(f"Unknown label '{lab}' for field '{f}' at idx={idx}")
                labels[f] = torch.tensor(LABEL2ID[f][lab], dtype=torch.long)
            item["labels"] = labels
        item["_index"] = idx
        return item


def esg_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "_index": torch.tensor([b["_index"] for b in batch], dtype=torch.long),
    }
    if "labels" in batch[0]:
        out["labels"] = {
            f: torch.stack([b["labels"][f] for b in batch]) for f in TASKS
        }
    return out
