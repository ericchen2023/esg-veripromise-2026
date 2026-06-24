"""Predict on a list of records and apply hierarchical post-processing."""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.dataset import ID2LABEL, TASKS
from src.inference.post_process import apply_constraints_batch


def predict_records(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    base_records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    """Return list of fully-populated records (post-processed) and probability tensors."""
    model.eval()
    probs_buf: dict[str, list[np.ndarray]] = {t: [] for t in TASKS}
    indices: list[int] = []
    autocast_ctx = torch.amp.autocast(device_type=device.type, enabled=use_amp)
    with torch.no_grad(), autocast_ctx:
        for batch in loader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            mask = batch["attention_mask"].to(device, non_blocking=True)
            logits = model(input_ids, mask)
            for t in TASKS:
                probs_buf[t].append(torch.softmax(logits[t].float(), dim=-1).cpu().numpy())
            indices.extend(batch["_index"].cpu().tolist())

    probs = {t: np.concatenate(probs_buf[t], axis=0) for t in TASKS}
    order = np.argsort(indices)
    probs = {t: probs[t][order] for t in TASKS}

    out: list[dict[str, Any]] = []
    for i, rec in enumerate(base_records):
        new = dict(rec)
        for t in TASKS:
            new[t] = ID2LABEL[t][int(probs[t][i].argmax())]
        out.append(new)
    out = apply_constraints_batch(out)
    return out, probs
