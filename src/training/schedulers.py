"""LR schedulers."""
from __future__ import annotations

from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup


def build_scheduler(name: str, optimizer, num_warmup_steps: int, num_training_steps: int):
    if name == "cosine":
        return get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
    if name == "linear":
        return get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
    raise ValueError(f"unknown scheduler: {name}")
