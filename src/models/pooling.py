"""Pooling strategies for transformer encoders."""
from __future__ import annotations

import torch
import torch.nn as nn


class Pooler(nn.Module):
    """Wraps several pooling modes with a unified interface.

    Modes:
      - cls       : last_hidden_state[:, 0]
      - mean      : mean over non-padding tokens
      - cls_mean  : concat(cls, mean) -> 2H
    """

    def __init__(self, mode: str = "cls_mean") -> None:
        super().__init__()
        if mode not in {"cls", "mean", "cls_mean"}:
            raise ValueError(f"unknown pooling mode: {mode}")
        self.mode = mode

    @property
    def expansion(self) -> int:
        return 2 if self.mode == "cls_mean" else 1

    def forward(
        self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        cls = last_hidden_state[:, 0]
        if self.mode == "cls":
            return cls
        mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
        summed = (last_hidden_state * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1.0)
        mean = summed / denom
        if self.mode == "mean":
            return mean
        return torch.cat([cls, mean], dim=-1)
