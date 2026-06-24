"""Multi-task transformer classifier for VeriPromise ESG 2026.

Shared encoder + per-task classification head. Returns dict of logits.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel

from src.models.pooling import Pooler


class MultiTaskClassifier(nn.Module):
    def __init__(
        self,
        backbone: str,
        num_labels: dict[str, int],
        pooling: str = "cls_mean",
        dropout: float = 0.1,
        msd_k: int = 1,
    ) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(backbone)
        hidden = self.encoder.config.hidden_size
        self.pooler = Pooler(pooling)
        feat_dim = hidden * self.pooler.expansion
        self.dropouts = nn.ModuleDict({k: nn.Dropout(dropout) for k in num_labels})
        self.heads = nn.ModuleDict(
            {k: nn.Linear(feat_dim, n) for k, n in num_labels.items()}
        )
        self.task_names: tuple[str, ...] = tuple(num_labels.keys())
        self.msd_k = int(msd_k)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        feat = self.pooler(out.last_hidden_state, attention_mask)
        if self.training and self.msd_k > 1:
            # Multi-Sample Dropout: average logits over K dropout passes
            result: dict[str, torch.Tensor] = {}
            for k in self.task_names:
                acc = 0.0
                for _ in range(self.msd_k):
                    acc = acc + self.heads[k](self.dropouts[k](feat))
                result[k] = acc / self.msd_k
            return result
        return {k: self.heads[k](self.dropouts[k](feat)) for k in self.task_names}
