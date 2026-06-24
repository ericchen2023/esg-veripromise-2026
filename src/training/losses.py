"""Loss utilities for the multi-task setup."""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_class_weights(
    y: list[str], domain: list[str], beta: float = 1.0
) -> torch.Tensor:
    """Inverse-frequency class weights, normalized so mean weight = 1.

    beta=1 -> pure inverse frequency; beta in [0,1] dampens the effect.
    Missing classes receive a weight of 1.0 (neutral).
    """
    counts = np.array([max(1, sum(1 for v in y if v == c)) for c in domain], dtype=np.float64)
    inv = 1.0 / counts
    inv = inv ** beta
    w = inv / inv.mean()
    return torch.tensor(w, dtype=torch.float32)


class FocalLoss(nn.Module):
    """Multi-class focal loss (Lin et al. 2017) with optional class weights.

    L = - sum_c alpha_c * (1 - p_c)^gamma * y_c * log(p_c)
    """

    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None,
                 label_smoothing: float = 0.0) -> None:
        super().__init__()
        self.gamma = float(gamma)
        self.register_buffer("weight", weight if weight is not None else None,
                              persistent=False)
        self.label_smoothing = float(label_smoothing)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # standard CE w/ optional smoothing -> per-sample loss
        logp = F.log_softmax(logits, dim=-1)
        with torch.no_grad():
            n_class = logits.size(-1)
            if self.label_smoothing > 0:
                onehot = torch.full_like(logp, self.label_smoothing / (n_class - 1))
                onehot.scatter_(1, target.unsqueeze(1), 1.0 - self.label_smoothing)
            else:
                onehot = torch.zeros_like(logp).scatter_(1, target.unsqueeze(1), 1.0)
        ce = -(onehot * logp).sum(dim=-1)              # [N]
        # focal modulator using true-class prob
        pt = logp.gather(1, target.unsqueeze(1)).exp().squeeze(1)  # [N]
        focal = (1.0 - pt).clamp(min=1e-8) ** self.gamma
        loss = focal * ce                              # [N]
        if self.weight is not None:
            w = self.weight[target]
            loss = loss * w
        return loss.mean()


class MultiTaskCE(nn.Module):
    """Sum of per-task cross-entropy with optional class & task weights.

    If `focal_tasks` is provided, those tasks use FocalLoss(gamma=focal_gamma)
    instead of plain CrossEntropyLoss.
    """

    def __init__(
        self,
        task_loss_weights: Mapping[str, float],
        class_weights: Mapping[str, torch.Tensor] | None = None,
        label_smoothing: float | Mapping[str, float] = 0.0,
        focal_tasks: list[str] | None = None,
        focal_gamma: float = 2.0,
    ) -> None:
        super().__init__()
        self.task_loss_weights = dict(task_loss_weights)
        focal_tasks = set(focal_tasks or [])
        # Normalize label_smoothing to per-task dict; missing tasks -> 0.0
        if isinstance(label_smoothing, Mapping):
            ls_map: dict[str, float] = {t: float(label_smoothing.get(t, 0.0))
                                        for t in self.task_loss_weights}
        else:
            ls_val = float(label_smoothing)
            ls_map = {t: ls_val for t in self.task_loss_weights}
        self.label_smoothing_map = ls_map
        self.criteria = nn.ModuleDict()
        for task, w in self.task_loss_weights.items():
            cw = class_weights.get(task) if class_weights else None
            ls_t = ls_map[task]
            if task in focal_tasks:
                self.criteria[task] = FocalLoss(
                    gamma=focal_gamma, weight=cw, label_smoothing=ls_t
                )
            else:
                self.criteria[task] = nn.CrossEntropyLoss(
                    weight=cw, label_smoothing=ls_t
                )

    def forward(
        self,
        logits: Mapping[str, torch.Tensor],
        labels: Mapping[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        total = 0.0
        parts: dict[str, float] = {}
        for task, w in self.task_loss_weights.items():
            l = self.criteria[task](logits[task], labels[task])
            total = total + w * l
            parts[task] = float(l.detach().item())
        return total, parts
