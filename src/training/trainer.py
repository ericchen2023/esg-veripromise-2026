"""Single-fold trainer for the multi-task ESG model.

Implements:
  - AMP / fp16 (torch.amp.autocast)
  - Gradient accumulation
  - Gradient clipping
  - Cosine LR schedule with warmup
  - Per-epoch validation with competition weighted score
  - Best-checkpoint saving
  - OOF probability persistence (np.float16)
  - JSONL logging
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.dataset import ID2LABEL, NUM_LABELS, TASKS
from src.eval.metrics import weighted_score
from src.training.losses import MultiTaskCE
from src.training.schedulers import build_scheduler


@dataclass
class FoldResult:
    fold: int
    seed: int
    best_epoch: int
    best_score: float
    per_task: dict[str, float]
    history: list[dict[str, Any]]
    ckpt_path: str
    oof_path: str


def _to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device, non_blocking=True)
        elif isinstance(v, dict):
            out[k] = {kk: vv.to(device, non_blocking=True) for kk, vv in v.items()}
        else:
            out[k] = v
    return out


def _build_param_groups(
    model: nn.Module, lr: float, wd: float, llrd_decay: float | None
) -> list[dict]:
    """Build optimizer param groups.

    If `llrd_decay` is set (e.g. 0.95), apply Layer-wise LR Decay across encoder
    layers (deeper layers get the configured `lr`; shallower layers shrink by
    `decay**(depth)`). Heads / pooler stay at base `lr`. Bias / LayerNorm get wd=0.
    """
    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")

    def _wd_for(n: str) -> float:
        return 0.0 if any(nd in n for nd in no_decay) else wd

    if not llrd_decay:
        return [
            {"params": [p for n, p in model.named_parameters() if _wd_for(n) > 0], "weight_decay": wd, "lr": lr},
            {"params": [p for n, p in model.named_parameters() if _wd_for(n) == 0], "weight_decay": 0.0, "lr": lr},
        ]

    # LLRD: figure out encoder layer count
    encoder = getattr(model, "encoder", None)
    n_layers = 0
    try:
        n_layers = len(encoder.encoder.layer)  # type: ignore[attr-defined]
    except Exception:
        n_layers = 12  # fallback for BERT-base

    groups: list[dict] = []
    seen: set[int] = set()

    def _add(params, group_lr: float, group_wd: float) -> None:
        params = [p for p in params if id(p) not in seen and p.requires_grad]
        for p in params:
            seen.add(id(p))
        if params:
            groups.append({"params": params, "lr": group_lr, "weight_decay": group_wd})

    # heads + pooler at base lr
    for n, p in model.named_parameters():
        if n.startswith("encoder."):
            continue
        _add([p], lr, _wd_for(n))

    # encoder embeddings -> shallowest (depth = n_layers + 1)
    emb_lr = lr * (llrd_decay ** (n_layers + 1))
    for n, p in model.named_parameters():
        if n.startswith("encoder.embeddings"):
            _add([p], emb_lr, _wd_for(n))

    # per-layer (layer 0 deepest depth, last layer shallowest depth=1)
    for li in range(n_layers):
        depth = n_layers - li  # layer 0 -> depth=n, layer n-1 -> depth=1
        layer_lr = lr * (llrd_decay ** depth)
        prefix = f"encoder.encoder.layer.{li}."
        for n, p in model.named_parameters():
            if n.startswith(prefix):
                _add([p], layer_lr, _wd_for(n))

    # remaining encoder params (e.g. encoder.pooler) -> base lr
    for n, p in model.named_parameters():
        if n.startswith("encoder.") and id(p) not in seen:
            _add([p], lr, _wd_for(n))

    return groups


class _EMA:
    """Exponential moving average of model params; stored on CPU to save VRAM."""

    def __init__(self, model: nn.Module, decay: float) -> None:
        self.decay = float(decay)
        self.shadow: dict[str, torch.Tensor] = {}
        self.backup: dict[str, torch.Tensor] = {}
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n] = p.detach().clone()

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        d = self.decay
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            s = self.shadow.get(n)
            if s is None:
                self.shadow[n] = p.detach().clone()
            else:
                s.mul_(d).add_(p.detach(), alpha=1.0 - d)

    @torch.no_grad()
    def apply_to(self, model: nn.Module) -> None:
        self.backup = {}
        for n, p in model.named_parameters():
            if n in self.shadow:
                self.backup[n] = p.detach().clone()
                p.data.copy_(self.shadow[n].data)

    @torch.no_grad()
    def restore(self, model: nn.Module) -> None:
        for n, p in model.named_parameters():
            if n in self.backup:
                p.data.copy_(self.backup[n].data)
        self.backup = {}


class _FGM:
    """Fast Gradient Method adversarial training on word embeddings."""

    def __init__(self, model: nn.Module, eps: float = 1.0,
                 emb_name_substr: str = "word_embeddings") -> None:
        self.model = model
        self.eps = float(eps)
        self.emb_name_substr = emb_name_substr
        self.backup: dict[str, torch.Tensor] = {}

    def attack(self) -> None:
        for n, p in self.model.named_parameters():
            if p.requires_grad and self.emb_name_substr in n and p.grad is not None:
                self.backup[n] = p.data.clone()
                norm = torch.norm(p.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = self.eps * p.grad / norm
                    p.data.add_(r_at)

    def restore(self) -> None:
        for n, p in self.model.named_parameters():
            if n in self.backup:
                p.data.copy_(self.backup[n])
        self.backup = {}


def _predict(
    model: nn.Module, loader: DataLoader, device: torch.device, use_amp: bool
) -> tuple[dict[str, np.ndarray], list[int]]:
    """Return per-task softmax probability matrices [N, C_t] and original indices."""
    model.eval()
    probs: dict[str, list[np.ndarray]] = {t: [] for t in TASKS}
    indices: list[int] = []
    autocast_ctx = torch.amp.autocast(device_type=device.type, enabled=use_amp)
    with torch.no_grad(), autocast_ctx:
        for batch in loader:
            batch = _to_device(batch, device)
            logits = model(batch["input_ids"], batch["attention_mask"])
            for t in TASKS:
                p = torch.softmax(logits[t].float(), dim=-1).cpu().numpy()
                probs[t].append(p)
            indices.extend(batch["_index"].cpu().tolist())
    return {t: np.concatenate(probs[t], axis=0) for t in TASKS}, indices


def train_fold(
    *,
    fold: int,
    seed: int,
    train_records: list[dict],
    val_records: list[dict],
    model: nn.Module,
    tokenizer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: dict,
    out_root: Path,
    log_path: Path,
    val_global_indices: list[int] | None = None,
) -> FoldResult:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    tcfg = cfg["training"]
    epochs = int(tcfg["epochs"])
    lr = float(tcfg["lr"])
    wd = float(tcfg.get("weight_decay", 0.0))
    grad_accum = int(tcfg.get("grad_accum", 1))
    warmup_ratio = float(tcfg.get("warmup_ratio", 0.1))
    grad_clip = float(tcfg.get("grad_clip", 1.0))
    use_amp = bool(tcfg.get("use_amp", True)) and device.type == "cuda"
    sched_name = tcfg.get("scheduler", "cosine")
    # label_smoothing may be a float (global) OR a dict {task_name: float}
    label_smoothing_cfg = tcfg.get("label_smoothing", 0.0)
    if isinstance(label_smoothing_cfg, dict):
        label_smoothing: float | dict = {k: float(v) for k, v in label_smoothing_cfg.items()}
    else:
        label_smoothing = float(label_smoothing_cfg)
    log_every = int(cfg.get("runtime", {}).get("log_every", 50))

    # Phase B options (all default off / no-op)
    llrd_decay = tcfg.get("llrd_decay")  # e.g. 0.95
    if llrd_decay is not None:
        llrd_decay = float(llrd_decay)
    ema_decay = tcfg.get("ema_decay")    # e.g. 0.999
    if ema_decay is not None:
        ema_decay = float(ema_decay)
    # U2: optional warm-start — skip EMA updates for first N epochs and rebuild
    # shadow weights at the start of epoch N+1. Defaults to 0 (legacy behaviour).
    ema_warmup_epochs = int(tcfg.get("ema_warmup_epochs", 0) or 0)
    fgm_eps = tcfg.get("fgm_eps")        # e.g. 1.0
    if fgm_eps is not None:
        fgm_eps = float(fgm_eps)
    focal_tasks = list(tcfg.get("focal_tasks", []) or [])
    focal_gamma = float(tcfg.get("focal_gamma", 2.0))
    rdrop_alpha = float(tcfg.get("rdrop_alpha", 0.0) or 0.0)

    # class weights
    class_weights = None
    if bool(tcfg.get("use_class_weight", False)):
        from src.training.losses import compute_class_weights

        class_weights = {}
        from src.data.dataset import LABEL_DOMAINS

        for task, domain in LABEL_DOMAINS.items():
            ys = [r.get(task, "N/A") for r in train_records]
            class_weights[task] = compute_class_weights(ys, domain).to(device)

    criterion = MultiTaskCE(
        task_loss_weights=tcfg.get("task_loss_weights", {t: 1.0 for t in TASKS}),
        class_weights=class_weights,
        label_smoothing=label_smoothing,
        focal_tasks=focal_tasks,
        focal_gamma=focal_gamma,
    ).to(device)

    params = _build_param_groups(model, lr=lr, wd=wd, llrd_decay=llrd_decay)
    optimizer = torch.optim.AdamW(params, lr=lr)

    steps_per_epoch = max(1, len(train_loader) // grad_accum)
    total_steps = steps_per_epoch * epochs
    warmup_steps = max(1, int(warmup_ratio * total_steps))
    scheduler = build_scheduler(sched_name, optimizer, warmup_steps, total_steps)

    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    ema = _EMA(model, ema_decay) if ema_decay else None
    ema_started = False  # becomes True at first epoch past ema_warmup_epochs
    fgm = _FGM(model, eps=fgm_eps) if fgm_eps else None

    out_root.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_fp = log_path.open("a", encoding="utf-8")

    def log(rec: dict) -> None:
        log_fp.write(json.dumps(rec, ensure_ascii=False) + "\n")
        log_fp.flush()

    history: list[dict] = []
    best_score = -1.0
    best_epoch = -1
    ckpt_path = out_root / "best.pt"
    oof_path = out_root / "oof_probs.npz"

    # U3 SWA: optionally save the last K epoch checkpoints for weight-space
    # averaging via src/tools/u3_swa_aggregate.py. Off by default.
    swa_last_k = int(tcfg.get("swa_last_k", 0) or 0)

    val_truth = {t: [r[t] for r in val_records] for t in TASKS}

    for epoch in range(1, epochs + 1):
        # U2 warm-start: re-initialise EMA shadow at start of first post-warmup
        # epoch so the average tracks only converged weights.
        if ema is not None and not ema_started and epoch > ema_warmup_epochs:
            for n, p in model.named_parameters():
                if p.requires_grad:
                    ema.shadow[n] = p.detach().clone()
            ema_started = True
        model.train()
        t0 = time.time()
        running = 0.0
        n_seen = 0
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(train_loader, start=1):
            batch = _to_device(batch, device)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                logits = model(batch["input_ids"], batch["attention_mask"])
                loss, _ = criterion(logits, batch["labels"])
                # R-Drop: second forward + symmetric KL on each task's logits
                if rdrop_alpha > 0:
                    logits2 = model(batch["input_ids"], batch["attention_mask"])
                    loss2, _ = criterion(logits2, batch["labels"])
                    kl_total = 0.0
                    for t in TASKS:
                        p = torch.log_softmax(logits[t].float(), dim=-1)
                        q = torch.log_softmax(logits2[t].float(), dim=-1)
                        kl_pq = torch.nn.functional.kl_div(p, q, reduction="batchmean", log_target=True)
                        kl_qp = torch.nn.functional.kl_div(q, p, reduction="batchmean", log_target=True)
                        kl_total = kl_total + 0.5 * (kl_pq + kl_qp)
                    loss = 0.5 * (loss + loss2) + rdrop_alpha * kl_total
                loss = loss / grad_accum

            scaler.scale(loss).backward()

            # FGM adversarial: extra forward-backward on perturbed embeddings
            if fgm is not None:
                fgm.attack()
                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    logits_adv = model(batch["input_ids"], batch["attention_mask"])
                    loss_adv, _ = criterion(logits_adv, batch["labels"])
                    loss_adv = loss_adv / grad_accum
                scaler.scale(loss_adv).backward()
                fgm.restore()

            if step % grad_accum == 0:
                if grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                if ema is not None and ema_started:
                    ema.update(model)

            running += float(loss.item()) * grad_accum * batch["input_ids"].size(0)
            n_seen += batch["input_ids"].size(0)
            if step % log_every == 0:
                cur = running / max(1, n_seen)
                lr_now = scheduler.get_last_lr()[0]
                log({"event": "step", "fold": fold, "seed": seed, "epoch": epoch,
                     "step": step, "loss": cur, "lr": lr_now})

        train_loss = running / max(1, n_seen)

        # Validation (use EMA weights if enabled and warm-up has finished)
        if ema is not None and ema_started:
            ema.apply_to(model)
        probs, idx = _predict(model, val_loader, device, use_amp)
        # idx is a permutation of range(len(val_records)) but loader is not shuffled
        # we still rebuild predictions in original order via idx
        order = np.argsort(idx)
        ordered_probs = {t: probs[t][order] for t in TASKS}
        preds_id = {t: ordered_probs[t].argmax(axis=1) for t in TASKS}
        preds_text = {t: [ID2LABEL[t][int(i)] for i in preds_id[t]] for t in TASKS}
        score_dict = weighted_score(val_truth, preds_text)

        epoch_rec = {
            "event": "epoch",
            "fold": fold,
            "seed": seed,
            "epoch": epoch,
            "train_loss": train_loss,
            "elapsed_sec": round(time.time() - t0, 2),
            "scores": score_dict,
        }
        history.append(epoch_rec)
        log(epoch_rec)
        print(
            f"[fold={fold} seed={seed}] epoch {epoch}/{epochs} "
            f"loss={train_loss:.4f} score={score_dict['final_weighted_score']:.5f} "
            f"({round(time.time()-t0,1)}s)"
        )

        if score_dict["final_weighted_score"] > best_score:
            best_score = score_dict["final_weighted_score"]
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "score": best_score,
                    "scores": score_dict,
                    "fold": fold,
                    "seed": seed,
                },
                ckpt_path,
            )
            # Save OOF probs (float16 to save space)
            local_sorted = np.asarray(idx)[order].astype(np.int32)
            if val_global_indices is not None:
                global_sorted = np.asarray(
                    [val_global_indices[i] for i in local_sorted], dtype=np.int32
                )
            else:
                global_sorted = local_sorted
            np.savez_compressed(
                oof_path,
                indices=global_sorted,
                local_indices=local_sorted,
                **{f"probs_{t}": ordered_probs[t].astype(np.float16) for t in TASKS},
                meta=json.dumps(
                    {
                        "fold": fold,
                        "seed": seed,
                        "best_epoch": epoch,
                        "score": float(best_score),
                        "tasks": list(TASKS),
                        "label_domains": {t: list(ID2LABEL[t].values()) for t in TASKS},
                    },
                    ensure_ascii=False,
                ),
            )

        # restore live weights after EMA-eval
        if ema is not None and ema_started:
            ema.restore(model)

        # U3 SWA: save last-K epoch state dicts (raw model weights, no optim).
        # Saved files: out_root/swa_epoch{epoch}.pt with key "model_state_dict".
        if swa_last_k > 0 and epoch > epochs - swa_last_k:
            swa_path = out_root / f"swa_epoch{epoch}.pt"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "scores": score_dict,
                    "fold": fold,
                    "seed": seed,
                },
                swa_path,
            )

    log_fp.close()

    per_task = {t: float(history[best_epoch - 1]["scores"][t]) for t in TASKS}
    return FoldResult(
        fold=fold,
        seed=seed,
        best_epoch=best_epoch,
        best_score=float(best_score),
        per_task=per_task,
        history=history,
        ckpt_path=str(ckpt_path),
        oof_path=str(oof_path),
    )


# Re-export for callers
__all__ = ["FoldResult", "train_fold", "NUM_LABELS"]
