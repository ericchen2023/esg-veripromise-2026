"""U32 — Phase 50: TAPT model on the TEST binary tasks (the only real lever).

Rigorous diagnosis (Phase 48 u28 + Phase 50 cascade probe) established:
  * The entire OOF->LB gap (0.699->0.604) is in the BINARY tasks T1 (+0.154)
    and T3 (+0.184); macro T2/T4 transfer faithfully.
  * T3 has NO legal post-hoc lever: schema forces promise=No -> evidence=N/A,
    and c3 already takes the optimal argmax{Yes,No} on promise=Yes rows. T3 is
    hard-capped by T1's ~21% test false-No rate.
  * Therefore the ONLY way to recover binary points is a MODEL that generalises
    better on the test binary tasks. OOF cannot measure this (binary OOF is
    untrustworthy + adaptively overfit), so it is strictly an LB experiment.

Phase 49 rejected the TAPT model on OOF (-0.0006), but that verdict was
dominated by the untrustworthy binary OOF. TAPT is domain-adapted (continued
MLM on the ESG corpus) and was the BEST single model ever on macro T2. It is
the cheapest structurally-different candidate to probe on the real test binary
tasks.

This script:
  1. Runs 5-fold test inference for the p49 TAPT stem (backbone=outputs/tapt/
     macbert_esg) and caches its probs.
  2. Builds two LB candidates from the cached 8-stem probs + TAPT:
       - equal9 : equal-weight mean over all 9 stems (per task).
       - tapt_only : TAPT stem alone (cleanest read on TAPT's test binary skill).
  3. Decodes with the identical constrained logic as the banked c3 and writes
     validated submissions.

Usage:
    python -m scripts.u32_phase50_tapt_test
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.data.dataset import ESGDataset, NUM_LABELS, TASKS, esg_collate
from scripts.u17_phase42_test_inference import (
    BATCH_SIZE,
    MAX_LEN,
    POOLING,
    DROPOUT,
    SUBMISSION_COLUMNS,
    load_test_records,
    probs_to_records,
    write_submission,
)

CACHED_PROBS = Path("outputs/submissions/phase43_test_probs.npz")
TAPT_BACKBONE = "outputs/tapt/macbert_esg"
TAPT_STEM = "p49_tapt_combo_best"
TAPT_CKPT_ROOT = Path("outputs/checkpoints") / TAPT_STEM / "seed42"
TAPT_PROBS_CACHE = Path("outputs/submissions/phase50_tapt_test_probs.npz")
OUT_DIR = Path("outputs/submissions")
N_FOLDS = 5

TV_STEMS = (
    "p2_combo_best_tv",
    "p2_combo_best_u10_pseudo_tv",
    "p2_combo_best_u10_pseudo_v2_tv",
    "p2_combo_best_u10_pseudo_v2_classw_focal_t4_g3_tv",
    "p2_combo_best_u10_pseudo_v3_classw_focal_t4_g3_tv",
    "p2_combo_best_classw_focal_u6pro_tv",
    "p2_combo_best_aug_plus_tv",
    "p2_combo_best_aug_plus_v2_tv",
)


@torch.no_grad()
def infer_tapt(records, device) -> dict[str, np.ndarray]:
    """5-fold mean softmax probs for the TAPT stem -> {task: [n, C_t]}."""
    from src.models.multitask import MultiTaskClassifier

    tokenizer = AutoTokenizer.from_pretrained(TAPT_BACKBONE)
    ds = ESGDataset(records, tokenizer, max_length=MAX_LEN, with_labels=False)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                        collate_fn=esg_collate, num_workers=0,
                        pin_memory=(device.type == "cuda"))
    n = len(records)
    acc = {t: np.zeros((n, NUM_LABELS[t]), dtype=np.float64) for t in TASKS}
    use_amp = device.type == "cuda"
    for fold in range(N_FOLDS):
        ckpt = TAPT_CKPT_ROOT / f"fold{fold}" / "best.pt"
        if not ckpt.exists():
            raise FileNotFoundError(ckpt)
        model = MultiTaskClassifier(backbone=TAPT_BACKBONE, num_labels=NUM_LABELS,
                                    pooling=POOLING, dropout=DROPOUT)
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model_state_dict"])
        model.to(device).eval()
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            for batch in loader:
                ids = batch["input_ids"].to(device, non_blocking=True)
                mask = batch["attention_mask"].to(device, non_blocking=True)
                idx = batch["_index"].cpu().numpy()
                logits = model(ids, mask)
                for t in TASKS:
                    acc[t][idx] += torch.softmax(logits[t].float(), -1).cpu().numpy()
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"  [tapt] fold{fold} done")
    for t in TASKS:
        acc[t] /= N_FOLDS
    return acc


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")
    records = load_test_records("vpesg4k_test_2000.csv")
    n = len(records)
    print(f"[data] {n} test records")

    # --- TAPT inference (cache to avoid recompute) ---
    if TAPT_PROBS_CACHE.exists():
        z = np.load(TAPT_PROBS_CACHE)
        tapt = {t: z[t] for t in TASKS}
        print(f"[tapt] loaded cached probs {TAPT_PROBS_CACHE.name}")
    else:
        print("[tapt] running 5-fold test inference (backbone=TAPT)")
        tapt = infer_tapt(records, device)
        np.savez_compressed(TAPT_PROBS_CACHE, **tapt)
        print(f"[tapt] cached -> {TAPT_PROBS_CACHE.name}")

    # --- load 8 cached stems ---
    d = np.load(CACHED_PROBS)
    per_stem = {s: {t: d[f"{s}__{t}"].astype(np.float64) for t in TASKS} for s in TV_STEMS}
    per_stem[TAPT_STEM] = tapt
    stems9 = list(per_stem.keys())
    print(f"[ensemble] {len(stems9)} stems: {stems9}")

    def blend(stem_list):
        out = {}
        for t in TASKS:
            arr = np.zeros_like(per_stem[stems9[0]][t])
            for s in stem_list:
                arr += per_stem[s][t]
            out[t] = arr / len(stem_list)
        return out

    candidates = {
        "phase50_equal9": blend(stems9),
        "phase50_tapt_only": blend([TAPT_STEM]),
    }

    for tag, mixed in candidates.items():
        constrained = probs_to_records(mixed, records)
        df = write_submission(constrained, OUT_DIR / f"{tag}_submission.csv")
        # validate (write_submission already aliased timeline; validate internal view)
        from src.tools.validate_submission import validate_submission_frame
        chk = df.copy()
        chk["verification_timeline"] = chk["verification_timeline"].replace(
            {"more_than_5_years": "longer_than_5_years"})
        rep = validate_submission_frame(chk[SUBMISSION_COLUMNS], mode="preds")
        dist = {t: df[t].value_counts().to_dict() for t in ("promise_status", "evidence_status")}
        print(f"[wrote] {tag}_submission.csv rows={len(df)} valid={rep.ok}")
        print(f"        T1={dist['promise_status']}  T3={dist['evidence_status']}")


if __name__ == "__main__":
    main()
