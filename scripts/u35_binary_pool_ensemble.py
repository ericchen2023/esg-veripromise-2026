"""U35 — Phase 53: diverse binary-specialist pool -> T1/T3 ensemble.

Phase 52 (LB 0.6113) proved two things:
  1. TAPT improves the TEST binary tasks (T1 +0.0103, T3 +0.0054 vs c3) and its
     promise=No cascade also lifts macro T2/T4 (better N/A assignment).
  2. The binary tasks T1/T3 (combined weight 0.50) are NOT an irreducible
     ceiling -- the earlier verdict was an OOF-validation failure. They are the
     largest remaining recoverable LB headroom.

Binary OOF cannot be trusted (company leakage + adaptive overfit), so the lever
is a POOL of structurally-different, domain-adapted models whose T1/T3
softmax probabilities are averaged, while the macro tasks T2/T4 keep the
faithful c3 base (equal8 + prior-correction a=0.3). Each new ensemble is a
single LB probe.

Pool members (each a TAPT-style domain-adapted stem trained on combined-2000):
  - macbert-TAPT seed42        (Phase 49, cached probs already on disk)
  - macbert-TAPT seed2024/...  (Phase 53 config exp_p53_tapt_macbert_ms)
  - roberta-large-TAPT seed42  (Phase 53 config exp_p54_tapt_robertalarge)
  - electra-large-TAPT seed42  (Phase 53 config exp_p55_tapt_electralarge)

The script auto-discovers which members are available (cached probs OR trained
checkpoints), runs + caches 5-fold test inference for the rest, then builds:
  - one binary candidate per individual member (isolates each model's skill),
  - one full-pool binary candidate (mean of all members' T1/T3 probs).
All candidates reuse the EXACT constrained decode + macro base as the banked
0.6113 winner, and are schema-validated before writing.

Usage:
    # sanity: rebuild the 0.6113 winner from the single macbert-TAPT member
    python -m scripts.u35_binary_pool_ensemble --reproduce

    # after training new stems: infer (if needed) + build all candidates
    python -m scripts.u35_binary_pool_ensemble --build
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.data.dataset import ESGDataset, NUM_LABELS, TASKS, esg_collate
from src.tools.validate_submission import validate_submission_frame
from scripts.u18_decoding_experiments import train_priors
from scripts.u20_binary_prior_correction import prior_correct_per_task
from scripts.u17_phase42_test_inference import (
    BATCH_SIZE,
    POOLING,
    DROPOUT,
    SUBMISSION_COLUMNS,
    load_test_records,
    probs_to_records,
    write_submission,
)

OUT_DIR = Path("outputs/submissions")
TV_PROBS = OUT_DIR / "phase43_test_probs.npz"
C3 = OUT_DIR / "FROZEN_BEST_0.6037_c3.csv"
CURRENT_BEST = OUT_DIR / "phase51_tapt_binary_submission.csv"
CURRENT_BEST_MD5 = "6b41fab2398426af6feb6b6a0976269f"
BINARY_TASKS = ("promise_status", "evidence_status")

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

# Binary-specialist pool. cache = pre-computed test probs ({task: [n,C]} npz);
# when absent, infer from ckpt_root/fold{0..n_folds-1}/best.pt with `backbone`.
BINARY_POOL = [
    {
        "name": "tapt_macbert_s42",
        "cache": OUT_DIR / "phase50_tapt_test_probs.npz",
        "backbone": "outputs/tapt/macbert_esg",
        "ckpt_root": Path("outputs/checkpoints/p49_tapt_combo_best/seed42"),
        "max_len": 384,
        "n_folds": 5,
    },
    {
        "name": "tapt_macbert_s2024",
        "cache": OUT_DIR / "p53_tapt_macbert_s2024_test_probs.npz",
        "backbone": "outputs/tapt/macbert_esg",
        "ckpt_root": Path("outputs/checkpoints/p53_tapt_macbert_ms/seed2024"),
        "max_len": 384,
        "n_folds": 5,
    },
    {
        "name": "tapt_macbert_s20260417",
        "cache": OUT_DIR / "p53_tapt_macbert_s20260417_test_probs.npz",
        "backbone": "outputs/tapt/macbert_esg",
        "ckpt_root": Path("outputs/checkpoints/p53_tapt_macbert_ms/seed20260417"),
        "max_len": 384,
        "n_folds": 5,
    },
    {
        "name": "tapt_macbert_s33",
        "cache": OUT_DIR / "p53_tapt_macbert_s33_test_probs.npz",
        "backbone": "outputs/tapt/macbert_esg",
        "ckpt_root": Path("outputs/checkpoints/p53_tapt_macbert_ms/seed33"),
        "max_len": 384,
        "n_folds": 5,
    },
    {
        "name": "tapt_macbert_s2025",
        "cache": OUT_DIR / "p53_tapt_macbert_s2025_test_probs.npz",
        "backbone": "outputs/tapt/macbert_esg",
        "ckpt_root": Path("outputs/checkpoints/p53_tapt_macbert_ms/seed2025"),
        "max_len": 384,
        "n_folds": 5,
    },
    {
        "name": "tapt_robertalarge_s42",
        "cache": OUT_DIR / "p54_tapt_robertalarge_test_probs.npz",
        "backbone": "outputs/tapt/roberta_large_esg",
        "ckpt_root": Path("outputs/checkpoints/p54_tapt_robertalarge/seed42"),
        "max_len": 256,
        "n_folds": 5,
    },
    {
        "name": "tapt_electralarge_s42",
        "cache": OUT_DIR / "p55_tapt_electralarge_test_probs.npz",
        "backbone": "outputs/tapt/electra_large_esg",
        "ckpt_root": Path("outputs/checkpoints/p55_tapt_electralarge/seed42"),
        "max_len": 256,
        "n_folds": 5,
    },
]


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def load_equal8() -> dict[str, np.ndarray]:
    z = np.load(TV_PROBS)
    out: dict[str, np.ndarray] = {}
    for task in TASKS:
        acc = None
        for stem in TV_STEMS:
            arr = z[f"{stem}__{task}"].astype(np.float64)
            acc = arr.copy() if acc is None else acc + arr
        out[task] = acc / len(TV_STEMS)
    return out


@torch.no_grad()
def infer_member(member: dict, records, device) -> dict[str, np.ndarray]:
    """5-fold mean softmax probs for one pool member -> {task: [n, C_t]}."""
    from src.models.multitask import MultiTaskClassifier

    backbone = member["backbone"]
    tokenizer = AutoTokenizer.from_pretrained(backbone)
    ds = ESGDataset(records, tokenizer, max_length=member["max_len"], with_labels=False)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                        collate_fn=esg_collate, num_workers=0,
                        pin_memory=(device.type == "cuda"))
    n = len(records)
    acc = {t: np.zeros((n, NUM_LABELS[t]), dtype=np.float64) for t in TASKS}
    use_amp = device.type == "cuda"
    for fold in range(member["n_folds"]):
        ckpt = member["ckpt_root"] / f"fold{fold}" / "best.pt"
        if not ckpt.exists():
            raise FileNotFoundError(ckpt)
        model = MultiTaskClassifier(backbone=backbone, num_labels=NUM_LABELS,
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
        print(f"    [{member['name']}] fold{fold} done")
    for t in TASKS:
        acc[t] /= member["n_folds"]
    return acc


def get_member_probs(member: dict, records, device) -> dict[str, np.ndarray] | None:
    """Load cached probs, else infer from checkpoints, else None (unavailable)."""
    cache = Path(member["cache"])
    if cache.exists():
        z = np.load(cache)
        return {t: z[t].astype(np.float64) for t in TASKS}
    if not (member["ckpt_root"] / "fold0" / "best.pt").exists():
        return None
    if records is None:
        return None
    print(f"  [infer] {member['name']} (no cache; running 5-fold test inference)")
    probs = infer_member(member, records, device)
    np.savez_compressed(cache, **probs)
    print(f"  [cache] -> {cache.name}")
    return probs


def macro_base() -> dict[str, np.ndarray]:
    """c3 base = equal8 + prior-correction a=0.3 on T2/T4 (identical to banked c3)."""
    priors = train_priors()
    return prior_correct_per_task(
        load_equal8(), priors,
        {"verification_timeline": 0.3, "evidence_quality": 0.3},
    )


def binary_candidate(base: dict, members: list[dict]) -> dict[str, np.ndarray]:
    """Mean the binary T1/T3 probs over `members`; keep macro tasks at c3 base."""
    mixed = {t: base[t].copy() for t in TASKS}
    for t in BINARY_TASKS:
        mixed[t] = np.mean([m[t] for m in members], axis=0)
    return mixed


def emit(tag: str, mixed: dict, records, c3_df: pd.DataFrame) -> Path:
    df = write_submission(probs_to_records(mixed, records), OUT_DIR / f"{tag}_submission.csv")
    chk = df.copy()
    chk["verification_timeline"] = chk["verification_timeline"].replace(
        {"more_than_5_years": "longer_than_5_years"})
    rep = validate_submission_frame(chk[SUBMISSION_COLUMNS], mode="preds")
    if not rep.ok:
        raise RuntimeError(f"{tag} INVALID: {rep.errors[:5]}")
    diffs = {c: int((df[c].values != c3_df[c].values).sum()) for c in SUBMISSION_COLUMNS[1:]}
    print(f"[wrote] {tag}_submission.csv  valid={rep.ok}  diffs_vs_c3={diffs}")
    print(f"        T1={df['promise_status'].value_counts().to_dict()}  "
          f"T3={df['evidence_status'].value_counts().to_dict()}")
    return OUT_DIR / f"{tag}_submission.csv"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reproduce", action="store_true",
                    help="Rebuild 0.6113 from the single macbert-TAPT member; assert md5.")
    ap.add_argument("--build", action="store_true",
                    help="Infer (if needed) + build per-member and full-pool candidates.")
    args = ap.parse_args()
    if not (args.reproduce or args.build):
        ap.error("pass --reproduce and/or --build")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    records = load_test_records("vpesg4k_test_2000.csv")
    c3_df = pd.read_csv(C3, keep_default_na=False).sort_values("id").reset_index(drop=True)
    base = macro_base()

    if args.reproduce:
        m0 = BINARY_POOL[0]
        probs = get_member_probs(m0, records, device)
        assert probs is not None, f"reproduce needs {m0['cache']} or its checkpoints"
        mixed = binary_candidate(base, [probs])
        out = emit("u35_reproduce_binary", mixed, records, c3_df)
        got = md5(out)
        ok = (got == CURRENT_BEST_MD5)
        print(f"\n[reproduce] md5={got}  expected={CURRENT_BEST_MD5}  "
              f"{'PASS (bit-identical to 0.6113 winner)' if ok else 'FAIL'}")
        if not ok:
            raise SystemExit("reproduce mismatch -- pipeline drifted, do not trust new candidates")

    if args.build:
        avail: list[tuple[str, dict]] = []
        for m in BINARY_POOL:
            p = get_member_probs(m, records, device)
            if p is None:
                print(f"  [skip] {m['name']} (no cache, no checkpoints yet)")
                continue
            avail.append((m["name"], p))
        print(f"\n[pool] {len(avail)} available members: {[n for n,_ in avail]}")
        if not avail:
            raise SystemExit("no pool members available -- train Phase 53 stems first")

        # per-member binary candidates (isolate each model's binary skill on LB)
        for name, probs in avail:
            emit(f"u35_{name}_binary", binary_candidate(base, [probs]), records, c3_df)
        # full-pool binary candidate (mean of all available members' T1/T3)
        if len(avail) >= 2:
            emit("u35_pool_binary", binary_candidate(base, [p for _, p in avail]),
                 records, c3_df)
        print("\n[done] candidates in outputs/submissions/ (u35_*_submission.csv).")
        print("       Upload order: probe the full-pool first; if it regresses, fall")
        print("       back to the best individual member. Each upload is one LB probe.")


if __name__ == "__main__":
    main()
