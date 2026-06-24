"""U17 — Phase 42 test-set inference + TV 8-stem ensemble submission.

The official test set (``vpesg4k_test_2000.csv``, ids 12001..14000) was released
2026-06-11. This script produces the final competition submission by:

  1. Loading the 2,000-row test set (id + data only, no labels).
  2. For each of the 8 Phase 41 TV stems, running stored-view inference with all
     5 fold checkpoints (``best.pt``) and averaging the per-task softmax probs.
  3. Blending stems per task using the warm-start weights from
     ``reports/analysis/_ensemble/tv_oof_ensemble_meta.json`` (U16 joint
     hillclimb, OOF=0.71033).
  4. argmax -> apply_constraints_batch (hierarchical label coupling).
  5. Writing an internal preds CSV (canonical labels) + validating it, then the
     final submission CSV (``more_than_5_years`` remap, literal "N/A", exact
     5-column order, 2000 rows).

Also writes two fallback submissions: equal-weight blend and best-single-stem.

All 8 TV stems share one architecture: hfl/chinese-macbert-base, cls_mean
pooling, max_length 384, no added vocab tokens — so a single tokenizer/model
spec is reused for every checkpoint.

Usage:
    python -m scripts.u17_phase42_test_inference
    python -m scripts.u17_phase42_test_inference --test-csv vpesg4k_test_2000.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.data.dataset import ESGDataset, ID2LABEL, NUM_LABELS, TASKS, esg_collate
from src.inference.post_process import apply_constraints_batch
from src.tools.validate_submission import validate_submission_frame

CKPT_ROOT = Path("outputs/checkpoints")
META_PATH = Path("reports/analysis/_ensemble/tv_oof_ensemble_meta.json")
OUT_DIR = Path("outputs/submissions")
SEED_DIR = "seed42"
N_FOLDS = 5

BACKBONE = "hfl/chinese-macbert-base"
POOLING = "cls_mean"
DROPOUT = 0.1
MAX_LEN = 384
BATCH_SIZE = 32

# 8 AP-D4 TV stems, ordered exactly as in u16_tv_oof_ensemble.py / the meta json.
TV_STEMS: tuple[str, ...] = (
    "p2_combo_best_tv",
    "p2_combo_best_u10_pseudo_tv",
    "p2_combo_best_u10_pseudo_v2_tv",
    "p2_combo_best_u10_pseudo_v2_classw_focal_t4_g3_tv",
    "p2_combo_best_u10_pseudo_v3_classw_focal_t4_g3_tv",
    "p2_combo_best_classw_focal_u6pro_tv",
    "p2_combo_best_aug_plus_tv",
    "p2_combo_best_aug_plus_v2_tv",
)

# Canonical training label -> official submission label.
SUBMISSION_ALIASES: dict[str, dict[str, str]] = {
    "verification_timeline": {"longer_than_5_years": "more_than_5_years"},
}
SUBMISSION_COLUMNS = ["id", "promise_status", "verification_timeline",
                      "evidence_status", "evidence_quality"]


def load_test_records(test_csv: str) -> list[dict]:
    """Load test CSV (id + data only); no label columns expected."""
    df = pd.read_csv(test_csv, encoding="utf-8")
    if "id" not in df.columns or "data" not in df.columns:
        raise ValueError(f"test CSV must have 'id' and 'data' columns, got {list(df.columns)}")
    records = [{"id": int(r["id"]), "data": str(r["data"])} for r in df.to_dict("records")]
    return records


@torch.no_grad()
def infer_stem_fold(
    ckpt_path: Path,
    loader: DataLoader,
    device: torch.device,
    n: int,
) -> dict[str, np.ndarray]:
    """Load one fold checkpoint, return per-task softmax probs [n, C_t] in record order."""
    from src.models.multitask import MultiTaskClassifier

    model = MultiTaskClassifier(
        backbone=BACKBONE, num_labels=NUM_LABELS, pooling=POOLING, dropout=DROPOUT
    )
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    model.to(device).eval()

    probs = {t: np.zeros((n, NUM_LABELS[t]), dtype=np.float64) for t in TASKS}
    use_amp = device.type == "cuda"
    autocast_ctx = torch.amp.autocast(device_type=device.type, enabled=use_amp)
    with autocast_ctx:
        for batch in loader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            mask = batch["attention_mask"].to(device, non_blocking=True)
            idx = batch["_index"].cpu().numpy()
            logits = model(input_ids, mask)
            for t in TASKS:
                p = torch.softmax(logits[t].float(), dim=-1).cpu().numpy()
                probs[t][idx] = p

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return probs


def infer_all_stems(records: list[dict], device: torch.device) -> dict[str, dict[str, np.ndarray]]:
    """Return {stem: {task: [n, C_t]}} = 5-fold mean stored-view probs per stem."""
    tokenizer = AutoTokenizer.from_pretrained(BACKBONE)
    ds = ESGDataset(records, tokenizer, max_length=MAX_LEN, with_labels=False)
    loader = DataLoader(
        ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=esg_collate,
        num_workers=0, pin_memory=(device.type == "cuda"),
    )
    n = len(records)

    per_stem: dict[str, dict[str, np.ndarray]] = {}
    for stem in TV_STEMS:
        acc = {t: np.zeros((n, NUM_LABELS[t]), dtype=np.float64) for t in TASKS}
        folds_used = 0
        for fold in range(N_FOLDS):
            ckpt = CKPT_ROOT / stem / SEED_DIR / f"fold{fold}" / "best.pt"
            if not ckpt.exists():
                raise FileNotFoundError(f"missing checkpoint: {ckpt}")
            fold_probs = infer_stem_fold(ckpt, loader, device, n)
            for t in TASKS:
                acc[t] += fold_probs[t]
            folds_used += 1
        for t in TASKS:
            acc[t] /= folds_used
        per_stem[stem] = acc
        print(f"  [infer] {stem:52s} {folds_used} folds averaged")
    return per_stem


def mix(per_stem: dict[str, dict[str, np.ndarray]],
        stem_weights_per_task: dict[str, list[float]]) -> dict[str, np.ndarray]:
    """Blend stems per task -> {task: [n, C_t]}; weights renormalized."""
    out: dict[str, np.ndarray] = {}
    for t in TASKS:
        ws = stem_weights_per_task[t]
        s = float(sum(ws))
        assert s > 0, f"zero stem weights for task {t}"
        arr = np.zeros_like(per_stem[TV_STEMS[0]][t], dtype=np.float64)
        for stem, w in zip(TV_STEMS, ws):
            if w == 0.0:
                continue
            arr += w * per_stem[stem][t]
        out[t] = arr / s
    return out


def probs_to_records(mixed: dict[str, np.ndarray], records: list[dict]) -> list[dict]:
    """Constrained decode of mixed probs into spec-compliant label records.

    The official logic table (submission guidelines §3) requires:
      - promise=No  -> timeline = evidence_status = quality = N/A
      - promise=Yes -> timeline != N/A (one of the 4 dated buckets)
                       evidence_status in {Yes, No}  (not N/A)
                       evidence=Yes -> quality in {Clear, Not Clear, Misleading}
                       evidence=No  -> quality = N/A

    Ground-truth promise=Yes rows never carry an N/A timeline/evidence/quality,
    so restricting the argmax to the non-N/A classes for those rows is always
    >= plain argmax in expected score while guaranteeing validity. The
    canonical `longer_than_5_years` label is kept here; the submission writer
    remaps it to `more_than_5_years`.
    """
    from src.data.dataset import LABEL2ID

    tl_na = LABEL2ID["verification_timeline"]["N/A"]
    es_na = LABEL2ID["evidence_status"]["N/A"]
    eq_na = LABEL2ID["evidence_quality"]["N/A"]
    es_yes = LABEL2ID["evidence_status"]["Yes"]

    def _argmax_exclude(p: np.ndarray, exclude: int) -> int:
        masked = p.copy()
        masked[exclude] = -np.inf
        return int(masked.argmax())

    out: list[dict] = []
    for i, rec in enumerate(records):
        promise = ID2LABEL["promise_status"][int(mixed["promise_status"][i].argmax())]
        row = {"id": rec["id"], "promise_status": promise}
        if promise == "No":
            row["verification_timeline"] = "N/A"
            row["evidence_status"] = "N/A"
            row["evidence_quality"] = "N/A"
        else:
            tl_idx = _argmax_exclude(mixed["verification_timeline"][i], tl_na)
            row["verification_timeline"] = ID2LABEL["verification_timeline"][tl_idx]
            es_idx = _argmax_exclude(mixed["evidence_status"][i], es_na)
            row["evidence_status"] = ID2LABEL["evidence_status"][es_idx]
            if es_idx == es_yes:
                eq_idx = _argmax_exclude(mixed["evidence_quality"][i], eq_na)
                row["evidence_quality"] = ID2LABEL["evidence_quality"][eq_idx]
            else:
                row["evidence_quality"] = "N/A"
        out.append(row)
    # Safety net (idempotent given the constrained decode above).
    return apply_constraints_batch(out)


def write_submission(constrained: list[dict], path: Path) -> pd.DataFrame:
    """Write the official 5-column submission CSV (more_than_5_years remap, literal N/A)."""
    df = pd.DataFrame(constrained)[SUBMISSION_COLUMNS].copy()
    df = df.sort_values("id").reset_index(drop=True)
    for col, mapping in SUBMISSION_ALIASES.items():
        df[col] = df[col].replace(mapping)
    # Ensure every cell is a non-blank string ("N/A" stays literal).
    for col in SUBMISSION_COLUMNS[1:]:
        df[col] = df[col].astype(str)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 42 test inference + ensemble submission")
    ap.add_argument("--test-csv", default="vpesg4k_test_2000.csv")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    print(f"[device] {device}")

    records = load_test_records(args.test_csv)
    n = len(records)
    ids = [r["id"] for r in records]
    print(f"[data] loaded {n} test records (id {min(ids)}..{max(ids)})")

    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    stem_w = meta["stem_weights_per_task"]
    print(f"[meta] loaded TV ensemble weights (OOF={meta['final_score']:.5f})")

    print(f"[infer] running 8 stems x {N_FOLDS} folds on {n} test rows")
    per_stem = infer_all_stems(records, device)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Primary: per-task hillclimb-weighted ensemble ---
    mixed = mix(per_stem, stem_w)
    constrained = probs_to_records(mixed, records)
    preds_df = pd.DataFrame(constrained)[["id", *TASKS]].sort_values("id").reset_index(drop=True)
    preds_path = OUT_DIR / "phase42_tv_ensemble_preds.csv"
    preds_df.to_csv(preds_path, index=False, encoding="utf-8")
    rep = validate_submission_frame(preds_df, mode="preds")
    print(f"[validate preds] ok={rep.ok} rows={rep.rows} errors={list(rep.errors)[:5]}")
    if not rep.ok:
        raise RuntimeError(f"preds validation failed: {rep.errors[:10]}")

    sub_path = OUT_DIR / "phase42_tv_ensemble_submission.csv"
    sub_df = write_submission(constrained, sub_path)
    print(f"[wrote] {sub_path} (rows={len(sub_df)})")

    # --- Fallback 1: equal-weight blend ---
    equal_w = {t: [1.0] * len(TV_STEMS) for t in TASKS}
    eq_constrained = probs_to_records(mix(per_stem, equal_w), records)
    eq_path = OUT_DIR / "phase42_equalweight_submission.csv"
    write_submission(eq_constrained, eq_path)
    print(f"[wrote] {eq_path}")

    # --- Fallback 2: best single stem (per meta) ---
    best_stem = meta["best_single_stem"]
    best_only = {t: [1.0 if s == best_stem else 0.0 for s in TV_STEMS] for t in TASKS}
    bs_constrained = probs_to_records(mix(per_stem, best_only), records)
    bs_path = OUT_DIR / "phase42_bestsingle_submission.csv"
    write_submission(bs_constrained, bs_path)
    print(f"[wrote] {bs_path} (best_single={best_stem})")

    # --- Label distribution sanity ---
    print("\n[submission label distribution — primary ensemble]")
    for t in TASKS:
        vc = sub_df[t].value_counts().to_dict()
        print(f"  {t:24s}: {vc}")

    # Final structural checks on the primary submission.
    assert len(sub_df) == n, f"expected {n} rows, got {len(sub_df)}"
    assert list(sub_df["id"]) == sorted(ids), "id order mismatch"
    assert list(sub_df.columns) == SUBMISSION_COLUMNS, "column order mismatch"
    assert not sub_df.isna().any().any(), "submission contains NaN/blank"
    valid_tl = set(SUBMISSION_COLUMNS) and {"already", "within_2_years",
        "between_2_and_5_years", "more_than_5_years", "N/A"}
    bad_tl = set(sub_df["verification_timeline"]) - valid_tl
    assert not bad_tl, f"unexpected timeline labels: {bad_tl}"
    print("\n[ok] primary submission passed all structural checks")
    print(f"[done] primary submission -> {sub_path}")


if __name__ == "__main__":
    main()
