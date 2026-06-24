"""M4 Stage A — Per-task TTA pseudo-label admission.

Loads the p2_combo_best ensemble (3 seeds x 5 folds) and runs inference on
data/processed/u10/corpus.jsonl. Averages softmax probabilities across all
checkpoints, then admits an example only when ALL FOUR per-task max-probs
meet their thresholds. Writes a CSV compatible with src/train_pseudo_kfold.py.

Per-task admission thresholds (per master plan):
    promise_status         T1 >= 0.80
    verification_timeline  T2 >= 0.60
    evidence_status        T3 >= 0.70
    evidence_quality       T4 >= 0.60
"""
from __future__ import annotations
import sys, io, os, json, argparse, time
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.dataset import ESGDataset, ID2LABEL, NUM_LABELS, TASKS, esg_collate
from src.models.multitask import MultiTaskClassifier

THRESHOLDS = {
    "promise_status": 0.80,
    "verification_timeline": 0.60,
    "evidence_status": 0.70,
    "evidence_quality": 0.60,
}

BACKBONE = "hfl/chinese-macbert-base"
MAX_LEN  = 384
BATCH    = 32
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP  = DEVICE.type == "cuda"

CKPT_ROOT = ROOT / "outputs" / "checkpoints" / "p2_combo_best"
SEEDS = [42, 2024, 20260417]
N_FOLDS = 5
CORPUS_JSONL = ROOT / "data" / "processed" / "u10" / "corpus.jsonl"
OUT_CSV = ROOT / "data" / "processed" / "u10" / "pseudo_labels.csv"
RPT_JSON = ROOT / "reports" / "experiments" / "u10" / "pseudo_label_stats.json"


def load_corpus():
    recs = []
    with open(CORPUS_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            r["id"] = 100000 + len(recs)  # avoid collision with labeled 10001-11000
            recs.append(r)
    return recs


def predict_probs(model, loader):
    """Return per-task array (N, n_labels)."""
    model.eval()
    bufs = {t: [] for t in TASKS}
    idxs = []
    autocast = torch.amp.autocast(device_type=DEVICE.type, enabled=USE_AMP)
    with torch.no_grad(), autocast:
        for batch in loader:
            ids = batch["input_ids"].to(DEVICE, non_blocking=True)
            mk  = batch["attention_mask"].to(DEVICE, non_blocking=True)
            logits = model(ids, mk)
            for t in TASKS:
                bufs[t].append(torch.softmax(logits[t].float(), -1).cpu().numpy())
            idxs.extend(batch["_index"].cpu().tolist())
    out = {t: np.concatenate(bufs[t], 0) for t in TASKS}
    order = np.argsort(idxs)
    return {t: out[t][order] for t in TASKS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--folds", type=int, default=N_FOLDS)
    args = ap.parse_args()

    print(f"Device: {DEVICE}, AMP: {USE_AMP}")
    records = load_corpus()
    print(f"Corpus records: {len(records)}")

    tokenizer = AutoTokenizer.from_pretrained(BACKBONE)
    ds = ESGDataset(records, tokenizer, max_length=MAX_LEN, text_field="text", with_labels=False)
    loader = DataLoader(ds, batch_size=BATCH, shuffle=False, collate_fn=esg_collate,
                        num_workers=0, pin_memory=True)

    sum_probs = {t: None for t in TASKS}
    n_models = 0
    for seed in args.seeds:
        for fi in range(args.folds):
            ckpt = CKPT_ROOT / f"seed{seed}" / f"fold{fi}" / "best.pt"
            if not ckpt.exists():
                print(f"  MISSING {ckpt}"); continue
            t0 = time.time()
            model = MultiTaskClassifier(backbone=BACKBONE, num_labels=NUM_LABELS,
                                        pooling="cls_mean", dropout=0.1, msd_k=1)
            payload = torch.load(ckpt, map_location="cpu", weights_only=False)
            state = payload.get("model_state_dict", payload)
            model.load_state_dict(state, strict=True)
            model.to(DEVICE)
            probs = predict_probs(model, loader)
            for t in TASKS:
                sum_probs[t] = probs[t] if sum_probs[t] is None else sum_probs[t] + probs[t]
            n_models += 1
            print(f"  seed={seed} fold={fi}  {time.time()-t0:.1f}s  (cum={n_models})", flush=True)
            del model; torch.cuda.empty_cache()

    if n_models == 0:
        raise RuntimeError("No checkpoints loaded")
    avg_probs = {t: sum_probs[t] / n_models for t in TASKS}
    print(f"\nAveraged over {n_models} checkpoints")

    # Per-task max prob & predicted label
    per_task_pred = {}
    per_task_conf = {}
    for t in TASKS:
        per_task_pred[t] = avg_probs[t].argmax(-1)
        per_task_conf[t] = avg_probs[t].max(-1)

    # Admission: ALL 4 tasks must meet threshold
    pass_mask = np.ones(len(records), dtype=bool)
    per_task_pass = {}
    for t in TASKS:
        m = per_task_conf[t] >= THRESHOLDS[t]
        per_task_pass[t] = int(m.sum())
        pass_mask &= m

    admitted = int(pass_mask.sum())
    print(f"\nAdmission per task: {per_task_pass}")
    print(f"Final admitted (all 4 pass): {admitted}/{len(records)}")

    # Build CSV rows
    import csv
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fields = ["id","data","esg_type","promise_status","promise_string",
              "verification_timeline","evidence_status","evidence_string",
              "evidence_quality","company","ticker","page_number","pdf_url",
              "company_source","confidence_min","conf_T1","conf_T2","conf_T3","conf_T4"]
    n_written = 0
    label_dist = {t: {} for t in TASKS}
    with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for i, r in enumerate(records):
            if not pass_mask[i]:
                continue
            row = {
                "id": r["id"],
                "data": r["text"],
                "esg_type": "",
                "promise_status": ID2LABEL["promise_status"][int(per_task_pred["promise_status"][i])],
                "promise_string": "",
                "verification_timeline": ID2LABEL["verification_timeline"][int(per_task_pred["verification_timeline"][i])],
                "evidence_status": ID2LABEL["evidence_status"][int(per_task_pred["evidence_status"][i])],
                "evidence_string": "",
                "evidence_quality": ID2LABEL["evidence_quality"][int(per_task_pred["evidence_quality"][i])],
                "company": "",
                "ticker": r["ticker"],
                "page_number": r["page"],
                "pdf_url": r["source_pdf"],
                "company_source": "u10_pseudo",
                "confidence_min": float(min(per_task_conf[t][i] for t in TASKS)),
                "conf_T1": float(per_task_conf["promise_status"][i]),
                "conf_T2": float(per_task_conf["verification_timeline"][i]),
                "conf_T3": float(per_task_conf["evidence_status"][i]),
                "conf_T4": float(per_task_conf["evidence_quality"][i]),
            }
            w.writerow(row)
            n_written += 1
            for t in TASKS:
                lab = row[t]
                label_dist[t][lab] = label_dist[t].get(lab, 0) + 1

    stats = {
        "n_corpus": len(records),
        "n_models_ensembled": n_models,
        "thresholds": THRESHOLDS,
        "per_task_pass": per_task_pass,
        "n_admitted": admitted,
        "n_written": n_written,
        "label_distribution_admitted": label_dist,
    }
    RPT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(RPT_JSON, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"\nLabel distribution (admitted):")
    for t in TASKS:
        print(f"  {t}: {label_dist[t]}")
    print(f"\nWrote {n_written} pseudo rows -> {OUT_CSV}")
    print(f"Stats -> {RPT_JSON}")


if __name__ == "__main__":
    main()
