"""M4-v2 — Per-class targeted pseudo-labeling.

Reads corpus_v2.jsonl (12,345 paras) and runs the 15-ckpt ensemble. Two-tier
admission to specifically pull in minority classes:

  Tier-1 (standard, applies to ALL rows):
      T1 >= 0.80, T2 >= 0.60, T3 >= 0.70, T4 >= 0.60     (same as v1)

  Tier-2 (minority boost, applies if Tier-1 fails):
      Row admitted IF (a) it's a heuristic candidate for a minority class
      AND (b) the model's prediction matches the heuristic class
      AND (c) per-task max-probs all exceed RELAXED thresholds:
          T1 >= 0.55, T2 >= 0.45, T3 >= 0.50, T4 >= 0.45

      Minority targets:
        T2: within_2_years, between_2_and_5_years, longer_than_5_years
        T4: Not_Clear, Misleading

The Tier-2 row is also capped per minority class (max_per_minority).

Output CSV: data/processed/u10/pseudo_labels_v2.csv  (same schema as v1)
Stats   : reports/experiments/u10/pseudo_label_v2_stats.json
"""
from __future__ import annotations
import sys, io, json, time
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
import numpy as np, torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.data.dataset import ESGDataset, ID2LABEL, NUM_LABELS, TASKS, esg_collate
from src.models.multitask import MultiTaskClassifier

# Same maps used by v1
THRESH_T1 = {"promise_status":0.80,"verification_timeline":0.60,"evidence_status":0.70,"evidence_quality":0.60}
THRESH_T2 = {"promise_status":0.55,"verification_timeline":0.45,"evidence_status":0.50,"evidence_quality":0.45}
MAX_PER_MINORITY = 60        # cap each minority class
MAX_TOTAL_TIER2 = 250        # global cap on Tier-2 admissions

BACKBONE = "hfl/chinese-macbert-base"
MAX_LEN = 384
BATCH = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = DEVICE.type == "cuda"

CKPT_ROOT = ROOT / "outputs" / "checkpoints" / "p2_combo_best"
SEEDS = [42, 2024, 20260417]; N_FOLDS = 5
CORPUS = ROOT / "data" / "processed" / "u10" / "corpus_v2.jsonl"
OUT_CSV = ROOT / "data" / "processed" / "u10" / "pseudo_labels_v2.csv"
RPT_JSON = ROOT / "reports" / "experiments" / "u10" / "pseudo_label_v2_stats.json"

# Map heuristic tag -> dataset label string used by ID2LABEL
T2_HEUR_TO_LABEL = {
    "already": "already",
    "within_2_years": "within_2_years",
    "between_2_and_5_years": "between_2_and_5_years",
    "longer_than_5_years": "longer_than_5_years",
}
T4_HEUR_TO_LABEL = {"Clear":"Clear","Not_Clear":"Not Clear"}

# Reverse maps to id
def _label_to_id(task, label):
    for i, l in ID2LABEL[task].items():
        if l == label:
            return i
    return None


def load_corpus():
    recs = []
    with open(CORPUS, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            r["id"] = 200000 + len(recs)
            recs.append(r)
    return recs


def predict_probs(model, loader):
    model.eval()
    bufs = {t: [] for t in TASKS}; idxs = []
    autocast = torch.amp.autocast(device_type=DEVICE.type, enabled=USE_AMP)
    with torch.no_grad(), autocast:
        for batch in loader:
            ids = batch["input_ids"].to(DEVICE, non_blocking=True)
            mk = batch["attention_mask"].to(DEVICE, non_blocking=True)
            logits = model(ids, mk)
            for t in TASKS:
                bufs[t].append(torch.softmax(logits[t].float(), -1).cpu().numpy())
            idxs.extend(batch["_index"].cpu().tolist())
    out = {t: np.concatenate(bufs[t], 0) for t in TASKS}
    order = np.argsort(idxs)
    return {t: out[t][order] for t in TASKS}


def main():
    print(f"Device: {DEVICE}, AMP: {USE_AMP}")
    records = load_corpus()
    print(f"Corpus rows: {len(records)}")
    tokenizer = AutoTokenizer.from_pretrained(BACKBONE)
    ds = ESGDataset(records, tokenizer, max_length=MAX_LEN, text_field="text", with_labels=False)
    loader = DataLoader(ds, batch_size=BATCH, shuffle=False, collate_fn=esg_collate, num_workers=0, pin_memory=True)

    sum_probs = {t: None for t in TASKS}; n_models = 0
    for seed in SEEDS:
        for fi in range(N_FOLDS):
            ck = CKPT_ROOT / f"seed{seed}" / f"fold{fi}" / "best.pt"
            if not ck.exists(): print(f"  MISS {ck}"); continue
            t0 = time.time()
            model = MultiTaskClassifier(backbone=BACKBONE, num_labels=NUM_LABELS, pooling="cls_mean", dropout=0.1, msd_k=1)
            payload = torch.load(ck, map_location="cpu", weights_only=False)
            state = payload.get("model_state_dict", payload)
            model.load_state_dict(state, strict=True); model.to(DEVICE)
            probs = predict_probs(model, loader)
            for t in TASKS:
                sum_probs[t] = probs[t] if sum_probs[t] is None else sum_probs[t] + probs[t]
            n_models += 1
            print(f"  seed={seed} fold={fi}  {time.time()-t0:.1f}s  ({n_models})", flush=True)
            del model; torch.cuda.empty_cache()

    avg = {t: sum_probs[t] / n_models for t in TASKS}
    pred = {t: avg[t].argmax(-1) for t in TASKS}
    conf = {t: avg[t].max(-1) for t in TASKS}

    # ---- Tier-1: standard admission ----
    tier1 = np.ones(len(records), dtype=bool)
    for t in TASKS:
        tier1 &= conf[t] >= THRESH_T1[t]
    print(f"Tier-1 admitted: {int(tier1.sum())}/{len(records)}")

    # ---- Tier-2: minority boost ----
    tier2_counts = {"T2_within_2_years":0,"T2_between_2_and_5_years":0,"T2_longer_than_5_years":0,
                    "T4_Not_Clear":0,"T4_Misleading":0}
    tier2_mask = np.zeros(len(records), dtype=bool)
    tier2_total = 0
    # Iterate sorted by min(conf) descending so highest-confidence candidates picked first
    relax_pass = np.ones(len(records), dtype=bool)
    for t in TASKS:
        relax_pass &= conf[t] >= THRESH_T2[t]
    candidate_idx = []
    for i, r in enumerate(records):
        if tier1[i] or not relax_pass[i]:
            continue
        # T2 minority match
        c2 = r.get("cand_t2")
        if c2 in ("within_2_years","between_2_and_5_years","longer_than_5_years"):
            target_id = _label_to_id("verification_timeline", T2_HEUR_TO_LABEL[c2])
            if target_id is not None and pred["verification_timeline"][i] == target_id:
                key = f"T2_{c2}"
                if tier2_counts[key] < MAX_PER_MINORITY:
                    candidate_idx.append((i, key, min(conf[t][i] for t in TASKS)))
                    continue
        # T4 minority match
        c4 = r.get("cand_t4")
        if c4 == "Not_Clear":
            target_id = _label_to_id("evidence_quality", "Not Clear")
            if target_id is not None and pred["evidence_quality"][i] == target_id:
                if tier2_counts["T4_Not_Clear"] < MAX_PER_MINORITY:
                    candidate_idx.append((i, "T4_Not_Clear", min(conf[t][i] for t in TASKS)))
                    continue
        # Direct model-predicted Misleading (no heuristic)
        ml_id = _label_to_id("evidence_quality", "Misleading")
        if ml_id is not None and pred["evidence_quality"][i] == ml_id:
            if tier2_counts["T4_Misleading"] < MAX_PER_MINORITY:
                candidate_idx.append((i, "T4_Misleading", min(conf[t][i] for t in TASKS)))

    # Sort by confidence desc and apply caps
    candidate_idx.sort(key=lambda x: -x[2])
    for i, key, c in candidate_idx:
        if tier2_total >= MAX_TOTAL_TIER2:
            break
        if tier2_counts[key] >= MAX_PER_MINORITY:
            continue
        tier2_mask[i] = True
        tier2_counts[key] += 1
        tier2_total += 1
    print(f"Tier-2 admitted: {int(tier2_mask.sum())}  by class: {tier2_counts}")

    final_mask = tier1 | tier2_mask
    print(f"Final admitted: {int(final_mask.sum())}")

    # ---- Write CSV ----
    import csv
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fields = ["id","data","esg_type","promise_status","promise_string",
              "verification_timeline","evidence_status","evidence_string",
              "evidence_quality","company","ticker","page_number","pdf_url",
              "company_source","confidence_min","conf_T1","conf_T2","conf_T3","conf_T4","tier"]
    n_written = 0
    label_dist = {t: {} for t in TASKS}
    with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for i, r in enumerate(records):
            if not final_mask[i]: continue
            row = {
                "id": r["id"],
                "data": r["text"],
                "esg_type": "",
                "promise_status": ID2LABEL["promise_status"][int(pred["promise_status"][i])],
                "promise_string": "",
                "verification_timeline": ID2LABEL["verification_timeline"][int(pred["verification_timeline"][i])],
                "evidence_status": ID2LABEL["evidence_status"][int(pred["evidence_status"][i])],
                "evidence_string": "",
                "evidence_quality": ID2LABEL["evidence_quality"][int(pred["evidence_quality"][i])],
                "company": "",
                "ticker": r["ticker"],
                "page_number": r["page"],
                "pdf_url": r["source_pdf"],
                "company_source": "u10_v2",
                "confidence_min": float(min(conf[t][i] for t in TASKS)),
                "conf_T1": float(conf["promise_status"][i]),
                "conf_T2": float(conf["verification_timeline"][i]),
                "conf_T3": float(conf["evidence_status"][i]),
                "conf_T4": float(conf["evidence_quality"][i]),
                "tier": "1" if tier1[i] else "2",
            }
            w.writerow(row); n_written += 1
            for t in TASKS:
                label_dist[t][row[t]] = label_dist[t].get(row[t], 0) + 1

    stats = {
        "n_corpus": len(records),
        "n_models_ensembled": n_models,
        "tier1_thresholds": THRESH_T1,
        "tier2_thresholds": THRESH_T2,
        "max_per_minority": MAX_PER_MINORITY,
        "max_total_tier2": MAX_TOTAL_TIER2,
        "n_tier1_admitted": int(tier1.sum()),
        "n_tier2_admitted": int(tier2_mask.sum()),
        "tier2_per_class": tier2_counts,
        "n_written": n_written,
        "label_distribution_admitted": label_dist,
    }
    RPT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(RPT_JSON, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print("\nLabel distribution (admitted):")
    for t in TASKS:
        print(f"  {t}: {label_dist[t]}")
    print(f"\nWrote {n_written} rows -> {OUT_CSV}")
    print(f"Stats -> {RPT_JSON}")


if __name__ == "__main__":
    main()
