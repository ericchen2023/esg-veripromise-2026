"""U42 — Phase 57: sentence-embedding macro model (the Chinese-Timing winner's class).

SemEval-2025 PromiseEval finding: CSECU-DSG won the CHINESE Timing subtask with
universal sentence embeddings (a model class distinct from the BERT pool), and
the organizers noted "universal sentence embeddings were particularly effective
for timing verification." T2 (verification_timeline, w=0.15) has headroom and is
a faithfully-transferring macro task, so a strong multilingual sentence-embedding
classifier is a genuinely-new, research-backed source for the per-task recombine.

This embeds train+val(2000) + test(2000) with multilingual-e5-large, trains a
per-task logistic-regression head on the SAME 5-fold StratifiedKFold splits as
the BERT stems (so OOF aligns), and writes OOF + test probs per task. It then
val-slice-validates T2 (and T4) under the BERT cascade. Only sources that improve
the faithful val-slice get used in the recombination (scripts/u41).

Usage:
    python -m scripts.u42_embed_macro --model intfloat/multilingual-e5-large
"""
from __future__ import annotations

import argparse
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from src.config import load_config
from src.data.loader import load_dataset
from src.data.splits import make_folds
from src.data.dataset import LABEL_DOMAINS, LABEL2ID
from scripts.u17_phase42_test_inference import probs_to_records, load_test_records
from scripts.u16_tv_oof_ensemble import _reconstruct_oof, TV_STEMS

N = 2000
TASKS = ("promise_status", "verification_timeline", "evidence_status", "evidence_quality")
COMBINED = "data/processed/train_val_combined.csv"
OUT_OOF = "outputs/submissions/u42_e5_oof.npz"
OUT_TEST = "outputs/submissions/u42_e5_test_probs.npz"
VA = np.arange(1000, 2000)


def embed(model_name, texts):
    from sentence_transformers import SentenceTransformer
    import torch
    m = SentenceTransformer(model_name, device="cuda" if torch.cuda.is_available() else "cpu")
    # e5 convention: prefix inputs with "query: "
    return m.encode(["query: " + str(t) for t in texts], batch_size=64,
                    normalize_embeddings=True, show_progress_bar=True).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="intfloat/multilingual-e5-large")
    args = ap.parse_args()

    cfg = load_config("configs/exp_p2_combo_best.yaml")
    records, df = load_dataset(COMBINED)
    folds = make_folds(df, n_splits=5, stratify_fields=cfg["split"]["stratify_fields"],
                       seed=42, mode="stratified_kfold")
    test_records = load_test_records("vpesg4k_test_2000.csv")
    test_df = pd.DataFrame(test_records)

    print(f"[embed] train+val {len(df)} + test {len(test_df)} with {args.model}")
    Xtr = embed(args.model, df["data"].tolist())
    Xte = embed(args.model, test_df["data"].tolist())

    oof = {t: np.zeros((N, len(LABEL_DOMAINS[t]))) for t in TASKS}
    test = {t: np.zeros((N, len(LABEL_DOMAINS[t]))) for t in TASKS}
    for t in TASKS:
        y = df[t].astype(str).map(LABEL2ID[t]).to_numpy()
        classes = list(range(len(LABEL_DOMAINS[t])))
        for fi, (tr, va) in enumerate(folds):
            clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
            clf.fit(Xtr[tr], y[tr])
            # map clf.classes_ back to full label space
            p_va = np.zeros((len(va), len(classes)))
            p_te = np.zeros((len(test_df), len(classes)))
            for j, c in enumerate(clf.classes_):
                p_va[:, c] = clf.predict_proba(Xtr[va])[:, j]
                p_te[:, c] = clf.predict_proba(Xte)[:, j]
            oof[t][va] = p_va
            test[t] += p_te / len(folds)
    np.savez_compressed(OUT_OOF, **oof)
    np.savez_compressed(OUT_TEST, **test)
    print(f"[saved] {OUT_OOF}, {OUT_TEST}")

    # ---- val-slice validation: BERT cascade, swap one macro task to e5 ----
    bert = {t: np.zeros((N, len(LABEL_DOMAINS[t]))) for t in TASKS}
    for s in TV_STEMS:
        o = _reconstruct_oof(s, N)
        for t in TASKS:
            bert[t] += o[t]
    for t in TASKS:
        bert[t] /= len(TV_STEMS)
    g = pd.read_csv(COMBINED, keep_default_na=False)

    def score(probs, col):
        pred = np.array([r[col] for r in probs_to_records(probs, [{"id": i} for i in range(N)])])
        gold = g[col].astype(str).to_numpy()
        return f1_score(gold[VA], pred[VA], labels=LABEL_DOMAINS[col], average="macro", zero_division=0)

    print("\n[val-slice macro F1: BERT vs e5 vs blend]  (faithful LB proxy)")
    for col in ("verification_timeline", "evidence_quality"):
        b = score(bert, col)
        sw = {k: bert[k].copy() for k in TASKS}; sw[col] = oof[col]
        e = score(sw, col)
        for w in (0.3, 0.5):
            bl = {k: bert[k].copy() for k in TASKS}; bl[col] = (1 - w) * bert[col] + w * oof[col]
            bw = score(bl, col)
            print(f"  {col:24s} BERT={b:.4f}  e5={e:.4f}  blend{w}={bw:.4f}")


if __name__ == "__main__":
    main()
