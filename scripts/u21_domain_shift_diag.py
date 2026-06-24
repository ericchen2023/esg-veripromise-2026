"""U21 — Phase 45 domain-shift & binary-confidence diagnostic.

Phase 44 localised the leaderboard collapse to the BINARY tasks (T1 promise
0.941->0.786, T3 evidence_status 0.861->0.675) while the MACRO tasks (T2, T4)
transferred almost perfectly. The corrective lever (prior-correction toward the
minority class "No") CANNOT be tuned on the OOF set, because the OOF shares the
exact train prior — any toward-No shift looks strictly worse on OOF by
construction. So before spending scarce leaderboard quota (3 uploads/day) we
gather TWO local signals that DO carry information about the test set:

  1. Adversarial validation (train vs test text separability).
     A char+word TF-IDF logistic-regression classifier is trained to tell train
     rows from test rows. CV ROC-AUC ~0.5 => same distribution (domain shift
     unlikely; binary collapse is genuine model error, correction won't help
     much). AUC >> 0.5 => real domain shift (the test paragraphs differ), which
     makes a prior/threshold correction the right tool.

  2. Binary-confidence comparison (OOF vs test) for T1 and T3.
     We reconstruct the equal-weight OOF blend and the equal-weight TEST blend
     (cached probs) and compare the p(Yes) distributions. If on TEST the model
     is LESS confident in Yes (lower mean p(Yes), more borderline 0.4-0.6 mass,
     more mass below 0.5) than on OOF, the model is over-committing to Yes on a
     set that contains more genuine "No" — i.e. toward-No correction is
     justified and we can even estimate how hard to push.

Neither signal touches the label of a single test row, so this is purely a
read-only diagnostic. Output is printed and (optionally) the per-row binary
p(Yes) arrays are returned for downstream candidate generation.

Usage:
    python -m scripts.u21_domain_shift_diag
"""
from __future__ import annotations

import collections

import numpy as np
import pandas as pd
import torch

from scripts.u16_tv_oof_ensemble import COMBINED_CSV, TV_STEMS, _reconstruct_oof
from scripts.u17_phase42_test_inference import load_test_records, mix
from scripts.u18_decoding_experiments import load_cached_probs
from src.data.dataset import LABEL2ID, TASKS
from src.data.loader import load_dataset

BINARY = ("promise_status", "evidence_status")
TEST_CSV = "vpesg4k_test_2000.csv"


def _equal_weights(n_stems: int) -> dict[str, list[float]]:
    return {t: [1.0] * n_stems for t in TASKS}


def adversarial_validation(train_texts: list[str], test_texts: list[str]) -> float:
    """Return CV ROC-AUC of a TF-IDF logistic classifier separating train vs test.

    ~0.5 => indistinguishable (no domain shift). Closer to 1.0 => strong shift.
    """
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score
        from sklearn.pipeline import Pipeline
    except ImportError:
        print("[adv] scikit-learn unavailable; skipping adversarial validation")
        return float("nan")

    texts = list(train_texts) + list(test_texts)
    y = np.array([0] * len(train_texts) + [1] * len(test_texts))
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4),
                                  min_df=3, max_features=40000)),
        ("clf", LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")),
    ])
    scores = cross_val_score(pipe, texts, y, cv=5, scoring="roc_auc", n_jobs=-1)
    return float(scores.mean())


def _pyes(probs_task: np.ndarray, task: str) -> np.ndarray:
    """Column of P(Yes) for a binary task."""
    return probs_task[:, LABEL2ID[task]["Yes"]]


def confidence_report(oof_pyes: np.ndarray, test_pyes: np.ndarray, task: str) -> None:
    """Print an OOF-vs-test comparison of the P(Yes) distribution for one task."""
    def stats(p: np.ndarray) -> dict[str, float]:
        return {
            "mean_pYes": float(p.mean()),
            "median_pYes": float(np.median(p)),
            "pred_Yes_rate": float((p > 0.5).mean()),
            "borderline_0.4_0.6": float(((p > 0.4) & (p < 0.6)).mean()),
            "below_0.5": float((p < 0.5).mean()),
            "high_conf_Yes_>0.8": float((p > 0.8).mean()),
        }

    o, t = stats(oof_pyes), stats(test_pyes)
    print(f"\n=== {task} — P(Yes) distribution (equal-weight blend) ===")
    print(f"  {'metric':22s} {'OOF':>9s} {'TEST':>9s} {'delta':>9s}")
    for k in o:
        d = t[k] - o[k]
        print(f"  {k:22s} {o[k]:9.3f} {t[k]:9.3f} {d:+9.3f}")
    verdict = ("toward-No SUPPORTED (test less confident in Yes)"
               if t["mean_pYes"] < o["mean_pYes"] - 1e-3
               else "toward-No NOT clearly supported by confidence")
    print(f"  -> {verdict}")


def main() -> None:
    device = torch.device("cpu")

    # --- load texts ---
    train_recs, _ = load_dataset(COMBINED_CSV)
    train_texts = [str(r.get("data", "")) for r in train_recs]
    test_records = load_test_records(TEST_CSV)
    test_texts = [r["data"] for r in test_records]
    n_test = len(test_records)
    print(f"[data] train={len(train_texts)} rows, test={n_test} rows")

    # --- (1) adversarial validation ---
    auc = adversarial_validation(train_texts, test_texts)
    print(f"\n[adv] train-vs-test TF-IDF CV ROC-AUC = {auc:.4f}")
    if not np.isnan(auc):
        if auc < 0.6:
            print("      -> distributions ~indistinguishable; domain shift WEAK")
        elif auc < 0.75:
            print("      -> moderate domain shift")
        else:
            print("      -> STRONG domain shift (test text differs markedly)")

    # --- (2) binary-confidence OOF vs test ---
    n_stems = len(TV_STEMS)
    eq = _equal_weights(n_stems)

    oof_per_stem = {stem: _reconstruct_oof(stem, len(train_recs)) for stem in TV_STEMS}
    oof_mixed = mix(oof_per_stem, eq)

    test_per_stem = load_cached_probs(test_records, device, use_cache=True)
    test_mixed = mix(test_per_stem, eq)

    for task in BINARY:
        confidence_report(_pyes(oof_mixed[task], task),
                          _pyes(test_mixed[task], task), task)

    # train marginal for reference
    print("\n[ref] train Yes-rates:")
    for task in BINARY:
        c = collections.Counter(str(r.get(task)) for r in train_recs)
        tot = sum(c.values())
        print(f"  {task:18s} Yes={c.get('Yes', 0) / tot:6.1%}")

    print("\n[done] diagnostic complete")


if __name__ == "__main__":
    main()
