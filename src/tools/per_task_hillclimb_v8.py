"""Per-task hillclimb v8: 12-way pool extending v7b with Sprint B p5 t6 time-token.

New member (vs v7b):
  11 p5_t6_time_token : seed=42, hfl/chinese-macbert-base + add_time_tokens
                       (single OOF=0.66231, T2=0.4874)

Warm-start = v7b winners with 0.0 appended at idx 11 so the search
degenerates back to v7b SOTA (0.68558) when random search fails.

Same biased-search recipe as v7b: every perturbation MUST include the
new member (idx 11). Other 0-2 indices sampled from {0..10}.
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.dataset import LABEL_DOMAINS, NUM_LABELS, TASKS
from src.data.loader import load_dataset
from src.eval.metrics import weighted_score
from src.inference.post_process import apply_constraints_batch
from src.tools.per_task_hillclimb_v7 import _combine as _combine_v7  # noqa: F401
from src.tools.per_task_hillclimb_v7 import _load_member as _load_member_v7
from src.tools.per_task_hillclimb_v7 import _score_one


EXPS = [
    "p2_combo_v2",            # 0
    "p2_combo_best",          # 1
    "p3_large_lr2e5",         # 2
    "p2ab_aug_mask10",        # 3
    "p2ac_aug_mix",           # 4
    "p2ad_rdrop05",           # 5
    "p2ae_msd5",              # 6
    "p2_combo_v3_s42",        # 7
    "p2_combo_v3_avg",        # 8
    "p4a_roberta_wwm_base",   # 9
    "p4b_bert_base_chinese",  # 10
    "p5_t6_time_token",       # 11   NEW (Sprint B)
]

# Loader spec for the new member; reuse v7's LOADERS for the rest.
from src.tools.per_task_hillclimb_v7 import LOADERS as _BASE_LOADERS

LOADERS = dict(_BASE_LOADERS)
LOADERS["p5_t6_time_token"] = ("single", "p5_t6_time_token", [42])

GRID = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
N_RANDOM = 12000
NEW_IDX = [11]                      # p5_t6_time_token
LEGACY_IDX = list(range(11))        # 0..10
RNG = random.Random(20260503)

# v7b winners (11-way) with 0.0 appended at idx 11.
SEED_W = {
    "promise_status":        (0.0, 0.0, 0.0, 0.25, 0.25, 0.75, 0.25, 1.75, 0.0, 0.0, 0.0, 0.0),
    "verification_timeline": (0.0, 0.75, 1.25, 0.25, 0.0, 0.0, 0.0, 1.75, 0.0, 0.5, 1.25, 0.0),
    "evidence_status":       (1.0, 0.75, 0.75, 0.0, 0.0, 1.0, 0.25, 0.25, 0.5, 0.0, 0.0, 0.0),
    "evidence_quality":      (1.0, 1.5, 1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
}


def _load_member(name: str, n: int):
    # patch LOADERS map used by v7's helper by temporarily monkeypatching
    import src.tools.per_task_hillclimb_v7 as _v7
    saved = _v7.LOADERS
    _v7.LOADERS = LOADERS
    try:
        return _load_member_v7(name, n)
    finally:
        _v7.LOADERS = saved


def _combine(per_exp_probs, w_tuple, task, n):
    arr = np.zeros((n, NUM_LABELS[task]))
    s = sum(w_tuple)
    if s == 0:
        return arr
    for exp, ww in zip(EXPS, w_tuple):
        if ww == 0:
            continue
        arr += ww * per_exp_probs[exp][task]
    return arr / s


def main() -> None:
    records, _ = load_dataset("data/raw/vpesg4k_train_1000 V1.csv")
    n = len(records)
    truth = {t: [r[t] for r in records] for t in TASKS}

    per_exp_probs = {name: _load_member(name, n) for name in EXPS}

    default = tuple([1.0] * len(EXPS))
    accum_default = {t: np.zeros((n, NUM_LABELS[t])) for t in TASKS}
    for exp, w in zip(EXPS, default):
        for t in TASKS:
            accum_default[t] += w * per_exp_probs[exp][t]
    for t in TASKS:
        accum_default[t] /= sum(default)

    best_per_task: dict[str, tuple] = {}
    best_acc_per_task: dict[str, np.ndarray] = {}

    for t in TASKS:
        seed_w = SEED_W[t]
        seed_arr = _combine(per_exp_probs, seed_w, t, n)
        seed_f1 = _score_one(truth, seed_arr, accum_default, t, records, n)
        best_w = seed_w
        best_score = seed_f1
        best_arr = seed_arr
        print(f"[task] {t:25s}  seed_w F1={seed_f1:.5f}")
        for _ in range(N_RANDOM):
            w = list(best_w)
            new_pick = NEW_IDX  # always force idx 11
            extra_k = RNG.randint(0, 2)
            extra_pick = RNG.sample(LEGACY_IDX, extra_k)
            for idx in new_pick + extra_pick:
                w[idx] = RNG.choice(GRID)
            w = tuple(w)
            if sum(w) == 0:
                continue
            arr = _combine(per_exp_probs, w, t, n)
            f1 = _score_one(truth, arr, accum_default, t, records, n)
            if f1 > best_score:
                best_score = f1
                best_w = w
                best_arr = arr
        best_per_task[t] = (best_w, best_score)
        best_acc_per_task[t] = best_arr
        print(f"[task] {t:25s}  best_w={best_w}  F1={best_score:.5f}")

    preds = []
    for i in range(n):
        row = {"id": records[i].get("id", i)}
        for t in TASKS:
            row[t] = LABEL_DOMAINS[t][int(best_acc_per_task[t][i].argmax())]
        preds.append(row)
    proc = apply_constraints_batch(preds)
    pd_dict = {t: [r[t] for r in proc] for t in TASKS}
    ws = weighted_score(truth, pd_dict)
    print()
    print(f"[FINAL v8] per-task weighted score = {ws['final_weighted_score']:.5f}")
    print(f"           T1={ws['promise_status']:.4f} T2={ws['verification_timeline']:.4f} "
          f"T3={ws['evidence_status']:.4f} T4={ws['evidence_quality']:.4f}")

    out_dir = Path("reports/analysis/_ensemble")
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(proc).to_csv(out_dir / "per_task_hillclimb_v8_preds.csv", index=False)
    rows = []
    for t in TASKS:
        w_tuple, f1 = best_per_task[t]
        row = {"task": t, "best_f1": f1}
        for exp, ww in zip(EXPS, w_tuple):
            row[f"w_{exp}"] = ww
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_dir / "per_task_hillclimb_v8_summary.csv", index=False)
    print(f"[wrote] {out_dir}/per_task_hillclimb_v8_summary.csv")
    print(f"[wrote] {out_dir}/per_task_hillclimb_v8_preds.csv")


if __name__ == "__main__":
    main()
