"""U7 / Joint hillclimb v11 — directly optimize the FINAL weighted score.

Why this exists (Phase 12 lesson):
  per_task_hillclimb_v{...} optimize each task's F1 INDEPENDENTLY while holding
  the other 3 tasks at the uniform baseline. The 4 best per-task tuples are
  then combined and `apply_constraints_batch` runs ONCE post-hoc. Because the
  hierarchical constraints couple tasks (e.g. promise=No forces timeline=N/A),
  marginal per-task gains can violate constraints on edge samples and lower
  the joint score. v10 demonstrated this: per-task internal F1 all >= v9 yet
  joint score regressed -0.00023.

Design:
  * Single weight matrix W of shape (4 tasks) x (M members).
  * Score function = weighted_score(apply_constraints_batch(arg-max(combine(W[t]))))
    -- the EXACT competition objective. No surrogate.
  * Hill climbing: warm-start from v9 winners (and 0.0 for new members). Each
    iter, perturb ONE (task, exp) cell by sampling from GRID. Accept iff joint
    score strictly improves.
  * N_ITERS large but each eval is O(N=1000) cheap — ~5-10 min CPU total.

Pool auto-detection:
  * Always includes the 14 v10 members.
  * Auto-includes p8_ema995 and p9_ls_t1t3 if their checkpoint folders exist
    (so this script works both before and after their training completes).
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.dataset import LABEL_DOMAINS, NUM_LABELS, TASKS
from src.data.loader import load_dataset
from src.eval.metrics import weighted_score
from src.inference.post_process import apply_constraints_batch
from src.tools.per_task_hillclimb_v7 import _load_member as _load_member_v7
from src.tools.per_task_hillclimb_v10 import LOADERS as _V10_LOADERS


# Base v10 members in fixed order
_V10_EXPS = [
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
    "p5_t6_time_token",       # 11
    "p6_t6v2_bucket_tok",     # 12
    "p7_focal_g3",            # 13
]

# v9 winners with 0 appended at idx 13 — also our v11 warm-start (extra zeros
# for any auto-added members).
_V9_WINNERS = {
    "promise_status":        (0.0, 0.0, 0.0, 1.5, 1.75, 0.75, 1.25, 1.25, 0.0, 0.0, 0.0, 2.0, 1.5, 0.0),
    "verification_timeline": (0.0, 0.75, 1.25, 0.25, 0.0, 0.0, 0.0, 1.75, 0.0, 0.5, 1.25, 0.0, 0.0, 0.0),
    "evidence_status":       (1.25, 0.75, 0.75, 0.0, 0.0, 1.5, 0.25, 0.25, 0.5, 0.0, 0.0, 2.0, 0.0, 0.0),
    "evidence_quality":      (1.0, 1.5, 1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
}


GRID = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
RNG = random.Random(20260506)
N_ITERS = 30000


def _build_pool() -> tuple[list[str], dict, dict[str, list[float]]]:
    """Return (EXPS, LOADERS, warm_start_winners) auto-extending v10 with
    any of {p8_ema995, p9_ls_t1t3} that finished training."""
    exps = list(_V10_EXPS)
    loaders = dict(_V10_LOADERS)
    extras = [
        ("p8_ema995", 42),
        ("p9_ls_t1t3", 42),
    ]
    for name, seed in extras:
        ckpt_dir = Path("outputs/checkpoints") / name / f"seed{seed}"
        # Require ALL 5 folds (fold0..fold4) to have best.pt before pooling,
        # otherwise OOF coverage is incomplete and v11 will see NaN/zero rows.
        complete = ckpt_dir.exists() and all(
            (ckpt_dir / f"fold{i}" / "best.pt").exists() for i in range(5)
        )
        if complete:
            print(f"[pool] auto-include {name} (5/5 folds ready)")
            exps.append(name)
            loaders[name] = ("single", name, [seed])
        else:
            n_done = sum((ckpt_dir / f"fold{i}" / "best.pt").exists() for i in range(5))
            print(f"[pool] skip {name} ({n_done}/5 folds ready)")
    # warm start: v9 winners + 0.0 padding for each extra
    n_extras = len(exps) - len(_V10_EXPS)
    winners = {t: list(_V9_WINNERS[t]) + [0.0] * n_extras for t in TASKS}
    return exps, loaders, winners


def _load_member(name: str, n: int, loaders: dict):
    import src.tools.per_task_hillclimb_v7 as _v7
    saved = _v7.LOADERS
    _v7.LOADERS = loaders
    try:
        return _load_member_v7(name, n)
    finally:
        _v7.LOADERS = saved


def _build_accum(per_exp_probs, exps, w_per_task, n):
    """Combine per-task weighted ensemble. Returns dict[task]->[N, C]."""
    out = {}
    for t in TASKS:
        w_tuple = w_per_task[t]
        arr = np.zeros((n, NUM_LABELS[t]))
        s = sum(w_tuple)
        if s == 0:
            # fallback: equal weights to avoid 0/0
            for exp in exps:
                arr += per_exp_probs[exp][t]
            arr /= len(exps)
        else:
            for exp, ww in zip(exps, w_tuple):
                if ww == 0:
                    continue
                arr += ww * per_exp_probs[exp][t]
            arr /= s
        out[t] = arr
    return out


def _score_joint(truth, accum, records, n) -> tuple[float, dict[str, float]]:
    """Run the COMPLETE pipeline: arg-max -> constraints -> weighted_score."""
    preds = []
    for i in range(n):
        row = {"id": records[i].get("id", i)}
        for t in TASKS:
            row[t] = LABEL_DOMAINS[t][int(accum[t][i].argmax())]
        preds.append(row)
    proc = apply_constraints_batch(preds)
    pd_dict = {t: [r[t] for r in proc] for t in TASKS}
    ws = weighted_score(truth, pd_dict)
    per_task = {t: ws[t] for t in TASKS}
    return ws["final_weighted_score"], per_task


def main() -> None:
    exps, loaders, w_init = _build_pool()
    M = len(exps)
    print(f"[pool] M={M} members")
    for i, e in enumerate(exps):
        print(f"  [{i:2d}] {e}")

    records, _ = load_dataset("data/raw/vpesg4k_train_1000 V1.csv")
    n = len(records)
    truth = {t: [r[t] for r in records] for t in TASKS}

    per_exp_probs = {name: _load_member(name, n, loaders) for name in exps}

    # --- evaluate warm-start ---
    accum = _build_accum(per_exp_probs, exps, w_init, n)
    best_score, best_per_task = _score_joint(truth, accum, records, n)
    best_w = {t: list(w_init[t]) for t in TASKS}
    print(f"\n[warm-start] joint={best_score:.5f} | "
          f"T1={best_per_task['promise_status']:.4f} "
          f"T2={best_per_task['verification_timeline']:.4f} "
          f"T3={best_per_task['evidence_status']:.4f} "
          f"T4={best_per_task['evidence_quality']:.4f}")

    # --- coordinate hill-climb on joint score ---
    accept = 0
    history = []
    for it in range(1, N_ITERS + 1):
        # Pick a random (task, exp) cell to perturb
        t = RNG.choice(TASKS)
        idx = RNG.randrange(M)
        new_val = RNG.choice(GRID)
        old_val = best_w[t][idx]
        if new_val == old_val:
            continue
        cand_w = {tt: list(best_w[tt]) for tt in TASKS}
        cand_w[t][idx] = new_val
        # only need to recompute task t's accum
        cand_accum = dict(accum)
        w_tuple = cand_w[t]
        s = sum(w_tuple)
        if s == 0:
            continue
        arr = np.zeros((n, NUM_LABELS[t]))
        for exp, ww in zip(exps, w_tuple):
            if ww == 0:
                continue
            arr += ww * per_exp_probs[exp][t]
        arr /= s
        cand_accum[t] = arr
        score, per_task = _score_joint(truth, cand_accum, records, n)
        if score > best_score:
            best_score = score
            best_per_task = per_task
            best_w = cand_w
            accum = cand_accum
            accept += 1
            history.append({"iter": it, "task": t, "idx": idx, "exp": exps[idx],
                            "new_val": new_val, "old_val": old_val,
                            "score": score, **{f"T{i+1}": per_task[tt]
                                                for i, tt in enumerate(TASKS)}})
            print(f"[iter {it:5d}] accept #{accept:3d}  {t:25s}[{idx:2d}={exps[idx]:25s}] "
                  f"{old_val:.2f}->{new_val:.2f}  joint={score:.5f}")

    print(f"\n[joint hillclimb v11] DONE  iters={N_ITERS}  accepted={accept}")
    print(f"[FINAL v11] joint={best_score:.5f} | "
          f"T1={best_per_task['promise_status']:.4f} "
          f"T2={best_per_task['verification_timeline']:.4f} "
          f"T3={best_per_task['evidence_status']:.4f} "
          f"T4={best_per_task['evidence_quality']:.4f}")
    print(f"[compare] v9 SOTA=0.68683  v10=0.68660  v11={best_score:.5f}  "
          f"delta_vs_v9={best_score - 0.68683:+.5f}")

    out_dir = Path("reports/analysis/_ensemble")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Final preds with the joint-optimal weights
    final_preds = []
    for i in range(n):
        row = {"id": records[i].get("id", i)}
        for t in TASKS:
            row[t] = LABEL_DOMAINS[t][int(accum[t][i].argmax())]
        final_preds.append(row)
    proc = apply_constraints_batch(final_preds)
    pd.DataFrame(proc).to_csv(out_dir / "joint_hillclimb_v11_preds.csv", index=False)

    # Summary CSV: weights per task
    rows = []
    for t in TASKS:
        row = {"task": t, "f1": best_per_task[t]}
        for exp, ww in zip(exps, best_w[t]):
            row[f"w_{exp}"] = ww
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_dir / "joint_hillclimb_v11_summary.csv", index=False)

    # History JSONL
    with (out_dir / "joint_hillclimb_v11_history.jsonl").open("w", encoding="utf-8") as f:
        for h in history:
            f.write(json.dumps(h) + "\n")

    # Top-line metrics JSON
    with (out_dir / "joint_hillclimb_v11_meta.json").open("w", encoding="utf-8") as f:
        json.dump({
            "members": exps,
            "M": M,
            "n_iters": N_ITERS,
            "accepted": accept,
            "warm_start_score": _score_joint(truth, _build_accum(per_exp_probs, exps, w_init, n), records, n)[0],
            "final_score": best_score,
            "per_task": best_per_task,
            "best_w": {t: best_w[t] for t in TASKS},
            "v9_sota": 0.68683,
            "delta_vs_v9": best_score - 0.68683,
        }, f, indent=2, ensure_ascii=False)

    print(f"[wrote] {out_dir}/joint_hillclimb_v11_summary.csv")
    print(f"[wrote] {out_dir}/joint_hillclimb_v11_preds.csv")
    print(f"[wrote] {out_dir}/joint_hillclimb_v11_history.jsonl")
    print(f"[wrote] {out_dir}/joint_hillclimb_v11_meta.json")


if __name__ == "__main__":
    main()
