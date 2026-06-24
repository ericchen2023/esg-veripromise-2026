"""U7 / Joint hillclimb v12 — extends v11 (16-way) with the 17th member
`p10_large_focal_fgm` (Phase 14 / N1, OOF=0.66874).

Design changes vs v11:
  * Pool size M=17 (v11's 16 + p10_large_focal_fgm).
  * Warm-start = v11 FINAL SOTA winners (0.68770) padded with 0.0 for idx 16.
  * Same coordinate hill-climb, same GRID, same N_ITERS, fresh RNG seed for
    independent exploration.

Member order (idx 16 is the new entry):
  0  p2_combo_v2              8  p2_combo_v3_avg
  1  p2_combo_best            9  p4a_roberta_wwm_base
  2  p3_large_lr2e5          10  p4b_bert_base_chinese
  3  p2ab_aug_mask10         11  p5_t6_time_token
  4  p2ac_aug_mix            12  p6_t6v2_bucket_tok
  5  p2ad_rdrop05            13  p7_focal_g3
  6  p2ae_msd5               14  p8_ema995
  7  p2_combo_v3_s42         15  p9_ls_t1t3
                             16  p10_large_focal_fgm  <-- NEW
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
from src.tools.joint_hillclimb_v11 import _build_pool as _build_pool_v11


# v11 FINAL SOTA winners (joint=0.68770) -- 16 members each.
# These are the warm-start; padded with 0.0 for the new member at idx 16.
_V11_WINNERS = {
    "promise_status":        [0.0, 0.0, 0.0, 1.5, 1.75, 0.75, 1.25, 1.25, 0.0, 0.0, 0.0, 2.0, 1.5, 0.0, 0.0, 0.0],
    "verification_timeline": [0.0, 0.75, 1.25, 0.25, 0.0, 0.0, 0.0, 1.75, 0.0, 0.5, 1.25, 0.0, 0.0, 0.0, 0.0, 0.0],
    "evidence_status":       [1.25, 0.75, 2.0, 0.25, 0.0, 1.0, 0.25, 0.75, 0.5, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0],
    "evidence_quality":      [1.0, 1.5, 1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
}

# New member to append (idx 16)
_NEW_MEMBER = ("p10_large_focal_fgm", 42)

GRID = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
RNG = random.Random(20260603)  # different seed than v11 for fresh exploration
N_ITERS = 30000

V11_SOTA = 0.68770


def _build_pool_v12() -> tuple[list[str], dict, dict[str, list[float]]]:
    """v11 pool + p10_large_focal_fgm if its 5 folds are ready."""
    exps, loaders, _ = _build_pool_v11()  # 16 members + v9 winners (length 16)
    name, seed = _NEW_MEMBER
    ckpt_dir = Path("outputs/checkpoints") / name / f"seed{seed}"
    complete = ckpt_dir.exists() and all(
        (ckpt_dir / f"fold{i}" / "best.pt").exists() for i in range(5)
    )
    if complete:
        print(f"[pool] auto-include {name} (5/5 folds ready)")
        exps.append(name)
        loaders[name] = ("single", name, [seed])
    else:
        n_done = sum((ckpt_dir / f"fold{i}" / "best.pt").exists() for i in range(5))
        raise RuntimeError(f"[pool] {name} not ready ({n_done}/5 folds). Train first.")
    # Warm start uses v11 winners + 0.0 padding (already length 16, +1 -> 17)
    winners = {t: list(_V11_WINNERS[t]) + [0.0] for t in TASKS}
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
    out = {}
    for t in TASKS:
        w_tuple = w_per_task[t]
        arr = np.zeros((n, NUM_LABELS[t]))
        s = sum(w_tuple)
        if s == 0:
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
    exps, loaders, w_init = _build_pool_v12()
    M = len(exps)
    print(f"[pool] M={M} members")
    for i, e in enumerate(exps):
        marker = "  <-- NEW" if i == M - 1 else ""
        print(f"  [{i:2d}] {e}{marker}")

    records, _ = load_dataset("data/raw/vpesg4k_train_1000 V1.csv")
    n = len(records)
    truth = {t: [r[t] for r in records] for t in TASKS}

    per_exp_probs = {name: _load_member(name, n, loaders) for name in exps}

    accum = _build_accum(per_exp_probs, exps, w_init, n)
    warm_score, warm_per_task = _score_joint(truth, accum, records, n)
    best_score = warm_score
    best_per_task = warm_per_task
    best_w = {t: list(w_init[t]) for t in TASKS}
    print(f"\n[warm-start v11-pad-0] joint={warm_score:.5f} "
          f"(v11 SOTA={V11_SOTA:.5f}, delta={warm_score - V11_SOTA:+.5f})")
    print(f"  T1={warm_per_task['promise_status']:.4f} "
          f"T2={warm_per_task['verification_timeline']:.4f} "
          f"T3={warm_per_task['evidence_status']:.4f} "
          f"T4={warm_per_task['evidence_quality']:.4f}")

    accept = 0
    history = []
    for it in range(1, N_ITERS + 1):
        t = RNG.choice(TASKS)
        idx = RNG.randrange(M)
        new_val = RNG.choice(GRID)
        old_val = best_w[t][idx]
        if new_val == old_val:
            continue
        cand_w = {tt: list(best_w[tt]) for tt in TASKS}
        cand_w[t][idx] = new_val
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
            new_marker = " *NEW*" if idx == M - 1 else ""
            print(f"[iter {it:5d}] accept #{accept:3d}  {t:25s}[{idx:2d}={exps[idx]:25s}]{new_marker} "
                  f"{old_val:.2f}->{new_val:.2f}  joint={score:.5f}")

    print(f"\n[joint hillclimb v12] DONE  iters={N_ITERS}  accepted={accept}")
    print(f"[FINAL v12] joint={best_score:.5f} | "
          f"T1={best_per_task['promise_status']:.4f} "
          f"T2={best_per_task['verification_timeline']:.4f} "
          f"T3={best_per_task['evidence_status']:.4f} "
          f"T4={best_per_task['evidence_quality']:.4f}")
    print(f"[compare] v11 SOTA={V11_SOTA:.5f}  v12={best_score:.5f}  "
          f"delta_vs_v11={best_score - V11_SOTA:+.5f}")

    # Stats: how often the new member (idx 16) was used
    new_uses = sum(1 for h in history if h["idx"] == M - 1)
    new_nonzero = sum(1 for t in TASKS if best_w[t][M - 1] != 0)
    print(f"[new member stats] p10_large_focal_fgm: "
          f"accepted_perturbations={new_uses}/{accept}, "
          f"final_nonzero_tasks={new_nonzero}/4")
    print(f"[new member final_w] " + " ".join(
        f"{t[:3]}={best_w[t][M-1]:.2f}" for t in TASKS))

    out_dir = Path("reports/analysis/_ensemble")
    out_dir.mkdir(parents=True, exist_ok=True)

    final_preds = []
    for i in range(n):
        row = {"id": records[i].get("id", i)}
        for t in TASKS:
            row[t] = LABEL_DOMAINS[t][int(accum[t][i].argmax())]
        final_preds.append(row)
    proc = apply_constraints_batch(final_preds)
    pd.DataFrame(proc).to_csv(out_dir / "joint_hillclimb_v12_preds.csv", index=False)

    rows = []
    for t in TASKS:
        row = {"task": t, "f1": best_per_task[t]}
        for exp, ww in zip(exps, best_w[t]):
            row[f"w_{exp}"] = ww
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_dir / "joint_hillclimb_v12_summary.csv", index=False)

    with (out_dir / "joint_hillclimb_v12_history.jsonl").open("w", encoding="utf-8") as f:
        for h in history:
            f.write(json.dumps(h) + "\n")

    with (out_dir / "joint_hillclimb_v12_meta.json").open("w", encoding="utf-8") as f:
        json.dump({
            "members": exps,
            "M": M,
            "n_iters": N_ITERS,
            "accepted": accept,
            "warm_start_score": warm_score,
            "final_score": best_score,
            "per_task": best_per_task,
            "best_w": {t: best_w[t] for t in TASKS},
            "v11_sota": V11_SOTA,
            "delta_vs_v11": best_score - V11_SOTA,
            "new_member": _NEW_MEMBER[0],
            "new_member_accepted_perturbations": new_uses,
            "new_member_final_nonzero_tasks": new_nonzero,
        }, f, indent=2, ensure_ascii=False)

    print(f"[wrote] {out_dir}/joint_hillclimb_v12_summary.csv")
    print(f"[wrote] {out_dir}/joint_hillclimb_v12_preds.csv")
    print(f"[wrote] {out_dir}/joint_hillclimb_v12_history.jsonl")
    print(f"[wrote] {out_dir}/joint_hillclimb_v12_meta.json")


if __name__ == "__main__":
    main()
