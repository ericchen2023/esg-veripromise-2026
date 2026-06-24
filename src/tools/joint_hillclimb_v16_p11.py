"""Phase 16 / N2 admission test for p11_electra_base.

This script keeps the Phase 14 v12 ensemble intact, warm-starts from the saved
v12 final weights, appends p11 as the 18th member, and optimizes the exact
post-constraint weighted objective.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.dataset import NUM_LABELS, TASKS, LABEL_DOMAINS
from src.data.loader import load_dataset
from src.inference.post_process import apply_constraints_batch
from src.eval.metrics import weighted_score
from src.tools.joint_hillclimb_v12 import _build_pool_v12, _load_member


NEW_MEMBER = ("p11_electra_base", 42)
GRID = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
RNG = random.Random(20260504)
N_ITERS = 30000
NEW_MEMBER_PROB = 0.70

V12_TRAINING_SOTA = 0.6882469764180601
U1_ACTIVE_SOTA = 0.6887881104620621


def _require_member_ready(name: str, seed: int) -> None:
    ckpt_dir = Path("outputs/checkpoints") / name / f"seed{seed}"
    complete = ckpt_dir.exists() and all(
        (ckpt_dir / f"fold{i}" / "best.pt").exists() for i in range(5)
    )
    if not complete:
        n_done = sum((ckpt_dir / f"fold{i}" / "best.pt").exists() for i in range(5))
        raise RuntimeError(f"[pool] {name} not ready ({n_done}/5 folds). Train first.")


def _load_v12_winners(exps: list[str]) -> dict[str, list[float]]:
    meta_path = Path("reports/analysis/_ensemble/joint_hillclimb_v12_meta.json")
    with meta_path.open("r", encoding="utf-8") as file:
        meta = json.load(file)
    if meta["members"] != exps:
        raise RuntimeError(
            "[warm-start] v12 member order mismatch; refuse to reuse saved weights."
        )
    return {task: list(meta["best_w"][task]) for task in TASKS}


def _build_pool_v16() -> tuple[list[str], dict, dict[str, list[float]]]:
    exps, loaders, _ = _build_pool_v12()
    winners = _load_v12_winners(exps)

    name, seed = NEW_MEMBER
    _require_member_ready(name, seed)
    exps.append(name)
    loaders[name] = ("single", name, [seed])
    winners = {task: weights + [0.0] for task, weights in winners.items()}
    return exps, loaders, winners


def _build_accum(per_exp_probs, exps, w_per_task, n):
    out = {}
    for task in TASKS:
        weights = w_per_task[task]
        arr = np.zeros((n, NUM_LABELS[task]))
        weight_sum = sum(weights)
        if weight_sum == 0:
            for exp in exps:
                arr += per_exp_probs[exp][task]
            arr /= len(exps)
        else:
            for exp, weight in zip(exps, weights):
                if weight == 0:
                    continue
                arr += weight * per_exp_probs[exp][task]
            arr /= weight_sum
        out[task] = arr
    return out


def _score_joint(truth, accum, records, n) -> tuple[float, dict[str, float]]:
    preds = []
    for i in range(n):
        row = {"id": records[i].get("id", i)}
        for task in TASKS:
            row[task] = LABEL_DOMAINS[task][int(accum[task][i].argmax())]
        preds.append(row)
    constrained = apply_constraints_batch(preds)
    pred_dict = {task: [row[task] for row in constrained] for task in TASKS}
    score = weighted_score(truth, pred_dict)
    per_task = {task: score[task] for task in TASKS}
    return score["final_weighted_score"], per_task


def main() -> None:
    exps, loaders, weights = _build_pool_v16()
    new_idx = len(exps) - 1
    print(f"[pool] M={len(exps)} members")
    for idx, exp in enumerate(exps):
        marker = "  <-- NEW" if idx == new_idx else ""
        print(f"  [{idx:2d}] {exp}{marker}")

    records, _ = load_dataset("data/raw/vpesg4k_train_1000 V1.csv")
    n = len(records)
    truth = {task: [row[task] for row in records] for task in TASKS}
    per_exp_probs = {name: _load_member(name, n, loaders) for name in exps}

    accum = _build_accum(per_exp_probs, exps, weights, n)
    warm_score, warm_per_task = _score_joint(truth, accum, records, n)
    best_score = warm_score
    best_per_task = warm_per_task
    best_weights = {task: list(weights[task]) for task in TASKS}
    print(
        f"\n[warm-start v12-pad-0] joint={warm_score:.10f} "
        f"delta_vs_v12={warm_score - V12_TRAINING_SOTA:+.10f} "
        f"delta_vs_u1={warm_score - U1_ACTIVE_SOTA:+.10f}"
    )
    print(
        f"  T1={warm_per_task['promise_status']:.6f} "
        f"T2={warm_per_task['verification_timeline']:.6f} "
        f"T3={warm_per_task['evidence_status']:.6f} "
        f"T4={warm_per_task['evidence_quality']:.6f}"
    )

    accepted = 0
    history = []
    for iteration in range(1, N_ITERS + 1):
        task = RNG.choice(TASKS)
        # Zero-padded new members are easy to under-explore in a saturated pool.
        # Bias most perturbations toward p11, while still allowing legacy cleanup.
        if RNG.random() < NEW_MEMBER_PROB:
            idx = new_idx
        else:
            idx = RNG.randrange(len(exps) - 1)
        new_value = RNG.choice(GRID)
        old_value = best_weights[task][idx]
        if new_value == old_value:
            continue

        cand_weights = {item: list(best_weights[item]) for item in TASKS}
        cand_weights[task][idx] = new_value
        task_weight_sum = sum(cand_weights[task])
        if task_weight_sum == 0:
            continue

        cand_accum = dict(accum)
        arr = np.zeros((n, NUM_LABELS[task]))
        for exp, weight in zip(exps, cand_weights[task]):
            if weight == 0:
                continue
            arr += weight * per_exp_probs[exp][task]
        arr /= task_weight_sum
        cand_accum[task] = arr

        score, per_task = _score_joint(truth, cand_accum, records, n)
        if score > best_score:
            best_score = score
            best_per_task = per_task
            best_weights = cand_weights
            accum = cand_accum
            accepted += 1
            history.append(
                {
                    "iter": iteration,
                    "task": task,
                    "idx": idx,
                    "exp": exps[idx],
                    "new_val": new_value,
                    "old_val": old_value,
                    "score": score,
                    **{f"T{i + 1}": per_task[item] for i, item in enumerate(TASKS)},
                }
            )
            marker = " *NEW*" if idx == new_idx else ""
            print(
                f"[iter {iteration:5d}] accept #{accepted:3d} "
                f"{task:25s}[{idx:2d}={exps[idx]:25s}]{marker} "
                f"{old_value:.2f}->{new_value:.2f} joint={score:.10f}"
            )

    new_uses = sum(1 for item in history if item["idx"] == new_idx)
    new_nonzero = sum(1 for task in TASKS if best_weights[task][new_idx] != 0)
    print(f"\n[joint hillclimb v16/p11] DONE iters={N_ITERS} accepted={accepted}")
    print(
        f"[FINAL v16/p11] joint={best_score:.10f} "
        f"delta_vs_v12={best_score - V12_TRAINING_SOTA:+.10f} "
        f"delta_vs_u1={best_score - U1_ACTIVE_SOTA:+.10f}"
    )
    print(
        f"  T1={best_per_task['promise_status']:.6f} "
        f"T2={best_per_task['verification_timeline']:.6f} "
        f"T3={best_per_task['evidence_status']:.6f} "
        f"T4={best_per_task['evidence_quality']:.6f}"
    )
    print(
        f"[new member stats] {NEW_MEMBER[0]} accepted_perturbations={new_uses}/{accepted}, "
        f"final_nonzero_tasks={new_nonzero}/4"
    )
    print(
        "[new member final_w] "
        + " ".join(f"{task[:3]}={best_weights[task][new_idx]:.2f}" for task in TASKS)
    )

    out_dir = Path("reports/analysis/_ensemble")
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = "joint_hillclimb_v16_p11"

    final_preds = []
    for i in range(n):
        row = {"id": records[i].get("id", i)}
        for task in TASKS:
            row[task] = LABEL_DOMAINS[task][int(accum[task][i].argmax())]
        final_preds.append(row)
    constrained = apply_constraints_batch(final_preds)
    pd.DataFrame(constrained).to_csv(out_dir / f"{tag}_preds.csv", index=False)

    rows = []
    for task in TASKS:
        row = {"task": task, "f1": best_per_task[task]}
        for exp, weight in zip(exps, best_weights[task]):
            row[f"w_{exp}"] = weight
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_dir / f"{tag}_summary.csv", index=False)

    with (out_dir / f"{tag}_history.jsonl").open("w", encoding="utf-8") as file:
        for item in history:
            file.write(json.dumps(item) + "\n")

    with (out_dir / f"{tag}_meta.json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                "members": exps,
                "M": len(exps),
                "n_iters": N_ITERS,
                "new_member_prob": NEW_MEMBER_PROB,
                "accepted": accepted,
                "warm_start_score": warm_score,
                "final_score": best_score,
                "per_task": best_per_task,
                "best_w": {task: best_weights[task] for task in TASKS},
                "v12_training_sota": V12_TRAINING_SOTA,
                "u1_active_sota": U1_ACTIVE_SOTA,
                "delta_vs_v12_training": best_score - V12_TRAINING_SOTA,
                "delta_vs_u1_active": best_score - U1_ACTIVE_SOTA,
                "new_member": NEW_MEMBER[0],
                "new_member_accepted_perturbations": new_uses,
                "new_member_final_nonzero_tasks": new_nonzero,
            },
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(f"[wrote] {out_dir}/{tag}_summary.csv")
    print(f"[wrote] {out_dir}/{tag}_preds.csv")
    print(f"[wrote] {out_dir}/{tag}_history.jsonl")
    print(f"[wrote] {out_dir}/{tag}_meta.json")


if __name__ == "__main__":
    main()