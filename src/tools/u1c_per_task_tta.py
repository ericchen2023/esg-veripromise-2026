"""U1-c — per-task / per-view weighted TTA over the v12 active pool.

Differences vs `u1_tta_oof.py` (which only supports equal-weight view averaging):

- Caches each active member's stored / middle / tail per-view probabilities
  to ``outputs/cache/u1c_tta/{member}_{view}.npz`` so subsequent runs are fast.
- Searches a per-task simplex weight ``alpha_{t,v}`` (sum_v alpha_{t,v} = 1)
  jointly across tasks via coordinate descent on a discrete grid (default
  step 0.05 → 231 simplex points per task per round).
- Re-uses the v12 per-(task, member) weights — only the per-view mixing is
  optimised, so the v12 ensemble structure is preserved.
- Reports baseline (stored only), equal-weight `stored+middle`, equal-weight
  `stored+middle+tail`, oracle per-task best-of-3-variants, and the U1-c
  searched solution. The search is constraint-aware via
  ``apply_constraints_batch``.

Outputs to ``reports/analysis/_ensemble``:

    u1c_per_task_tta_summary.csv
    u1c_per_task_tta_preds.csv
    u1c_per_task_tta_meta.json
"""
from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from src.data.dataset import LABEL_DOMAINS, NUM_LABELS, TASKS
from src.data.loader import load_dataset
from src.eval.metrics import weighted_score
from src.inference.post_process import apply_constraints_batch
from src.tools.joint_hillclimb_v12 import _build_pool_v12, _load_member
from src.tools.u1_tta_oof import (
    DEFAULT_DATA,
    DEFAULT_META,
    active_members,
    load_meta,
    predict_one_view,
)


CACHE_DIR = Path("outputs/cache/u1c_tta")
OUTPUT_DIR = Path("reports/analysis/_ensemble")
VIEWS = ("stored", "middle", "tail")


def _cache_path(member: str, view: str) -> Path:
    return CACHE_DIR / f"{member}_{view}.npz"


def _save_view_probs(path: Path, probs: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        **{f"probs_{task}": probs[task].astype(np.float32) for task in TASKS},
    )


def _load_view_probs(path: Path, n: int) -> dict[str, np.ndarray]:
    data = np.load(path)
    out: dict[str, np.ndarray] = {}
    for task in TASKS:
        arr = data[f"probs_{task}"].astype(np.float64)
        if arr.shape != (n, NUM_LABELS[task]):
            raise RuntimeError(
                f"Cached view shape mismatch for {path}: got {arr.shape}, expected ({n}, {NUM_LABELS[task]})"
            )
        out[task] = arr
    return out


def _ensure_view_probs(
    *,
    member_name: str,
    loader_spec,
    view: str,
    records: list[dict[str, Any]],
    device: torch.device,
    use_amp: bool,
    batch_size_override: int | None,
) -> dict[str, np.ndarray]:
    cache = _cache_path(member_name, view)
    if cache.exists():
        try:
            probs = _load_view_probs(cache, len(records))
            print(f"[cache hit] {member_name}/{view} <- {cache}")
            return probs
        except Exception as exc:
            print(f"[cache invalid] {member_name}/{view} ({exc}); recomputing")
    probs = predict_one_view(
        records=records,
        member_name=member_name,
        loader_spec=loader_spec,
        view=view,
        device=device,
        use_amp=use_amp,
        batch_size_override=batch_size_override,
    )
    _save_view_probs(cache, probs)
    print(f"[cache write] {member_name}/{view} -> {cache}")
    return probs


def _ensemble_per_view(
    *,
    members: list[str],
    weights_per_task: dict[str, list[float]],
    member_view_probs: dict[str, dict[str, dict[str, np.ndarray]]],
    n: int,
) -> dict[str, dict[str, np.ndarray]]:
    """Return {view: {task: [N, C_t]}} after applying v12 per-task member weights."""
    out: dict[str, dict[str, np.ndarray]] = {v: {t: np.zeros((n, NUM_LABELS[t]), dtype=np.float64) for t in TASKS} for v in VIEWS}
    for view in VIEWS:
        for task in TASKS:
            ws = [float(w) for w in weights_per_task[task]]
            tot = float(sum(ws))
            if tot == 0:
                raise RuntimeError(f"v12 zero-weight task {task}")
            acc = np.zeros((n, NUM_LABELS[task]), dtype=np.float64)
            for member, w in zip(members, ws):
                if w == 0.0:
                    continue
                acc += w * member_view_probs[member][view][task]
            out[view][task] = acc / tot
    return out


def _score_with_alphas(
    *,
    alpha_per_task: dict[str, dict[str, float]],
    ensemble_per_view: dict[str, dict[str, np.ndarray]],
    records: list[dict[str, Any]],
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    n = len(records)
    mixed = {t: np.zeros((n, NUM_LABELS[t]), dtype=np.float64) for t in TASKS}
    for task in TASKS:
        a = alpha_per_task[task]
        s = sum(a.values())
        if s <= 0:
            raise RuntimeError(f"alpha sum=0 for task {task}")
        for view in VIEWS:
            mixed[task] += (a[view] / s) * ensemble_per_view[view][task]
    preds: list[dict[str, Any]] = []
    for i, rec in enumerate(records):
        row = {"id": rec.get("id", i)}
        for task in TASKS:
            row[task] = LABEL_DOMAINS[task][int(mixed[task][i].argmax())]
        preds.append(row)
    constrained = apply_constraints_batch(preds)
    pred_dict = {t: [r[t] for r in constrained] for t in TASKS}
    truth = {t: [r[t] for r in records] for t in TASKS}
    return weighted_score(truth, pred_dict), constrained


def _simplex_grid(step: float) -> list[tuple[float, float, float]]:
    n = int(round(1.0 / step))
    pts: list[tuple[float, float, float]] = []
    for i in range(n + 1):
        for j in range(n + 1 - i):
            k = n - i - j
            pts.append((i / n, j / n, k / n))
    return pts


def coordinate_descent_search(
    *,
    ensemble_per_view: dict[str, dict[str, np.ndarray]],
    records: list[dict[str, Any]],
    grid_step: float = 0.05,
    max_rounds: int = 6,
    seed_alpha: dict[str, dict[str, float]] | None = None,
    verbose: bool = True,
) -> tuple[dict[str, dict[str, float]], dict[str, float], list[dict[str, Any]]]:
    grid = _simplex_grid(grid_step)
    if seed_alpha is None:
        # Start from current SOTA: stored=middle=0.5, tail=0
        alpha = {t: {"stored": 0.5, "middle": 0.5, "tail": 0.0} for t in TASKS}
    else:
        alpha = {t: dict(seed_alpha[t]) for t in TASKS}
    score, _ = _score_with_alphas(alpha_per_task=alpha, ensemble_per_view=ensemble_per_view, records=records)
    best_score = score["final_weighted_score"]
    if verbose:
        print(f"[u1c init] {best_score:.10f} alpha={alpha}")

    for rd in range(1, max_rounds + 1):
        improved = False
        for task in TASKS:
            best_local = best_score
            best_pt = (alpha[task]["stored"], alpha[task]["middle"], alpha[task]["tail"])
            for s, m, t in grid:
                trial = {tt: dict(alpha[tt]) for tt in TASKS}
                trial[task] = {"stored": s, "middle": m, "tail": t}
                sc, _ = _score_with_alphas(alpha_per_task=trial, ensemble_per_view=ensemble_per_view, records=records)
                if sc["final_weighted_score"] > best_local + 1e-12:
                    best_local = sc["final_weighted_score"]
                    best_pt = (s, m, t)
            if best_local > best_score + 1e-12:
                improved = True
                best_score = best_local
                alpha[task] = {"stored": best_pt[0], "middle": best_pt[1], "tail": best_pt[2]}
                if verbose:
                    print(f"[u1c round {rd}] {task}: alpha={alpha[task]} -> {best_score:.10f}")
        if not improved:
            if verbose:
                print(f"[u1c round {rd}] converged.")
            break
    final_score, final_preds = _score_with_alphas(alpha_per_task=alpha, ensemble_per_view=ensemble_per_view, records=records)
    return alpha, final_score, final_preds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta", default=str(DEFAULT_META))
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--grid-step", type=float, default=0.05)
    parser.add_argument("--max-rounds", type=int, default=6)
    parser.add_argument("--tag", default="u1c_per_task_tta")
    args = parser.parse_args()

    meta = load_meta(Path(args.meta))
    members = active_members(meta)
    records, _ = load_dataset(args.data)
    n = len(records)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    use_amp = bool(device.type == "cuda" and not args.no_amp)
    print(f"[u1c] N={n} active_members={len(members)} device={device}")

    exps, loaders, _ = _build_pool_v12()
    loader_map = {member: loaders[member] for member in exps}

    # Per-member, per-view probs (cached on disk)
    member_view_probs: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    stored_member_probs = {member: _load_member(member, n, loaders) for member in meta["members"]}
    for member in meta["members"]:
        member_view_probs[member] = {"stored": stored_member_probs[member]}
    for member in members:
        for view in ("middle", "tail"):
            member_view_probs[member][view] = _ensure_view_probs(
                member_name=member,
                loader_spec=loader_map[member],
                view=view,
                records=records,
                device=device,
                use_amp=use_amp,
                batch_size_override=args.batch_size,
            )
    # Inactive members keep stored only — for completeness fill middle/tail = stored
    for member in meta["members"]:
        for view in ("middle", "tail"):
            member_view_probs[member].setdefault(view, member_view_probs[member]["stored"])

    weights_per_task = meta["best_w"]
    ens_per_view = _ensemble_per_view(
        members=meta["members"],
        weights_per_task=weights_per_task,
        member_view_probs=member_view_probs,
        n=n,
    )

    # Reference variants
    ref_alphas = {
        "stored_only": {t: {"stored": 1.0, "middle": 0.0, "tail": 0.0} for t in TASKS},
        "stored_middle_eq": {t: {"stored": 0.5, "middle": 0.5, "tail": 0.0} for t in TASKS},
        "stored_tail_eq": {t: {"stored": 0.5, "middle": 0.0, "tail": 0.5} for t in TASKS},
        "stored_middle_tail_eq": {t: {"stored": 1/3, "middle": 1/3, "tail": 1/3} for t in TASKS},
    }
    ref_scores: dict[str, dict[str, float]] = {}
    for name, alpha in ref_alphas.items():
        sc, _ = _score_with_alphas(alpha_per_task=alpha, ensemble_per_view=ens_per_view, records=records)
        ref_scores[name] = sc
        print(f"[ref] {name}: {sc['final_weighted_score']:.10f}")

    # Oracle per-task best-of-3-variants (warm start for U1-c search)
    oracle_alpha: dict[str, dict[str, float]] = {}
    for task in TASKS:
        best = None
        for name, alpha in ref_alphas.items():
            sc, _ = _score_with_alphas(
                alpha_per_task={tt: (alpha[tt] if tt == task else ref_alphas["stored_middle_eq"][tt]) for tt in TASKS},
                ensemble_per_view=ens_per_view,
                records=records,
            )
            cand = (sc[task], name)
            if best is None or cand > best:
                best = cand
        chosen = best[1]
        oracle_alpha[task] = dict(ref_alphas[chosen][task])
        print(f"[oracle] task={task} -> {chosen} alpha={oracle_alpha[task]}")
    oracle_score, _ = _score_with_alphas(alpha_per_task=oracle_alpha, ensemble_per_view=ens_per_view, records=records)
    print(f"[oracle joint] {oracle_score['final_weighted_score']:.10f}")

    # U1-c coordinate descent (warm start = oracle)
    alpha_star, score_star, preds_star = coordinate_descent_search(
        ensemble_per_view=ens_per_view,
        records=records,
        grid_step=args.grid_step,
        max_rounds=args.max_rounds,
        seed_alpha=oracle_alpha,
        verbose=True,
    )
    sota = ref_scores["stored_middle_eq"]["final_weighted_score"]
    delta = score_star["final_weighted_score"] - sota
    print(f"[u1c FINAL] joint={score_star['final_weighted_score']:.10f} delta_vs_active_SOTA={delta:+.10f}")
    print("[u1c per-task] " + " ".join(f"{t}={score_star[t]:.6f}" for t in TASKS))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    for name, sc in ref_scores.items():
        summary_rows.append({"variant": name, **{t: sc[t] for t in TASKS}, "weighted_score": sc["final_weighted_score"]})
    summary_rows.append({"variant": "oracle_per_task_best_of_eq3", **{t: oracle_score[t] for t in TASKS}, "weighted_score": oracle_score["final_weighted_score"]})
    summary_rows.append({"variant": "u1c_searched", **{t: score_star[t] for t in TASKS}, "weighted_score": score_star["final_weighted_score"]})
    pd.DataFrame(summary_rows).to_csv(OUTPUT_DIR / f"{args.tag}_summary.csv", index=False)
    pd.DataFrame(preds_star).to_csv(OUTPUT_DIR / f"{args.tag}_preds.csv", index=False)
    out_meta = {
        "active_SOTA_baseline": sota,
        "ref_scores": {k: dict(v) for k, v in ref_scores.items()},
        "oracle_alpha": oracle_alpha,
        "oracle_score": dict(oracle_score),
        "u1c_alpha_star": alpha_star,
        "u1c_score": dict(score_star),
        "delta_vs_active_SOTA": delta,
        "grid_step": args.grid_step,
        "max_rounds": args.max_rounds,
        "active_members": members,
        "v12_meta": str(args.meta),
    }
    (OUTPUT_DIR / f"{args.tag}_meta.json").write_text(json.dumps(out_meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[wrote] {OUTPUT_DIR / (args.tag + '_summary.csv')}")
    print(f"[wrote] {OUTPUT_DIR / (args.tag + '_preds.csv')}")
    print(f"[wrote] {OUTPUT_DIR / (args.tag + '_meta.json')}")


if __name__ == "__main__":
    main()
