"""Per-task / per-view weighted TTA over U10 and AP-D stem stacks.

Default U10 stems (each = 3 seeds x 5 folds = 15 ckpts):
  - p2_combo_best                  (baseline, no pseudo)
  - p2_combo_best_u10_pseudo       (U10 v1 pseudo, 211 rows)
  - p2_combo_best_u10_pseudo_v2    (U10 v2 pseudo, 3,904 rows)

Use ``--stems`` to override this default for AP-D3/AP-D4/AP-D5 style stacks.

Existing equal-weight stack (3 stems, stored view only) = 0.67746.

This tool searches:

  1. Per-task per-stem mix weights w_{t,k} (k in {baseline,v1,v2}, sum=1).
  2. Per-task per-view alpha weights a_{t,v} (v in {stored,middle,tail}, sum=1).

Two-stage coordinate descent on disjoint simplices keeps the search cheap.
The scoring path uses ``FastTTAEvaluator`` for exact post-constraint scoring
with cached probability tensors; ``_eval_full_reference`` remains as a slow
auditable fallback for tests.
Saves results to ``reports/analysis/_ensemble``:

    u10_per_task_tta_summary.csv
    u10_per_task_tta_meta.json
    u10_per_task_tta_preds.csv   (final predictions after constraints)
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from src.data.dataset import LABEL_DOMAINS, NUM_LABELS, TASKS
from src.data.loader import load_dataset
from src.eval.metrics import weighted_score
from src.inference.post_process import apply_constraints_batch
from src.tools.oof_ensemble import _build_seed_oof
from src.tools.tta_fast_eval import FastTTAEvaluator
from src.tools.u1_tta_oof import predict_one_view


U10_STEMS = ("p2_combo_best", "p2_combo_best_u10_pseudo", "p2_combo_best_u10_pseudo_v2")
VIEWS = ("stored", "middle", "tail")
CACHE_DIR = Path("outputs/cache/u10_tta")
OUTPUT_DIR = Path("reports/analysis/_ensemble")
REFERENCE_SOTA_SCORE = 0.71608


def _stored_probs_for_stem(stem: str, n: int) -> dict[str, np.ndarray]:
    """Average stored OOF probs across all seeds of a stem."""
    exp_dir = Path("outputs/checkpoints") / stem
    splits_dir = Path("data/splits") / stem
    seeds = sorted({int(p.name.replace("seed", "")) for p in exp_dir.iterdir() if p.name.startswith("seed")})
    accum = {t: np.zeros((n, NUM_LABELS[t]), dtype=np.float64) for t in TASKS}
    for s in seeds:
        sp = _build_seed_oof(exp_dir, splits_dir, n, s)
        for t in TASKS:
            accum[t] += sp[t]
    for t in TASKS:
        accum[t] /= len(seeds)
    return accum


def _cache_path(stem: str, view: str) -> Path:
    return CACHE_DIR / f"{stem}_{view}.npz"


def _save_view(path: Path, probs: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{f"probs_{t}": probs[t].astype(np.float32) for t in TASKS})


def _load_view(path: Path, n: int) -> dict[str, np.ndarray]:
    z = np.load(path)
    out: dict[str, np.ndarray] = {}
    for t in TASKS:
        arr = z[f"probs_{t}"].astype(np.float64)
        if arr.shape != (n, NUM_LABELS[t]):
            raise RuntimeError(f"cache shape mismatch {path}: got {arr.shape}")
        out[t] = arr
    return out


def _ensure_nonstored_view(
    *,
    stem: str,
    view: str,
    records: list[dict[str, Any]],
    device: torch.device,
    use_amp: bool,
    batch_size: int | None,
) -> dict[str, np.ndarray]:
    cache = _cache_path(stem, view)
    if cache.exists():
        try:
            probs = _load_view(cache, len(records))
            print(f"[cache hit] {stem}/{view} <- {cache}")
            return probs
        except Exception as exc:
            print(f"[cache invalid] {stem}/{view} ({exc}); recomputing")
    probs = predict_one_view(
        records=records,
        member_name=stem,
        loader_spec=("auto", stem, []),
        view=view,
        device=device,
        use_amp=use_amp,
        batch_size_override=batch_size,
    )
    _save_view(cache, probs)
    print(f"[cache write] {stem}/{view} -> {cache}")
    return probs


def _score(probs_per_task: dict[str, np.ndarray], records: list[dict[str, Any]]) -> tuple[dict[str, float], list[dict[str, Any]]]:
    n = len(records)
    raw: list[dict[str, Any]] = []
    for i in range(n):
        row = {"id": records[i].get("id", i)}
        for t in TASKS:
            row[t] = LABEL_DOMAINS[t][int(probs_per_task[t][i].argmax())]
        raw.append(row)
    constrained = apply_constraints_batch(raw)
    truth = {t: [r[t] for r in records] for t in TASKS}
    pred = {t: [r[t] for r in constrained] for t in TASKS}
    return weighted_score(truth, pred), constrained


def _simplex_grid(step: float, k: int = 3) -> list[tuple[float, ...]]:
    """K-D simplex grid points (sum=1, multiples of step)."""
    n = int(round(1.0 / step))
    out: list[tuple[float, ...]] = []

    def _rec(prefix: list[int], remain: int, slots: int) -> None:
        if slots == 1:
            out.append(tuple((x / n) for x in prefix + [remain]))
            return
        for i in range(remain + 1):
            _rec(prefix + [i], remain - i, slots - 1)

    _rec([], n, k)
    return out


def _mix_stems(
    *,
    per_stem_per_view: dict[str, dict[str, dict[str, np.ndarray]]],
    stem_weights_per_task: dict[str, tuple[float, ...]],
) -> dict[str, dict[str, np.ndarray]]:
    """Return {view: {task: [N, C_t]}} by mixing stems per task."""
    out: dict[str, dict[str, np.ndarray]] = {v: {} for v in VIEWS}
    for view in VIEWS:
        for task in TASKS:
            ws = stem_weights_per_task[task]
            assert len(ws) == len(U10_STEMS), f"stem weight len {len(ws)} != stems {len(U10_STEMS)}"
            s = sum(ws)
            assert s > 0, f"stem weights zero for task {task}"
            arr = np.zeros_like(per_stem_per_view[U10_STEMS[0]][view][task], dtype=np.float64)
            for stem, w in zip(U10_STEMS, ws):
                if w == 0.0:
                    continue
                arr += w * per_stem_per_view[stem][view][task]
            out[view][task] = arr / s
    return out


def _mix_views(
    *,
    per_view: dict[str, dict[str, np.ndarray]],
    view_alpha_per_task: dict[str, tuple[float, float, float]],
) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for task in TASKS:
        a = view_alpha_per_task[task]
        s = sum(a)
        assert s > 0, f"view alpha zero for task {task}"
        arr = np.zeros_like(per_view[VIEWS[0]][task], dtype=np.float64)
        for view, w in zip(VIEWS, a):
            if w == 0.0:
                continue
            arr += w * per_view[view][task]
        out[task] = arr / s
    return out


def _eval_full_reference(
    *,
    per_stem_per_view: dict[str, dict[str, dict[str, np.ndarray]]],
    stem_weights_per_task: dict[str, tuple[float, ...]],
    view_alpha_per_task: dict[str, tuple[float, float, float]],
    records: list[dict[str, Any]],
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    pv = _mix_stems(per_stem_per_view=per_stem_per_view, stem_weights_per_task=stem_weights_per_task)
    final = _mix_views(per_view=pv, view_alpha_per_task=view_alpha_per_task)
    return _score(final, records)


def _eval_full(
    *,
    per_stem_per_view: dict[str, dict[str, dict[str, np.ndarray]]],
    stem_weights_per_task: dict[str, tuple[float, ...]],
    view_alpha_per_task: dict[str, tuple[float, float, float]],
    records: list[dict[str, Any]],
    evaluator: FastTTAEvaluator | None = None,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    if evaluator is None:
        evaluator = FastTTAEvaluator(
            per_stem_per_view=per_stem_per_view,
            records=records,
            stems=U10_STEMS,
            views=VIEWS,
        )
    return evaluator.score_and_predictions(
        stem_weights_per_task=stem_weights_per_task,
        view_alpha_per_task=view_alpha_per_task,
    )


def _move_simplex_mass(weights: tuple[float, ...], *, step: float, rng: random.Random) -> tuple[float, ...]:
    if len(weights) < 2 or step <= 0.0:
        return weights
    vals = list(weights)
    donors = [idx for idx, value in enumerate(vals) if value >= step - 1e-12]
    if not donors:
        return weights
    donor = rng.choice(donors)
    receivers = [idx for idx in range(len(vals)) if idx != donor]
    receiver = rng.choice(receivers)
    vals[donor] = max(0.0, vals[donor] - step)
    vals[receiver] += step
    total = sum(vals)
    if total <= 0.0:
        return weights
    return tuple(round(value / total, 10) for value in vals)


def random_refine_search(
    *,
    evaluator: FastTTAEvaluator,
    init_stem: dict[str, tuple[float, ...]],
    init_alpha: dict[str, tuple[float, float, float]],
    n_iters: int,
    step: float,
    seed: int,
    stem_probability: float = 0.75,
) -> tuple[
    dict[str, tuple[float, ...]],
    dict[str, tuple[float, float, float]],
    dict[str, float],
    list[dict[str, Any]],
]:
    """Budgeted local search over adjacent simplex points.

    This is meant for AP-D5-style follow-up searches where exhaustive 0.05
    grids are too expensive with 8+ stems. It never accepts a candidate unless
    the exact post-constraint weighted score improves.
    """
    rng = random.Random(seed)
    best_stem = {task: tuple(init_stem[task]) for task in TASKS}
    best_alpha = {task: tuple(init_alpha[task]) for task in TASKS}
    best_score = evaluator.score(stem_weights_per_task=best_stem, view_alpha_per_task=best_alpha)
    history: list[dict[str, Any]] = []

    for iteration in range(1, n_iters + 1):
        candidate_stem = dict(best_stem)
        candidate_alpha = dict(best_alpha)
        task = rng.choice(TASKS)
        group = "stem" if rng.random() < stem_probability else "view"
        if group == "stem":
            candidate_stem[task] = _move_simplex_mass(best_stem[task], step=step, rng=rng)
            if candidate_stem[task] == best_stem[task]:
                continue
        else:
            candidate_alpha[task] = _move_simplex_mass(best_alpha[task], step=step, rng=rng)
            if candidate_alpha[task] == best_alpha[task]:
                continue

        candidate_score = evaluator.score(
            stem_weights_per_task=candidate_stem,
            view_alpha_per_task=candidate_alpha,
        )
        if candidate_score["final_weighted_score"] > best_score["final_weighted_score"] + 1e-12:
            best_stem = candidate_stem
            best_alpha = candidate_alpha
            best_score = candidate_score
            event = {
                "iter": iteration,
                "group": group,
                "task": task,
                "score": best_score["final_weighted_score"],
            }
            history.append(event)
            print(
                f"[random refine {iteration:5d}] {group}/{task} -> "
                f"{best_score['final_weighted_score']:.10f}"
            )

    print(f"[random refine done] iters={n_iters} accepted={len(history)}")
    return best_stem, best_alpha, best_score, history


def stage_a_stem_search(
    *,
    per_stem_per_view: dict[str, dict[str, dict[str, np.ndarray]]],
    records: list[dict[str, Any]],
    grid_step: float,
    init_stem: dict[str, tuple[float, ...]] | None,
    fixed_view_alpha: dict[str, tuple[float, float, float]],
    max_rounds: int,
    evaluator: FastTTAEvaluator | None = None,
) -> tuple[dict[str, tuple[float, ...]], dict[str, float]]:
    """Per-task coordinate descent over stem simplex with fixed view alpha."""
    k = len(U10_STEMS)
    grid = _simplex_grid(grid_step, k)
    stem = init_stem or {t: tuple([1.0 / k] * k) for t in TASKS}
    score, _ = _eval_full(
        per_stem_per_view=per_stem_per_view,
        stem_weights_per_task=stem,
        view_alpha_per_task=fixed_view_alpha,
        records=records,
        evaluator=evaluator,
    )
    best = score["final_weighted_score"]
    print(f"[stage A init] {best:.10f} stem={stem}")
    for rd in range(1, max_rounds + 1):
        improved = False
        for task in TASKS:
            best_local, best_pt = best, stem[task]
            for pt in grid:
                trial = dict(stem)
                trial[task] = pt
                sc, _ = _eval_full(
                    per_stem_per_view=per_stem_per_view,
                    stem_weights_per_task=trial,
                    view_alpha_per_task=fixed_view_alpha,
                    records=records,
                    evaluator=evaluator,
                )
                if sc["final_weighted_score"] > best_local + 1e-12:
                    best_local, best_pt = sc["final_weighted_score"], pt
            if best_local > best + 1e-12:
                best, stem[task] = best_local, best_pt
                improved = True
                print(f"[stage A r{rd}] {task} -> {best_pt} -> {best:.10f}")
        if not improved:
            print(f"[stage A r{rd}] converged.")
            break
    score, _ = _eval_full(
        per_stem_per_view=per_stem_per_view,
        stem_weights_per_task=stem,
        view_alpha_per_task=fixed_view_alpha,
        records=records,
        evaluator=evaluator,
    )
    return stem, score


def stage_b_view_search(
    *,
    per_stem_per_view: dict[str, dict[str, dict[str, np.ndarray]]],
    records: list[dict[str, Any]],
    grid_step: float,
    fixed_stem: dict[str, tuple[float, ...]],
    init_alpha: dict[str, tuple[float, float, float]] | None,
    max_rounds: int,
    evaluator: FastTTAEvaluator | None = None,
) -> tuple[dict[str, tuple[float, float, float]], dict[str, float]]:
    grid = _simplex_grid(grid_step, 3)
    alpha = init_alpha or {t: (0.5, 0.5, 0.0) for t in TASKS}
    score, _ = _eval_full(
        per_stem_per_view=per_stem_per_view,
        stem_weights_per_task=fixed_stem,
        view_alpha_per_task=alpha,
        records=records,
        evaluator=evaluator,
    )
    best = score["final_weighted_score"]
    print(f"[stage B init] {best:.10f} alpha={alpha}")
    for rd in range(1, max_rounds + 1):
        improved = False
        for task in TASKS:
            best_local, best_pt = best, alpha[task]
            for pt in grid:
                trial = dict(alpha)
                trial[task] = pt
                sc, _ = _eval_full(
                    per_stem_per_view=per_stem_per_view,
                    stem_weights_per_task=fixed_stem,
                    view_alpha_per_task=trial,
                    records=records,
                    evaluator=evaluator,
                )
                if sc["final_weighted_score"] > best_local + 1e-12:
                    best_local, best_pt = sc["final_weighted_score"], pt
            if best_local > best + 1e-12:
                best, alpha[task] = best_local, best_pt
                improved = True
                print(f"[stage B r{rd}] {task} -> {best_pt} -> {best:.10f}")
        if not improved:
            print(f"[stage B r{rd}] converged.")
            break
    score, _ = _eval_full(
        per_stem_per_view=per_stem_per_view,
        stem_weights_per_task=fixed_stem,
        view_alpha_per_task=alpha,
        records=records,
        evaluator=evaluator,
    )
    return alpha, score


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/raw/vpesg4k_train_1000 V1.csv")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--grid-step", type=float, default=0.05)
    ap.add_argument("--max-rounds", type=int, default=4)
    ap.add_argument("--joint-rounds", type=int, default=2,
                    help="Alternating A->B refinement rounds after the first pass.")
    ap.add_argument("--random-refine-iters", type=int, default=0,
                    help="Budgeted local random refinement iterations after coordinate descent.")
    ap.add_argument("--random-refine-step", type=float, default=None,
                    help="Simplex mass moved per random refinement step; defaults to --grid-step.")
    ap.add_argument("--random-seed", type=int, default=20260525)
    ap.add_argument("--tag", default="u10_per_task_tta")
    ap.add_argument("--stems", nargs="+", default=None,
                    help="Override the default 3-stem U10 stack.")
    args = ap.parse_args()

    global U10_STEMS  # noqa: PLW0603
    if args.stems:
        U10_STEMS = tuple(args.stems)

    records, _ = load_dataset(args.data)
    n = len(records)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    use_amp = bool(device.type == "cuda" and not args.no_amp)
    print(f"[u10-tta] N={n} device={device} stems={U10_STEMS}")

    # Per-(stem, view) probs ---------------------------------------------------
    per_stem_per_view: dict[str, dict[str, dict[str, np.ndarray]]] = {s: {} for s in U10_STEMS}
    for stem in U10_STEMS:
        per_stem_per_view[stem]["stored"] = _stored_probs_for_stem(stem, n)
        for view in ("middle", "tail"):
            per_stem_per_view[stem][view] = _ensure_nonstored_view(
                stem=stem, view=view, records=records, device=device,
                use_amp=use_amp, batch_size=args.batch_size,
            )

    evaluator = FastTTAEvaluator(
        per_stem_per_view=per_stem_per_view,
        records=records,
        stems=U10_STEMS,
        views=VIEWS,
    )

    # Reference baselines -----------------------------------------------------
    K = len(U10_STEMS)
    EQ_STEM = tuple([1.0 / K] * K)
    eq_stem_pt = {t: EQ_STEM for t in TASKS}
    refs = {
        "u10_stack_stored_only":   {t: (1.0, 0.0, 0.0) for t in TASKS},
        "u10_stack_stored_middle": {t: (0.5, 0.5, 0.0) for t in TASKS},
        "u10_stack_stored_tail":   {t: (0.5, 0.0, 0.5) for t in TASKS},
        "u10_stack_three_eq":      {t: (1/3, 1/3, 1/3) for t in TASKS},
    }
    ref_scores: dict[str, dict[str, float]] = {}
    for name, alpha in refs.items():
        sc, _ = _eval_full(
            per_stem_per_view=per_stem_per_view,
            stem_weights_per_task=eq_stem_pt,
            view_alpha_per_task=alpha,
            records=records,
            evaluator=evaluator,
        )
        ref_scores[name] = sc
        print(f"[ref] eq-stem / {name}: {sc['final_weighted_score']:.10f}")

    sota_baseline = ref_scores["u10_stack_stored_only"]["final_weighted_score"]

    # Stage A: stem mix on stored only ----------------------------------------
    stored_only_alpha = {t: (1.0, 0.0, 0.0) for t in TASKS}
    stem_star, sa_score = stage_a_stem_search(
        per_stem_per_view=per_stem_per_view,
        records=records,
        grid_step=args.grid_step,
        init_stem=eq_stem_pt,
        fixed_view_alpha=stored_only_alpha,
        max_rounds=args.max_rounds,
        evaluator=evaluator,
    )
    print(f"[stage A done] {sa_score['final_weighted_score']:.10f} stem*={stem_star}")

    # Stage B: view alpha given stem* -----------------------------------------
    init_b = {t: (0.5, 0.5, 0.0) for t in TASKS}
    alpha_star, sb_score = stage_b_view_search(
        per_stem_per_view=per_stem_per_view,
        records=records,
        grid_step=args.grid_step,
        fixed_stem=stem_star,
        init_alpha=init_b,
        max_rounds=args.max_rounds,
        evaluator=evaluator,
    )
    print(f"[stage B done] {sb_score['final_weighted_score']:.10f} alpha*={alpha_star}")

    # Joint alternating refinement -------------------------------------------
    cur_score = sb_score["final_weighted_score"]
    for j in range(1, args.joint_rounds + 1):
        stem_star, sa = stage_a_stem_search(
            per_stem_per_view=per_stem_per_view,
            records=records,
            grid_step=args.grid_step,
            init_stem=stem_star,
            fixed_view_alpha=alpha_star,
            max_rounds=args.max_rounds,
            evaluator=evaluator,
        )
        alpha_star, sb = stage_b_view_search(
            per_stem_per_view=per_stem_per_view,
            records=records,
            grid_step=args.grid_step,
            fixed_stem=stem_star,
            init_alpha=alpha_star,
            max_rounds=args.max_rounds,
            evaluator=evaluator,
        )
        if sb["final_weighted_score"] - cur_score < 1e-9:
            print(f"[joint r{j}] converged.")
            break
        cur_score = sb["final_weighted_score"]
        print(f"[joint r{j}] {cur_score:.10f}")

    random_score: dict[str, float] | None = None
    random_history: list[dict[str, Any]] = []
    if args.random_refine_iters > 0:
        random_step = args.random_refine_step if args.random_refine_step is not None else args.grid_step
        stem_star, alpha_star, random_score, random_history = random_refine_search(
            evaluator=evaluator,
            init_stem=stem_star,
            init_alpha=alpha_star,
            n_iters=args.random_refine_iters,
            step=random_step,
            seed=args.random_seed,
        )
        print(f"[random refine score] {random_score['final_weighted_score']:.10f}")

    final_score, final_preds = _eval_full(
        per_stem_per_view=per_stem_per_view,
        stem_weights_per_task=stem_star,
        view_alpha_per_task=alpha_star,
        records=records,
        evaluator=evaluator,
    )
    delta_vs_stack = final_score["final_weighted_score"] - sota_baseline
    delta_vs_reference_sota = final_score["final_weighted_score"] - REFERENCE_SOTA_SCORE
    print(f"[u10-tta FINAL] {final_score['final_weighted_score']:.10f} "
          f"delta_vs_u10_stack={delta_vs_stack:+.10f} "
          f"delta_vs_AP_D4_SOTA({REFERENCE_SOTA_SCORE:.5f})={delta_vs_reference_sota:+.10f}")
    for t in TASKS:
        print(f"  task {t}: {final_score[t]:.6f}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, sc in ref_scores.items():
        rows.append({"variant": name, **{t: sc[t] for t in TASKS}, "weighted_score": sc["final_weighted_score"]})
    rows.append({"variant": "u10_stage_a_stem_search", **{t: sa_score[t] for t in TASKS}, "weighted_score": sa_score["final_weighted_score"]})
    rows.append({"variant": "u10_stage_b_view_search", **{t: sb_score[t] for t in TASKS}, "weighted_score": sb_score["final_weighted_score"]})
    if random_score is not None:
        rows.append({
            "variant": "u10_random_refine_search",
            **{t: random_score[t] for t in TASKS},
            "weighted_score": random_score["final_weighted_score"],
        })
    rows.append({"variant": "u10_per_task_tta_FINAL", **{t: final_score[t] for t in TASKS}, "weighted_score": final_score["final_weighted_score"]})
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / f"{args.tag}_summary.csv", index=False)
    pd.DataFrame(final_preds).to_csv(OUTPUT_DIR / f"{args.tag}_preds.csv", index=False)
    meta = {
        "stems": list(U10_STEMS),
        "ref_scores": {k: dict(v) for k, v in ref_scores.items()},
        "stem_star": {t: list(stem_star[t]) for t in TASKS},
        "alpha_star": {t: list(alpha_star[t]) for t in TASKS},
        "final_score": dict(final_score),
        "u10_stack_baseline": sota_baseline,
        "delta_vs_u10_stack": delta_vs_stack,
        "reference_sota_score": REFERENCE_SOTA_SCORE,
        "delta_vs_reference_sota": delta_vs_reference_sota,
        "delta_vs_active_SOTA": delta_vs_reference_sota,
        "grid_step": args.grid_step,
        "max_rounds": args.max_rounds,
        "joint_rounds": args.joint_rounds,
        "fast_eval": True,
        "random_refine_iters": args.random_refine_iters,
        "random_refine_step": args.random_refine_step if args.random_refine_step is not None else args.grid_step,
        "random_seed": args.random_seed,
        "random_refine_accepts": len(random_history),
        "random_refine_history": random_history,
    }
    (OUTPUT_DIR / f"{args.tag}_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[wrote] {OUTPUT_DIR / (args.tag + '_summary.csv')}")
    print(f"[wrote] {OUTPUT_DIR / (args.tag + '_meta.json')}")
    print(f"[wrote] {OUTPUT_DIR / (args.tag + '_preds.csv')}")


if __name__ == "__main__":
    main()
