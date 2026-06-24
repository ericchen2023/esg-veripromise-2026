"""U27 — Phase 47: isolate the ONE positive LB lever (T4 rare-class recovery).

Phase 46 LB verdict (uploaded mw_a03 = stored macro-weighted + corr0.3 -> 0.6029):
    task                 c3 (0.6037)   mw_a03 (0.6029)   delta
    promise_status        0.786          0.7864           +0.0004 (noise)
    verification_timeline 0.606          0.5959           -0.0101  <- macro WEIGHTING overfit T2
    evidence_status       0.675          0.675             0.0000  (binary frozen, identical)
    evidence_quality      0.437          0.4392           +0.0022  <- T4 weighting HELPED

Lessons:
  * macro-only stem WEIGHTING still overfits OOF (T2 dropped 0.010 despite +0.023 OOF).
    The Phase 45 "macro transfers faithfully" finding was about CONFIDENCE
    DISTRIBUTIONS, not about OOF weight optima. EQUAL weights win on LB for T2.
  * the ONLY positive signal was T4 (evidence_quality, the HEAVIEST task w=0.35):
    shifting its argmax toward rare classes (Misleading 0% -> 0.7%) gained +0.0022.

Because `mix` blends each task INDEPENDENTLY and the constraint cascade depends
only on T1/T3 (kept at equal weight), a candidate built from
    equal weights on T1/T2/T3   +   the learned T4 stem-weight   +   corr 0.3
reproduces EXACTLY c3's T1/T2/T3 (so T2 stays at its faithful 0.606) AND
mw_a03's T4 (0.4392). The LB is therefore decomposable:
    0.20*0.786 + 0.15*0.606 + 0.30*0.675 + 0.35*0.4392 ~= 0.6043  >  banked 0.6037.

This script:
  (1) OOF-validates the T4 lever: a T4-ONLY hillclimb (freeze T1/T2/T3) and a T4
      prior-correction alpha sweep, reporting evidence_quality macro-F1 so we can
      avoid the known "aggressive T4 correction craters to 0.49" failure mode;
  (2) emits the lever-isolated TEST candidates:
        p47_t4w_a03      equal T1/T2/T3 + learned-T4-weight + corr{T2:0.3,T4:0.3}
                         (== c3 T1/T2/T3 + mw_a03 T4; near-certain ~0.6043)
        p47_t4hc_a03     equal T1/T2/T3 + T4-only-hillclimb weight + corr 0.3
        p47_t4w_a04/a05  as p47_t4w_a03 but stronger T4 corr (only if OOF-safe)
        p47_t4w_tta_a03  TTA blend + learned-T4-weight + corr 0.3 (stack lever B)

Usage:
    python -m scripts.u27_phase47_t4_lever
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scripts.u16_tv_oof_ensemble import (
    COMBINED_CSV,
    TV_STEMS,
    _mix,
    _move_simplex_mass,
    _reconstruct_oof,
    _score,
)
from scripts.u17_phase42_test_inference import (
    OUT_DIR,
    load_test_records,
    mix,
    probs_to_records,
    write_submission,
)
from scripts.u18_decoding_experiments import load_cached_probs, marginal_table, train_priors
from scripts.u20_binary_prior_correction import prior_correct_per_task
from src.data.dataset import TASKS
from src.data.loader import load_dataset
from src.tools.validate_submission import validate_submission_frame

MACRO_META = Path("reports/analysis/_ensemble/macro_weighting_meta.json")
T4 = "evidence_quality"
T2 = "verification_timeline"
STORED_NPZ = OUT_DIR / "phase43_test_probs.npz"
MIDDLE_NPZ = OUT_DIR / "phase46_tta_middle_probs.npz"
TAIL_NPZ = OUT_DIR / "phase46_tta_tail_probs.npz"


# --------------------------------------------------------------------------- OOF
def t4_only_hillclimb(per_stem, records, *, init, n_iters, step, seed):
    """Post-constraint hillclimb perturbing ONLY the evidence_quality weight."""
    rng = random.Random(seed)
    best = {t: tuple(init[t]) for t in TASKS}
    best_score = _score(_mix(per_stem, best), records)
    for it in range(1, n_iters + 1):
        cand = dict(best)
        cand[T4] = _move_simplex_mass(best[T4], step=step, rng=rng)
        if cand[T4] == best[T4]:
            continue
        cs = _score(_mix(per_stem, cand), records)
        if cs["final_weighted_score"] > best_score["final_weighted_score"] + 1e-12:
            best, best_score = cand, cs
    return best, best_score


def oof_validate(per_stem_oof, records_oof, priors, t4_weight_meta):
    equal = {t: tuple([1.0] * len(TV_STEMS)) for t in TASKS}
    base = _score(_mix(per_stem_oof, equal), records_oof)
    print(f"[OOF] equal baseline = {base['final_weighted_score']:.6f}  "
          f"(T2={base[T2]:.4f}  T4={base[T4]:.4f})")

    # (a) learned meta T4 weight (from u23) applied with equal elsewhere
    w_meta = {t: equal[t] for t in TASKS}
    w_meta[T4] = tuple(t4_weight_meta)
    s_meta = _score(_mix(per_stem_oof, w_meta), records_oof)
    print(f"[OOF] equal + meta-T4-weight = {s_meta['final_weighted_score']:.6f}  "
          f"(T2={s_meta[T2]:.4f}  T4={s_meta[T4]:.4f})  dT4={s_meta[T4]-base[T4]:+.4f}")

    # (b) T4-only hillclimb (freeze T1/T2/T3)
    w_hc, s_hc = t4_only_hillclimb(per_stem_oof, records_oof, init=equal,
                                   n_iters=8000, step=0.1, seed=42)
    print(f"[OOF] T4-only hillclimb     = {s_hc['final_weighted_score']:.6f}  "
          f"(T2={s_hc[T2]:.4f}  T4={s_hc[T4]:.4f})  dT4={s_hc[T4]-base[T4]:+.4f}")

    # (c) T4 prior-correction alpha sweep on the equal blend (detect cratering)
    mixed_eq = _mix(per_stem_oof, equal)
    print("[OOF] T4 prior-corr alpha sweep (equal blend, T4 macro-F1):")
    safe_alphas = []
    for a in (0.2, 0.3, 0.4, 0.5, 0.6):
        corrected = prior_correct_per_task(mixed_eq, priors, {T4: a})
        sa = _score(corrected, records_oof)
        flag = "OK" if sa[T4] >= base[T4] - 1e-4 else "DROP"
        if sa[T4] >= base[T4] - 1e-4:
            safe_alphas.append(a)
        print(f"        alpha={a:.1f}  T4={sa[T4]:.4f}  total={sa['final_weighted_score']:.6f}  [{flag}]")
    return tuple(w_hc[T4]), safe_alphas


# --------------------------------------------------------------------------- TEST
def emit(name, mixed, records, priors):
    constrained = probs_to_records(mixed, records)
    preds_df = (pd.DataFrame(constrained)[["id", *TASKS]]
                .sort_values("id").reset_index(drop=True))
    rep = validate_submission_frame(preds_df, mode="preds")
    if not rep.ok:
        raise RuntimeError(f"[{name}] validation failed: {list(rep.errors)[:10]}")
    preds_df.to_csv(OUT_DIR / f"{name}_preds.csv", index=False, encoding="utf-8")
    sub_df = write_submission(constrained, OUT_DIR / f"{name}_submission.csv")
    q = sub_df["evidence_quality"].value_counts(normalize=True)
    print(f"\n=== {name} ===  (ok={rep.ok}, rows={rep.rows})")
    print(f"  T1 Yes={ (sub_df['promise_status']=='Yes').mean():.1%}  "
          f"T3 Yes={ (sub_df['evidence_status']=='Yes').mean():.1%}  "
          f"T4: Clear={q.get('Clear',0):.1%} NotClear={q.get('Not Clear',0):.1%} "
          f"Misleading={q.get('Misleading',0):.1%} NA={q.get('N/A',0):.1%}")
    return sub_df


def _load_view(path):
    data = np.load(path)
    return {stem: {t: data[f"{stem}__{t}"] for t in TASKS} for stem in TV_STEMS}


def _blend_views(views):
    out = {}
    for stem in TV_STEMS:
        out[stem] = {t: np.stack([v[stem][t] for v in views], axis=0).mean(axis=0)
                     for t in TASKS}
    return out


def main():
    if not MACRO_META.exists():
        raise SystemExit("run `python -m scripts.u23_macro_weighting` first")
    meta = json.loads(MACRO_META.read_text(encoding="utf-8"))
    t4_weight_meta = meta["stem_weights_per_task"][T4]
    print(f"[meta] learned T4 stem-weight = "
          f"{[round(x,3) for x in t4_weight_meta]}")

    priors = train_priors()

    # ---- OOF validation -----------------------------------------------------
    records_oof, _ = load_dataset(COMBINED_CSV)
    n = len(records_oof)
    per_stem_oof = {stem: _reconstruct_oof(stem, n) for stem in TV_STEMS}
    print(f"[OOF] {n} records, {len(TV_STEMS)} stems reconstructed")
    t4_hc_weight, safe_alphas = oof_validate(per_stem_oof, records_oof, priors, t4_weight_meta)

    # ---- TEST candidates ----------------------------------------------------
    device = torch.device("cpu")
    records = load_test_records("vpesg4k_test_2000.csv")
    per_stem = load_cached_probs(records, device, use_cache=True)  # stored view
    equal = {t: [1.0] * len(TV_STEMS) for t in TASKS}

    # PRIMARY: equal T1/T2/T3 + meta-T4-weight + corr{T2:0.3,T4:0.3}
    #          == c3 T1/T2/T3 + mw_a03 T4  ->  near-certain ~0.6043
    w_t4 = {t: equal[t] for t in TASKS}; w_t4[T4] = list(t4_weight_meta)
    mixed_t4 = mix(per_stem, w_t4)
    emit("phase47_t4w_a03", prior_correct_per_task(mixed_t4, priors, {T2: 0.3, T4: 0.3}),
         records, priors)

    # T4-only-hillclimb weight (independent OOF optimum) + corr 0.3
    w_hc = {t: equal[t] for t in TASKS}; w_hc[T4] = list(t4_hc_weight)
    mixed_hc = mix(per_stem, w_hc)
    emit("phase47_t4hc_a03", prior_correct_per_task(mixed_hc, priors, {T2: 0.3, T4: 0.3}),
         records, priors)

    # stronger T4 corr on top of the learned T4 weight (only OOF-safe alphas)
    for a in (0.4, 0.5):
        if a in safe_alphas:
            emit(f"phase47_t4w_a{int(a*10):02d}",
                 prior_correct_per_task(mixed_t4, priors, {T2: 0.3, T4: a}),
                 records, priors)
        else:
            print(f"[skip] T4 corr alpha={a} unsafe on OOF; not emitting")

    # STACK lever B: TTA blend + learned-T4-weight + corr 0.3
    if MIDDLE_NPZ.exists() and TAIL_NPZ.exists():
        views = [_load_view(STORED_NPZ), _load_view(MIDDLE_NPZ), _load_view(TAIL_NPZ)]
        per_stem_tta = _blend_views(views)
        w_t4_tta = {t: equal[t] for t in TASKS}; w_t4_tta[T4] = list(t4_weight_meta)
        mixed_tta = mix(per_stem_tta, w_t4_tta)
        emit("phase47_t4w_tta_a03",
             prior_correct_per_task(mixed_tta, priors, {T2: 0.3, T4: 0.3}),
             records, priors)
    else:
        print("[info] TTA views missing; skipping T4w+TTA stack")

    print("\n[done] Phase 47 candidates written to", OUT_DIR)


if __name__ == "__main__":
    main()
