"""U19 — quantify the u16(score)/u17(submission) decode mismatch on the OOF set.

u16's hillclimb scored each candidate with PLAIN argmax + apply_constraints
(so a promise=Yes row may still resolve to timeline/evidence=N/A), but the u17
submission writer FORCES non-N/A for promise=Yes rows (argmax-exclude-N/A).
The OOF=0.71033 weights were optimised for the FORMER decode. If re-scoring the
OOF with u17's exact constrained decode drops well below 0.71033, the decode
mismatch is a real contributor to the 0.5962 test collapse.

This reconstructs OOF probs for all 8 stems, mixes with the meta hillclimb
weights, and computes the weighted_score under THREE decodes:
  A. plain argmax + apply_constraints           (u16 scoring — the 0.71033 path)
  B. u17 constrained (force non-N/A for Yes)     (the actual submission path)
  C. equal-weight blend under decode A and B     (overfit check)

Usage:
    python -m scripts.u19_decode_mismatch_oof
"""
from __future__ import annotations

import json

from pathlib import Path

import numpy as np

from scripts.u16_tv_oof_ensemble import TV_STEMS, _mix, _reconstruct_oof
from src.data.dataset import ID2LABEL, LABEL2ID, TASKS
from src.data.loader import load_dataset
from src.eval.metrics import weighted_score
from src.inference.post_process import apply_constraints_batch

META_PATH = Path("reports/analysis/_ensemble/tv_oof_ensemble_meta.json")
TRAIN_CSV = "data/processed/train_val_combined.csv"


def decode_plain(mixed: dict[str, np.ndarray], records: list[dict]) -> list[dict]:
    """u16 decode: plain argmax over the full domain + apply_constraints_batch."""
    raw = []
    for i, rec in enumerate(records):
        row = {"id": rec.get("id", i)}
        for t in TASKS:
            row[t] = ID2LABEL[t][int(mixed[t][i].argmax())]
        raw.append(row)
    return apply_constraints_batch(raw)


def decode_constrained(mixed: dict[str, np.ndarray], records: list[dict]) -> list[dict]:
    """u17 decode: promise drives a hierarchical argmax-exclude-N/A cascade."""
    tl_na = LABEL2ID["verification_timeline"]["N/A"]
    es_na = LABEL2ID["evidence_status"]["N/A"]
    eq_na = LABEL2ID["evidence_quality"]["N/A"]
    es_yes = LABEL2ID["evidence_status"]["Yes"]

    def amx(p, exclude):
        m = p.copy()
        m[exclude] = -np.inf
        return int(m.argmax())

    out = []
    for i, rec in enumerate(records):
        promise = ID2LABEL["promise_status"][int(mixed["promise_status"][i].argmax())]
        row = {"id": rec.get("id", i), "promise_status": promise}
        if promise == "No":
            row["verification_timeline"] = "N/A"
            row["evidence_status"] = "N/A"
            row["evidence_quality"] = "N/A"
        else:
            row["verification_timeline"] = ID2LABEL["verification_timeline"][amx(mixed["verification_timeline"][i], tl_na)]
            es_idx = amx(mixed["evidence_status"][i], es_na)
            row["evidence_status"] = ID2LABEL["evidence_status"][es_idx]
            if es_idx == es_yes:
                row["evidence_quality"] = ID2LABEL["evidence_quality"][amx(mixed["evidence_quality"][i], eq_na)]
            else:
                row["evidence_quality"] = "N/A"
        out.append(row)
    return apply_constraints_batch(out)


def score(constrained: list[dict], records: list[dict]) -> dict[str, float]:
    truth = {t: [r[t] for r in records] for t in TASKS}
    pred = {t: [r[t] for r in constrained] for t in TASKS}
    return weighted_score(truth, pred)


def main() -> None:
    records, _ = load_dataset(TRAIN_CSV)
    n = len(records)
    print(f"[data] {n} OOF rows")

    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    stem_w = {t: tuple(meta["stem_weights_per_task"][t]) for t in TASKS}
    equal_w = {t: tuple([1.0] * len(TV_STEMS)) for t in TASKS}

    per_stem = {stem: _reconstruct_oof(stem, n) for stem in TV_STEMS}
    print(f"[oof] reconstructed {len(per_stem)} stems")

    mixed_hc = _mix(per_stem, stem_w)
    mixed_eq = _mix(per_stem, equal_w)

    results = {
        "A_hillclimb_plain   (u16 score path)": score(decode_plain(mixed_hc, records), records),
        "B_hillclimb_forced  (u17 submit path)": score(decode_constrained(mixed_hc, records), records),
        "C_equal_plain": score(decode_plain(mixed_eq, records), records),
        "D_equal_forced": score(decode_constrained(mixed_eq, records), records),
    }

    print("\n=== OOF weighted_score by (weights x decode) ===")
    for k, v in results.items():
        ws = v["final_weighted_score"]
        per = {t: round(v[t], 4) for t in TASKS if t in v}
        print(f"  {k:40s} weighted={ws:.5f}  per-task={per}")

    wa = results["A_hillclimb_plain   (u16 score path)"]["final_weighted_score"]
    wb = results["B_hillclimb_forced  (u17 submit path)"]["final_weighted_score"]
    print(f"\n[decode mismatch impact on OOF] A(plain)={wa:.5f} -> B(forced)={wb:.5f}  delta={wb-wa:+.5f}")


if __name__ == "__main__":
    main()
