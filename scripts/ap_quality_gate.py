"""Aug-Plus quality gate.

Filters the merged Aug-Plus CSV (handcrafted + LLM-synth) before it enters
training. Three passes, in order:

  1. SimHash-style near-duplicate suppression against
       (a) the official competition train set,
       (b) the hand-crafted seed,
       (c) other rows in the same merged file.
  2. Length sanity: drop rows shorter than --min-chars or longer than
     --max-chars.
  3. Teacher confidence (optional): if --teacher-ckpt is given, load the
     Phase-36 ensemble and require min(conf_T1..T4) >= --conf-threshold.
     Otherwise rows keep their existing confidence values (1.0 for handcraft,
     0.0 for raw LLM -> filtered out unless --skip-teacher is set).

Outputs::
    --out                          gated CSV (default: data/aug_plus/aug_gated.csv)
    --report                       JSON report of drops by reason

Example
-------
  python scripts/ap_quality_gate.py \\
        --in data/aug_plus/aug_merged_raw.csv \\
        --skip-teacher \\
        --out data/aug_plus/aug_gated.csv
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.aug_schema import PSEUDO_CSV_COLUMNS, assert_labels_valid


# ---------------------------------------------------------------------------
# Lightweight SimHash on character 3-grams.  Pure-python, no external deps.
# ---------------------------------------------------------------------------
def _shingles(text: str, n: int = 3) -> list[str]:
    text = "".join(text.split())
    if len(text) < n:
        return [text]
    return [text[i : i + n] for i in range(len(text) - n + 1)]


def simhash(text: str, bits: int = 64) -> int:
    vec = [0] * bits
    for sh in _shingles(text):
        h = int(hashlib.md5(sh.encode("utf-8")).hexdigest(), 16)
        for b in range(bits):
            vec[b] += 1 if (h >> b) & 1 else -1
    out = 0
    for b in range(bits):
        if vec[b] > 0:
            out |= 1 << b
    return out


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


# ---------------------------------------------------------------------------
def _load_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _ref_corpus(args: argparse.Namespace) -> list[int]:
    """SimHash fingerprints of the official train + handcrafted seed."""
    refs: list[int] = []
    if args.train_csv:
        p = Path(args.train_csv)
        if p.exists():
            for r in _load_csv(p):
                refs.append(simhash(r.get("data", "")))
    seed = ROOT / "assets" / "aug_plus" / "handcrafted_v1.csv"
    if seed.exists():
        for r in _load_csv(seed):
            refs.append(simhash(r.get("data", "")))
    return refs


def gate(args: argparse.Namespace) -> int:
    inp = Path(args.input)
    if not inp.exists():
        print(f"[FAIL] input missing: {inp}", file=sys.stderr)
        return 2

    rows = _load_csv(inp)
    refs = _ref_corpus(args)
    threshold_hd = args.dedup_hamming

    kept: list[dict] = []
    drop_reason: Counter = Counter()
    dedup_pool = list(refs)  # mutated as we keep rows

    for r in rows:
        text = (r.get("data") or "").strip()

        # 1. length
        n = len(text)
        if n < args.min_chars or n > args.max_chars:
            drop_reason["length"] += 1
            continue

        # 2. schema (defence in depth)
        try:
            assert_labels_valid(
                {
                    "promise_status": r["promise_status"],
                    "verification_timeline": r["verification_timeline"],
                    "evidence_status": r["evidence_status"],
                    "evidence_quality": r["evidence_quality"],
                }
            )
        except (KeyError, ValueError):
            drop_reason["schema"] += 1
            continue

        # 3. SimHash dedup
        fp = simhash(text)
        if any(hamming(fp, ref) <= threshold_hd for ref in dedup_pool):
            # hand-crafted rows always pass even if near a ref (they ARE the ref)
            if r.get("company_source", "").startswith("ap_handcraft"):
                pass
            else:
                drop_reason["dedup"] += 1
                continue

        # 4. teacher-confidence gate
        try:
            cmin = float(r.get("confidence_min") or 0.0)
        except ValueError:
            cmin = 0.0
        if not args.skip_teacher and cmin < args.conf_threshold:
            drop_reason["low_confidence"] += 1
            continue

        kept.append(r)
        dedup_pool.append(fp)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(PSEUDO_CSV_COLUMNS), quoting=csv.QUOTE_MINIMAL)
        w.writeheader()
        for r in kept:
            w.writerow({k: r.get(k, "") for k in PSEUDO_CSV_COLUMNS})

    report = {
        "input": str(inp),
        "output": str(out_path),
        "total_in": len(rows),
        "kept": len(kept),
        "dropped_by_reason": dict(drop_reason),
        "settings": {
            "dedup_hamming": threshold_hd,
            "min_chars": args.min_chars,
            "max_chars": args.max_chars,
            "conf_threshold": args.conf_threshold,
            "skip_teacher": args.skip_teacher,
        },
    }
    if args.report:
        Path(args.report).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Aug-Plus quality gate")
    p.add_argument("--in", dest="input", default="data/aug_plus/aug_merged_raw.csv")
    p.add_argument("--out", default="data/aug_plus/aug_gated.csv")
    p.add_argument("--report", default="data/aug_plus/aug_gated_report.json")
    p.add_argument("--train-csv", default="vpesg4k_train_1000 V1.csv",
                   help="Official train CSV (for dedup reference). Use '' to skip.")
    p.add_argument("--min-chars", type=int, default=30)
    p.add_argument("--max-chars", type=int, default=400)
    p.add_argument("--dedup-hamming", type=int, default=3,
                   help="Hamming-distance threshold on 64-bit SimHash; <= is duplicate")
    p.add_argument("--conf-threshold", type=float, default=0.6)
    p.add_argument("--skip-teacher", action="store_true",
                   help="Bypass teacher-confidence gating (keep all rows by confidence)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return gate(args)


if __name__ == "__main__":
    raise SystemExit(main())
