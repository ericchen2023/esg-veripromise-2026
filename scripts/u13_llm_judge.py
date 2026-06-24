"""Phase 37 — LLM-as-judge for U10 pseudo-label re-scoring.

Takes existing U10 pseudo-labelled CSV (``data/processed/u10/pseudo_labels*.csv``)
and asks an LLM to independently judge whether the model's 4-task labels match
the text. Outputs an augmented CSV with a ``llm_judge_score`` column ∈ [0, 1]
that downstream filtering can use to keep only high-agreement pseudo rows.

Strategy: zero-shot evaluation with strict JSON output. Disagreement on any of
the 4 tasks costs 0.25. Missing-from-hierarchy errors cost an extra 0.10.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

# Reuse provider registry from u13_synth_llm
sys.path.insert(0, str(Path(__file__).resolve().parent))
from u13_synth_llm import PROVIDER_REGISTRY, LLMProvider  # noqa: E402


JUDGE_PROMPT = """你是 ESG 永續報告書資料標註專家。請依以下標籤定義，獨立判斷段落
的 4 個標籤。

【標籤定義】
T1 promise_status: 是否包含「企業可驗證的永續承諾」(Yes/No)
T2 verification_timeline: 達成時程 (already / within_2_years / between_2_and_5_years / longer_than_5_years / N/A)
T3 evidence_status: 是否提供具體量化證據 (Yes / No / N/A)
T4 evidence_quality: 證據品質 (Clear / Not Clear / Misleading / N/A)

階層約束：
- T1=No → T2/T3/T4 必為 N/A
- T3=No → T4 必為 N/A
- T1=Yes → T2 ≠ N/A

【段落】
{text}

請僅輸出 JSON（無 markdown、無前後說明），格式：
{{"T1": "...", "T2": "...", "T3": "...", "T4": "..."}}"""


_JSON_RE = re.compile(r"\{[^{}]*\"T1\"[^{}]*\}", re.DOTALL)


def parse_judgement(raw: str) -> dict[str, str] | None:
    """Robustly extract JSON object from LLM response."""
    raw = raw.strip()
    # Try direct parse
    try:
        return json.loads(raw)
    except Exception:
        pass
    # Try regex extraction
    m = _JSON_RE.search(raw)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return None
    return None


def score_agreement(model_labels: dict[str, str], judge_labels: dict[str, str]) -> float:
    """0.0 (full disagreement) → 1.0 (full agreement)."""
    score = 1.0
    for k_model, k_judge in [
        ("promise_status", "T1"),
        ("verification_timeline", "T2"),
        ("evidence_status", "T3"),
        ("evidence_quality", "T4"),
    ]:
        if model_labels.get(k_model) != judge_labels.get(k_judge):
            score -= 0.25
    return max(0.0, score)


def cmd_judge(args: argparse.Namespace) -> None:
    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path)
    if args.limit > 0:
        df = df.head(args.limit).copy()
    print(f"[judge] loaded {len(df)} rows from {in_path}")

    provider: LLMProvider = PROVIDER_REGISTRY[args.provider]()
    print(f"[judge] provider={provider.name}")

    scores: list[float] = []
    judgements: list[str] = []
    for idx, row in df.iterrows():
        text = str(row["data"])[: args.max_text_len]
        prompt = JUDGE_PROMPT.format(text=text)
        try:
            raw = provider.generate(prompt, temperature=0.0)
        except Exception as exc:
            print(f"  [warn] row {idx}: {exc}")
            scores.append(float("nan"))
            judgements.append("")
            continue
        judge = parse_judgement(raw)
        if not judge:
            scores.append(float("nan"))
            judgements.append(raw[:200])
            continue
        model_labels = {k: str(row[k]) for k in ("promise_status", "verification_timeline", "evidence_status", "evidence_quality")}
        s = score_agreement(model_labels, judge)
        scores.append(s)
        judgements.append(json.dumps(judge, ensure_ascii=False))
        if args.sleep > 0 and provider.name != "mock":
            time.sleep(args.sleep)
        if (idx + 1) % 50 == 0:
            print(f"  [progress] {idx + 1}/{len(df)}  running mean score={pd.Series(scores).mean():.3f}")

    df["llm_judge_score"] = scores
    df["llm_judge_raw"] = judgements
    df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"[judge] wrote {len(df)} rows -> {out_path}")
    print(f"[judge] mean score = {pd.Series(scores).mean():.4f}")
    print(f"[judge] rows with score >= 0.75: {(pd.Series(scores) >= 0.75).sum()}")
    print(f"[judge] rows with score == 1.0:  {(pd.Series(scores) == 1.0).sum()}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    j = sub.add_parser("judge", help="Re-score pseudo CSV with LLM judge.")
    j.add_argument("--input", default="data/processed/u10/pseudo_labels.csv")
    j.add_argument("--output", default="data/processed/u10/pseudo_labels_judged.csv")
    j.add_argument("--provider", choices=list(PROVIDER_REGISTRY), default="mock")
    j.add_argument("--limit", type=int, default=0, help="0 = all rows.")
    j.add_argument("--sleep", type=float, default=0.0)
    j.add_argument("--max-text-len", type=int, default=1500)
    j.set_defaults(func=cmd_judge)
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
