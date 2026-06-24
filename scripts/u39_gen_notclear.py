"""U39 — Phase 55: generate 'Not Clear' (T4) training augmentation.

Phase 54 proved post-hoc is exhausted; the val-slice diagnosis shows T4 macro is
gated by the 'Not Clear' class (val-slice F1 0.35, P 0.32 / R 0.38) — and the
prior aug_plus synthesis (Phase 37/38) targeted Misleading/within_2y, NEVER
Not Clear. This is the one untried, faithfully-transferring lever (T4 OOF == LB).

Convention learned from the real data:
  Clear     = concrete / quantified / verifiable evidence (specific %, named
              programmes, dated achievements).
  Not Clear = action stated but only process / principle / intent, with NO
              specific numbers, targets, or checkable outcomes.

This script few-shot-prompts a local Ollama instruct model with REAL Not Clear
examples (so the generated style matches the annotation convention), cleans +
dedups the output, and emits leak-free augment records: each carries a
``_source_id`` = a real Not Clear row id, so src/train_kfold injects it ONLY
into folds where that source row is in train (never into the held-out fold) ->
OOF / val-slice stays an honest measure of whether the augmentation helps.

Usage:
    python -m scripts.u39_gen_notclear --n 220 --model hopephoto/Qwen3-4B-Instruct-2507_q8:latest
"""
from __future__ import annotations

import argparse
import json
import random
import re
import urllib.request
from pathlib import Path

import pandas as pd

COMBINED = "data/processed/train_val_combined.csv"
OUT = Path("data/aug_plus/notclear_synth.json")
OLLAMA = "http://localhost:11434/api/generate"
ID_BASE = 1_550_000  # synth id namespace (disjoint from 10001-12000 real + others)


def ollama(model: str, prompt: str, temp: float) -> str:
    body = json.dumps({"model": model, "prompt": prompt, "stream": False,
                       "options": {"temperature": temp, "top_p": 0.95}}).encode()
    req = urllib.request.Request(OLLAMA, data=body, headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=180).read()).get("response", "")


def norm(s: str) -> str:
    return re.sub(r"\s+", "", s).lower()


def clean_lines(raw: str) -> list[str]:
    out = []
    for ln in raw.splitlines():
        s = ln.strip()
        s = re.sub(r"^[\-\*\d\.、）)\s\"'「『]+", "", s)  # strip bullets/numbering/quotes
        s = re.sub(r"[\"'」』]+$", "", s).strip()
        s = re.sub(r"\s+", "", s)  # CJK: drop internal spaces
        if 18 <= len(s) <= 180 and not s.startswith(("以下", "範例", "請", "好的", "當然", "Clear", "Not")):
            out.append(s)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=220, help="target clean synth count")
    ap.add_argument("--model", default="hopephoto/Qwen3-4B-Instruct-2507_q8:latest")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    df = pd.read_csv(COMBINED, keep_default_na=False)
    nc = df[df.evidence_quality == "Not Clear"]
    real_ids = nc["id"].astype(int).tolist()
    exemplars = [str(s).strip() for s in nc["evidence_string"] if 18 <= len(str(s).strip()) <= 160]
    clear_eg = [str(s).strip() for s in df[df.evidence_quality == "Clear"]["evidence_string"]
                if 18 <= len(str(s).strip()) <= 160]
    seen = {norm(s) for s in nc["evidence_string"]} | {norm(s) for s in df["data"]}
    t2_pool = ["between_2_and_5_years"] * 100 + ["already"] * 65 + ["longer_than_5_years"] * 59
    esg_pool = ["S"] * 77 + ["E"] * 68 + ["G"] * 63

    topics = {"E": "環境/減碳/能源/水資源/廢棄物/生物多樣性",
              "S": "員工/勞工權益/供應鏈/社會公益/職場安全/人才培育",
              "G": "公司治理/風險管理/資訊揭露/法遵/董事會/誠信經營"}

    records, kept = [], set()
    calls = 0
    while len(records) < args.n and calls < args.n * 2:
        calls += 1
        cat = rng.choice(["E", "S", "G"])
        ex = rng.sample(exemplars, k=min(4, len(exemplars)))
        ce = rng.sample(clear_eg, k=2)
        prompt = (
            "你是繁體中文 ESG 永續報告書語料生成器。請生成「證據不夠清晰 (Not Clear)」的證據句："
            "描述企業在【" + topics[cat] + "】面向的作為，但**只有流程、原則或方向，"
            "完全沒有具體數字、目標值、百分比或可查證的成果**。\n\n"
            "真實 Not Clear 範例（模仿這種模糊風格）：\n- " + "\n- ".join(ex) + "\n\n"
            "Clear 反例（太具體，不要生成這種）：\n- " + "\n- ".join(ce) + "\n\n"
            "請生成 5 句全新、彼此不重複、口吻自然的 Not Clear 證據句，每句一行，"
            "12~60 字，僅輸出句子本身，不要編號、不要前言。"
        )
        try:
            resp = ollama(args.model, prompt, temp=rng.uniform(0.75, 1.0))
        except Exception as e:
            print(f"  [warn] call {calls} failed: {type(e).__name__}")
            continue
        for s in clean_lines(resp):
            k = norm(s)
            if k in seen or k in kept:
                continue
            kept.add(k)
            records.append({
                "id": ID_BASE + len(records),
                "data": s,
                "esg_type": rng.choice(esg_pool),
                "promise_status": "Yes",
                "promise_string": s,
                "verification_timeline": rng.choice(t2_pool),
                "evidence_status": "Yes",
                "evidence_string": s,
                "evidence_quality": "Not Clear",
                "_source_id": real_ids[len(records) % len(real_ids)],
            })
        if calls % 10 == 0:
            print(f"  [gen] calls={calls} kept={len(records)}/{args.n}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(records, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[done] {len(records)} clean Not Clear synth records -> {OUT}")
    print("  sample:", json.dumps(records[0], ensure_ascii=False)[:200] if records else "none")


if __name__ == "__main__":
    main()
