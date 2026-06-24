"""Phase 37 — multi-provider LLM synthetic-data generator for ESG VeriPromise 2026.

Officially authorised by competition organisers: "在競賽過程中，僅『主辦方提供之
測試資料集』和『最後參賽者上傳的預測結果』絕對不能有人工介入。" → we may freely
craft / LLM-generate ADDITIONAL training data so long as we do not touch the
official test set or final predictions.

Pipeline (3 subcommands, idempotent):

    1. `generate`  — call LLM provider, write raw responses to JSONL.
    2. `validate`  — parse / dedup / hierarchy-check / length-filter → filtered JSONL.
    3. `promote`   — JSONL → CSV in pseudo-label schema (consumed by train_pseudo_kfold).

Providers (auto-detected from env / CLI):
    --provider mock      : deterministic template-based (default; no API needed)
    --provider openai    : needs OPENAI_API_KEY  (model default gpt-4o-mini)
    --provider anthropic : needs ANTHROPIC_API_KEY (claude-3-5-haiku-latest)
    --provider gemini    : needs GEMINI_API_KEY  (gemini-2.0-flash)
    --provider ollama    : needs local ollama server (default qwen2.5:7b)

Safety:
    * `mock` provider is the default → script runs with zero credentials.
    * Real providers gated behind explicit `--provider` flag.
    * Generated rows assert hierarchical-constraint validity before write.
    * Stable hash-based IDs guarantee idempotency across reruns.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.data.loader import LABEL_DOMAINS, load_dataset
from src.data.synth_schema import (
    PHASE37_TARGET_COUNTS,
    PSEUDO_CSV_COLUMNS,
    SynthRow,
    assert_hierarchy,
    assert_labels_valid,
    iter_target_specs,
)


# ============================================================================
# Prompt engineering
# ============================================================================

TASK_PROMPT_HEADER = """你是繁體中文 ESG 永續報告書資料增廣專家。請依照規格產生 1 段
真實感極強的中文段落，內容必須與企業 ESG 承諾／驗證／證據相關。

【標籤定義】
- promise_status (T1): 該段是否包含「企業可驗證的永續承諾」(Yes/No)
- verification_timeline (T2): 承諾完成時間 (already=已達成 / within_2_years / between_2_and_5_years / longer_than_5_years / N/A)
- evidence_status (T3): 是否提供具體量化證據 (Yes / No / N/A)
- evidence_quality (T4): 證據品質 (Clear=明確量化 / Not Clear=模糊 / Misleading=誇大或矛盾 / N/A)

【階層約束】
- 若 T1=No，T2/T3/T4 必為 N/A
- 若 T3=No，T4 必為 N/A
- 若 T1=Yes，T2 不可為 N/A

【產生規格】
- 標籤組合：{label_block}
- 字數：120~280 中文字
- 風格：模擬上市公司永續報告書段落（如台積電、台達電、中華電信等）
- 內容主題：請涵蓋 ESG 三大支柱中的隨機一項（環境 E / 社會 S / 公司治理 G）
- 嚴禁複製真實段落，務必原創；嚴禁出現「以下為範例」「本段為虛構」等元敘述

【輸出格式】
直接輸出段落本文，不要 JSON、不要編號、不要任何前後綴說明。"""


def build_prompt(labels: dict[str, str], seed_hint: str = "") -> str:
    label_block = (
        f"T1={labels['promise_status']}, "
        f"T2={labels['verification_timeline']}, "
        f"T3={labels['evidence_status']}, "
        f"T4={labels['evidence_quality']}"
    )
    body = TASK_PROMPT_HEADER.format(label_block=label_block)
    if seed_hint:
        body += f"\n\n【本次主題種子】{seed_hint}"
    return body


# Seed hints rotated per generation to enforce topic diversity.
SEED_HINTS_ENV = [
    "再生能源採購比例", "範疇一/二/三 碳排放", "用水回收與節水",
    "廢棄物減量與循環經濟", "綠色供應鏈管理", "RE100 與 SBTi 承諾",
    "ISO 14064 / ISO 14067", "PUE 與資料中心節能", "電動車隊轉型",
    "塑膠包材減量",
]
SEED_HINTS_SOC = [
    "員工教育訓練時數", "工安事故率", "供應商人權盡職調查",
    "社區回饋與志工時數", "性別薪資差距", "員工身心健康方案",
    "TFI 員工敬業度", "DEI 多元共融", "供應鏈勞工權益稽核",
]
SEED_HINTS_GOV = [
    "獨立董事比例", "風險管理框架", "資安治理 ISO 27001",
    "反賄賂 / 反洗錢內控", "薪酬委員會運作", "ESG 連結高階主管薪酬",
    "永續報告書第三方確信", "氣候相關財務揭露 TCFD", "永續委員會職能",
]
ALL_SEED_HINTS = SEED_HINTS_ENV + SEED_HINTS_SOC + SEED_HINTS_GOV


# ============================================================================
# Provider abstraction
# ============================================================================

class LLMProvider:
    name: str = "base"

    def generate(self, prompt: str, *, temperature: float = 0.85) -> str:
        raise NotImplementedError


class MockProvider(LLMProvider):
    """Deterministic template-based fallback (no API).

    Generates plausible but obviously-synthetic ESG text. Useful for unit tests
    and dry-runs. Output should NOT be promoted to training data in
    production — it is template noise.
    """

    name = "mock"

    _T1_NO_TEMPLATES = [
        "本章節概述本公司目前營運範疇與組織架構，未涉及具體永續承諾。{hint}相關現況請參考管理階層說明。",
        "本段落為公司治理結構之說明，內容包含董事會組成、稽核委員會職權範圍等基本資訊。{hint}",
    ]

    _T1_YES_TEMPLATES = [
        "本公司承諾於{deadline}完成{hint}相關目標，{evidence_phrase}",
        "為實現{hint}永續轉型，集團規劃於{deadline}達成階段性里程碑，{evidence_phrase}",
        "依循國際倡議，{hint}預計於{deadline}全面落實，{evidence_phrase}",
    ]

    _DEADLINE = {
        "already": "2023 年第四季",
        "within_2_years": "2026 年底前",
        "between_2_and_5_years": "2029 年",
        "longer_than_5_years": "2035 年",
    }

    _EVIDENCE_BY_T4 = {
        "Clear": "經第三方查證，2024 年實際達成率為 {pct}%，較基準年下降 {pct2} 百分點。",
        "Not Clear": "目前持續推動中，已啟動相關專案。",
        "Misleading": "本公司為業界永續標竿，相關指標領先全球同業平均水準甚多。",
        "N/A": "細節仍在規劃中。",
    }

    def __init__(self, seed: int = 42):
        self._rng = random.Random(seed)

    def generate(self, prompt: str, *, temperature: float = 0.85) -> str:
        # Parse label hint from prompt (mock doesn't actually call an LLM)
        m = re.search(r"T1=(\w+/?\w*), T2=([\w/]+), T3=(\w+/?\w*), T4=([\w /]+)", prompt)
        t1, t2, t3, t4 = (m.group(1), m.group(2), m.group(3), m.group(4)) if m else ("Yes", "already", "Yes", "Clear")
        hint_match = re.search(r"【本次主題種子】(.+?)$", prompt, re.DOTALL)
        hint = hint_match.group(1).strip() if hint_match else self._rng.choice(ALL_SEED_HINTS)

        if t1 == "No":
            tpl = self._rng.choice(self._T1_NO_TEMPLATES)
            return tpl.format(hint=hint)
        tpl = self._rng.choice(self._T1_YES_TEMPLATES)
        evidence_phrase = self._EVIDENCE_BY_T4.get(t4, "").format(
            pct=self._rng.randint(30, 95),
            pct2=self._rng.randint(2, 18),
        )
        return tpl.format(
            deadline=self._DEADLINE.get(t2, "未來年度"),
            hint=hint,
            evidence_phrase=evidence_phrase,
        )


class OpenAIProvider(LLMProvider):
    name = "openai"

    def __init__(self, model: str = "gpt-4o-mini"):
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as exc:
            raise RuntimeError("pip install openai") from exc
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY env var not set")
        self._client = OpenAI()
        self._model = model

    def generate(self, prompt: str, *, temperature: float = 0.85) -> str:
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=600,
        )
        return resp.choices[0].message.content or ""


class AnthropicProvider(LLMProvider):
    name = "anthropic"

    def __init__(self, model: str = "claude-3-5-haiku-latest"):
        try:
            import anthropic  # type: ignore
        except ImportError as exc:
            raise RuntimeError("pip install anthropic") from exc
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY env var not set")
        self._client = anthropic.Anthropic()
        self._model = model

    def generate(self, prompt: str, *, temperature: float = 0.85) -> str:
        msg = self._client.messages.create(
            model=self._model,
            max_tokens=800,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(block.text for block in msg.content if hasattr(block, "text"))


class GeminiProvider(LLMProvider):
    name = "gemini"

    def __init__(self, model: str = "gemini-2.0-flash"):
        try:
            import google.generativeai as genai  # type: ignore
        except ImportError as exc:
            raise RuntimeError("pip install google-generativeai") from exc
        if not os.environ.get("GEMINI_API_KEY"):
            raise RuntimeError("GEMINI_API_KEY env var not set")
        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        self._model = genai.GenerativeModel(model)

    def generate(self, prompt: str, *, temperature: float = 0.85) -> str:
        resp = self._model.generate_content(
            prompt,
            generation_config={"temperature": temperature, "max_output_tokens": 800},
        )
        return resp.text or ""


class OllamaProvider(LLMProvider):
    name = "ollama"

    def __init__(self, model: str = "qwen2.5:7b", host: str = "http://localhost:11434"):
        try:
            import requests  # type: ignore  # noqa
        except ImportError as exc:
            raise RuntimeError("pip install requests") from exc
        self._model = model
        self._host = host

    def generate(self, prompt: str, *, temperature: float = 0.85) -> str:
        import requests
        r = requests.post(
            f"{self._host}/api/generate",
            json={"model": self._model, "prompt": prompt, "stream": False, "options": {"temperature": temperature}},
            timeout=120,
        )
        r.raise_for_status()
        return r.json().get("response", "")


PROVIDER_REGISTRY: dict[str, Callable[[], LLMProvider]] = {
    "mock": MockProvider,
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
    "gemini": GeminiProvider,
    "ollama": OllamaProvider,
}


# ============================================================================
# Subcommand: generate
# ============================================================================

def cmd_generate(args: argparse.Namespace) -> None:
    provider = PROVIDER_REGISTRY[args.provider]()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    total_target = sum(s["n"] for s in iter_target_specs())
    print(f"[gen] provider={provider.name} target={total_target} rows -> {out_path}")
    n_written = 0
    with out_path.open("w", encoding="utf-8") as f:
        for spec in iter_target_specs():
            n = spec["n"]
            labels = spec["labels"]
            bucket = spec["bucket"]
            print(f"[gen] bucket={bucket:40s} n={n}")
            for i in range(n):
                hint = rng.choice(ALL_SEED_HINTS)
                prompt = build_prompt(labels, seed_hint=hint)
                try:
                    text = provider.generate(prompt, temperature=args.temperature)
                except Exception as exc:
                    print(f"  [warn] provider error at i={i}: {exc}; skipping")
                    continue
                row = {
                    "raw_text": text,
                    "labels": labels,
                    "provider": provider.name,
                    "bucket": bucket,
                    "seed_hint": hint,
                    "prompt_hash": hashlib.blake2b(prompt.encode("utf-8"), digest_size=8).hexdigest(),
                    "ts": time.time(),
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_written += 1
                if args.sleep > 0 and provider.name != "mock":
                    time.sleep(args.sleep)
    print(f"[gen] wrote {n_written} rows to {out_path}")


# ============================================================================
# Subcommand: validate
# ============================================================================

def _clean_text(raw: str) -> str:
    t = raw.strip()
    # Strip common LLM preambles
    for prefix in ("以下是", "以下為", "段落：", "段落:", "範例：", "範例:"):
        if t.startswith(prefix):
            t = t.split("\n", 1)[1].strip() if "\n" in t else t[len(prefix):].strip()
    # Strip surrounding quotes
    if t.startswith('"') and t.endswith('"'):
        t = t[1:-1].strip()
    if t.startswith("「") and t.endswith("」"):
        t = t[1:-1].strip()
    # Collapse whitespace
    t = re.sub(r"\s+", " ", t)
    return t


def _is_acceptable(text: str, *, min_len: int, max_len: int) -> tuple[bool, str]:
    if len(text) < min_len:
        return False, f"too_short({len(text)})"
    if len(text) > max_len:
        return False, f"too_long({len(text)})"
    if "本段為虛構" in text or "as an AI" in text.lower() or "i cannot" in text.lower():
        return False, "meta_disclaimer"
    # Must contain Chinese characters
    if not re.search(r"[\u4e00-\u9fff]", text):
        return False, "no_chinese"
    return True, "ok"


def cmd_validate(args: argparse.Namespace) -> None:
    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    official_records, _ = load_dataset(args.official)
    official_texts = {r["data"].strip() for r in official_records}

    seen_hashes: set[str] = set()
    kept = 0
    drops: dict[str, int] = {}

    with in_path.open(encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            row = json.loads(line)
            text = _clean_text(row["raw_text"])
            ok, reason = _is_acceptable(text, min_len=args.min_len, max_len=args.max_len)
            if not ok:
                drops[reason] = drops.get(reason, 0) + 1
                continue
            # Dedup vs official
            if text in official_texts:
                drops["dup_official"] = drops.get("dup_official", 0) + 1
                continue
            # Dedup within synth (SimHash-lite: 8-byte blake2b of normalised text)
            h = hashlib.blake2b(text.encode("utf-8"), digest_size=8).hexdigest()
            if h in seen_hashes:
                drops["dup_synth"] = drops.get("dup_synth", 0) + 1
                continue
            seen_hashes.add(h)
            # Hierarchy check (should already hold, but guard)
            try:
                assert_labels_valid(row["labels"])
            except ValueError as exc:
                drops[f"hier:{exc}"] = drops.get(f"hier:{exc}", 0) + 1
                continue
            row["text"] = text
            del row["raw_text"]
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            kept += 1

    print(f"[val] kept={kept}  dropped={sum(drops.values())}")
    for reason, n in sorted(drops.items(), key=lambda x: -x[1]):
        print(f"  - {reason:30s} {n}")
    print(f"[val] -> {out_path}")


# ============================================================================
# Subcommand: promote
# ============================================================================

def cmd_promote(args: argparse.Namespace) -> None:
    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    with in_path.open(encoding="utf-8") as f:
        for line in f:
            j = json.loads(line)
            row = SynthRow(
                text=j["text"],
                labels=j["labels"],
                source=f"u13_synth_llm:{j.get('provider', 'unknown')}",
                generator_meta={"seed_hint": j.get("seed_hint", ""), "bucket": j.get("bucket", "")},
                confidence=float(args.confidence),
            )
            rows.append(row.to_csv_dict())

    df = pd.DataFrame(rows, columns=list(PSEUDO_CSV_COLUMNS))
    # Final dedup on stable_id (idempotency guarantee)
    df = df.drop_duplicates(subset=["id"], keep="first")
    df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"[promote] {len(df)} rows -> {out_path}")
    # Print bucket distribution for QA
    grp = df.groupby(["verification_timeline", "evidence_quality"]).size()
    print("[promote] bucket distribution:")
    for (t2, t4), n in grp.items():
        print(f"  T2={t2:24s} T4={t4:12s}  {n}")


# ============================================================================
# Subcommand: merge
# ============================================================================

def cmd_merge(args: argparse.Namespace) -> None:
    """Concatenate multiple pseudo-label CSVs (e.g. U10 v3 + U13 synth)."""
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frames = []
    for f in args.inputs:
        df = pd.read_csv(f)
        print(f"[merge] {f}: {len(df)} rows")
        frames.append(df)
    merged = pd.concat(frames, ignore_index=True)
    # ID-based dedup (synth IDs are deterministic hashes -> idempotent)
    before = len(merged)
    merged = merged.drop_duplicates(subset=["id"], keep="first")
    print(f"[merge] dedup: {before} -> {len(merged)} (dropped {before - len(merged)})")
    merged.to_csv(out_path, index=False, encoding="utf-8")
    print(f"[merge] wrote {len(merged)} rows -> {out_path}")


# ============================================================================
# CLI
# ============================================================================

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="Call LLM and write raw JSONL.")
    g.add_argument("--provider", choices=list(PROVIDER_REGISTRY), default="mock")
    g.add_argument("--output", default="data/processed/u13/synth_raw.jsonl")
    g.add_argument("--temperature", type=float, default=0.85)
    g.add_argument("--seed", type=int, default=20260518)
    g.add_argument("--sleep", type=float, default=0.0, help="Seconds between API calls (rate limiting).")
    g.set_defaults(func=cmd_generate)

    v = sub.add_parser("validate", help="Parse / dedup / hierarchy-check raw JSONL.")
    v.add_argument("--input", default="data/processed/u13/synth_raw.jsonl")
    v.add_argument("--output", default="data/processed/u13/synth_filtered.jsonl")
    v.add_argument("--official", default="vpesg4k_train_1000 V1.csv")
    v.add_argument("--min-len", type=int, default=80)
    v.add_argument("--max-len", type=int, default=600)
    v.set_defaults(func=cmd_validate)

    pr = sub.add_parser("promote", help="Filtered JSONL -> pseudo-label CSV.")
    pr.add_argument("--input", default="data/processed/u13/synth_filtered.jsonl")
    pr.add_argument("--output", default="data/processed/u13/synth_pseudo.csv")
    pr.add_argument("--confidence", type=float, default=1.0,
                    help="Per-row confidence. LLM-generated: 0.95; manual: 1.0.")
    pr.set_defaults(func=cmd_promote)

    m = sub.add_parser("merge", help="Concat U10 pseudo CSV with U13 synth CSV.")
    m.add_argument("--inputs", nargs="+", required=True,
                   help="CSV files to concatenate (e.g. u10/pseudo_labels_v3.csv u13/synth_pseudo.csv).")
    m.add_argument("--output", required=True)
    m.set_defaults(func=cmd_merge)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
