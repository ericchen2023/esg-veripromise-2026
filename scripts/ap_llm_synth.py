"""Aug-Plus LLM synthesis CLI.

Generates synthetic Traditional-Chinese ESG-disclosure rows for the minority
classes identified by Phase-36 bottleneck analysis. Provider-agnostic with a
zero-dependency ``mock`` backend so the pipeline is fully testable without
any network or API key.

Subcommands
-----------
  generate   produce raw LLM rows  -> data/aug_plus/llm_synth_<target>.jsonl
  validate   schema/hierarchy check raw output, summarise errors
  merge      merge handcrafted + validated synth into a single training CSV
  promote    move gated CSV to data/processed/aug_plus/aug_plus_v1.csv

Example
-------
  # End-to-end with mock provider (works offline, deterministic):
  python scripts/ap_llm_synth.py generate --target misleading --n 40 \\
        --provider mock --seed 42
  python scripts/ap_llm_synth.py validate --target misleading
  python scripts/ap_llm_synth.py merge --out data/aug_plus/aug_merged_raw.csv

Providers are plugins in ``ap_providers()``. Real backends require the
corresponding SDK and an env-var-supplied API key (never hard-coded).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path
from typing import Callable

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.aug_schema import (
    AP_LLM_BASE,
    PSEUDO_CSV_COLUMNS,
    SynthRow,
    stable_synth_id,
    validate_rows,
)

PROMPT_DIR = ROOT / "configs" / "prompts"
OUT_DIR = ROOT / "data" / "aug_plus"
SEED_CSV = ROOT / "assets" / "aug_plus" / "handcrafted_v1.csv"

TARGETS: dict[str, str] = {
    "misleading":    "ap_misleading.yaml",
    "within_2_years": "ap_within_2_years.yaml",
}


# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------
ProviderFn = Callable[[dict, int, int], list[dict]]


def _mock_provider(spec: dict, n: int, seed: int) -> list[dict]:
    """Deterministic offline provider for tests and dry-runs.

    Builds synthetic rows by mixing template slots seeded from the prompt's
    fewshot examples. Output covers the exact label schema requested.
    """
    rng = random.Random(seed)
    label = spec["target"]["label"]
    fewshot = spec.get("fewshot", [])
    if not fewshot:
        raise ValueError("mock provider needs at least one fewshot exemplar")

    # Template slots
    companies = ["集團", "本公司", "公司", "我們", "本集團"]
    sectors = ["製造", "金融", "零售", "科技", "運輸", "食品", "電子", "化工"]
    metrics = [
        "減碳目標", "再生能源使用率", "女性主管比例", "供應鏈稽核覆蓋率",
        "廢棄物回收率", "員工教育訓練時數", "職安事故率", "水資源回收率",
        "包材循環使用比例", "董事會多元化比例",
    ]
    years_misleading = ["2028", "2029", "2030", "2031", "2032", "2034", "2035"]
    years_within2 = ["2026年底", "2027年初", "2027年第二季前", "明年底", "未來18個月內"]
    vague_evidence_phrases = [
        "已連續多年榮獲企業永續獎",
        "高度重視相關議題並設置專責委員會",
        "積極參與多場國際永續論壇",
        "已啟動跨部門研究小組蒐集相關資料",
        "持續宣導並懸掛標語以提升員工意識",
        "公司高層親自出席多場相關活動",
    ]

    out: list[dict] = []
    seen: set[str] = set()
    attempts = 0
    while len(out) < n and attempts < n * 6:
        attempts += 1
        co = rng.choice(companies)
        m = rng.choice(metrics)
        sec = rng.choice(sectors)
        if label == "T4_Misleading":
            yr = rng.choice(years_misleading)
            yi = int(yr)
            t2 = (
                "within_2_years" if yi - 2026 <= 2
                else "between_2_and_5_years" if yi - 2026 <= 5
                else "longer_than_5_years"
            )
            promise = f"承諾於{yr}年達成{sec}產業{m}領先水準"
            ev = rng.choice(vague_evidence_phrases)
            data = f"{co}{promise}，{ev}。"
            row = {
                "data": data,
                "promise_string": promise,
                "evidence_string": ev,
                "verification_timeline": t2,
            }
        elif label == "T2_within_2_years":
            yr = rng.choice(years_within2)
            promise = f"承諾於{yr}前完成{sec}業務之{m}改善計畫"
            has_ev = rng.random() < 0.55
            if has_ev:
                ev_quality = rng.choice(["Clear", "Not Clear"])
                ev = "目前已進入規劃中期階段並編列預算" if ev_quality == "Clear" else "正在進行內部討論"
                row = {
                    "data": f"{co}{promise}，{ev}。",
                    "promise_string": promise,
                    "evidence_string": ev,
                    "evidence_status": "Yes",
                    "evidence_quality": ev_quality,
                }
            else:
                row = {
                    "data": f"{co}{promise}。",
                    "promise_string": promise,
                    "evidence_string": "",
                    "evidence_status": "No",
                    "evidence_quality": "N/A",
                }
        else:
            raise ValueError(f"mock provider does not support label {label}")

        if row["data"] in seen:
            continue
        seen.add(row["data"])
        out.append(row)
    return out


def _openai_provider(spec: dict, n: int, seed: int) -> list[dict]:  # pragma: no cover
    raise NotImplementedError(
        "OpenAI provider stub. Implement using openai>=1.0 and "
        "OPENAI_API_KEY env var. Left intentionally out-of-scope for the "
        "scaffold so unit tests need no network."
    )


def _anthropic_provider(spec: dict, n: int, seed: int) -> list[dict]:  # pragma: no cover
    raise NotImplementedError("Anthropic stub.")


def _gemini_provider(spec: dict, n: int, seed: int) -> list[dict]:  # pragma: no cover
    raise NotImplementedError("Gemini stub.")


def _ollama_provider(spec: dict, n: int, seed: int) -> list[dict]:
    """Local-LLM provider using an Ollama daemon at ``OLLAMA_HOST``.

    Environment
    -----------
    OLLAMA_HOST  default ``http://localhost:11434``
    OLLAMA_MODEL default ``qwen2.5:7b-instruct``

    Requires only the Python stdlib (``urllib``) so the scaffold has no
    network deps unless this provider is actually selected.
    """
    import urllib.error
    import urllib.request

    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
    model = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b-instruct")

    target_label = spec["target"]["label"]
    system_prompt = spec.get("system_prompt", "").strip()
    fewshot = spec.get("fewshot", [])
    if not fewshot:
        raise ValueError("ollama provider needs at least one fewshot exemplar")

    # Render fewshot as JSONL block (single-line objects) so the model sees the
    # exact output shape it must emit.
    fewshot_block = "\n".join(json.dumps(ex, ensure_ascii=False) for ex in fewshot)
    user_template = spec.get("user_prompt_template", "請產生 {n} 則符合上述標註的 JSONL。")

    out: list[dict] = []
    seen: set[str] = set()
    rng = random.Random(seed)
    round_idx = 0
    max_rounds = 6  # over-generate to compensate for dup / invalid lines

    # Per-round generation target (slightly oversized to absorb losses)
    per_round = max(4, min(n, 16))

    while len(out) < n and round_idx < max_rounds:
        round_idx += 1
        req_n = min(per_round, n - len(out) + 4)
        round_seed = seed * 1000 + round_idx
        user_msg = user_template.format(n=req_n)
        # Append fewshot demonstration as an assistant turn so the model sees
        # the exact JSONL output shape and won't wrap output in code fences.
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"請先示範{len(fewshot)} 行符合規範的 JSONL（僅給 reference，不要重覆於後續輸出）。"},
            {"role": "assistant", "content": fewshot_block},
            {"role": "user", "content": user_msg + "\n\n請只輸出 JSONL，不要任何說明文字、不要 markdown code fence。"},
        ]
        body = json.dumps({
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": 0.85,
                "top_p": 0.92,
                "seed": round_seed,
                "num_predict": 1400,
            },
        }, ensure_ascii=False).encode("utf-8")

        url = f"{host}/api/chat"
        req = urllib.request.Request(url, data=body, method="POST",
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=240) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as e:
            print(f"[ollama] request failed (round {round_idx}): {e}", file=sys.stderr)
            continue

        text = (payload.get("message") or {}).get("content", "")
        if not text:
            print(f"[ollama] empty response (round {round_idx})", file=sys.stderr)
            continue

        # Strip accidental code fences, parse line-by-line.
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("```"):
                continue
            # Some models wrap as [ {...}, {...} ] — try array first.
            if line.startswith("[") or line.startswith("{") and line.endswith("},"):
                try:
                    arr = json.loads(line.rstrip(","))
                    candidates = arr if isinstance(arr, list) else [arr]
                except Exception:
                    candidates = []
            else:
                try:
                    candidates = [json.loads(line)]
                except Exception:
                    continue
            for obj in candidates:
                if not isinstance(obj, dict) or "data" not in obj:
                    continue
                txt = (obj.get("data") or "").strip()
                if not txt or txt in seen:
                    continue
                # minimal schema patch — required by _raw_to_synth_row downstream
                if target_label == "T4_Misleading":
                    if not obj.get("verification_timeline"):
                        continue
                    if obj["verification_timeline"] not in {
                        "within_2_years", "between_2_and_5_years", "longer_than_5_years",
                    }:
                        continue
                elif target_label == "T2_within_2_years":
                    es = obj.get("evidence_status")
                    if es not in {"Yes", "No"}:
                        # allow model to omit -> default to "No"
                        obj["evidence_status"] = "No"
                        obj["evidence_quality"] = "N/A"
                    elif es == "Yes" and obj.get("evidence_quality") not in {"Clear", "Not Clear"}:
                        continue
                seen.add(txt)
                out.append(obj)
                if len(out) >= n:
                    break
            if len(out) >= n:
                break

    if len(out) < n:
        print(
            f"[ollama] WARN: only produced {len(out)}/{n} rows after {round_idx} rounds",
            file=sys.stderr,
        )
    # deterministic order with a tiny shuffle keyed on seed for reproducibility
    rng.shuffle(out)
    return out[:n]


def ap_providers() -> dict[str, ProviderFn]:
    return {
        "mock":      _mock_provider,
        "openai":    _openai_provider,
        "anthropic": _anthropic_provider,
        "gemini":    _gemini_provider,
        "ollama":    _ollama_provider,
    }


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------
def load_prompt(target: str) -> dict:
    if target not in TARGETS:
        raise ValueError(f"unknown target {target!r}; valid: {list(TARGETS)}")
    path = PROMPT_DIR / TARGETS[target]
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _raw_to_synth_row(raw: dict, label_for: str) -> SynthRow:
    """Convert a provider raw JSON object into a SynthRow with conf=0.0."""
    if label_for == "misleading":
        return SynthRow(
            id=stable_synth_id(raw["data"], namespace="ap_llm"),
            data=raw["data"],
            promise_status="Yes",
            verification_timeline=raw["verification_timeline"],
            evidence_status="Yes",
            evidence_quality="Misleading",
            promise_string=raw.get("promise_string", ""),
            evidence_string=raw.get("evidence_string", ""),
            company_source="ap_llm_synth",
        )
    if label_for == "within_2_years":
        return SynthRow(
            id=stable_synth_id(raw["data"], namespace="ap_llm"),
            data=raw["data"],
            promise_status="Yes",
            verification_timeline="within_2_years",
            evidence_status=raw["evidence_status"],
            evidence_quality=raw["evidence_quality"],
            promise_string=raw.get("promise_string", ""),
            evidence_string=raw.get("evidence_string", ""),
            company_source="ap_llm_synth",
        )
    raise ValueError(f"unsupported label_for {label_for!r}")


def cmd_generate(args: argparse.Namespace) -> int:
    spec = load_prompt(args.target)
    providers = ap_providers()
    if args.provider not in providers:
        print(f"[FAIL] unknown provider {args.provider!r}", file=sys.stderr)
        return 2
    raw = providers[args.provider](spec, args.n, args.seed)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"llm_synth_{args.target}.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for r in raw:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[OK] generated {len(raw)} raw rows -> {out}")
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    path = OUT_DIR / f"llm_synth_{args.target}.jsonl"
    if not path.exists():
        print(f"[FAIL] missing {path}; run generate first", file=sys.stderr)
        return 2
    raws = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [_raw_to_synth_row(r, args.target) for r in raws]
    n_ok, errors = validate_rows(rows)
    print(f"[validate] target={args.target}  total={len(rows)}  ok={n_ok}  errors={len(errors)}")
    for rid, msg in errors[:5]:
        print(f"  - id={rid}: {msg}")
    return 0 if not errors else 1


def cmd_merge(args: argparse.Namespace) -> int:
    """Concatenate handcrafted seed + every validated LLM-synth jsonl."""
    if not SEED_CSV.exists():
        print(f"[FAIL] seed CSV missing: {SEED_CSV}", file=sys.stderr)
        return 2

    out_rows: list[dict] = []

    # Seed
    with SEED_CSV.open(encoding="utf-8", newline="") as f:
        out_rows.extend(csv.DictReader(f))

    # LLM synth
    for target in TARGETS:
        jp = OUT_DIR / f"llm_synth_{target}.jsonl"
        if not jp.exists():
            continue
        for line in jp.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            raw = json.loads(line)
            srow = _raw_to_synth_row(raw, target)
            try:
                srow.validate()
            except ValueError:
                continue
            out_rows.append({k: str(v) for k, v in srow.to_csv_row().items()})

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(PSEUDO_CSV_COLUMNS), quoting=csv.QUOTE_MINIMAL)
        w.writeheader()
        for r in out_rows:
            w.writerow({k: r.get(k, "") for k in PSEUDO_CSV_COLUMNS})
    print(f"[OK] merged {len(out_rows)} rows -> {out_path}")
    return 0


def cmd_promote(args: argparse.Namespace) -> int:
    """Copy the gated CSV to data/processed/aug_plus/ for training.

    With ``--with-u10`` the gated AP rows are concatenated to the U10 v2
    pseudo CSV and written under a distinct filename so the trainer can
    consume both pools via its single ``data.pseudo_csv_path`` field.
    """
    src = Path(args.gated_csv)
    if not src.exists():
        print(f"[FAIL] gated CSV missing: {src}", file=sys.stderr)
        return 2
    out_dir = ROOT / "data" / "processed" / "aug_plus"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.with_u10:
        u10 = Path(args.u10_csv)
        if not u10.exists():
            print(f"[FAIL] U10 CSV missing: {u10}", file=sys.stderr)
            return 2
        dst = out_dir / "aug_plus_v1_with_u10v2.csv"
        with dst.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(PSEUDO_CSV_COLUMNS), quoting=csv.QUOTE_MINIMAL)
            w.writeheader()
            total = 0
            for path in (src, u10):
                for r in _load_csv_safe(path):
                    w.writerow({k: r.get(k, "") for k in PSEUDO_CSV_COLUMNS})
                    total += 1
        print(f"[OK] promoted {total} rows (AP + U10v2) -> {dst}")
        return 0

    dst = out_dir / "aug_plus_v1.csv"
    dst.write_bytes(src.read_bytes())
    print(f"[OK] promoted {src} -> {dst}")
    return 0


def _load_csv_safe(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Aug-Plus LLM synthesis CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="generate raw synth rows")
    g.add_argument("--target", required=True, choices=list(TARGETS))
    g.add_argument("--n", type=int, default=40)
    g.add_argument("--provider", default="mock", choices=list(ap_providers()))
    g.add_argument("--seed", type=int, default=42)
    g.set_defaults(func=cmd_generate)

    v = sub.add_parser("validate", help="schema-check generated rows")
    v.add_argument("--target", required=True, choices=list(TARGETS))
    v.set_defaults(func=cmd_validate)

    m = sub.add_parser("merge", help="merge handcrafted + synth into one CSV")
    m.add_argument("--out", default="data/aug_plus/aug_merged_raw.csv")
    m.set_defaults(func=cmd_merge)

    pr = sub.add_parser("promote", help="copy gated CSV into data/processed/aug_plus/")
    pr.add_argument("--gated-csv", required=True)
    pr.add_argument("--with-u10", action="store_true",
                    help="Concatenate with U10 v2 pseudo CSV at promote time")
    pr.add_argument("--u10-csv", default="data/processed/u10/pseudo_labels_v2.csv")
    pr.set_defaults(func=cmd_promote)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
