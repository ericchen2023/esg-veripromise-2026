"""M3-v3 — Even more aggressive PDF paragraph extraction (per §F path 3).

Changes vs v2 (data/processed/u10/corpus_v2.jsonl):
  1. Length range: 30 <= L <= 1200 (was 40 <= L <= 800).
  2. Remove TOC space-ratio rule: drop the `s.count(' ') / len(s) >= 0.25`
     filter inside `looks_like_toc()` -- many ESG paragraphs in mixed-Chinese
     reports legitimately exceed that ratio after CJK-aware whitespace
     normalisation.
  3. Lower CJK ratio threshold to 0.50 (was 0.55) to admit more bilingual
     blocks.
  4. Keep all other heuristics (ESG keyword OR quantified cue, SimHash dedup,
     T2/T4 candidate tagging) unchanged so downstream u10_pseudo_label_v3.py
     can reuse the v2 schema verbatim.

Output: data/processed/u10/corpus_v3.jsonl
Stats : reports/experiments/u10/extract_v3_stats.json
"""
from __future__ import annotations
import sys, io, os, re, json, glob, time

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
import pdfplumber

# Reuse v2 helpers we want unchanged
from scripts.u10_pdf_extract_v2 import (
    RE_CJK, RE_WS, RE_MULTI_NL, RE_PAGE_NUM, RE_NOISE, RE_URL, RE_PURE_NUM,
    RE_SENT, TOC_TOKENS,
    cjk_ratio, normalize,
    ESG_KEYWORDS, RE_ESG, RE_PCT, RE_YEAR, RE_NUM, has_quant_cue,
    RE_ALREADY, RE_TARGET_YEAR, tag_t2,
    RE_VAGUE, RE_MEASURABLE, tag_t4,
    shingles, simhash64, hamming,
    DEDUP_LABELED, DEDUP_SELF,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw", "u10")
OUT_DIR = os.path.join(ROOT, "data", "processed", "u10")
RPT_DIR = os.path.join(ROOT, "reports", "experiments", "u10")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(RPT_DIR, exist_ok=True)
OUT_JSONL = os.path.join(OUT_DIR, "corpus_v3.jsonl")
STATS = os.path.join(RPT_DIR, "extract_v3_stats.json")
LABELED_JSON = os.path.join(ROOT, "vpesg4k_train_1000 V1.json")

# v3 length / quality thresholds
MIN_LEN = 30
MAX_LEN = 1200
CJK_MIN = 0.50

# v3 "looks like TOC" -- drop the space-ratio rule, keep token-hit + ASCII rule.
def looks_like_toc_v3(s: str) -> bool:
    toc_hits = sum(1 for t in TOC_TOKENS if t in s)
    if toc_hits >= 4:
        return True
    en = sum(1 for c in s if c.isascii() and (c.isalpha() or c == ' '))
    if en / max(1, len(s)) >= 0.50:
        return True
    return False


def split_paragraphs(page_text: str, target_min: int = MIN_LEN, target_max: int = MAX_LEN):
    if not page_text:
        return
    blocks = RE_MULTI_NL.split(page_text)
    for blk in blocks:
        flat = blk.replace("\r", "").replace("\n", "")
        flat = normalize(flat)
        if not flat:
            continue
        if target_min <= len(flat) <= target_max:
            yield flat
            continue
        if len(flat) > target_max:
            sents = [s for s in RE_SENT.split(flat) if s.strip()]
            buf = ""
            for s in sents:
                if len(buf) + len(s) <= target_max:
                    buf += s
                else:
                    if len(buf) >= target_min:
                        yield buf
                    buf = s
                    if len(buf) > target_max:
                        yield buf[:target_max]
                        buf = ""
            if len(buf) >= target_min:
                yield buf


def emit_page_paragraphs(page_text: str):
    blocks = RE_MULTI_NL.split(page_text or "")
    flats = []
    for blk in blocks:
        flat = blk.replace("\r", "").replace("\n", "")
        flat = normalize(flat)
        if flat:
            flats.append(flat)
    leftovers = []
    for f in flats:
        if MIN_LEN <= len(f) <= MAX_LEN:
            yield f
        elif len(f) > MAX_LEN:
            yield from split_paragraphs(f)
        else:
            leftovers.append(f)
    buf = ""
    for f in leftovers:
        if len(buf) + len(f) + 1 <= MAX_LEN:
            buf = (buf + " " + f).strip() if buf else f
        else:
            if len(buf) >= MIN_LEN:
                yield buf
            buf = f
    if len(buf) >= MIN_LEN:
        yield buf


def main() -> None:
    print(f"[v3] length=[{MIN_LEN},{MAX_LEN}] cjk_min={CJK_MIN} TOC space-rule=OFF", flush=True)
    print("Loading labeled set...", flush=True)
    with open(LABELED_JSON, "r", encoding="utf-8") as f:
        labeled = json.load(f)
    labeled_fps = []
    for r in labeled:
        t = (r.get("data") or "").strip()
        if len(t) >= MIN_LEN:
            labeled_fps.append(simhash64(t))
    print(f"  labeled fingerprints: {len(labeled_fps)}", flush=True)

    def is_dup_vs_labeled(fp):
        return any(hamming(fp, lf) <= DEDUP_LABELED for lf in labeled_fps)

    PDFS = sorted(glob.glob(os.path.join(RAW, "*_FY*.pdf")))
    print(f"PDFs to process: {len(PDFS)}", flush=True)

    stats = {
        "n_pdfs": len(PDFS),
        "per_pdf": {},
        "totals": {
            "raw": 0, "filtered": 0, "kept": 0,
            "kept_with_esg": 0, "kept_with_quant": 0,
            "cand_t2": {"already": 0, "within_2_years": 0, "between_2_and_5_years": 0, "longer_than_5_years": 0},
            "cand_t4": {"Clear": 0, "Not_Clear": 0},
        },
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {"min_len": MIN_LEN, "max_len": MAX_LEN, "cjk_min": CJK_MIN, "toc_space_rule": False},
    }

    written = 0
    seen_self: list[int] = []
    with open(OUT_JSONL, "w", encoding="utf-8") as fo:
        for idx, pdf_path in enumerate(PDFS, 1):
            fname = os.path.basename(pdf_path)
            m = re.match(r"(\d+)_FY(\d{4})\.pdf$", fname)
            if not m:
                print(f"  skip {fname}")
                continue
            ticker, year = m.group(1), int(m.group(2))
            t0 = time.time()
            per = {"raw": 0, "filtered": 0, "kept": 0}
            try:
                with pdfplumber.open(pdf_path) as pdf:
                    for page_no, page in enumerate(pdf.pages, 1):
                        try:
                            text = page.extract_text() or ""
                        except Exception:
                            text = ""
                        paras = list(emit_page_paragraphs(text))
                        for pi, para in enumerate(paras):
                            per["raw"] += 1
                            L = len(para)
                            if L < MIN_LEN or L > MAX_LEN:
                                continue
                            if RE_PAGE_NUM.match(para):
                                continue
                            if RE_PURE_NUM.match(para):
                                continue
                            if cjk_ratio(para) < CJK_MIN:
                                continue
                            if looks_like_toc_v3(para):
                                continue
                            has_esg = bool(RE_ESG.search(para))
                            has_q = has_quant_cue(para)
                            if not (has_esg or has_q):
                                continue
                            per["filtered"] += 1
                            fp = simhash64(para)
                            if any(hamming(fp, sf) <= DEDUP_SELF for sf in seen_self):
                                continue
                            if is_dup_vs_labeled(fp):
                                continue
                            seen_self.append(fp)
                            c2 = tag_t2(para, year)
                            c4 = tag_t4(para)
                            rec = {
                                "ticker": ticker, "year": year, "page": page_no, "para_id": pi,
                                "text": para, "simhash": fp, "source_pdf": fname,
                                "has_esg_kw": has_esg, "has_quant_cue": has_q,
                                "cand_t2": c2, "cand_t4": c4,
                            }
                            fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
                            written += 1
                            per["kept"] += 1
                            if has_esg: stats["totals"]["kept_with_esg"] += 1
                            if has_q: stats["totals"]["kept_with_quant"] += 1
                            if c2: stats["totals"]["cand_t2"][c2] += 1
                            if c4: stats["totals"]["cand_t4"][c4] += 1
            except Exception as e:
                print(f"  ERR {fname}: {e}")
                per["error"] = str(e)
            dt = time.time() - t0
            stats["per_pdf"][fname] = {**per, "sec": round(dt, 1)}
            stats["totals"]["raw"] += per["raw"]
            stats["totals"]["filtered"] += per["filtered"]
            stats["totals"]["kept"] += per["kept"]
            print(f"  [{idx:>2}/{len(PDFS)}] {fname} raw={per['raw']:>4} filt={per['filtered']:>3} kept={per['kept']:>3} ({dt:.1f}s)", flush=True)

    stats["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    stats["totals"]["written"] = written
    with open(STATS, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"\n=== TOTAL written {written} -> {OUT_JSONL}")
    print(f"=== Cand T2: {stats['totals']['cand_t2']}")
    print(f"=== Cand T4: {stats['totals']['cand_t4']}")


if __name__ == "__main__":
    main()
