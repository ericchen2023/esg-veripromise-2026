"""M3-v2 — Aggressive PDF paragraph extraction.

Improvements over v1 (510 paras / 5.7% yield):
  1. Smarter sentence-level splitting + merge into 60~800 char paragraphs.
  2. Length range relaxed: 40 <= L <= 800 (was 50 <= L <= 500).
  3. CJK ratio 0.55 (was 0.60), TOC filter only on >=4 hits (was 3).
  4. Drop ESG-keyword hard requirement; instead require >=1 of:
       (a) ESG keyword OR
       (b) commitment/timeline cue (數值 + 年份/百分比/年度詞)
     -> capture *quantified statements* even if keyword stem differs.
  5. Per-task minority-class candidate tagging so M4 can preferentially
     pseudo-label rare categories (T2 within_2_years, T4 Misleading/Not Clear).

Output schema (additive to v1):
  ticker, year, page, para_id, text, simhash, source_pdf,
  has_esg_kw (bool), has_quant_cue (bool),
  cand_t2 (str|None: heuristic timeline class hint),
  cand_t4 (str|None: heuristic quality class hint)
"""
from __future__ import annotations
import sys, io, os, re, json, glob, hashlib, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
import pdfplumber

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw", "u10")
OUT_DIR = os.path.join(ROOT, "data", "processed", "u10")
RPT_DIR = os.path.join(ROOT, "reports", "experiments", "u10")
os.makedirs(OUT_DIR, exist_ok=True); os.makedirs(RPT_DIR, exist_ok=True)
OUT_JSONL = os.path.join(OUT_DIR, "corpus_v2.jsonl")
STATS = os.path.join(RPT_DIR, "extract_v2_stats.json")
LABELED_JSON = os.path.join(ROOT, "vpesg4k_train_1000 V1.json")

# ===== text utils =====
RE_CJK = re.compile(r"[\u4e00-\u9fff]")
RE_WS = re.compile(r"[ \t\u3000]+")
RE_MULTI_NL = re.compile(r"\n{2,}")
RE_PAGE_NUM = re.compile(r"^\s*(?:第\s*\d+\s*頁|\d{1,3}\s*/\s*\d{1,3}|page\s*\d+)\s*$", re.I)
RE_NOISE = re.compile(r"[•·◆■▲►●◇□△▷○※☆★◎▼▽◀]+")
RE_URL = re.compile(r"https?://\S+")
RE_PURE_NUM = re.compile(r"^[\s\d.,%／/:：()\-—–]+$")
TOC_TOKENS = ["目錄","前言","附錄","關於本報告","章節","Chapter","CHAPTER","GRI 準則","GRI準則","SASB","TCFD"]
# Sentence boundary (Chinese full-stop / question / exclamation / semicolon)
RE_SENT = re.compile(r"(?<=[。！？；])")

def cjk_ratio(s: str) -> float:
    return len(RE_CJK.findall(s)) / max(1, len(s)) if s else 0.0

def normalize(s: str) -> str:
    s = RE_URL.sub("", s)
    s = RE_NOISE.sub("", s)
    s = RE_WS.sub(" ", s)
    return s.strip()

def looks_like_toc(s: str) -> bool:
    toc_hits = sum(1 for t in TOC_TOKENS if t in s)
    if toc_hits >= 4:
        return True
    en = sum(1 for c in s if c.isascii() and (c.isalpha() or c == ' '))
    if en / max(1, len(s)) >= 0.50:
        return True
    if s.count(' ') / max(1, len(s)) >= 0.25:
        return True
    return False

# ===== ESG / quant cue =====
ESG_KEYWORDS = [
    "永續","環境","碳","減碳","減排","排放","溫室氣體","節能","能源","再生能源","綠電","氣候","氣候變遷",
    "水資源","廢棄物","循環經濟","生物多樣性","污染","ISO 14001","ISO 14064","SBTi","RE100","淨零","碳中和",
    "社會","員工","人權","勞工","健康","安全","職安","訓練","培訓","多元","包容","平等","婦女","女性",
    "供應鏈","供應商","社區","公益","客戶","顧客","資訊安全","隱私","ISO 45001",
    "治理","董事","監察","風險","法遵","合規","稽核","誠信","反貪","利益衝突","資訊揭露",
    "承諾","目標","預計","規劃","期望","致力","計劃","推動","落實","達成",
]
RE_ESG = re.compile("|".join(map(re.escape, ESG_KEYWORDS)))
# Quantified statement cues — number + (year | %) within same paragraph
RE_PCT = re.compile(r"\d+(?:\.\d+)?\s*%")
RE_YEAR = re.compile(r"(?:20\d{2}|民國\s*1[01]\d|至\s*20\d{2}|於\s*20\d{2})")
RE_NUM = re.compile(r"\d+(?:[,.]\d+)*")

def has_quant_cue(s: str) -> bool:
    return bool(RE_PCT.search(s)) or bool(RE_YEAR.search(s))

# ===== heuristic minority-class candidate tagging =====
# T2 timeline classes: already / within_2_years / between_2_and_5_years / longer_than_5_years / N/A
RE_ALREADY = re.compile(r"(已|已經|過去|歷年|本年度|今年|本季|去年).{0,12}(達|完成|達成|實施|建置|取得|通過)")
RE_TARGET_YEAR = re.compile(r"(?:於|至|預計|目標|將於|計畫於|預期於)\s*(20\d{2})\s*年")

def tag_t2(s: str, doc_year: int):
    """Return ('already'|'within_2_years'|'between_2_and_5_years'|'longer_than_5_years'|None)."""
    if RE_ALREADY.search(s):
        return "already"
    m = RE_TARGET_YEAR.search(s)
    if m:
        target = int(m.group(1))
        diff = target - doc_year
        if diff <= 0:
            return "already"
        if diff <= 2:
            return "within_2_years"
        if diff <= 5:
            return "between_2_and_5_years"
        return "longer_than_5_years"
    return None

# T4 quality classes: Clear / Not Clear / Misleading / N/A
# Heuristic: misleading often combined with vague aspiration words, no quant.
# Not-Clear: strong commitment word but missing measurable target.
RE_VAGUE = re.compile(r"(致力|努力|期許|期望|希望|盼望|積極|持續推動|樂見|盼能)")
RE_MEASURABLE = re.compile(r"(達\s*\d|減少\s*\d|降低\s*\d|提升\s*\d|\d+\s*%|\d+\s*MW|\d+\s*噸|\d+\s*tCO2)")

def tag_t4(s: str):
    has_meas = bool(RE_MEASURABLE.search(s))
    has_vague = bool(RE_VAGUE.search(s))
    if has_meas:
        return "Clear"           # measurable -> likely Clear
    if has_vague and not has_meas:
        return "Not_Clear"       # commitment language but no measurable -> Not Clear candidate
    return None

# ===== sentence-merge paragraph builder =====
def split_paragraphs(page_text: str, target_min: int = 60, target_max: int = 800):
    """Yield merged paragraphs targeting 60~800 chars, falling back gracefully."""
    if not page_text:
        return
    blocks = RE_MULTI_NL.split(page_text)
    for blk in blocks:
        flat = blk.replace("\r", "").replace("\n", "")
        flat = normalize(flat)
        if not flat:
            continue
        # Already in target range -> emit
        if target_min <= len(flat) <= target_max:
            yield flat; continue
        # Too long -> split by sentences then greedy-merge
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
                    # If single sentence > target_max keep it (truncated reading model still ok)
                    if len(buf) > target_max:
                        yield buf[:target_max]; buf = ""
            if len(buf) >= target_min:
                yield buf
            continue
        # Too short -> drop (will be re-merged at higher level if context wraps)
        # No-op: short fragments dropped here, but page-level merge below handles them.

def emit_page_paragraphs(page_text: str):
    """Wrap split_paragraphs with one extra layer: if page has many short fragments
    sequence them and greedy-merge into 60~800 char buckets."""
    blocks = RE_MULTI_NL.split(page_text or "")
    flats = []
    for blk in blocks:
        flat = blk.replace("\r", "").replace("\n", "")
        flat = normalize(flat)
        if flat:
            flats.append(flat)
    # First pass: yield naturally-sized blocks
    leftovers = []
    for f in flats:
        if 60 <= len(f) <= 800:
            yield f
        elif len(f) > 800:
            yield from split_paragraphs(f)
        else:
            leftovers.append(f)
    # Second pass: greedy-merge short leftovers
    buf = ""
    for f in leftovers:
        if len(buf) + len(f) + 1 <= 800:
            buf = (buf + " " + f).strip() if buf else f
        else:
            if len(buf) >= 60:
                yield buf
            buf = f
    if len(buf) >= 60:
        yield buf

# ===== SimHash (same as v1) =====
def shingles(text: str, n: int = 4):
    text = re.sub(r"\s+", "", text)
    if len(text) < n:
        return [text] if text else []
    return [text[i:i+n] for i in range(len(text) - n + 1)]

def simhash64(text: str) -> int:
    grams = shingles(text, 4)
    if not grams: return 0
    v = [0]*64
    for g in grams:
        h = int.from_bytes(hashlib.blake2b(g.encode("utf-8"), digest_size=8).digest(), "big")
        for i in range(64):
            v[i] += 1 if (h >> i) & 1 else -1
    out = 0
    for i in range(64):
        if v[i] > 0: out |= (1 << i)
    return out

def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()

# ===== labeled fingerprints =====
print("Loading labeled set...", flush=True)
with open(LABELED_JSON, "r", encoding="utf-8") as f:
    labeled = json.load(f)
labeled_fps = []
for r in labeled:
    t = (r.get("data") or "").strip()
    if len(t) >= 30:
        labeled_fps.append(simhash64(t))
print(f"  labeled fingerprints: {len(labeled_fps)}", flush=True)

DEDUP_LABELED = 8
DEDUP_SELF = 6

def is_dup_vs_labeled(fp):
    return any(hamming(fp, lf) <= DEDUP_LABELED for lf in labeled_fps)

# ===== iterate PDFs =====
PDFS = sorted(glob.glob(os.path.join(RAW, "*_FY*.pdf")))
print(f"PDFs to process: {len(PDFS)}", flush=True)

stats = {
    "n_pdfs": len(PDFS), "per_pdf": {},
    "totals": {"raw":0, "filtered":0, "kept":0,
               "kept_with_esg":0, "kept_with_quant":0,
               "cand_t2": {"already":0,"within_2_years":0,"between_2_and_5_years":0,"longer_than_5_years":0},
               "cand_t4": {"Clear":0,"Not_Clear":0}},
    "started": time.strftime("%Y-%m-%d %H:%M:%S"),
}

written = 0
seen_self = []
with open(OUT_JSONL, "w", encoding="utf-8") as fo:
    for idx, pdf_path in enumerate(PDFS, 1):
        fname = os.path.basename(pdf_path)
        m = re.match(r"(\d+)_FY(\d{4})\.pdf$", fname)
        if not m:
            print(f"  skip {fname}"); continue
        ticker, year = m.group(1), int(m.group(2))
        t0 = time.time()
        per = {"raw":0, "filtered":0, "kept":0}
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
                        if L < 40 or L > 800: continue
                        if RE_PAGE_NUM.match(para): continue
                        if RE_PURE_NUM.match(para): continue
                        if cjk_ratio(para) < 0.55: continue
                        if looks_like_toc(para): continue
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
        stats["per_pdf"][fname] = {**per, "sec": round(dt,1)}
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
