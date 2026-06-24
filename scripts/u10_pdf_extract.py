"""M3 — U10-NEW PDF paragraph extraction + SimHash dedup vs labeled set.

Input  : data/raw/u10/{ticker}_FY{year}.pdf (62 files)
Output : data/processed/u10/corpus.jsonl
         reports/experiments/u10/extract_stats.json

Filters:
  - CJK ratio >= 0.60
  - 50 <= length <= 500 chars (token budget at max_length=384 leaves headroom)
  - contains >= 1 ESG keyword
  - 64-bit SimHash hamming > 8 vs every labeled `data` paragraph
"""
import sys, io, os, re, json, glob, hashlib, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
import pdfplumber

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW  = os.path.join(ROOT, "data", "raw", "u10")
OUT_DIR = os.path.join(ROOT, "data", "processed", "u10")
RPT_DIR = os.path.join(ROOT, "reports", "experiments", "u10")
os.makedirs(OUT_DIR, exist_ok=True); os.makedirs(RPT_DIR, exist_ok=True)
OUT_JSONL = os.path.join(OUT_DIR, "corpus.jsonl")
STATS = os.path.join(RPT_DIR, "extract_stats.json")
LABELED_JSON = os.path.join(ROOT, "vpesg4k_train_1000 V1.json")

# -------------------- text utils --------------------
RE_CJK = re.compile(r"[\u4e00-\u9fff]")
RE_WS  = re.compile(r"[ \t\u3000]+")
RE_MULTI_NL = re.compile(r"\n{2,}")
RE_PAGE_NUM = re.compile(r"^\s*(?:第\s*\d+\s*頁|\d{1,3}\s*/\s*\d{1,3}|page\s*\d+)\s*$", re.I)
RE_NOISE_TOKENS = re.compile(r"[•·◆■▲►●◇□△▷○※☆★◎▼▽◀]+")
RE_URL  = re.compile(r"https?://\S+")
RE_PURE_NUM = re.compile(r"^[\s\d.,%／/:：()\-—–]+$")
# TOC / header / footer signatures — usually short broken fragments scattered with chapter markers
TOC_TOKENS = ["目錄","前言","附錄","關於本報告","章節","Chapter","CHAPTER","GRI 準則","GRI準則","GRI 3","SASB","TCFD"]
RE_BROKEN_NL = re.compile(r"[A-Za-z]\s*\n\s*[\u4e00-\u9fff]")  # broken bilingual lines

def looks_like_toc(s: str) -> bool:
    # Many chapter markers
    toc_hits = sum(1 for t in TOC_TOKENS if t in s)
    if toc_hits >= 3:
        return True
    # high english/symbol ratio (header band) — relaxed
    en = sum(1 for c in s if c.isascii() and (c.isalpha() or c == ' '))
    if en / max(1, len(s)) >= 0.40:
        return True
    # Repeated short fragments separated by lots of spaces (visual layout junk)
    spaces = s.count(' ')
    if spaces / max(1, len(s)) >= 0.20:
        return True
    return False


ESG_KEYWORDS = [
    # E
    "永續","環境","碳","減碳","減排","排放","溫室氣體","節能","能源","再生能源","綠電","氣候",
    "水資源","廢棄物","循環經濟","生物多樣性","污染","ISO 14001","ISO 14064","SBTi","RE100","淨零",
    # S
    "社會","員工","人權","勞工","健康","安全","職安","訓練","培訓","多元","包容","平等","婦女","女性",
    "供應鏈","供應商","社區","公益","客戶","顧客","資訊安全","隱私","ISO 45001",
    # G
    "治理","董事","監察","風險","法遵","合規","稽核","誠信","反貪","利益衝突","資訊揭露",
    # 承諾語氣詞
    "承諾","目標","預計","規劃","期望","致力","計劃","推動","落實","達成","於 20","至 20",
]
RE_ESG = re.compile("|".join(map(re.escape, ESG_KEYWORDS)))

def cjk_ratio(s: str) -> float:
    if not s: return 0.0
    return len(RE_CJK.findall(s)) / max(1, len(s))

def normalize(s: str) -> str:
    s = RE_URL.sub("", s)
    s = RE_NOISE_TOKENS.sub("", s)
    s = RE_WS.sub(" ", s)
    return s.strip()

def split_paragraphs(page_text: str):
    if not page_text: return []
    # Treat 2+ newlines as paragraph break; then merge wrapped lines within paragraph
    raw = RE_MULTI_NL.split(page_text)
    out = []
    for chunk in raw:
        # collapse single newlines within paragraph (PDF line wraps)
        para = chunk.replace("\r", "").replace("\n", "")
        para = normalize(para)
        if not para: continue
        out.append(para)
    return out

# -------------------- 64-bit SimHash --------------------
def shingles(text: str, n: int = 4):
    text = re.sub(r"\s+", "", text)
    if len(text) < n: return [text] if text else []
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

# -------------------- build labeled fingerprints --------------------
print("Loading labeled set...", flush=True)
with open(LABELED_JSON, "r", encoding="utf-8") as f:
    labeled = json.load(f)
labeled_fps = []
for r in labeled:
    t = (r.get("data") or "").strip()
    if len(t) >= 30:
        labeled_fps.append(simhash64(t))
print(f"  labeled fingerprints: {len(labeled_fps)}", flush=True)

DEDUP_THRESHOLD = 8  # hamming > 8 to keep

def is_dup_vs_labeled(fp: int) -> bool:
    for lf in labeled_fps:
        if hamming(fp, lf) <= DEDUP_THRESHOLD:
            return True
    return False

# -------------------- iterate PDFs --------------------
PDFS = sorted(glob.glob(os.path.join(RAW, "*_FY*.pdf")))
print(f"PDFs to process: {len(PDFS)}", flush=True)

stats = {
    "n_pdfs": len(PDFS),
    "per_pdf": {},
    "totals": {"raw_paras":0, "after_filter":0, "after_self_dedup":0, "after_label_dedup":0},
    "started": time.strftime("%Y-%m-%d %H:%M:%S"),
}

written = 0
seen_self_fps = []   # for in-corpus dedup
SELF_THRESHOLD = 6

with open(OUT_JSONL, "w", encoding="utf-8") as fo:
    for idx, pdf_path in enumerate(PDFS, 1):
        fname = os.path.basename(pdf_path)
        m = re.match(r"(\d+)_FY(\d{4})\.pdf$", fname)
        if not m:
            print(f"  skip {fname} (bad name)"); continue
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
                    paras = split_paragraphs(text)
                    for pi, para in enumerate(paras):
                        per["raw"] += 1
                        L = len(para)
                        if L < 50 or L > 500: continue
                        if RE_PAGE_NUM.match(para): continue
                        if RE_PURE_NUM.match(para): continue
                        if cjk_ratio(para) < 0.60: continue
                        if not RE_ESG.search(para): continue
                        if looks_like_toc(para): continue
                        per["filtered"] += 1
                        fp = simhash64(para)
                        # in-corpus dedup
                        if any(hamming(fp, sf) <= SELF_THRESHOLD for sf in seen_self_fps):
                            continue
                        if is_dup_vs_labeled(fp):
                            continue
                        seen_self_fps.append(fp)
                        rec = {
                            "ticker": ticker, "year": year, "page": page_no, "para_id": pi,
                            "text": para, "simhash": fp, "source_pdf": fname,
                        }
                        fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        written += 1
                        per["kept"] += 1
        except Exception as e:
            print(f"  ERR {fname}: {e}")
            per["error"] = str(e)
        dt = time.time() - t0
        stats["per_pdf"][fname] = {**per, "sec": round(dt,1)}
        stats["totals"]["raw_paras"] += per["raw"]
        stats["totals"]["after_filter"] += per["filtered"]
        stats["totals"]["after_label_dedup"] += per["kept"]
        print(f"  [{idx:>2}/{len(PDFS)}] {fname} raw={per['raw']:>4} filt={per['filtered']:>3} kept={per['kept']:>3} ({dt:.1f}s)", flush=True)

stats["totals"]["after_self_dedup"] = written
stats["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")

with open(STATS, "w", encoding="utf-8") as f:
    json.dump(stats, f, ensure_ascii=False, indent=2)

print(f"\nDONE. Wrote {written} paragraphs -> {OUT_JSONL}")
print(f"Stats -> {STATS}")
