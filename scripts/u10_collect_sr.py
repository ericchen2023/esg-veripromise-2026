"""U10-NEW SR PDF crawler.

Strategy:
  1. For each of 30 tickers, fetch its ESG landing page from u10_sources.COMPANY_SOURCES.
  2. Try plain HTTP first (requests + browser UA). If 0 PDF candidates, fall back to Selenium (handles JS-rendered SPAs).
  3. Extract all <a href="...pdf"> links. Also extract relative links containing keywords (sustain|esg|csr|report|永續|報告) and follow ONE level deep on same-domain.
  4. Filter candidates by:
       - URL/text contains SR keywords: 永續|sustain|esg|csr|報告書|sustainability|社會責任
       - URL/text does NOT contain: 年報|annual|股東會|法說|10-K|proxy|interim
       - Prefer links whose nearby text (or filename) mentions year 2023/2024/2025/112/113/114
  5. HEAD/GET to verify Content-Type is application/pdf or first 4 bytes == %PDF; require >=100KB.
  6. Save to data/raw/u10/{ticker}_{year}_{idx}_{md5short}.pdf and append manifest row.
  7. Cap at 3 PDFs per company (one per year preferred).

Run:
  python scripts/u10_collect_sr.py             # all 30
  python scripts/u10_collect_sr.py 1102 2357   # specific tickers
"""
from __future__ import annotations
import sys, os, re, csv, time, hashlib, urllib3, io
from urllib.parse import urljoin, urlparse
from datetime import datetime

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
urllib3.disable_warnings()

import requests
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from u10_sources import COMPANY_SOURCES

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(ROOT, "data", "raw", "u10")
LOG_DIR = os.path.join(ROOT, "reports", "experiments", "u10")
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
MANIFEST = os.path.join(LOG_DIR, "collect_manifest.csv")
LOG_PATH = os.path.join(LOG_DIR, "collect.log")

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")
HEADERS = {"User-Agent": UA, "Accept-Language": "zh-TW,zh;q=0.9,en;q=0.7"}

YEAR_TOKENS = ["2025", "2024", "2023", "114", "113", "112"]
SR_INC = re.compile(r"(永續|sustain|esg|csr|社會責任|報告書|sustainability|tcfd)", re.I)
SR_EXC = re.compile(r"(年報|annual|股東會|法說|10-?K|proxy|interim|季報|月報|quarter)", re.I)
TARGET_PER_COMPANY = 3
PDF_MIN_BYTES = 100_000


def log(msg: str):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def fetch_html(url: str, timeout: int = 25) -> str | None:
    try:
        r = requests.get(url, headers=HEADERS, timeout=timeout, verify=False, allow_redirects=True)
        if r.status_code != 200:
            log(f"    HTTP {r.status_code} for {url}")
            return None
        ct = r.headers.get("Content-Type", "").lower()
        if "html" not in ct and "text" not in ct:
            return None
        if not r.encoding or r.encoding.lower() == "iso-8859-1":
            r.encoding = r.apparent_encoding or "utf-8"
        return r.text
    except Exception as e:
        log(f"    fetch fail {url}: {type(e).__name__}: {str(e)[:120]}")
        return None


_driver = None
def fetch_html_selenium(url: str) -> str | None:
    global _driver
    try:
        if _driver is None:
            from selenium import webdriver
            from selenium.webdriver.chrome.options import Options
            opts = Options()
            opts.add_argument("--headless=new")
            opts.add_argument("--disable-gpu")
            opts.add_argument("--no-sandbox")
            opts.add_argument("--window-size=1400,1000")
            opts.add_argument(f"--user-agent={UA}")
            _driver = webdriver.Chrome(options=opts)
            _driver.set_page_load_timeout(45)
        _driver.get(url)
        time.sleep(4)
        return _driver.page_source
    except Exception as e:
        log(f"    selenium fail {url}: {type(e).__name__}: {str(e)[:120]}")
        return None


def extract_links(html: str, base_url: str) -> list[tuple[str, str]]:
    out = []
    for m in re.finditer(r'<a[^>]+href=["\']([^"\']+)["\'][^>]*>(.*?)</a>', html, re.I | re.S):
        href = m.group(1).strip()
        text = re.sub(r"<[^>]+>", " ", m.group(2)).strip()
        if not href or href.startswith("javascript:") or href.startswith("#"):
            continue
        try:
            absu = urljoin(base_url, href)
        except Exception:
            continue
        out.append((absu, text))
    return out


def looks_like_sr_pdf(url: str, text: str) -> bool:
    blob = (url + " " + text).lower()
    if not url.lower().split("?")[0].endswith(".pdf"):
        return False
    if SR_EXC.search(blob):
        return False
    return bool(SR_INC.search(blob))


def detect_year(url: str, text: str) -> str | None:
    blob = url + " " + text
    for tok in YEAR_TOKENS:
        if tok in blob:
            if tok == "114": return "2025"
            if tok == "113": return "2024"
            if tok == "112": return "2023"
            return tok
    return None


def download_pdf(url: str, dest: str) -> int:
    try:
        r = requests.get(url, headers=HEADERS, timeout=120, verify=False, stream=True, allow_redirects=True)
        if r.status_code != 200:
            return 0
        chunks = []
        first = b""
        for chunk in r.iter_content(chunk_size=65536):
            if chunk:
                if not first:
                    first = chunk[:4]
                    if first[:4] != b"%PDF":
                        return 0
                chunks.append(chunk)
        data = b"".join(chunks)
        if len(data) < PDF_MIN_BYTES:
            return 0
        with open(dest, "wb") as f:
            f.write(data)
        return len(data)
    except Exception as e:
        log(f"    download fail {url[:120]}: {type(e).__name__}")
        return 0


def collect_for(ticker, cn, en, landing, manifest_writer):
    log(f"\n=== {ticker} {cn} ({en}) ===")
    log(f"  landing: {landing}")
    html = fetch_html(landing)
    if not html or html.count("<a") < 5:
        log("  -> retry via Selenium")
        html = fetch_html_selenium(landing)
    if not html:
        log("  no HTML obtained, SKIP")
        return 0

    links = extract_links(html, landing)
    pdf_links = [(u, t) for (u, t) in links if u.lower().split("?")[0].endswith(".pdf")]
    log(f"  raw links: {len(links)}, pdf links: {len(pdf_links)}")

    if not pdf_links:
        host = urlparse(landing).netloc
        sub_pages = [(u, t) for (u, t) in links
                     if urlparse(u).netloc == host and SR_INC.search(u + " " + t)]
        log(f"  no direct PDFs; following {min(len(sub_pages),5)} sub-pages")
        seen = set()
        for su, st in sub_pages[:5]:
            if su in seen:
                continue
            seen.add(su)
            sub_html = fetch_html(su)
            if not sub_html:
                continue
            sub_links = extract_links(sub_html, su)
            for u, t in sub_links:
                if u.lower().split("?")[0].endswith(".pdf"):
                    pdf_links.append((u, t))
        log(f"  pdf links after expand: {len(pdf_links)}")

    sr_pdfs = [(u, t, detect_year(u, t)) for (u, t) in pdf_links if looks_like_sr_pdf(u, t)]
    log(f"  SR-filtered: {len(sr_pdfs)}")
    for u, t, y in sr_pdfs[:10]:
        log(f"    [{y}] {t[:40]} -> {u[:120]}")

    def prio(item):
        u, t, y = item
        order = {"2025": 0, "2024": 1, "2023": 2}.get(y, 3)
        return (order, -len(t))
    sr_pdfs.sort(key=prio)

    seen_years = set()
    saved = 0
    for u, t, y in sr_pdfs:
        if saved >= TARGET_PER_COMPANY:
            break
        if y and y in seen_years:
            continue
        md5 = hashlib.md5(u.encode()).hexdigest()[:8]
        ystr = y or "NA"
        fname = f"{ticker}_{ystr}_{saved+1}_{md5}.pdf"
        dest = os.path.join(RAW_DIR, fname)
        if os.path.exists(dest):
            log(f"  already have {fname}, skip")
            saved += 1
            if y:
                seen_years.add(y)
            continue
        size = download_pdf(u, dest)
        if size > 0:
            log(f"  + saved {fname} ({size/1024:.0f} KB)")
            manifest_writer.writerow({
                "ticker": ticker, "company_cn": cn, "company_en": en,
                "year": ystr, "size_bytes": size, "filename": fname,
                "url": u, "landing": landing, "anchor_text": t[:200],
            })
            saved += 1
            if y:
                seen_years.add(y)
        time.sleep(1.0)
    log(f"  === {ticker} done: {saved} PDFs ===")
    return saved


def main():
    only = sys.argv[1:]
    targets = [(tk, cn, en, url) for tk, (cn, en, url) in COMPANY_SOURCES.items()
               if not only or tk in only]
    log(f"\n###### U10-NEW collect: {len(targets)} companies ######")
    new_manifest = not os.path.exists(MANIFEST)
    with open(MANIFEST, "a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["ticker", "company_cn", "company_en", "year",
                                          "size_bytes", "filename", "url", "landing", "anchor_text"])
        if new_manifest:
            w.writeheader()
        total = 0
        for tk, cn, en, url in targets:
            try:
                total += collect_for(tk, cn, en, url, w)
                f.flush()
            except Exception as e:
                log(f"  EXCEPTION {tk}: {type(e).__name__}: {str(e)[:200]}")
    log(f"\n###### ALL DONE: {total} PDFs total ######")
    if _driver is not None:
        try: _driver.quit()
        except: pass


if __name__ == "__main__":
    main()
