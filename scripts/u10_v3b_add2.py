"""v3b — add 2 disjoint tickers (1503, 9933)"""
import sys, io, os, re, time, urllib.parse, csv
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
import requests, urllib3
urllib3.disable_warnings()
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw", "u10")
RPT = os.path.join(ROOT, "reports", "experiments", "u10")
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124.0 Safari/537.36"}

TARGETS = {
    "1503": "士林電機廠股份有限公司",
    "9933": "中鼎工程股份有限公司",
}
WANT = {2023, 2024}

opts = Options(); opts.add_argument("--headless=new"); opts.add_argument("--window-size=1400,1000")
opts.add_argument(f"user-agent={UA['User-Agent']}")
drv = webdriver.Chrome(options=opts); drv.set_page_load_timeout(60)

RE_CARD = re.compile(r'href="(/reports/([^/]+)/([a-f0-9-]{36})/)"')
RE_DL = re.compile(
    r'<a\s+href="(https://esggenplus\.twse\.com\.tw/api/api/MopsSustainReport/data/FileStream\?id=[a-f0-9-]+)"'
    r'[^>]*data-ga-event="(detail_download_click|detail_history_download_click)"'
    r'[^>]*data-report-info="([^"]+)"', re.I)

def page(url, wait=4):
    drv.get(url); time.sleep(wait); return drv.page_source

def download_pdf(url, out):
    r = requests.get(url, headers=UA, timeout=180, stream=True, verify=False)
    if r.status_code != 200 or "pdf" not in r.headers.get("content-type","").lower():
        return False, f"http={r.status_code}"
    first = r.raw.read(4)
    if first[:4] != b"%PDF":
        return False, "no_magic"
    size = len(first)
    with open(out, "wb") as f:
        f.write(first)
        for c in r.iter_content(131072):
            if c: f.write(c); size += len(c)
    if size < 2_000_000:
        os.remove(out); return False, f"too_small={size}"
    return True, size

# Try multiple keyword variants per ticker
SEARCH_KEYWORDS = {
    "1503": ["士林電機","1503","士電"],
    "9933": ["中鼎","中鼎工程","9933"],
}

for ticker, cn in TARGETS.items():
    print(f"\n=== {ticker} {cn} ===")
    uuids, seen = [], set()
    for kw in SEARCH_KEYWORDS[ticker]:
        html = page(f"https://www.sustaihub.com/reports/?keyword={urllib.parse.quote(kw)}", wait=4)
        for _, name_enc, uid in RE_CARD.findall(html):
            n = urllib.parse.unquote(name_enc)
            if n == cn and uid not in seen:
                uuids.append(uid); seen.add(uid)
    print(f"  cards: {len(uuids)}")
    if not uuids:
        # show what was returned
        print("  Tried keywords found these names:")
        for kw in SEARCH_KEYWORDS[ticker]:
            html = page(f"https://www.sustaihub.com/reports/?keyword={urllib.parse.quote(kw)}", wait=3)
            for _, name_enc, uid in RE_CARD.findall(html):
                print(f"    kw={kw!r} -> {urllib.parse.unquote(name_enc)}")
                break
        continue

    pdfs, seen_pdf = [], set()
    for uid in uuids[:5]:
        dh = page(f"https://www.sustaihub.com/reports/{urllib.parse.quote(cn)}/{uid}/", wait=5)
        for url, _evt, info in RE_DL.findall(dh):
            m = re.search(r"-\s*(\d{4})", info)
            if not m: continue
            y = int(m.group(1))
            if y not in WANT or cn not in info: continue
            if (y, url) in seen_pdf: continue
            seen_pdf.add((y, url)); pdfs.append((url, y))
        if {2023, 2024} <= {y for _, y in pdfs}:
            break
    print(f"  candidates: {sorted({y for _, y in pdfs})}")
    for url, y in sorted(pdfs, key=lambda x: -x[1]):
        out = os.path.join(RAW, f"{ticker}_FY{y}.pdf")
        if os.path.exists(out): print(f"  [{y}] exists"); continue
        ok, info = download_pdf(url, out)
        print(f"  [{y}] {'ok' if ok else 'FAIL'} {info}")

drv.quit()
