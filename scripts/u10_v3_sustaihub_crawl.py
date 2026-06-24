"""v3 sustaihub crawler — fill missing tickers + add fresh ones.

Goal: bring U10-NEW SR corpus to 30 unique tickers, each with FY2023+FY2024.
Naming: {ticker}_FY{year}.pdf  (final unified format)
"""
import sys, io, os, re, time, urllib.parse, csv
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
import requests, urllib3
urllib3.disable_warnings()
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw", "u10")
RPT = os.path.join(ROOT, "reports", "experiments", "u10")
os.makedirs(RAW, exist_ok=True); os.makedirs(RPT, exist_ok=True)
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124.0 Safari/537.36"}

# Round-1 leftover tickers (re-collect via sustaihub) + brand-new picks
TARGETS = {
    "1326": "台灣化學纖維股份有限公司",
    "1402": "遠東新世紀股份有限公司",
    "2357": "華碩電腦股份有限公司",
    "2382": "廣達電腦股份有限公司",
    "2603": "長榮海運股份有限公司",
    "2890": "永豐金融控股股份有限公司",
    "8299": "群聯電子股份有限公司",
    "9910": "豐泰企業股份有限公司",
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

summary = {"ok":0, "fail_no_card":0, "fail_no_pdf":0}
manifest_rows = []

for ticker, cn in TARGETS.items():
    print(f"\n=== {ticker} {cn} ===")
    html = page(f"https://www.sustaihub.com/reports/?keyword={urllib.parse.quote(cn)}", wait=4)
    uuids, seen = [], set()
    for _, name_enc, uid in RE_CARD.findall(html):
        if urllib.parse.unquote(name_enc) == cn and uid not in seen:
            uuids.append(uid); seen.add(uid)
    print(f"  cards: {len(uuids)}")
    if not uuids:
        summary["fail_no_card"] += 1
        manifest_rows.append([ticker, cn, "", "", "no_card", 0, ""])
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
    years_found = sorted({y for _, y in pdfs})
    print(f"  candidates: {years_found}")
    if not pdfs:
        summary["fail_no_pdf"] += 1
        manifest_rows.append([ticker, cn, "", "", "no_pdf", 0, ""])
        continue

    got_any = False
    for url, y in sorted(pdfs, key=lambda x: -x[1]):
        out = os.path.join(RAW, f"{ticker}_FY{y}.pdf")
        if os.path.exists(out) and os.path.getsize(out) > 2_000_000:
            print(f"  [{y}] exists -> skip")
            manifest_rows.append([ticker, cn, y, url, "exists", os.path.getsize(out), out])
            got_any = True; continue
        ok, info = download_pdf(url, out)
        if ok:
            print(f"  [{y}] ok size={info:,}")
            manifest_rows.append([ticker, cn, y, url, "ok", info, out])
            got_any = True
        else:
            print(f"  [{y}] FAIL {info}")
            manifest_rows.append([ticker, cn, y, url, f"fail:{info}", 0, ""])
    if got_any:
        summary["ok"] += 1
drv.quit()

with open(os.path.join(RPT, "u10_v3_manifest.csv"), "w", encoding="utf-8", newline="") as cf:
    w = csv.writer(cf)
    w.writerow(["ticker","cn_name","year","url","status","size","path"])
    w.writerows(manifest_rows)

print("\n===== SUMMARY v3 =====")
for k, v in summary.items():
    print(f"  {k}: {v}")
