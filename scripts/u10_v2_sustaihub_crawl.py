"""U10-NEW-v2: SR PDF crawler via sustaihub.com -> TWSE ESG GenPlus API.

Pipeline per ticker:
  1. Open sustaihub `/reports/?keyword={cn_name}` (Selenium, 4s).
  2. Parse cards: keep links whose path-segment (公司中文全名) == target cn_name.
  3. For each matching card, open detail page (Selenium, 5s).
  4. Extract <a href="esggenplus.twse.com.tw/api/.../FileStream?id=UUID">
     restricted to data-ga-event in {detail_download_click, detail_history_download_click}
     (skip detail_industry_download_click).
  5. From data-report-info "{name} - {YYYY}" parse year; keep YYYY in {2023, 2024}.
  6. Download PDF via requests (stream + %PDF magic + size>=2MB).
  7. Save: data/raw/u10/{ticker}_FY{year}_sustai.pdf

Notes:
  - sustaihub search is fuzzy; we must filter by EXACT company name.
  - One detail page actually lists historical reports of the SAME company
    (detail_history_download_click) so a single visit yields multi-year PDFs.
"""
import sys, io, os, re, time, csv, hashlib, urllib.parse
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
import requests, urllib3
urllib3.disable_warnings()
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW  = os.path.join(ROOT, "data", "raw", "u10")
RPT  = os.path.join(ROOT, "reports", "experiments", "u10")
os.makedirs(RAW, exist_ok=True); os.makedirs(RPT, exist_ok=True)

UA = {"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
      "Accept-Language":"zh-TW,zh;q=0.9"}

# 24 tickers needing supplemental download. Use FULL company name (must match sustaihub card text exactly).
TARGETS = {
    "1102": "亞洲水泥股份有限公司",
    "1605": "華新麗華股份有限公司",
    "1722": "台灣肥料股份有限公司",
    "2353": "宏碁股份有限公司",
    "2354": "鴻準精密工業股份有限公司",
    "2356": "英業達股份有限公司",
    "2376": "技嘉科技股份有限公司",
    "2408": "南亞科技股份有限公司",
    "2474": "可成科技股份有限公司",
    "2610": "中華航空股份有限公司",
    "2618": "長榮航空股份有限公司",
    "2727": "王品餐飲股份有限公司",
    "2812": "台中商業銀行股份有限公司",
    "2823": "中國人壽保險股份有限公司",
    "2867": "三商美邦人壽保險股份有限公司",
    "3037": "欣興電子股份有限公司",
    "3443": "創意電子股份有限公司",
    "4958": "臻鼎科技控股股份有限公司",
    "5269": "祥碩科技股份有限公司",
    "6239": "力成科技股份有限公司",
    "6415": "矽力杰半導體技術(杭州)有限公司",  # 6415 矽力-KY corporate name varies, fallback below
    "8046": "南亞電路板股份有限公司",
    "9904": "寶成工業股份有限公司",
    "9921": "巨大機械工業股份有限公司",
}
WANTED_YEARS = {2023, 2024}

def make_driver():
    opts = Options()
    opts.add_argument("--headless=new")
    opts.add_argument("--window-size=1400,1000")
    opts.add_argument("--disable-blink-features=AutomationControlled")
    opts.add_argument(f"user-agent={UA['User-Agent']}")
    drv = webdriver.Chrome(options=opts)
    drv.set_page_load_timeout(60)
    return drv

DRV = None
def page(url, wait=4):
    global DRV
    if DRV is None: DRV = make_driver()
    DRV.get(url); time.sleep(wait)
    return DRV.page_source

# Card pattern on listing
RE_CARD_HREF = re.compile(r'href="(/reports/([^/]+)/([a-f0-9-]{36})/)"')
# Detail-page own/historical download buttons (skip industry recommendations)
RE_DL = re.compile(
    r'<a\s+href="(https://esggenplus\.twse\.com\.tw/api/api/MopsSustainReport/data/FileStream\?id=[a-f0-9-]+)"'
    r'[^>]*data-ga-event="(detail_download_click|detail_history_download_click)"'
    r'[^>]*data-report-info="([^"]+)"',
    re.I)

def find_uuid_for_company(cn_name):
    """Return list of (uuid, name_path) cards whose decoded path == cn_name."""
    html = page(f"https://www.sustaihub.com/reports/?keyword={urllib.parse.quote(cn_name)}", wait=4)
    out, seen = [], set()
    for path, name_enc, uid in RE_CARD_HREF.findall(html):
        name = urllib.parse.unquote(name_enc)
        if name == cn_name and uid not in seen:
            seen.add(uid); out.append(uid)
    return out

def collect_pdfs_from_detail(uuid, cn_name):
    """Return list of (api_url, year:int) for cn_name in WANTED_YEARS."""
    url = f"https://www.sustaihub.com/reports/{urllib.parse.quote(cn_name)}/{uuid}/"
    try:
        html = page(url, wait=5)
    except Exception as e:
        print(f"   ! detail load failed: {e}")
        return []
    out = []
    seen = set()
    for api_url, _evt, info in RE_DL.findall(html):
        # info = "{cn_name} - {year}"
        m = re.search(r"-\s*(\d{4})", info)
        if not m: continue
        year = int(m.group(1))
        if year not in WANTED_YEARS: continue
        if cn_name not in info: continue
        key = (year, api_url)
        if key in seen: continue
        seen.add(key)
        out.append((api_url, year))
    return out

def download_pdf(api_url, ticker, year):
    out_fp = os.path.join(RAW, f"{ticker}_FY{year}_sustai.pdf")
    if os.path.exists(out_fp) and os.path.getsize(out_fp) > 1_000_000:
        return out_fp, "exists", os.path.getsize(out_fp)
    try:
        r = requests.get(api_url, headers=UA, timeout=120, stream=True, verify=False)
        if r.status_code != 200:
            return None, f"http_{r.status_code}", 0
        if "pdf" not in (r.headers.get("content-type","").lower()):
            return None, "not_pdf_ct", 0
        first = r.raw.read(4)
        if first[:4] != b"%PDF":
            return None, "no_magic", 0
        size = len(first)
        with open(out_fp, "wb") as f:
            f.write(first)
            for chunk in r.iter_content(chunk_size=131072):
                if chunk:
                    f.write(chunk); size += len(chunk)
        if size < 2_000_000:
            os.remove(out_fp); return None, f"too_small_{size}", size
        return out_fp, "ok", size
    except Exception as e:
        return None, f"err:{type(e).__name__}", 0

def main():
    log_fp = open(os.path.join(RPT, "u10_v2_collect.log"), "w", encoding="utf-8")
    csv_fp = open(os.path.join(RPT, "u10_v2_manifest.csv"), "w", encoding="utf-8", newline="")
    wr = csv.writer(csv_fp); wr.writerow(["ticker","cn_name","year","api_url","status","size","local_path"])

    def L(msg):
        print(msg); log_fp.write(msg+"\n"); log_fp.flush()

    summary = {"ok":0, "fail_no_card":0, "fail_no_pdf":0, "skipped":0}
    for ticker, cn in TARGETS.items():
        L(f"\n=== {ticker} {cn} ===")
        try:
            uuids = find_uuid_for_company(cn)
        except Exception as e:
            L(f"  search failed: {e}")
            summary["fail_no_card"] += 1
            continue
        if not uuids:
            L(f"  ✗ no exact-name match cards")
            wr.writerow([ticker, cn, "", "", "no_card", 0, ""])
            summary["fail_no_card"] += 1
            continue
        L(f"  found {len(uuids)} card(s) matching exact name")
        # Visit FIRST card only (detail page already lists all historical years for same company)
        pdfs = collect_pdfs_from_detail(uuids[0], cn)
        if not pdfs:
            L(f"  ✗ detail page exposes no PDF in years {WANTED_YEARS}")
            wr.writerow([ticker, cn, "", "", "no_pdf_in_year", 0, ""])
            summary["fail_no_pdf"] += 1
            continue
        # If no FY2023 in first card, try other cards
        years_have = {y for _, y in pdfs}
        if WANTED_YEARS - years_have and len(uuids) > 1:
            for uid in uuids[1:]:
                more = collect_pdfs_from_detail(uid, cn)
                for u, y in more:
                    if y not in years_have:
                        pdfs.append((u, y)); years_have.add(y)
                if not (WANTED_YEARS - years_have): break
        L(f"  candidates: {sorted({y for _,y in pdfs})}")
        any_ok = False
        for api_url, year in sorted(pdfs, key=lambda x: -x[1]):
            local, status, size = download_pdf(api_url, ticker, year)
            L(f"    [{year}] {status}  size={size:,}  -> {local}")
            wr.writerow([ticker, cn, year, api_url, status, size, local or ""])
            if status in ("ok", "exists"): any_ok = True
        if any_ok: summary["ok"] += 1
        else: summary["fail_no_pdf"] += 1

    L("\n===== SUMMARY =====")
    for k,v in summary.items(): L(f"  {k}: {v}")
    log_fp.close(); csv_fp.close()
    if DRV is not None: DRV.quit()

if __name__ == "__main__":
    main()
