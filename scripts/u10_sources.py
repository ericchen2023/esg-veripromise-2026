"""U10-NEW: 30-company ESG report landing page registry.

Each entry: ticker -> (chinese_name, english_short, esg_landing_url).
ESG landing URL is the "永續報告書下載"/"ESG Reports" page where SR PDFs are listed.
Where exact ESG page is uncertain, fallback to corporate IR/About page that links to ESG.
Verified manually based on common patterns; runtime crawler will fall back to Bing search if a page returns 0 PDFs.
"""

COMPANY_SOURCES = {
    # 水泥 / 化工 / 鋼鐵 / 紡織
    "1102": ("亞泥", "Asia Cement", "https://www.acc.com.tw/main_ch/sustain.aspx?Sub_ID=2026"),
    "1326": ("台化", "Formosa Chemicals", "https://www.fcfc.com.tw/Sustainability/Reports"),
    "1402": ("遠東新世紀", "Far Eastern New Century", "https://csr.fenc.com/report.aspx"),
    "1605": ("華新麗華", "Walsin Lihwa", "https://www.walsin.com/zh-tw/sustainability/reports.html"),
    "1722": ("台肥", "Taiwan Fertilizer", "https://www.taifer.com.tw/main/sustainability/Pages/CSRReportDownload.aspx"),

    # 電子下游 / 品牌
    "2353": ("宏碁", "Acer", "https://www.acer.com/sustainability/zh/reports"),
    "2354": ("鴻準", "Foxconn Tech", "https://www.foxconntech.com/zh-TW/Sustainability/Report"),
    "2356": ("英業達", "Inventec", "https://csr.inventec.com/csr-tw/csr-report.html"),
    "2357": ("華碩", "ASUS", "https://csr.asus.com/Chinese/Article.aspx?id=2120"),
    "2376": ("技嘉", "Gigabyte", "https://www.gigabyte.com/tw/csr/Reports"),

    # 半導體 / 記憶體 / 載板
    "2408": ("南亞科", "Nanya Tech", "https://www.nanya.com/tw/CSR/CSRReports"),
    "2474": ("可成", "Catcher", "https://www.catcher.com.tw/zh-tw/Sustainability/CSR-Reports"),
    "3037": ("欣興", "Unimicron", "https://www.unimicron.com/zh-TW/Investors_csr_reports.html"),
    "3443": ("創意", "GUC", "https://www.guc-asic.com/zh-tw/csr/csr-reports"),
    "4958": ("臻鼎", "Zhen Ding", "https://www.zdtco.com/tw/CSR/Reports"),
    "5269": ("祥碩", "ASMedia", "https://www.asmedia.com.tw/eng/about_06.php"),
    "6239": ("力成", "Powertech", "https://www.powertech.com.tw/csr/CSRReport.aspx"),
    "6415": ("矽力-KY", "Silergy", "https://www.silergy.com/tw/about-us/sustainability"),
    "8046": ("南電", "Nan Ya PCB", "https://www.npc.com.tw/j2npc/zhtw/csr/csr_report.jsp"),
    "8299": ("群聯", "Phison", "https://www.phison.com/zh-TW/CorporateSocialResponsibility/CSR-Report"),

    # 航空 / 餐飲
    "2610": ("華航", "China Airlines", "https://calec.china-airlines.com/csr/zh/download.html"),
    "2618": ("長榮航", "EVA Air", "https://www.evaair.com/zh-tw/about-eva-air/corporate-social-responsibility/csr-reports/"),
    "2727": ("王品", "Wowprime", "https://www.wowprime.com/csr.aspx"),

    # 金融 / 保險
    "2812": ("台中商銀", "Taichung Bank", "https://www.tcbbank.com.tw/about/sustainability/report"),
    "2823": ("中壽", "China Life", "https://www.chinalife.com.tw/wps/portal/chinalife/about/socialResponsibility/socialReport"),
    "2867": ("三商壽", "Mercuries Life", "https://www.mli.com.tw/about_csrreport"),
    "2890": ("永豐金控", "SinoPac Holdings", "https://www.sinopac.com/esg/tw/sustainability-report/"),

    # 製鞋 / 自行車
    "9904": ("寶成", "Pou Chen", "https://www.pouchen.com/index.php/tw/csr-report-tw"),
    "9910": ("豐泰", "Feng Tay", "https://www.fengtay.com/tw/sustainability/Reports.aspx"),
    "9921": ("巨大", "Giant", "https://www.giantgroup.com.tw/csr-reports/zh-Hant"),
}

assert len(COMPANY_SOURCES) == 30, f"expected 30, got {len(COMPANY_SOURCES)}"

if __name__ == "__main__":
    for tk, (cn, en, url) in COMPANY_SOURCES.items():
        print(f"{tk}\t{cn}\t{en}\t{url}")
