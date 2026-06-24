"""Aug-Plus hand-crafted seed builder (v1).

Single-source-of-truth for the hand-authored minority-class ESG rows that
seed the Aug-Plus pipeline.  Run from the repo root::

    python assets/aug_plus/build_handcrafted_v1.py

It validates every row through ``src.data.aug_schema`` and writes the
canonical CSV to ``assets/aug_plus/handcrafted_v1.csv``.  Both files are
tracked in git so the seed is reproducible without external tools.

Target distribution (based on Phase 36 bottleneck analysis):
  * T4 = Misleading            : 20  rows  (train: 1   -> +20  : ~21x lift)
  * T2 = within_2_years        : 15  rows  (train: 13  -> +15  : ~2.2x)
  * T4 = Not Clear             : 8   rows  (train: 124 -> +8   : balance)
  * T2 = longer_than_5_years   : 7   rows  (train: 197 -> +7   : balance)

All rows are realistic Traditional-Chinese ESG-disclosure prose; none are
copied verbatim from any external corpus.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

# Allow running from anywhere
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.aug_schema import (
    AP_HANDCRAFT_BASE,
    PSEUDO_CSV_COLUMNS,
    SynthRow,
    validate_rows,
)


# ---------------------------------------------------------------------------
# Seed catalogue.  Each entry is a dict with the four labels + data text.
# IDs are assigned sequentially from AP_HANDCRAFT_BASE in catalogue order.
# ---------------------------------------------------------------------------

# T4 = Misleading  (T1=Yes, T2!=N/A, T3=Yes, T4=Misleading)
# Pattern: a concrete promise paired with vague / irrelevant / tangential
# evidence -- enough to count as evidence_status=Yes but failing to support
# the promise.
MISLEADING: list[dict] = [
    {
        "data": "本公司承諾於2030年達成範疇一及範疇二溫室氣體淨零排放，集團已連續五年榮獲台灣企業永續獎金獎，並積極參與多項環保公益活動。",
        "promise_string": "承諾於2030年達成範疇一及範疇二溫室氣體淨零排放",
        "evidence_string": "集團已連續五年榮獲台灣企業永續獎金獎，並積極參與多項環保公益活動",
        "verification_timeline": "between_2_and_5_years",
    },
    {
        "data": "集團目標2028年女性管理職比例提升至40%，本公司一向重視性別平等，並設置多元共融委員會及員工協助方案。",
        "promise_string": "目標2028年女性管理職比例提升至40%",
        "evidence_string": "重視性別平等，並設置多元共融委員會及員工協助方案",
        "verification_timeline": "between_2_and_5_years",
    },
    {
        "data": "公司計畫於2027年底前全面使用再生能源，目前已於屏東廠導入太陽能板示範案，可滿足該廠約5%電力需求。",
        "promise_string": "計畫於2027年底前全面使用再生能源",
        "evidence_string": "於屏東廠導入太陽能板示範案，可滿足該廠約5%電力需求",
        "verification_timeline": "within_2_years",
    },
    {
        "data": "本公司宣示2035年達成範疇三供應鏈減碳50%，今年度已邀請主要供應商共同舉辦兩場永續說明會。",
        "promise_string": "宣示2035年達成範疇三供應鏈減碳50%",
        "evidence_string": "已邀請主要供應商共同舉辦兩場永續說明會",
        "verification_timeline": "longer_than_5_years",
    },
    {
        "data": "我們承諾2029年將產品包裝塑膠用量減少30%，目前正在進行包材替代材料的研發評估，並已成立跨部門專案小組。",
        "promise_string": "承諾2029年將產品包裝塑膠用量減少30%",
        "evidence_string": "正在進行包材替代材料的研發評估，並已成立跨部門專案小組",
        "verification_timeline": "between_2_and_5_years",
    },
    {
        "data": "公司預定於2027年導入100%電動物流車隊，目前車隊中有3台油電混合車輛作為試行，運行狀況良好。",
        "promise_string": "預定於2027年導入100%電動物流車隊",
        "evidence_string": "目前車隊中有3台油電混合車輛作為試行",
        "verification_timeline": "within_2_years",
    },
    {
        "data": "本公司目標2030年董事會多元化比例達50%，現任董事會成員具備豐富產業經驗及高度多元背景。",
        "promise_string": "目標2030年董事會多元化比例達50%",
        "evidence_string": "現任董事會成員具備豐富產業經驗及高度多元背景",
        "verification_timeline": "between_2_and_5_years",
    },
    {
        "data": "我們承諾2028年水資源回收率達到80%，本公司高度重視水資源永續，每年舉辦世界水資源日宣導活動。",
        "promise_string": "承諾2028年水資源回收率達到80%",
        "evidence_string": "每年舉辦世界水資源日宣導活動",
        "verification_timeline": "between_2_and_5_years",
    },
    {
        "data": "集團宣布2026年底完成所有營運據點的ISO 14001認證，目前已開始相關文件準備及員工教育訓練。",
        "promise_string": "宣布2026年底完成所有營運據點的ISO 14001認證",
        "evidence_string": "目前已開始相關文件準備及員工教育訓練",
        "verification_timeline": "within_2_years",
    },
    {
        "data": "本公司目標2032年生物多樣性正向影響面積達1,000公頃，公司長期支持地方造林活動，董事長親自參與多場植樹典禮。",
        "promise_string": "目標2032年生物多樣性正向影響面積達1,000公頃",
        "evidence_string": "公司長期支持地方造林活動，董事長親自參與多場植樹典禮",
        "verification_timeline": "longer_than_5_years",
    },
    {
        "data": "我們承諾於2031年達成全產品線取得綠色標章，產品研發團隊已參加多場綠色設計國際研討會。",
        "promise_string": "承諾於2031年達成全產品線取得綠色標章",
        "evidence_string": "產品研發團隊已參加多場綠色設計國際研討會",
        "verification_timeline": "longer_than_5_years",
    },
    {
        "data": "公司預期2028年員工教育訓練平均時數達50小時，今年度已盤點各單位現有訓練資源並建立內部講師資料庫。",
        "promise_string": "預期2028年員工教育訓練平均時數達50小時",
        "evidence_string": "已盤點各單位現有訓練資源並建立內部講師資料庫",
        "verification_timeline": "between_2_and_5_years",
    },
    {
        "data": "本公司承諾2027年取得SBTi科學基礎減量目標認證，今年已委請外部顧問評估初步減量路徑。",
        "promise_string": "承諾2027年取得SBTi科學基礎減量目標認證",
        "evidence_string": "今年已委請外部顧問評估初步減量路徑",
        "verification_timeline": "within_2_years",
    },
    {
        "data": "集團目標2029年食物里程縮短20%，本公司持續推動在地採購政策，並表揚優良在地供應商。",
        "promise_string": "目標2029年食物里程縮短20%",
        "evidence_string": "持續推動在地採購政策，並表揚優良在地供應商",
        "verification_timeline": "between_2_and_5_years",
    },
    {
        "data": "我們宣示2030年成為碳負排企業，公司去年購買大量自願性碳權，並出席多場國際氣候峰會。",
        "promise_string": "宣示2030年成為碳負排企業",
        "evidence_string": "公司去年購買大量自願性碳權，並出席多場國際氣候峰會",
        "verification_timeline": "between_2_and_5_years",
    },
    {
        "data": "本公司承諾2034年職場安全零事故，現行已實施每月安全宣導早會並懸掛安全標語於各廠區。",
        "promise_string": "承諾2034年職場安全零事故",
        "evidence_string": "現行已實施每月安全宣導早會並懸掛安全標語於各廠區",
        "verification_timeline": "longer_than_5_years",
    },
    {
        "data": "公司預定2028年水耗量較基準年減少25%，現有節水措施包含廁所感應式水龍頭及員工節水提案活動。",
        "promise_string": "預定2028年水耗量較基準年減少25%",
        "evidence_string": "現有節水措施包含廁所感應式水龍頭及員工節水提案活動",
        "verification_timeline": "between_2_and_5_years",
    },
    {
        "data": "我們承諾2027年底所有自有車輛完成電動化，公司本年度購置一輛特斯拉Model Y作為高階主管接駁示範車。",
        "promise_string": "承諾2027年底所有自有車輛完成電動化",
        "evidence_string": "本年度購置一輛特斯拉Model Y作為高階主管接駁示範車",
        "verification_timeline": "within_2_years",
    },
    {
        "data": "本公司承諾2030年新進員工女性比例達45%，公司人資部門已重新設計招募網頁，採用更多元化的視覺呈現。",
        "promise_string": "承諾2030年新進員工女性比例達45%",
        "evidence_string": "公司人資部門已重新設計招募網頁，採用更多元化的視覺呈現",
        "verification_timeline": "between_2_and_5_years",
    },
    {
        "data": "集團目標於2028年完成全廠區廢棄物零掩埋，本公司已與當地清潔隊建立良好溝通管道並簽訂合作備忘錄。",
        "promise_string": "目標於2028年完成全廠區廢棄物零掩埋",
        "evidence_string": "已與當地清潔隊建立良好溝通管道並簽訂合作備忘錄",
        "verification_timeline": "between_2_and_5_years",
    },
]

# T2 = within_2_years  (T1=Yes, T2=within_2_years, T3/T4 varied)
# Pattern: clear short-term promise with concrete timeline of ≤2 yrs.
WITHIN_2Y: list[dict] = [
    {
        "data": "本公司承諾於2026年第四季前完成總部大樓LED燈具汰換工程，預期可降低照明電力消耗約35%。",
        "promise_string": "承諾於2026年第四季前完成總部大樓LED燈具汰換工程",
        "evidence_string": "",
        "evidence_status": "No",
        "evidence_quality": "N/A",
    },
    {
        "data": "集團目標於2027年中前在全台50個營業據點導入無紙化簽核系統，目前已於5個據點完成試行並回收良好反饋。",
        "promise_string": "目標於2027年中前在全台50個營業據點導入無紙化簽核系統",
        "evidence_string": "已於5個據點完成試行並回收良好反饋",
        "evidence_status": "Yes",
        "evidence_quality": "Clear",
    },
    {
        "data": "我們將於明年底前完成董事會獨立董事比例提升至三分之一的目標，已於上次股東會通過相關章程修訂案。",
        "promise_string": "明年底前完成董事會獨立董事比例提升至三分之一的目標",
        "evidence_string": "已於上次股東會通過相關章程修訂案",
        "evidence_status": "Yes",
        "evidence_quality": "Clear",
    },
    {
        "data": "公司預定於2026年12月前導入新版供應商行為準則並要求前50大供應商簽署。",
        "promise_string": "預定於2026年12月前導入新版供應商行為準則並要求前50大供應商簽署",
        "evidence_string": "",
        "evidence_status": "No",
        "evidence_quality": "N/A",
    },
    {
        "data": "本公司承諾未來18個月內將員工年度健康檢查項目擴充至涵蓋癌症早期篩檢，現已完成方案規劃並編列預算。",
        "promise_string": "承諾未來18個月內將員工年度健康檢查項目擴充至涵蓋癌症早期篩檢",
        "evidence_string": "現已完成方案規劃並編列預算",
        "evidence_status": "Yes",
        "evidence_quality": "Clear",
    },
    {
        "data": "集團計畫於2027年初前完成新北物流中心屋頂太陽能板建置，預估年發電量可達120萬度。",
        "promise_string": "計畫於2027年初前完成新北物流中心屋頂太陽能板建置",
        "evidence_string": "",
        "evidence_status": "No",
        "evidence_quality": "N/A",
    },
    {
        "data": "我們承諾於2026年底發布首份依TNFD框架編製的自然相關財務揭露報告，目前正進行重大議題鑑別。",
        "promise_string": "承諾於2026年底發布首份依TNFD框架編製的自然相關財務揭露報告",
        "evidence_string": "目前正進行重大議題鑑別",
        "evidence_status": "Yes",
        "evidence_quality": "Not Clear",
    },
    {
        "data": "本公司預定2027年第一季前完成所有作業流程的個人資料風險評估並導入隱私強化技術。",
        "promise_string": "預定2027年第一季前完成所有作業流程的個人資料風險評估並導入隱私強化技術",
        "evidence_string": "",
        "evidence_status": "No",
        "evidence_quality": "N/A",
    },
    {
        "data": "公司宣布於2026年底前將全體員工最低薪資調整至月薪35,000元以上，相關預算已納入下年度營運計畫。",
        "promise_string": "宣布於2026年底前將全體員工最低薪資調整至月薪35,000元以上",
        "evidence_string": "相關預算已納入下年度營運計畫",
        "evidence_status": "Yes",
        "evidence_quality": "Clear",
    },
    {
        "data": "我們將於明年第三季前完成全集團ISO 27001資訊安全管理系統認證，內部稽核已展開第二輪。",
        "promise_string": "明年第三季前完成全集團ISO 27001資訊安全管理系統認證",
        "evidence_string": "內部稽核已展開第二輪",
        "evidence_status": "Yes",
        "evidence_quality": "Clear",
    },
    {
        "data": "集團承諾2027年農曆年前完成總部空調系統全面汰換為高效率變頻機組，工程招標已於本月公告。",
        "promise_string": "承諾2027年農曆年前完成總部空調系統全面汰換為高效率變頻機組",
        "evidence_string": "工程招標已於本月公告",
        "evidence_status": "Yes",
        "evidence_quality": "Clear",
    },
    {
        "data": "本公司預期2026年底前將公司治理評鑑成績提升至前5%等級，已聘請外部專業顧問協助治理結構強化。",
        "promise_string": "預期2026年底前將公司治理評鑑成績提升至前5%等級",
        "evidence_string": "已聘請外部專業顧問協助治理結構強化",
        "evidence_status": "Yes",
        "evidence_quality": "Not Clear",
    },
    {
        "data": "公司預定於未來兩年內取得B型企業（B Corp）認證。",
        "promise_string": "預定於未來兩年內取得B型企業（B Corp）認證",
        "evidence_string": "",
        "evidence_status": "No",
        "evidence_quality": "N/A",
    },
    {
        "data": "我們承諾於2027年中前讓所有外籍移工居住空間達到Fair Wear Foundation標準，已啟動現況盤點。",
        "promise_string": "承諾於2027年中前讓所有外籍移工居住空間達到Fair Wear Foundation標準",
        "evidence_string": "已啟動現況盤點",
        "evidence_status": "Yes",
        "evidence_quality": "Not Clear",
    },
    {
        "data": "集團目標2026年底前董事會通過並對外公布人權政策正式版本，相關草案已提交審計委員會審議。",
        "promise_string": "目標2026年底前董事會通過並對外公布人權政策正式版本",
        "evidence_string": "相關草案已提交審計委員會審議",
        "evidence_status": "Yes",
        "evidence_quality": "Clear",
    },
]

# T4 = Not Clear  (T1=Yes, T2!=N/A, T3=Yes, T4=Not Clear)
NOT_CLEAR: list[dict] = [
    {
        "data": "本公司承諾2030年範疇一二排放較2020基準年減少40%，去年範疇一二總排放量為12萬公噸CO2e，較前年下降約3%。",
        "promise_string": "承諾2030年範疇一二排放較2020基準年減少40%",
        "evidence_string": "去年範疇一二總排放量為12萬公噸CO2e，較前年下降約3%",
        "verification_timeline": "between_2_and_5_years",
    },
    {
        "data": "我們目標2028年女性主管比例達35%，現任課級以上女性主管比例為28.4%，較三年前提升4個百分點。",
        "promise_string": "目標2028年女性主管比例達35%",
        "evidence_string": "現任課級以上女性主管比例為28.4%，較三年前提升4個百分點",
        "verification_timeline": "between_2_and_5_years",
    },
    {
        "data": "集團承諾2027年所有產品包材100%可回收，目前主要產品線回收材料使用比例約為65%。",
        "promise_string": "承諾2027年所有產品包材100%可回收",
        "evidence_string": "目前主要產品線回收材料使用比例約為65%",
        "verification_timeline": "within_2_years",
    },
    {
        "data": "公司預定2031年廢水回收率提升至70%，去年廢水回收率為52%，預計明年完成第二期回收系統建置。",
        "promise_string": "預定2031年廢水回收率提升至70%",
        "evidence_string": "去年廢水回收率為52%，預計明年完成第二期回收系統建置",
        "verification_timeline": "longer_than_5_years",
    },
    {
        "data": "我們承諾2029年員工敬業度調查分數達80分，最近一次調查分數為72分，較前次上升3分。",
        "promise_string": "承諾2029年員工敬業度調查分數達80分",
        "evidence_string": "最近一次調查分數為72分，較前次上升3分",
        "verification_timeline": "between_2_and_5_years",
    },
    {
        "data": "本公司目標2027年產品單位能耗較基準年下降20%，今年度單位能耗已下降約8%。",
        "promise_string": "目標2027年產品單位能耗較基準年下降20%",
        "evidence_string": "今年度單位能耗已下降約8%",
        "verification_timeline": "within_2_years",
    },
    {
        "data": "集團計畫2030年供應商ESG稽核覆蓋率達90%，目前前100大供應商完成稽核者為58家。",
        "promise_string": "計畫2030年供應商ESG稽核覆蓋率達90%",
        "evidence_string": "目前前100大供應商完成稽核者為58家",
        "verification_timeline": "between_2_and_5_years",
    },
    {
        "data": "我們承諾2028年職災千人率降至0.5以下，去年職災千人率為0.9，較前年的1.2有所改善。",
        "promise_string": "承諾2028年職災千人率降至0.5以下",
        "evidence_string": "去年職災千人率為0.9，較前年的1.2有所改善",
        "verification_timeline": "between_2_and_5_years",
    },
]

# T2 = longer_than_5_years  (T1=Yes, T2=longer_than_5_years, T3/T4 varied)
LONGER_5Y: list[dict] = [
    {
        "data": "本公司承諾於2050年達成全價值鏈淨零排放，並通過SBTi長期目標審查。",
        "promise_string": "承諾於2050年達成全價值鏈淨零排放",
        "evidence_string": "通過SBTi長期目標審查",
        "evidence_status": "Yes",
        "evidence_quality": "Clear",
    },
    {
        "data": "集團設定2040年成為100%循環經濟企業之長期願景。",
        "promise_string": "設定2040年成為100%循環經濟企業之長期願景",
        "evidence_string": "",
        "evidence_status": "No",
        "evidence_quality": "N/A",
    },
    {
        "data": "我們宣示2045年達成水資源正影響，目前正進行流域水風險評估。",
        "promise_string": "宣示2045年達成水資源正影響",
        "evidence_string": "目前正進行流域水風險評估",
        "evidence_status": "Yes",
        "evidence_quality": "Not Clear",
    },
    {
        "data": "公司預定2050年董事會半數成員具備氣候變遷專業背景。",
        "promise_string": "預定2050年董事會半數成員具備氣候變遷專業背景",
        "evidence_string": "",
        "evidence_status": "No",
        "evidence_quality": "N/A",
    },
    {
        "data": "本公司承諾於2040年將全廠區可再生能源使用比例提升至100%，並已採購20年期長期再生能源購電合約。",
        "promise_string": "承諾於2040年將全廠區可再生能源使用比例提升至100%",
        "evidence_string": "已採購20年期長期再生能源購電合約",
        "evidence_status": "Yes",
        "evidence_quality": "Clear",
    },
    {
        "data": "集團目標2055年成為碳負排企業領導者，目前正委託學術單位研擬轉型藍圖。",
        "promise_string": "目標2055年成為碳負排企業領導者",
        "evidence_string": "目前正委託學術單位研擬轉型藍圖",
        "evidence_status": "Yes",
        "evidence_quality": "Not Clear",
    },
    {
        "data": "我們承諾於2050年完全停用化石燃料，公司正在制定為期25年的逐步淘汰時程表。",
        "promise_string": "承諾於2050年完全停用化石燃料",
        "evidence_string": "公司正在制定為期25年的逐步淘汰時程表",
        "evidence_status": "Yes",
        "evidence_quality": "Not Clear",
    },
]


def build_seed_rows() -> list[SynthRow]:
    rows: list[SynthRow] = []
    next_id = AP_HANDCRAFT_BASE

    def emit(spec: dict, *, t4_default: str, t2_default: str | None = None,
             t3_default: str = "Yes", company_source: str = "ap_handcraft_v1") -> SynthRow:
        nonlocal next_id
        row = SynthRow(
            id=next_id,
            data=spec["data"],
            promise_status="Yes",
            verification_timeline=spec.get("verification_timeline", t2_default or ""),
            evidence_status=spec.get("evidence_status", t3_default),
            evidence_quality=spec.get("evidence_quality", t4_default),
            promise_string=spec.get("promise_string", ""),
            evidence_string=spec.get("evidence_string", ""),
            company_source=company_source,
            confidence_min=1.0,
            conf_T1=1.0,
            conf_T2=1.0,
            conf_T3=1.0,
            conf_T4=1.0,
        )
        next_id += 1
        return row

    for spec in MISLEADING:
        rows.append(emit(spec, t4_default="Misleading"))
    for spec in WITHIN_2Y:
        rows.append(emit(spec, t4_default="Clear", t2_default="within_2_years"))
    for spec in NOT_CLEAR:
        rows.append(emit(spec, t4_default="Not Clear"))
    for spec in LONGER_5Y:
        rows.append(emit(spec, t4_default="Clear", t2_default="longer_than_5_years"))
    return rows


def write_csv(rows: list[SynthRow], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(PSEUDO_CSV_COLUMNS), quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        for r in rows:
            writer.writerow(r.to_csv_row())


def main() -> int:
    rows = build_seed_rows()
    n_ok, errors = validate_rows(rows)
    if errors:
        print(f"[FAIL] {len(errors)} rows failed validation:", file=sys.stderr)
        for rid, msg in errors[:10]:
            print(f"  id={rid}: {msg}", file=sys.stderr)
        return 1

    out = ROOT / "assets" / "aug_plus" / "handcrafted_v1.csv"
    write_csv(rows, out)

    # Quick stats
    from collections import Counter
    t4 = Counter(r.evidence_quality for r in rows)
    t2 = Counter(r.verification_timeline for r in rows)
    print(f"[OK] wrote {n_ok} rows -> {out}")
    print(f"     T4 distribution: {dict(t4)}")
    print(f"     T2 distribution: {dict(t2)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
