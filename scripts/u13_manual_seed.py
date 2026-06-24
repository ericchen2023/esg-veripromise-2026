"""Phase 37 — manual annotation seed template generator.

Produces an empty CSV with the same schema as the pseudo-label pipeline,
pre-filled with rows targeting the minority classes that need human-crafted
examples. Author fills in the ``data`` column (and optionally tweaks labels);
the file then promotes directly via train_pseudo_kfold.

Also ships 20 hand-crafted seed examples (Chinese ESG) to bootstrap the file
so the author has a worked reference style.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.data.synth_schema import (
    PHASE37_TARGET_COUNTS,
    PSEUDO_CSV_COLUMNS,
    SynthRow,
    iter_target_specs,
    stable_synth_id,
)


# 20 hand-crafted seed examples covering the highest-value buckets.
# These are author-written, fully synthetic, and intentionally diverse in topic
# (E / S / G), tone (formal / progress-update / aspirational), and company type.
SEED_EXAMPLES: list[dict] = [
    # T2=within_2_years, T4=Clear
    {"data": "本公司承諾於 2026 年底前將範疇一與範疇二溫室氣體排放總量較 2020 年基準降低 30%。截至 2024 年第三季，已實際減量 23.4%，並經 SGS 依 ISO 14064-1:2018 完成第三方確信。",
     "labels": {"promise_status": "Yes", "verification_timeline": "within_2_years", "evidence_status": "Yes", "evidence_quality": "Clear"}},
    {"data": "為達 RE100 階段性目標，集團規劃於 2026 年前完成台灣廠區綠電採購比例達 60%。2024 年實際達成 47%，新增 87 MW 太陽能 PPA，數據揭露於 CDP Climate 問卷 C8.2a。",
     "labels": {"promise_status": "Yes", "verification_timeline": "within_2_years", "evidence_status": "Yes", "evidence_quality": "Clear"}},
    {"data": "公司資安治理委員會承諾於 2025 年底前取得 ISO 27001:2022 全廠區認證。目前 6 座主要廠區已完成驗證，剩餘 2 座預計 2025 Q2 通過稽核，認證範圍與時程表已揭露於 2023 永續報告書第 142 頁。",
     "labels": {"promise_status": "Yes", "verification_timeline": "within_2_years", "evidence_status": "Yes", "evidence_quality": "Clear"}},
    # T2=within_2_years, T4=Misleading
    {"data": "本集團承諾於 2026 年成為亞洲最環保的科技領導者，相關碳中和進度全面領先業界。為展現決心，本年度已啟動數十項綠色專案。",
     "labels": {"promise_status": "Yes", "verification_timeline": "within_2_years", "evidence_status": "Yes", "evidence_quality": "Misleading"}},
    {"data": "我們承諾於兩年內完成淨零排放，成為全球永續典範。透過卓越管理與創新技術，相關指標已遠超國際同業平均水準。",
     "labels": {"promise_status": "Yes", "verification_timeline": "within_2_years", "evidence_status": "Yes", "evidence_quality": "Misleading"}},
    # T2=longer_than_5_years, T4=Misleading
    {"data": "本公司鄭重承諾於 2050 年達成淨零碳排，致力打造世界級永續企業。藉由集團強大的執行力與廣大員工的支持，我們對達成此一宏遠目標充滿信心。",
     "labels": {"promise_status": "Yes", "verification_timeline": "longer_than_5_years", "evidence_status": "Yes", "evidence_quality": "Misleading"}},
    {"data": "邁向 2040 年水資源正效益願景，本集團將持續引領產業變革。本年度已舉辦多場節水論壇並獲得業界高度肯定。",
     "labels": {"promise_status": "Yes", "verification_timeline": "longer_than_5_years", "evidence_status": "Yes", "evidence_quality": "Misleading"}},
    # T2=longer_than_5_years, T4=Clear (already covered)
    {"data": "本公司科學基礎減量目標 (SBTi) 已通過審核：承諾 2035 年範疇一+二排放絕對量較 2021 年基準減少 50.4%，範疇三排放強度減少 30%。2024 年實際進度為範疇一+二 -18.7%、範疇三強度 -9.2%，揭露於 CDP 報告 C4.2a。",
     "labels": {"promise_status": "Yes", "verification_timeline": "longer_than_5_years", "evidence_status": "Yes", "evidence_quality": "Clear"}},
    # T2=already, T4=Misleading
    {"data": "本公司已成功完成歷史性的綠色轉型，相關 ESG 指標全面領先業界。本年度榮獲多項國內外永續大獎，足證集團永續實力之卓越。",
     "labels": {"promise_status": "Yes", "verification_timeline": "already", "evidence_status": "Yes", "evidence_quality": "Misleading"}},
    {"data": "去年度本集團已實現碳中和承諾，成為產業永續標竿。透過全體同仁努力與創新技術應用，我們持續在 ESG 領域樹立新典範。",
     "labels": {"promise_status": "Yes", "verification_timeline": "already", "evidence_status": "Yes", "evidence_quality": "Misleading"}},
    # T2=between_2_and_5_years, T4=Misleading
    {"data": "本公司預計於 2028 年完成全廠區綠能轉型，相關規劃符合國際最高標準。集團將以業界領導者之姿，引領產業邁向永續未來。",
     "labels": {"promise_status": "Yes", "verification_timeline": "between_2_and_5_years", "evidence_status": "Yes", "evidence_quality": "Misleading"}},
    # T2=already, T4=Not Clear
    {"data": "去年度本公司已完成 ISO 14001 環境管理系統的全面導入。相關認證範圍涵蓋多個營運據點，並持續推動環境績效改善計畫。",
     "labels": {"promise_status": "Yes", "verification_timeline": "already", "evidence_status": "Yes", "evidence_quality": "Not Clear"}},
    {"data": "供應鏈管理面向，本集團已建立完整的供應商行為準則，並要求所有一階供應商簽署承諾書。相關稽核工作已陸續展開。",
     "labels": {"promise_status": "Yes", "verification_timeline": "already", "evidence_status": "Yes", "evidence_quality": "Not Clear"}},
    # T2=within_2_years, T4=Not Clear
    {"data": "本公司預計於 2026 年前推動全面性的多元共融計畫。內容包括但不限於性別平等、世代共融、跨文化包容等多重面向，相關工作小組已成立並啟動規劃。",
     "labels": {"promise_status": "Yes", "verification_timeline": "within_2_years", "evidence_status": "Yes", "evidence_quality": "Not Clear"}},
    {"data": "為強化資訊安全防護，集團規劃於 2025 年完成資安人才培訓計畫升級。具體訓練時數與覆蓋率將於後續報告書中持續揭露。",
     "labels": {"promise_status": "Yes", "verification_timeline": "within_2_years", "evidence_status": "Yes", "evidence_quality": "Not Clear"}},
    # T2=longer_than_5_years, T4=Not Clear
    {"data": "本公司長期致力於循環經濟轉型，預計於 2032 年達成主要產品 70% 回收料佔比的階段性目標。相關技術研發與供應鏈合作正持續推進中。",
     "labels": {"promise_status": "Yes", "verification_timeline": "longer_than_5_years", "evidence_status": "Yes", "evidence_quality": "Not Clear"}},
    # T1=No anchors
    {"data": "本章節說明本公司的組織架構與營運範疇。集團總部設於台北，並於亞太地區設有多處分支機構，員工總數約 12,000 人。",
     "labels": {"promise_status": "No", "verification_timeline": "N/A", "evidence_status": "N/A", "evidence_quality": "N/A"}},
    {"data": "本永續報告書係依循 GRI Standards 2021 核心選項編製，並參考 SASB 與 TCFD 框架揭露相關資訊。報告期間為 2024 年 1 月 1 日至 12 月 31 日。",
     "labels": {"promise_status": "No", "verification_timeline": "N/A", "evidence_status": "N/A", "evidence_quality": "N/A"}},
    {"data": "本年度董事會共召開 8 次會議，董事平均出席率為 94.6%。獨立董事 3 名，佔董事會席次 33.3%，符合主管機關相關規範。",
     "labels": {"promise_status": "No", "verification_timeline": "N/A", "evidence_status": "N/A", "evidence_quality": "N/A"}},
    {"data": "本公司主要產品線包括半導體封裝測試、印刷電路板組裝、系統整合服務三大事業群，2024 年合併營收達新台幣 854 億元。",
     "labels": {"promise_status": "No", "verification_timeline": "N/A", "evidence_status": "N/A", "evidence_quality": "N/A"}},
]


def cmd_emit_template(args: argparse.Namespace) -> None:
    """Generate empty CSV template with target rows + 20 seed examples filled."""
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []

    # Seed rows (with actual text + labels)
    for ex in SEED_EXAMPLES:
        sr = SynthRow(text=ex["data"], labels=ex["labels"], source="u13_manual_seed", confidence=1.0)
        rows.append(sr.to_csv_dict())

    # Template rows (empty `data`, labels pre-filled, awaiting author fill-in)
    n_seed = len(rows)
    placeholder_counter = 0
    for spec in iter_target_specs():
        n_target = spec["n"]
        # Subtract seed count where possible to avoid over-target
        n_in_seed = sum(
            1 for ex in SEED_EXAMPLES
            if ex["labels"]["verification_timeline"] == spec["labels"]["verification_timeline"]
            and ex["labels"]["evidence_quality"] == spec["labels"]["evidence_quality"]
        )
        n_remaining = max(0, n_target - n_in_seed)
        for _ in range(n_remaining):
            placeholder_counter += 1
            placeholder_text = f"<TODO_FILL_{placeholder_counter:03d}_BUCKET_{spec['bucket'].replace('/', '_').replace(' ', '_').replace('=', '')}>"
            # Use placeholder id offset to avoid collision
            row = {col: "" for col in PSEUDO_CSV_COLUMNS}
            row["id"] = 400000 + placeholder_counter
            row["data"] = placeholder_text
            for k, v in spec["labels"].items():
                row[k] = v
            row["company_source"] = "u13_manual_template"
            row["confidence_min"] = 1.0
            row["conf_T1"] = 1.0
            row["conf_T2"] = 1.0
            row["conf_T3"] = 1.0
            row["conf_T4"] = 1.0
            rows.append(row)

    df = pd.DataFrame(rows, columns=list(PSEUDO_CSV_COLUMNS))
    df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"[manual] wrote template with {len(df)} rows ({n_seed} pre-filled, {len(df) - n_seed} TODO) -> {out_path}")
    print("[manual] open in Excel/CSV editor and fill <TODO_FILL_NNN_...> placeholders with real text.")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("emit", help="Emit annotation template CSV.")
    e.add_argument("--output", default="data/processed/u13/manual_template.csv")
    e.set_defaults(func=cmd_emit_template)
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
