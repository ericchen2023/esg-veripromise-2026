<!-- markdownlint-disable MD025 MD031 MD032 MD033 MD036 MD040 MD041 MD058 MD060 -->

# VeriPromise ESG 2026 — ESG 永續承諾驗證競賽參賽程式碼

> 繁中 ESG 永續承諾驗證競賽（**ESG VeriPromise 2026 / AI CUP 2026**）參賽程式碼。
> 目標：讓 AI 自動判讀企業永續報告書段落的「承諾、驗證時間、證據、證據清晰度」四件事。
> 核心方法為**多模型集成 + 逐任務最佳重組**，並以本機 LLM 合成與弱監督資料補強少數類樣本。
>
> **第一次接觸或非技術讀者**：請先看 [專案總覽_白話版.md](專案總覽_白話版.md)（無術語、5 分鐘看懂全貌）。
>
> **隊伍**：TEAM_10057
>
> 分數說明（重要）：
>
> - **真實競賽 leaderboard**：public **0.6218**（0.6217591，rank 23 / 143，TEAM_10057）/ **private 官方計分 0.6356308**（rank 26 / 143）。private 高於 public +0.0139，模型未過擬合、穩健泛化至隱藏切片（正式得獎名次 2026-07-23 公佈）。
> - **OOF 代理分數**：train+val 5-Fold 交叉驗證 **weighted = 0.71608**。此為「自我測驗」分數，**對二元任務系統性樂觀、不等於真實 leaderboard**；落差成因見 [MASTER_PLAN §60–§68](MASTER_PLAN_AND_PROGRESS.md#60-phase-43--真實-leaderboard-05962-vs-oof-071033-根因分析2026-06-11)。
>
> **想複現 OOF 管線？請看 [REPRODUCE.md](REPRODUCE.md)，從 clone 到 OOF 0.71608 的一站式說明書。**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch 2.2+](https://img.shields.io/badge/pytorch-2.2%2B-ee4c2c)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

---

## 目錄

1. [競賽簡介](#1-競賽簡介)
2. [結果速覽](#2-結果速覽)
3. [專案結構](#3-專案結構)
4. [快速開始](#4-快速開始)
5. [完整重現流程](#5-完整重現流程)
6. [方法概要](#6-方法概要)
7. [文件總覽](#7-文件總覽)
8. [硬體與環境](#8-硬體與環境)
9. [常見問題](#9-常見問題)
10. [授權與引用](#10-授權與引用)

---

## 1. 競賽簡介

ESG VeriPromise 2026 競賽要求對企業永續報告書中的中文段落同時預測四個任務：

| 代號 | 欄位 | 標籤域 | 評分 | 權重 |
| :-: | :-- | :-- | :-- | :-: |
| T1 | `promise_status` | Yes / No | F1（positive=Yes） | 0.20 |
| T2 | `verification_timeline` | already / within_2_years / between_2_and_5_years / longer_than_5_years / N/A | macro F1 | 0.15 |
| T3 | `evidence_status` | Yes / No / N/A | F1（positive=Yes） | 0.30 |
| T4 | `evidence_quality` | Clear / Not Clear / Misleading / N/A | macro F1 | 0.35 |

最終分數：

```
S = 0.20 · F1(T1, Yes) + 0.15 · macroF1(T2)
  + 0.30 · F1(T3, Yes) + 0.35 · macroF1(T4)
```

訓練樣本 **1,000 筆**；測試集競賽方未公開。詳見 [`ESG_永續承諾驗證競賽_2026.md`](ESG_永續承諾驗證競賽_2026.md)。

---

## 2. 結果速覽

### 2.1 真實競賽 leaderboard（主要事實）

| 指標 | 值 |
| :-- | :-- |
| 最佳 leaderboard 分數 | public **0.6218** / private 官方 **0.6356308**（Phase 56 `u41_macbert7T1_pool7T3`，md5 `d3745d70`）|
| 名次 | public 23 / 143 / private 官方 26 / 143（private 高於 public +0.0139，無過擬合；正式名次 2026-07-23 公佈） |
| 逐任務 LB（現行最佳）| T1 0.8056 / T2 0.6147 / T3 0.6974 / T4 0.4549 |
| 分數軌跡 | 0.6037 → 0.6113 → 0.6161 → 0.6176 → 0.6195 → 0.6210 → **0.6218**（分數階梯式提升）|

二元任務 T1/T3 的 OOF 因公司層級洩漏而灌水，真實 LB 遠低於 OOF；macro 任務 T2/T4 則忠實轉移。0.6037 → 0.6218（public）的提升來自「二元模型集成擴大 + promise 級聯連帶抬升全任務」的逐任務重組。**私榜驗證：private 0.6356308 = public 0.6217591 + 0.0139，方向 private > public，代表模型未過擬合、穩健泛化至隱藏測試集。** 完整根因見 [MASTER_PLAN §60–§68](MASTER_PLAN_AND_PROGRESS.md#60-phase-43--真實-leaderboard-05962-vs-oof-071033-根因分析2026-06-11)。

### 2.2 OOF 代理分數（AP-D4，非 leaderboard）

Phase 38 **AP-D4**（2026-05-20）— 5-Fold OOF（seed42）。此為交叉驗證代理，僅供方法演進追蹤，二元任務分數不可轉移到 leaderboard：

| 指標 | AP-D4（OOF 代理）| AP-D3 | Phase 36 |
| :-- | :-: | :-: | :-: |
| 加權總分（OOF）| **0.71608** | 0.71364 | 0.71018 |
| T1 promise_status (F1, Yes) | **0.94387** | 0.94337 | 0.94210 |
| T2 verification_timeline (macro F1) | **0.63150** | 0.63061 | 0.62778 |
| T3 evidence_status (F1, Yes) | 0.88011 | **0.88045** | 0.87774 |
| T4 evidence_quality (macro F1) | **0.48157** | 0.47496 | 0.46934 |

Δ apples-to-apples（vs AP-D3 同 seed42 × 同 3-view 設定）= **+0.00244**（T1+0.00050、T2+0.00089、T3−0.00034、**T4+0.00661**）；AP-D4 加入的 stem #8 以 **Ollama qwen2.5:7b-instruct** 生成 80 列 Misleading + 60 列 within_2_years 作為訓練語料補充，針對 T4 / T2 低頻類別。重現指令見 [REPRODUCE.md §5](REPRODUCE.md#5-集成產出-sota-oof-071608)。

### Phase 32 → 36 進化軌跡

| 階段 | 主要差異 | OOF |
| :-: | :-- | :-: |
| Phase 32 | p2_combo_best baseline + per-task hillclimb（單 stem） | 0.6694 |
| Phase 33 | + class-weighted CE / Focal γ=3 on T4 | 0.6741 |
| Phase 34 | + U10 v1/v2 偽標籤兩階段微調 | 0.6878 |
| Phase 35 | + 5-way 6 stems × 3-view TTA | 0.7034 |
| **Phase 36** | + U6-pro 反翻譯擴增 & 6-way × 3-view × per-task hillclimb | **0.71018** |
| Phase 37 | + Aug-Plus 47 列手工種子（single-stem ablation, seed=42）→ 7-way × 3-view（AP-D3） | 0.71364 |
| **Phase 38** | + Ollama qwen2.5:7b-instruct 本機 LLM 合成 （stem #8）→ **8-way × 3-view（AP-D4）** | **0.71608** |
| Phase 39 | AP-D5 grid 0.05 細粒度權重搜索可行性評估（單機 Stage A 單輪 ≈3h，列為負面消融；SOTA 維持 AP-D4） | 0.71608（unchanged） |
| 2026-05-25 工程硬化 | submission validator + fast TTA evaluator + 小預算 refinement 入口；SOTA anchor 不變 | 0.71608（unchanged） |

完整實驗紀錄見 [MASTER_PLAN_AND_PROGRESS.md](MASTER_PLAN_AND_PROGRESS.md)。

---

## 3. 專案結構

```
esg-veripromise-2026/
├── SOTA_Reproduction_Phase36.ipynb    # 互動式重現手冊（檔名保留歷史 Phase 36，內容已延伸至 Phase 38）
├── MASTER_PLAN_AND_PROGRESS.md        # 完整研究決策總控（Phase 1~39 + 2026-05-25 工程硬化）
├── ESG_永續承諾驗證競賽_2026.md         # 競賽規則
├── README.md
├── requirements.txt
├── pyproject.toml
│
├── vpesg4k_train_1000 V1.csv          # 官方訓練集（1,000 筆）
├── vpesg4k_train_1000 V1.json
├── [External]_VeriPromiseESG_..._Baseline_Code_ZH.ipynb  # 官方 baseline
│
├── configs/                # 8 個 AP-D4 SOTA stem 的 YAML 配方
│   ├── base.yaml
│   ├── exp_p2_combo_best.yaml
│   ├── exp_p2_combo_best_u10_pseudo.yaml
│   ├── exp_p2_combo_best_u10_pseudo_v2.yaml
│   ├── exp_p2_combo_best_u10_pseudo_v2_classw_focal_t4_g3.yaml
│   ├── exp_p2_combo_best_u10_pseudo_v3_classw_focal_t4_g3.yaml
│   ├── exp_p2_combo_best_classw_focal_u6pro.yaml
│   ├── exp_p2_combo_best_aug_plus.yaml
│   └── exp_p2_combo_best_aug_plus_v2.yaml
│
├── src/                    # 框架原始碼
│   ├── seed.py             # 全域種子
│   ├── config.py           # YAML 遞迴 extends 載入
│   ├── train_kfold.py      # 5-fold 訓練入口
│   ├── train_pseudo_kfold.py  # 兩階段（pseudo + real → real-only）
│   ├── data/               # loader / dataset / splits
│   ├── models/             # MultiTaskClassifier + Pooler
│   ├── training/           # trainer / losses / schedulers
│   ├── inference/          # post_process（階層約束）
│   ├── eval/               # metrics（weighted_score）
│   └── tools/              # OOF 集成 / hillclimb / 3-view TTA / submission validator
│
├── scripts/
│   ├── build_sota_notebook.py   # 重新產生 SOTA notebook
│   ├── u10_collect_sr.py        # M1: 永續報告書 PDF 爬取
│   ├── u10_pdf_extract*.py      # M3: PDF 段落擷取（v1/v2/v3）
│   ├── u10_pseudo_label*.py     # M4: 偽標籤閘門（v1/v2/v3）
│   ├── u10_v{2,3}_sustaihub_crawl.py  # 擴充來源
│   ├── u10_sources.py           # 公司清單
│   └── u6_backtranslate_pro.py  # NLLB-200 雙樞紐回譯 + ESG 術語表
│
└── tests/                  # pytest 單元測試（評分、後處理、validator、fast evaluator）
```

> **本機獨有（不上傳）**：
> - `outputs/checkpoints/`（AP-D4 8 stems × seed42 約 44 GB；若保留歷史多 seed/多 Phase checkpoint，可能達百 GB 以上）
> - `data/raw/` ≈ 1.3 GB（U10 永續報告書原始 PDF）
> - `data/processed/`、`data/splits/`（可重生）
> - `outputs/cache/`（per-view OOF 快取）
> - `reports/{logs,runs,experiments,u6}/` 與 `reports/analysis/` 中未追蹤的 per-run 中介物（核心 AP-D summary/meta 已保留）
>
> 完整理由與重生方式見 §5 與 [`.gitignore`](.gitignore)。

---

## 4. 快速開始

### 4.1 安裝

```powershell
# Windows / PowerShell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

```bash
# Linux / macOS
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 4.2 跑單元測試

```powershell
pytest -q
```

### 4.3 開啟互動式 notebook（legacy 對照）

```powershell
jupyter notebook SOTA_Reproduction_Phase36.ipynb
```

此 notebook 檔名保留 Phase 36 歷史脈絡，適合作互動式對照；AP-D4 標準主線仍以 [REPRODUCE.md](REPRODUCE.md) 為準。筆記本第 §1 章可選擇執行模式：

| 模式 | 用途 | 預估耗時（RTX 5060 Laptop 8GB） |
| :-- | :-- | :-- |
| `explain` | 僅閱讀，不執行模型 | 0 分鐘 |
| `load_oof` | **預設**。從既有 checkpoints/cache 重建 Phase 36 OOF + hillclimb 對照分數 0.71018 | 8 ~ 12 分鐘 |
| `demo` | 對單一 stem 跑 1 fold smoke-train | 3 ~ 5 分鐘 |
| `full` | legacy Phase 36 從零訓練 6 stems × 3 seeds × 5 folds | 30 ~ 40 GPU 小時 |

> **注意**：`demo` / `full` 需要 GPU 自行訓練 `outputs/checkpoints/best.pt`。
> 公開 GitHub 倉儲僅含程式碼與設定檔，不含模型權重；AP-D4 權重需依 [REPRODUCE.md](REPRODUCE.md) 重新訓練或由本隊另行提供。

#### `load_oof` 模式快速重現（無 GPU 也可）

`load_oof` 模式利用 §13 的 TTA cache 命中機制，**不需要 best.pt**，只需以下三類輕量檔案：

| 路徑 | 檔案數 | 大小 |
| :-- | :-: | :-: |
| `outputs/checkpoints/{stem}/seed{S}/fold{F}/oof_probs.npz` | 105 | ~0.76 MB |
| `outputs/cache/u10_tta/{stem}_{middle,tail}.npz` | 12 | ~1.2 MB |
| `data/splits/{stem}/...` | 12 | ~0.3 MB |

完整輕量 bundle 已預先打好（`load_oof_bundle_phase36.zip`，約 1.4 MB），請洽本隊
或於 [GitHub Releases](https://github.com/YOUR-ACCOUNT/esg-veripromise-2026/releases) 取得。
解壓步驟：

```powershell
# 在 repo 根目錄
Expand-Archive load_oof_bundle_phase36.zip -DestinationPath . -Force
# 之後設 MODE="load_oof" 執行 SOTA notebook 即可重現 0.71018
```

---

## 5. 完整重現流程

從零端到端重訓 Phase 38 AP-D4 SOTA（不依賴任何既有 checkpoint）。

### 5.1 重生資料層

```powershell
# 5.1.1 確認官方資料就位（已隨 repo 上傳）
ls "vpesg4k_train_1000 V1.csv", "vpesg4k_train_1000 V1.json"

# 5.1.2 （可選）重跑 U10 永續報告書管線
#   M1: 從公開資訊觀測站爬 30 家公司 PDF（約 1.3 GB / 60 分鐘）
python scripts\u10_collect_sr.py
#   M3: PDF 段落擷取 + ESG 關鍵字過濾 + SimHash 去重
python scripts\u10_pdf_extract_v3.py
#   M4: 對未標註段落跑教師模型 + 四任務聯合閘門（先需先有 p2_combo_best 15 ckpts）
python scripts\u10_pseudo_label_v3.py

# 5.1.3 （可選）重跑 U6-pro 反翻譯擴增（NLLB-200，約 3 GB 模型 / 90 分鐘 GPU）
python scripts\u6_backtranslate_pro.py
```

> 上述步驟涉及外部網站結構與 PDF 解析版本依賴，**byte-exact 重現可能漂移 ±0.003**。
> 若僅需重現官方資料 baseline 而不在意 U10/U6 增強，請跳過 5.1.2 / 5.1.3。

### 5.2 訓練 8 個 AP-D4 stem

```powershell
# 預估 7~8 GPU 小時 / RTX 5060 Laptop 8GB；單 fold 約 5~7 分鐘
python -m src.train_kfold        --config configs\exp_p2_combo_best.yaml
python -m src.train_pseudo_kfold --config configs\exp_p2_combo_best_u10_pseudo.yaml
python -m src.train_pseudo_kfold --config configs\exp_p2_combo_best_u10_pseudo_v2.yaml
python -m src.train_pseudo_kfold --config configs\exp_p2_combo_best_u10_pseudo_v2_classw_focal_t4_g3.yaml
python -m src.train_pseudo_kfold --config configs\exp_p2_combo_best_u10_pseudo_v3_classw_focal_t4_g3.yaml
python -m src.train_pseudo_kfold --config configs\exp_p2_combo_best_classw_focal_u6pro.yaml
python -m src.train_pseudo_kfold --config configs\exp_p2_combo_best_aug_plus.yaml
python -m src.train_pseudo_kfold --config configs\exp_p2_combo_best_aug_plus_v2.yaml
```

每個 stem 會在 `outputs/checkpoints/{stem}/seed{S}/fold{F}/` 寫入 `best.pt` 與 `oof_probs.npz`。

### 5.3 集成與重現分數

建議直接依 [REPRODUCE.md §5](REPRODUCE.md#5-集成產出-sota-oof-071608) 執行 AP-D4 8-way × 3-view ensemble。預期輸出：

```
[u10-tta FINAL] 0.7160840624
  promise_status         F1 = 0.94387
  verification_timeline  F1 = 0.63150
  evidence_status        F1 = 0.88011
  evidence_quality       F1 = 0.48157
```

實際數字會因 cuDNN 非決定性、torch 浮點累加在 ±0.0005 內漂移。

---

## 6. 方法概要

詳細推導與消融實驗見 [MASTER_PLAN_AND_PROGRESS.md](MASTER_PLAN_AND_PROGRESS.md) §4–§18、§47–§56。

### 6.1 模型

- 主幹：`hfl/chinese-macbert-base`（中文 MacBERT，BERT 架構＋MLM-as-correction 預訓練）
- 池化：`cls_mean`（拼接 [CLS] 與 attention-mask 加權平均向量；2 × hidden_size = 1536）
- 分類頭：4 個獨立的 `nn.Linear(1536, C_t)` + 任務獨立 dropout

> 已驗證 large 模型（macbert-large、xlm-r-base、nezha-base、ernie-base、electra-base）在 1k 樣本下均 < base，故維持 base 主幹。

### 6.2 訓練策略

| 組件 | 設定 |
| :-- | :-- |
| 切分 | 5-Fold StratifiedKFold（複合鍵 `promise_status \| evidence_status`），3 seeds |
| 優化器 | AdamW，lr=3e-5，wd=0.01 |
| 學習率排程 | Cosine + 10% warmup |
| 數值精度 | AMP fp16 + grad_clip 1.0 |
| 批量 | bs=8 + grad_accum=2，max_len=384 |
| 損失 | 多任務 CE；T1/T2/T3 用類別加權；T4 用 Focal Loss (γ=3) |
| 提前停止 | patience=2，依 fold OOF 加權分數 |

### 6.3 半監督與資料擴增

- **U10 弱監督管線**（M1~M5）：從公開資訊觀測站爬 30 家公司永續報告書 → PDF 段落擷取＋SimHash 去重 → 教師模型四任務聯合閘門（T1≥0.80 / T2≥0.60 / T3≥0.70 / T4≥0.60）→ 兩階段微調（Stage A：pseudo+real / Stage B：real-only）。
- **U6-pro 反翻譯**：NLLB-200-distilled-600M × 雙樞紐（en、ja）× 雙溫度，搭配 109 條 ESG 術語表保護與 ChrF 過濾，對 minority 類別（T2 within_2y / longer_5y、T4 Misleading / Not Clear）產生 434 筆增強樣本，僅在 fold-train 端注入，**永不洩漏 val/OOF**。
- **U13 LLM 合成 + 人工標註**（Phase 37，官方 2026-05-15 裁示明確授權）：
  - `scripts/u13_synth_llm.py`：4 子命令（generate / validate / promote / merge），5 個 provider（OpenAI / Anthropic / Gemini / Ollama / Mock），鎖定 T2 within_2y (13→143)、T4 Misleading (1→131) 兩大瓶頸類。
  - `scripts/u13_manual_seed.py`：emit 20 筆本隊手寫種子 + 311 筆 `<TODO_FILL>` 模板（含目標標籤分佈）。
  - `scripts/u13_llm_judge.py`：LLM-as-judge 對既有 U10 偽標籤重新打分（`llm_judge_score` ∈ [0,1]）。
  - 設定檔：`configs/exp_p2_combo_best_classw_focal_u13_synth.yaml`（繼承 Phase 36 最佳配方 + U13 合成資料）。
  - 測試：`tests/test_u13_synth.py` 12 個測試全綠。
  - **詳見 [MASTER_PLAN §54](MASTER_PLAN_AND_PROGRESS.md#54-phase-37-並行路線--u13llm-合成--人工標註--llm-評審規劃中llm-合成已於-55-落地)**。

### 6.4 集成（Phase 36 到 AP-D4）

AP-D4 沿用 Phase 36 已驗證的 per-task stem/view 權重搜尋設計，並擴充為 8-way × 3-view。對每個任務同時搜尋兩組 simplex 權重：

- **Stem 權重** $w^{(t)}$（AP-D4 為 8 個 stem 加總為 1）
- **View 權重** $\alpha^{(t)} \in \Delta^2$（stored + middle + tail 三視角加總為 1）

AP-D4 標準重現使用 grid step = 0.1 的座標下降，A↔B 交替 2 輪即收斂；2026-05-25 起評分核心接上 fast evaluator，避免每個候選都走逐列 string/sklearn pipeline。下表為 Phase 36 6-way legacy 權重範例，AP-D4 最終權重請以 `reports/analysis/_ensemble/ap_d4_8way_3view_meta.json` 為準：

| 任務 | Stem w*（依 stem #1~#6 順序） | View α*（stored, middle, tail） |
| :-: | :-- | :-- |
| T1 | (0.10, 0, 0, 0, 0.50, 0.40) | (0, 0, 1) |
| T2 | (0.10, 0.10, 0, 0.40, 0.30, 0.10) | (0.4, 0, 0.6) |
| T3 | (0.30, 0.20, 0.30, 0, 0, 0.20) | (0.5, 0.5, 0) |
| T4 | (0.10, 0.10, 0, 0.40, 0, 0.40) | (0.5, 0.5, 0) |

### 6.5 階層後處理

```python
if promise_status == "No":
    verification_timeline = evidence_status = evidence_quality = "N/A"
elif evidence_status == "No":
    evidence_quality = "N/A"
```

該約束**必須在 hillclimb 評分迴圈內即套用**，否則 hillclimb 會選到不合法但 OOF F1 較高的權重。

### 6.5 Aug-Plus 擴增模組（Phase 37 · 已完成訓練 single-stem ablation）

2026-05-17 主辦單位裁示：「自行查詢與標註資料」屬合法資料擴增；**手工撰寫或 LLM 合成皆允許**，唯一禁區是測試集與最終預測。基此新增 **Aug-Plus（AP1~AP5）** 模組，鎖定 Phase 36 雙瓶頸：T4 `Misleading`（train 僅 1 筆）與 T2 `within_2_years`（train 僅 13 筆）。

| 模組 | 檔案 | 角色 |
| --- | --- | --- |
| AP1 schema | [`src/data/aug_schema.py`](src/data/aug_schema.py) | 欄位、階層驗證、ID 空間切分 |
| AP2 種子 | [`assets/aug_plus/build_handcrafted_v1.py`](assets/aug_plus/build_handcrafted_v1.py) → `handcrafted_v1.csv` | 50 列親撰繁中 ESG 段落（T4 Misleading × 20、T2 within_2y × 22） |
| AP3 合成 | [`scripts/ap_llm_synth.py`](scripts/ap_llm_synth.py) | `generate/validate/merge/promote`；多 provider（Mock 預設、OpenAI/Anthropic/Gemini/Ollama 為 stub） |
| AP3 prompt | [`configs/prompts/ap_*.yaml`](configs/prompts/) | 系統提示 + fewshot + user 模板 |
| AP4 閘門 | [`scripts/ap_quality_gate.py`](scripts/ap_quality_gate.py) | 長度、schema 防禦、SimHash 去重、teacher 信心 hook |
| AP5 訓練 | [`configs/exp_p2_combo_best_aug_plus.yaml`](configs/exp_p2_combo_best_aug_plus.yaml) | 疊在 Phase 36 best 之上、不改任何程式碼 |

端到端煙霧（已通過 25/25 單元測試）：
```powershell
python assets/aug_plus/build_handcrafted_v1.py
python scripts/ap_llm_synth.py generate --target misleading --n 30 --provider mock --seed 42
python scripts/ap_llm_synth.py generate --target within_2_years --n 20 --provider mock --seed 7
python scripts/ap_llm_synth.py merge --out data/aug_plus/aug_merged_raw.csv
python scripts/ap_quality_gate.py --skip-teacher
python scripts/ap_llm_synth.py promote --gated-csv data/aug_plus/aug_gated.csv --with-u10
# → 訓練：python src/train_pseudo_kfold.py --config configs/exp_p2_combo_best_aug_plus.yaml
```

**狀態：Phase 37 G3 smoke + G4 5-fold 全訓練皆已完成**（seed=42 × 5 folds · 約 53 min 於 RTX 5060 Laptop）。Single-stem OOF = **0.66966**（vs Phase 36 stem #6 baseline 0.67044，**Δ = −0.00078**），落在 U12 OOF noise budget ±0.0045 內，未升單 stem SOTA；其中 **T4 +0.00483 與設計意圖一致**，T2 因 47 列種子以 within_2y 為主造成 −0.02105 regression。後續已在 AP-D3 7-way × 3-view ensemble 證明 diversity 價值（0.71364），並由 AP-D4 stem #8 推進至 0.71608。詳 [MASTER_PLAN §53](MASTER_PLAN_AND_PROGRESS.md#53-phase-37--aug-plus-hand-crafted-minority-訓練與-single-stem-ablation2026-05-18)。

---

## 7. 文件總覽

| 文件 | 內容 |
| :-- | :-- |
| [`SOTA_Reproduction_Phase36.ipynb`](SOTA_Reproduction_Phase36.ipynb) | 17 章可執行重現手冊；首選入口 |
| [MASTER_PLAN_AND_PROGRESS.md](MASTER_PLAN_AND_PROGRESS.md) | 從 Phase 1 到 Phase 39 的完整研究決策總控（含 AP-D4 與 Phase 39 負面消融） |
| [`競賽報告.md`](競賽報告.md) | 完整競賽技術報告（圖表、參考文獻、可重現性查核）|
| [`專案總覽_白話版.md`](專案總覽_白話版.md) | 無術語白話總覽 |
| [`ESG_永續承諾驗證競賽_2026.md`](ESG_永續承諾驗證競賽_2026.md) | 競賽官方規則 |
| [`[External]_VeriPromiseESG_..._Baseline_Code_ZH.ipynb`](%5BExternal%5D_VeriPromiseESG_2026_ESG_Promise_Verification_Competition_Baseline_Code_ZH.ipynb) | 主辦單位提供的 baseline notebook |

---

## 8. 硬體與環境

### 開發環境

| 項目 | 版本 |
| :-- | :-- |
| OS | Windows 11 / WSL2 Ubuntu 22.04 |
| Python | 3.10 ~ 3.13 |
| PyTorch | 2.2 ~ 2.4 + CUDA 12.x |
| transformers | 4.41 ~ 4.45 |
| GPU | RTX 4060/5060 Laptop 8GB（最低）／ A6000 48GB（推薦） |

### 訓練成本

| 項目 | 估計 |
| :-- | :-- |
| 單 fold 訓練（base，max_len=384，bs=8） | 5~7 分鐘 |
| 單 stem（seed42 × 5 folds） | 約 30~40 分鐘 |
| AP-D4 8 stems 訓練 | 約 7~8 GPU 小時 |
| 完整歷史多 seed / 多 Phase 實驗重跑 | 30~40 GPU 小時以上 |

---

## 9. 常見問題

**Q1. clone 之後為何沒有 `outputs/checkpoints/`？**
A. AP-D4 8 stems × seed42 checkpoint 約 44 GB；若連歷史多 seed / 多 Phase 權重一起保留，會達百 GB 等級，超過 GitHub 與 Git LFS 的合理上限。請依 §5 重訓，或洽本隊取得權重備份。

**Q2. `MODE="load_oof"` 但筆記本提示 checkpoint missing？**
A. 你需要先完成 §5.2 訓練。或將別處備份的 `outputs/checkpoints/{stem}/seed{S}/fold{F}/best.pt` 與 `oof_probs.npz` 放回原位。

**Q3. 為什麼用 base 而非 large 主幹？**
A. 1k 樣本對 large 容量過度。已驗證 macbert-large 在 8GB GPU 上 bs 必須降至 2，OOF 反而 −0.018。詳見 MASTER_PLAN §A.7。

**Q4. T4 為何單獨用 Focal γ=3？**
A. T4 的 `Misleading` 與 `Not Clear` 類別計數合計 < 80，是分數天花板。Focal Loss 透過 $(1-p_t)^\gamma$ 抑制易類別主導，γ=3 為 Phase 33 grid search 結果。

**Q5. U10 偽標籤如何避免污染驗證集？**
A. 偽標籤只進 fold-train，永遠不寫入 fold-val；OOF 索引嚴格依官方 1,000 筆全域索引拼接。

**Q6. 我的分數比 0.71608 低 / 高 0.005，正常嗎？**
A. 若只重跑 AP-D4 ensemble 且使用既有 OOF/cache，漂移應接近 ±0.0005；若從 U10 / U6 / Aug-Plus 資料或模型訓練重新生成，外部 PDF 解析、tokenizer 與 cuDNN 非決定性可能再帶來 ±0.003 級漂移。

---

## 10. 授權與引用

### 程式碼

本倉儲程式碼（`src/`、`scripts/`、`configs/`、notebook）採 MIT 授權。

### 資料

- `vpesg4k_train_1000 V1.csv` / `.json`：競賽主辦單位釋出，僅供 ESG VeriPromise 2026 競賽研究用。轉用請參照官方授權條款。
- U10 永續報告書（本機 `data/raw/`）：來源為公開資訊觀測站；著作權歸各企業所有，**不隨本倉儲發佈**。

### 引用

若本程式碼對您的研究有幫助，請引用 MASTER_PLAN：

```bibtex
@misc{veripromise2026_phase38,
  title  = {VeriPromise ESG 2026 — Phase 38 AP-D4 SOTA Pipeline (0.71608 OOF)},
  author = {TEAM\_10057},
  year   = {2026},
  howpublished = {\url{https://github.com/YOUR-ACCOUNT/esg-veripromise-2026}}
}
```

---

> 任何問題、bug、再現失敗，歡迎開 issue 或 pull request。
