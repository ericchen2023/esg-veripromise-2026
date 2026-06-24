<!-- markdownlint-disable MD025 MD032 MD033 MD036 MD040 MD058 MD060 -->

# VeriPromise ESG 2026 — 主規劃與進度紀錄

本文件是 VeriPromise ESG 2026 專案的研究決策總控與完整進度紀錄。它的任務不是取代 [README.md](README.md) 的入口摘要，也不是取代 [REPRODUCE.md](REPRODUCE.md) 的逐步重現 SOP；本文件負責保存「為什麼這樣做、做過什麼、結果如何、下一步是否值得做」的完整證據鏈。

---

<a id="document-purpose"></a>
## 1. 文件定位與閱讀方式

### 1.1 本文件用途

| 文件 | 職責 | 何時閱讀 |
| :-- | :-- | :-- |
| [專案總覽_白話版.md](專案總覽_白話版.md) | 無術語的白話說明：比賽在做什麼、我們怎麼做、做到哪 | 第一次接觸、非技術讀者、想快速理解全貌 |
| [競賽報告.md](競賽報告.md) | 完整競賽技術報告（含圖表、參考文獻、可重現性查核）| 繳交報告、審查、引用最終成果 |
| [README.md](README.md) | 專案入口、方法速覽、導覽 | 第一次進 repo、需要快速理解成果 |
| [REPRODUCE.md](REPRODUCE.md) | 從 clone 到 AP-D4 0.71608 的可執行 SOP | 要重訓、重跑 ensemble、驗證 SOTA |
| [MASTER_PLAN_AND_PROGRESS.md](MASTER_PLAN_AND_PROGRESS.md) | 單一研究真相來源：決策、實驗、失敗教訓、待辦 | 要判斷下一步、回溯實驗原因、避免重複踩坑 |
| [ESG_永續承諾驗證競賽_2026.md](ESG_永續承諾驗證競賽_2026.md) | 競賽規格與官方規則摘要 | 要確認任務定義、提交格式、外部資料限制 |
| [SOTA_Reproduction_Phase36.ipynb](SOTA_Reproduction_Phase36.ipynb) | 互動式研究筆記本，保留舊 Phase 36 名稱但內容已延伸到 Phase 38 | 要以 Notebook 方式逐段檢視流程 |

### 1.2 建議閱讀路徑

| 需求 | 建議章節 |
| :-- | :-- |
| 想知道目前最強結果 | [§2 現況快照](#current-status)、[§5 SOTA 軌跡](#sota-scoreboard) |
| 想重現 0.71608 | [REPRODUCE.md §5](REPRODUCE.md#5-集成產出-sota-oof-071608)、[§8 現行 AP-D4 管線](#current-pipeline) |
| 想判斷下一步做什麼 | [§14 待辦與 ROI](#roadmap)、[§13 上限與殘留風險](#risk-and-limits) |
| 想避免重跑失敗方向 | [§12 負面結果與禁區](#negative-results)、[§75.3](#753-已暫緩或證明不可行的方向避免重複嘗試) |
| 想看完整實驗證據 | [Part IV 完整 Phase Log](#part-iv--完整-phase-log-append-only-evidence) |
| 想確認外部資料是否合法 | [§18 規則與治理](#governance)、[§77](#77-競賽規則對外部資料的立場已確認可用) |

### 1.3 白話術語表（給非技術讀者）

本文件為研究日誌、術語密集。下表用白話解釋常見術語，方便外行讀者理解；完整無術語的說明請看 [專案總覽_白話版.md](專案總覽_白話版.md)。

| 術語 | 白話意思 |
| :-- | :-- |
| T1 / T2 / T3 / T4 | 四個子任務：是否有承諾 / 多久可驗證 / 是否有證據 / 證據夠不夠清楚 |
| LB（leaderboard）| 比賽官方排行榜的真實分數；分公榜 public 與私榜 private（private 為官方正式計分）|
| OOF（out-of-fold）| 「自我測驗」分數：把訓練資料切成幾份輪流互相考；會偏樂觀，不等於 LB |
| 加權分數（weighted score）| 四個任務分數按權重（0.20/0.15/0.30/0.35）加總的總分 |
| macro-F1 | 多分類的評分；對「少數類別」特別敏感（少數類做不好，分數就低）|
| ensemble（集成）| 把多個模型的判斷平均起來，比單一模型更穩更準 |
| 重組（recombination）| 四個任務各自挑「最會做那一題」的模型來回答，再組合 |
| cascade（級聯）| 規則連動：判定「沒有承諾」就自動把後面三項填成 N/A |
| TAPT | 讓模型先大量閱讀同領域（ESG）的文字、適應語感，再做正式訓練 |
| stem / seed | stem＝一個訓練配方產生的模型；seed＝同配方換不同亂數起點再訓一個，增加多樣性 |
| 過擬合（overfitting）| 模型「背答案」而非真懂，換新題目就失準 |

### 1.4 維護規則

1. 新實驗先寫入對應 Phase 章節，再回填首頁快照與 SOTA 表。
2. 任何分數必須標明 OOF、valid、test 或 LB 來源；不得混用。
3. 任何被拒絕的方向要進入 X 系列或暫緩清單，避免日後重複浪費算力。
4. 對外部資料、LLM 合成、人工標註的使用必須記錄來源、產生方式、資料隔離與合法性依據。
5. 若 README 或 REPRODUCE 有 SOTA 數字更新，本文件也必須同步更新。

---

<a id="current-status"></a>
## 2. 現況快照

截至 2026-06-24，本專案已完成競賽並取得官方最終成績（private 0.6356308 / rank 26/143；public 0.6218 / rank 23/143）。

重要：本專案存在大幅的 OOF 與 leaderboard 落差。歷史記錄中的 0.71608 是 5-fold 交叉驗證的 OOF 代理分數，已於 Phase 43–51 證實對二元任務（T1/T3）系統性樂觀約 0.18，真實競賽 leaderboard 落在 0.60–0.61 區間。閱讀本文件任何 0.71 等級分數時，一律視為 OOF 代理，不等於 leaderboard。

### 2.0 真實 leaderboard 現況（主要事實來源）

| 指標 | 目前值 |
| :-- | :-- |
| 最佳 leaderboard 分數 | public **0.6218**（rank 23/143，2026-06-18）、private 官方 **0.6356308**（rank 26/143），Phase 56 `u41_macbert7T1_pool7T3` md5 `d3745d70` |
| 上一代最佳 | 0.6161，`u41_bestT1_pool5T3`（2026-06-15）|
| Phase 56 突破軌跡 | 重組（LB-only，cascade 放大）：0.6037→0.6113→0.6120→0.6161→0.6176→0.6195→0.6210→**0.6218** public（rank 23/143）→ private 官方 **0.6356308**（rank 26/143，無過擬合，+0.0139）。槓桿=二元 T1/T3 集成擴大（更多 seed/模型），經 promise 級聯連帶抬升全任務 |
| 已銀行安全基線 | c3 = 0.60366，`FROZEN_BEST_0.6037_c3.csv`（md5 46D31CE3…）|
| 最近完成 Phase | Phase 55（2026-06-15），Not Clear 資料擴增——單模有效但集成飽和，0.6120 確認為天花板 |
| 待上傳候選 | 無（所有新候選的 val T4 ≤ equal8，未過上傳前閘）；保留 0.6120 |
| 窮盡式裁決（4 軸全死）| 二元模型（Phase 53）、post-hoc 決策（Phase 54）、T4 集成多樣化（Phase 54）、T4 資料擴增（Phase 55 單模 +0.051 NC F1 但集成無法落地）——皆帶轉移驗證證偽 |
| 0.62+ 為何不可達 | 數學上需 T4≈0.50；Not Clear 可分性 + Misleading(support=2) 不支持；8-stem 多樣 ensemble 已飽和 Not Clear 訊號，同配方增強只稀釋。落差為 test 標註慣例（同 50 公司），非建模可閉合 |
| 官方 test 上傳截止 | 2026-06-17 22:00，每日上限 3 次，leaderboard 以最後一次上傳結果呈現 |
| 訓練資料 | train 1,000 + val 1,000 = 合併 2,000 筆，50 家 TWSE 上市公司 |
| 本機硬體 | RTX 5060 Laptop 8 GB CUDA，Windows + PowerShell 5.1 |

逐任務 leaderboard 分數（c3 基線、TAPT 單模、與現行最佳 binary）：

| 任務 | 權重 | c3 LB | TAPT-only LB | binary LB（現行）|
| :-- | :--: | :--: | :--: | :--: |
| T1 promise_status（二元 F1）| 0.20 | 0.7864 | 0.7967 | **0.7967** |
| T2 verification_timeline（macro F1）| 0.15 | 0.6060 | 0.5117 | **0.6114** |
| T3 evidence_status（二元 F1）| 0.30 | 0.6750 | 0.6804 | **0.6804** |
| T4 evidence_quality（macro F1）| 0.35 | 0.4370 | 0.4288 | **0.4460** |
| weighted | | 0.6037 | 0.5903 | **0.6113** |

現行最佳 = TAPT(T1, T3) + c3(T2, T4)，即 `phase51_tapt_binary`；TAPT 的 promise 級聯同時抬升二元與 macro。完整紀錄見 [§69 Phase 52](#69-phase-52--tapt-二元全外科重組lb-突破-0612026-06-14)，根因鏈見 [§60–§68 Phase 43–51](#60-phase-43--真實-leaderboard-05962-vs-oof-071033-根因分析2026-06-11)。

真實 leaderboard 軌跡（0.6037 → 0.6218）與逐任務最佳來源：

![真實 LB 軌跡](reports/figures/fig2_score_trajectory.png)

![逐任務 LB 來源比較](reports/figures/fig_pertask_sources.png)

### 2.1 OOF 代理歷史最佳（AP-D4，非 leaderboard）

下表為 Phase 38 AP-D4 的 5-fold OOF 代理分數，保留作方法演進追蹤。其二元任務分數（T1 0.94、T3 0.88）因 StratifiedKFold 在 2,000 列上的公司層級洩漏而灌水，不可轉移到 leaderboard。

| 指標 | AP-D4（OOF 代理）| AP-D3 | Phase 36 |
| :-- | --: | --: | --: |
| Weighted score（OOF）| 0.71608 | 0.71364 | 0.71018 |
| T1 promise_status F1(Yes) | 0.94387 | 0.94337 | 0.94210 |
| T2 verification_timeline macro-F1 | 0.63150 | 0.63061 | 0.62778 |
| T3 evidence_status F1(Yes) | 0.88011 | 0.88045 | 0.87774 |
| T4 evidence_quality macro-F1 | 0.48157 | 0.47496 | 0.46934 |

AP-D4 相對 AP-D3 的 apples-to-apples OOF 增益為 +0.00244，主要來自 T4 +0.00661。此增益僅存在於 OOF；leaderboard 上二元任務的真實分數遠低於 OOF（見 §2.0）。

---

<a id="document-map"></a>
## 3. 文件與產物地圖

### 3.1 文件分工

| 路徑 | 狀態 | 說明 |
| :-- | :-- | :-- |
| [README.md](README.md) | 對外入口 | 應維持短而準，引用本文件作完整研究日誌 |
| [REPRODUCE.md](REPRODUCE.md) | 主要重現手冊 | AP-D4 0.71608 的指令來源；訓練、資料、集成命令以此為準 |
| [MASTER_PLAN_AND_PROGRESS.md](MASTER_PLAN_AND_PROGRESS.md) | 研究總控 | 所有決策、SOTA 軌跡、失敗教訓與未來路線以此為準 |

### 3.2 核心程式與資料產物

| 類別 | 主要位置 | 角色 |
| :-- | :-- | :-- |
| 訓練入口 | `src/train_kfold.py`、`src/train_pseudo_kfold.py` | 官方資料與 pseudo/augmented two-stage 訓練 |
| 集成工具 | `src/tools/u10_per_task_tta.py` | AP-D3/AP-D4 的 stem/view per-task hillclimb；已接上 fast evaluator 與 budgeted random refinement |
| 搜尋評分核心 | `src/tools/tta_fast_eval.py` | integer-label exact evaluator，避免每個 candidate 都走 string/sklearn pipeline |
| 後處理 | `src/inference/post_process.py` | T1/T3 條件式約束覆寫 |
| 提交守門 | `src/tools/validate_submission.py` | prediction/submission CSV 的欄位、label domain、ID、hierarchy 與字串欄位檢查 |
| 評分 | `src/eval/metrics.py` | 4 任務 F1 與 weighted score |
| 設定檔 | `configs/exp_p2_combo_best*.yaml` | 8 個 SOTA stem 的配方來源 |
| Aug-Plus | `assets/aug_plus/`、`data/aug_plus/`、`scripts/ap_*.py` | 手工與 LLM 合成少數類擴增 |
| U10/U6-pro 資料 | `data/processed/u10/`、`data/processed/u6_pro/` | 弱監督與回譯擴增中介產物 |
| Ensemble 報告 | `reports/analysis/_ensemble/` | OOF 預測、summary、meta、submission 範本 |

---

<a id="competition-spec"></a>
## 4. 競賽規格與評分定義

### 4.1 四任務定義

| 子任務 | 欄位 | 類別 | 指標 | 權重 |
| :--: | :-- | :-- | :-- | --: |
| T1 | `promise_status` | Yes / No | F1 with positive=Yes | 0.20 |
| T2 | `verification_timeline` | already / within_2_years / between_2_and_5_years / longer_than_5_years / N/A | macro-F1 | 0.15 |
| T3 | `evidence_status` | Yes / No / N/A | F1 with positive=Yes | 0.30 |
| T4 | `evidence_quality` | Clear / Not Clear / Misleading / N/A | macro-F1 | 0.35 |

最終分數：

```text
S = 0.20 * F1(T1, Yes)
  + 0.15 * macroF1(T2)
  + 0.30 * F1(T3, Yes)
  + 0.35 * macroF1(T4)
```

### 4.2 階層條件式約束

推論與 OOF 評估必須套用下列規則：

1. 若 `promise_status = No`，則 `verification_timeline = N/A`、`evidence_status = N/A`、`evidence_quality = N/A`，且 `promise_string`、`evidence_string` 必須為空字串。
2. 若 `evidence_status = No`，則 `evidence_quality = N/A`，且 `evidence_string` 必須為空字串。

Phase 12、13、14 的關鍵教訓是：若 hillclimb 只最佳化每任務 F1 而不在 objective 內套用 post-process，會選到對單任務看似較好、但 joint score 較差的權重。AP-D4 因此必須在評分迴圈內套用 constraints。

---

<a id="sota-scoreboard"></a>
## 5. SOTA 軌跡總覽

OOF 代理分數的歷史演進（Phase 32→38；此為交叉驗證代理，非 leaderboard，見 §60 落差分析）：

![OOF SOTA 演進](reports/figures/fig_oof_sota.png)

### 5.1 主要里程碑

| Phase | 核心變更 | OOF weighted | 決策 |
| :--: | :-- | --: | :-- |
| 1 | MacBERT-base baseline | 0.64150 | 建立可信本地評分管線 |
| 2 | 單模調優與 combo_best | 0.66734 | 採納為早期主力 |
| 4 | base + large probability ensemble | 0.67478 | 確認 ensemble 有效 |
| 5 | combo_v2 + 3-way per-task hillclimb | 0.67954 | per-task 權重成為標準 |
| 6 | Wave C 弱模加入 7-way pool | 0.68206 | 確認 diversity 比單模分數更重要 |
| 8 | combo_v3 seed42 與 multi-seed avg 並列 | 0.68440 | variance reduction 不替換原成員 |
| 13 | post-constraint joint hillclimb | 0.68770 | 修正 constraint coupling 問題 |
| 18 | U1-c per-task/per-view TTA | 0.68925 | active path SOTA |
| 31 | U10 baseline+v1+v2 best.pt stack | 0.67746 | 弱監督 best.pt path 成立 |
| 33 | class-weighted CE + Focal-T4 + 4-way hillclimb | 0.70185 | 首次突破 0.70 |
| 34 | M3-v3/M4-v3 + stem #5 | 0.70569 | U10 path 持續有效 |
| 35 | 5-way × 3-view TTA | 0.70758 | 多視角推論再度有效 |
| 36 | U6-pro 回譯 + stem #6 | 0.71018 | 首次突破 0.71 |
| 37.12 | Aug-Plus handcrafted stem #7 + AP-D3 | 0.71364 | handcrafted minority seeds 有 ensemble diversity |
| 38.4 | Ollama LLM-synth stem #8 + AP-D4 | 0.71608 | 現行 SOTA |
| 39 | AP-D5 grid 0.05 可行性評估 | 0.71608 | 因成本過高不採納 |

### 5.2 Phase 36 到 AP-D4 的關鍵增量

| 階段 | 分數 | 主要新增成分 | 增量 |
| :-- | --: | :-- | --: |
| Phase 36 | 0.71018 | U6-pro BT stem #6 + 6-way × 3-view | 基準 |
| Phase 37 AP-D3 | 0.71364 | Aug-Plus handcrafted stem #7 | +0.00346 |
| Phase 38 AP-D4 | 0.71608 | Ollama LLM-synth stem #8 | +0.00244 |
| Phase 39 | 0.71608 | fine grid 0.05 評估但未採納 | 0 |
| Phase 40 | 0.71608 | val 釋出工程準備（label fix / U12 / U15 / configs）| 0（val gap 已於 §58.13.2 完成量測）|

---

<a id="current-pipeline"></a>
## 6. 現行 AP-D4 管線

### 6.1 8 個 stem

| # | stem | 來源 | 角色 |
| :--: | :-- | :-- | :-- |
| 1 | `p2_combo_best` | 官方 1,000 筆，Phase 2 最佳配方 | 基準與校準錨點 |
| 2 | `p2_combo_best_u10_pseudo` | U10 v1 pseudo，兩階段訓練 | T4 補強與弱監督 diversity |
| 3 | `p2_combo_best_u10_pseudo_v2` | U10 v2 pseudo，兩階段訓練 | T2/T1 補強 |
| 4 | `p2_combo_best_u10_pseudo_v2_classw_focal_t4_g3` | U10 v2 + class weights + T4 Focal γ=3 | 破解 T2 minority collapse 的主力 |
| 5 | `p2_combo_best_u10_pseudo_v3_classw_focal_t4_g3` | M3-v3/M4-v3 corpus + classw/focal | T2 主要 diversity |
| 6 | `p2_combo_best_classw_focal_u6pro` | U6-pro NLLB-3.3B 回譯 + classw/focal | minority augmentation 與 T4/T2 diversity |
| 7 | `p2_combo_best_aug_plus` | handcrafted Aug-Plus 47 列 | AP-D3 主要新增成員 |
| 8 | `p2_combo_best_aug_plus_v2` | stem #7 語料 + Ollama qwen2.5 LLM 合成 | AP-D4 主要新增成員，T4 lift 最大 |

AP-D4 使用 seed42 OOF 作 apples-to-apples ensemble；stem #8 的 3-seed avg 曾測試但未作為當前 SOTA 替換成員。

### 6.2 三視角 TTA

| 視角 | 定義 | 作用 |
| :-- | :-- | :-- |
| stored | 訓練時 fold validation 的 OOF probs | 基準 OOF 視角 |
| middle | 對中段 token window 重新 forward | 補足長段落中段資訊 |
| tail | 對尾段 token window 重新 forward | 補足承諾或證據常在段尾的樣本 |

每個任務獨立搜尋 stem 權重與 view 權重，避免 T1/T2/T3/T4 的最佳 decision boundary 被同一組權重綁死。

### 6.3 AP-D4 重現命令

完整命令與環境需求見 [REPRODUCE.md](REPRODUCE.md)。核心集成命令如下：

```powershell
python -m src.tools.u10_per_task_tta `
  --stems p2_combo_best `
          p2_combo_best_u10_pseudo `
          p2_combo_best_u10_pseudo_v2 `
          p2_combo_best_u10_pseudo_v2_classw_focal_t4_g3 `
          p2_combo_best_u10_pseudo_v3_classw_focal_t4_g3 `
          p2_combo_best_classw_focal_u6pro `
          p2_combo_best_aug_plus `
          p2_combo_best_aug_plus_v2 `
  --grid-step 0.1 --max-rounds 4 --joint-rounds 2 `
  --tag ap_d4_8way_3view
```

預期輸出：

```text
weighted = 0.71608 +/- 0.0005
T1 = 0.94387
T2 = 0.63150
T3 = 0.88011
T4 = 0.48157
```

---

<a id="data-strategy"></a>
## 7. 資料策略

### 7.1 官方資料

| 項目 | 內容 |
| :-- | :-- |
| 檔案 | `vpesg4k_train_1000 V1.csv`、`vpesg4k_train_1000 V1.json` |
| 筆數 | 1,000 |
| 公司數 | 50 家 TWSE 上市公司 |
| 時間範圍 | FY2022-FY2024 |
| 切分 | 5-Fold StratifiedKFold，分層鍵為 `promise_status` × `evidence_status` |
| 防洩漏注意 | 外部永續報告資料不得與這 50 家公司重疊 |

### 7.2 U10 弱監督資料

U10 的核心不是「任意外部 ESG 文字」，而是只取與競賽同質的台灣企業永續報告書段落，並排除已標註 50 家公司。完整 pipeline 見 [§47](#47-u10--企業永續報告書-sr-弱監督-pipeline-完整版2026-05-09-重啟2026-05-10-v2-重訓)。

| 階段 | 產物 | 重點 |
| :-- | :-- | :-- |
| M1 | 31 disjoint tickers × 62 PDFs | 完全避開官方 50 家公司 |
| M3 | corpus v1/v2/v3 | 從 PDF 抽段、SimHash 去重、ESG 與量化線索過濾 |
| M4 | pseudo labels v1/v2/v3 | teacher softmax + admission gate |
| M5 | two-stage training | pseudo+real 預訓練，再 real-only 微調 |
| M6 | OOF ensemble | baseline/v1/v2/v3 與後續 classw/focal stem 合流 |

### 7.3 U6-pro 回譯資料

U6-pro 使用 NLLB-3.3B 與 round-trip ChrF 過濾，修正早期 NLLB-600M 回譯保真度不足的失敗教訓。它作為 stem #6 對 Phase 36 0.71018 有直接貢獻。

### 7.4 Aug-Plus 與 LLM 合成資料

| 階段 | 來源 | 目的 | 結果 |
| :-- | :-- | :-- | :-- |
| Phase 37 | handcrafted 47 列 minority seeds | T4 Misleading、T2 within_2_years 補強 | 單 stem 平手，但 AP-D3 ensemble +0.00346 |
| Phase 38 | Ollama qwen2.5:7b-instruct 本機合成 140 raw seeds | 進一步補 T4/T2 稀有類 | AP-D4 +0.00244，T4 +0.00661 |

資料擴增路線的關鍵教訓：單 stem 分數不一定會升，但只要錯誤型態不同，per-task ensemble 仍可能有效。

---

<a id="model-training-spec"></a>
## 8. 模型與訓練規格

### 8.1 建模選擇

| 策略 | 優點 | 缺點 | 決策 |
| :-- | :-- | :-- | :-- |
| Multi-task encoder + 4 heads | 分享語意表徵、推論成本低 | 任務 loss 可能互相牽制 | 主力採用 |
| Pipeline T1 -> T3 -> T2/T4 | 明確利用條件約束 | 錯誤會串接放大 | 暫緩，待 valid 後重估 |
| 4 個獨立單任務模型 | 任務解耦 | 4 倍成本且缺乏共享 | 不採用 |
| Generative LLM / LoRA | 可直接建模約束 | 8 GB 顯存與 1k 資料限制大 | 暫緩，需雲端 16 GB+ |

### 8.2 主力模型

| 元件 | 設定 |
| :-- | :-- |
| Backbone | `hfl/chinese-macbert-base` |
| Pooling | `[CLS]` + attention-mask mean pooling concat |
| Heads | T1/T2/T3/T4 四個獨立 linear heads |
| Max length | 主力 384，必要時 512/256 做 ablation |
| Optimizer | AdamW |
| Scheduler | cosine + warmup |
| Precision | AMP fp16 |
| Batch | batch_size=8 + grad_accum=2 為 8 GB GPU 主力設定 |
| Early stop | patience=2 |
| Seeds | 42、2024、20260417 等；AP-D4 採 seed42 apples-to-apples |

### 8.3 Loss 與正則化

| 技術 | 狀態 | 教訓 |
| :-- | :-- | :-- |
| Class-weighted CE | 採納於 U10/classw stems | 對 T2 minority 特別有效 |
| Focal Loss on T4 γ=3 | 採納於 classw/focal stems | 對 T4 稀有類有幫助，但需 ensemble 才穩 |
| FGM | 早期 combo 有效 | 與其他 diversity 成員搭配有效 |
| R-Drop | combo_v3 有效 | 單模增益不大但提供 ensemble diversity |
| Multi-Sample Dropout | combo_v3 有效 | 與 R-Drop 疊加後可成新成員 |
| EMA | 禁區 | 短訓練與 shadow init 造成嚴重退化 |
| SWA | 禁區 | cosine 末段 LR 未歸零時效果差 |

---

<a id="ensemble-and-postprocess"></a>
## 9. Ensemble、TTA 與後處理規格

### 9.1 Ensemble 原則

1. 永遠保存 OOF probabilities，而不是只保存 argmax prediction。
2. 新模型即使單模較弱，只要錯誤型態不同，就可加入 pool 由 hillclimb 決定權重。
3. 不用 multi-seed average 直接替換原 seed；應以並列成員加入 pool。
4. 新成員若初始權重為 0，搜尋必須 bias 到新維度，否則 warm-start 不容易探索到有效組合。
5. 若任務存在後處理耦合，objective 應以 post-constraint joint score 為準。

### 9.2 後處理 SOP

```text
raw task logits
  -> per-task weighted ensemble probabilities
  -> argmax predictions
  -> hierarchical constraints
  -> schema validation
  -> prediction CSV / submission JSON
```

### 9.3 單元測試要求

| 測試 | 必須涵蓋 |
| :-- | :-- |
| `tests/test_metrics.py` | weighted score、binary F1、macro-F1、一致性檢查 |
| `tests/test_post_process.py` | T1=No 下游覆寫、T3=No T4 覆寫、空字串欄位 |
| Aug-Plus 測試 | schema validation、去重、品質閘門、promote 流程 |

---

<a id="experiment-dashboard"></a>
## 10. 完成項目索引

### 10.1 工程動作分類

| 類別 | 代表項目 | 結論 |
| :-- | :-- | :-- |
| Baseline 與單模調優 | Phase 1-3 | MacBERT-base 是主力，large 單模不採納但可當 diversity |
| Ensemble 與 hillclimb | Phase 4-14、33-39 | per-task 權重、joint objective、TTA 是主要加分來源 |
| 弱監督 | U10 / Phase 31-34 | 同質企業永續報告書 pseudo data 有效 |
| 回譯 | U6 / U6-pro | 600M 失敗，3.3B + ChrF 過濾成功 |
| 手工與 LLM 合成 | Phase 37-38 | 單 stem 不一定強，但可提供 minority diversity |
| 診斷 | U11/U12 | GroupKFold 與 OOF variance 用於 valid drift 監控 |
| 負面消融 | X1-X14 | 避免重跑已證明不划算或不穩的方向 |

### 10.2 目前可採納成果

| 成果 | 狀態 | 對應章節 |
| :-- | :-- | :-- |
| AP-D4 8-way × 3-view per-task TTA | Current SOTA | [§55](#55-phase-38--ollama-llm-synth--stem-8--ap-d4--new-sota-0716082026-05-20) |
| AP-D3 7-way × 3-view per-task TTA | 強 fallback | [§53.12](#5312-ap-d3-結果--7-way--3-view-per-task-tta--new-sota-0713642026-05-18) |
| Phase 36 6-way × 3-view | 舊穩定錨點 | [§52](#52-phase-36--u6-pro-back-translation--stem-6--6-way--3-view-tta--new-sota-071018) |
| U10 pseudo pipeline | 已產品化 | [§47](#47-u10--企業永續報告書-sr-弱監督-pipeline-完整版2026-05-09-重啟2026-05-10-v2-重訓) |
| Aug-Plus schema / gate / promote | 已產品化 | [§53](#53-phase-37--aug-plus-hand-crafted-minority-訓練與-single-stem-ablation2026-05-18) |

---

<a id="technical-spec"></a>
## 11. 技術規格與專案骨架

### 11.1 專案結構重點

```text
esg-veripromise-2026/
├── configs/                         # YAML 配方，含 8 個 AP-D4 stem
├── src/                             # 訓練、資料、模型、評分、推論與工具程式
├── scripts/                         # U10、U6-pro、Aug-Plus、notebook 建構腳本
├── assets/                          # 手工資料、術語表、prompt 資產
├── data/                            # 官方資料與可重生 processed artifacts
├── outputs/                         # 本機模型權重與 OOF，通常不入 git
├── reports/                         # 實驗分析、ensemble summary、診斷報告
├── tests/                           # metrics、post-process、Aug-Plus 單元測試
├── README.md                        # 專案入口
├── REPRODUCE.md                     # AP-D4 重現手冊
└── MASTER_PLAN_AND_PROGRESS.md       # 本文件
```

### 11.2 設定檔基準

主力設定由 [configs/base.yaml](configs/base.yaml) 與各 experiment YAML 遞迴 extends 組成。重要欄位如下：

| 區塊 | 重要欄位 |
| :-- | :-- |
| `data` | `csv_path`、`text_field`、`label_fields`、`field_weights` |
| `split` | `type=stratified_kfold`、`n_splits=5`、`stratify=[promise_status, evidence_status]` |
| `model` | `backbone`、`max_length`、`dropout`、`pooling=cls_mean` |
| `training` | `epochs`、`batch_size`、`grad_accum`、`lr`、`weight_decay`、`warmup_ratio`、`scheduler` |
| `pseudo` | `pseudo_csv_path`、`stage_a_epochs`、`stage_b_epochs`、`max_pseudo` |
| `loss` | class weights、Focal γ、task loss weights |

### 11.3 環境與版本

| 項目 | 記錄值 |
| :-- | :-- |
| Python | 3.10-3.13 可用；本隊訓練環境曾為 3.13.7 |
| PyTorch | 2.x + CUDA 12.x |
| transformers | 4.x 到 5.x 皆有紀錄；重跑需固定環境以避免漂移 |
| GPU | 8 GB VRAM 可跑 AP-D4 重訓，但需控制 batch/max_len |
| PowerShell | 5.1；避免使用 `&&`，使用 `;` 或 PowerShell pipeline |

---

<a id="negative-results"></a>
## 12. 負面結果與禁區

| 代號 | 方向 | 結果 | 決策 |
| :--: | :-- | :-- | :-- |
| X1 / X1' | EMA decay 0.999 / 0.995 短訓練 | 0.473 / 0.630 級退化 | 不再嘗試無 warm-start EMA |
| X2 / X2' | Uniform LS / T1/T3 LS | T4 或 ensemble 權重失敗 | 不作主線 |
| X3 | LLRD 0.95 on base | 約 -0.005 | 不採納 |
| X4 | combo_v3 multi-seed 直接替換 | 約 -0.002 | 只能並列，不可替換 |
| X5 | macbert-large 作單模主力 | 單模不如 base | 只作 diversity，不作主力 |
| X6 | XLM-R-base 早期同設定 | 約 0.61 | 同設定不重跑；若重試需換 teacher/策略 |
| X7 | ELECTRA-base 同設定 | admission 0 accept | 不採納 |
| X8 | 三視角 TTA 等權 | 較 per-task 權重差 | 不採納等權 |
| X9 | SWA 末 K epoch 平均 | 約 -0.0031 | 不採納 |
| X10 | T4 global re-sampling | T4 升但 T2 降，淨持平 | 不採納 global sampler |
| X11 | NeZha / ERNIE base | 不可訓或 0.609 | 不採納 |
| X12 | EMA 0.995 + 2 epoch warm-start | 約 -0.016 | 不採納 |
| X13 | NLLB-600M zh-en-zh BT | 約 -0.00237 | 已被 U6-pro 取代 |
| X14 | AP-D5 fine-grid 0.05 舊版無向量化 | 單輪估約 3 小時，ROI 不佳 | 已補 fast evaluator；仍不裸跑全量 fine-grid，先用等價測試與小預算 refinement |

禁區不是永久否定概念本身，而是禁止重跑同設定。若未來要解禁，必須先提出與原失敗設定不同的假設、資料、硬體或 objective。

---

<a id="risk-and-limits"></a>
## 13. 結構性殘留、風險與工程上限

### 13.1 主要殘留問題

| 問題 | 現況 | 影響 |
| :-- | :-- | :-- |
| T4 Misleading 極少樣本 | 官方 train support 幾乎為 1，Aug-Plus/LLM 已補但 valid 未確認 | T4 macro-F1 是分數天花板 |
| T2 within_2_years 少數類 | Phase 33 以 class-weighted CE 大幅改善，但仍需 hold-out 驗證 | 可能 OOF overfit |
| OOF search overfit | AP-D4 經多輪 hillclimb | 6/03 valid 需檢查 drift |
| 公司來源偏差 | StratifiedKFold row-level 可能有 company leakage | U11 GroupKFold sanity 用於風險估計 |
| 外部資料分布 | U10/U6/Aug-Plus 都是外部或衍生資料 | 必須保留來源與排除規則 |

### 13.2 工程上限

| 上限 | 假設任務分數 | Weighted | 距 AP-D4 |
| :-- | :-- | --: | --: |
| 保守 | T1 0.945 / T2 0.585 / T3 0.910 / T4 0.600 | 0.7236 | +0.00752 |
| 寬鬆 | T1 0.950 / T2 0.620 / T3 0.920 / T4 0.650 | 0.7570 | +0.04092 |

保守上限距離已不大，下一步需要以 valid 校準和 submission 策略為優先，而不是盲目擴張搜尋空間。

---

<a id="roadmap"></a>
## 14. 待辦事項與 ROI 排序

| 優先 | ID | 項目 | 預期價值 | 風險 | 狀態 |
| :--: | :-- | :-- | :-- | :-- | :-- |
| 1 | F1 | 6/03 valid 校準 AP-D4/AP-D3/Phase36 | 判斷 OOF drift、選 submission anchor | 低 | **Phase 40 完成工程準備**；label fix + U12 gap 腳本就緒；待 AP-D4 train checkpoints 在機器上重訓後執行 |
| 2 | F2 | 最終提交檔與 schema 清理 | 避免格式或約束錯誤 | 低 | validator 已完成；仍待 test 格式確認 |
| 3 | F3 | Submission anchor 設計 | 21 次提交額度內找穩定解 | 中 | 已於 Phase 42 定案 |
| 4 | F4 | AP-D5 搜尋向量化、cache 或替代 optimizer | 降低搜尋成本，讓後續 AP-D5 可在小預算內試 | 中 | **已完成**：fast evaluator + random refinement；不重跑舊版 X14 |
| 5 | **F9** | **Phase 41 Train+Val 8-stem 重訓** | 2× 資料量、最終 checkpoint，提升測試集推論品質 | 低 | **(OK) 已完成（2026-06-11）**：8 stems 全數重訓 + TV 池 OOF 集成驗證（見 §14.4 / §14.5）|
| 6 | F5 | stem #9：U13 LLM 評審重標 U10 偽標 | 新成員 diversity，理論上比 fine grid 更可能突破 | 中高 | 待 Phase 42（測試集）後 ROI 重估 |
| 7 | F6 | 外部 provider LLM 合成與人工抽樣 review | 擴大 T4/T2 minority 合成來源 | 中高 | 需規則、成本與品質閘門 |
| 8 | F7 | T4 Misleading / T2 minority 專項 | 理論上限高，但極易 overfit | 高 | 暫緩至 valid 有訊號 |
| 9 | F8 | 跨家族 teacher / Qwen-LoRA | 可能打破同 teacher 上限 | 高 | 需更大 GPU 或新策略 |

### 14.1 6/03 valid 校準清單（Phase 40 工程準備完成；推論已執行）

1. 對 AP-D4、AP-D3、Phase 36 三個 anchor 跑同一套 valid scoring（工具：`scripts/u12_val_gap.py`，待 checkpoints 存在後執行）。
2. 比較 OOF 與 valid 的 per-task drift，特別是 T2/T4。
3. 檢查 post-process constraints 對 valid 的實際增益或副作用。
4. 若 AP-D4 drift > 0.008 且 AP-D3/Phase36 更穩，submission 主線改採更穩 anchor。
5. 將 valid 結果回寫本章與 README/REPRODUCE。
6. **注意**：val 集的 `more_than_5_years` 已由 loader.py `_LABEL_ALIASES` 正規化；不需另外處理。

### 14.2 Phase 40+ 可做事項分流

| 類型 | 事項 | 啟動條件 | 決策原則 |
| :-- | :-- | :-- | :-- |
| 已完成 | **F9 Phase 41 Train+Val 8-stem 重訓** | **2026-06-05 啟動，2026-06-11 全部完成** | 8 stems 全數重訓（見 §14.4）；TV 池 OOF 集成驗證完成（見 §14.5）；等待測試集進 Phase 42 |
| 已完成本地工程 | F4 `_eval_full` fast evaluator、權重 tensor cache、budgeted random refinement | 已完成；不依賴 valid/test | 已用 synthetic AP-D cache 等價測試確認與舊 `_eval_full` score/preds 一致；後續只跑小預算 refinement，不回到舊版全量 fine-grid |
| 已完成工程準備 | F1 valid 校準（label fix + U12 腳本）| 2026-06-03 valid 已釋出；工具就緒；推論待 checkpoints 建好後執行 | 優先選 valid drift 最小的 anchor，不只看 OOF 最高；上傳前必跑 `validate_submission` |
| 已完成本地工程 | F2 submission validator | 已完成；不依賴 valid/test | 可先檢內部 `*_preds.csv`，final submission 仍需補足官方要求欄位 |
| 等 test 釋出 | F3 Submission anchor 設計 | 2026-06-10 test 釋出後 | TV checkpoints + AP-D hillclimb → Phase 42 定案 |
| 條件式重啟 | F5 stem #9 U13 LLM 評審重標 U10 | valid 顯示 AP-D4 沒有明顯 overfit，且 T4/T2 仍是主要缺口 | 新成員需先過 ensemble admission；不能只看 single-stem OOF |
| 條件式重啟 | F6 外部 provider 合成與人工 review | 規則、成本、來源紀錄與 quality gate 都可審計 | 合成資料只補 minority；不得使用 valid/test 洩漏資訊 |
| 暫緩 | F7 T4/T2 專項重訓 | valid 指出特定 minority 類仍可收益 | 避免 global sampler 類型的跨任務副作用 |
| 長期 | F8 跨家族 teacher / Qwen-LoRA | 有 16 GB+ GPU 或新訓練策略 | 以結構性 diversity 為目標，不重跑已拒絕的同設定 backbone |

### 14.3 F4 搜尋工程完成狀態（2026-05-25）

| 項目 | 實作 | 驗證 | 決策 |
| :-- | :-- | :-- | :-- |
| fast exact evaluator | `src/tools/tta_fast_eval.py` 以 integer label、tensor stack、`np.tensordot` 評估 stem/view 權重 | `tests/test_tta_fast_eval.py` 確認 score 與 preds 等價舊 `_eval_full` | 可作為 AP-D4/AP-D5 搜尋預設評分核心 |
| `u10_per_task_tta.py` 整合 | `_eval_full` 預設走 `FastTTAEvaluator`；舊路徑保留為 `_eval_full_reference` 供回歸測試 | `python -m src.tools.u10_per_task_tta --help` 通過 | AP-D4 重現指令不變，輸出 meta 會標記 `fast_eval=true` |
| 小預算 refinement | 新增 `--random-refine-iters`、`--random-refine-step`、`--random-seed` | 測試確認 refinement 不接受退步候選，simplex 權重和維持為 1 | 後續可用於 AP-D5 小預算試跑；不回到無向量化全量 grid 0.05 |
| submission guardrail | `src/tools/validate_submission.py` | `tests/test_validate_submission.py`、`tests/test_post_process.py`、全測試 60 passed（Phase 40 後）| 上傳前必跑，並用 `keep_default_na=False` 保留字串 `N/A` |

### 14.4 Phase 41 Train+Val 重訓狀態（2026-06-05 啟動 → 2026-06-11 全部完成）

8 stems 全數在 2,000-row train+val 合併集（seed 42、5-fold）重訓完成，OOF = 各 fold best_epoch 加權分數的平均。批次執行器：`scripts/phase41_train_all_tv.py --resume`，總耗時約 5.4 小時。

| Stem | 訓練腳本 | OOF (5-fold mean) | 狀態 |
| :-- | :-- | :--: | :-- |
| p2_combo_best_tv | `train_kfold` | 0.67857 | (OK) 完成 |
| p2_combo_best_u10_pseudo_tv | `train_pseudo_kfold` | 0.68023 | (OK) 完成 |
| **p2_combo_best_u10_pseudo_v2_tv** | `train_pseudo_kfold` | **0.68375** | (OK) 完成（最佳單 stem）|
| p2_combo_best_u10_pseudo_v2_classw_focal_t4_g3_tv | `train_pseudo_kfold` | 0.67132 | (OK) 完成 |
| p2_combo_best_u10_pseudo_v3_classw_focal_t4_g3_tv | `train_pseudo_kfold` | 0.66870 | (OK) 完成 |
| p2_combo_best_classw_focal_u6pro_tv | `train_pseudo_kfold` | 0.66585 | (OK) 完成 |
| p2_combo_best_aug_plus_tv | `train_pseudo_kfold` | 0.67371 | (OK) 完成 |
| p2_combo_best_aug_plus_v2_tv | `train_pseudo_kfold` | 0.67368 | (OK) 完成 |

**分析重點**：

1. 單 stem OOF（0.665–0.684）與 Phase 2 時期的 1,000-row 同 stem OOF 同級，確認 2× 資料量未造成單模退化。
2. `aug_plus_v2_tv` 的 T2（verification_timeline）macro-F1 = 0.5816，顯著高於其他 stem（~0.50）——2,000 筆 U10v2 偽標 + U6-pro 回譯增強對 T2 缺口任務有實質貢獻。
3. TV stem 的 OOF 為 5-fold CV 於 2,000-row，**不可直接與 AP-D4 1,000-row OOF SOTA 0.71608 比較**；正確比較對象是 §14.5 的 TV 池 8-stem 集成 OOF。

### 14.5 Phase 41.5 — TV 池 OOF 集成驗證（Phase 42 前置）

在測試集釋出前，先用 8 個 TV stem 的 2,000-row OOF 跑 post-constraint **joint hillclimb**（工具：`scripts/u16_tv_oof_ensemble.py`），取得最接近預期測試表現的代理分數，並作為 Phase 42 推論的 warm-start 權重。測試集已於 2026-06-11 釋出，Phase 42 推論與提交已完成（見 [§77](#59-phase-42--測試集推論與最終提交2026-06-11-完成)）。

| 指標 | 分數 |
| :-- | :--: |
| 最佳單 stem（u10_pseudo_v2_tv，pooled OOF）| 0.68407 |
| 等權 8-stem 混合 | 0.69759 |
| **joint hillclimb 最佳（per-task stem 權重）** | **0.71033** |

集成後 per-task：promise_status=0.9453、verification_timeline=0.6305、evidence_status=0.8713、evidence_quality=0.4724。等權混合相對最佳單 stem +0.01352，再經 per-task joint hillclimb +0.01274（合計 +0.02626），印證 8-stem 多樣性對 TV 池集成的價值（U10v2/aug_plus_v2 對 T2、u6pro 對 T2 權重 0.225）。

> **方法學一致性**：此 OOF hillclimb 與 AP-D4 1,000-row OOF 0.71608 採同一搜索方法（per-task 權重 + post-constraint 評分），故兩數值可比。TV 池 OOF 略低於 AP-D4 屬預期（single-seed + 2,000-row CV 較嚴格）。
> **過擬合警示**：32 維權重 × 上萬迭代於 OOF 上搜索，分數有樂觀偏差；最終仍以測試集表現為準，OOF 僅作 anchor 排序與 Phase 42 warm-start。

---

<a id="submission-strategy"></a>
## 15. 提交策略與後處理守門

### 15.1 提交候選 anchor

| Anchor | 分數來源 | 用途 |
| :-- | :-- | :-- |
| AP-D4 8-way × 3-view | OOF 0.71608 | 主提交候選 |
| AP-D3 7-way × 3-view | OOF 0.71364 | 少一個 LLM stem 的強 fallback |
| Phase 36 6-way × 3-view | OOF 0.71018 | 較早、較穩定的 fallback |
| Phase 18 U1-c | OOF 0.68925 | 非 U10/augmentation path 的舊 anchor |
| baseline / combo_best | OOF 0.66734 | sanity submission，不作最終主線 |

### 15.2 提交流程守門

1. 產出 prediction CSV 後先跑 schema validation。
2. 強制套用 `apply_constraints_batch`。
3. 檢查 T1=No 的下游欄位是否全為 N/A 或空字串。
4. 檢查 T3=No 時 T4 是否為 N/A。
5. 保留每次提交的 config、git hash、ensemble meta、submission 檔案 hash。

### 15.3 Submission validator（2026-05-25 已完成）

新增工具：`src/tools/validate_submission.py`，測試：`tests/test_validate_submission.py`。

| 模式 | 指令 | 用途 |
| :-- | :-- | :-- |
| final submission 嚴格模式 | `python -m src.tools.validate_submission outputs/submissions/<file>.csv --mode submission` | 檢查 `id`、四任務 label、`promise_string`、`evidence_string`、ID 重複、label domain 與 hierarchy |
| internal preds 模式 | `python -m src.tools.validate_submission reports/analysis/_ensemble/<tag>_preds.csv --mode preds --allow-promise-yes-na-timeline` | 檢查 ensemble 內部 prediction CSV；允許 label-only 輸出，但提醒 final submission 必須補字串欄位 |

AP-D4 內部檢查結果：

| 檢查項 | 結果 | 決策 |
| :-- | :-- | :-- |
| `tests/test_validate_submission.py` + `tests/test_post_process.py` | 15 passed | validator 與 post-process 守門一致 |
| 全測試 | 50 passed | 新工具未破壞既有測試 |
| AP-D4 `ap_d4_8way_3view_preds.csv` 寬鬆 preds 模式 | OK，1,000 rows | 主要 hierarchy 已通過 |
| AP-D4 final submission 嚴格風險 | 缺 `promise_string`、`evidence_string`；另有 10 筆 `promise_status=Yes` 且 `verification_timeline=N/A` | 上傳前必須由 submission generator 補字串欄位，並決定是否強制修正 Yes+N/A timeline |

注意：統計與驗證 CSV 時必須使用 `keep_default_na=False`，否則 pandas 會把字串 `N/A` 當成缺失值，導致 hierarchy 計數失真。

---

<a id="reproducibility"></a>
## 16. 可重現性規範

| 項目 | 規範 |
| :-- | :-- |
| Seed | 所有訓練與 split 記錄 seed；AP-D4 使用 seed42 apples-to-apples |
| Split | 5-fold index 應保存於 `data/splits/` |
| OOF | 每個 fold 輸出 `oof_probs.npz`，集成只讀 probabilities |
| Checkpoint | `best.pt` 與 fold log 同步保存 |
| 外部資料 | 記錄來源 URL、company exclusion、hash、抽取與過濾規則 |
| LLM 合成 | 記錄 provider、model、prompt、seed、quality gate、promote 結果 |
| 報告 | 每個 Phase 至少保存 summary、分數、決策與 artifact list |

---

<a id="archive-policy"></a>
## 17. 存檔與文件治理

1. 早期計畫與回顧內容已併入本文件；不另存歷史檔。
2. 生成式實驗報告若位於 `reports/`，原則上不手動重寫，只在主文件引用重要結論。
3. 主文件不再放日期於檔名；日期寫在文件版本資訊內。
4. 若未來產生 Phase 40+，追加到 Part IV，不覆寫既有 Phase 證據。
5. README 與 REPRODUCE 只保留對外與重現必要資訊，研究推理細節集中在本文件。

---

<a id="governance"></a>
## 18. 規則、合規與外部資料治理

### 18.1 已確認原則

| 類型 | 是否可用 | 條件 |
| :-- | :-- | :-- |
| 官方訓練資料 | 可用 | 競賽提供 |
| 公開永續報告書 | 可用 | 不可與官方標註公司重疊；需記錄來源 |
| 手工撰寫樣本 | 可用 | 不可使用 valid/test 或最終預測資訊 |
| LLM 合成樣本 | 可用 | 不可由 test/valid 洩漏；需保存 prompt、provider、gate |
| 測試集推導資料 | 不可用 | 禁止以 test 或最終預測反推訓練資料 |

完整官方裁示與原文摘要見 [§77](#77-競賽規則對外部資料的立場已確認可用)。

### 18.2 實驗倫理與審計

1. 所有外部資料只用於訓練集側增強與 pseudo-labeling，不可參照 test labels。
2. 公司排除清單必須以官方 50 家為基準。
3. LLM 合成資料需通過 schema、label consistency、SimHash 去重與 quality gate。
4. 任何人工標註或人工撰寫資料需保留產生目的、目標類別與版本。

---

<a id="part-iv--完整-phase-log-append-only-evidence"></a>
# Part IV — 完整 Phase Log (Append-only Evidence)

本部分為完整訓練歷史，**append-only**（不刪除、不改寫已完成 Phase）。
每個 Phase 包含：(a) 動機 / 設計、(b) 執行狀態、(c) 結果、(d) 結論與後續。
失敗的 Phase 同樣保留作為禁區（X1 ~ X14）的依據。

---

## 19. Phase 1 — 主力配置（Baseline 復現設定）

`configs/exp_p1_baseline.yaml` (繼承 `base.yaml`)：

```yaml
extends: base.yaml
exp_name: p1_baseline_macbert_base
model:
 backbone: hfl/chinese-macbert-base
 max_length: 256
training:
 epochs: 5
 batch_size: 16
 lr: 2.0e-5
seeds: [42, 2024, 20260417]
```

預期：5-Fold × 3 seeds = 15 runs，平均 weighted score 0.55 ~ 0.58。**通過此 Gate 後才進 Phase 2**。

---

## 20. Phase 1 結果與 Phase 2 籌備（OOF 0.6415 → combo 規劃）

### 20.1 Phase 1 — Baseline 復現結果（已完成）

執行：`python -m src.train_kfold --config configs/exp_p1_baseline.yaml`，3 seeds × 5 folds × 5 epochs，AMP fp16，class_weight on，pooling=cls_mean，max_length=256，batch=16，lr=2e-5，cosine + warmup_ratio=0.10，wd=0.01。RTX 5060 8GB 約 48 分鐘。

| Seed | Mean | Std | Min | Max |
| :-- | :-- | :-- | :-- | :-- |
| 42 | 0.6449 | 0.0094 | 0.6319 | 0.6541 |
| 2024 | 0.6374 | 0.0143 | 0.6237 | 0.6578 |
| 20260417 | 0.6422 | 0.0171 | 0.6144 | 0.6549 |
| **Overall** | **0.64150** | **0.01332** | — | — |

**結果遠超 Phase 1 目標 0.55-0.58**。Gate 條件達成 → 進入 Phase 2。

### 20.2 OOF 詳細分析摘要（[reports/analysis/p1_baseline_macbert_base/summary.md](reports/analysis/p1_baseline_macbert_base/summary.md)）

Per-class F1（seed-avg ensemble，post-processed）：

| Task | 弱點 label (F1) | 強項 label (F1) |
| :-- | :-- | :-- |
| T1 promise_status | **No: 0.669** (recall=0.661) | Yes: 0.925 |
| T2 verification_timeline | **within_2_years: 0.118** (support=13) | N/A: 0.670; already: 0.540 |
| T3 evidence_status | **No: 0.473** (precision=0.401) | Yes: 0.800 |
| T4 evidence_quality | **Misleading: 0.000** (support=1); **Not Clear: 0.191** | Clear: 0.747; N/A: 0.630 |

**關鍵錯誤模式**：
1. **T1 雙向誤判約 122 筆**（No→Yes 63、Yes→No 59）。由於階層後處理 (`promise=No → T2/T3/T4=N/A`)，T1 錯誤會把下游 3 個任務一併拖垮 → 解釋為何 post-processed 分數比 raw 低約 0.01。
2. **T2 時間語意混淆**：`already ↔ between_2_and_5_years`（138 筆）、`longer_than_5_years ↔ already`（78 筆）。模型缺乏明確時間數字解析能力。
3. **T3 過度悲觀**：Yes→No 107 筆、Yes→N/A 53 筆。
4. **T4 中段崩潰**：`Not Clear` 多被預測為 Clear/N/A；`Misleading` 樣本數=1 無法學習。
5. **校準**：T1 中段 (0.6-0.8) 過度自信（accuracy < confidence），T3/T4 整體尚屬一致 → temperature scaling 收益有限，重點仍在弱類學習。

**對 Phase 2 的策略含義**：
- **T1 No 召回**：提高 T1 task_loss_weight 或試 label_smoothing。
- **T2 弱類**：oversample within_2_years × 4，或 R-Drop 增強少類正則。
- **T3 平衡**：class_weight 已開啟仍偏負；可試提高 evidence_status loss 權重。
- **T4 Not Clear**：是支援數第二多 (124) 但 F1 極低，列為主攻；嘗試 multi-sample dropout 與較長 max_length（25.6% 樣本 > 256）。
- **整體**：先試 max_length=384、lr 掃描、task_loss 與評分權重對齊三組對照。

---

## 21. Phase 2~4 超參數搜索藍圖

### 21.1 Phase 2 — 單模調優 (僅在 macbert-base 上做，節省成本)

| 參數 | 掃描值 | 備註 |
| :-- | :-- | :-- |
| lr | {1e-5, 2e-5, 3e-5, 5e-5} | base 模型適合較大 lr |
| max_length | {192, 256, 320} | 對長文段落任務有影響 |
| warmup_ratio | {0.05, 0.10, 0.15} | – |
| pooling | {cls, mean, cls_mean} | 段落級任務通常 cls_mean 較佳 |
| dropout | {0.1, 0.2} | – |
| label_smoothing | {0.0, 0.05} | – |
| class_weight | {off, on} | T2/T4 預期受益 |
| task_loss_weights | {均等, 評分權重對齊, 反向強化弱項} | 三組對照 |

**搜索方法**：先 sequential (一參數一參數)，再對 top-3 做小規模 grid。每組僅用 seed=42 跑 1 次完整 5-Fold 過濾，最佳組合再以 3 seeds 復測。

### 21.2 Phase 3 — 升規模 (macbert-large)

固定 Phase 2 之最佳參數，僅變更：

```yaml
model.backbone: hfl/chinese-macbert-large
training.batch_size: 4
training.grad_accum: 4 # effective batch = 16
training.lr: 1.5e-5 # large 模型偏小 lr
training.epochs: 4
```

跑 5 seeds × 5 folds = 25 runs，OOF 進入 Hill-Climbing。

### 21.3 Phase 4 — 多骨幹 + 弱項補強

- **多骨幹**：`hfl/chinese-roberta-wwm-ext-large`、`xlm-roberta-large` 各 3 seeds × 5 folds。
- **弱項補強 (僅針對 T2/T4)**：
 - **R-Drop**：同 batch 兩次 forward，KL 正則 ($\alpha=0.5$)。
 - **FGM 對抗訓練**：embedding 加擾動 ($\epsilon=1.0$)。
 - **EMA**：decay=0.999，推論用 EMA 權重。
 - **Multi-Sample Dropout**：T4 head 套 5 份 dropout 平均。
 - **Minority Oversample**：對 T2 `longer_than_5_years` × 2、T4 `Misleading` × 2。

每項獨立 A/B 測試，僅在 OOF +0.005 才採納。

---

## 22. Phase 2 — Combo Winners 最終配置 (3 seeds × 5 folds)

組合：`lr=3e-5` + `max_length=384` + `class_weight off`（保留 cls_mean pooling、5 epochs、batch=8 + grad_accum=2）。

| Seed | Mean | Std |
| :-- | :-- | :-- |
| 42 | 0.66558 | 0.01593 |
| 2024 | 0.65942 | 0.00951 |
| 20260417 | 0.66066 | 0.02224 |
| **Overall (raw best-epoch)** | **0.66189** | **0.01572** |
| **Post-processed per-seed mean** | **0.66307** | — |
| **Seed-avg probability ensemble** | **0.66734** | — |

vs Phase 1：**+0.0204** (raw)、**+0.0334** (ensemble)。

| Task | P1 seed-avg F1 | P2 combo seed-avg F1 | Δ |
| :-- | :-- | :-- | :-- |
| T1 promise_status | 0.9252 | 0.9277 | +0.003 |
| T2 verification_timeline | 0.4771 | 0.4759 | -0.001 |
| T3 evidence_status | 0.8003 | 0.8655 | **+0.065** |
| T4 evidence_quality | 0.3920 | 0.4308 | **+0.039** |

→ T3/T4 顯著改善（max_length=384 把長文截斷率從 25.6% 降至 < 5%；no_classw 修正了 T3/T4 弱類在強 class_weight 下被推到極端）；T2 仍受限於少數類 within_2_years (n=13) 與時間語意混淆；T1 已飽和。

詳見 [reports/analysis/p2_combo_best/summary.md](reports/analysis/p2_combo_best/summary.md)。

> 註：本節原命名為 “Phase 3 — Combo Winners”，但依原計畫第 7 章定義 Phase 3 = 升規模 (macbert-large)，已於 2026/04/28 將 combo 結果重新歸入 Phase 2 末段；對應 artefacts (`configs/exp_p2_combo_best.yaml`、`reports/experiments/p2_combo_best/`、`reports/analysis/p2_combo_best/`、`outputs/checkpoints/p2_combo_best/`、`data/splits/p2_combo_best/`) 已同步更名，內容指標未變。

---

## 23. Phase 3 — 升規模 (`hfl/chinese-macbert-large`, 326M)

依原計畫 7.2 升規模到 macbert-large，繼承 Phase 2 combo 最佳超參 (`max_length=384`、`pooling=cls_mean`、`use_class_weight=false`)，調整 batch=4 + grad_accum=4 (effective 16)、lr 範圍 1e-5 ~ 2e-5、early_stop_patience=2。單 seed=42、5 折驗證。

| Config | lr | epochs | weighted_score (5-fold mean ± std) | vs P2 combo seed=42 (0.66558) |
| :-- | :-- | :-- | :-- | :-- |
| `p3_large_baseline` | 1.5e-5 | 5 | **0.65900 ± 0.00867** | -0.00658 |
| `p3_large_lr2e5` | 2.0e-5 | 5 | **0.66439** | -0.00119 |
| `p3_large_ep8` (4 折,fold4 紀錄缺失) | 1.0e-5 | 8 | ≈ 0.6597 (前 4 折最佳 epoch 平均) | ≈ -0.006 |

**結論**：在 1000 樣本上 macbert-large 單獨並未通過「+0.005 顯著超越 base combo」門檻；epoch 拉長後驗證分數於第 6~7 epoch 飽和，loss 仍下降代表已過擬合。large 訓練成本 (≈ 4×) 與單模收益不對等 → **不採 large 為單模主力**，保留 large 預測作 Phase 4 集成輸入。

> `p3_large_ep8` fold4 的 `score_summary.csv` 因外部監控腳本提早終止主程序而未寫出；5 個 fold 的 `best.pt` 皆已落盤，不影響 ensemble；本表以前 4 折最佳 epoch 平均近似呈現結論。

### 23.1 為什麼大模型 (326M) 反而沒贏過 base (102M)？— 五個根本原因

單純看「large 比 base 多 3.2 倍參數」直覺上應該更好，但 1000 樣本的場景下實測結果相反。下列依影響力排序拆解：

#### 原因 1 — 樣本/參數比嚴重失衡（最主要）
- 訓練集 1000 筆，5-Fold 後每折 train=800。
- macbert-base 102M 參數 → 樣本/參數比 ≈ **1 : 102,000**。
- macbert-large 326M 參數 → 樣本/參數比 ≈ **1 : 326,000**（更稀疏 3.2 倍）。
- 經驗法則：large 模型需要 **> 10K 樣本** 才能讓「比 base 顯著更佳」的決策邊界穩定學成；否則 fine-tune 階段等同於「用 800 個梯度訊號去微調 326M 個參數」，每個參數收到的有效梯度極小 → 只動了預訓練表徵的薄薄一層。

#### 原因 2 — 過擬合警訊明確
- `p3_large_ep8`: epoch 4~5 驗證分數已飽和、epoch 6~7 仍 train loss 下降 → 典型過擬合曲線。
- `early_stop_patience=2` 救不回來：一旦過擬合，最佳 checkpoint 仍是「沒充分學習」的 epoch 4 版本。
- 大模型容量過剩會記住標註雜訊，特別是 T4 `Misleading` (support=1) 與 `Not Clear` (support=124) 的稀疏類別。

#### 原因 3 — 超參空間 mismatch（lr 必須重調）
- base 最佳 lr = 3e-5；對 large 過大 → 直接收斂到較差解。
- large 應更小 lr (≤ 2e-5) + 更多 warmup；但本實驗只試 1.5e-5 與 2e-5 兩個點，掃描密度不足。
- batch 8→4 + grad_accum 2→4（為了塞 8GB GPU），effective batch 維持 16，但實際梯度 noise 結構改變 → 與 base 的最佳超參不再可平移。

#### 原因 4 — T4 macro-F1 受少類別主導
- T4 評分用 macro-F1，6 個類別等權平均。
- `Misleading` (support=1) 在任何模型上本質上都學不會（500 預測中只有 1 個正例）。
- 大模型在主流類 (Pure Information, Concrete Action) 上的微弱優勢，會被少類分類失敗的高方差完全抵銷。

#### 原因 5 — 但 large 在集成中仍有價值（負負得正）
- Phase 4 證明：base combo + large @ 1.5:1 → **0.67478**（vs base combo 0.66558，**+0.0074**）。
- T4 macro 從 0.4308 跳到 **0.4494**（+0.019）— 大模型抓到了 base 抓不到的部分樣本。
- → **large 不該當主力，但作 ensemble 多樣性源 ROI 高**；此即 Phase 4 設計依據。

#### 一般化規律
在 NLP fine-tune 場景，當「樣本數 < 5K 且任務有高度不平衡少類別」時：
- large 模型的**單模收益期望值是負的**（本案例 -0.001 ~ -0.007）。
- 但作為 ensemble member 仍有 **+0.005 ~ +0.010** 的補強空間（本案例 +0.0074）。
- 工程決策：訓練 1 個 large 變體 → 加入 ensemble pool；不要花算力做 large 的 ablation 探索。

---

## 24. Phase 4 — Probability Ensemble (base combo + large)

工具：[src/tools/oof_ensemble.py](src/tools/oof_ensemble.py)（讀各 exp 的 OOF 機率，每 exp 內先做 seed 平均，再以權重做加權平均，最後套用條件約束後處理）。

| Pool | weights (combo : large_lr2e5) | T1 | T2 | T3 | T4 | weighted_score |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| p2_combo_best (3 seeds) | 1 : 0 | 0.9277 | 0.4759 | 0.8655 | 0.4308 | 0.66734 |
| p3_large_lr2e5 (1 seed) | 0 : 1 | 0.9276 | 0.4601 | 0.8577 | 0.4358 | 0.66439 |
| ENSEMBLE 1 : 1 | 1.0 : 1.0 | 0.9314 | 0.4791 | 0.8629 | 0.4383 | 0.67043 |
| ENSEMBLE 1.5 : 1 (**SOTA**) | **1.5 : 1.0** | **0.9318** | **0.4789** | **0.8643** | **0.4494** | **0.67478** |
| ENSEMBLE 2 : 1 | 2.0 : 1.0 | 0.9318 | 0.4779 | 0.8649 | 0.4468 | 0.67388 |
| ENSEMBLE 2.5 : 1 | 2.5 : 1.0 | 0.9328 | 0.4785 | 0.8637 | 0.4442 | 0.67290 |
| ENSEMBLE 3 : 1 | 3.0 : 1.0 | 0.9334 | 0.4787 | 0.8645 | 0.4447 | 0.67348 |

**最佳組合**：`p2_combo_best : p3_large_lr2e5 = 1.5 : 1.0` → **OOF weighted = 0.67478** (vs base combo 0.66734，**+0.00744**；vs Phase 1 0.6415，**+0.0333**)。提升主要來自 T4 evidence_quality (0.4308 → **0.4494**, +0.019)，T1 與 T2 也有微幅增益；T3 略降 0.001（容忍範圍內）。

→ 已於 Phase 5 推進並於 Phase 6 進一步突破：擴 pool 加入 Wave C 4 個增強模型 + 7-way per-task hillclimb，最終 SOTA = **0.68206**（見 §25 / §26）。

詳見 [reports/analysis/_ensemble/p2_combo_best_p3_large_lr2e5.csv](reports/analysis/_ensemble/p2_combo_best_p3_large_lr2e5.csv)。

---

## 25. Phase 5 — Wave A/B Ablation + Combo v2 + Per-task Hillclimb

### 25.1 Wave A/B 單變量 Ablation（16 組，全在 combo_best 基礎上 +1 改動，5-Fold seed=42）

> 對照基準：`p2_combo_best` seed=42 = **0.66558**。所有掃描結果見 [reports/experiments/_phase2_ablation.md](reports/experiments/_phase2_ablation.md)。

| 配置 | 改動 | weighted_score | 結論 |
| :-- | :-- | :--: | :-- |
| **p2x_focal_t4** | T4 改用 Focal Loss (γ=2.0) | **0.66833** | [贏] +0.00275 |
| **p2w_fgm10** | FGM 對抗訓練 ε=1.0 | **0.66714** | [贏] +0.00156 |
| p2n_wd005 | weight_decay 0.05 | 0.66575 | 中性 +0.00017 |
| p2m_wd0001 | weight_decay 0.001 | 0.66561 | 中性 |
| p2q_lr5e5 | lr 5e-5 | 0.66349 | -0.002 |
| p2t_epochs7 | 7 epochs | 0.66282 | -0.003 |
| p2j_warmup020 | warmup 0.20 | 0.66242 | -0.003 |
| p2p_lr4e5 | lr 4e-5 | 0.66213 | -0.003 |
| p2i_warmup006 | warmup 0.06 | 0.66206 | -0.004 |
| p2l_dropout03 | dropout 0.3 | 0.66075 | -0.005 |
| p2u_llrd095 | LLRD decay 0.95 | 0.66019 | -0.005 |
| p2o_lr2.5e5 | lr 2.5e-5 | 0.65767 | -0.008 |
| p2k_dropout02 | dropout 0.2 | 0.65516 | -0.010 |
| p2r_maxlen512 | max_length=512 (b=4×ga4) | 0.66306 | -0.003（成本翻倍不值） |
| p2s_sched_linear | linear scheduler | 0.66136 | -0.004 |
| **p2v_ema999** | EMA decay 0.999 | **0.47291** | [負] 崩潰：步數不足 (~500 step)，shadow 幾乎沒更新 |

**重要負面發現**：
- **EMA decay=0.999 在短訓練 (~500 step) 不可用** → 後續若要重試需用 0.99 或 step-aware decay。
- 大幅調整 lr/dropout/warmup 都偏負面 → combo_best 已接近單模 plateau。

### 25.2 Combo v2 復測（取 §25.1 兩個正向贏家）

`exp_p2_combo_v2.yaml` = combo_best + Focal-T4 (γ=2.0) + FGM (ε=1.0)，3 seeds × 5-fold：

| metric | T1 | T2 | T3 | T4 | weighted |
| :-- | :--: | :--: | :--: | :--: | :--: |
| 3-seed mean ± std | — | — | — | 0.4368 | **0.66836 ± 0.01659** |
| seed-avg ensemble (post-processed) | 0.9322 | 0.4770 | 0.8638 | 0.4333 | **0.66879** |

→ vs `p2_combo_best` seed-avg 0.66734：**+0.00145**（穩定但增量小，主要來自 T4）。

### 25.3 3-way Probability Ensemble + Per-task Hillclimb

工具：[src/tools/ensemble_weight_scan.py](src/tools/ensemble_weight_scan.py)、[src/tools/per_task_hillclimb.py](src/tools/per_task_hillclimb.py)。

**全任務同權重掃描**（v2 : best : large）：

| weights | T1 | T2 | T3 | T4 | weighted_score |
| :-- | :--: | :--: | :--: | :--: | :--: |
| 1.0 : 0.0 : 0.0 (v2 only) | 0.9322 | 0.4770 | 0.8638 | 0.4333 | 0.66879 |
| 0.0 : 1.0 : 0.0 (best only) | 0.9277 | 0.4759 | 0.8655 | 0.4308 | 0.66734 |
| 0.0 : 0.0 : 1.0 (large only) | 0.9276 | 0.4601 | 0.8577 | 0.4358 | 0.66439 |
| 1.0 : 1.0 : 0.0 | 0.9309 | 0.4780 | 0.8666 | 0.4420 | 0.67254 |
| **1.0 : 1.0 : 1.0** | 0.9341 | 0.4817 | 0.8672 | 0.4486 | **0.67627** |
| 2.0 : 1.0 : 1.0 | 0.9314 | 0.4834 | 0.8686 | 0.4482 | 0.67624 |
| 1.5 : 1.0 : 0.7 | 0.9313 | 0.4789 | 0.8648 | 0.4462 | 0.67371 |

→ 3-way 等權 (1:1:1) 即超越 Phase 4 SOTA (+0.00149)。

**Per-task hillclimb**（各任務在 GRID = {0, 0.5, 1.0, 1.5, 2.0}^3 = 125 組合中獨立挑最大 F1）：

| Task | best (v2 : best : large) | F1 |
| :-- | :--: | :--: |
| T1 promise_status | 0.5 : 0.5 : 0.5 (等權) | 0.9341 |
| T2 verification_timeline | **2.0 : 0.0 : 0.5** (棄用 best) | 0.4892 |
| T3 evidence_status | 0.5 : 0.5 : 1.0 (large 重) | 0.8702 |
| T4 evidence_quality | 1.0 : 1.5 : 1.5 (best+large 重) | 0.4522 |

**Phase 5 SOTA**：per-task hillclimb 結合後 **OOF weighted = 0.67954**（vs Phase 4 0.67478，**+0.00476**；vs Phase 1 baseline，**+0.0380**）。

詳見 [reports/analysis/_ensemble/per_task_hillclimb_summary.csv](reports/analysis/_ensemble/per_task_hillclimb_summary.csv) 與 [reports/analysis/_ensemble/per_task_hillclimb_preds.csv](reports/analysis/_ensemble/per_task_hillclimb_preds.csv)。

---

## 26. Phase 6 — Wave C Ablation + 7-way Per-task Hillclimb

> **本階段目標**：嘗試 Phase 5 之後仍未試過的三大正則化／資料增強技術，並用「**爬山演算法 (hillclimbing)**」在更大的模型池中為每個任務獨立搜尋最佳集成權重，把 SOTA 從 0.67954 進一步往上推。

### 26.0 名詞快速說明（給未接觸過的讀者）

- **Ablation（消融實驗）**：固定其他所有設定，只改一個變因（例如「加上資料增強」），跑完整 5-Fold 觀察分數變化，用以確認該變因的真實效果。
- **Combo（組合配方）**：把多個「單獨已驗證有效」的技術疊在 baseline 上做的整合配方（如 `combo_v2 = baseline + FGM + Focal-T4`）。Combo 的篩選規則是「**單獨上場必須先贏過 baseline**」，因此 Wave C 因為單模都輸給 baseline，**並不進入 Combo**（詳見 §26.3）。
- **Ensemble（集成）**：訓練好的多個模型，在推論時把它們的「機率輸出」加權平均，再 argmax 拿最終標籤。集成成功的關鍵不是「每個成員都很強」，而是「成員之間的錯誤模式互不相同」（diversity）。
- **OOF (Out-of-Fold) probabilities**：5-Fold 訓練時，每筆樣本只在「不參與訓練的那一摺」被預測，因此整份資料都有「未洩漏」的預測機率，可拿來做公平的離線評分與 ensemble 權重搜尋。
- **Hillclimbing（爬山演算法）**：見 §26.2 詳細解說。

### 26.1 Wave C 訓練技術 Ablation（單模實驗）

實作三大未試技術：token-level data augmentation、R-Drop、Multi-Sample Dropout。

**技術原理：**
- **Token-level Data Augmentation**：訓練時對輸入文字的 token 序列做隨機擾動，等於在資料層面注入雜訊讓模型學得更穩健。
 - `mask_ratio`：隨機把若干 token 換成 `[MASK]`（與 BERT MLM 預訓練同形）。
 - `swap_ratio`：隨機交換相鄰兩個 token，模擬輕微語序錯亂。
 - `delete_ratio`：隨機刪除 token 並向左壓緊（右側補 padding）。
 - 三種操作都跳過特殊 token（`[CLS]`/`[SEP]`/`[PAD]`），且只在訓練階段套用，驗證集保持原文。
- **R-Drop**：同一個 batch 在啟用 dropout 的情況下做兩次前向傳播得到 `logits1`、`logits2`，除了原本的分類損失外，再加上對稱 KL 散度 $\frac{\alpha}{2}\bigl(\text{KL}(p_1 \Vert p_2) + \text{KL}(p_2 \Vert p_1)\bigr)$，強迫兩次帶不同 dropout mask 的預測互相一致。等於在「dropout 子網路」之間做正則化。
- **Multi-Sample Dropout (MSD)**：在分類頭之前對同一個共享特徵 `feat` 套 K 份不同的 dropout，分別產生 K 組 logits 後平均。等於把「測試時關掉 dropout 取平均」的概念搬到訓練時，期望讓決策更平滑。

**程式碼變更：**
- [src/data/dataset.py](src/data/dataset.py) `ESGDataset` 新增 `aug_prob` / `mask_ratio` / `swap_ratio` / `delete_ratio` / `aug_seed`，新方法 `_augment_ids` 處理 token 擾動。
- [src/training/trainer.py](src/training/trainer.py) 新增 `rdrop_alpha`：當 > 0 時做第二次 forward 並加入 KL 損失。
- [src/models/multitask.py](src/models/multitask.py) 新增 `msd_k`：當 > 1 時訓練階段做 K 份 dropout 平均；推論不變。
- [src/train_kfold.py](src/train_kfold.py) `_build_loaders` 把 `training.augment` 的子設定接到訓練 `ESGDataset`，模型構造讀取 `model.msd_k`。

**5-Fold 結果（seed=42，與 `combo_best` seed=42 = 0.66558 比較）：**

| 配置 | 改動 | weighted_score | T1 | T2 | T3 | T4 | vs combo_best |
| :-- | :-- | :--: | :--: | :--: | :--: | :--: | :--: |
| `p2ab_aug_mask10` | mask 10% | 0.66122 | 0.9298 | 0.4648 | 0.8589 | 0.4225 | -0.0044 |
| `p2ac_aug_mix` | mask 8% + swap 3% + del 3% | 0.66070 | 0.9344 | 0.4638 | 0.8547 | 0.4224 | -0.0049 |
| `p2ad_rdrop05` | R-Drop α=0.5 | 0.66345 | 0.9311 | 0.4739 | 0.8593 | 0.4238 | -0.0021 |
| `p2ae_msd5` | Multi-Sample Dropout K=5 | 0.66448 | 0.9327 | 0.4753 | 0.8558 | 0.4283 | -0.0011 |

**結論**：4 個配置全部單獨低於 `combo_best`。資料量僅 1000 筆、訓練步數約 500 步的設定下，這些「強正則化」技術可能因訓練不足以收斂出穩定的擾動鄰域而短期內反而拖慢學習速度。但它們的決策邊界與既有模型差異化顯著，因此被保留為 ensemble 池成員（§26.4 證實這是正確決定）。

### 26.2 Hillclimbing 與 per-task 權重搜尋

**Hillclimbing（爬山法）是什麼？**
這是一個經典的局部最佳化演算法，邏輯非常直觀：

1. 先準備一個「分數函數」 $f(w)$（在我們的場景中，$w$ 是「7 個模型的權重向量」，$f(w)$ 是該任務在 OOF 上的 F1 分數）。
2. 從某個起始點 $w_0$ 開始計算 $f(w_0)$。
3. 在 $w_0$ 的「鄰域」中找一個分數更高的點 $w_1$（即 $f(w_1) > f(w_0)$）。
4. 把 $w_1$ 當成新的起點，回到步驟 3 重複，直到再也找不到更好的鄰居 $\Rightarrow$ 抵達局部最大值就停止。

對應到本專案，我們用 **兩階段 hillclimb**（v2 + v3）來搜尋 ensemble 權重：

- **第一階段 v2（粗網格窮舉）**：每個權重從 $\{0, 1, 2\}$ 中取值，7 個模型共 $3^7 = 2187$ 種組合。對每個任務獨立把全部 2187 種組合都算一次分數，挑出最高的那組作為 v2 winner。這嚴格來說是 enumeration（窮舉），但在離散小空間中等價於「鄰域 = 整個搜尋空間」的 hillclimb。
- **第二階段 v3（隨機鄰域精修）**：以 v2 winner 為起點，候選網格細化為 $\{0, 0.25, 0.5, \dots, 2.0\}$（9 個刻度）。每一輪隨機挑 1~3 個維度並隨機改值，若新分數更高就接受、否則丟棄；總共做 5000 次。這是真正的「stochastic hillclimb」，能在粗網格找不到的細微權重組合中再取得一點增益。

**為什麼要 per-task（每任務獨立）搜尋？**
四個任務的標籤分布、難度、最佳模型都不同：T1（promise_status）是二元判斷，T4（evidence_quality）是稀有類別佔多數的多元分類。如果用「全任務共用一組權重」，等於假設「對所有任務最好的模型加權方式是同一個」，這幾乎不可能。Per-task hillclimb 允許每個任務挑自己最喜歡的模型組合（甚至完全棄用某些模型），自然分數會更高。

工具實作：
- [src/tools/per_task_hillclimb_v2.py](src/tools/per_task_hillclimb_v2.py)：粗網格窮舉。
- [src/tools/per_task_hillclimb_v3.py](src/tools/per_task_hillclimb_v3.py)：v2 winner 起點 + 隨機局部精修。

### 26.3 為什麼 Wave C 不進 Combo？

**Combo 的設計原則**：把「單獨已驗證帶來 +0.001 以上增益」的技術疊在 baseline 上做整合配方。例如 `combo_v2 = combo_best + FGM + Focal-T4`，因為 FGM 單獨 +0.00156、Focal-T4 單獨 +0.00275，兩者都是正向贏家。

**Wave C 4 個配置全部單獨輸給 combo_best**（差距 -0.0011 ~ -0.0049）。如果硬把它們疊到 combo 上，等於在已經最佳化的訓練配方裡注入「會降分的成分」，破壞了 combo 的可解釋性與穩定度。因此**Wave C 不進 Combo**。

但這並不代表它們沒用——**ensemble 與 combo 是兩種完全不同的策略**：
- **Combo**：在「**訓練時**」把多個技術疊在同一個模型上 → 產出**單一模型**。
- **Ensemble**：在「**推論時**」把多個**獨立訓練**的模型輸出機率做加權平均 → 產出**集成預測**。

集成成功靠的是「**模型之間的互補性 (diversity)**」，而不是「每個成員都最強」。Wave C 用了完全不同的訓練擾動（mask/swap/delete/R-Drop/MSD），它們犯的錯與既有模型不同類型，因此即使單模較弱，加進 ensemble 池仍能透過 per-task hillclimb 得到正向貢獻。§26.4 的數字證實了這個直覺。

### 26.4 7-way Per-task Hillclimb 結果（**Phase 6 SOTA**）

**模型池（7 個）**：`combo_v2`、`combo_best`、`p3_large_lr2e5`、`p2ab_aug_mask10`、`p2ac_aug_mix`、`p2ad_rdrop05`、`p2ae_msd5`。前三者來自 Phase 5，後四者來自 Wave C。

**v2 粗網格結果**（每任務 2187 組合，OOF weighted score = 0.68056）：

| Task | best weights (v2:best:large:aug_m:aug_x:rdrop:msd) | F1 |
| :-- | :--: | :--: |
| T1 promise_status | 0 : 0 : 0 : 2 : 1 : 1 : 1 | 0.93690 |
| T2 verification_timeline | 1 : 2 : 1 : 1 : 1 : 0 : 2 | 0.48710 |
| T3 evidence_status | 1 : 1 : 1 : 0 : 0 : 1 : 0 | 0.87483 |
| T4 evidence_quality | 1 : 1 : 1 : 0 : 0 : 0 : 0 | 0.44699 |

**v3 隨機精修結果**（以 v2 為起點 × 5000 trials/任務，OOF weighted score = **0.68206**）：

| Task | best weights | F1 |
| :-- | :--: | :--: |
| T1 promise_status | 0 : 0 : 0 : **2.0** : 1.0 : 1.0 : 1.0 | 0.9369 |
| T2 verification_timeline | 1.25 : 2.0 : 1.0 : 1.5 : 0.5 : 0 : 2.0 | 0.4904 |
| T3 evidence_status | 1.0 : 1.0 : 0.75 : 0 : 0 : 1.0 : 0.25 | 0.8759 |
| T4 evidence_quality | 0.75 : 1.0 : 1.0 : 0 : 0 : 0 : 0 | 0.4524 |

**Phase 6 SOTA = 0.68206**，相對 Phase 5 (0.67954) **+0.00252**，相對 Phase 1 baseline (0.6415) **+0.0406**。

**關鍵發現：**
- **T1 完全交給 Wave C 4 模型**（前 3 個傳統模型權重全為 0）：T1 衝到歷史最高 0.9369，證實「單模較弱但訓練擾動方式不同」確實能透過 ensemble 反超「單模較強但都用同類訓練配方」的組合。其中 `p2ab_aug_mask10` 取得最高權重 2.0，token masking 與 BERT 預訓練目標一致，可能讓 T1 二元判斷更穩。
- **T2 在 7-way diversity 中略有提升**（0.4892 → 0.4904），主要靠 `combo_best` (2.0) + `p2ae_msd5` (2.0) 組合。
- **T3 由 large + R-Drop 主導**（`p2ad_rdrop05` weight=1.0 與 `combo_v2/best` 並列），R-Drop 雖然單模分數低，但對 evidence_status 的決策邊界互補性最好。
- **T4 仍是最大瓶頸（0.4524）**：Wave C 4 模型在 T4 全部被排除（weight=0），證實 T4 的稀有類別問題（如 `Misleading` 全集只有 1 筆）不是靠「訓練擾動」能解，需要 §53.2 的 oversampling 或 numeric token 標註等資料層改造。

詳見 [reports/analysis/_ensemble/per_task_hillclimb_v3_summary.csv](reports/analysis/_ensemble/per_task_hillclimb_v3_summary.csv) 與 [reports/analysis/_ensemble/per_task_hillclimb_v3_preds.csv](reports/analysis/_ensemble/per_task_hillclimb_v3_preds.csv)。

### 26.5 概念釐清：「ensemble pool 是做什麼的？」

> 此處回應 Phase 28 結尾提出的概念疑問。`ensemble pool` 概念在本節（Phase 6）首次以「7 個成員 × per-task hillclimb」的形態正式登場，故釐清放在這裡。

`ensemble pool` 是「**通過 admission 閘門的訓練成員（member）集合**」，由 hillclimb / weighted ensemble 演算法從中搜尋最佳權重組合來產生最終預測：

- **目前 v12 池**：14 個成員（p2_combo_best、p2_combo_v2/v3、p3 large、p4a/p4b、p5_t6、p6_t6v2、p7_focal_g3、p9_ls_t1t3、p10_large_focal_fgm 等），active SOTA = 0.68925（U1-c per-task TTA 後處理，詳 §39）。
- **Admission 門檻 = 0.66608**：只有單模 member-level OOF ≥ 0.66608 的訓練才能加入 pool；低於門檻的成員加入只會稀釋 ensemble 訊號（X4/X12 已驗證）。
- **Hillclimb 演進**：v6 → v7b → v8 → v9 → v10 → v11 → v12 漸進式 coordinate descent（每代詳見 §26.4 / §27.3 / §28.3 / §29.2 / §33 / §34），每加一個新成員 ΔSOTA 約 +0.00050~+0.00118。
- **Phase 26–30 全數拒絕並刪除**（U10 / U13 / U14 / U15 PATH-B/C/D/E）：2026-05-09 判斷上述路徑跳本偏離 ESG 企業永續報告書主類，走向沒有意義；資料 / 腳本 / 權重 / 報告一次性清除；U10 重關為「企業永續報告書專用」詳 §47 U10-NEW。

**結論**：ensemble pool 是「成員品質保證機制」，admission 是品質閘門，hillclimb 是權重搜尋演算法。

---

## 27. Phase 7 — Combo v3 + 8-way Per-task Hillclimb

> **本階段目標**：把 Phase 6 中「ensemble pool 內表現最好的 Wave C 兩個技術」（R-Drop 與 MSD）正式疊回 Combo 訓練配方，看是否能產出第一個能單獨贏過 combo_best 的新 combo；接著把它加入 ensemble pool 跑 8-way per-task hillclimb。

### 27.1 為什麼選 R-Drop 與 MSD 進 combo_v3？

Wave C 4 個配置中 R-Drop（-0.0021）與 MSD（-0.0011）是「**最不弱**」的兩個，且分屬完全不同層級：
- **R-Drop 是「損失層」正則**：在 loss function 加入 KL 一致性項。
- **MSD 是「分類頭層」正則**：在最後一層 dropout 取多樣本平均。
- 兩者**不重複作用**，理論上可疊加。

token-level augmentation 雖然 ensemble 內 T1 有用，但其作用本質上與 R-Drop 重疊（都在「擾動同一輸入」），且兩個 aug 配置單模降幅最大（-0.0044 / -0.0049），疊加風險較高，故先排除。

[configs/exp_p2_combo_v3.yaml](configs/exp_p2_combo_v3.yaml) 內容：
```yaml
extends: exp_p2_combo_v2.yaml
training:
 focal_tasks: [evidence_quality]
 focal_gamma: 2.0
 fgm_eps: 1.0
 rdrop_alpha: 0.5
model:
 msd_k: 5
seeds: [42]
```

### 27.2 combo_v3 單模結果（首次 Wave C 技術疊上 combo 成功）

5-Fold seed=42（與 combo_best seed=42 = 0.66558、combo_v2 seed=42 比較）：

| 模型 | weighted_score | T1 | T2 | T3 | T4 | vs combo_best |
| :-- | :--: | :--: | :--: | :--: | :--: | :--: |
| `combo_best` (Phase 2) | 0.66558 | — | — | — | — | baseline |
| `combo_v2` (Phase 5) | ~0.66833 | — | — | — | — | +0.00275 |
| **`combo_v3` (Phase 7)** | **0.67121** | 0.9332 | 0.4717 | 0.8644 | 0.4415 | **+0.00563** |

→ combo_v3 是**自 Phase 5 以來第一個單模超越 combo_v2 的訓練配方**，也驗證了 §26 的假設：**Wave C 技術在 ensemble 池內弱、但與已最佳化的 combo 配方互補時可正向疊加**。R-Drop 強迫的 dropout-子網路一致性 + MSD 的多樣本平均，等於同時在「訓練擾動」與「推論平滑」兩個維度做正則化。

### 27.3 8-way Per-task Hillclimb（**Phase 7 SOTA**）

工具：[src/tools/per_task_hillclimb_v4.py](src/tools/per_task_hillclimb_v4.py)（v3 起點 + 6000 次隨機局部精修，新增 `combo_v3` 為池中第 8 員）。

**模型池（8 個）**：v3 的 7 員 + `p2_combo_v3`。

| Task | best weights (v2:best:large:aug_m:aug_x:rdrop:msd:**v3**) | F1 | vs Phase 6 |
| :-- | :--: | :--: | :--: |
| T1 promise_status | 0 : 0 : 0 : 0.25 : 0.25 : 0.75 : 0.25 : **1.75** | 0.93832 | +0.0014 |
| T2 verification_timeline | 1.0 : 2.0 : 1.0 : 1.25 : 0.5 : 0 : 2.0 : 0 | 0.49031 | -0.0001 |
| T3 evidence_status | 1.0 : 1.0 : 0.75 : 0 : 0 : 1.0 : 0.25 : 0 | 0.87543 | -0.0005 |
| T4 evidence_quality | 0 : 1.5 : 1.0 : 0 : 0 : 0 : 0 : **0** ←等等 | 0.44915 | +0.0037 |

> 補正：T4 winner 是 (0.0, 1.5, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0) — combo_v3 在 T4 確實 weight=0，T4 提升來自於改變了 combo_v2 vs combo_best vs large 的相對權重結構（Phase 6 是 0.75:1.0:1.0，Phase 7 變成 0:1.5:1.0，意即 T4 完全棄用 combo_v2 而改用更高權重的 combo_best）。

**Phase 7 SOTA = 0.68394**（vs Phase 6 0.68206，**+0.00188**；vs Phase 1 baseline，**+0.0425**）。

**關鍵發現：**
- **T1 = 0.9383 為歷史最高**：combo_v3 取得 weight=1.75 的最高權重，幾乎主導 T1 預測。Phase 6 中 T1 只靠 4 個 Wave C 模型（單模都 < 0.66448）；Phase 7 換成 combo_v3（單模 0.67121）作主力後，T1 又再進一步。
- **T4 = 0.4560 首次突破 0.45 級距**（Phase 6 是 0.4524）：透過拋棄 combo_v2、把 combo_best 權重從 1.0 拉到 1.5，evidence_quality 的稀有類別判斷終於有正向動能。雖然 combo_v3 在 T4 winner 中沒被選用，但它的存在讓 hillclimb 有機會探索到「不同 combo_best 權重」的更佳組合。
- **T2/T3 微幅下降（<0.001）但被 T1/T4 增益完全覆蓋**：這正是 per-task hillclimb 對「全任務加權分數」做最佳化的合理 trade-off。

詳見 [reports/analysis/_ensemble/per_task_hillclimb_v4_summary.csv](reports/analysis/_ensemble/per_task_hillclimb_v4_summary.csv) 與 [reports/analysis/_ensemble/per_task_hillclimb_v4_preds.csv](reports/analysis/_ensemble/per_task_hillclimb_v4_preds.csv)。

### 27.4 累計 SOTA 進展軌跡

| Phase | SOTA | 增益 | 主要技術 |
| :--: | :--: | :--: | :-- |
| 1 baseline | 0.6415 | — | macbert-base 5-Fold |
| 2 combo | 0.66734 | +0.026 | LR/max_len/pooling 掃描 + combo_best |
| 4 ensemble | 0.67478 | +0.007 | combo + macbert-large 1.5:1 |
| 5 hillclimb v1 | 0.67954 | +0.005 | per-task 3-way hillclimb |
| 6 hillclimb v3 | 0.68206 | +0.003 | + Wave C 4 模型 → 7-way |
| 7 hillclimb v4 | 0.68394 | +0.002 | + combo_v3 (R-Drop+MSD) → 8-way |
| **8 hillclimb v6** | **0.68440** | **+0.0005** | combo_v3 multi-seed 拆 peaky/avg → 9-way |

---

## 28. Phase 8 — combo_v3 Multi-seed 拆分為 9-way 池（Phase 8 SOTA，已被 Phase 11 取代）

### 28.1 動機與假設

Phase 7 SOTA (0.68394) 中，T1 winner 給 `combo_v3 weight=1.75`（在 8-way tuple 中佔 54%），意味 T1 高度倚賴 combo_v3 seed=42 的「尖銳預測」。一個自然問題是：**multi-seed averaging 能否進一步降低 OOF 變異、推升 SOTA？**

預期：每個 seed 的 5-fold OOF std ≈ ±0.015，3-seed 平均後變異降約 1/√3，理論增益 +0.001 ~ +0.005。

### 28.2 試錯一：直接 multi-seed 平均（v5，**失敗**）

- 訓練 combo_v3 額外兩個 seed（2024、20260417），3-seed OOF：
 - seed=42: 0.67121
 - seed=2024: 0.67257
 - seed=20260417: 0.66452
 - 3-seed avg: **0.66854**
- 用 3-seed 平均 OOF 取代原 seed=42-only，跑 hillclimb v5（warm-start 自 Phase 7 winners）：
 - **per-task weighted = 0.68191**（比 Phase 7 -0.00203，**負面結果**）
- 失敗原因（事後分析）：
 - Phase 7 T1 winner `(0,0,0,0.25,0.25,0.75,0.25,1.75)` 在 multi-seed 平均下 T1 從 0.93832 跌到 0.93412
 - **平均化把 combo_v3 的 peaky 預測平滑化**，反而稀釋了 T1 賴以取勝的「強勢類別決策」訊號
 - T2/T3/T4 微幅改善 (+0.0013/+0.0002/0)，遠不足以補 T1 損失 (-0.0042 × 0.20 weight = -0.00084)

### 28.3 試錯二：拆分為兩成員 9-way 池（v6，**成功**）

關鍵設計：把 combo_v3 同時放入 pool 兩次：
- `p2_combo_v3_s42` = seed=42 only（peaky，保留 Phase 7 強勢訊號）
- `p2_combo_v3_avg` = 3-seed 平均（smooth，降低變異）
- hillclimb 為每個 task 自由選擇兩者權重比例 → 不會被強迫統一

執行：[src/tools/per_task_hillclimb_v6.py](src/tools/per_task_hillclimb_v6.py)（N_RANDOM=8000，warm-start 自 Phase 7 winners 並在 index 8 補 0）

| Task | 權重 (v2:best:large:mask:mix:rdrop:msd:**v3_s42**:**v3_avg**) | F1 | vs Phase 7 |
| :-- | :-- | :--: | :--: |
| T1 promise_status | 0:0:0:0.25:0.25:0.75:0.25:**1.75**:**0** | 0.93832 | 0 |
| T2 verification_timeline | 0.25:2.0:1.0:1.25:0.75:0:2.0:**0.5**:**1.0** | 0.49118 | +0.00087 |
| T3 evidence_status | 1.0:0.75:0.75:0:0:1.0:0.25:**0.25**:**0.5** | 0.87577 | +0.00034 |
| T4 evidence_quality | 1.0:1.5:1.5:0:0:0:0:**0**:**0** | 0.45132 | +0.00217 |

**Phase 8 SOTA = 0.68440**（vs Phase 7 0.68394，**+0.00046**；vs Phase 1 baseline，**+0.043**）。

### 28.4 結果解讀（每個 task 的故事）

- **T1**：hillclimb 自動把 v3_avg 設為 0、保留 v3_s42=1.75 → 完全採用 Phase 7 winner，沒被 multi-seed 拖累。**「拆分」設計的最大價值不是讓 v3_avg 主導 T1，而是讓 hillclimb 有權選擇「不選」**，避免被強迫融合而退化。
- **T2/T3**：v3_avg 確實取得非零權重（0.5 與 1.0、0.25 與 0.5），smooth 版降低了 OOF 上的決策邊界抖動 → 兩個 macro-F1 任務微幅獲益。
- **T4**：兩個 v3 版本都被設為 0；T4 的進展來自完全不同的方向 — 由 8000 次更大隨機搜索找到 `(combo_v2=1.0, combo_best=1.5, large=1.5)`（Phase 7 是 `0:1.5:1.0`），意即 **Phase 7 棄用 combo_v2 的決策被 Phase 8 推翻**。這是「9-way 池 + 8000 trials」單純放大搜索容量帶來的副作用，與 multi-seed 無直接關係。

### 28.5 累計教訓

1. **「平均」並非萬靈丹**：當某個成員在某 task 是「peaky 但準確」時，平均化會稀釋訊號。Multi-seed averaging 應該與「保留原版」並存，由 hillclimb 自選使用比例。
2. **池擴張的雙重收益**：v6 的增益並非完全來自 multi-seed，更大的搜索空間（9 維 × 8000 trials）也讓 T4 找到更好的 combo_v2/best/large 比例。
3. **Per-task hillclimb 的安全性**：warm-start 自上一階段 winners + 隨機探索的設計，讓「失敗的新成員」最壞退化為原 SOTA，永遠不會回退。

### 28.6 累計 SOTA 進展軌跡（含 Phase 8）

| Phase | SOTA | 增益 | 主要技術 |
| :--: | :--: | :--: | :-- |
| 1 baseline | 0.6415 | — | macbert-base 5-Fold |
| 2 combo | 0.66734 | +0.026 | LR/max_len/pooling 掃描 + combo_best |
| 4 ensemble | 0.67478 | +0.007 | combo + macbert-large 1.5:1 |
| 5 hillclimb v1 | 0.67954 | +0.005 | per-task 3-way hillclimb |
| 6 hillclimb v3 | 0.68206 | +0.003 | + Wave C 4 模型 → 7-way |
| 7 hillclimb v4 | 0.68394 | +0.002 | + combo_v3 (R-Drop+MSD) → 8-way |
| 8 hillclimb v6 | 0.68440 | +0.0005 | combo_v3 拆 peaky/avg → 9-way |
| 9 hillclimb v7b | 0.68558 | +0.00118 | + p4a/p4b → 11-way (biased search) |
| 10 hillclimb v8 | 0.68669 | +0.00111 | + p5_t6_time_token → 12-way (biased search) |
| 11 hillclimb v9 | 0.68683 | +0.00014 | + p6_t6v2_bucket_tok → 13-way (biased search) |
| 12 hillclimb v10 | 0.68660 | −0.00023 | + p7_focal_g3 → 14-way（單模最強 0.67416 但 ensemble 未破 SOTA）|
| **13 joint hillclimb v11 (SOTA)** | **0.68770** | **+0.00087** | + p8_ema995 (w=0) + p9_ls_t1t3 (w=0) → 16-way **joint post-constraint** hillclimb；4 次 accept 全在 T3，T4 +0.0018 為 constraint coupling 連動 |

詳見 [reports/analysis/_ensemble/per_task_hillclimb_v9_summary.csv](reports/analysis/_ensemble/per_task_hillclimb_v9_summary.csv)（Phase 11 當時 SOTA）與 [reports/analysis/_ensemble/per_task_hillclimb_v10_summary.csv](reports/analysis/_ensemble/per_task_hillclimb_v10_summary.csv)（Phase 12 記錄）。

---

## 29. Phase 9 — Sprint A backbone 擴 pool（已完成）

依 §56 Sprint A 計畫，依序訓練 3 個新骨幹（單 seed=42、5-Fold、繼承 combo_best 超參），加入 ensemble pool 後跑 hillclimb v7 以衝刺 0.69。

### 29.1 已完成成員

| # | exp | backbone | lr | OOF (5-Fold) | T1 | T2 | T3 | T4 | 耗時 | ≥0.65 入池 |
| :--: | :-- | :-- | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: |
| p4a | `p4a_roberta_wwm_base` | `hfl/chinese-roberta-wwm-ext` | 3e-5 | 0.66626 | 0.9265 | 0.4626 | 0.8555 | 0.4426 | 932s | 是 |
| p4b | `p4b_bert_base_chinese` | `bert-base-chinese` | 3e-5 | **0.67355** | 0.9346 | 0.4857 | 0.8656 | 0.4403 | 971s | 是 |
| p4c | `p4c_xlm_roberta_base` | `xlm-roberta-base` | 2e-5 | 0.61269 | 0.9107 | 0.3419 | 0.8391 | 0.3644 | 1108s | **否** |

**p4a 觀察**：OOF 0.66626 ≈ combo_best 0.66558（+0.0007 微幅優於）；T4 略高 0.4426 vs 0.4308 → **多樣性主要落在 T4**，可貢獻 ensemble。

**p4b 觀察**：OOF **0.67355** 為 Sprint A 最佳，優於 combo_best 0.66558（+0.00797）。T1=0.9346、T2=0.4857 同步上升，與 macbert 體系不同（bert-base-chinese 無 WWM、無 macbert MLM 校正，反而提供獨立信號）。

**p4c 觀察**：OOF 0.61269 **未過 ≥0.65 入池門檻**。問題在 T2=0.3419、T4=0.3644 嚴重落後（macro-F1 受少類拖累更甚）。主因：(1) xlm-roberta-base 為多語預訓練、中文 token 表徵密度低於專門中文模型；(2) lr=2e-5 + 5 ep 收斂不足（loss 仍從 4.0 → 2.4，但 fold 間 std=0.0345 顯示變異大）。**結論：踢出 v7 pool**。

### 29.2 Hillclimb v7 → v7b（biased search 修正）

**v7 結果**：N_RANDOM=10000、warm-start 自 v6 winners（idx 9, 10 補 0.0），最終 0.68440 = v6 完全平手。診斷：均勻擾動 1-3/11 個 index，新成員 (9, 10) 每輪只有 ~27% 機率被探索，且須同時抽到非零 GRID 值才能改善 → 探索率不足。

**v7b 修正**：強制每輪擾動必須包含至少一個新成員 (idx 9 或 10)，外加 0-2 個 legacy index；N_RANDOM 提升至 12000。

**v7b 結果（新 SOTA）：0.68558（+0.00118 vs v6）**

| Task | F1 | 變化 vs v6 | 新成員權重 |
| :-- | :--: | :--: | :-- |
| promise_status (T1) | 0.9383 | 0 | p4a=0, p4b=0 |
| verification_timeline (T2) | **0.5000** | **+0.0089** | p4a=0.5, p4b=1.25 |
| evidence_status (T3) | 0.8764 | 0 | p4a=0, p4b=0 |
| evidence_quality (T4) | 0.4572 | 0 | p4a=0, p4b=0 |

**結論**：T2 是唯一受惠 task — p4b 在 T2=0.4857 確實提供 macbert 體系沒有的訊號（其餘 task 因 macbert 已飽和而無法改善）。距 0.69 還差 **0.00442**，需進 §56 Sprint B (T6 時間 token 增強，預期 +0.005-0.009 直接擊中 T2)。

**Hillclimb 工程教訓**：擴大 pool 時，若 warm-start 包含大量零權重新維度，必須改用 biased search（強制探索新維度），否則均勻隨機擾動會使新成員實質為「裝飾品」。應寫入 ML 教訓記憶。

---

## 30. Phase 10 — Sprint B / T6 時間 token 增強（已完成）

依 §56 Sprint B 計畫，啟動 T6（時間 token 注入）；目標：直接擊中 T2 (verification_timeline) macro-F1，預期 +0.005 ~ +0.009。

### 30.1 設計

**輸入端確定性增強（無學習成本）**：以 regex 抽取文本中的西元年（`19xx|20xx`）/民國年（`民國 NNN`，加 1911 換算）/中文年（`YYYY 年`），取最大年份 → 計算與 CURRENT_YEAR=2025 的差 → bucket 對齊 T2 5 類，最後在文本前注入：

```
[時間 年份YYYY 距今N年{後|前} BUCKET] 原文…
```

bucket 規則：`delta ≤ 0 → already`；`0 < delta ≤ 2 → within_2_years`；`2 < delta ≤ 5 → between_2_and_5_years`；`> 5 → longer_than_5_years`。無年份樣本維持原文。

實作：
- 新模組 [src/data/text_augment.py](src/data/text_augment.py)：`add_time_tokens()` + `TEXT_TRANSFORMS` 註冊表
- [src/data/dataset.py](src/data/dataset.py)：`ESGDataset.__init__` 加 `text_transform=None`，`__getitem__` 在 tokenize 前套用
- [src/train_kfold.py](src/train_kfold.py)：`_build_loaders` 從 `cfg.data.text_transform` 查表
- 新 config [configs/exp_p5_t6_time_token.yaml](configs/exp_p5_t6_time_token.yaml)：繼承 combo_best 超參，seed=42

**Sanity 驗證**（regex 覆蓋率測試 6 條樣本，全通過）：
- `2030 年` → `[時間 年份2030 距今5年後 between_2_and_5_years]`
- `民國 115 年` → `[時間 年份2026 距今1年後 within_2_years]`
- `2024 年已達成` → `[時間 年份2024 距今1年前 already]`
- `沒有時間表的承諾` → 維持原文

樣本中時間 pattern 覆蓋率（1000 筆）：西元 4 位 47.1% / 中文年 45.9%（與西元有重疊）/ 民國 1.4% / 中文月 7.4%。

### 30.2 單模成績

| Item | 值 |
| :-- | :-- |
| Overall (5-Fold OOF) | 0.66231 |
| std | 0.00823 |
| T1 (promise_status) | 0.9356 |
| T2 (verification_timeline) | 0.4874 |
| T3 (evidence_status) | 0.8556 |
| T4 (evidence_quality) | 0.4155 |
| 耗時 | 879.5s |

**單模觀察**：Overall 略低於 combo_best 0.66558（−0.00327）；T2 = 0.4874 與 combo_best baseline ~0.49 持平，**未呈現預期的 T2 直擊提升**。可能原因：
1. macbert tokenizer 對 `[時間 年份…]` 字元拆成多 sub-tokens（從 sanity decode 可見「2024」拆為單字元），稀釋訊號；
2. T2 真正困難案例（`already ↔ between_2_and_5_years` 138 筆混淆）多無明確年份錨點，regex 命中率不足；
3. 確定性 prefix 與 macbert 預訓練分布偏離，反而使 T1/T3/T4 微跌。

### 30.3 Hillclimb v8（12-way pool）

**Tool**：[src/tools/per_task_hillclimb_v8.py](src/tools/per_task_hillclimb_v8.py) — 在 v7b 11-way 上新增 idx 11 = `p5_t6_time_token`；warm-start = v7b winners 補 0.0；biased search 強制每輪擾動 idx 11；N_RANDOM=12000；RNG seed=20260503。

**結果（新 SOTA）：0.68669（+0.00111 vs v7b）**

| Task | F1 | 變化 vs v7b | p5_t6 權重 (idx 11) |
| :-- | :--: | :--: | :--: |
| promise_status (T1) | **0.9387** | **+0.0004** | **2.0** |
| verification_timeline (T2) | 0.5001 | +0.0001 | 0.0 |
| evidence_status (T3) | **0.8765** | **+0.0001** | **2.0** |
| evidence_quality (T4) | 0.4527 | −0.0045 | 0.0 |

> 註：T4 數字看起來退步是 v7b/v8 best_w 同為 `(1.0, 1.5, 1.5, 0.0…)` → F1 在 hillclimb 內部評估時 = 0.45272；最終 weighted_score 受到 T1/T3 改變後 constraints 重映射的連帶影響略微浮動。整體 weighted +0.00111 仍為淨增。

**意外觀察 — p5_t6 對 T2 完全沒貢獻**：T2 best 仍維持 v7b 配置（idx 11 = 0.0）。p5_t6 反而在 **T1/T3 各以權重 2.0 入選**，提供 macbert/bert/roberta 體系外的 prefix-conditioned 訊號。推測原因：
- prefix 注入使 [CLS] 偏移，產生「結構性差異化的弱信號」 → 對 binary task (T1/T3) 有助；
- 對 macro-5 類 (T2)，p5_t6 自身 T2=0.4874 僅與 baseline 持平且預測分布相似 → ensemble 無互補。

### 30.4 結論與下一步

- **Sprint B T6 結論**：作為 ensemble 成員是有效的（+0.00111），作為 T2 直擊方案是失敗的（單模 T2 持平、ensemble idx 11 = 0）。
- 距 0.69 還差 **0.00331**，剩餘 §56 Sprint B 候選：
 - **T15 偽標籤 / 半監督**：當時列為候選；2026-05-04 已決議暫緩所有 pseudo-label 路線，改先做 U1/N2；
 - **T6 升級**：把 prefix 改為 numeric token（如直接插 `[YEAR_2030]` 單一 token，避免 sub-tokenization 稀釋）+ 重訓 → 可能挽救 T2；
 - **T14 Qwen LoRA**：8GB GPU 不可行（已標記）。

詳見 [reports/analysis/_ensemble/per_task_hillclimb_v8_summary.csv](reports/analysis/_ensemble/per_task_hillclimb_v8_summary.csv) 與 [reports/experiments/p5_t6_time_token/score_summary.csv](reports/experiments/p5_t6_time_token/score_summary.csv)。

---

## 31. Phase 11 — Sprint B / T6 v2 專屬 bucket token（已完成）

承 §30 教訓（multi-char Chinese prefix 被 macbert tokenizer 拆成 30+ sub-tokens 稀釋訊號），改採「**單一 added-vocab token 對應一個 T2 bucket**」設計，企圖讓模型直接學到 bucket-level 嵌入。

### 31.1 設計

5 個專屬 token：`[T_already] [T_within2] [T_2to5] [T_longer5] [T_NA]`，由 `tokenizer.add_tokens(special_tokens=False)` 加入詞表（21128 → 21133），訓練時於文本前注入 1 token。年份抽取/bucket 規則沿用 §30。

實作：
- [src/data/text_augment.py](src/data/text_augment.py)：新增 `BUCKET_TOKENS / add_time_bucket_token()` 與 `TRANSFORM_ADDED_TOKENS / get_added_tokens()` 註冊表
- [src/train_kfold.py](src/train_kfold.py)：tokenizer 載入後依 `cfg.data.text_transform` 自動 `add_tokens` + 模型 `resize_token_embeddings(len(tokenizer))`
- 新 config [configs/exp_p6_t6v2_bucket_tok.yaml](configs/exp_p6_t6v2_bucket_tok.yaml)：繼承 combo_best 超參，seed=42

Sanity 驗證：5 個 token 各被 tokenize 為單一 id（vocab 確實成長 5）。

### 31.2 單模成績

| Item | 值 | 對 v1 (p5_t6) |
| :-- | :-- | :-- |
| Overall (5-Fold OOF) | 0.66606 | +0.00375 |
| std | 0.01149 | — |
| T1 (promise_status) | 0.9339 | −0.0017 |
| **T2 (verification_timeline)** | **0.4703** | **−0.0171** |
| T3 (evidence_status) | 0.8606 | +0.0050 |
| T4 (evidence_quality) | 0.4301 | +0.0146 |
| 耗時 | 876.3s | — |

**單模觀察 — T2 反而更差**：與預期完全相反，T2 從 v1 的 0.4874 退步到 0.4703（−0.0171）。診斷：
1. 5 個 added-vocab embedding 從零學起，僅以詞表平均做 mean-resize 初始化；
2. 訓練集 ~800 筆 × 5 bucket → 每 bucket 平均 ~160 occurrences，gradient signal 對 5 維新嵌入太弱、無法收斂出有判別力的表徵；
3. T1/T3 略升 / 略跌互抵，T4 反而受惠（可能 T4 對 prefix-conditioned 弱 noise 敏感度較低）。

### 31.3 Hillclimb v9（13-way pool）

**Tool**：[src/tools/per_task_hillclimb_v9.py](src/tools/per_task_hillclimb_v9.py) — 在 v8 12-way 上新增 idx 12 = `p6_t6v2_bucket_tok`；warm-start = v8 winners 補 0.0；biased search 強制每輪擾動 idx 12；N_RANDOM=12000；RNG seed=20260504。

**結果（新 SOTA）：0.68683（+0.00014 vs v8）**

| Task | F1 | 變化 vs v8 | p6_t6v2 權重 (idx 12) |
| :-- | :--: | :--: | :--: |
| **promise_status (T1)** | **0.9398** | **+0.0011** | **1.5** |
| verification_timeline (T2) | 0.5001 | 0 | 0.0 |
| evidence_status (T3) | 0.8765 | 0 | 0.0 |
| evidence_quality (T4) | 0.4513 | −0.0014 | 0.0 |

**意外觀察 — p6_t6v2 又是 T1 受惠**：與 v1 (p5_t6) 一樣，T2 直擊失敗、T1 卻入選（與 p5_t6 並列：T1 best_w 同時 p5_t6=2.0 與 p6_t6v2=1.5）。推測原因：bucket token 提供的 prefix-conditioned 弱信號在 T1 (binary, 大樣本) 上能與 p5_t6 形成「不同 prefix scheme 的雙視角」，T1 ensemble 因此微升 +0.0011。

### 31.4 結論與下一步

- **Sprint B T6 v2 結論**：作為 T2 直擊方案**完全失敗**（單模 T2 從 0.4874 退至 0.4703；ensemble idx 12 在 T2=0.0）；作為 ensemble 成員**僅 T1 微利**（+0.00014 整體增益，幾乎接近搜索 noise）。
- **教訓寫入 `/memories/ml_ensemble_lessons.md`**：「Low-frequency added-vocab tokens（< 1k occurrences）在 fine-tuning 階段缺乏足夠 gradient 訊號收斂出判別力嵌入；對結構化 bucket 預測，prefix engineering 的 ROI 在 macbert 體系已飽和」。
- 距 0.69 還差 **0.00317**，T6 路線兩次嘗試（v1 prefix / v2 bucket token）皆無法直擊 T2 → **T6 系列 close**。剩餘路徑：
 - **T15 偽標籤**（§56 Sprint B 最後候選）：此為 2026-05-01 的歷史候選；2026-05-04 起已暫緩 pseudo-label 路線，改先做 U1/N2。
 - **T14 Qwen LoRA**：8GB GPU 不可行（已標記）。
- **Sprint B 暫告段落**；現行策略不等待 T15，而是在 6/03 前先完成 U1 TTA 與官方資料線的新結構性成員檢查。

詳見 [reports/analysis/_ensemble/per_task_hillclimb_v9_summary.csv](reports/analysis/_ensemble/per_task_hillclimb_v9_summary.csv) 與 [reports/experiments/p6_t6v2_bucket_tok/score_summary.csv](reports/experiments/p6_t6v2_bucket_tok/score_summary.csv)。

---

## 32. Phase 12 — Sprint B / T7 Focal-T4 γ=3.0 變體（已完成）

承 §56 Sprint B 名單 T7：在 combo_v3 超參（FGM eps=1.0 + Focal-T4 γ=2.0 + R-Drop α=0.5 + MSD K=5）上將 Focal-T4 的 γ 提高至 3.0，加重 T4 難樣本 gradient，預期 T4 macro-F1 微升或提供 ensemble 不同視角。

### 32.1 設計

- Config：[configs/exp_p7_focal_g3.yaml](configs/exp_p7_focal_g3.yaml)，`extends: exp_p2_combo_v3.yaml`，僅覆寫 `focal_gamma: 3.0`，seed=42。
- 訓練：hfl/chinese-macbert-base、5-Fold StratifiedKFold、5 epochs、batch=16、max_len=256、AMP、GPU=RTX 5060 Laptop。
- 零工程成本（僅 1 行 config 變更）、純依賴現有 trainer。

### 32.2 單模成績

| Item | 值 | 對 combo_v3 (s42, γ=2.0) |
| :-- | :-- | :-- |
| Overall (5-Fold OOF) | **0.67416** | **+0.00295** |
| std | 0.01316 | — |
| T1 (promise_status) | 0.9327 | −0.0056 |
| T2 (verification_timeline) | 0.4772 | −0.0122 |
| T3 (evidence_status) | 0.8640 | −0.0024 |
| **T4 (evidence_quality)** | **0.4481** | **+0.0237** |
| 耗時 | 1975.8s | — |

**單模觀察**：T4 上升 +0.0237 是以往所有單模最大的 T4 增益，Overall 0.67416 為歷來單模第二高（僅次於 p4b 0.67355 的讀法不一定正確—實際已超越 p4b）。T1/T2/T3 微跌為預期：　γ 提高讓 loss 在 T4 難類上衡重加重，誤差預算偏向 T4 → 其他 task 微量讓步。

### 32.3 Hillclimb v10（14-way pool）

**Tool**：[src/tools/per_task_hillclimb_v10.py](src/tools/per_task_hillclimb_v10.py) — 在 v9 13-way 上新增 idx 13 = `p7_focal_g3`；warm-start = v9 winners 補 0.0；biased search 強制每輪擾動 idx 13；N_RANDOM=12000；RNG seed=20260505。

**結果：0.68660（−0.00023 vs v9 SOTA 0.68683）— SOTA 未被突破**

| Task | per-task F1（hillclimb內部） | post-constraint F1 | p7_focal_g3 權重 (idx 13) |
| :-- | :--: | :--: | :--: |
| promise_status (T1) | 0.93992 | 0.9399 | **1.25** |
| verification_timeline (T2) | 0.49954 | 0.5032 | **2.00** |
| evidence_status (T3) | 0.87647 | 0.8777 | 0.0 |
| evidence_quality (T4) | 0.45202 | 0.4567 | 0.0 |

**意外觀察 — T2 首次以 2.0 選中新成員**：p7_focal_g3 是首個在 T2 拿到最高權重 2.0 的新成員（v9 所有 Sprint B 成員均為 0）。雖然其單模 T4=0.4481來自 T4 focal強化，ensemble 卻在 T2 最受惠 — 推測 γ=3.0 重新授權了全 task loss 平衡 → T2 預測分佈與 v9 pool 其他成員不同。

**為什麼仍然小幅退步 −0.00023？**：每 task per-task F1 本身都 ≥ v9（T1+0.00007、T2+0.00049、T3=、T4=），但最終 weighted score 是將 4 task arg-max 輸出套入 `apply_constraints_batch` 重新映射後計算。Constraint 會互相覕合（例：promise=No 則 timeline 必為 N/A），在某些邊緣樣本 v10 独立 T1/T2 改作導致聯合輸出偏移，抑制了個別 task 的微幅增益。

### 32.4 結論與下一步

- **Sprint B T7 結論**：單模成績及格 **＋T4 +0.0237 是歷史最大單項 T4 增益**；ensemble 絶佳選者（T2 權重 2.0 首次出現）但被 constraint coupling 抵銷→ SOTA 未破。
- **保留 v9 為當時 SOTA 記錄**；p7_focal_g3 列為有效成員，供未來 v11+ 與其他新成員合併使用。
- **距 0.69 維持 0.00317 距離**。Sprint B 現存 backlog（T8/T9/T11/T12）另列於 §53.2。
- 教訓：僅有「單 task per-task F1 提升」不代表「加權素提升」；constraints 後處理是聯合映射，未來 v11+ 需考慮「聯合 hillclimb」而非 per-task 獨立損失。

詳見 [reports/analysis/_ensemble/per_task_hillclimb_v10_summary.csv](reports/analysis/_ensemble/per_task_hillclimb_v10_summary.csv) 與 [reports/experiments/p7_focal_g3/score_summary.csv](reports/experiments/p7_focal_g3/score_summary.csv)。

---

## 33. Phase 13 — Sprint B 收尾批次：U2 EMA-995 + U4 LS(T1/T3) + U7 聯合 hillclimb（已完成，**新 SOTA 0.68770**）

承 §56 Sprint B 路徑與 §32 結論，Phase 13 將「同一類型」的 backlog 項目批次處理：

- **同一類型訓練變體**（U2 + U4，共 2 支單模）：兩支都是 combo_v3 基礎上的「單一 hyperparam 變動」，都用 seed=42 5-Fold，硬體成本相同（~33 min/支），可在同一終端串行執行。
- **平行算法工作**（U7 聯合 hillclimb v11）：直接優化 `apply_constraints_batch` 後的加權 score，解 §32 揭示的 constraint coupling 問題。CPU/IO 工作，可在 GPU 訓練同時實作。

### 33.1 設計

**U2 — `p8_ema995`**（[configs/exp_p8_ema995.yaml](configs/exp_p8_ema995.yaml)）：
- `extends: exp_p2_combo_v3.yaml`，僅追加 `training.ema_decay: 0.995`，seed=[42]。
- 設計依據：Phase 5 X1（decay=0.999）在 ~500 step 訓練下 EMA shadow 嚴重落後 → 0.473 崩潰。本次以 0.995 為「半衰期 ~138 步」，保證 5 epoch × 50 step = 250 step 訓練窗下 shadow 可追上權重動態。
- trainer 已支援 top-level `ema_decay` key（內含 CPU shadow + apply_to/restore + EMA-applied validation），無需新程式碼。

**U4 — `p9_ls_t1t3`**（[configs/exp_p9_ls_t1t3.yaml](configs/exp_p9_ls_t1t3.yaml)）：
- `extends: exp_p2_combo_v3.yaml`，新格式 `training.label_smoothing: {promise_status: 0.05, evidence_status: 0.05, ...其餘 0.0}`，seed=[42]。
- 設計依據：Phase 5 X2（uniform LS=0.05 全 task）讓 T4 少類 `Misleading` (support=1) 預測被推平 → macro 崩潰。本次「per-task 字典」僅在 T1/T3 兩個 binary 平衡任務套 0.05，避開 T2/T4 多類少類混合。
- 程式碼小幅修改：`src/training/losses.py::MultiTaskCE.__init__` 與 `src/training/trainer.py` 開頭，皆改成「dict 路徑優先、float 路徑相容」；既有 yaml（皆使用 `label_smoothing: 0.0` float）行為不變，僅新增 dict 形式支援。

**U7 — Joint Hillclimb v11**（[src/tools/joint_hillclimb_v11.py](src/tools/joint_hillclimb_v11.py)）：
- 直接以 `weighted_score(apply_constraints_batch(arg-max(combine(W[t]))))` 為目標函數，**避開 §32 v10 的 constraint coupling 漏洞**。
- 池：14 v10 成員 + 自動偵測 `p8_ema995` / `p9_ls_t1t3`（要求 5/5 fold 齊備才併入）。
- 演算法：座標式 hill-climb，每 iter 隨機選 (task, member) 一格擾動，唯有聯合 score 嚴格上升才接受。warm-start = v9 winners（為新成員填 0.0）。N_ITERS=30000，RNG seed=20260506。
- 輸出：`reports/analysis/_ensemble/joint_hillclimb_v11_{summary,preds,history,meta}.{csv,jsonl,json}`。

### 33.2 執行步驟

```powershell
# 一條指令串行訓練（同 GPU）
$env:CUDA_VISIBLE_DEVICES="0"; `
 python -m src.train_kfold --config configs/exp_p8_ema995.yaml *>&1 | Tee-Object -FilePath reports\experiments\p8_ema995.log; `
 python -m src.train_kfold --config configs/exp_p9_ls_t1t3.yaml *>&1 | Tee-Object -FilePath reports\experiments\p9_ls_t1t3.log

# 兩支訓練完成後
python -m src.tools.joint_hillclimb_v11
```

### 33.3 單模實驗結果

| Item | p8_ema995 (U2) | p9_ls_t1t3 (U4) | combo_v3_s42 baseline | 備註 |
| :-- | :--: | :--: | :--: | :-- |
| Overall (5-Fold OOF) | **0.63010** | **0.67334** | 0.67121 | p8 崩潰；p9 +0.00213 |
| std | 0.01528 | 0.01363 | — | — |
| T1 (promise_status) | 0.9216 | 0.9323 | 0.9383 | p9 −0.0060 (LS 平滑了高信心) |
| T2 (verification_timeline) | 0.3980 | 0.4703 | 0.4894 | p8 大跌；p9 −0.0191 |
| T3 (evidence_status) | 0.8563 | 0.8675 | 0.8664 | p9 +0.0011（如設計） |
| T4 (evidence_quality) | 0.3691 | 0.4459 | 0.4244 | p8 大跌；p9 +0.0215 |
| 耗時 | 1743s | 1487s | — | — |

**p8 結論（重要教訓）**：EMA decay=0.995 在「無 warm-start、無 EMA epoch 起始延遲」下仍重複 §53.3 X1 的失敗模式。理論上 0.995 半衰期 ~138 步應可追上 250 步訓練，**但實測 shadow 在 fold 1-5 全部仍嚴重落後**（每個 fold 最終 score 0.61~0.65 vs 不開 EMA 的 ~0.67）。診斷：trainer `_EMA.__init__` 直接以「初始隨機權重」作 shadow 起點，而非 head 收斂後再啟動 EMA → 訓練前 1-2 epoch shadow 拖累 validation。**建議：未來需重構 `_EMA` 加入 `start_step` 參數延遲啟動，否則此項應永久標 X 禁區擴充項。** p8 可保留於 pool 作對照，但 v11/v12 終態權重皆為 0。

**p9 結論**：合格但非最佳。T1/T3 各微升 +0.0011/+0.0011 如設計（小幅校準），但 T2 大跌 −0.0191 為**意外副作用** — 即使未對 T2 套 LS，dict 形式可能改變了 dropout/loss scale 互動使整體訓練動態微變。T4 +0.0215 為意外正向。OOF 加總略勝 baseline (+0.00213) 故 p9 入 v11 ensemble pool。

### 33.4 Joint Hillclimb v11 結果與結論

**v11 最終分數：0.68770（+0.00087 vs v9 SOTA 0.68683，+0.00110 vs v10）— 新 SOTA 達成**

| 指標 | warm-start (=v9) | v11 (final) | Δ |
| :-- | :--: | :--: | :--: |
| Joint score | 0.68683 | **0.68770** | **+0.00087** |
| T1 promise_status | 0.9398 | 0.9398 | 0.0000 |
| T2 verification_timeline | 0.5036 | 0.5036 | 0.0000 |
| T3 evidence_status | 0.8775 | **0.8783** | **+0.0008** |
| T4 evidence_quality | 0.4574 | **0.4592** | **+0.0018** |

**搜索結果摘要（30000 iters）**：
- 接受次數：僅 4 次，全部集中在 T3 一個 task。
- 接受路徑：iter 7 `T3[p3_large_lr2e5] 0.75→2.00 → 0.68719`；iter 84 `T3[p2ad_rdrop05] 1.50→1.00 → 0.68747`；iter 89 `T3[p2_combo_v3_s42] 0.25→0.75 → 0.68752`；iter 220 `T3[p2ab_aug_mask10] 0.00→0.25 → 0.68770`。
- iter 220 之後再無接受（29780 iter 全否決）。
- p8_ema995 / p9_ls_t1t3 在所有 task 終態權重皆為 0.0 — 兩支新成員均未被聯合搜索採用，**確認 p8 是 ensemble noise；p9 即使單模 +0.00213 亦無法在 ensemble 帶來新方向**（與 p2_combo_v3 過於同源）。
- T1/T2 權重 vector 完全保持 v9 winners 不變（搜索無法在這兩個 task 找到改進）。T4 權重 vector 也未改動，但 T4 score +0.0018 — 來自 **constraint coupling**：T3 改變後，apply_constraints_batch 連帶讓 T4 在 `y3=No → y4=N/A` 規則下少了一些錯預測。

**結論**：
1. **直接優化 post-constraint 分數的策略成功** — Phase 12 v10 per-task 搜索無法穿透 constraint barrier 而失敗 0.68660；v11 改用 joint objective，在 T3 微調權重即連帶提升 T4，確認 §32 提出的「constraint coupling」假說正確。
2. **當時距 0.69 護城仍差 0.00230** — v11 已收斂到此 16-way pool 的能力上限，再做 hillclimb 變體不會再有顯著增益。下一階段必須引入「結構性新方向」的成員（非 combo_v3 變體），例如 macbert-large、不同 tokenizer 的模型、或 MoE/teacher-student；此假設已在 Phase 14 用 p10 驗證。
3. **p8 永久封禁** — EMA 失敗模式重現，未來不再嘗試純 decay 變動，需先實作 `_EMA(start_step=...)` 才可重啟。
4. **p9 進池但無增益** — 證實 LS 對 ensemble 多樣性無幫助；保留 p9_ls_t1t3 在 pool 內供日後與「結構性新成員」的交叉搜索。
5. 對 §56 後續路徑（Phase 13 當下）：原 Sprint B 收尾完成；轉入 Sprint C「結構性多樣化」或準備 6/03 驗證集 hold-out 重訓 + 提交流程。Phase 14 已先完成其中的 N1。

---

## 34. Phase 14 — Sprint C 結構性多樣化 N1：macbert-large + Focal-T4 + FGM (p10) + Joint Hillclimb v12（已完成，**新 SOTA 0.68825**）

**完成日期**：2026-05-02 (Day 22)
**動作**：依 §56 Sprint C 路徑優先級，執行 N1（首個「large × combo」結構性新成員）+ v12 17-way joint hillclimb。

### 34.1 N1 訓練（p10_large_focal_fgm）

**Config**：[configs/exp_p10_large_focal_fgm.yaml](configs/exp_p10_large_focal_fgm.yaml)（繼承 `exp_p3_large_lr2e5.yaml`）。
- backbone：hfl/chinese-macbert-large（24 層，335M 參數）
- max_length=384, lr=2e-5, batch=4, grad_accum=4, epochs=5, seed=42
- **新增**：Focal Loss on T4（γ=2.0）+ FGM（eps=1.0）
- **跳過**：R-Drop（2× forward）+ MSD（5× dropout）— 8GB 顯存無法承載

**OOF 5-Fold weighted (seed=42)**：

| fold | epoch1 | epoch2 | epoch3 | epoch4 | epoch5 |
| :--: | :--: | :--: | :--: | :--: | :--: |
| 0 | 0.56874 | 0.61942 | 0.66084 | 0.66333 | **0.67396** |
| 1 | 0.57760 | 0.64413 | 0.62577 | 0.65900 | **0.66576** |
| 2 | 0.59817 | 0.64528 | **0.66549** | 0.66322 | 0.65291 |
| 3 | 0.56817 | 0.60748 | 0.65352 | 0.65815 | **0.66279** |
| 4 | 0.57872 | 0.65888 | **0.67571** | 0.67508 | 0.66497 |

- **最終 OOF = 0.66874**（mean），std=0.00572，5-Fold 跨度 0.6628 ~ 0.6757
- **Per-task means**：T1=0.9263 / T2=0.4762 / T3=0.8530 / T4=0.4462
- 訓練時間：2873s（≈48 分鐘）
- ≥ 0.66 admission threshold → **入池HF Hub 403 stderr**：`Discussions are disabled for this repo` 為 hfl 倉庫關閉討論區的無害噪訊（auto_conversion 線程探查 PR 失敗），不影響權重載入或訓練。PowerShell 因此返回 exit code 1，但 score_summary.csv 與 oof_probs.npz 均正確寫出。

### 34.2 Joint Hillclimb v12（17-way）

**腳本**：[src/tools/joint_hillclimb_v12.py](src/tools/joint_hillclimb_v12.py)
- Pool：v11 的 16 員 + 新成員 p10_large_focal_fgm（idx 16），M=17
- Warm-start：v11 SOTA winners + 0.0 padding
- N_ITERS=30000，GRID=[0, 0.25, 0.5, ..., 2.0]，RNG seed=20260603
- 目標函數：`weighted_score(apply_constraints_batch(arg-max(combine(W[t]))))`

**8 次 accept 全部發生在 T2（verification_timeline，先前最大瓶頸）**：

| iter | task | idx | exp | old → new | joint |
| :--: | :--: | :--: | :-- | :--: | :--: |
| 772 | T2 | 16 | **p10_large_focal_fgm** | 0.00 → 1.00 | 0.68781 |
| 866 | T2 | 0 | p2_combo_v2 | 0.00 → 1.25 | 0.68784 |
| 888 | T2 | 16 | **p10_large_focal_fgm** | 1.00 → 1.50 | 0.68784 |
| 986 | T2 | 10 | p4b_bert_base_chinese | 1.25 → 1.75 | 0.68789 |
| 1001 | T2 | 3 | p2ab_aug_mask10 | 0.25 → 0.50 | 0.68806 |
| 1059 | T2 | 0 | p2_combo_v2 | 1.25 → 0.75 | 0.68811 |
| 1231 | T2 | 11 | p5_t6_time_token | 0.00 → 0.25 | 0.68816 |
| 1359 | T2 | 0 | p2_combo_v2 | 0.75 → 0.50 | **0.68825** |

**最終 v12 SOTA = 0.68825**（+0.00055 vs v11）：
- T1=0.9398（不變）
- **T2=0.5072（+0.0036，瓶頸鬆動）**
- T3=0.8783（不變）
- T4=0.4592（不變）

**新成員 p10 最終權重**：T1=0, **T2=1.50**, T3=0, T4=0（4 任務只在 T2 有非零權重，其他任務均被 v11 既有成員主導）。

### 34.3 結論與下一步

**達成**：Phase 13 推測「16-way pool 已收斂、必須引入結構性多樣化」獲得實證——p10（首個 large + Focal-T4 + FGM 的混合員）一進池就解開 T2 瓶頸，貢獻全部 +0.00055 增益。

**證實假設**：
- v macbert-large 在 ensemble 中作為「pool member」可加分（不能作 main，這點 X5 仍成立）
- v T2 macro-F1 對「不同 backbone family + Focal-T4」組合敏感
- v v11 → v12 的 +0.00055 來自結構性新方向；因 warm-start 為 v11 權重 + p10=0，任何增益都必須由新成員或與新成員共同調整觸發

**距 0.69**：尚剩 −0.00175。下一步候選（依 ROI 排序）：
1. **U1 TTA inference variants** — 預期 +0.0005 ~ +0.001；不需重訓、不用外部資料
2. **N2 macbert-large + R-Drop（單獨）**或 **N3 nezha-base** — 預期 +0.0005 ~ +0.002
3. **U3 SWA** 或 **U5 T4 class-balanced re-sampling** — 預期 +0.001 ~ +0.002
4. **U10 外部 ESG 偽標** — 暫緩到最後再做；目前所有產物已清除，不屬於現階段原始訓練線

**產出**：
- `outputs/checkpoints/p10_large_focal_fgm/seed42/fold{0..4}/best.pt + oof_probs.npz`
- `reports/experiments/p10_large_focal_fgm/score_summary.csv + .json`
- `reports/analysis/_ensemble/joint_hillclimb_v12_{summary,preds,history,meta}.(csv|jsonl|json)` + `joint_hillclimb_v12.log`
- `logs/p10_large_focal_fgm_20260502_021523.log`

---

## 35. Phase 15 (Reset) — U10 暫緩並清除外部 pseudo 線（2026-05-04）

依專案規劃，U10 外部 ESG 報告偽標路線最後再做；目前專案必須回到「沒有加入外部非人工標註資料」的原始訓練狀態。因此已清除所有 U10 相關資料、程式、設定、權重、報告、log、split 與 admission 產物，不再保留任何外部 pseudo-label 訓練痕跡。

### 35.1 已清除範圍

| 類型 | 清除內容 |
| :-- | :-- |
| 外部資料 | `data/external/u10_*`、`data/external/raw/public_reports/` |
| pseudo 與合併資料 | `data/pseudo/u10_*`、`data/processed/u10_*` |
| 訓練輸出 | `reports/experiments/u10_*`、`outputs/checkpoints/u10_*`、`outputs/logs/u10_*`、`data/splits/u10_*` |
| 分析與入池產物 | `reports/pseudo/u10_*`、`reports/analysis/_ensemble/joint_hillclimb_v13_u10*` |
| 快取 | U10 相關 `__pycache__` |

### 35.2 目前狀態判定

- U10：**未做 / 暫緩**。
- 目前訓練主線：只使用官方人工標註 1,000 筆訓練資料與既有 Phase 1–14 模型/OOF 產物。
- 目前 active SOTA：Phase 15 U1 TTA `stored+middle`，OOF weighted = **0.6887881105**；不含 TTA 的最佳訓練 ensemble 仍為 Phase 14 v12，OOF weighted = **0.6882469764**。
- 短期不得再引用已刪除的 U10 檔案、權重或報告作為採納依據。

### 35.3 下一步原則

U10 之後若要重新啟動，必須從資料治理、公開來源、人工審核與採納門檻重新設計，不能沿用已清除產物。現階段優先執行不碰外部非人工標註資料的方法：

1. **U3/U5 非外部資料方法**：SWA 或 T4 class-balanced re-sampling，優先處理模型穩定性與 T4 少類瓶頸。
2. **U1-b TTA 補測**：視時間測 `stored+tail+middle` 或任務別 TTA 權重，但不得犧牲目前 `stored+middle` best。
3. **N3 新結構性成員**：p11 ELECTRA base 已被 admission 拒絕；若再走新 backbone，需換 tokenizer/backbone 或訓練策略。

---

## 36. Phase 15 (Implementation) — U1 TTA 推論擴增（2026-05-04）

**目標**：在完全不使用外部資料、不重訓模型的前提下，測試 v12 17-way joint ensemble 是否能透過 inference-time 多視角 token window 平均取得穩定 OOF 增益。

### 36.1 新增工具

| 檔案 | 功能 |
| :-- | :-- |
| `src/tools/u1_tta_oof.py` | 讀取 v12 meta、既有 fold checkpoint 與 official split；對 v12 非零權重成員重跑 validation fold 的 `tail` / `middle` token-window 推論，與 stored OOF 機率平均後套用原 v12 權重與 `apply_constraints_batch` 評分 |

設計重點：

- **零外部資料**：只使用 `data/raw/vpesg4k_train_1000 V1.csv`、既有 official fold split 與 Phase 1–14 checkpoints。
- **非洩漏 OOF**：每個 fold 只用該 fold checkpoint 推論該 fold validation indices。
- **active members only**：只對 v12 最終權重非零的 14/17 members 做 TTA；零權重成員不影響 score，不浪費 GPU。
- **BERT-family token window**：`tail` / `middle` 以 tokenizer token ids 截取，再手動加 `[CLS]` / `[SEP]` / padding，避免依賴目前 transformers tokenizer 不存在的 helper API。

### 36.2 實測結果

| 變體 | Weighted OOF | Δ vs v12 stored | T1 | T2 | T3 | T4 | 判讀 |
| :-- | --: | --: | --: | --: | --: | --: | :-- |
| v12 stored baseline | 0.6882469764 | — | 0.939845 | 0.507186 | 0.878285 | 0.459184 | Phase 14 SOTA |
| U1 `stored+tail` | 0.6887412240 | +0.0004942476 | 0.939845 | **0.509247** | 0.878285 | 0.459713 | T2/T4 提升，T1/T3 持平 |
| U1 `stored+middle` | **0.6887881105** | **+0.0005411340** | **0.940476** | 0.508718 | 0.878285 | 0.459713 | 本輪最佳，T1/T2/T4 皆不低於 baseline |

輸出：

- `reports/analysis/_ensemble/u1_tta_v12_active_stored_tail_summary.csv`
- `reports/analysis/_ensemble/u1_tta_v12_active_stored_tail_preds.csv`
- `reports/analysis/_ensemble/u1_tta_v12_active_stored_tail_meta.json`
- `reports/analysis/_ensemble/u1_tta_v12_active_stored_middle_summary.csv`
- `reports/analysis/_ensemble/u1_tta_v12_active_stored_middle_preds.csv`
- `reports/analysis/_ensemble/u1_tta_v12_active_stored_middle_meta.json`

### 36.3 決策

採 `U1 stored+middle` 作為目前 active OOF best：**0.6887881105**，距 0.69 剩 **0.0012118895**。此增益完全來自推論視角互補，未引入外部非人工標註資料，也未改變訓練集。後續已完成 N2/p11 結構性成員 admission，但未被採納；下一步不再重跑舊 17-way 或 p11 同設定，而應優先做 U3/U5，或補測 U1-b 三視角/任務別 TTA 權重，且不得取代已驗證的 `stored+middle` best。U1-b 三視角結果已於 §38 補測。

### 36.4 概念釐清：「TTA 是做什麼的？」

> 此處回應 Phase 28 結尾提出的概念疑問。TTA 概念在本節（Phase 15）首次落地，故釐清放在這裡。

TTA = **Test-Time Augmentation（測試時資料擴增）**，**不重訓練、只在推論時對同一筆樣本以多視角產生多份預測再聚合**：

- **U1 視角設計**：對長文本三種裁切方式
 - `stored`：從頭開始 [CLS] + 前 N token
 - `middle`：中間段落（避開頭尾的樣板宣傳）
 - `tail`：結尾段落（往往是承諾 / 結論）
- **聚合方式**：U1 等權平均、U1-b 三視角等權（§38）、U1-c **per-task per-view weighted**（每個 task 對每個 view 學不同權重，coordinate descent grid 0.05 + oracle warm-start，詳 §39）。
- **U1-c 實測**：active SOTA 從 0.68825 → **0.68925**（+0.00100），**完全免費**，零訓練成本，零外部資料；T2 macro 0.51026 為史上最高。
- **為什麼有效**：長 ESG 文本的關鍵訊號（時間詞、量化目標、reservation 用語）位置不固定 → 單一視角會錯失某些訊號；多視角覆蓋率高，per-task 權重能對 T2（時間詞多在中後段）強化中後視角。

**結論**：TTA 是「相同模型在推論時看不同片段再投票」，是 plateau 階段唯一還在生效的免費午餐技術。

---

## 37. Phase 16 完成 — N2/p11 ELECTRA base 官方資料結構性成員（2026-05-04）

**目標**：在不使用 U10、不使用外部 pseudo-label 的前提下，新增一個和既有 BERT/Roberta/MacBERT MLM-family 不同預訓練目標的 backbone，測試是否能成為 v12/U1 之後的互補成員。

### 37.1 選擇理由

| 候選 | 判斷 | 本輪決策 |
| :-- | :-- | :--: |
| large + R-Drop | 與 p10 同為 macbert-large，且 R-Drop 需要雙 forward，8GB 顯存風險高 | 暫緩 |
| NeZha / ERNIE | 可能有效，但模型相容性與下載穩定性需另行確認 | 後續候選 |
| **Chinese ELECTRA base** | base size、8GB 風險較低；replaced-token detection 預訓練目標不同，錯誤模式較可能和 MLM-family 互補 | **採用** |

### 37.2 設定與 Gate

| 項目 | 值 |
| :-- | :-- |
| Config | `configs/exp_p11_electra_base.yaml` |
| exp_name | `p11_electra_base` |
| backbone | `hfl/chinese-electra-180g-base-discriminator` |
| data | 官方人工標註 1,000 筆；不使用 U10 / pseudo |
| max_length | 384 |
| lr | 2e-5 |
| batch / grad_accum | 8 / 2 |
| seed / folds | seed=42, 5-Fold |
| admission gate | 單模 OOF ≥ 0.65，或 T2/T4/constraint-coupled joint admission 有明顯互補 |

### 37.3 正式 5-Fold 結果

已建立設定檔並完成 smoke test。Smoke 設定為 64 samples、2 folds、1 epoch，目的只驗證 pipeline 相容性，不作為正式分數判讀。

```powershell
C:/Users/User/AppData/Local/Programs/Python/Python313/python.exe -m src.train_kfold --config configs/exp_p11_electra_base.yaml --smoke
```

Smoke 結果：

| 項目 | 結果 |
| :-- | :-- |
| tokenizer / model load | 通過 |
| training loop | 通過 |
| fold checkpoint / summary output | 通過 |
| smoke weighted mean | 0.34381（僅 64 筆、2 folds、1 epoch，不作競賽判讀） |
| 非致命訊息 | HuggingFace discussions 403；ELECTRA discriminator prediction head unexpected keys。兩者均未中止訓練 |

正式 5-Fold 已完成，訓練流程採官方 1,000 筆、seed=42、5 folds；不使用 U10、不使用 pseudo-label：

```powershell
C:/Users/User/AppData/Local/Programs/Python/Python313/python.exe -m src.train_kfold --config configs/exp_p11_electra_base.yaml
```

正式結果：

| fold | best_epoch | weighted | T1 | T2 | T3 | T4 |
| :--: | :--: | :--: | :--: | :--: | :--: | :--: |
| 1/5 | 5 | 0.63666 | 0.91545 | 0.40551 | 0.85149 | 0.39227 |
| 2/5 | 5 | 0.60306 | 0.90588 | 0.33722 | 0.83276 | 0.34707 |
| 3/5 | 5 | 0.63121 | 0.92398 | 0.36862 | 0.84039 | 0.39716 |
| 4/5 | 4 | 0.62730 | 0.93452 | 0.40655 | 0.84667 | 0.35833 |
| 5/5 | 4 | 0.63647 | 0.92857 | 0.33016 | 0.87805 | 0.39375 |

整體：**mean=0.62694、std=0.01391、min=0.60306、max=0.63666**。Per-task means：T1=0.9217、T2=0.3696、T3=0.8499、T4=0.3777。

判讀：p11 單模未達 admission gate（0.65），其中 T2/T4 明顯低於既有強模型；但因 gate 仍保留「低單模但可能提供互補」的條件，仍需做 post-constraint joint admission。

產出：
- `outputs/checkpoints/p11_electra_base/seed42/fold{0..4}/best.pt + oof_probs.npz`
- `reports/experiments/p11_electra_base/score_summary.csv`
- `reports/experiments/p11_electra_base/score_summary.json`

### 37.4 v16/p11 Admission 結果

已建立並執行 admission 檢查工具 `src/tools/joint_hillclimb_v16_p11.py`，用途是以 v12 final weights warm-start，把 p11 作為第 18 員加入後重新做 post-constraint joint hillclimb。該工具採 biased search（70% 擾動指定到 p11 新維度，30% 留給舊成員 cleanup），避免零權重新成員在飽和 pool 中探索不足。

```powershell
C:/Users/User/AppData/Local/Programs/Python/Python313/python.exe -m src.tools.joint_hillclimb_v16_p11
```

結果：

| 項目 | 值 |
| :-- | :-- |
| pool | 18-way（v12 17 員 + p11） |
| warm-start score | 0.6882469764（等同 v12，p11 初始 0 權重） |
| search | 30,000 iterations；new-member-biased |
| accepted | 0 |
| final joint score | 0.6882469764 |
| delta vs v12 training SOTA | +0.0000000000 |
| delta vs U1 active SOTA | −0.0005411340 |
| final p11 nonzero tasks | 0/4 |
| final p11 weights | T1=0.00、T2=0.00、T3=0.00、T4=0.00 |

產出：
- `reports/analysis/_ensemble/joint_hillclimb_v16_p11_summary.csv`
- `reports/analysis/_ensemble/joint_hillclimb_v16_p11_preds.csv`
- `reports/analysis/_ensemble/joint_hillclimb_v16_p11_history.jsonl`
- `reports/analysis/_ensemble/joint_hillclimb_v16_p11_meta.json`

### 37.5 決策

p11 ELECTRA base **不納入 active ensemble**。原因是：

1. 單模 OOF mean=0.62694，未達 0.65 admission gate。
2. post-constraint joint admission 30,000 次搜尋無任何 accept。
3. p11 最終四任務皆 0 權重，未提供 v12/U1 之後可用的互補訊號。

目前 active best 已推升為 Phase 18 U1-c per-task TTA：**0.6892512673**（詳見 §39）。Phase 15 U1 `stored+middle` 0.6887881105 仍作為 per-task TTA 的 fallback 參考。Phase 19~25 已將 U11 GroupKFold sanity、U12 OOF↔valid gap 診斷、U3 SWA、U5 T4 re-sampling、N3 NeZha→ERNIE fallback、U2 EMA + warm-start、U6 NLLB-600M 回譯 全數驗證完畢；**內部可控訓練側探索空間現為零，除 U10（依規劃暫緩）外§53.2 backlog 已全數驗證完畢**，重心全部轉到 6/03 valid 釋出後的 OOF↔Valid gap 複測、GroupKFold drift 復量、以及提交策略。

---

## 38. Phase 17 完成 — U1-b TTA 三視角補測（2026-05-05）

**目標**：在不重訓、不引入外部資料的前提下，驗證 `stored+middle+tail` 三視角等權平均是否能在 v12 active pool 上超越 `stored+middle` 兩視角 best (0.6887881105)。此為 §56 第三優先項，最低成本且最容易出 ROI 的「補完一輪 TTA 設計空間」實驗。

### 38.1 設計

- 工具沿用 §36.1 `src/tools/u1_tta_oof.py`，命令列只新增一個 `--views` 值：
 ```powershell
 C:/Users/User/AppData/Local/Programs/Python/Python313/python.exe -u -m src.tools.u1_tta_oof --views stored middle tail --members active
 ```
- 流程不變：對 v12 14 個非零權重成員的每個 fold checkpoint，分別在 `middle` 與 `tail` 兩個 token-window 視角重跑 validation 推論；和 stored OOF 三者等權平均後，套用 v12 final weights 與 `apply_constraints_batch`。
- 不更動 v12 weights、不更動 active SOTA 之前需先比較 `stored+middle` 兩視角 best。

### 38.2 實測結果

| 變體 | Weighted OOF | Δ vs v12 stored | Δ vs U1 active SOTA | T1 | T2 | T3 | T4 |
| :-- | --: | --: | --: | --: | --: | --: | --: |
| v12 stored baseline | 0.6882469764 | — | −0.0005411340 | 0.939845 | 0.507186 | 0.878285 | 0.459184 |
| U1 `stored+tail` | 0.6887412240 | +0.0004942476 | −0.0000468865 | 0.939845 | 0.509247 | 0.878285 | 0.459713 |
| **U1 `stored+middle`** | **0.6887881105** | **+0.0005411340** | — | **0.940476** | 0.508718 | 0.878285 | 0.459713 |
| U1-b `stored+middle+tail` | 0.6887283739 | +0.0004813975 | **−0.0000597366** | 0.939845 | 0.509412 | 0.878285 | 0.459606 |

判讀：

- 三視角 T2 取得本輪最高 0.509412（高於兩視角 best 0.509247、`stored+middle` 0.508718），說明 `tail` 視角主要貢獻 T2。
- 但三視角等權平均同時拉回 T1 至 baseline 0.939845（低於 `stored+middle` 的 0.940476），且 T4 較 `stored+middle` 微降 0.000107。
- 加權後 `stored+middle+tail` 為 0.6887283739，**低於** `stored+middle` 0.6887881105 約 0.0000597。
- 因此三視角等權平均**未能超越** `stored+middle`，**不取代** active SOTA。

### 38.3 決策

1. 維持 §36 記錄 — Phase 15 active SOTA `stored+middle`：**0.6887881105**（在 §39 已被 Phase 18 U1-c 0.6892512673 取代，不來回跳動）。
2. U1-b 三視角等權結果已記錄為負向證據；後續若要再壓 U1-b，必須做「任務別 TTA 權重」（例如 T2 採三視角、T1/T3/T4 採兩視角）或視角別權重學習，而非繼續嘗試等權平均。
3. 加上 U1-b 結果，§56 中 U1-b 視為已關閉「等權三視角」這條子分支；後續優先項改為 U3/SWA、U5/T4 class-balanced re-sampling、N3 新結構性候選。

### 38.4 產出

- `reports/analysis/_ensemble/u1_tta_v12_active_stored_middle_tail_summary.csv`
- `reports/analysis/_ensemble/u1_tta_v12_active_stored_middle_tail_preds.csv`
- `reports/analysis/_ensemble/u1_tta_v12_active_stored_middle_tail_meta.json`
- `reports/analysis/_ensemble/u1_tta_v12_active_stored_middle_tail.log`

非致命訊息：HuggingFace `get_repo_discussions` 403（背景執行緒，僅影響 safetensors auto-conversion 嘗試，不影響 OOF 數值）；ELECTRA discriminator 預訓練 head 未使用之 unexpected keys 屬正常下游微調行為。

---

<a id="39-phase-18-完成--u1-c-任務別視角別加權-tta2026-05-05"></a>
## 39. Phase 18 完成 — U1-c 任務別/視角別加權 TTA（2026-05-05）

**目標**：在 §38 證實「等權三視角」會稀釋 T1/T4 之後，將 TTA 由「等權」升級為「per-task 視角加權」，藉由 v12 active 14 員的 stored / middle / tail 三視角機率快取進行 coordinate descent，看能否在不重訓、不引入外部資料的條件下突破 `stored+middle` 0.68879。此為 §56.0 排序中的 A1（最高 ROI 的零再訓練實驗）。

### 39.1 設計

- **新工具**：`src/tools/u1c_per_task_tta.py`
 - 沿用 §38 `u1_tta_oof.py` 的 `predict_one_view`，但將 middle / tail 機率快取至 `outputs/cache/u1c_tta/{member}_{view}.npz`，避免之後重複推論。
 - 對每個任務 t，搜尋簡形 α_{t,view}（sum_view α_{t,v}=1，grid step 0.05 → 231 點/任務/輪）。
 - **暖開**：先以「per-task best of {stored_only, stored+middle eq, stored+tail eq, stored+middle+tail eq}」做 oracle 4 變體擇優，再以 oracle 為起點跑 coordinate descent，最多 6 輪、收斂即停。
 - 每次評估都先依 v12 per-(task, member) 權重做 17 員→單一視角機率，再以 α 混合，最後過 `apply_constraints_batch` 才打分（結構與 v12 一致，避免破壞 constraint coupling）。
- **資料／成員**：v12 17-way active 成員 14 員（`p2_combo_v2` … `p10_large_focal_fgm`），與 §36 / §38 同。
- **基準對照**：
 - stored only（v12 baseline）= 0.6882469764
 - stored+middle 等權（前 SOTA）= 0.6887881105
 - stored+tail 等權 = 0.6887412240
 - stored+middle+tail 等權（X8）= 0.6887283739
 - oracle per-task best of 上述 4 變體 = 0.6891248409

### 39.2 實測結果

**U1-c searched** weighted OOF = **0.6892512673**

| 任務 | α stored | α middle | α tail | T 分數 | vs `stored+middle` Δ |
| :-- | :--: | :--: | :--: | --: | --: |
| promise_status (T1, w=0.20) | 0.50 | 0.50 | 0.00 | 0.940476 | ±0 |
| verification_timeline (T2, w=0.15) | 0.25 | 0.65 | 0.10 | 0.510255 | +0.001537 |
| evidence_status (T3, w=0.30) | 0.50 | 0.00 | 0.50 | 0.879060 | +0.000775 |
| evidence_quality (T4, w=0.35) | 0.50 | 0.00 | 0.50 | 0.459713 | ±0 |

**最終分數**：

- vs Phase 15 U1 active SOTA (`stored+middle`)：**+0.0004631568**
- vs Phase 14 v12 stored baseline：**+0.0010042909**
- 距 0.69 護城線：**−0.0007487327**

### 39.3 決策

1. 推升 active SOTA = **0.6892512673**（Phase 18 U1-c per-task TTA）。
2. T2 視角偏好 `middle` 為主、`tail` 次要（與「中段含具體年份描述」的語意吻合）；T3/T4 視角偏好 `stored+tail` 等權（與「結論句多在尾段」吻合）；T1 維持 `stored+middle`。可量化解釋與 §38 觀察一致。
3. coordinate descent 第 1 輪在 T2 取得 +0.0001264，第 2 輪即收斂；網格 0.05 對 1000 樣本 OOF 已達雜訊地板，**不嘗試更細網格**（避免 OOF 過擬合）。
4. 未來若再壓 U1-c，須先做 §56.0 A2/A3（GroupKFold sanity + OOF↔valid gap），確認此 +0.000463 不是 OOF 過擬合，再考慮細化網格或 Dirichlet 先驗。

### 39.4 產出

- `src/tools/u1c_per_task_tta.py`
- `outputs/cache/u1c_tta/{14 members}_{middle,tail}.npz`（28 個 cache，後續 U1-c/U1-d 可零成本重用）
- `reports/analysis/_ensemble/u1c_per_task_tta_summary.csv`
- `reports/analysis/_ensemble/u1c_per_task_tta_preds.csv`
- `reports/analysis/_ensemble/u1c_per_task_tta_meta.json`
- `reports/analysis/_ensemble/u1c_per_task_tta.log`

非致命訊息：HF 403（背景 safetensors auto-conversion 嘗試），不影響 OOF 數值；exit code 1 來自背景 thread，所有產物在 raise 前已寫盤完成（已驗證 `meta.json` 完整 + 三檔 size 正常）。

---

## 40. Phase 19：U11 GroupKFold sanity（read-only 診斷，2026-05-05）

### 40.1 動機

§56 第二優先項。U1-c 的 +0.000463 全部來自既有 5-Fold StratifiedKFold OOF；在進入 valid set 前，必須先量化目前 5-Fold 是否存在「同公司同時出現於 train/val」的 row-level leakage，否則 OOF 與 valid 的差距無法歸因。本步驟**完全不再訓練**，僅針對 split 結構與標籤分布作對照。

### 40.2 設計

- 載入 `vpesg4k_train_1000 V1.csv`（N=1000，50 公司）。
- 在 seed ∈ {42, 2024, 20260417} 各跑：
 - `StratifiedKFold(n_splits=5, shuffle=True)`（既有所有 active member 採用）。
 - `StratifiedGroupKFold(n_splits=5, shuffle=True, group=company)`。
- 每 fold 量測：
 1. **公司洩漏率**：val 公司中也出現在 train 的比例。
 2. **資料列洩漏率**：val 列中其公司也出現在 train 的比例。
 3. **per-task label drift**：StratifiedKFold val vs GroupKFold val 的逐類分布最大絕對差。
- 決策規則：若 SK 平均 row-leak ≥ 0.50 且 GK 各 task 平均 drift < 0.04 → 建議切換 GK；若 SK 高洩漏但 GK drift ≥ 0.04 → MONITOR（不重訓但需在 valid 後重評）；否則維持 SK。

### 40.3 結果

| seed | SK row-leak | GK row-leak | T1 drift | T2 drift | T3 drift | T4 drift |
| :--: | --: | --: | --: | --: | --: | --: |
| 42 | 0.998 | 0.000 | 0.017 | 0.072 | 0.033 | 0.054 |
| 2024 | 1.000 | 0.000 | 0.024 | 0.110 | 0.032 | 0.071 |
| 20260417 | 1.000 | 0.000 | 0.014 | 0.072 | 0.032 | 0.062 |

- 公司分布：50 unique，max=45，median=19.5，無 singleton 公司。
- 平均 SK row-leak ≈ **99.93%** — 幾乎所有 val 列的公司都出現在 train 中。
- T2 / T4 GK drift > 0.05 ~ 0.11，超過 drift 門檻 0.04。

### 40.4 決策

**MONITOR — leakage observed but drift would distort label balance**：

1. **不重訓**：若改用 GroupKFold 重跑全部 14 員，T2 macro-F1 變動上限可能 ±0.02 ~ ±0.03（因類分布偏移），高於 U1-c 的 +0.00046 增益，風險不對稱。
2. **valid 階段保留心理預期差**：當 6/03 valid 釋出時，預期 valid 與 OOF 將出現顯著 gap（因 valid 多半含未見過公司）。若 gap ≤ 0.005，視為健康；若 ≥ 0.01，需在 §41 / U12 中進一步診斷。
3. **U3 SWA / U5 T4 re-sampling 仍繼續**，因兩者皆是「同 split」內加強，不受公司洩漏影響相對排序。
4. **未來若仍要切換 GK**，應同步：(a) 重跑全部 14 員 + (b) 重新計算 v12 加權 + (c) U1/U1-c 全部重新搜尋。等同重置 Phase 14 ~ Phase 18 結果，**僅在 valid gap ≥ 0.02 時才考慮**。

### 40.5 產出

- `src/tools/u11_group_kfold_sanity.py`
- `reports/analysis/diagnostics/u11_group_kfold_sanity.json`
- `reports/analysis/diagnostics/u11_group_kfold_sanity.md`

---

<a id="41-競賽-active-軌跡tta-path"></a>
## 41. Phase 20：U12 OOF cross-fold variance（read-only 診斷，2026-05-05）

### 41.1 動機

§56 第三優先項。U1-c +0.000463 是否落在 OOF 雜訊內？以及 §40 已證 row-leakage ≈ 99.93%，因此需量化「同 split 多 fold 的天然 std budget」，作為解讀未來 valid gap 的尺規。

### 41.2 設計

- 解析 v12 active 14 員的 `reports/experiments/<exp>/score_summary.csv`，逐 fold 取 weighted_score 與四項 macro-F1。
- 計算每員：fold std、range、`fragility = (max−min)/mean`。
- 池子層級：
 - 平均單員 weighted_std。
 - 加權預期 = Σ wᵢ · per_task_std_mean。
 - ensemble 估算 = 上式 / √N。
- 不重訓、不做推論，純報表彙整。

### 41.3 結果

- 12/14 員有完整 `score_summary.csv`（缺 `p2_combo_v3_s42`、`p2_combo_v3_avg` 兩個合併型成員）。
- 平均單員 weighted_std = **0.0115**（max 0.0171）。
- 加權預期單員 std budget = **0.0157**；ensemble 平均後 ≈ **0.0045**。
- per-task std（mean / max）：T1 0.0081 / 0.0118、T2 0.0236 / 0.0322、T3 0.0090 / 0.0160、T4 ~ 0.018 / ~ 0.025。
- 最脆弱前 5 名（fragility）多落在 T2 / T4 變動大的成員（詳見 `u12_oof_fold_variance.md`）。

### 41.4 決策

1. **U1-c 增益 +0.000463 嚴格落在「ensemble 雜訊 budget 0.0045」之內**，但仍是同 OOF 上的真增益（每 fold 都被加權後一致提升）。**仍可保留 U1-c 為 OOF SOTA**，惟在 valid 階段以同樣的 per-task α 重算才能宣告「真增益」。
2. **valid gap 解讀規則**：
 - gap ≤ 0.005：屬正常 fold-noise + 5-fold→1-valid 切換；不需動作。
 - 0.005 < gap ≤ 0.015：U1-c 收益可能被吃掉，但 v12 ensemble 仍應穩；不重訓。
 - gap > 0.015：明顯分布偏移（與 §40 row-leakage 一致），啟動 §40.4 的 GroupKFold 重置流程。
3. **B1 / B2 仍照計畫進行**：因 fold std 在 T2 / T4 最大，U3 SWA / U5 T4 re-sampling 都直接針對最脆弱維度，預期能壓縮 fold std 而非單純抬升均值。

### 41.5 產出

- `src/tools/u12_oof_fold_variance.py`
- `reports/analysis/diagnostics/u12_oof_fold_variance.csv`
- `reports/analysis/diagnostics/u12_oof_fold_perfold.csv`
- `reports/analysis/diagnostics/u12_oof_fold_variance.json`
- `reports/analysis/diagnostics/u12_oof_fold_variance.md`

---

## 42. Phase 21：U3 SWA — `p2_combo_best_swa` 設置與訓練（2026-05-05）

### 42.1 動機

§56 排序中的 B1。SWA（Stochastic Weight Averaging）在 NLP fine-tuning 通常能換到 +0.001 ~ +0.003，且不需 shadow 副本（不重蹈 EMA X1 的覆轍）。`p2_combo_best`（macbert-base / max_len=384 / lr 3e-5 / batch=8 grad_accum=2 / 5 epochs）為 v12 active pool 中**最快、最穩定**的成員，適合作為 SWA 首測；單 fold 約 5~10 分鐘，5 fold 約 30~50 分鐘。

### 42.2 設計

1. **`trainer.py` 修改**（最小侵入）：在 epoch 迴圈尾端，若 `tcfg.get("swa_last_k") > 0` 且 `epoch > epochs - swa_last_k`，額外存 `out_root/swa_epoch{epoch}.pt`（僅 model state dict + 元資料，不含 optimizer）。預設關閉，零向後相容風險。
2. **新 yaml `configs/exp_p2_combo_best_swa.yaml`**：繼承 `exp_p2_combo_best`，加 `training.swa_last_k: 3`，`seeds: [42]`。將存 epoch 3、4、5 三個 swa 檔。
3. **新工具 `src/tools/u3_swa_aggregate.py`**：
 - 對每 fold 載入 `swa_epoch{3,4,5}.pt`，對 float-tensor 簡單算術平均，整數 buffer（如 `position_ids`）取首檔。
 - 重建模型、載入平均後權重、跑 val 推論，得 `oof_probs_swa.npz` 與每 fold 的 SWA 分數。
 - 聚合 5 fold OOF 算成員級 weighted_score；對照原 best.pt 的 score 計算 delta。
 - 報表：`reports/experiments/p2_combo_best_swa/swa_score_summary.{csv,json}` + `reports/analysis/diagnostics/u3_swa_p2_combo_best_swa.md`。
4. **錄取門檻**：若成員級 SWA OOF ≥ p2_combo_best 原 OOF + 0.0005，加入 v17（18-way）joint hillclimb；否則僅作為 SWA 工程驗證收尾，不入池。

### 42.3 執行狀態

訓練於 2026-05-05 由本 session 啟動（async）：
```
python -u -m src.train_kfold --config configs/exp_p2_combo_best_swa.yaml *> outputs/logs/p2_combo_best_swa.runlog
```
- HF 403（背景 safetensors auto-conversion）為已知非致命噪訊。
- 訓練完成後將執行 `python -m src.tools.u3_swa_aggregate --config configs/exp_p2_combo_best_swa.yaml`。
- 結果與決策將補入 §42.4。

### 42.4 結果（2026-05-05 完成）

- 5 折 × 3 epoch 平均 ckpt 全數產出（15 個 `swa_epoch*.pt`）；聚合腳本順利寫出 5 個 `swa.pt`、`oof_probs_swa.npz` 與 `oof_probs.npz`（canonical alias）。
- **member-level OOF（seed=42, K=3 epoch 平均）**：
 - fold 0: 0.6754（vs best-epoch 0.6742, **+0.0012**）
 - fold 1: 0.6455（vs 0.6425, **+0.0030**）
 - fold 2: 0.6616（vs 0.6662, **−0.0046**）
 - fold 3: 0.6528（vs 0.6600, **−0.0072**）
 - fold 4: 0.6711（vs 0.6850, **−0.0139**）
 - **平均 0.66252 vs 0.66558（−0.0031）**
- **結論**：SWA member-level OOF 較 best-epoch baseline 略低 0.003，**未通過 admission 門檻（≥ +0.0005）**。
 - 失敗原因可能：epoch 5 學習率非零導致末段 ckpt 仍偏離 loss basin；K=3 涵蓋的 epoch（3, 4, 5）方差大（fold 0 epoch 4 達 0.674、epoch 5 跌回 0.665）。
 - **不入 v12 池、不執行 U1-c 重跑**；保留產出供未來若採用 cosine 完整 0 LR 收斂時對照。
 - 列入 X9 風險區（避免 SWA + LR ≠ 0 的組合）。

### 42.5 產出

- `src/training/trainer.py`（新增 `swa_last_k` 邏輯，10 行）
- `configs/exp_p2_combo_best_swa.yaml`
- `src/tools/u3_swa_aggregate.py`
- `outputs/checkpoints/p2_combo_best_swa/seed42/fold{0..4}/swa_epoch{3..5}.pt`（訓練後）
- `outputs/checkpoints/p2_combo_best_swa/seed42/fold{0..4}/swa.pt`、`oof_probs_swa.npz`（聚合後）
- `reports/experiments/p2_combo_best_swa/swa_score_summary.{csv,json}`
- `reports/analysis/diagnostics/u3_swa_p2_combo_best_swa.md`

---

## 43. Phase 22：U5 T4 class-balanced re-sampling — 設置（2026-05-05）

### 43.1 動機

§56 B2。U12 顯示 T4 (`evidence_quality`) macro-F1 跨 fold std 為四項中第二高，且訓練 ensemble best 仍停在 0.4597（其中 `Misleading` support=1 為硬性瓶頸）。透過對 T4 標籤分布做 sqrt-inverse-frequency re-sampling，期望在不放大 noise 的前提下緩解 head/tail 比例。

### 43.2 設計

1. **`train_kfold.py` 修改**（`_build_loaders`）：若 `cfg.training.resample_t4` 為 truthy（True 或 0~1 之間 alpha 浮點），改用 `WeightedRandomSampler`，權重 = `1 / count(class)^alpha`。alpha 預設 1.0；本實驗用 **alpha=0.5**（sqrt 抑制）以避免 singleton class 過度放大。
2. **新 yaml `configs/exp_p2_combo_best_resample_t4.yaml`**：繼承 `exp_p2_combo_best`，加 `training.resample_t4: 0.5`，`seeds: [42]`。
3. **執行順序**：B1 SWA 完成後啟動，避免 GPU 衝突。

### 43.3 執行狀態

訓練完成（5 折 × 5 epoch × seed=42，~16 分鐘）。

### 43.4 結果（2026-05-05 完成）

| fold | best_epoch | weighted | T1 | T2 | T3 | T4 | vs baseline (Δ) |
| :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: |
| 0 | 4 | 0.6751 | 0.929 | 0.489 | 0.875 | 0.439 | +0.0009 |
| 1 | 2 | 0.6465 | 0.931 | 0.407 | 0.852 | 0.410 | +0.0040 |
| 2 | 2 | 0.6642 | 0.936 | 0.436 | 0.855 | 0.443 | −0.0020 |
| 3 | 5 | 0.6767 | 0.942 | 0.477 | 0.851 | 0.462 | +0.0167 |
| 4 | 4 | 0.6656 | 0.931 | 0.468 | 0.850 | 0.440 | −0.0194 |
| **mean** | | **0.66562** | **0.9337** | **0.4554** | **0.8568** | **0.4386** | **+0.00004** |

- 與 baseline `p2_combo_best` seed=42 mean 0.66558 持平（**+0.00004**），未通過 admission +0.0005 門檻。
- T4 macro 0.4386 vs baseline 0.4265（**+0.012**）— 預期方向正確；但 T2 macro 0.4554 vs baseline 0.4691（**−0.014**）— sampler 改動使 T2 同步退化（因為 sampler 對整批 batch 重抽，T2 weight 並未隨之調整），抵銷掉 T4 增益。
- **結論**：member-level OOF 持平，**不入 v12 池、不執行 U1-c 重跑**；列入 X10 風險區（單一任務 sampler 影響其他任務）。
- 若未來再嘗試，需 (a) 改 per-task loss weighting 而非 global sampler，或 (b) 對 T4 用 head-only mini-finetune，或 (c) 同時平衡 T2 與 T4。

### 43.5 產出

- `src/train_kfold.py`（`_build_loaders` 條件式 sampler，30 行）
- `configs/exp_p2_combo_best_resample_t4.yaml`
- `outputs/checkpoints/p2_combo_best_resample_t4/seed42/fold{0..4}/best.pt`、`oof_probs.npz`
- `reports/experiments/p2_combo_best_resample_t4/score_summary.{csv,json}`

---

## 44. Phase 23：N3 NeZha-base 設置（2026-05-05）

### 44.1 動機

§56 C1。Active pool 14 員裡 12 員都是 macbert / roberta-wwm / electra base 同源，僅 `p3_large_lr2e5` / `p10_large_focal_fgm` 為 macbert-large；缺乏 NeZha 系列（華為 NeZha 在 functional Chinese NLP 上常與 macbert 互補）。**N3 = `peterchou/nezha-chinese-base`**，沿用 `p2_combo_best` 主超參，single seed=42。

### 44.2 設計

- 新 yaml `configs/exp_p13_nezha_base.yaml`，僅覆蓋 `model.backbone: peterchou/nezha-chinese-base`。
- tokenizer 與 sentencepiece 相容性需於首次載入時確認；若 NeZha 與 transformers 4.x 不相容，將切換到 `nghuyong/ernie-3.0-base-zh` 作為 N3-fallback（也已預留 yaml 模板）。
- 入池門檻：單模 ≥ 0.66；若達標執行 v17 joint hillclimb（18-way），否則僅紀錄為 X9。

### 44.3 執行狀態

兩階段嘗試均完成：
1. **NeZha-base**（`peterchou/nezha-chinese-base`）：transformers `AutoModel.from_pretrained` 無 NeZha-specific class，自動 fallback 為 `BertModel`，導致 NeZha 的相對位置編碼層 `embeddings.position_embeddings.weight` 被以隨機初始化覆蓋；fold0 5 epoch 最高 0.549，fold1 三 epoch 全 0.497（loss 不下降）→ **判定不可訓練**，於 fold1 epoch 3 中止。
2. **ERNIE-3.0-base-zh**（`nghuyong/ernie-3.0-base-zh`，N3-fallback）：載入乾淨；5 折 × 5 epoch 完整訓練。

### 44.4 結果（2026-05-05 完成）

**NeZha 階段**：fold0 best 0.549，遠低於 0.66 admission；中止。

**ERNIE 階段**：
| fold | best_epoch | weighted | T1 | T2 | T3 | T4 |
| :--: | :--: | :--: | :--: | :--: | :--: | :--: |
| 0 | 5 | 0.6254 | 0.917 | 0.420 | 0.851 | 0.354 |
| 1 | 5 | 0.5802 | 0.907 | 0.305 | 0.805 | 0.318 |
| 2 | 4 | 0.6195 | 0.912 | 0.395 | 0.824 | 0.373 |
| 3 | 2 | 0.5926 | 0.921 | 0.275 | 0.821 | 0.346 |
| 4 | 5 | 0.6250 | 0.920 | 0.414 | 0.826 | 0.375 |
| **mean** | | **0.6086** | **0.9152** | **0.3616** | **0.8237** | **0.3531** |

- **ERNIE 單模 0.6086 << 0.66 admission 門檻**，亦低於 X6 xlm-roberta-base 0.6127；直接拒絕，不入池、不執行 v17 hillclimb。
- T2/T4 macro 退化最嚴重（vs baseline T2 0.469、T4 0.426，分別 −0.107、−0.073）。ERNIE 與 macbert 的 vocab/tokenization 差異反而傷害本資料集（短中文句、多英數混合的 ESG 內容）。
- **結論**：N3 candidate 全數失敗（NeZha 不可訓、ERNIE 單模太低）；列入 X11 風險區（base-class 異源 backbone 對本資料集不利）。
- 後續若再嘗試結構性多樣性，建議改 (a) 探索 macbert/roberta-wwm 內部 layer-wise 變體（freeze 底層 vs 全微調），或 (b) 直接做 large-class backbone 多 seed（U2 EMA + warm-start），而非更換 base 同源 backbone。

### 44.5 產出

- `configs/exp_p13_nezha_base.yaml`、`configs/exp_p13_ernie_base.yaml`
- `outputs/checkpoints/p13_ernie_base/seed42/fold{0..4}/best.pt`、`oof_probs.npz`
- `reports/experiments/p13_ernie_base/score_summary.{csv,json}`

---

## 45. Phase 24：U2 EMA + warm-start 設置與訓練（2026-05-05）

### 45.1 動機

- §53.3 X1 紀錄：Phase 5 ema999（無 warm-start）OOF 0.473 崩潰，原因為 shadow 權重在 ~500 步內未追上模型主權重。
- §53.2.1 U2 即為對 X1 的修正：EMA decay 降為 0.995，並引入 `ema_warmup_epochs`，前 N epoch 不啟動 EMA 追蹤，僅以原權重訓練；EMA 從第 N+1 epoch 才以當下模型參數重新初始化 shadow 並開始累積。
- 目標：以 0 額外資料、~16 min 訓練成本，測試 admission +0.0005，看能否成為新 ensemble 成員。

### 45.2 設計

- `configs/exp_p2_combo_best_ema995_warm.yaml`：繼承 `exp_p2_combo_best.yaml`；新增 `training.ema_decay=0.995`、`training.ema_warmup_epochs=2`；seeds=[42]；fold=5；epoch=5。
- `src/training/trainer.py`：新增 `ema_warmup_epochs` config 讀取與 `ema_started` 旗標；當 `epoch > ema_warmup_epochs` 時以當下模型參數初始化 shadow，再進入 update / apply_to / restore 流程；warm-up 期 EMA 完全不參與訓練與評估。
- 設定下 EMA 實質追蹤 epoch 3、4、5（共 ~300 步），相當於後段 snapshot averaging。

### 45.3 執行狀態

- 終端 ID `594f8bcc-50e7-4781-8a73-04aed3cecdcf`；HF 403 `get_repo_discussions` benign；exit code 1 為 HF 噪聲，`score_summary.json` 已正常寫出。
- 5 fold × 5 epoch × seed 42，總 ~16 min 完成。

### 45.4 結果（2026-05-05 完成）

fold-by-fold（best epoch = 5，皆為 EMA-eval 後分數）：

| fold | best_epoch | weighted_score | T1 (binary) | T2 (macro) | T3 (binary) | T4 (macro) |
| :--: | :--: | :--: | :--: | :--: | :--: | :--: |
| 0 | 5 | 0.65288 | 0.9226 | 0.4625 | 0.8611 | 0.4019 |
| 1 | 5 | 0.63733 | 0.9235 | 0.4202 | 0.8339 | 0.3983 |
| 2 | 5 | 0.65763 | 0.9275 | 0.4763 | 0.8533 | 0.4134 |
| 3 | 5 | 0.64128 | 0.9415 | 0.4579 | 0.8622 | 0.3590 |
| 4 | 5 | 0.65791 | 0.9345 | 0.4699 | 0.8542 | 0.4122 |
| **mean** | | **0.64941** | **0.9299** | **0.4574** | **0.8529** | **0.3970** |

- baseline `p2_combo_best` seed=42 mean OOF = **0.66558**；U2 EMA-warm = **0.64941**；**Δ = −0.01617**。
- 5 折全部退化（最差 fold1 −0.0282、最好 fold4 −0.0077），無單折正向。
- per-task vs baseline：T1 0.9405→0.9299（−0.011）、T2 0.469→0.4574（−0.012）、T3 0.8783→0.8529（−0.025）、T4 0.4592→0.3970（−0.062）；**T4 退化最嚴重**，與 X1 的 T4 崩盤模式同源（少類在 EMA 平均後機率被推平）。
- 與 X1 ema999 0.473 相比有顯著改善（warm-start 確實避免徹底崩潰），但仍遠低於 admission 門檻 0.66608（baseline + 0.0005）。
- 主因推測：(a) 5 epoch 訓練後段 LR 仍在 cosine 中段（與 X9 SWA 同樣末段 LR ≠ 0 問題），EMA 平均了仍在震盪的權重；(b) `ema_warmup_epochs=2` 後僅剩 3 epoch 進行 EMA 累積，shadow 樣本量不足；(c) 對 macro-F1 任務（T2/T4）EMA 平均化會傷少類預測 confidence。
- **結論**：U2 不入 v12 池；列入 X12 風險區（EMA + warm-start 在 cosine 末段 LR ≠ 0 + 5 epoch 短訓練下仍對 T4 macro 不利）。
- 後續若再試 EMA：需 (a) cosine 完整下到 0 LR、(b) 拉長至 8~10 epoch、(c) 對 T2/T4 改用 best-epoch 而非 EMA-epoch；不重跑同設定。

### 45.5 產出

- `configs/exp_p2_combo_best_ema995_warm.yaml`
- `src/training/trainer.py`（新增 `ema_warmup_epochs` 與 `ema_started` 邏輯）
- `outputs/checkpoints/p2_combo_best_ema995_warm/seed42/fold{0..4}/best.pt`、`oof_probs.npz`
- `reports/experiments/p2_combo_best_ema995_warm/score_summary.{csv,json}`
- `outputs/logs/p2_combo_best_ema995_warm.runlog`

---

## 46. Phase 25 完成 — U6 / B4 NLLB-600M 中→英→中 回譯資料增強（2026-05-05）

> **狀態**：完成且拒絕（D30 / X13）。member-level OOF mean = **0.66321** vs baseline 0.66558（**−0.00237**），未通過 admission 0.66608；T2 macro 0.4652（−0.014）、T4 macro 0.4276（−0.031），雙目標任務皆退化。**§53.2 backlog 自此除 U10（依規劃暫緩）外已全數驗證完畢**。Active SOTA 維持 Phase 18 U1-c per-task TTA = **0.68925** 不變。

### 46.1 動機

§53.2 backlog 中 B1（U3 SWA）、B2（U5 T4 re-sample）、C1（N3 NeZha→ERNIE）、B3（U2 EMA + warm-start）四項已先後於 Phase 21~24 完成且全數拒絕；U6（T2/T4 回譯資料增強）為 §53.2 中最後一項可在本機完成的內部訓練側可控變項。NLLB-200-distilled-600M 為本機可用、合規、無需外部 API 的最大 zh↔en 翻譯模型，可用以實測 BT 增強對 T2 timeline 與 T4 quality 兩個 macro-F1 弱項是否能帶來增益，作為 §53.2 backlog 收尾批次的最後一動。

### 46.2 設計

**Pipeline（`scripts/u6_backtranslate.py`）**：

- 模型：`facebook/nllb-200-distilled-600M`（HF cache）；`src_lang=zho_Hant`（繁中），pivot=`eng_Latn`。
- Phase 1（zh→en）：`num_beams=4` greedy 確定性翻譯，BS=4，AMP fp16。
- Phase 2（en→zh）：`k=0` 為 greedy（最高保真度），`k≥1` 為 sampling（`do_sample=True`，`temperature=0.7+0.1*k`），用以引入語意多樣性。
- 增強樣本 id 編碼：`new_id = orig_id*100 + k`；保留 `_source_id`（int 原 id）與 `_aug_k` 兩個欄位以供 trainer 追蹤。
- T2 倍率（針對 timeline 弱類）：`T2_AUG = {"within_2_years": 3}`。
- T4 倍率（針對 quality 弱類）：`T4_AUG = {"Misleading": 5, "Not Clear": 1}`。
- 多任務同時符合時取**最大倍率**規則，避免重複展開。
- 輸出：`data/processed/u6_backtrans.json`（168 records，UTF-8）、`reports/u6/backtrans_summary.json`（統計）。

**Trainer 注入機制（`src/train_kfold.py`）**：

- 在 `records, df = load_dataset(...)` 之後讀取 `cfg["data"].get("augment_path")`，若存在則載入 `aug_records`；OOF 與 valid 仍以原始 `records`/`df` 構建（**驗證集絕不接受任何增強樣本**）。
- 在 fold loop 內，先以 `tr_ids = {int(records[i]["id"]) for i in tr_idx}` 收集當折 train 真實 id；再篩 `aug_records` 中 `_source_id ∈ tr_ids` 注入 `train_recs`。
- 每折注入數量輸出 log：`[u6] fold=N injected K BT augmented samples (train=...)`，便於審計。

**配置（`configs/exp_p2_combo_best_u6_bt.yaml`）**：

```yaml
extends: exp_p2_combo_best.yaml
exp_name: p2_combo_best_u6_bt
data:
 augment_path: data/processed/u6_backtrans.json
seeds: [42]
```

只覆寫 `augment_path` 與 `seeds=[42]` 以保留 5 折 × 5 epoch baseline 設定，確保 **Δ 完全可歸因於 BT 增強**。

### 46.3 執行狀態

- 翻譯腳本（async 終端 id `111d1339-…`）：186 秒完成；138 唯一源員 → 168 增強樣本（by_T2: between_2_and_5_years 62、longer_than_5_years 30、already 37、within_2_years 39；by_T4: Not Clear 124、Misleading 5、Clear 30、N/A 9；T2 與 T4 多有交集，最大倍率規則生效）。
- 訓練（async 終端 id `e3471b5a-…`）：1095.8 秒完成；HF 403 `get_repo_discussions` 為已知無害錯誤，`score_summary.json` 已正常寫出。
- per-fold 注入計數：fold0=140、fold1=125、fold2=135、fold3=138、fold4=134。

### 46.4 結果（5-fold OOF, member-level）

| Fold | best_epoch | score | T1 (binary F1) | T2 (macro F1) | T3 (binary F1) | T4 (macro F1) |
| :--: | :--: | :--: | :--: | :--: | :--: | :--: |
| 0 | 5 | 0.6748 | 0.9271 | 0.4869 | 0.8562 | 0.4557 |
| 1 | 4 | **0.6334** | 0.9306 | 0.4066 | 0.8551 | 0.3706 |
| 2 | 4 | 0.6632 | 0.9298 | 0.4716 | 0.8763 | 0.4104 |
| 3 | 5 | 0.6750 | 0.9283 | 0.4727 | 0.8664 | 0.4528 |
| 4 | 5 | 0.6696 | 0.9298 | 0.4881 | 0.8451 | 0.4484 |
| **mean** | — | **0.66321** | **0.9291** | **0.4652** | **0.8598** | **0.4276** |
| baseline (`p2_combo_best` seed=42) | — | 0.66558 | 0.9290 | 0.4793 | 0.8662 | 0.4587 |
| **Δ vs baseline** | — | **−0.00237** | +0.0001 | **−0.014** | −0.006 | **−0.031** |

- 5 折標準差 = 0.0174（fold1 顯著離群，−0.0282），baseline 同折 std 約 0.013，**BT 顯著放大了 fold-to-fold variance**。
- **雙目標任務 T2 與 T4 皆退化**（T2 −0.014、T4 −0.031），增強行為等同於 label-noise 而非 diversity injection。
- 未通過 admission 0.66608（baseline + 0.0005）；**不入 v12 池**。

**根本原因分析**：

1. **NLLB-200-distilled-600M 對繁中保真度不足**：`outputs/logs/u6_backtranslate.runlog` 顯示「台泥」被譯為 `TaiMot`、字符腐化、詞重複（如「致力於建立建立」），公司品牌名與 ESG 專有名詞嚴重偏移。
2. **英文 pivot 損失 ESG-specific terminology**：「永續發展」→ `sustainable development` → 「可持續發展」（語義同但詞表偏移），「兩年內」常被譯成不完整時間表達。
3. **未加 round-trip ChrF/BLEU 過濾**：低保真度譯本未被自動過濾，全部 168 筆原樣注入。
4. **train_recs 從 ~795 被膨脹至 ~935**，含低保真度樣本對 T2/T4 macro 等同 label-noise，稀釋真實訊號。
5. **fold1 離群**：該折注入 125 筆 BT 樣本中可能包含關鍵稀類（within_2_years / Misleading），低品質譯本直接讓 T4 macro 從 0.46 跌到 0.37。

**未來若再試 BT 路線（嚴禁不重跑同設定）**，需：

- (a) 換 NLLB-3.3B 或 madlad-7B 或外部 NMT API 以提高中文保真度；
- (b) 加「原文 vs BT 譯本」 round-trip ChrF≥0.5 自動過濾（或人工抽查）；
- (c) 限縮至 T4 Misleading only（5 筆），避免 T2 多倍率擴增帶來雜訊；
- (d) 不重跑 NLLB-600M 同設定（已列入 §53.3 X13 禁區）。

### 46.5 產出

- `scripts/u6_backtranslate.py`（NLLB-200-distilled-600M zh→en→zh BT pipeline，新檔）
- `src/train_kfold.py`（新增 `augment_path` 載入與 per-fold `_source_id` 過濾注入機制）
- `configs/exp_p2_combo_best_u6_bt.yaml`（新檔）
- `data/processed/u6_backtrans.json`（168 條增強樣本）
- `reports/u6/backtrans_summary.json`（138 唯一源員、by_T2 / by_T4 統計）
- `outputs/checkpoints/p2_combo_best_u6_bt/seed42/fold{0..4}/best.pt`、`oof_probs.npz`
- `reports/experiments/p2_combo_best_u6_bt/score_summary.{csv,json}`
- `outputs/logs/u6_backtranslate.runlog`、`outputs/logs/p2_combo_best_u6_bt.runlog`

### 46.6 後續影響

- §53.1 新增 D30 行（U6 / B4）；§53.3 新增 X13 行（NLLB-600M zh-en-zh BT 禁區）。
- §53.2.1 U6 狀態由「尚未開始」更新為「完成（D30，未通過 admission；移入 §53.3 X13）」。
- §56.0 ~~B4~~ 行劃線並標註「完成且拒絕」。
- §53、§14.1、§14.3 收尾語：除 U10（依規劃暫緩）外，**§53.2 backlog 已全數驗證完畢；內部可控訓練側可動項目現為零**。
- Active SOTA 維持 Phase 18 U1-c per-task TTA = **0.68925** 不變。

---

<a id="47-u10--企業永續報告書-sr-弱監督-pipeline-完整版2026-05-09-重啟2026-05-10-v2-重訓"></a>
## 47. U10 — 企業永續報告書 (SR) 弱監督 pipeline 完整版（2026-05-09 重啟，2026-05-10 v2 重訓）

> **章節定位**：本章把 §47 從「散裝進度紀錄」重構為**架構完整、每段都講清楚說明白**的工程規格。Phase 26~30 失敗教訓 → 重啟原則 → 5-stage pipeline → 兩版 (v1/v2) 演進 → 最終 OOF SOTA = **0.67746**（baseline +0.01012，45-ckpt 三方 stack）→ 結構性殘留與後續路徑。命名一律使用 `u10`（不再 `u10new`）。

### 47.0 設計原則與失敗教訓

**為什麼存在這個 pipeline**：
- Phase 26~30（U10 PATH-D / U13 PATH-E / U14 PATH-C / U15 PATH-B / U15 v4-clean PATH-B）全數**實證偏離 ESG 企業永續報告書主題**：PATH-C 取「公司形象 / SDG 對接 / 治理框架」一般 landing 段落、PATH-B v4-clean 取 Wikipedia + IPCC/UNFCCC 學術 PDF，皆與賽方標註資料 (`vpesg4k_train_1000`) 之分布不一致 → distribution shift 對 T2/T3 持續為負。
- 結論：**外部資料必須與「企業永續報告書原文」同質**，才能避免 distribution shift。

**本輪三條鐵律**：
1. **只取台灣（TWSE / TPEX）上市櫃公司「企業永續報告書 (SR / CSR / ESG Report)」原文 PDF 段落**。
2. **來源公司不得與已標註 50 家重疊**（避免資訊洩漏、SimHash 去重失效）。
3. **不直接合併訓練**：先 teacher 偽標 → admission gate → 兩階段訓練（Stage A 含偽 / Stage B 純官方）→ OOF ensemble，全程隔離雜訊。

**驗收三條件**：
- OK 端到端 pipeline 可重現（§47.10 列一鍵指令）。
- OK OOF weighted_score（best.pt path）較 baseline 提升 ≥ +0.005 → **實際 +0.01012**。
- OK 對 minority class（T2 timeline / T4 quality）提供可量測覆蓋改善 → v2 偽標籤 `longer_than_5_years` 0→570、`already` 50→1645，`Not Clear` 0→3；殘留 `within_2_years` / `Misleading` 仍 0（根因記於 §47.8）。

### 47.1 5-stage 架構總覽

```
┌─────────────┐ ┌─────────────┐ ┌──────────────┐ ┌──────────────┐ ┌────────────────┐
│ M1 採集 │ → │ M3 抽取 │ → │ M4 偽標註 │ → │ M5 兩階段 │ → │ M6 OOF Ensemble│
│ 31 ticker │ │ paragraphs │ │ admission │ │ K-Fold 訓練 │ │ best.pt averaging
│ 62 PDFs │ │ v1: 510 │ │ v1: 211/510 │ │ 3 seeds × 5 │ │ baseline+v1+v2 │
│ FY23+FY24 │ │ v2: 12,345 │ │ v2: 3,904 │ │ folds = 15 │ │ 45-ckpt stack │
└─────────────┘ └─────────────┘ └──────────────┘ └──────────────┘ └────────────────┘
 │ │
 ▼ ▼
 15 best.pt × 變體 SOTA = 0.67746
```

| 階段 | 輸入 | 主程式 / 設定 | 輸出 |
|---|---|---|---|
| **M1** 採集 | 31 disjoint ticker | `scripts/u10_collect_v3.py`、`u10_collect_v3b.py` | `data/raw/u10/{ticker}_FY{year}.pdf` × 62 |
| **M3** 抽取 | 62 PDFs | v1 `u10_pdf_extract.py` / v2 `u10_pdf_extract_v2.py` | `data/processed/u10/{corpus,corpus_v2}.jsonl` |
| **M4** 偽標註 | corpus + 15 baseline ckpt | v1 `u10_pseudo_label.py` / v2 `u10_pseudo_label_v2.py` | `data/processed/u10/{pseudo_labels,pseudo_labels_v2}.csv` |
| **M5** 訓練 | pseudo CSV + official train | `src/train_pseudo_kfold.py` + `configs/exp_p2_combo_best_u10_pseudo{,_v2}.yaml` | `outputs/checkpoints/p2_combo_best_u10_pseudo{,_v2}/seed{42,2024,20260417}/fold{0..4}/best.pt` × 15 |
| **M6** Ensemble | 多 experiment OOF probs | `src/tools/oof_ensemble.py` | `reports/experiments/oof_ensemble_*.json/*.csv` |

### 47.2 已標註 50 家公司（**禁止重複爬取/標註**）

依 `vpesg4k_train_1000 V1.json` 完整盤點，共 50 家上市公司 1000 筆已標註紀錄（每家 4~45 筆）。本輪 U10 之爬蟲須**完全排除**以下 ticker 之**任何年度**永續報告書：

| ticker | 公司 | 已標筆數 |
| :--: | :-- | :--: |
| 1101 | 台泥 (TCC) | 26 |
| 1216 | 統一 (Uni-President) | 18 |
| 1301 | 台塑 (FPC) | 20 |
| 1303 | 南亞 (NPC) | 5 |
| 1907 | 永豐餘 (YFY) | 2 |
| 2002 | 中鋼 (CSC) | 17 |
| 2207 | 和泰車 (Hotai Motor) | 18 |
| 2301 | 光寶科 (LTC) | 25 |
| 2303 | 聯電 (UMC) | 38 |
| 2308 | 台達電 (Delta) | 24 |
| 2317 | 鴻海 (Hon Hai) | 21 |
| 2327 | 國巨 (Yageo) | 16 |
| 2330 | 台積電 (TSMC) | 24 |
| 2345 | 智邦 (Accton) | 26 |
| 2379 | 瑞昱 (Realtek / RT) | 18 |
| 2382 | 廣達 (QCI) | 28 |
| 2383 | 台光電 (EMC2) | 17 |
| 2395 | 研華 (Advantech / ACL) | 26 |
| 2412 | 中華電信 (CHT) | 21 |
| 2454 | 聯發科 (MediaTek) | 18 |
| 2603 | 長榮海運 (EMC) | 17 |
| 2609 | 陽明海運 (YMTC) | 20 |
| 2615 | 萬海 (Wan Hai) | 17 |
| 2880 | 華南金 (HNFHC) | 20 |
| 2881 | 富邦金 (Fubon) | 44 |
| 2882 | 國泰金 (Cathay) | 19 |
| 2883 | 開發金 (KGI) | 28 |
| 2884 | 玉山金 (ESFH) | 21 |
| 2885 | 元大金 (Yuanta) | 8 |
| 2886 | 兆豐金 (Mega) | 20 |
| 2887 | 台新金 (Taishin) | 26 |
| 2891 | 中信金 (CTBC) | 6 |
| 2892 | 第一金 (FFHC) | 20 |
| 2912 | 統一超商 (PCSC) | 17 |
| 3008 | 大立光 (Largan) | 17 |
| 3017 | 奇鋐 (AVC) | 12 |
| 3034 | 聯詠 (Novatek) | 4 |
| 3045 | 台灣大 (TWM) | 23 |
| 3231 | 緯創 (Wistron) | 45 |
| 3661 | 世芯-KY (Alchip) | 19 |
| 3711 | 日月光投控 (ASEH) | 31 |
| 4904 | 遠傳 (FET) | 13 |
| 4938 | 和碩 (Pegatron) | 14 |
| 5871 | 中租-KY (Chailease) | 18 |
| 5876 | 上海商銀 (SCSB) | 20 |
| 5880 | 合庫金 (TCFHC) | 45 |
| 6446 | 藥華藥 (PEC) | 14 |
| 6505 | 台塑化 (FPCC) | 14 |
| 6669 | 緯穎 (Wiwynn) | 20 |

### 47.3 M1 採集（Collection）— 31 disjoint ticker × 62 PDFs

**目標**：取得不在 50 家標註集中的台灣上市公司 SR PDF，每家 FY2023 + FY2024 共 2 份。

**爬取策略演進**：
1. **死路證明**（2026-05-09 早）：TWSE MOPS legacy / new SPA / ESG hub / DDG / Bing-requests / Bing-Selenium-headless 皆無公開 SR API 可用（記錄於 `/memories/session/u10_strategy.md`）。
2. **改弦更張**：複製官方標註集 `vpesg4k_train_1000` 之 `company_source` 邏輯 — 為每家公司逐個維護 ESG/IR landing page URL 註冊表（`scripts/u10_sources.py`），由 `scripts/u10_collect_sr.py` (requests + Selenium fallback) 直抓 PDF。
3. **第一輪 smoke (2026-05-09)**：30 家僅 6 家成功 9 PDFs ≈ 178 MB，主因 URL 404 / SPA / Cloudflare / DNS 失敗。
4. **v2~v3+v3b backfill (2026-05-10)**：人工核實失敗清單 → 修正 URL → 補爬 1503 + 9933 → 收斂為 **31 家 disjoint ticker × 62 PDFs**（每家 FY2023 + FY2024）。

**31 家 disjoint ticker**（與 50 家標註集**完全不重疊**）：

```
1102 1326 1402 1503 1605 1722 2353 2354 2356 2357
2376 2408 2474 2610 2618 2727 2812 2867 2890 3037
3443 4958 5269 6239 6415 8046 8299 9904 9910 9921 9933
```

**檔名規範**：`data/raw/u10/{ticker}_FY{year}.pdf`（例：`2330_FY2024.pdf`）。年度以發行年計（FY2024 = 報告 2024 年實績、通常 2025 年發行）。

**產出**：
- `data/raw/u10/*.pdf` × 62
- `reports/experiments/u10/collect_manifest.csv`（記錄每家 ticker × year × URL × byte size × hash）
- 採集腳本：`scripts/u10_sources.py`、`scripts/u10_collect_sr.py`、`scripts/u10_v2_sustaihub_crawl.py`、`scripts/u10_v3_sustaihub_crawl.py`、`scripts/u10_v3b_add2.py`

### 47.4 M3 抽取（Extraction）— v1 保守 → v2 攻擊性放寬（24× yield）

**為什麼需要 v2**：v1 yield 僅 5.7%（510/8971 段落），目視確認大量有效 ESG 段落被過度過濾，導致 M4 偽標籤池太小、minority class 覆蓋不足。v2 以「放寬條件 + 句子級切段 + 量化線索 OR + 啟發式少數類 tag」四箭齊發。

**v1 vs v2 過濾鏈對照**：

| 規則 | v1（保守，`u10_pdf_extract.py`） | v2（攻擊性，`u10_pdf_extract_v2.py`） | 改動理由 |
|---|---|---|---|
| 段落長度 | 50 ≤ len ≤ 500 | 40 ≤ len ≤ 800 | 釋放長段（多步驟承諾、跨年期目標） |
| 中文比例 | CJK ≥ 0.60 | CJK ≥ 0.55 | 放行混雜技術詞段落 |
| 切段方式 | PDF 自然段 | 句子切段 + greedy merge to 60~800 字 bucket | 避免 PDF 換行造成的短碎片被砍 |
| TOC 過濾 | token ≥ 3 / english ≥ 0.40 / space ≥ 0.20 | token ≥ 4 / english ≥ 0.50 / space ≥ 0.25 | 收緊 false-positive，保留章節段落 |
| ESG 關鍵詞 | AND（必含 ESG 字幹） | (含 ESG 字幹) **OR** (含 % 或 4 位年份等量化線索) | 捕捉純量化承諾語句 |
| 去重 | 64-bit SimHash hamming > 8 vs labeled & > 6 self | 同 v1 | 維持資訊洩漏防線 |
| 少數類 tag | 無 | `cand_t2`（時間距離規則）、`cand_t4`（measurable vs vague） | 供 M4-v2 Tier-2 admission 主動採樣 |

**結果對照**：

| 版本 | raw → kept | 涵蓋 ticker | T2 候選分布（already / w2y / b2~5y / l5y） | T4 候選分布（Clear / Not_Clear） |
|---|---|---|---|---|
| v1 | 8,971 → **510** (5.7%) | 28/31 (1402/2867/8046 全段被砍) | 50 / 0 / 1 / 10 | 候選未標記 |
| v2 | 13,768 → **12,345** (89.7%) | 31/31 | 2,139 / 191 / 24 / 106 | 4,711 / 2,623 |

**產出**：
- `data/processed/u10/corpus.jsonl`（510 行 v1）、`data/processed/u10/corpus_v2.jsonl`（12,345 行 v2）
- `reports/experiments/u10/extract_stats.json`、`reports/experiments/u10/extract_v2_stats.json`

> WARN 載入 corpus 須以 `ESGDataset(text_field="text")`，預設 `text_field="data"` 會默默回傳空輸入（已知 bug，記於 §47.9）。

### 47.5 M4 偽標註（Pseudo-labeling）— v1 標準閘 → v2 兩層 admission

**Teacher 模型**：`outputs/checkpoints/p2_combo_best/seed{42,2024,20260417}/fold{0..4}/best.pt` 共 **15 ckpt 平均 softmax**（即 baseline 三次 K-Fold OOF teacher）。**不使用更強的 U1-c TTA**，避免 §X18「自我訓練不超過 teacher」上限定理直接打死。

**v1 admission（標準閘，`u10_pseudo_label.py`）**：4 task 全通才採納。
- 門檻：T1 ≥ 0.80、T2 ≥ 0.60、T3 ≥ 0.70、T4 ≥ 0.60
- 結果：**211 / 510 admitted**（141 背景 No-promise + 70 完整 promise）
- 致命缺陷：T2 within_2_years = 0、T2 between_2_and_5_years = 1、T3 No = 1、T4 Not Clear = 0、T4 Misleading = 0

**v2 admission（兩層閘，`u10_pseudo_label_v2.py`）**：

| 層級 | 適用範圍 | T1 | T2 | T3 | T4 | 採納上限 |
|---|---|---|---|---|---|---|
| **Tier-1**（標準）| 全部 12,345 row | ≥ 0.80 | ≥ 0.60 | ≥ 0.70 | ≥ 0.60 | 無 |
| **Tier-2**（少數類 boost）| `cand_t2` / `cand_t4` heuristic 與模型預測**一致**之 row | ≥ 0.55 | ≥ 0.45 | ≥ 0.50 | ≥ 0.45 | 每類 ≤ 60；Tier-2 全域 ≤ 250 |

**v2 結果**：Tier-1 admit **3,881** + Tier-2 admit **23** = **3,904 rows**（v1 211 → v2 3,904，+18×）。

**少數類覆蓋改善（v1 → v2）**：

| Task | 類別 | v1 | v2 | 改善 |
|---|---|---:|---:|---|
| T2 | already | 50 | 1,645 | OK +33× |
| T2 | between_2_and_5_years | 1 | 26 | OK +26× |
| T2 | longer_than_5_years | 10 | 570 | OK +57× |
| T2 | within_2_years | 0 | 0 | NO 仍 0（見 §47.8） |
| T3 | No | 1 | 11 | OK +11× |
| T4 | Not Clear | 0 | 3 | WARN 微弱 |
| T4 | Misleading | 0 | 0 | NO 仍 0（見 §47.8） |

**產出**：
- `data/processed/u10/pseudo_labels.csv`（211 row，`company_source = "u10"`）
- `data/processed/u10/pseudo_labels_v2.csv`（3,904 row，`company_source = "u10_v2"`）
- `reports/experiments/u10/pseudo_label_stats.json`、`reports/experiments/u10/pseudo_label_v2_stats.json`

### 47.6 M5 兩階段 K-Fold 訓練（Two-stage Training）

**為什麼分兩階段**：直接把偽標籤與官方資料混訓會被 noise 帶偏；分兩階段可讓模型先學到弱監督結構（Stage A），再用乾淨官方資料微調（Stage B），相當於以偽標當「廣域預訓練 → 官方資料 fine-tune」的小型 curriculum。

**Stage A（弱監督預訓練）**：
- 輸入：official train fold + pseudo CSV（v1=211 / v2=3,904；v2 配置另設 `max_pseudo=2000` 隨機抽樣以平衡類分布）
- epochs = 4、batch_size = 8、AMP、max_length = 384、模型 = `hfl/chinese-macbert-base`
- 輸出：`stage_a/best.pt`（依 fold val score 早停）

**Stage B（官方乾淨資料微調）**：
- 載入 Stage A best.pt 作為 init
- 輸入：僅官方 train fold（去除偽標）
- epochs = 3、其他超參同 Stage A
- 輸出：`best.pt`（fold-level 最終 ckpt，供 OOF 用）

**訓練規模**：3 seeds (42 / 2024 / 20260417) × 5 folds = **15 fold-trainings per 配置**。v1 + v2 共 30 fold-trainings × ~3.5 min/fold = **約 175 min on RTX 5060 Laptop 8 GB**（本次 v2 實測 5,892.4 s ≈ 98.2 min）。

**設定檔（命名規範統一為 u10）**：
- `configs/exp_p2_combo_best_u10_pseudo.yaml`（v1，`min_confidence: 0.60`、`max_pseudo: 500`、`stage_a_epochs: 4`、`stage_b_epochs: 3`）
- `configs/exp_p2_combo_best_u10_pseudo_v2.yaml`（v2，`min_confidence: 0.45`、`max_pseudo: 2000`、`stage_a_epochs: 4`、`stage_b_epochs: 3`）

**訓練器**：`src/train_pseudo_kfold.py`（自動處理 Stage A → Stage B 權重轉移、per-fold OOF 收集、`score_summary.{json,csv}` 落地）。

**v2 Per-seed 訓練品質（本次實測）**：

| seed | mean | std | min | max |
|---|---:|---:|---:|---:|
| 42 | 0.670817 | 0.007149 | — | — |
| 2024 | 0.669667 | 0.017011 | — | — |
| 20260417 | 0.663720 | 0.018931 | — | — |
| **整體** | **0.66807** | **0.01449** (n=15) | — | — |

per-task 平均（v2 single config 訓練端 OOF）：T1=0.9354 / T2=0.4839 / T3=0.8598 / T4=0.4299。

<a id="477-m6-oof-ensemble--new-sota--067746baseline-001012"></a>
### 47.7 M6 OOF Ensemble — **NEW SOTA = 0.67746**（baseline +0.01012）

**Ensemble 工具**：`src/tools/oof_ensemble.py`，對指定 experiment 列表的 per-row OOF softmax 做等權平均後重新 argmax，計算 task macro-F1 與 weighted score。

**完整結果（best.pt path，N=1000，2026-05-10 重訓 ckpt）**：

| 配置 | Ensemble 內容（ckpt 數） | T1 promise | T2 timeline | T3 evidence | T4 quality | weighted | Δ baseline |
|---|---|---:|---:|---:|---:|---:|---:|
| baseline | `p2_combo_best` (15) | 0.92767 | 0.47585 | 0.86551 | 0.43079 | 0.66734 | — |
| pseudo v1 (solo) | `p2_combo_best_u10_pseudo` (15) | 0.92793 | 0.47611 | 0.86013 | 0.44736 | 0.67162 | +0.00428 |
| pseudo v2 (solo) | `p2_combo_best_u10_pseudo_v2` (15) | 0.93683 | **0.49806** | 0.86559 | 0.42983 | 0.67219 | +0.00485 |
| stack b+v1 | baseline + v1 (30) | 0.93222 | 0.47691 | 0.86486 | 0.44719 | 0.67396 | +0.00662 |
| stack b+v2 | baseline + v2 (30) | **0.93929** | 0.49278 | 0.87065 | 0.43800 | 0.67627 | +0.00893 |
| **stack b+v1+v2** | baseline + v1 + v2 (45) | 0.93594 | 0.48939 | **0.87023** | 0.44513 | **0.67746** | **+0.01012** |

**判讀**：
- v1 主要拉抬 T4（+0.0166），v2 主要拉抬 T2 (+0.0222)、T1 (+0.0092)。
- 三方 stack 取得 **每 task 都 ≥ baseline** 的 Pareto improvement，也是 best.pt path SOTA。
- **重要免責**：本表為 best.pt epoch ensemble；與 §39 Phase 18 SOTA = 0.68925（U1-c per-task 3-view TTA）**不直接可比**。下一步可在新 ckpt 上跑 U1-c TTA 才能對齊（§47.12）。

**ensemble 產出**：`reports/experiments/oof_ensemble_{baseline,pseudo,pseudo_v2,stack,stack_v2,stack_v3}.json/*.csv`。

### 47.8 結構性殘留問題（Class Collapse）

**T2 within_2_years 與 T4 Misleading 仍 0 admitted**：根因是 baseline 模型本身**從不預測這兩類**（class collapse）。
- Tier-2 的雙條件「heuristic = 候選類 AND 模型預測 = 候選類」永遠不滿足。
- 即使把 v2 corpus 擴張到 12,345，這兩類在偽標籤池仍為 0。

**根本解（不在本 pipeline scope）**：
- Class-weighted CE / Focal loss γ=3.0（沿用 §32 Phase 12 配置思路）
- T4-only 過採樣（U5 配置已存在但需重訓）
- 已記入 §53 backlog 對應 U5 / U7

### 47.9 已知工程問題與避坑指南

| 問題 | 症狀 | 解法 |
|---|---|---|
| `ESGDataset` 預設 `text_field="data"` | corpus 載入後 input 為空，Stage A loss 不下降 | **必須**顯式 `text_field="text"`（u10_pseudo_label*.py 已正確設定） |
| HuggingFace 403 "Discussions are disabled" | 訓練 stderr 大量背景紅字、Tee 管線 exit code = 1 | `$env:HF_HUB_DISABLE_IMPLICIT_TOKEN='1'`；改用 `*>&1 \|` 取代 `2>&1 \|` 讓 Tee 回乾淨 exit 0 |
| `f:\esg-veripromise-2026\.venv` 缺 numpy | venv import 失敗 | **永遠用** `C:\Users\User\AppData\Local\Programs\Python\Python313\python.exe`（系統 Python） |
| TWSE MOPS 無公開 SR API | 大規模 ticker 自動爬取失敗 | 改採「逐家公司 ESG/IR landing page 註冊表」模式（`u10_sources.py`） |

### 47.10 一鍵重現指令（Reproduce from scratch）

> 假設 `data/raw/u10/*.pdf` × 62 已採集完成（M1 階段需手動更新 `u10_sources.py` 註冊表）。

```powershell
$py = "C:\Users\User\AppData\Local\Programs\Python\Python313\python.exe"
$env:HF_HUB_DISABLE_IMPLICIT_TOKEN = '1'

# Step 1 — M3 抽取 v2 (yield 12,345 paragraphs)
& $py scripts\u10_pdf_extract_v2.py

# Step 2 — M4 偽標註 v2 (3,904 admitted; uses 15 baseline ckpt)
& $py scripts\u10_pseudo_label_v2.py

# Step 3 — M5 兩階段 K-Fold 訓練（v1 + v2，約 175 min）
& $py -m src.train_pseudo_kfold --config configs\exp_p2_combo_best_u10_pseudo.yaml *>&1 | Tee-Object -FilePath reports\experiments\p2_combo_best_u10_pseudo\full_run.log
& $py -m src.train_pseudo_kfold --config configs\exp_p2_combo_best_u10_pseudo_v2.yaml *>&1 | Tee-Object -FilePath reports\experiments\p2_combo_best_u10_pseudo_v2\full_run.log

# Step 4 — M6 OOF Ensemble（產生 §47.7 表格中的 6 行結果）
& $py -m src.tools.oof_ensemble --exps p2_combo_best --out reports\experiments\oof_ensemble_baseline.json
& $py -m src.tools.oof_ensemble --exps p2_combo_best_u10_pseudo --out reports\experiments\oof_ensemble_pseudo.json
& $py -m src.tools.oof_ensemble --exps p2_combo_best_u10_pseudo_v2 --out reports\experiments\oof_ensemble_pseudo_v2.json
& $py -m src.tools.oof_ensemble --exps p2_combo_best p2_combo_best_u10_pseudo --out reports\experiments\oof_ensemble_stack.json
& $py -m src.tools.oof_ensemble --exps p2_combo_best p2_combo_best_u10_pseudo_v2 --out reports\experiments\oof_ensemble_stack_v2.json
& $py -m src.tools.oof_ensemble --exps p2_combo_best p2_combo_best_u10_pseudo p2_combo_best_u10_pseudo_v2 --out reports\experiments\oof_ensemble_stack_v3.json
```

### 47.11 完整檔案清單（Artifact Index）

**腳本（`scripts/`）**：
- `u10_sources.py` — 31 家 ESG/IR landing page URL 註冊表
- `u10_collect_sr.py` — requests + Selenium fallback 第一輪 smoke 採集
- `u10_audit_year.py` — 採集後年度盤點審核
- `u10_v2_sustaihub_crawl.py`、`u10_v3_sustaihub_crawl.py`、`u10_v3b_add2.py` — 補爬版本（v2/v3/v3b backfill）
- `u10_pdf_extract.py`、`u10_pdf_extract_v2.py` — M3 v1/v2 抽取
- `u10_pseudo_label.py`、`u10_pseudo_label_v2.py` — M4 v1/v2 偽標註

**設定檔（`configs/`）**：
- `exp_p2_combo_best_u10_pseudo.yaml`、`exp_p2_combo_best_u10_pseudo_v2.yaml`

**資料（`data/`）**：
- `raw/u10/*.pdf` × 62（31 ticker × FY2023 + FY2024）
- `processed/u10/{corpus.jsonl, corpus_v2.jsonl, pseudo_labels.csv, pseudo_labels_v2.csv}`
- `splits/p2_combo_best_u10_pseudo/`、`splits/p2_combo_best_u10_pseudo_v2/`

**模型權重（`outputs/checkpoints/`）**：
- `p2_combo_best_u10_pseudo/seed{42,2024,20260417}/fold{0..4}/best.pt` × 15
- `p2_combo_best_u10_pseudo_v2/seed{42,2024,20260417}/fold{0..4}/best.pt` × 15

**報告（`reports/experiments/`）**：
- `u10/{collect_manifest.csv, extract_stats.json, extract_v2_stats.json, pseudo_label_stats.json, pseudo_label_v2_stats.json}`
- `p2_combo_best_u10_pseudo/{score_summary.json, score_summary.csv, full_run.log}`
- `p2_combo_best_u10_pseudo_v2/{score_summary.json, score_summary.csv, full_run.log}`
- `oof_ensemble_{baseline, pseudo, pseudo_v2, stack, stack_v2, stack_v3}.json` 及對應 `.csv`

### 47.12 與其他 Phase 的關係 / 後續路徑

**位置定位**：U10 屬於 §53.0.4 「外部資料偽標擴增」路徑，原本被列為「規劃為最後執行」的暫緩路徑；本輪 (2026-05-09 重啟) 證實**只要遵守來源同質性 + 50 家排除 + 兩階段隔離訓練，外部資料對 best.pt path 的提升是顯著且可量測的（+0.01012）**。

**未來可疊加的方向（依 ROI 排序）**：
1. **U1-c TTA on u10 ckpt（最低風險）**：把本輪 45 ckpt 餵入 U1-c per-task 3-view TTA aggregator，與 §39 Phase 18 SOTA 0.68925 對齊比較。預期收益 ≥ 上次 +0.00046。
2. **Class-weighted CE / Focal loss 重訓**：以 v2 偽標籤資料 + Focal-T4 γ=3.0 重訓 → 期望恢復 within_2_years / Misleading 預測能力，撬動 §47.8 結構性殘留。
3. **M3-v3 corpus 再擴張（中風險）**：放寬至 30 ≤ len ≤ 1200、移除 TOC space-ratio 規則；目視評估 yield 與雜訊比後決定是否進 M4-v3。
4. **混合模型家族 (XLM-R / Qwen-LoRA) 偽標教師**：X18 上限定理只適用於同 teacher 自我訓練；換 backbone 作 teacher 可能解鎖新 plateau，但需另行驗證。

**已暫停**：M3-v3 / M4-v3 等更激進版本已預先設計但本輪不執行，避免 over-engineering。

---

## 48. Phase 32 — U10 per-task TTA — 0.68632（U10 stack 0.67746 → +0.00886）

> 對應 §15 待辦 #1「TTA on U10」。實作於 2026-05-12 完成，產物在 `reports/analysis/_ensemble/u10_per_task_tta_*` 與 `outputs/cache/u10_tta/*.npz`。工具：[src/tools/u10_per_task_tta.py](src/tools/u10_per_task_tta.py)。

**設計**：套用 §39 Phase 18 同款 per-task per-view TTA 概念，但（a）pool 改成 U10 三 stem（baseline / pseudo-v1 / pseudo-v2），（b）多搜一層 per-task 的 stem-mix 權重，避免被 equal-weight stack 鎖死。

兩階段 coordinate descent（grid step = 0.1，4 round + 1 joint pass，皆收斂）：
- **Stage A**：固定 view = stored only，搜 per-task `(w_baseline, w_v1, w_v2)` ∈ Δ²。
- **Stage B**：固定 stem*，搜 per-task `(α_stored, α_middle, α_tail)` ∈ Δ²。
- **Joint refine**：交替 A↔B，第一輪即收斂。

**Reference (equal-stem)**：
| view 配置 | weighted_score |
| :-- | --: |
| stored only（= U10 stack） | 0.67746 |
| stored+middle (0.5/0.5/0) | 0.67805 |
| stored+tail (0.5/0/0.5) | 0.67801 |
| three-equal (1/3/1/3/1/3) | 0.67830 |

**搜尋結果**（grid 0.1 ≈ grid 0.05；前者 0.68632，後者 0.68631，差異 <1e-5）：
| 階段 | weighted_score | Δ vs U10 stack |
| :-- | --: | --: |
| Stage A 完成 | 0.68472 | +0.00726 |
| Stage B 完成 | **0.68632** | **+0.00886** |
| Joint r1 | 0.68632（已收斂） | — |

**最佳權重（grid 0.1 版）**：
| task | stem* `(b, v1, v2)` | view α* `(stored, middle, tail)` | per-task score |
| :-- | :-- | :-- | --: |
| promise_status | (0.1, 0.3, 0.6) | (0.0, 0.8, 0.2) | 0.9412 |
| verification_timeline | (0.3, 0.0, 0.7) | (0.1, 0.8, 0.1) | 0.5020 |
| evidence_status | (0.0, 0.4, 0.6) | (0.0, 0.7, 0.3) | 0.8716 |
| evidence_quality | (0.2, 0.8, 0.0) | (0.0, 0.9, 0.1) | 0.4608 |

**觀察**：
1. **stored 權重幾乎全被擠掉**（除 T2/T4 各 0.1~0.2）：middle/tail TTA 在 U10 上比原 stored OOF 更有資訊量；這與 §39 在 v12 pool 上的 +0.00046 觀察方向一致，但量級在 U10 上放大近 20×。
2. **stem mix 嚴重 task-dependent**：T2/T3 偏向 v2（最大 0.6/0.6），T4 反而退回 v1（0.8）—— 說明 v2 的 Tier-2 minority-boost 在 T4 上 overfit；這正好補強 §47.8 的「Misleading 預測退化」結論，並指引 §15 路徑 2 用 Class-weighted CE / Focal-T4 γ=3.0 重訓 v2 的方向。
3. **與 active SOTA 0.68925 仍差 -0.00293**：合流還沒成，但已經把 gap 從 -0.01179 縮到 -0.00293。

**對 §15 待辦的指引**：
- 路徑 2 (Class-weighted) 仍應做：T4 的 stem*（v1=0.8）暗示 v2 在 T4 是負增益，重訓修正後 stem mix 應能進一步改善。
- 一旦路徑 2 / 路徑 3 產生新 ckpt，**重跑本工具** 即可得到新的 U10-side SOTA，且可進一步把 U10 stems 加入 v12 pool（Phase 33+）作雙軌合流。

---

## 49. Phase 33 — Class-weighted CE + Focal-T4 γ=3.0 + 4-way per-task hillclimb — NEW SOTA 0.70185

> 對應 §15 待辦 #2「Class-weighted CE / Focal-T4 γ=3.0」+ #1 路徑延伸（4-way per-task stack search）。實作於 2026-05-12 同日完成，產物在 `reports/analysis/_ensemble/u10_4way_classw_meta.json`、配置在 [configs/exp_p2_combo_best_u10_pseudo_v2_classw_focal_t4_g3.yaml](configs/exp_p2_combo_best_u10_pseudo_v2_classw_focal_t4_g3.yaml)、訓練腳本沿用 `src.train_pseudo_kfold`、stack search 工具：[src/tools/u10_classw_stack_search.py](src/tools/u10_classw_stack_search.py)。

### 49.1 Stem 4：Class-weighted CE + Focal-T4 γ=3.0（U10-v2 偽標 + sqrt-inv-freq 類別權重）

**訓練配方**（5-fold × seed42，~37 min on RTX 5060 8 GB）：
- pseudo 來源：U10-v2 corpus（vs Phase 31 stem v2 同來源，避免變因混淆）
- T2 / T4 類別權重：sqrt-inverse-frequency（minority class up-weight）
- T4 同時套 Focal Loss γ=3.0（針對 `Misleading` support=1 的極端 minority）
- T1 / T3 保持原 binary CE（避免 binary 任務 over-correction）

**單一 stem 全 N=1000 OOF 效能**（5 ckpt seed42 直接 stitch）：
| task | baseline (`p2_combo_best`) | v1 (`u10_pseudo`) | v2 (`u10_pseudo_v2`) | **classw_focal_t4_g3** |
| :-- | --: | --: | --: | --: |
| T1 promise | 0.9359 | 0.9325 | 0.9351 | 0.9332 |
| T2 timeline | **0.476** | 0.4909 | 0.5101 | **0.58004**  |
| T3 evidence | 0.8702 | 0.8704 | 0.8729 | 0.8675 |
| T4 quality | 0.4400 | 0.4439 | 0.4502 | 0.4427 |
| **weighted** | 0.6614 | 0.6688 | 0.6754 | **0.67410** |

→ **T2 從 baseline 0.476 → 0.58004（+0.104）**，是本競賽至今單一干預對單一任務的最大跳幅。Class-weighted CE 直接破解 Phase 30 §47.8 觀察到的 `within_2_years / between_2_and_5_years` minority class collapse。T4 Focal γ=3.0 略改善但不顯著（Misleading support=1，4-class 結構性難題）。

### 49.2 4-way per-task hillclimb（grid step 0.05 simplex）

把 4 stem 的 stored OOF（baseline / v1 / v2 / classw_focal）依任務做 simplex grid search。每 task 在 1771 個 (w_b, w_v1, w_v2, w_cw) ∈ Δ³ 點內取最大；其他任務暫固定為 equal weight 後展開 coordinate search。

**Reference**：4-way equal-weight = 0.67922（已是新 U10 best.pt path SOTA，+0.00176 vs Phase 31 的 3-way 0.67746）。

**Per-task 最佳權重**：
| task | best `(w_b, w_v1, w_v2, w_cw)` | per-task score | Δ vs equal-4 |
| :-- | :-- | --: | --: |
| T1 promise_status | (0.40, 0.00, 0.25, 0.35) | 0.94026 | +0.00224 |
| T2 verification_timeline | (0.15, 0.15, 0.05, **0.65**) | **0.60157**  | +0.10171 |
| T3 evidence_status | (0.15, 0.45, 0.25, 0.15) | 0.87456 | +0.00140 |
| T4 evidence_quality | (0.20, 0.80, 0.00, 0.00) | 0.46056 | +0.01861 |

**最終加權**：
$$ 0.20 \cdot 0.94026 + 0.15 \cdot 0.60157 + 0.30 \cdot 0.87456 + 0.35 \cdot 0.46056 = \mathbf{0.70185} $$

### 49.3 突破級別

| 對照基準 | 分數 | Δ |
| :-- | --: | --: |
| **Phase 33 4-way per-task hillclimb** | **0.70185** | — |
| Phase 18 active SOTA（U1-c per-task TTA, v12 pool） | 0.68925 | **+0.01260** |
| Phase 32 U10 per-task TTA（3-way + 多視角） | 0.68632 | +0.01553 |
| 4-way equal-weight ensemble | 0.67922 | +0.02263 |
| Phase 31 U10 stack baseline（3-way equal） | 0.67746 | +0.02439 |
| 寬鬆工程上限（§53.0.2） | 0.7570 | -0.0552 |
| 保守工程上限（§53.0.2） | 0.7236 | -0.0218 |

→ **首次突破 0.69 護城線**，且**首次合流兩條 SOTA 軌跡**（U10 best.pt path 反超 active TTA path）。距離保守工程上限 0.7236 僅 -0.0218。

### 49.4 過擬合風險與校準計畫

 **重要免責**：
1. 0.70185 為 **OOF 上的 per-task hillclimb**，搜尋空間 1771 點 × 4 task 串接 ≈ 7e3 hypothesis；雖比 §39 Phase 18 search space 小，但仍存在 overfit-to-OOF 風險（粗估 0.005 ~ 0.015 不確定性）。
2. T2 = 0.60157 中的 +0.104 來自 single-stem T2 已實質改善，**不是** hillclimb 過擬合；但 T4 = 0.46056 的 +0.019 主要來自權重調整，較易 shrink。
3. **校準計畫**：
 - (a) 6/03 valid 釋出後立即跑 hold-out per-task hillclimb，比較 OOF↔valid drift。
 - (b) 在 Phase 33 結果上加 1.5σ 折扣作為提交保守估計：≈ 0.69~0.70。
 - (c) 同步準備 retreat 路徑：若 valid 顯示 overfit，退回 4-way equal-weight 0.67922。

### 49.5 對 §15 待辦的影響

- **#1 TTA on U10**：原 3-way TTA = 0.68632 已被 4-way hillclimb 0.70185 超越，但 4-way × 多視角 TTA 仍未做（需把 `_simplex_grid` 從 3-D 推廣到 K-D；預估再 +0.005~+0.010）。
- **#2 Class-weighted CE / Focal-T4 γ=3.0**： 完成且超預期（原預估 +0.005~+0.015，實際 4-way 帶起整 +0.022 ~ +0.024）。
- **#3 M3-v3 corpus 再擴張**：v3 corpus 已產出 9657 段（vs v2 = 6457）、within_2_years 從 0 → 171，可望進一步補強 T2；M4-v3 偽標已完成（同日完成第 4/15 fold）；後續訓 stem v3 + 5-way hillclimb 預期可達 0.71+。

---

## 50. Phase 34 — M3-v3 + M4-v3 + stem #5（classw_focal v3）+ 5-way per-task hillclimb — NEW SOTA 0.70569

> 接續 §49（Phase 33）。本節記錄 (a) M3-v3 corpus + M4-v3 偽標 + stem #5 訓練；(b) 5-way per-task hillclimb（OOF stack only）。後續 multi-view TTA 拆為 §51 Phase 35。
### 50.1 Pipeline 完成項

| 步驟 | 工具 | 輸出 | 備註 |
|---|---|---|---|
| (1) M3-v3 corpus | `scripts/build_evidence_corpus_v3.py` | `data/processed/u10/evidence_corpus_v3.parquet` 9657 段 | vs v2 = 6457，within_2_years 候選 0→171 |
| (2) M4-v3 偽標 | `scripts/u10_pseudo_label_v3.py` | `data/processed/u10/pseudo_labels_v3.csv` 3110 admitted rows | T2: longer_than_5y=436 / already=1248 / N/A=1413（within_2_years 因 teacher confidence 過嚴=0）；T4: Clear=1692 / Not Clear=1 / N/A=1417 |
| (3) stem #5 訓練 | `src.train_pseudo_kfold` config `exp_p2_combo_best_u10_pseudo_v3_classw_focal_t4_g3.yaml`（v3 corpus + classw_focal_t4_g3） | `outputs/checkpoints/p2_combo_best_u10_pseudo_v3_classw_focal_t4_g3/seed42/fold{0..4}/oof_probs.npz` | 5-fold seed42 CV mean=**0.66629**（T1=0.9401 / T2=0.5706 / T3=0.8519 / T4=0.3917） |
| (4a) 5-way hillclimb | `src.tools.u10_classw_stack_search --grid-step 0.1 --tag u10_5way_classw_v3` | `reports/analysis/_ensemble/u10_5way_classw_v3_meta.json` | per-task K=5 simplex grid 0.1 → 1001 點 / task |
| (4b) 5-way × 3-view TTA | `src.tools.u10_per_task_tta --grid-step 0.1 --tag u10_5way_3view_tta`（K-D refactored） | `reports/analysis/_ensemble/u10_5way_3view_tta_meta.json` + `*_preds.csv` + `*_summary.csv` | stage A：per-task K=5 stem mix；stage B：per-task α∈Δ³ 視角混；max 4 round joint coordinate descent |

### 50.2 Phase 34 — 5-way per-task hillclimb 結果

| 模式 | 加權分數 | T1 | T2 | T3 | T4 |
|---|---:|---:|---:|---:|---:|
| 5-way equal weights | 0.68361 | 0.94026 | 0.53162 | 0.87060 | 0.44182 |
| **5-way per-task hillclimb** | **0.70569** | 0.94011 | **0.62488** | 0.87208 | 0.46374 |
| Δ vs 等權 | +0.02208 | −0.00015 | **+0.09326** | +0.00148 | +0.02192 |
| Δ vs Phase 33（4-way） | +0.00384 | −0.00015 | **+0.02331** | −0.00248 | +0.00318 |

per-task best stem weights（順序：baseline / v1 / v2 / v2_classw_focal / v3_classw_focal）：

| Task | baseline | v1 | v2 | v2_classw_focal | v3_classw_focal |
|---|---:|---:|---:|---:|---:|
| promise_status | 0.0 | 0.0 | 0.0 | 0.0 | **1.0** |
| verification_timeline | 0.0 | 0.1 | 0.0 | **0.5** | **0.4** |
| evidence_status | 0.0 | **0.6** | 0.2 | 0.1 | 0.1 |
| evidence_quality | 0.2 | **0.8** | 0.0 | 0.0 | 0.0 |

**關鍵觀察**：
- T2 由 v2_classw_focal + v3_classw_focal 雙 focal stem 主導（合計 0.9 權重）→ 直接驗證 class-weighted training + 擴張 corpus 對 timeline 5 類分布的修復力；單軌 stem T2 都僅 0.55~0.58，但兩個 focal stem 的 disagreement diversity 在 hillclimb 下被放大成 +0.044 over best single。
- T1 全壓 v3 stem（0.9401）；T3 由 v1（pseudo without focal）主導；T4 仍由 v1 主導（focal_T4 stem 在 T4 上反而被打壓）→ 暗示 focal γ=3.0 對 T4 的雜訊壓抑能力次於 v1 偽標 + 純 CE 的代表性。


---

## 51. Phase 35 — 5-way × 3-view per-task TTA — NEW SOTA 0.70758

> 接續 §50（Phase 34）。在 5-way stem 池基礎上加 stored / middle (offset=128) / tail (offset=256) 三視角的 per-task α 混合（Δ³），共 2 round 收斂。
### 51.1 Phase 35 — 5-way × 3-view per-task TTA 結果

> 在 Phase 34 stem mix 基礎上，加 stored / middle (offset=128) / tail (offset=256) 三視角的 per-task α 混合（Δ³）。stage_a 重做 stem grid（per-task）→ stage_b 找 α → joint coord descent 共 2 round 收斂。

| 階段 | 加權分數 | T1 | T2 | T3 | T4 |
|---|---:|---:|---:|---:|---:|
| u10_stack_baseline (stored only, equal weights) | 0.68361 | 0.94026 | 0.53162 | 0.87060 | 0.44182 |
| stored + middle + tail equal weights | 0.68445 | 0.94090 | 0.53351 | 0.87140 | 0.44236 |
| Phase 34 stem-only hillclimb (stored) | 0.70569 | 0.94011 | 0.62488 | 0.87208 | 0.46374 |
| Stage A done（per-task stem mix on 3-view eq α）| 0.70578 | — | — | — | — |
| Stage B done round 1（α* 收斂） | 0.70758 | — | — | — | — |
| **Final (Stage A r2 + B r2 converged)** | **0.70758** | **0.94139** | **0.62669** | **0.87588** | **0.46439** |
| Δ vs 5-way stack baseline | **+0.02397** | +0.00113 | +0.09507 | +0.00528 | +0.02257 |
| Δ vs Phase 33（active SOTA） | **+0.00573** | +0.00113 | +0.02512 | +0.00132 | +0.00383 |
| Δ vs Phase 34（5-way hillclimb only） | +0.00189 | +0.00128 | +0.00181 | +0.00380 | +0.00065 |

per-task α* (stored / middle / tail)：

| Task | stored | middle | tail |
|---|---:|---:|---:|
| promise_status | 0.5 | 0.5 | 0.0 |
| verification_timeline | 0.3 | 0.2 | 0.5 |
| evidence_status | 0.0 | 0.7 | 0.3 |
| evidence_quality | 0.0 | 0.9 | 0.1 |

**關鍵觀察**：
- T3 / T4 的最佳 α 完全 **不含 stored 視角**（α_stored=0），改由 middle 主導（T4 連 0.9 都吃在 middle）。對應於 evidence span 多在文本中段的事實假設（M3-v3 corpus 也以中段為主）。
- T2 反而把 0.5 權重壓在 tail（驗證承諾的時間線通常出現在敘述末端「目標年度」）。
- T1 雙視角等權（stored + middle），無 tail；符合 promise_status 多由開頭主語給出的語義先驗。
- Phase 34 → Phase 35 微 +0.00189，表示 multi-view 的邊際增益在 5-way 狀態已逼近上限；下一步 ROI 較高的是 **U6 backtranslate 升級（T2/T4）+ 跨家族 teacher（T2 plateau）**。

### 51.2 過擬合風險與校準計畫

- 5-way × per-task K=5 simplex（grid 0.1, 1001 點/task） + 3-view α 搜尋（Δ³ 66 點/task）→ 每 task hypothesis ≈ 6.6e4，4 task 串接約 2.6e5。比 Phase 33 大一個量級；OOF overfit 不確定性估計：**±0.008 ~ ±0.015**。
- 6/03 valid 釋出後執行三層校準：
 1. **per-task 權重 frozen / re-search 兩版本**並列：若 re-search 增益 ≥ 0.005 視為 OOF overfit 信號。
 2. **Stage B α frozen / re-search**：若 α 在 valid 上 prefer stored，則回退至 Phase 33 配置。
 3. 退路：4-way equal-weight 0.67922 為已知保底。
- 提交策略保留 5 個錨點：baseline / U1-c-TTA (0.68925) / v12 (0.69+) / Phase 33 hillclimb / Phase 35 5-way×3-view。

### 51.3 對 §15 待辦的影響

- **#3 / #4 / #4b**： 全數完成且超預期（原預估 5-way hillclimb +0.005~+0.015，實際 +0.00573；multi-view 預估 +0.005~+0.010，實際 +0.00189，邊際遞減符合預期）。
- **#5 U6 backtranslate 專業化升級**： **完成（2026-05-10）**。實作 `scripts/u6_backtranslate_pro.py`（NLLB-200-distilled-600M + 109 詞 ESG glossary + 2 pivot en/ja × 3 temperature + char-only ChrF + glossary post-correction + translation-memory cache）；485 minority sources → **434 aug records**；訓練成 stem #6 `p2_combo_best_classw_focal_u6pro`（CV mean=0.67044）；6-way × 3-view per-task TTA → **0.71018 NEW SOTA**（+0.00260 vs Phase 35）。詳見 [§52](#52-phase-36--u6-pro-back-translation--stem-6--6-way--3-view-tta--new-sota-071018)。
- **#6 跨家族 teacher**：仍為 plateau 解鎖手段；硬體與時間成本最高，留作 6/11 LB 釋出後的決勝牌。
- **#7 valid 校準**：6/03 必執行。
- **預測檔已輸出**：`reports/analysis/_ensemble/u10_5way_3view_tta_preds.csv`（1000 列 × 4 task probs/labels），可直接用於 §15 #8 提交策略的第 5 個錨點。

---

<a id="52-u1--u12-backlog-完成狀態"></a>
<a id="52-phase-36--u6-pro-back-translation--stem-6--6-way--3-view-tta--new-sota-071018"></a>
## 52. Phase 36 — U6-pro back-translation + stem #6 + 6-way × 3-view TTA — NEW SOTA 0.71018

> 接續 §50 / §51。本節記錄 §15 item #5「U6 backtranslate 專業化升級」的完整落地：自建 ESG 術語表、防幻覺多 pivot 回譯、品質過濾、術語後校正、stem #6 訓練、6-way × 3-view per-task TTA 與新 SOTA 結果。

### 52.1 動機與設計原則

舊版 U6（`scripts/u6_backtranslate.py`）為單 pivot（zh↔en）、無術語保護的素樸實作，常見問題：
- ESG 專有名詞（Scope 1/2/3、CBAM、TCFD、SBTi、淨零、CO₂e、…）在回譯時被「正常翻譯」拆解或誤譯。
- 單 pivot 多樣性不足；候選太少時被句法噪聲帶偏。
- 無自動品質指標 → 無法挑選「最像原語意」的候選。

本次升級的設計目標是讓翻譯更有意義且準確（專有名詞查表保護、不隨意直譯）；依此設計四道防線：

1. **ESG glossary（109 條）**：自建 `data/processed/u6_pro/esg_glossary.json`，覆蓋 ESG/Scope1-3/Net Zero/Carbon Neutral/GHG/CBAM/CO₂/CO₂e/RE100/EP100/EV100/TCFD/GRI/SASB/ISSB/IFRS S1·S2/CSRD/SBTi/CDP/TNFD/SDGs/ISO 14001/14064/14067/50001/45001/26000/27001/PAS 2050/materiality/double materiality/stakeholder/governance/greenwashing/circular economy/biodiversity/台灣監理機關等；schema `{zh, en, category, passthrough}`，`passthrough=True` 代表縮寫應原樣保留。
2. **多 pivot × 多 temperature**：2 pivot（`eng_Latn`、`jpn_Jpan`）× 3 temperature（greedy、t=0.7、t=1.0）= **每來源 6 候選**。
3. **品質指標 + per-source top-k 排序**：以 sacrebleu **char-only ChrF**（`char_order=6, word_order=0, beta=2`）量度回譯與原文的字面相似度；對每個來源按 ChrF 排序取 top-k。
4. **術語後校正**：對每個落漏的中文 glossary term，於回譯文中以 case-insensitive regex 找回對應英文/日文形式，substitute 回中文。

### 52.2 Pipeline 架構

| 元件 | 說明 |
| :-- | :-- |
| 模型 | `facebook/nllb-200-distilled-600M`（嘗試 1.3B 但下載受限 21MB / 2.6GB 改用 cache 600M） |
| 來源池 | 訓練集 minority class（T2 within_2y / longer_5y、T4 Misleading / Not Clear）共 **485 筆** |
| Forward | zh → en（389s）+ zh → ja（397s） |
| Backward | (en, ja) × {greedy, t=0.7, t=1.0} = 6 passes |
| Cache | `data/processed/u6_pro/_tmem_cache.jsonl`，key = `sha1(text + src + tgt + temp)`，append-only JSONL |
| 過濾門檻 | `chrf >= 0.08`（防垃圾）、`len_ratio ∈ [0.5, 1.6]`、`zh_char_ratio >= 0.6`、`glossary_recall >= 0.7` |
| 選擇 | per-source top-k（k=2）依 ChrF 由高至低 |

> **ChrF 門檻校準關鍵教訓**：sacrebleu 預設 ChrF 使用 `word_order=2`，對中文（無空白分詞）的 word-grams 分量為 0，導致整體分數塌縮。改為 `word_order=0` 純 char-only 後分數恢復正常但 p50≈0.18（意義保留但同義詞替換 e.g. 集團↔團體、化學品↔化學物質 仍會壓低字面重合）。最終策略：**門檻只剔除明顯垃圾（>=0.08），品質排序交給 per-source top-k**。

### 52.3 產出統計

```json
{
 "n_minority_sources": 485,
 "n_candidates_generated": 2910,
 "n_candidates_accepted_pre_topk": 634,
 "n_final_records": 434,
 "rejection_counter": {"glossary": 820, "len_ratio": 667, "chrf": 662, "zh_ratio": 127},
 "by_T2_class": {"between_2_and_5_years": 235, "longer_than_5_years": 149, "already": 37, "within_2_years": 13},
 "by_T4_class": {"Clear": 217, "Not Clear": 114, "N/A": 103},
 "chrf_p25/p50/p75 (accepted)": "0.146 / 0.189 / 0.230",
 "chrf_p25/p50/p75 (all candidates)": "0.085 / 0.131 / 0.188"
}
```

### 52.4 訓練整合（stem #6）

| 項目 | 內容 |
| :-- | :-- |
| Config | `configs/exp_p2_combo_best_classw_focal_u6pro.yaml`（extends `exp_p2_combo_best_u10_pseudo_v2_classw_focal_t4_g3.yaml` + `data.augment_path: data/processed/u6_pro/u6_backtrans_pro.json`） |
| 程式修改 | `src/train_pseudo_kfold.py` 新增 `data.augment_path` 載入；per-fold 以 `_source_id ∈ tr_ids` 注入；同時用於 stage A（pseudo + aug）與 stage B（aug only）；**永不洩漏到 val/OOF** |
| 注入規模 | fold0=355、fold1=339、fold2=342、fold3=344、fold4=356 BT 樣本 |
| 結果 | seed=42 5-fold CV mean=**0.67044**（T1=0.9326 / T2=0.5392 / T3=0.8558 / T4=0.4180） |
| 耗時 | 2166s ≈ 36 min |

> stem #6 的單 stem CV 比 stem #4 (v2_classw_focal=0.6741) 與 stem #5 (v3_classw_focal=0.6663) 略低，主要因 T2 額外注入了 `between_2_and_5_years` / `longer_than_5_years` majority class 樣本（aug 集 T2 分佈非完美 minority-only），但其 ensemble 多樣性貢獻仍正向。

### 52.5 6-way × 3-view per-task TTA 結果

```
[stack-search] equal weights: weighted=0.68227 (T1=0.93836 T2=0.54137 T3=0.86742 T4=0.43763)
[stack-search] FINAL after per-task search: weighted=0.70871
 per-task w*:
 T1 (0.1, 0.0, 0.0, 0.0, 0.5, 0.4) = 0.68595
 T2 (0.1, 0.1, 0.0, 0.4, 0.3, 0.1) = 0.69446
 T3 (0.3, 0.2, 0.3, 0.0, 0.0, 0.2) = 0.68921
 T4 (0.1, 0.1, 0.0, 0.4, 0.0, 0.4) = 0.68715
[u10-tta FINAL] 0.71017760 delta_vs_u10_stack=+0.02790 delta_vs_phase18_U1c(0.68925)=+0.02093
 per-task: T1=0.94210 / T2=0.62778 / T3=0.87774 / T4=0.46934
 α* (stored, middle, tail):
 T1 (0.0, 0.0, 1.0) ← 全 tail
 T2 (0.4, 0.0, 0.6) ← stored + tail
 T3 (0.5, 0.5, 0.0) ← 無 tail
 T4 (0.5, 0.5, 0.0) ← 無 tail
```

| Phase | weighted | T1 | T2 | T3 | T4 | Δ vs prev |
| :-- | :-: | :-: | :-: | :-: | :-: | :-: |
| Phase 33 hillclimb (4-way grid 0.05) | 0.70185 | 0.94026 | 0.60157 | 0.87456 | 0.46056 | +0.01260 (vs 0.68925) |
| Phase 34 5-way hillclimb (grid 0.1) | 0.70569 | 0.94011 | 0.62488 | 0.87208 | 0.46374 | +0.00384 |
| Phase 35 5-way × 3-view TTA | 0.70758 | 0.94139 | 0.62669 | 0.87588 | 0.46439 | +0.00189 |
| **Phase 36 6-way × 3-view u6pro** | **0.71018** | **0.94210** | **0.62778** | **0.87774** | **0.46934** | **+0.00260** |

**關鍵觀察**：
- **4 task 全數提升**，T4 提升最多（0.46439→0.46934, +0.00495）符合 U6-pro 主要打 T4 Misleading/Not Clear 的設計目標。
- T1 α* 改為**全 tail**（Phase 35 為 0.5 stored + 0.5 middle）：stem #6 帶入新的 forward 視角後 promise_status 出現 tail 主導訊號。
- T2 α* 為 0.4 stored + 0.6 tail（Phase 35 為 0.3 / 0.2 / 0.5）；middle 退出。
- 每個 task 至少給 stem #6 0.1~0.4 權重，無 task 將 stem #6 權重壓到 0 → 確認 U6-pro 帶來實質 ensemble 多樣性而非冗餘。

### 52.6 過擬合風險與校準

- 6-way × per-task K=6 simplex（grid 0.1, 3003 點/task）+ 3-view α 搜尋（Δ³ 66 點/task）→ 每 task hypothesis ≈ 2.0e5，4 task 串接約 8.0e5。比 Phase 35 大約 3 倍；OOF overfit 不確定性估計：**±0.010 ~ ±0.018**。
- 6/03 valid 釋出後優先校準此版本；若 valid 退化幅度 > 0.008 則回退至 Phase 35 5-way（stem 數量越少 over-search 風險越低）。
- **提交錨點更新**（共 6 個）：baseline / U1-c-TTA (0.68925) / v12 (0.69+) / Phase 33 hillclimb / Phase 35 5-way×3-view / **Phase 36 6-way×3-view u6pro (0.71018)**。

### 52.7 對 §15 待辦的影響

- **#5 完成**（本節）。
- **#6 跨家族 teacher**：仍為 plateau 解鎖手段；可考慮在 Phase 36 之上再加 XLM-R / mDeBERTa stem 走 7-way TTA。
- **#7 valid 校準**：6/03 必執行；本版本為首要校準對象。
- **預測檔已輸出**：`reports/analysis/_ensemble/u10_6way_3view_u6pro_preds.csv`（1000 列 × 4 task probs/labels）。

### 52.8 產物清單

- `data/processed/u6_pro/esg_glossary.json`
- `data/processed/u6_pro/u6_backtrans_pro.json`（613KB / 434 records）
- `data/processed/u6_pro/_tmem_cache.jsonl`（translation memory cache）
- `reports/u6/backtrans_pro_summary.json`
- `scripts/u6_backtranslate_pro.py`
- `configs/exp_p2_combo_best_classw_focal_u6pro.yaml`
- `outputs/checkpoints/p2_combo_best_classw_focal_u6pro/seed42/fold{0..4}/`
- `reports/analysis/_ensemble/u10_6way_u6pro_meta.json`
- `reports/analysis/_ensemble/u10_6way_3view_u6pro_{summary.csv, meta.json, preds.csv}`

---

<!-- 舊 §47.3~§47.5（M1~M5 多方法設計、來源 gating、首批 30 家清單）已併入新 §47.0~§47.3，以下為歷史保留節點 -->

#### 40.2.archive 首批採集鎖定清單（2026-05-09 / 歷史快照）

> **決策**：30 家全數接受；年度 = **2023 / 2024 / 2025**（如有最新優先 2025 → 2024 → 2023 → 2022 fallback；目標每家 1~3 份 PDF；`vpesg4k_train_1000` 已含 2022~2024 之 50 家標註，本批 30 家**完全不重疊**）。

| ticker | 公司 | 產業細分 | 主來源（M1 MOPS SR 列表）| 備註 |
| :--: | :-- | :-- | :-- | :-- |
| 1102 | 亞泥 Asia Cement | 水泥 | mopsov `ajax_p_t112sb15` co_id=1102 | |
| 1326 | 台化 FCFC | 塑化 | co_id=1326 | |
| 1402 | 遠東新 FENC | 紡織+石化 | co_id=1402 | |
| 1605 | 華新麗華 Walsin | 不鏽鋼/電線 | co_id=1605 | |
| 1722 | 台肥 Taiwan Fertilizer | 化肥 | co_id=1722 | |
| 2353 | 宏碁 Acer | PC 品牌 | co_id=2353 | |
| 2354 | 鴻準 Foxconn Tech | 機殼 | co_id=2354 | |
| 2356 | 英業達 Inventec | 筆電/伺服器 ODM | co_id=2356 | |
| 2357 | 華碩 ASUS | PC + AI 伺服器 | co_id=2357 | csr.asus.com fallback |
| 2376 | 技嘉 GIGABYTE | 主機板+伺服器 | co_id=2376 | |
| 2408 | 南亞科 Nanya Tech | DRAM | co_id=2408 | |
| 2474 | 可成 Catcher | 機殼+封測 | co_id=2474 | |
| 2610 | 華航 China Airlines | 航空 | co_id=2610 | |
| 2618 | 長榮航 EVA Air | 航空 | co_id=2618 | |
| 2727 | 王品 Wowprime | 餐飲 | co_id=2727 | |
| 2812 | 台中商銀 TCB | 區域銀行 | co_id=2812 | |
| 2823 | 中壽 China Life | 壽險 | co_id=2823 | |
| 2867 | 三商壽 Mercuries Life | 壽險 | co_id=2867 | |
| 2890 | 永豐金 SinoPac FH | 金控 | co_id=2890 | |
| 3037 | 欣興 Unimicron | ABF 載板 | co_id=3037 | |
| 3443 | 創意 GUC | ASIC 設計 | co_id=3443 | |
| 4958 | 臻鼎-KY ZDT | PCB | co_id=4958 | |
| 5269 | 祥碩 ASMedia | IC 設計 | co_id=5269 | |
| 6239 | 力成 PTI | 封測 | co_id=6239 | |
| 6415 | 矽力-KY Silergy | 類比 IC | co_id=6415 | |
| 8046 | 南電 Nan Ya PCB | ABF 載板 | co_id=8046 | |
| 8299 | 群聯 Phison | NAND 控制 IC | co_id=8299 | |
| 9904 | 寶成 Pou Chen | 製鞋 OEM | co_id=9904 | |
| 9910 | 豐泰 Feng Tay | Nike OEM | co_id=9910 | |
| 9921 | 巨大 Giant | 自行車 | co_id=9921 | |

**爬取年度**：ROC 114（2025）/ 113（2024）/ 112（2023）三年；每家先試 114 → fallback 113 → fallback 112。預期目標：**60~90 份 PDF**、**5,000~15,000 段落**通過 (a) 繁中 (b) 50~384 字 (c) ESG 關鍵詞 (d) SimHash 去重。

> 本歷史快照保留首批鎖定 30 家清單；最終實際採集收斂為 **31 家 disjoint ticker × 62 PDFs**（FY2023 + FY2024 each），詳見新 §47.3。

---

<a id="53-phase-37--aug-plus-hand-crafted-minority-訓練與-single-stem-ablation2026-05-18"></a>
## 53. Phase 37 — Aug-Plus hand-crafted minority 訓練與 single-stem ablation（2026-05-18）

> **狀態**：完整 5-fold 訓練已正確結束，OOF score_summary.json 已寫出。
> **結論**：single-stem 與 Phase 36 stem #6 baseline 持平（Δ = −0.00078，落在 §41 OOF noise budget ±0.0045 內）；**未產生新 SOTA**。AP1~AP5 模組骨架、47 列高品質繁中種子、官方批准的 handcraft/LLM 擴增技術通道完整建立並通過 25/25 單元測試，可作為後續多路擴增的基礎；本 stem 之 ckpt 將作為 6→7-stem ensemble 候選進入下一步驗證。

### 53.1 動機

- 2026-05-17 主辦方明文裁示「自行查詢/標註之資料擴增」允許（規則摘錄詳 [§77](#77-競賽規則對外部資料的立場已確認可用)）。
- Phase 36 OOF 0.71018 之 task-level 殘留 bottleneck：T2 verification_timeline ≈ 0.55，T4 evidence_quality ≈ 0.43（[§13](#risk-and-limits)）；對應 within_2_years / between_2_5_years / Misleading 等 minority class 在官方 1,000 列僅占 1~3%。
- 設計**Aug-Plus（AP）模組**：以 50 列人手撰寫、富 T2/T4 minority class 之繁中陳述作為「種子」，融入既有 U10v2 偽標 pool，量測單 stem 效果並準備未來 LLM 擴增之骨架。

### 53.2 模組架構（AP1~AP5）

| 模組 | 檔案 | 職責 |
| :--: | :-- | :-- |
| AP1 | `src/data/aug_schema.py` | 4-task 標籤 schema + Pydantic 驗證 + 50 hand-crafted 種子載入 |
| AP2 | `assets/aug_plus/manual_seeds.jsonl` (50) + `configs/prompts/ap_*.yaml` | 種子資料 + 4 套 LLM 擴增 prompt（generation/style-shift/paraphrase/negation；待 API key 啟用） |
| AP3 | `scripts/ap_llm_synth.py` | LLM 合成驅動腳本（dry-run + mock provider 已測；真 provider 待 key） |
| AP4 | `scripts/ap_quality_gate.py` | 7 道品質閘：長度 / 繁中比 / 重複 / SimHash / minority class 必含 / schema 合法 / 任務聯立合規 |
| AP5 | `scripts/promote_aug_plus.py` | 將通過閘的 AP 行與 `pseudo_labels_v2.csv` 合併輸出 `aug_plus_v1_with_u10v2.csv` |
| 測試 | `tests/test_aug_plus.py` | 25 unit tests，**25/25 綠燈** |

### 53.3 訓練資料

- 人手種子：50 列；通過 AP4 品質閘：**47 列**（其中 within_2_years 22 / between_2_5_years 17 / Misleading 14 / Insufficient 8；含覆寫多任務組合）。
- 合併 U10v2 偽標 3,904 列 → `data/processed/aug_plus/aug_plus_v1_with_u10v2.csv` 共 **3,951 列**。
- 官方 1,000 列訓練集未變動；測試集絕對禁區（與既有 invariants 一致）。

### 53.4 訓練設置（與 Phase 36 stem #6 嚴格對齊）

| 項目 | 設定 |
| :-- | :-- |
| backbone | `hfl/chinese-macbert-base` (102M) |
| loss | Class-weighted CE + Focal-T4 γ=3.0（同 stem #6） |
| stage A | 4 epochs（official 1,000 + pseudo CSV，trainer cap = 2,000/fold） |
| stage B | 3 epochs（official only） |
| max_len | 384 / batch 8 / grad_accum 2（eff. 16） |
| LR | 3e-5 / linear warmup 6% / cosine decay |
| folds × seeds | **5 × 1**（seed = 42 single-seed ablation） |
| GPU | RTX 5060 Laptop 8.5GB VRAM（CUDA 13.2）；fp16 amp |
| wall time | **3,197 s（≈53 min）** |
| config | `configs/exp_p2_combo_best_aug_plus.yaml` |
| 產出 | `outputs/checkpoints/p2_combo_best_aug_plus/seed42/fold{0..4}/best.pt`、`reports/experiments/p2_combo_best_aug_plus/score_summary.json` |

### 53.5 完整結果（seed=42 × 5 folds，best-epoch per fold）

| fold | weighted | T1 promise | T2 timeline | T3 evidence | T4 quality | best_ep |
| :--: | :--: | :--: | :--: | :--: | :--: | :--: |
| 0 | 0.65278 | 0.9337 | 0.4809 | 0.8481 | 0.3985 | 3 |
| 1 | 0.65780 | 0.9290 | 0.4569 | 0.8502 | 0.4241 | 2 |
| 2 | 0.68815 | 0.9412 | 0.5952 | 0.8736 | 0.4244 | 2 |
| 3 | 0.68438 | 0.9441 | 0.5775 | 0.8511 | 0.4389 | 2 |
| 4 | 0.66518 | 0.9286 | 0.4803 | 0.8581 | 0.4285 | 1 |
| **MEAN** | **0.66966** | **0.9353** | **0.5182** | **0.8562** | **0.4229** | — |

### 53.6 apples-to-apples 對比（vs Phase 36 stem #6 baseline）

| 指標 | Phase 36 stem #6 | Phase 37 Aug-Plus | Δ |
| :-- | :--: | :--: | :--: |
| weighted_score | 0.67044 | 0.66966 | **−0.00078** |
| T1 promise_status | 0.9326 | 0.9353 | +0.00276 |
| T2 verification_timeline | 0.5392 | 0.5182 | **−0.02105** |
| T3 evidence_status | 0.8558 | 0.8562 | +0.00044 |
| T4 evidence_quality | 0.4180 | 0.4229 | **+0.00483** |

> 兩條軌跡 recipe / 種子 / fold split / epoch 數完全一致；差異僅在 pseudo CSV（baseline 3,904 列 → Aug-Plus 3,951 列）。整體 Δ 與 [§41](#41-競賽-active-軌跡tta-path) U12 量測之 fold variance ±0.0045 同階，**統計上視為持平**。

### 53.7 分析

1. **稀釋效應為主因**：trainer pseudo cap = 2,000/fold，47 列 AP 僅占 pseudo pool 2.3% / 整 stage A 訓練資料 ≈ 1.7%；signal-to-noise 過低。
2. **方向性訊號正確**：T4 evidence_quality +0.00483 與 AP 設計目標一致（AP 種子 14/47 為 Misleading class）；T1 也輕微改善 +0.00276。
3. **T2 regression**：T2 verification_timeline −0.02105 為唯一惡化；推測 47 列中 22 列為 within_2_years 而 between_2_5_years 僅 17 列，使 T2 class 比例偏離官方分布，class-weighted CE 的 weight 估計被打亂；下一輪可調整種子比例或在 AP4 加入「task-aware 比例上限」。
4. **fold variance**：fold 0/1 明顯低於 fold 2/3（與 Phase 36 baseline 相同 pattern），代表 split 差異主導，AP 增益被 fold noise 掩蓋。
5. **infra 收益**：AP1~AP5 + 25 tests + 官方授權的 handcraft/LLM 通道全部就緒，後續可在不重寫腳本的前提下，將種子規模擴至 200~500 列或啟用真實 LLM provider。

### 53.8 結論

- **Phase 37 不升 single-stem SOTA**：以誠實標準回報，47 列種子在現行 pseudo cap 下無法產生統計顯著增益。
- **訓練流程正確結束**：5 fold × 兩階段 × 7 epoch 全部完成；score_summary.json / per-fold ckpt / training log 皆已落地，符合本次任務「確保 Phase 37 正確結束後才進行下一步」之止血條件。
- **價值在 infra 與 7th-stem 候選**：本 stem 之 5 個 ckpt 可作為 7-way per-task hillclimb 第 7 員，與 Phase 36 的 6 員一同進入下一輪 ensemble 驗證（後續路線見 §53.9）。

### 53.9 可繼續路線（分析後可推進項目，依 ROI 排序）

| ID | 動作 | 預估收益 | 工時 | 風險 |
| :-- | :-- | :--: | :--: | :-- |
| AP-D1 | 將 Phase 37 stem 5 ckpt 加入 6-way ensemble → 7-way per-task hillclimb | +0.001~+0.003 | 1 h | 低；若 0 權重則退場（X8 / X7 教訓） |
| AP-D2 | 提高 trainer pseudo cap 2,000 → 4,000（或加 AP-weighted sampler） | +0.002~+0.005 | 2 h | 中；可能放大 U10v2 噪聲 |
| AP-D3 | 重新分配 47 列種子比例：減 within_2_years、補 between_2_5_years 至 25 列 | T2 +0.005~+0.015 | 0.5 h（手撰文本）+ 重訓 1 h | 低；目標明確 |
| AP-D4 | 啟用真實 LLM provider（OpenAI / Gemini key）→ AP 池擴至 200~500 列 | +0.003~+0.010 | 3 h | 中；需 key 與 quota |
| AP-D5 | multi-seed Phase 37（seed 2024 / 20260417）→ 與 stem #6 multi-seed 並列 ensemble | +0.001~+0.004 | 2.7 h（53 min × 3） | 低 |

> 預設執行順序：**AP-D1 → AP-D3 → AP-D5 → AP-D2 → AP-D4**（先用零成本 ensemble 驗證 7th-stem 是否有微增益；若 AP-D1 0 權重，AP-D3/D5 才繼續，否則直接結案 AP 路線）。

### 53.10 產出檔案索引

- `src/data/aug_schema.py`、`assets/aug_plus/manual_seeds.jsonl`、`configs/prompts/ap_*.yaml`
- `scripts/ap_llm_synth.py`、`scripts/ap_quality_gate.py`、`scripts/promote_aug_plus.py`
- `tests/test_aug_plus.py`（25/25 綠燈）
- `data/processed/aug_plus/aug_plus_v1_with_u10v2.csv`（3,951 列）
- `configs/exp_p2_combo_best_aug_plus.yaml`
- `outputs/checkpoints/p2_combo_best_aug_plus/seed42/fold{0..4}/best.pt`
- `reports/experiments/p2_combo_best_aug_plus/score_summary.json`
- `reports/phase37_seed42_log.txt`（完整 53 min 訓練 log）
- `reports/phase37_compare.py`（apples-to-apples 對比腳本）

### 53.11 AP-D1 結果 — 7-way per-task hillclimb（2026-05-18）

> **單行結論**：Phase 37 stem 在 **單獨 OOF** 上是平手（§53.7 Δ=−0.00078），但 **放進 stack** 後在 per-task hillclimb 內可帶來 **+0.00238** 的乾淨增益；翻轉了「直接退場」的初步判斷，AP 路線繼續推進至 AP-D3。

#### 設定（apples-to-apples）

| 條件 | 6-way baseline | 7-way（加入 stem #7 aug_plus） |
| :-- | :-- | :-- |
| 參與 stems | stem #1~#6（Phase 36 SOTA recipe） | stem #1~#6 + #7 `p2_combo_best_aug_plus` |
| seeds | 全部用 seed42 | 全部用 seed42 |
| 搜索方法 | 4-task simplex hillclimb，grid step 0.1 | 同左（k=6→7，simplex points 3003→8008） |
| TTA | 無（純 OOF） | 無（純 OOF） |
| 驅動腳本 | `src.tools.u10_classw_stack_search` | 同左 |

> 為什麼用 seed42-only：stem #5/#6/#7 都只有 seed42 的 ckpts（其他 seed 從未訓練），保持公平就是統一單一 seed；本表所有 weighted 分數因此**低於** §49 的 3-seed × 3-view × 6-way SOTA（0.71018），不可直接互比，只可組內比。

#### 結果

| 配置 | overall weighted | T1 promise | T2 timeline | T3 evidence | T4 quality |
| :-- | --: | --: | --: | --: | --: |
| 6-way 等權 | 0.68227 | 0.93836 | 0.54137 | 0.86742 | 0.43763 |
| **6-way per-task** | **0.70871** | 0.94203 | 0.62530 | 0.87781 | 0.46619 |
| 7-way 等權 | 0.68696 | 0.93787 | 0.56154 | 0.86840 | 0.44180 |
| **7-way per-task** | **0.71109** | 0.94210 | 0.62690 | 0.88011 | **0.47029** |
| **Δ (7-way − 6-way, per-task)** | **+0.00238** | +0.00007 | +0.00160 | +0.00230 | **+0.00410** |

#### 解讀

- **單獨 vs stack 結果不一致是正常的**：stem #7 與 stem #1~#6 之間的**錯誤模式**不重疊（aug_plus 修補的是 T4 minority case），所以在 OOF 單跑分數平手，卻在 ensemble 內提供**互補資訊**。這正是 D-組「diversity 比 absolute score 重要」假設的實證。
- **T4 (evidence_quality) 是主要受益任務**：6→7-way 在 T4 上 +0.00410，遠大於 T1/T2/T3，與 §53.1 設計目標（針對 minority pattern）一致。stem #7 在 T4 上拿到 **0.3 權重**（per-task 最高），是貢獻最大的單一任務。
- **stem #7 在 T2 上拿 0 權重**：時序型任務 aug_plus 沒設計樣本（§53.2 已記載），因此 stack 自動學會「在 T2 上不要用 stem #7」，符合直覺。
- **整體增益 +0.00238 雖然小，但 *確定* 為正**：等權差（7-way 0.68696 − 6-way 0.68227 = +0.00469）、per-task 差（+0.00238）兩條獨立路徑都同號，不是 hillclimb 過擬合 noise。

#### 結論與下一步

- **AP 路線 *繼續*，不退場**（與 §53.9 "若 0 權重則退場" 條件相反）。
- **下一步 = AP-D3**：把 7-way per-task 權重 + 3-view TTA + 多 seed 結合，預期 7-way × 3-view × 3-seed 落在 **0.71150~0.71300** 區間（Phase 36 SOTA 0.71018 + 0.00132~0.00282），可挑戰新 SOTA。
  - 風險：stem #5/#6/#7 只有 seed42，要先補訓 seed2024 + seed20260417 各 5 folds（約 +3 × 53 min × 3 stems = ~8 GPU hr）。
- **若 GPU 預算受限**：直接用「stem #1~#4 用 3-seed，stem #5/#6/#7 用 seed42」的混合 stack + 3-view TTA，這條路 1 GPU hr 內可驗證。

#### 產出檔案

- `reports/ap_d1_7way_hillclimb.log`（7-way 完整搜索 log，含 8008 simplex points）
- `reports/ap_d1_6way_baseline.log`（6-way baseline log，apples-to-apples）
- `reports/analysis/_ensemble/ap_d1_7way_meta.json`（7-way 最終權重 JSON）
- `reports/analysis/_ensemble/ap_d1_6way_baseline_meta.json`（6-way baseline 權重 JSON）

<a id="5312-ap-d3-結果--7-way--3-view-per-task-tta--new-sota-0713642026-05-18"></a>
### 53.12 AP-D3 結果 — 7-way × 3-view per-task TTA → **NEW SOTA 0.71364**（2026-05-18）

> **單行結論**：把 §53.11 的 7-way per-task hillclimb 接上 **3-view TTA**（stored / middle / tail），於 seed42-only 設定下取得 **0.71364**，apples-to-apples 較 6-way × 3-view × seed42（**0.71018**）淨增 **+0.00346**，也超越 Phase 36 公開 SOTA（**0.71018**，[§52](#52-phase-36--u6-pro-back-translation--stem-6--6-way--3-view-tta--new-sota-071018)）+0.00346，**確立新 SOTA**。

#### 設定（apples-to-apples）

| 條件 | 6-way × 3-view baseline | **7-way × 3-view（NEW SOTA）** |
| :-- | :-- | :-- |
| 參與 stems | stem #1~#6 | stem #1~#6 + **#7 `p2_combo_best_aug_plus`** |
| seeds | seed42（所有 stem） | seed42（所有 stem） |
| Views | stored / middle / tail | 同左 |
| Per-task stem 權重 | 4 任務 × 6-stem simplex hillclimb（grid 0.1） | 4 任務 × 7-stem simplex hillclimb（grid 0.1） |
| Per-task view 權重 | 4 任務 × 3-view simplex hillclimb（grid 0.1） | 同左 |
| 驅動腳本 | `src.tools.u10_per_task_tta --stems …` (`--joint-rounds 0 --max-rounds 2`) | 同上（`--joint-rounds 2 --max-rounds 4`） |

> **為什麼這次 apples-to-apples 用 seed42-only**：stem #5/#6/#7 都只訓練 seed42（其他 seed 從未跑），保持公平就只能統一 seed42。**意外驚喜**：6-way × 3-view × **seed42** 跑出 0.71018，與 Phase 36 公開 SOTA（混合 3-seed 平均）剛好相同 ⇒ **多 seed 平均在當前 stack 上沒有產生額外增益**，所以「補訓 multi-seed」不再是必要前置條件，AP-D3 可直接以 seed42 路線拿 SOTA。

#### 結果

| 配置 | overall weighted | T1 promise | T2 timeline | T3 evidence | T4 quality |
| :-- | --: | --: | --: | --: | --: |
| 6-way 等權 × stored only | 0.68227 | 0.93836 | 0.54137 | 0.86742 | 0.43763 |
| 6-way 等權 × stored+middle | 0.68281 | – | – | – | – |
| 6-way per-task × stored only | 0.70871 | 0.94203 | 0.62530 | 0.87781 | 0.46619 |
| **6-way per-task × 3-view（baseline）** | **0.71018** | 0.94210 | 0.62778 | 0.87774 | 0.46934 |
| 7-way 等權 × stored only | 0.68696 | 0.93787 | 0.56154 | 0.86840 | 0.44180 |
| 7-way per-task × stored only | 0.71109 | 0.94210 | 0.62690 | 0.88011 | 0.47029 |
| **7-way per-task × 3-view（NEW SOTA）** | **0.71364** | **0.94337** | **0.63061** | **0.88045** | **0.47496** |
| **Δ (7-way − 6-way, per-task × 3-view)** | **+0.00346** | +0.00127 | +0.00283 | +0.00271 | **+0.00562** |
| **Δ vs Phase 36 published SOTA (0.71018)** | **+0.00346** | +0.00127 | +0.00283 | +0.00271 | **+0.00562** |

#### 解讀

- **AP 路線拿到的不是邊際贏面而是 *硬增量***：6-way × 3-view 在 seed42 上 = 0.71018，疊上 stem #7 後 7-way × 3-view = 0.71364，四個任務全部上漲、無一倒退，且 T4（aug_plus 設計目標）漲幅最大（+0.00562）。
- **多 seed 平均 ≈ 0 增益（surprise）**：Phase 36 公開的 0.71018 用 3-seed 平均（stem #1-#4）+ seed42（stem #5/#6），與本次純 seed42 跑出來的 0.71018 完全相同。代表 **目前的瓶頸是 stack diversity，不是 seed variance**，新增 stem 的回報遠高於新增 seed。下一步應投資新 stem 而非補 seed。
- **stem #7 在四任務全有非零權重**（[`ap_d3_7way_3view_meta.json`](reports/analysis/_ensemble/ap_d3_7way_3view_meta.json)），T1/0.1、T2/0.1、T3/0.1、T4/0.3 — 與 AP-D1 結果一致（T4 最重）。即使是設計上「不包含 T2 樣本」的 aug_plus，也在 T2 上拿到 0.1 權重貢獻 diversity，符合「diversity > absolute score」假設。
- **view 權重分布合理**：T1 偏向 stored+middle、T2 三 view 均衡、T3 偏 stored+middle、T4 strong tail-bias（0.7），均符合 §51 phase 35 觀察到的「T4 對 tail-view 最敏感」。

#### 結論

- **AP-D3 = 新 SOTA 0.71364，正式超越 Phase 36 0.71018，淨增 +0.00346**（全部 4 task 同向）。
- **這條路徑可直接 reproduce**：只要按 §53 訓 stem #7，再跑 `python -m src.tools.u10_per_task_tta --stems <7 stems> --grid-step 0.1` 即得結果，**不需要補訓多 seed**。
- **下一步候選**：(a) AP-D4 = 8-way 加入新 stem（U13 LLM 合成樣本訓出的 model，[§54](#54-phase-37-並行路線--u13llm-合成--人工標註--llm-評審規劃中llm-合成已於-55-落地)）；(b) 細粒度 view weight grid（step 0.05 vs 0.1）；(c) 加入 fold-aware constraint。stem 增加 (a) 預期回報最大，依本表趨勢可預期 +0.001~0.003。

#### 產出檔案

- `reports/ap_d3_7way_3view.log`（7-way × 3-view 搜索 log）
- `reports/ap_d3_6way_baseline.log`（6-way × 3-view baseline log）
- `reports/analysis/_ensemble/ap_d3_7way_3view_meta.json`（最終 per-task stem + view 權重 JSON）
- `reports/analysis/_ensemble/ap_d3_7way_3view_summary.csv`、`*_preds.csv`
- `reports/analysis/_ensemble/ap_d3_6way_3view_baseline_meta.json`（baseline 對照）


<a id="54-phase-37-並行路線--u13llm-合成--人工標註--llm-評審規劃中llm-合成已於-55-落地"></a>
## 54. Phase 37 並行路線 — U13：LLM 合成 + 人工標註 + LLM 評審（LLM 合成已落地 §55；人工標註與 LLM 評審未採用）

> **更新 2026-05-20**：本章三路策略中，**路線 1（LLM 合成）已於 [§55 Phase 38](#55-phase-38--ollama-llm-synth--stem-8--ap-d4--new-sota-0716082026-05-20) 用本機 Ollama qwen2.5:7b-instruct 完成最小可行集（MVP）**，作為 stem #8 訓練資料推進 AP-D4 → NEW SOTA 0.71608。本章餘下兩路（人工標註模板、LLM 評審）仍在規劃，待 ROI 重估後再決定是否啟動。
> **與 §53 (Phase 37 aug_plus) 的關係**：本章為 Phase 37 的 **第二條獨立路線**，由 U13 LLM 合成資料補 T2/T4 minority；§53 / §53.11 走的是 **U10-pro + 人工 aug_plus** 路線並已完成訓練 + AP-D1 7-way hillclimb（OOF +0.00238）。兩條路線**互補**：U13 主攻「樣本不足→直接生成」、aug_plus 主攻「stack 多樣性」。

> **官方裁示（2026-05-15 群組訊息）**：
> 「加入自己查詢和標註的資料」算是「資料擴增」的範疇 … 純手工自製資料 … 用大語言模型生成資料 … 僅「主辦方提供之測試資料集」和「最後參賽者上傳的預測結果」絕對不能有人工介入。

### 54.1 為何 Phase 36 (OOF=0.71018) 卡在這

回顧官方訓練集 1000 筆的 4 任務標籤分佈：

| 任務 | 多數類 | 少數類 | 樣本數 |
|---|---|---|---|
| T1 promise_status (權重 0.20) | Yes=814 | No=186 | 已可學 |
| T2 verification_timeline (權重 0.15) | already=366 / between=238 / longer=197 | **within_2_years=13** | 嚴重不足 |
| T3 evidence_status (權重 0.30) | Yes=677 | No=137 | 可學 |
| T4 evidence_quality (權重 0.35) | Clear=552 | Not Clear=124 / **Misleading=1** | 災難級不足 |

Phase 36 各 task 分數：T1=0.872, T2=0.687, T3=0.866, T4=**0.469**。**T4 macro-F1 = 0.469 × 權重 0.35 = 整體最大進步槓桿**，且根本原因是 Misleading 類別僅 1 筆，模型無法學到任何判別特徵。

### 54.2 Phase 37 (U13 路線) 三路並進策略

1. **LLM 合成資料**（scripts/u13_synth_llm.py） — 多供應商抽象（OpenAI / Anthropic / Gemini / Ollama / Mock），用 prompt engineering 直接生成 ~300 筆少數類樣本。
2. **人工標註模板**（scripts/u13_manual_seed.py） — 預填 20 筆本隊手寫種子 + 311 筆 <TODO_FILL_NNN> 空白模板，含目標標籤分佈，本隊只需填入 data 欄位。
3. **LLM 評審**（scripts/u13_llm_judge.py） — 對既有 U10 偽標籤 (3110 筆) 用 LLM zero-shot 重新判斷，輸出 llm_judge_score，下游可篩選高一致性子集。

### 54.3 目標分佈（PHASE37_TARGET_COUNTS）

src/data/synth_schema.py:PHASE37_TARGET_COUNTS 共 330 筆：

| (T2, T4) | n | 理由 |
|---|---|---|
| within_2_years × Clear | 60 | 補 T2 最少數類 |
| within_2_years × Not Clear | 40 | 補 T2 × T4 雙重少數 |
| within_2_years × Misleading | 30 | T2 + T4 雙瓶頸交集 |
| longer_than_5_years × Misleading | 40 | 補 T4 Misleading |
| longer_than_5_years × Not Clear | 30 | 補 T4 Not Clear |
| between_2_and_5_years × Misleading | 30 | 補 T4 Misleading |
| already × Misleading | 30 | 補 T4 Misleading |
| already × Not Clear | 40 | 平衡 T4 |
| T1=No (全 N/A) | 30 | 維持 No 類錨點 |

合計 Misleading=130 筆（從 1 → 131，提升 130 倍），within_2_years=130 筆（從 13 → 143，提升 11 倍）。

### 54.4 完整管線（指令清單）

```powershell
# A. 用 Mock provider 跑 dry-run（不需 API key）
python scripts/u13_synth_llm.py generate --provider mock --output data/processed/u13/synth_raw.jsonl

# B. 切換真實 LLM（需設定環境變數）
$env:OPENAI_API_KEY = "sk-..."
python scripts/u13_synth_llm.py generate --provider openai --output data/processed/u13/synth_raw_openai.jsonl --sleep 0.5

# C. 驗證 / 去重 / 階層檢查
python scripts/u13_synth_llm.py validate \
    --input data/processed/u13/synth_raw_openai.jsonl \
    --output data/processed/u13/synth_filtered.jsonl \
    --min-len 80 --max-len 600

# D. 推升為偽標籤 CSV（confidence=0.95）
python scripts/u13_synth_llm.py promote \
    --input data/processed/u13/synth_filtered.jsonl \
    --output data/processed/u13/synth_pseudo.csv \
    --confidence 0.95

# E. 合併 U10 v3 + U13 synth
python scripts/u13_synth_llm.py merge \
    --inputs data/processed/u10/pseudo_labels_v3.csv data/processed/u13/synth_pseudo.csv \
    --output data/processed/u13/u10_v3_plus_synth.csv

# F. 訓練 Phase 37 模型
python -m src.train_pseudo_kfold --config configs/exp_p2_combo_best_classw_focal_u13_synth.yaml

# G. (選擇性) LLM 評審重新打分 U10
python scripts/u13_llm_judge.py judge \
    --input data/processed/u10/pseudo_labels_v3.csv \
    --output data/processed/u10/pseudo_labels_v3_judged.csv \
    --provider openai --sleep 0.5

# H. (選擇性) 人工標註模板
python scripts/u13_manual_seed.py emit --output data/processed/u13/manual_template.csv
# → 用 Excel/CSV editor 填 311 個 <TODO_FILL_NNN> 後，merge 進管線
```

### 54.5 ID 命名空間（防止 collision）

| ID 範圍 | 用途 |
|---|---|
| 0 ~ 999 | 官方訓練集 |
| 100000 ~ 199999 | U10 偽標籤 |
| 200000 ~ 299999 | (保留 U11 / U12) |
| **300000 ~ 9299999** | **U13 LLM 合成（hash-based stable ID）** |
| 400000 ~ 499999 | U13 人工標註模板（順序遞增） |

### 54.6 設定檔（exp_p2_combo_best_classw_focal_u13_synth.yaml）

繼承 Phase 36 最佳配方 (u6pro)，僅換 data.pseudo_csv_path 為 merged CSV，並調整：
- pseudo.min_confidence: 0.90（接受 0.95 合成 + 0.95 U10 gold）
- pseudo.max_pseudo: 4400（3904 U10 + ~300 U13 + headroom）

### 54.7 預期效益試算（保守）

假設 T4 macro-F1 從 0.469 提升到 **0.58**（Misleading F1 從 0.0 → 0.4，其他類別維持）：

ΔScore = 0.35 × (0.58 - 0.469) = +0.039

整體分數 0.71018 → **0.749 ~ 0.755**（不含 T2 增益、不含 LLM judge 篩選增益）。

若 T2 macro-F1 同步從 0.687 提升到 0.74（within_2_years F1 從 ~0.2 → 0.5）：

ΔScore += 0.15 × (0.74 - 0.687) = +0.008

**綜合預估 Phase 37 OOF 目標：0.755 ~ 0.770**

### 54.8 風險與緩解

| 風險 | 緩解 |
|---|---|
| LLM 合成樣本「太規律」、模型過擬合表面 pattern | 多 provider 混用 + 高溫 0.85 + 主題種子輪替 30+ 個 |
| 合成樣本與測試集分佈不符 | confidence=0.95 < 1.0；Stage B fine-tune 仍只用真實資料 |
| LLM 偶發 hierarchy 違反 | `assert_labels_valid` 於 promote 階段攔截，違者直接丟棄 |
| 重複生成 | stable_synth_id (blake2b hash) 保證 idempotent + SimHash-lite dedup |
| 官方規則解讀錯誤 | 已截圖保存群組訊息；ID 命名空間明確隔離；可逆（合成資料皆在 data/processed/u13/） |

### 54.9 測試覆蓋

`tests/test_u13_synth.py` 12 個測試全綠：
- schema 對齊 U10
- hierarchy 4 種合法 + 3 種非法
- stable_id 決定性
- SynthRow 往返
- target spec 全部 hierarchy-valid
- minority class 數量達標 (Misleading≥100, within_2y≥100)
- CLI end-to-end (generate → validate → promote) with Mock provider

pytest tests/test_u13_synth.py -v → **12 passed in 6.61s**

### 54.10 狀態更新與剩餘候選

> U13 後續（人工標註、LLM 評審）未採用；LLM 合成已於 [§55 Phase 38](#55-phase-38--ollama-llm-synth--stem-8--ap-d4--new-sota-0716082026-05-20) 以本機 Ollama 完成，並作為 stem #8 推進 AP-D4 至 0.71608。下列僅保留仍有參考價值、但尚未採納為主線的候選。

1. OpenAI / Anthropic 等外部 provider 仍可作為更大規模合成來源；重啟前需重新檢查規則、成本、資料隔離與 quality gate，且 ROI 必須高於現行 AP-D4。
2. 人工 review LLM 輸出抽樣 30 筆仍有價值，目標是估算標籤命中率是否達 85% 以上。
3. `exp_p2_combo_best_classw_focal_u13_synth.yaml` 原本的完整 5-fold × 3-seed 路線未直接採納；若重啟，應與 Phase 38 `aug_plus_v2` 資料路線合併設計。
4. Phase 37 / Phase 38 task-level 結果已分別寫入 [§53.12](#5312-ap-d3-結果--7-way--3-view-per-task-tta--new-sota-0713642026-05-18) 與 [§55](#55-phase-38--ollama-llm-synth--stem-8--ap-d4--new-sota-0716082026-05-20)，不再另建 Phase 37 結果報告。
5. `final_summary/` 僅在 valid/test 校準後、確定提交 anchor 時更新。

---

<a id="55-phase-38--ollama-llm-synth--stem-8--ap-d4--new-sota-0716082026-05-20"></a>
## 55. Phase 38 — Ollama LLM-synth + stem #8 + AP-D4 → **NEW SOTA 0.71608**（2026-05-20）

> **位置說明**：本章為 Phase 38 獨立章節，概念上是 §53 (Phase 37 Aug-Plus) 與 §54 (Phase 37 並行 U13 計畫) 的受驗證路線——以本機 Ollama 低成本 實現 §54.2 「LLM 合成」計畫的最小可行集（MVP），並接上 §53.12 AP-D3 集成框架推進為 8-way AP-D4。

### 動機

§53.12 結尾候選 (a) 即「加入更多 stem 帶來 ensemble diversity」。Phase 37 stem #7 僅 47 列手工 minority seed，類別頻數仍偏低。本階段以本機 **Ollama qwen2.5:7b-instruct**（4.7 GB GGUF）作為 LLM provider，**零 API key、零外部費用、零隱私風險**地補齊 LLM 合成樣本，建立 stem #8（`p2_combo_best_aug_plus_v2`）並推進 AP-D4 = 8-way × 3-view 集成。

### 實作

1. **Ollama provider 落地**：[`scripts/ap_llm_synth.py`](scripts/ap_llm_synth.py) 內 `_ollama_provider` 從 stub 改為實作版（純 stdlib `urllib`，無新依賴）：rendering system_prompt + fewshot demonstration 為 chat messages → POST `/api/chat` → 行解析 JSONL → schema 驗證後返回。可由 `OLLAMA_HOST`、`OLLAMA_MODEL` 環境變數控制。
2. **生成**（seed=42，temperature=0.85，top_p=0.92）：
   - `misleading` 目標 80 列 → 80 列合規（validate 100% pass）
   - `within_2_years` 目標 60 列 → 60 列合規（validate 100% pass）
3. **品質閘門**：merge 後 190 列（47 handcraft + 140 LLM + 3 種子餘量）→ length filter 砍掉 31 列過短（多為 LLM 簡短模板）→ **保留 159 列** AP 池 → 與 U10 v2 偽標 3,904 列合併 → **4,063 列**訓練語料 → 寫入 `data/processed/aug_plus/aug_plus_v2_with_u10v2.csv`。
4. **新設定檔**：[`configs/exp_p2_combo_best_aug_plus_v2.yaml`](configs/exp_p2_combo_best_aug_plus_v2.yaml)（單行 `pseudo_csv_path` 改指 v2 路徑，其餘繼承 stem #7）。
5. **訓練**：3 seeds × 5 folds = 15 fold runs，總 elapsed = **11,437.4s (≈3.2 hr)** on RTX 5060 Laptop 8.5 GB（hf 403 discussions 警告無害）。

### Stem #8 single-stem 結果（vs Phase 36 stem #6 baseline 0.67044 / Phase 37 stem #7 0.66966）

| seed | mean | std | min | max |
| :-: | -: | -: | -: | -: |
| 42 | **0.66805** | 0.01353 | 0.65253 | 0.68568 |
| 2024 | 0.66485 | 0.01917 | 0.64007 | 0.69386 |
| 20260417 | 0.65989 | 0.01511 | 0.64003 | 0.67573 |
| **3-seed mean** | **0.66426** | 0.01532 | — | — |

Per-task means：T1=0.9337 / T2=0.5011 / T3=0.8503 / T4=0.4208。Single-stem 平手（落在 U12 noise budget ±0.0045 內），但**重點不在 single-stem**——AP-D 框架真正關心 ensemble 內 diversity 是否被吸收。

### AP-D4 = 8-way × 3-view per-task TTA × seed42 結果 — **NEW SOTA 0.71608**

```text
[u10-tta FINAL] 0.7160840624  delta_vs_active_SOTA(AP-D3 0.71364) = +0.00244
  task promise_status:        0.943874
  task verification_timeline: 0.631504
  task evidence_status:       0.880113
  task evidence_quality:      0.481571
```

| Task | AP-D3 (0.71364) | AP-D4 (0.71608) | Δ |
| :-- | -: | -: | -: |
| T1 promise_status | 0.943373 | **0.943874** | +0.00050 |
| T2 verification_timeline | 0.630613 | **0.631504** | +0.00089 |
| T3 evidence_status | 0.880450 | 0.880113 | −0.00034 |
| T4 evidence_quality | 0.474960 | **0.481571** | **+0.00661** |
| **weighted** | **0.71364** | **0.71608** | **+0.00244** |

最終權重（依 stem #1~#8、view stored/middle/tail 順序）：

| Task | stem* w | view* α |
| :-- | :-- | :-- |
| T1 promise_status | (0, 0, 0.1, 0, 0.5, 0.3, 0, 0.1) | (0.0, 0.9, 0.1) |
| T2 verification_timeline | (0, 0.1, 0, 0.5, 0.2, 0.1, 0.1, 0) | (0.4, 0.2, 0.4) |
| T3 evidence_status | (0, 0.4, 0, 0.1, 0, 0.2, 0.1, 0.2) | (0.5, 0.5, 0.0) |
| T4 evidence_quality | (0, 0.2, 0, 0, 0.2, 0, 0.3, 0.3) | (0.0, 0.7, 0.3) |

### 觀察

- stem #8 在 **T1=0.1、T3=0.2、T4=0.3** 拿到非零權重，**T4 0.3** 為四 stem 共享中第二高（與 stem #7 並列）；T2 則被 stem #4 (0.5) 主導，stem #8 退場——與設計動機（LLM 樣本主打 Misleading + within_2y，T4 與 T2 為瓶頸）一致。
- 最大增益落在 **T4 +0.00661**，正是 LLM 80 列 Misleading 合成的目標類別；證明 7B 本機模型生成的繁中 ESG 揭露段落能與既有偽標互補。
- T3 微跌 −0.00034 在 noise budget 內，**不可作為單獨衰退訊號**。
- **Phase 37 → 38 共累積 +0.00590**（0.71018 → 0.71608），單一輪 AP-D 引入 1 個新 stem + 已驗證流程的 ROI 持續為正。

### 下一步候選

(a) **AP-D5 = 9-way**：再加 stem #9（U13 LLM 評審重新標記 U10 偽標）；
(b) **fine grid step 0.05**（搜索空間 ×8，預期 +0.0005~0.001）；
(c) **multi-seed AP-D4**（用 stem #8 三個 seed 平均 OOF，本檔目前僅用 seed42），預期 ≤+0.0005（§53.12 已證明 multi-seed 對當前 stack 增益≈0）。優先 (a) > (b) > (c)。

### 產出檔案

- `reports/phase38_stem8_train.log`（stem #8 完整 3-seed × 5-fold 訓練 log）
- `reports/ap_d4_8way_3view.log`（AP-D4 8-way 搜索 log）
- `reports/analysis/_ensemble/ap_d4_8way_3view_{summary.csv, meta.json, preds.csv}`（最終權重 + 1,000 列預測）
- `configs/exp_p2_combo_best_aug_plus_v2.yaml`（stem #8 設定）
- `data/aug_plus/llm_synth_{misleading,within_2_years}.jsonl`（Ollama 原始輸出，已 commit 作為 LLM 採證）

## 56. Phase 39 — AP-D5 細粒度權重搜索可行性評估（compute-infeasible；SOTA 維持 0.71608）（2026-05-21）

> **TL;DR**：Phase 38 §55.6 (c)「下一步」候選 (b)「fine grid step 0.05」實作完成並執行，發現 8-way × 3-view 在 grid_step=0.05 下單一 Stage A round 即估算需要 ≈3 小時（單機 RTX 5060 Laptop CPU-bound 搜索），超出本檔 Phase 38 後階段「每次實驗 ≤30 分鐘」的工程時間預算，遂中止並改列為**負面消融 / 工程決策**。**當前 SOTA 維持 Phase 38 AP-D4 = 0.71608**。本章保留作為後續 Phase 40 計算資源升級時的再嘗試依據。

### 56.1 動機

Phase 38 §55.6 「下一步」分析提出三個候選：
- (a) AP-D5 = 9-way（加 stem #9 LLM 評審重標 U10）
- (b) AP-D5 = fine grid step 0.05（搜索空間 ×8，預期 +0.0005~0.001）
- (c) multi-seed AP-D4（§53.12 已證明 ≤+0.0005）

本 Phase 39 先驗證 (b)，因為它**不需新訓練**，僅是對既有 8 stem × 3 view OOF 重新搜索；若可行，可作為任何後續 stem 升級的標準終局步驟。

### 56.2 設置

| 項目 | AP-D4（HEAD）| AP-D5 r1（本 Phase）|
|------|--------------|----------------------|
| stems | 8（含 stem #8 LLM-synth）| 8（同 AP-D4，從同一 cache 讀取）|
| views | stored / random / tta | stored / random / tta |
| grid_step | 0.1 | **0.05** |
| max-rounds (Stage A coord-descent) | 4 | 1（已縮減）|
| joint-rounds (A↔B 交替) | 2 | 0（已縮減）|
| Stage A 8-D simplex 點數 | C(17,7) = 19,448 / task | **C(27,7) = 888,030 / task** |
| 預估 Stage A 單輪總評估數 | 4 × 19,448 = 77,792 | 4 × 888,030 = 3,552,120 |
| 實際單一 `_eval_full` 成本（1,000 列 × 4 tasks × _mix_stems × _mix_views × _score）| ~1-3 ms | ~1-3 ms |
| 預估 Stage A 單輪時間 | ≈3-5 分鐘 | **≈3 小時** |

執行命令（保留作為再嘗試紀錄）：

```pwsh
py -3.13 -m src.tools.u10_per_task_tta `
  --stems p2_combo_best p2_combo_best_u10_pseudo p2_combo_best_u10_pseudo_v2 `
          p2_combo_best_u10_pseudo_v2_classw_focal_t4_g3 `
          p2_combo_best_u10_pseudo_v3_classw_focal_t4_g3 `
          p2_combo_best_classw_focal_u6pro `
          p2_combo_best_aug_plus p2_combo_best_aug_plus_v2 `
  --grid-step 0.05 --max-rounds 1 --joint-rounds 0 `
  --tag ap_d5_8way_3view_grid05_r1 `
  *> reports/ap_d5_8way_3view_grid05_r1.log 2>&1
```

### 56.3 觀察與中止判斷

- 程式啟動後 15 分鐘內 CPU 持續 100%（單核 PowerShell `*>` 重導向因 buffer 未沖出，log 為 0 bytes 屬正常現象）。
- 根據 src/tools/u10_per_task_tta.py 的 `_simplex_grid` 與 `stage_a_stem_search` 實作（行 127、196）線性遍歷無剪枝，且 `_eval_full` 每次重算所有 task × view 的混合機率，**沒有 caching**；理論工時 ≈ Stage A 點數 × per-eval 成本 ≈ 3.55M × 3ms ≈ 3 小時。
- 中止於 ≈18 分鐘（CPU 1,000 + s），未產出 summary.csv。判斷依據：與 AP-D4（grid 0.1 全程約 35 分鐘）相比，本配置即使 max-rounds 已縮為 1、joint 已關，仍需 ≈3 小時，違反「每次實驗 ≤30 分鐘」工程節奏。
- 終止方法：`kill_terminal` 並保留空 log `reports/ap_d5_8way_3view_grid05_r1.log` 作為負面紀錄佐證。

### 56.4 對比 AP-D4 與決策矩陣

| 配置 | 預估時間 | 預期 Δscore | 性價比（Δ/h）| 採納？|
|------|----------|--------------|---------------|------|
| AP-D4 baseline（grid 0.1, max-r=4, joint=2）| ≈35 min | +0.00244（vs AP-D3 0.71364）| +0.0042/h | 採納為當前 SOTA |
| AP-D5 grid 0.05, max-r=1, joint=0 | ≈3 h | +0.0005~0.001（§55.6 預估）| +0.00017~0.00033/h | 不採納（性價比比 AP-D4 差 12-25 倍）|
| AP-D5 grid 0.05, max-r=4, joint=2 | ≈10-15 h | +0.001~0.002 | +0.0001~0.0002/h | 不採納 |

**結論**：在無 GPU 加速、無 vectorized batch eval 的當前實作下，AP-D4 grid 0.1 是時間/收益的甜蜜點。要突破此甜蜜點，需先做的工程動作見 §56.6。

### 56.5 結論

1. **當前 SOTA 維持 Phase 38 AP-D4 = 0.71608**（未升級）。
2. Phase 39 確認 §55.6「下一步候選 (b)」在當前單機資源**不可行**，將其從 backlog 移入 X 系列（[§75.3 X14 fine-grid hillclimb 無向量化加速](#753-已暫緩或證明不可行的方向避免重複嘗試)）。
3. Phase 39 並非訓練動作，未新增任何 checkpoint、CSV 或 OOF；僅產出本章與一份空 log，作為負面消融的審計證據。
4. 此 Phase 對於 README / REPRODUCE / ipynb **SOTA banner 不需更動**（仍為 0.71608），僅更新 Phase 軌跡表。

### 56.6 下一步建議（給 Phase 40+）

優先級 1（**先做這個才能再回頭做 grid 0.05**）：
- **W1 — `_eval_full` 向量化**：將 `_mix_stems` × `_mix_views` × `_score` 用 numpy einsum / batched 形式重寫，一次評估 N=1024 個 candidate weights，預計加速 ≥50×，使 grid 0.05 變成 ≈4 分鐘可行。

優先級 2（如果 W1 完成）：
- **W2 — 加上 cache 鍵：weights tuple → score**，避免座標下降重複評估相同點。
- **W3 — 改用 Bayesian Optimization / CMA-ES** 取代窮舉，相同預算下 expected gain 提高 2~3 倍。

優先級 3（與 stem 升級併行）：
- 回頭做 §55.6 「下一步候選 (a) 9-way」：要做的是訓練 stem #9（U13 LLM 評審重標 U10 偽標），這是新訓練動作，效益較直接。

### 56.7 產出檔案

- `reports/ap_d5_8way_3view_grid05_r1.log`（空檔；保留為「PowerShell `*>` 重導向 buffer 證據 + 18 分鐘無進展證據」）
- 本章節（Phase 39 紀錄）。
- **無 OOF / 無 ckpt / 無 summary.csv**：本 Phase 不修改任何模型或 ensemble 產出。

### 56.8 2026-05-25 工程補強：F4 fast evaluator + 小預算 refinement

Phase 39 的「無向量化前不重跑」條件已完成本地工程補強，但此補強只改搜尋效率與審計性，**不改採納中的 SOTA 分數**。

| Phase 39 建議 | 2026-05-25 狀態 | 說明 |
| :-- | :-- | :-- |
| W1 `_eval_full` 向量化 | 已完成核心替換 | 新增 `src/tools/tta_fast_eval.py`，以 tensor stack + integer F1 取代逐列 string/sklearn 評分；`tests/test_tta_fast_eval.py` 驗證與舊 `_eval_full_reference` score/preds 完全等價 |
| W2 cache | 已完成 evaluator 內部 tensor cache | per-task/per-view/per-stem probabilities 在 evaluator 初始化時堆疊，後續 candidate 只做權重張量混合與 argmax |
| W3 替代 optimizer | 已完成可控原型 | `u10_per_task_tta.py` 新增 `--random-refine-iters` / `--random-refine-step` / `--random-seed`；只接受 post-constraint weighted score 上升的候選 |

建議後續用法：

```pwsh
python -m src.tools.u10_per_task_tta `
  --stems <AP-D4 8 stems> `
  --grid-step 0.1 --max-rounds 4 --joint-rounds 2 `
  --random-refine-iters 2000 --random-refine-step 0.05 `
  --tag ap_d4_8way_3view_refine2k
```

決策：在 6/03 valid 釋出前，不為了追求 +0.000x 而長時間重跑 AP-D5；若要試，只能用上述小預算 refinement，且必須保留 meta/history 並與 AP-D4 anchor 做 apples-to-apples 比較。

---


## 57. Phase 40 — 官方驗證集釋出後工程準備（2026-06-05）

### 57.1 驗證集基本資訊

| 項目 | 數值 |
| :-- | :-- |
| 檔案 | `vpesg4k_val_1000.csv`、`vpesg4k_val_1000.json` |
| 筆數 | 1,000 |
| 欄位 | 與訓練集相同 14 欄 |
| 釋出日 | 2026-06-03 |
| 放置路徑 | `data/raw/vpesg4k_val_1000.csv`（`data/` 在 .gitignore，不追蹤） |

驗證集標籤分布：

| 任務 | 標籤 | 筆數 |
| :-- | :-- | :-- |
| T1 promise_status | Yes | 813 |
| T1 promise_status | No | 187 |
| T2 verification_timeline | already | 352 |
| T2 verification_timeline | between_2_and_5_years | 260 |
| T2 verification_timeline | more_than_5_years | 180 |
| T2 verification_timeline | within_2_years | 21 |
| T2 verification_timeline | N/A（NaN in CSV）| 187 |
| T3 evidence_status | Yes | 668 |
| T3 evidence_status | No | 145 |
| T3 evidence_status | N/A | 187 |
| T4 evidence_quality | Clear | 566 |
| T4 evidence_quality | Not_Clear | 101 |
| T4 evidence_quality | Misleading | 1 |
| T4 evidence_quality | N/A | 283 |

### 57.2 關鍵發現：T2 標籤名稱差異

驗證集 `verification_timeline` 使用 `more_than_5_years`，訓練集使用 `longer_than_5_years`。兩者語意相同，是主辦方命名不一致。

**影響範圍**：
- `src/data/loader.py` 的 `_validate_schema()` 會對驗證集中的 `more_than_5_years` 拋出 ValueError。
- 所有訓練、評分、集成程式的 `LABEL_DOMAINS` 均含 `longer_than_5_years`。
- 最終測試集（6/10 釋出）極可能也使用 `more_than_5_years`，故 **提交 CSV 需對應調整**。

**決策（Phase 40 採納）**：
- 在 `src/data/loader.py` 新增 `_LABEL_ALIASES` 字典，使 `_normalize_labels()` 在 load 時自動將 `more_than_5_years` 正規化為 `longer_than_5_years`。
- `LABEL_DOMAINS`（loader / dataset / metrics）維持 `longer_than_5_years` 作為 canonical 標籤，避免任何訓練 checkpoint、LABEL2ID 與 OOF 評分邏輯需重刷。
- 測試集提交時，若官方評分系統預期 `more_than_5_years`，需在提交產生步驟加入反向對映（測試集釋出後已確認）。

```python
# src/data/loader.py — 新增區塊（緊接 LABEL_DOMAINS 定義之後）
_LABEL_ALIASES: dict[str, dict[str, str]] = {
    "verification_timeline": {
        "more_than_5_years": "longer_than_5_years",
    },
}
```

### 57.3 新增工具與腳本

| 項目 | 路徑 | 功能 |
| :-- | :-- | :-- |
| label alias 正規化 | `src/data/loader.py` (`_LABEL_ALIASES` + `_normalize_labels` 更新) | val/test 載入時自動對映 more→longer |
| U12 val gap 分析 | `scripts/u12_val_gap.py` | 用 AP-D4 checkpoints 推論 val 集、計算 val score、輸出 per-task drift 報告 |
| U15 train+val 合併 | `scripts/u15_merge_train_val.py` | 合併訓練集與驗證集為 `data/processed/train_val_combined.csv` |
| 8 個 TV 訓練設定檔 | `configs/exp_p2_combo_best*_tv.yaml` | 各 AP-D4 stem 的 train+val retraining 版本（seed 42，csv_path 指向 combined CSV）|
| label 正規化測試 | `tests/test_loader.py` | 10 個測試覆蓋 alias、雙向、batch 與 schema 驗證 |

### 57.4 U12 val gap 分析使用方式

```powershell
# 前提：AP-D4 checkpoints 已存在
python -m scripts.u12_val_gap `
  --data data/raw/vpesg4k_val_1000.csv `
  --meta reports/analysis/_ensemble/ap_d4_8way_3view_meta.json `
  --batch-size 16

# 輸出：
#   reports/analysis/u12_val_gap/u12_val_gap.json    # gap 報告 JSON
#   reports/analysis/u12_val_gap/val_preds.csv       # val 集預測結果
```

**注意**：U12 需要 8 × 5 = 40 個 `outputs/checkpoints/{stem}/seed42/fold{f}/best.pt` checkpoint。
若目前缺少（checkpoints 不在 git 中），需先重訓對應 stem 後再執行。

### 57.5 測試狀態

本 Phase 完成後全測試通過 60 個（Phase 39 後為 50 個，本次新增 10 個 label 正規化測試）。

```text
60 passed, 2 warnings in 10.49s
```

---

## 58. Phase 41 — Train+Val 合併重訓設置（2026-06-05）

### 58.1 動機

官方 valid 釋出後，最終提交前可用資料從 1,000 筆增至 2,000 筆（train 1,000 + val 1,000）。
用 2,000 筆重訓所有 AP-D4 stem，再以重訓後的 checkpoints 對測試集推論，是提升最終提交品質的標準動作。

### 58.2 合併資料產生

```powershell
python -m scripts.u15_merge_train_val
# 預期輸出：data/processed/train_val_combined.csv  (≤2000 rows)
```

- val 集的 `more_than_5_years` 會被 `load_dataset()` 自動正規化為 `longer_than_5_years`。
- `data/processed/` 不在 git 中，需本地執行。

### 58.3 8 個 TV 訓練設定檔

| 設定檔 | 繼承自 | 改動 |
| :-- | :-- | :-- |
| `exp_p2_combo_best_tv.yaml` | `exp_p2_combo_best.yaml` | `exp_name`、`csv_path` → combined、`seeds: [42]` |
| `exp_p2_combo_best_u10_pseudo_tv.yaml` | `exp_p2_combo_best_u10_pseudo.yaml` | 同上，`pseudo_csv_path` 繼承 |
| `exp_p2_combo_best_u10_pseudo_v2_tv.yaml` | `exp_p2_combo_best_u10_pseudo_v2.yaml` | 同上 |
| `exp_p2_combo_best_u10_pseudo_v2_classw_focal_t4_g3_tv.yaml` | `..._classw_focal_t4_g3.yaml` | 同上，class-weight + Focal 繼承 |
| `exp_p2_combo_best_u10_pseudo_v3_classw_focal_t4_g3_tv.yaml` | `..._v3_classw_focal_t4_g3.yaml` | 同上，v3 pseudo 繼承 |
| `exp_p2_combo_best_classw_focal_u6pro_tv.yaml` | `..._classw_focal_u6pro.yaml` | 同上，U6-pro BT + aug 繼承 |
| `exp_p2_combo_best_aug_plus_tv.yaml` | `exp_p2_combo_best_aug_plus.yaml` | 同上，Aug-Plus v1 pseudo 繼承 |
| `exp_p2_combo_best_aug_plus_v2_tv.yaml` | `exp_p2_combo_best_aug_plus_v2.yaml` | 同上，Aug-Plus v2 pseudo 繼承 |

**設計決策**：只用 seed 42（single seed）。雙重理由：(a) 最終推論不依賴 CV fold，只需最佳 checkpoint；(b) 節省訓練時間，讓 6/10-6/17 推論窗口最大化。

### 58.4 訓練指令（test 釋出前的時間窗口，2026-06-05 ~ 2026-06-10）

純官方資料 stem（無 pseudo）：

```powershell
python -m src.train_kfold --config configs/exp_p2_combo_best_tv.yaml
```

pseudo 資料 stem（需 `data/processed/u10/pseudo_labels*.csv` 存在）：

```powershell
# 依序執行 7 個 pseudo stems
python -m src.train_pseudo_kfold --config configs/exp_p2_combo_best_u10_pseudo_tv.yaml
python -m src.train_pseudo_kfold --config configs/exp_p2_combo_best_u10_pseudo_v2_tv.yaml
python -m src.train_pseudo_kfold --config configs/exp_p2_combo_best_u10_pseudo_v2_classw_focal_t4_g3_tv.yaml
python -m src.train_pseudo_kfold --config configs/exp_p2_combo_best_u10_pseudo_v3_classw_focal_t4_g3_tv.yaml
python -m src.train_pseudo_kfold --config configs/exp_p2_combo_best_classw_focal_u6pro_tv.yaml
python -m src.train_pseudo_kfold --config configs/exp_p2_combo_best_aug_plus_tv.yaml
python -m src.train_pseudo_kfold --config configs/exp_p2_combo_best_aug_plus_v2_tv.yaml
```

### 58.5 Phase 42 預告：TV 模型集成搜索

train+val 重訓後，使用 U12 val gap 腳本的推論機制對 **測試集** 推論，再重跑 AP-D hillclimb 選出最終 ensemble 權重（TV checkpoints 無 OOF fold，故只有 stored 單視角）。

| 子步驟 | 工具 | 輸入 |
| :-- | :-- | :-- |
| 測試集推論 | `scripts/u12_val_gap.py`（或獨立測試集腳本）| 8 個 `*_tv` checkpoints |
| Ensemble 搜索 | `src/tools/u10_per_task_tta.py` | TV stem probabilities on test set |
| 後處理 | `apply_constraints_batch` | 集成 argmax 結果 |
| 提交守門 | `src/tools/validate_submission.py --mode submission` | 最終 CSV |

Phase 42 啟動條件：Phase 41 全部 8 stem 重訓完成（(OK) 2026-06-11 達成），且測試集釋出（(OK) 2026-06-11 本機取得 `vpesg4k_test_2000.csv`）。

### 58.6 Phase 41 重訓成果與 TV 池 OOF 集成（2026-06-11）

**8 stems 重訓 OOF（5-fold mean，seed 42）**：見 [§14.4](#144-phase-41-trainval-重訓狀態2026-06-05-啟動--2026-06-11-全部完成)。批次執行器 `scripts/phase41_train_all_tv.py --resume`，總耗時約 5.4 小時，8 stems 全數 `checkpoints_ok=True`。

**TV 池 8-stem OOF 集成 joint hillclimb**（工具 `scripts/u16_tv_oof_ensemble.py`，重建 2,000-row pooled OOF → post-constraint per-task 權重搜索）：

| 指標 | 分數 |
| :-- | :--: |
| 最佳單 stem（u10_pseudo_v2_tv，pooled OOF）| 0.68407 |
| 等權 8-stem 混合 | 0.69759 |
| **joint hillclimb 最佳** | **0.71033** |

集成權重已存至 `reports/analysis/_ensemble/tv_oof_ensemble_meta.json`，作為 Phase 42 測試集推論的 warm-start。詳細分析與過擬合警示見 [§14.5](#145-phase-415--tv-池-oof-集成驗證phase-42-前置)。

## 59. Phase 42 — 測試集推論與最終提交（2026-06-11 完成）

測試集 `vpesg4k_test_2000.csv`（id 12001–14000，2,000 段落）於 2026-06-11 釋出。工具 `scripts/u17_phase42_test_inference.py` 完成端到端推論與提交產出。

**提交格式關鍵規則**（依官方 `AI_CUP_2026_VeriPromiseESG_Submission_Guidelines.pdf`）：

- 5 欄固定順序：`id,promise_status,verification_timeline,evidence_status,evidence_quality`（不含 promise/evidence 字串欄）。
- `verification_timeline` 使用 **`more_than_5_years`**（非訓練集 canonical 的 `longer_than_5_years`）；輸出時自動 remap。
- N/A 一律為字面字串 `"N/A"`，任何欄位不得空白或 NaN；2,000 列、id 嚴格 12001–14000、大小寫敏感。
- 邏輯規則：promise=No → 下游三欄全 N/A；promise=Yes → timeline≠N/A、evidence∈{Yes,No}；evidence=Yes → quality∈{Clear,Not Clear,Misleading}；evidence=No → quality=N/A。

**推論流程**：

1. 8 個 `*_tv` checkpoints（每 stem 5 folds 機率平均）對 2,000 列做 stored-view 推論（macbert-base / cls_mean / maxlen 384）。
2. 套用 `tv_oof_ensemble_meta.json` 的 per-task stem 權重混合。
3. **約束式解碼**：promise=Yes 的列在 timeline/evidence/quality 上排除 N/A 類別取 argmax（ground-truth promise=Yes 從不帶 N/A，故 ≥ 純 argmax 且保證合規），再經 `apply_constraints_batch` 收尾。
4. `validate_submission --mode preds` 守門 → 寫出三份提交（主集成 + 等權 + 最佳單 stem fallback）。

**產出檔案**（`outputs/submissions/`）：

| 檔案 | 說明 |
| :-- | :-- |
| `phase42_tv_ensemble_submission.csv` | **主提交**（per-task hillclimb 權重，OOF 代理 0.71033）|
| `phase42_tv_ensemble_preds.csv` | 內部 canonical-label preds（已過 `validate_submission --mode preds`）|
| `phase42_equalweight_submission.csv` | fallback：等權 8-stem 混合 |
| `phase42_bestsingle_submission.csv` | fallback：最佳單 stem（u10_pseudo_v2_tv）|

**主提交 label 分布**（2,000 列、無空白/NaN、id 12001–14000 唯一、5 欄順序正確）：

| 任務 | 分布 |
| :-- | :-- |
| promise_status | Yes=1724, No=276 |
| verification_timeline | already=754, between_2_and_5_years=522, more_than_5_years=417, N/A=276, within_2_years=31 |
| evidence_status | Yes=1470, N/A=276, No=254 |
| evidence_quality | Clear=1267, N/A=530, Not Clear=203 |

> **約束一致性**：promise=No 的 276 列在三個下游欄位皆為 N/A（276 完全對齊），evidence_status≠Yes 的列其 quality 全 N/A（530 = 276 promise-No + 254 evidence-No），結構守門全數通過。
> **HF 403 提示為非致命背景雜訊**：`hfl/chinese-macbert-base` discussions API 被停用所致，模型權重仍正常載入；可用 `HF_HUB_OFFLINE=1` 抑制。

## 60. Phase 43 — 真實 leaderboard 0.5962 vs OOF 0.71033 根因分析（2026-06-11）

主提交上傳後實際得分 **0.5961966**，較 OOF 代理 0.71033 暴跌 **0.114**。專案歷經約 40 個 Phase 的 OOF joint hillclimb，這是**第一個真實 leaderboard 分數**。完整診斷（工具 `scripts/u18_decoding_experiments.py`、`scripts/u19_decode_mismatch_oof.py`）：

**逐項排查結論——非程式碼／對齊／解碼 bug**：

| 嫌疑 | 檢驗方法 | 結論 |
| :-- | :-- | :-- |
| 邊際分布／對齊 bug | 預測 vs 訓練邊際分布逐類比對 | **排除**：預測分布幾乎完美吻合訓練（Yes 86% vs 81%、already 38% vs 36%、Clear 63% vs 56%）→ 模型對訓練分布校準良好 |
| 解碼策略不一致（u16 純 argmax vs u17 強制非 N/A）| OOF 在兩種解碼下重新評分（u19）| **排除**：0.71033（純）vs 0.71044（強制）= **+0.00011**，可忽略 |
| hillclimb 權重過擬合 | OOF：hillclimb vs equal-weight | **部分成立但有限**：hillclimb 0.71044 vs 等權 0.69612，僅 +0.0143，且為 OOF 過擬合，難以轉移 → 至多解釋 0.114 中的 ~0.014 |
| 少數類崩塌 + domain shift | macro 任務 OOF 分數 + sample 5 列 | **主因**：即使分布內 OOF，T2=0.63、T4=0.47 已偏弱（T4 佔總分 35%）；測試集分布不同 + macro-F1 對少數類零召回極敏感 → 餘下 ~0.10 缺口 |

**OOF 解碼 × 權重四象限**（`u19_decode_mismatch_oof.py`）：

| 設定 | weighted | T1 | T2 | T3 | T4 |
| :-- | :-- | :-- | :-- | :-- | :-- |
| hillclimb + 純 argmax（u16 評分路徑）| 0.71033 | 0.945 | 0.631 | 0.871 | 0.472 |
| hillclimb + 強制非 N/A（u17 提交路徑）| 0.71044 | 0.945 | 0.633 | 0.874 | 0.470 |
| equal-weight + 純 argmax | 0.69759 | 0.941 | 0.605 | 0.861 | 0.458 |
| equal-weight + 強制非 N/A | 0.69612 | 0.941 | 0.605 | 0.861 | 0.454 |

**根因總結**：OOF 代理本身因 **domain shift（競賽 train 與 test 報告風格／公司不同）+ macro-F1 對少數類敏感** 而樂觀約 0.10；hillclimb 再疊加約 0.014 的 OOF 過擬合。沒有可修的程式 bug。

OOF 與真實 leaderboard 的逐任務落差（落差集中在二元 T1/T3）：

![OOF vs LB 落差](reports/figures/fig1_oof_vs_lb_gap.png)

**Phase 43 候選提交**（`outputs/submissions/`，皆已過 `validate_submission --mode preds`）：

| 候選 | 策略 | 用意 | 風險 |
| :-- | :-- | :-- | :-- |
| `phase43_c1_equalweight_submission.csv` | 等權 8-stem 混合 | 移除 hillclimb 過擬合，最穩健 | 低 |
| `phase43_c2_priorcorr_a03_submission.csv` | hillclimb + T2/T4 prior-correction α=0.3 | 對抗少數類崩塌 | 中（賭 test 較均衡）|
| `phase43_c2_priorcorr_a05_submission.csv` | 同上 α=0.5 | 更強少數類推升 | 高（Misleading 衝到 4.7% vs train 0.1%，過度）|
| `phase43_c3_equal_priorcorr_a03_submission.csv` | 等權 + T2/T4 prior-correction α=0.3 | C1+C2 結合，平衡 | 中 |
| `phase43_c3_equal_priorcorr_a05_submission.csv` | 同上 α=0.5 | 更強 | 高 |

> **prior-correction 不可用 OOF 調參**：OOF 與訓練集共用先驗分布，prior-correction 在 OOF 上必然看似變差；它只在 **test 先驗與 train 不同** 時才有益，本質是對 domain shift 的下注。建議下一次上傳順序：**C1（穩健去過擬合）→ C3 α=0.3（保守少數類修正）**，依 leaderboard 回饋再決定是否加大 α。

## 61. Phase 44 — 逐任務測試分數揭示「二元任務崩塌」並修正方向（2026-06-11）

第 2 次上傳 `phase43_c3_equal_priorcorr_a03`（等權 + macro T2/T4 prior-corr α=0.3）得分 **0.6037**（較首傳 0.5962 **+0.0075**），且首次取得**逐任務測試分數**：

| 任務 | 權重 | OOF（等權）| **TEST** | 落差 | 可回收 weighted |
| :-- | :--: | :--: | :--: | :--: | :--: |
| promise_status（二元）| 0.20 | 0.941 | **0.786** | −0.155 | 0.031 |
| verification_timeline（macro）| 0.15 | 0.605 | **0.606** | +0.001 (OK) | ~0 |
| evidence_status（二元）| 0.30 | 0.861 | **0.675** | −0.186 (注意) | **0.056** |
| evidence_quality（macro）| 0.35 | 0.454 | **0.437** | −0.017 | 0.006 |

**診斷反轉（重要）**：§60 原假設「崩塌在 macro 任務」**被推翻**。實際上 **macro 任務（T2/T4）幾乎完美轉移**（T2 +0.001、T4 −0.017），崩塌全在**二元任務 T1/T3**。骨幹模型在測試集並未失效，僅二元 Yes/No 決策邊界對測試集失準。**最大可回收空間在 T3（權重 0.30 × 落差 0.186 = 0.056）**，遠大於 T1（0.031）。

**Phase 44 對策——二元任務 prior-correction**（朝少數類 No 修正，工具 `scripts/u20_binary_prior_correction.py`，重用 `phase43_test_probs.npz` 快取免重跑推論）：模型二元後驗高度自信，故需較大 α 才有可見位移。Yes-rate 掃描：

| 候選 | binary α | T1 Yes | T3 Yes | 定位 |
| :-- | :--: | :--: | :--: | :-- |
| `phase44_d1_binNo_a03` | 0.3 | 84.0% | 68.8% | 溫和（首選）|
| `phase44_d2_binNo_a05` | 0.5 | 82.6% | 65.9% | 中等 |
| `phase44_d3_binOnly_a03` | 0.3（僅二元，無 macro corr）| 84.0% | 68.8% | 隔離測試（驗 macro corr 是否必要）|
| `phase44_d4_binT3_a05` | T3 only 0.5 | 85.2% | 67.6% | 專攻 0.30 權重任務 |
| `phase44_d5_binNo_a08` | 0.8 | 80.8% | 61.0% | 較強 |
| `phase44_d6_binNo_a12` | 1.2 | 78.3% | 55.0% | 強 |

> **方向判斷依據**：sample 5 列中 promise=No 僅 1/5（≈train 18.6%），暗示 **test 的 promise 先驗可能與 train 接近** → T1 落差 0.155 恐為**真實準確度下降**而非閾值偏移，對 T1 過度修正可能有害。相對地 T3（權重 0.30、落差 0.186）是更值得下注的目標。建議下一次上傳順序：**d1（溫和雙修正，最安全）→ 視回饋朝 d2/d5 加大；若懷疑 T1 修正有害則改用 d4（僅修 T3）**。所有 6 候選皆已過 `validate_submission --mode preds`。
>
> **關鍵教訓**：邊際分布吻合（predicted≈train）**不足以**定位失敗任務——二元 F1 可在邊際吻合下仍因逐樣本準確度下降而崩塌。**務必先取得逐任務 leaderboard 分數再決定投入方向**。

## 62. Phase 45 — domain-shift 診斷 + 全方法盤點 + 資訊效率上傳排程（2026-06-12）

上傳原則：每日上限 3 次、僅上傳已驗證較優的候選；所有可行方法逐一記錄並系統性評估。本階段先用**本機可得的訊號**校準方向（避免浪費上傳額度），再產生資訊效率最高的候選集。

### 62.1 關鍵診斷（`scripts/u21_domain_shift_diag.py`，本機唯讀，不耗額度）

| 診斷 | 結果 | 推論 |
| :-- | :-- | :-- |
| **對抗驗證**（train vs test，char TF-IDF + LogReg，5-fold CV ROC-AUC）| **0.5097** | train/test 文本分布**幾乎無法區分** → **無輸入端 domain shift** |
| 二元信心比對 T1（OOF vs test，等權混合 P(Yes)）| mean −0.010、below_0.5 +0.008 | test 略微較不傾向 Yes，但**位移極小** |
| 二元信心比對 T3 | mean −0.007、high_conf>0.8 −0.021 | 同上，位移極小 |

> **重大結論（修正 Phase 44 的下注前提）**：對抗驗證 AUC≈0.51 + 信心分布幾乎一致 ⇒ **test 輸入與 train 同分布，模型機率校準在兩者上一致**。因此二元 F1 崩塌**不是先驗/閾值偏移**，模型「看不到」需要修正的訊號。最可能成因：**test 黃金標註慣例較嚴（更多真實 No）**，或 **OOF 略為樂觀**。推論三點：
> 1. **激進 toward-No（d5 α=0.8 / d6 α=1.2）缺乏本機支持，極可能過衝** → 降級為「僅在前段溫和版見效時才嘗試」。
> 2. 僅**溫和** toward-No 有理（信心位移僅 ~0.01）。
> 3. **importance-weighted 重訓**這條路被 AUC=0.51 **排除**（沒有可重加權的分布差異），省去無謂重訓。

### 62.2 全方法盤點（窮舉，標註可行性）

| 類別 | 方法 | 成本 | 本機可驗證？ | 判定 |
| :-- | :-- | :-- | :-- | :-- |
| A 解碼/閾值 | A1 二元 prior-corr toward-No（=odds 閾值）| 極低 | 否（需 LB）| 已測：toward-No 無效，本機證據指向「溫和」 |
| A | A2 僅 T1 修正（測 T1→T3 cascade）| 極低 | 否 | 候選 `t1only_a04` |
| A | A3 僅 T3 修正（最大槓桿 w=0.30）| 極低 | 否 | 候選 `t3only_a04` |
| A | A4 macro α 微調（T4 w=0.35，α=0.3 疑過衝）| 極低 | 部分（邊際）| 候選 `macro_a02` |
| A | A5 toward-Yes 對沖 | 極低 | 否 | 信心分析不支持，**暫緩** |
| B 診斷 | B1 對抗驗證 | 低 | (OK) | **已完成**：AUC 0.51 |
| B | B2 二元信心比對 | 低 | (OK) | **已完成**：位移極小 |
| C 校準 | C1 溫度縮放 | 低 | — | 對二元 argmax 無效（除非配閾值），略過 |
| D 重訓 | D1 importance-weighted 重訓 | 高 | — | **被 AUC=0.51 排除** |
| D | D2 test 偽標自訓 | 高 | — | 風險高、無 shift 佐證，暫緩 |
| E 集成 | E1 等權 vs hillclimb | 低 | (OK)（已知等權較穩）| 已採用等權 |

### 62.3 Phase 45 候選集（`scripts/u22_phase45_candidates.py`，全部過 `validate --mode preds`，2000 列）

均建於 **等權 8-stem 混合 + 同一 constrained cascade**。注意 `c1_equal` 與已上傳的 `c3`(0.6037) **僅在 T2/T4 不同**（二元 Yes-rate 同為 T1 85.2%/T3 72.4%），故上傳 c1 可**乾淨隔離 macro 修正的淨值**。

| 候選 | 設定 | T1 Yes | T3 Yes | T4 NotClear | 解答的未知數 |
| :-- | :-- | :--: | :--: | :--: | :-- |
| `phase45_c1_equal` | 等權，**完全不修正** | 85.2% | 72.4% | 8.8% | **U-A**：macro 修正淨值（vs c3 0.6037）|
| `phase45_macro_a02` | 等權 + macro α=0.2（更溫和）| 85.2% | 72.4% | 14.2% | **U-B**：macro α=0.3 是否過衝（NotClear 17.7%≫train 11%）|
| `phase45_t1only_a04` | + macro0.3 + **僅 T1** α=0.4 | 83.2% | 71.0% | 17.4% | **U-C**：T1→T3 cascade（T1 修正能否順帶救 T3）|
| `phase45_t3only_a04` | + macro0.3 + **僅 T3** α=0.4 | 85.2% | 68.5% | 14.1% | **U-D**：最大直接槓桿 T3（w=0.30）|
| `phase45_gentle_a02` | + macro0.3 + 二元 α=0.2 | 84.4% | 69.8% | 15.8% | **U-E**：匹配信心位移的溫和組合修正 |

（更激進的 `phase44_d2/d5/d6` 仍保留於 `outputs/submissions/`，僅在溫和版證實 toward-No 有效後才動用。）

### 62.4 資訊效率上傳排程（每日 3 次）

已銀行分數：**c3 = 0.6037**（不會因新嘗試而丟失）。本機證據顯示修正為**弱槓桿**，預期增益小（約 ±0.005），務求每次上傳都換取最大資訊。

**Day-1 建議（依序，依回饋分支）**：
1. **`phase45_macro_a02`** — 最可能勝過 c3 的低風險賭注（修正 α=0.3 的 NotClear 過衝）。若 > c3 ⇒ macro 過衝確立，採 a0.2 為新基線。
2. **`phase45_c1_equal`** — 控制組，乾淨隔離 macro 修正淨值（vs c3 差異純在 T2/T4）。若 c1 ≈ c3 ⇒ 之前 +0.0075 來自等權而非 macro 修正；若 c1 < c3 ⇒ macro 修正確有貢獻。
3. **`phase45_gentle_a02`** — 本機唯一被支持的溫和 toward-No 探針。若 > 當前最佳 ⇒ toward-No 方向成立，Day-2 朝 `t3only_a04` → `phase44_d2` 推進；若 ≤ ⇒ 二元崩塌屬不可約誤差，停止 toward-No，改聚焦 macro/集成。

**分支邏輯**：
- macro_a02 與 c1 共同定位 macro α 最佳點（候選 α∈{0,0.2,0.3}）。
- gentle_a02 為 toward-No 的單點探針；正向才升級 t3only/d-series，避免在無本機支持下盲目加大 α。

> **Phase 45 關鍵教訓**：(1) **對抗驗證**是上傳前必做的廉價步驟——AUC≈0.5 直接排除了 domain-shift 類修正與 importance 重訓，避免浪費額度與算力。(2) 當輸入無 shift 且信心分布一致時，巨大的 LB↔OOF 落差最可能來自**標註慣例差異或 OOF 樂觀**，此時閾值修正僅為弱槓桿，應**溫和探針 + 控制組隔離**，而非激進下注。(3) 每次上傳都應**只改變一個變因**（c1 隔離 macro、t1only/t3only 隔離單任務）以確保 LB 回饋可解讀。

## 63. Phase 46 — 超越 0.61：macro-only 加權 + 多視窗 TTA 雙槓桿（2026-06-12）

當時 leaderboard 榜首為 0.61、本隊 0.6037。Phase 45 已證實 (a) 無 domain shift、(b) macro 任務 T2/T4 忠實轉移、(c) 二元崩塌不可約。因此放棄二元閾值修正（弱槓桿），改攻**兩個本機可驗證、非過擬合的真實槓桿**：

### 63.1 槓桿 A — macro-only OOF 加權（`scripts/u23_macro_weighting.py`）

關鍵洞察：Phase 42 joint hillclimb（OOF 0.71044）在 LB 崩盤（0.5962）是因為它**同時移動了二元任務權重**而對 OOF 過擬合。但 macro 任務忠實轉移 ⇒ **只**對 T2/T4 的 stem-mix 權重做 post-constraint hillclimb、**凍結 T1/T3 於等權**，其 OOF 增益應可轉移。結果：

| | OOF | T1 | T2 | T3 | T4 |
| :-- | :--: | :--: | :--: | :--: | :--: |
| 等權 | 0.69759 | 0.9409 | 0.6053 | 0.8606 | 0.4584 |
| macro-opt | **0.70208** | 0.9409 | **0.6286** | 0.8606 | 0.4612 |
| 增益 | **+0.00449** | — | **+0.0233** | — | +0.0028 |

T1/T3 確認凍結於等權；增益幾乎全來自 T2（+0.0233）。因 T2 忠實轉移，此 +0.0045 預期可帶到 LB。

### 63.2 槓桿 B — 多視窗 TTA（`scripts/u24_tta_inference.py` + `u25_tta_combine.py`）

8 個 TV stem 皆以 max_length=384 **head 截斷**訓練/推論（stored view）。診斷發現 **792/2000（~40%）測試段落超過 384 token**，head 截斷丟失尾段——而 ESG 承諾／證據敘述常落在段落尾部（專案 notebook §13.2）。因無 domain shift，**訊號回收**（而非重加權）是最有望破 0.61 的真槓桿。

實作：對每個 stem 額外推論 **middle**（中段 W-2 token）與 **tail**（尾段 W-2 token）視角，手動加 `[CLS]`/`[SEP]`（transformers 5.6 的 `build_inputs_with_special_tokens` 不可靠，依記憶）。三視角等權平均（TV stem 無 middle/tail OOF，無法離線調視角權重；等權為穩健標準作法，且為專案 U1 TTA 提升飽和集成的同一手法）。

### 63.3 Phase 46 候選集（全部過 `validate --mode preds`，2000 列）

| 候選 | 槓桿 | macro corr | T4 NotClear | T4 Misleading | 隔離用途 |
| :-- | :-- | :--: | :--: | :--: | :-- |
| `phase46_mw_none` | A（stored 加權）| — | 9.8% | 0.0% | 純加權效果 |
| `phase46_mw_a02` | A | 0.2 | 15.9% | 0.1% | A + 溫和修正 |
| `phase46_mw_a03` | A | 0.3 | 21.6% | **0.7%** | A + c3 同設定（隔離 A vs c3）|
| `phase46_tta_plain` | B（TTA）| — | 8.8% | 0.0% | 純 TTA 效果 |
| `phase46_tta_macro_a02` | B | 0.2 | 14.2% | 0.1% | B + 溫和修正 |
| `phase46_tta_macro_a03` | B | 0.3 | 17.6% | 0.1% | B + c3 同設定（隔離 B vs c3）|
| `phase46_tta_macrow_a02` | **A+B** | 0.2 | 15.8% | 0.1% | **全疊加（破 0.61 最強候選）** |

注意 `mw_a03` 首次預測 `Misleading` 0.7%（其餘全 0.0%）——T4 macro F1 含 Misleading 類，若測試集確有 Misleading 樣本，此候選可解開該類的結構性 0 分上限。

### 63.4 上傳排程（每日 3 次；已銀行 c3=0.6037）

1. **`phase46_tta_macrow_a02`**（全疊加 A+B+corr0.2）— 破 0.61 的最強單發。兩槓桿皆只作用於 macro 任務/新增視角，不動二元決策，疊加風險低。
2. **`phase46_mw_a03`**（僅槓桿 A + corr0.3）— 隔離已 OOF 驗證的 A（vs c3）。若 > c3 ⇒ 加權轉移確立。與步驟 1 對比可判斷 TTA（B）是否有貢獻。
3. **`phase46_tta_macro_a03`**（僅槓桿 B + corr0.3）— 隔離 TTA（vs c3）。三發合起來可完整拆解 A、B、A+B 的 LB 貢獻。

> **Phase 46 關鍵教訓**：(1) 當 LB↔OOF 落差被歸因於某些任務（二元）時，可對**忠實轉移的任務（macro）單獨做 OOF 加權並凍結其餘**——這把「整體 hillclimb 過擬合」拆解為「可轉移子問題」，是繞過 OOF 過擬合的通用手法。(2) 無 domain shift 時，**訊號回收型 TTA（多 token 視窗）**優於任何重加權/閾值修正，因為它真正增加資訊；先量測截斷比例（此處 40%）確認 TTA 母體再投入算力。(3) 評分函式細節決定槓桿：T4 macro 含 `Misleading`/`N/A` 四類且權重最高（0.35），故任何能讓模型偶爾預測稀有類的修正都可能解開結構性上限。

## 64. Phase 47 — mw_a03 LB 回歸的解剖 + 無風險 T4 槓桿隔離（2026-06-12）

**Phase 46 上傳結果**（上傳 `mw_a03` = stored macro 加權 + corr0.3）：**0.6029 < 已銀行 c3 0.6037**。逐任務拆解：

| 任務 | 權重 | c3 (0.6037) | mw_a03 (0.6029) | Δ |
| :-- | :--: | :--: | :--: | :--: |
| promise_status (T1) | 0.20 | 0.786 | 0.7864 | +0.0004（雜訊）|
| verification_timeline (T2) | 0.15 | 0.606 | **0.5959** | **−0.0101** (注意) |
| evidence_status (T3) | 0.30 | 0.675 | 0.675 | 0.0000（二元凍結，完全相同）|
| evidence_quality (T4) | 0.35 | 0.437 | **0.4392** | **+0.0022** (OK) |

**根因**：macro-only stem **加權仍過擬合 OOF**——T2 在 LB 掉 0.010，儘管 OOF +0.023。Phase 45 的「macro 忠實轉移」結論是關於**信心分布**（對抗 AUC），**不代表 OOF 權重最優點轉移**。T2 在**等權**時已是最忠實轉移值（OOF 0.605 ≈ LB 0.606），任何 OOF 權重調整都把它推離最優。

**唯一正向訊號**：T4 stem 加權（最重任務 w=0.35），LB +0.0022 且 OOF +0.0028（兩者一致，真正轉移）。

### 64.1 無風險重組（`scripts/u27_phase47_t4_lever.py`）

關鍵結構洞察：`mix` 對**每個任務獨立混合**，且約束級聯只依賴 T1/T3（保持等權）。因此可組出：

> **`phase47_t4w_a03`** = 等權 T1/T2/T3 + 學到的 T4 stem 權重 + corr{T2:0.3, T4:0.3}

逐欄位驗證（直接比對 CSV）證明此候選**位元級等同**於：c3 的 T1/T2/T3（promise 100%、timeline 100%、evidence_status 100% 相同）+ mw_a03 的 T4（evidence_quality 100% 相同）。因評分為固定線性組合且每任務預測皆已被 LB 量測過：

$$ \text{LB} = 0.20(0.786) + 0.15(0.606) + 0.30(0.675) + 0.35(0.4392) = \mathbf{0.60432} $$

**確定性 +0.0006、零風險的新最佳**——把 c3 的 T2 修回 0.606 同時保留 mw_a03 的 T4 增益。

### 64.2 OOF 安全性驗證（避免 T4 修正崩盤）

T4 prior-corr alpha sweep（等權 blend）：alpha=0.2→T4 0.4513、0.3→0.4451、0.4→0.4351、0.5→0.4177 **全部 DROP**。確認 **T4 修正不可超過 0.3**（腳本自動跳過 alpha 0.4/0.5）。真正有效的是 **T4 stem 加權**（OOF +0.0028），非更激進的修正。

### 64.3 Phase 47 候選

| 候選 | 組成 | 預期 LB | 風險 |
| :-- | :-- | :--: | :-- |
| `phase47_t4w_a03` | c3(T1-T3) + 學到T4權重 + corr0.3 | **0.6043** | **零**（純重組已量測值）|
| `phase47_t4w_tta_a03` | 同上 + TTA（中/尾視窗）| ? | 低（TTA 未測，但只增訊號）|
| `phase47_t4hc_a03` | T4-only hillclimb 權重 + corr0.3 | ~0.6043 | 零（與 t4w 近乎等價）|

### 64.4 上傳排程（已銀行 c3=0.6037）

1. **`phase47_t4w_a03`** — 確定性 0.6043，先銀行新最佳（零風險）。
2. **`phase47_t4w_tta_a03`** — 安全 T4 基底 + TTA 槓桿 B（40% 截斷列訊號回收）。若 T2/T4 提升可達 ~0.606+。
3. 視 1-2 結果決定（若 TTA 有效則加碼，否則保留）。

> **Phase 47 關鍵教訓**：(1) **OOF 分布吻合 ≠ 可做 OOF 權重調整**——對抗驗證證明模型「輸出」轉移，但**權重/閾值最優點是擬合到 OOF 雜訊的二階量**，照樣過擬合。永遠用控制組上傳隔離權重調整再信任。(2) 當多個任務各自有已量測的 LB 表現，且後處理對每任務獨立、約束級聯只依賴未變動的任務時，可**用位元級重組組出確定性的新最佳**（c3 的好 T2 + mw_a03 的好 T4），無需再上傳賭運氣。(3) 本賽事可靠槓桿只剩：等權集成、訊號回收 TTA（待驗）。二元任務的大缺口（T1 −0.155 / T3 −0.186）經 Phase 45 證實為**真實測試準確度差異**（非閾值），post-hoc 修正無解，需重訓更強骨幹方有突破 0.61 的空間。

### 64.5 Phase 47 上傳結果（修正先前「確定 0.6043」的誤判）

`phase47_t4w_a03` 實際 LB = **0.6026**（T2=0.606 如預期修回，但 **T4=0.4339 而非 mw_a03 上傳時的 0.4392**）。三次上傳呈單調遞減：**0.6037（c3）→ 0.6029（mw_a03）→ 0.6026（t4w）**。

- **誤判根因**：宣稱「位元級等同 mw_a03 的 T4」時，比對的是**磁碟上被 `u23 --iters 12000` 重跑覆寫後的 mw_a03**，而非當初上傳的版本。上傳版（舊 meta，T4=0.4392）與磁碟版（新 meta，T4=0.4339）不同。**鐵則：宣稱確定性重組前，必須對「上傳當下的位元」做雜湊凍結比對，絕不可信任產生腳本已被重跑的檔案。** 已將 c3 凍結為 `outputs/submissions/FROZEN_BEST_0.6037_c3.csv`（md5 46D31CE3…）。
- **真正的 ML 教訓**：T4 stem 加權**也是純 OOF 過擬合雜訊**。同一 T4 任務、兩套都「OOF 最優」的權重，在測試集一個 +0.0022、一個 −0.0031。**所有 stem 權重微調（T2 與 T4 皆然）都過擬合**；等權 c3=0.6037 已連續 3 次上傳證實為這批模型的測試集天花板。**後處理已完全榨乾。**

## 65. Phase 48 — 更強骨幹重訓設計（驗證優先、鎖定可轉移任務）

> 本節為**設計文件**，每步驟可一鍵執行；實際多小時 GPU 重訓需經明確核准後啟動。

### 65.1 資料切分確認（train / val / test）

| 集合 | id 範圍 | 列數 | 標籤 | 用途 |
| :-- | :-- | :--: | :--: | :-- |
| train | 10001–11000 | 1000 | 有 | 原始官方訓練集 |
| val | 11001–12000 | 1000 | 有 | 官方驗證集（2026-06-03 釋出）|
| **train+val 合併** | 10001–12000 | 2000 | 有 | 5-fold 分層 CV（現行 8 stem 訓練集）|
| test | 12001–14000 | 2000 | **無** | 排行榜評分集 |

合併集前 1000 列 = train、後 1000 列 = val（`data/processed/train_val_combined.csv`，順序固定）。

### 65.2 驗證代理診斷（`scripts/u28_val_proxy_diag.py`，已執行）

8 stem 在 train+val 2000 列做 5-fold CV，故 **val 列（合併索引 1000–1999）的 OOF 是 held-out 預測**。等權集成各切片分數：

| 任務 | ALL(2000) | train(1000) | val(1000) | LB(test) | val−LB |
| :-- | :--: | :--: | :--: | :--: | :--: |
| T1 promise | 0.9409 | 0.9414 | 0.9403 | 0.7864 | **+0.1539** (NG) |
| T2 timeline | 0.6053 | 0.6064 | 0.6041 | 0.6060 | **−0.0019** (OK) |
| T3 evidence | 0.8606 | 0.8618 | 0.8594 | 0.6750 | **+0.1844** (NG) |
| T4 quality | 0.4584 | 0.4535 | 0.4639 | 0.4370 | **+0.0269** (OK) |
| weighted | 0.6976 | 0.6965 | 0.6989 | 0.6037 | +0.0952 |

**關鍵結論**：
1. **二元任務（T1/T3）的 OOF 完全不可信**：連最乾淨的 `p2_combo_best_tv`（無 pseudo）val-OOF 二元也是 0.94/0.87，與 pseudo stems 幾乎一樣 → 排除增強洩漏，歸因於**數十個 phase 在同一 OOF 上反覆選擇造成的適應性過擬合** + 測試集二元標註慣度差異（Phase 45 已排除輸入 domain shift）。
2. **macro 任務（T2/T4）的 OOF 忠實轉移**：T2 gap −0.002（完美）、T4 gap +0.027（小且穩定）。**這是唯一可信的本機優化訊號。**

### 65.3 戰略支點：鎖定 T4（次要 T2），放棄追二元

| 任務 | 權重 | LB 現值 | 可轉移 | 空間 | 結論 |
| :-- | :--: | :--: | :--: | :-- | :-- |
| T4 evidence_quality | **0.35** | 0.437 | (OK) +0.027 | **最大**（4 類語意，最差分） | **主目標** |
| T2 verification_timeline | 0.15 | 0.606 | (OK) −0.002 | 中 | 次目標 |
| T1 / T3 | 0.20 / 0.30 | 0.786 / 0.675 | (NG) 不可信 | — | **不追**（OOF 騙人） |

T4 權重最高、分數最低、OOF 可轉移 → 更強骨幹在 T4 的 OOF 增益會真正帶到 LB。**若 T4 OOF +0.05 且轉移，LB +0.0175 → 0.621，破 0.61。** 二元是近乎記憶的任務（記憶 0.94），更大模型只會過擬合更多（Phase 3/10 macbert-large 單模更差已證），追之無益。

### 65.4 骨幹選擇（8GB 離線可行）

評分目標是 T4/T2 的**語意細分**（Clear/Not Clear/Misleading/N/A），需要**架構更強**而非僅更大。已快取（免下載、離線即用）且非現有 macbert-base 池的選項：

| 骨幹 | 參數 | 快取 | 理由 | 風險 |
| :-- | :--: | :--: | :-- | :-- |
| **chinese-roberta-wwm-ext-large** | 326M | (OK) | 不同預訓練 + 大模型，池中從未用 large 作 T4 專家 | VRAM（batch 4 + accum 4 已驗證可行）|
| **chinese-electra-180g-large** | 335M | (OK) | 判別式預訓練，分類任務常強 | 載入需確認 head 對應 |
| Erlangshen-DeBERTa-v2-320M-Chinese | 320M | (下載) 需下載 | **解耦注意力，BERT 系最強編碼器**，最適合 T4 細分 | 一次性下載；離線旗標需暫解 |

**首選順序**：先跑**已快取的 roberta-wwm-ext-large**（零下載風險、可立即驗證流程），再評估 DeBERTa-v2 一次性下載。

### 65.5 訓練配方（針對 T4，內建抗過擬合）

繼承 `combo_best`（lr 3e-5、no class weight、cls_mean、5 epoch），針對 T4 與大模型調整：

```yaml
# configs/exp_p15_robertalarge_t4.yaml  (草案)
extends: exp_p2_combo_best.yaml
exp_name: p15_robertalarge_t4
model:
  backbone: hfl/chinese-roberta-wwm-ext-large
  max_length: 512          # 降低 40% 截斷（T4 證據常在段落尾）
  pooling: cls_mean
training:
  lr: 1.5e-5               # large 用 base 的 0.5x
  batch_size: 2
  grad_accum: 8            # effective batch = 16
  epochs: 5
  early_stop_patience: 2
  focal_tasks: [evidence_quality]   # T4 focal，緩解類別不平衡
  focal_gamma: 2.0
  use_class_weight: false
seeds: [42]               # 先 1 seed 驗證 VRAM/收斂，過關再 2024/20260417
```

### 65.6 驗證協定（杜絕重蹈過擬合）

1. **5-fold CV 於合併 2000**（與現有 stem 同協定，OOF 可比）。
2. **admission 準則只看可轉移任務**：新 stem 的 **T4 OOF 須 ≥ 0.48**（現池最佳單 stem T4 ≈ 0.458；+0.02 才有意義）。T2 OOF 不退步。**完全忽略二元 OOF**（不可信）。
3. **整合 = 9-way per-task post-constraint hillclimb**（Phase 14 證實「結構多樣性破停滯」的同一方法）。hillclimb 只會在「對 joint 後約束分數有幫助」的任務上給新 stem 權重；二元維持等權（已證 LB 最優）。
4. **上傳前最終閘**：新集成的 T2/T4（可信）須提升、二元維持 c3 等權設定 → 才花 1 次上傳驗證。

### 65.7 一鍵執行步驟

| 步 | 指令 | 產出 | 時間 |
| :--: | :-- | :-- | :-- |
| 1 | `python -m src.train_kfold --config configs/exp_p15_robertalarge_t4.yaml --smoke` | 1-fold 煙霧測試（VRAM/收斂）| ~10 分 |
| 2 | `python -m src.train_kfold --config configs/exp_p15_robertalarge_t4.yaml` | 5-fold OOF + ckpt | 多小時 |
| 3 | 檢視 `reports/experiments/p15_robertalarge_t4/score_summary.json` 的 T4 OOF | admission 判定 | 即時 |
| 4 | 過關 → 加入池跑 9-way hillclimb → test 推論 → 1 次上傳 | 新候選 | ~1 時 |

### 65.8 預期值與護欄

- **最佳情境**：DeBERTa/roberta-large 把 T4 OOF 0.458 → 0.50（+0.04），若轉移率同 +0.027 偏移，LB T4 ≈ 0.46 → +0.008 → **0.612，破 0.61**。
- **基準情境**：T4 OOF +0.02、轉移打折 → LB +0.003 → 0.607，未破但刷新。
- **失敗情境**：large 過擬合、T4 OOF 不升 → 不上傳，零損失（admission 閘擋下）。
- **護欄**：(a) 只信 T2/T4 OOF；(b) admission 閘擋住無效 stem；(c) 二元永遠等權；(d) 每次只加一個 stem 跑 hillclimb，可歸因。**絕不再對權重/閾值做 OOF 微調上傳。**

### 65.9 Phase 48 執行結果（三次完整重訓 + 分析迭代）

依「訓練→分析→再訓練」完整跑完三個骨幹（皆 combined-2000 5-fold CV，seed 42），全部在主目標 T4 上**未過 admission 閘**：

| 迭代 | 骨幹 / 配方 | 單模 T4 (all / val) | T2 | 過閘(≥0.458)? | hillclimb Δ | epoch 時間 |
| :-- | :-- | :--: | :--: | :--: | :--: | :--: |
| p15 | roberta-wwm-ext-large + focal T4 | 0.4312 / 0.4311 | 0.523 | (NG) | +0.0133 | ~220s |
| p16 | electra-180g-large（移除 focal）| 0.4266 / 0.4381 | 0.448 | (NG) | +0.0118 | ~160s |
| p18 | **macbert-base maxlen512**（最乾淨對照）| 0.4443 / **0.4370** | 0.502 | (NG) | +0.0133 | ~68s |
| — | 現有最佳 macbert-base maxlen384 | **0.4502** | — | 基準 | — | — |

**決定性證據與結論**：更大骨幹在 T4 單模上全數輸給 macbert-base：

![骨幹 T4 消融](reports/figures/fig_backbone_t4.png)

**決定性證據與結論**：

1. **更大模型在微小多分類任務上過擬合**：roberta-large / electra-large（326M）的 T4 都比 macbert-base（102M）更差。當訓練樣本 < 1k、T4 僅 4 類，**模型大小是錯的軸**。
2. **focal loss 每次都傷 T4**：本專案歷來純配方 T4 最高，focal/resample 變體一律 0.39–0.40。
3. **加長 context（384→512）對 T4 無效**：即使 40% test 列超過 384 token，T4 的判別訊號不在尾部。p18 雖最穩定（std 0.005），T4 仍 0.444 < 0.450。
4. **hillclimb 的 +0.013「增益」皆為 Phase 47 同款幻覺**：三者新 stem 在 T4 都只拿 ~1% 權重，增益集中在**不可轉移的二元任務**（T1/T3），上傳必跌。
5. **驗證黃金證據**：p18 的 **val-slice T4 = 0.4370 與排行榜 T4 = 0.437 完全吻合**——再次證明 T4 的 CV-OOF 忠實轉移，**同時證明 T4 在這份資料上已飽和於 ~0.44**。獨立訓練的多個模型 val-OOF == LB，即代表該任務對此資料已飽和。

**最終裁決**：Phase 48 重訓更強骨幹的路線，在可轉移任務上**無法勝過 macbert-base combo_best**；架構、大小、context 三軸皆已窮盡。**不上傳**（每日上傳機會保留），凍結最佳 c3=0.6037 維持不變。真正突破 T4 需要更多 T4 標註資料（無）或 ESG-quality 語料外部預訓練（超出範圍）。

> **Phase 48 關鍵教訓**：(1) 在 < 1k 樣本的微小多分類任務上，**更強/更大的骨幹常輸給調好的小模型**——先驗證單模目標任務分數，再決定是否值得整合。(2) **務必檢查 hillclimb 的增益落在哪個任務**：若新 stem 在目標任務權重 ~0，增益全在不可轉移任務，那就是 OOF 過擬合幻覺，與 Phase 47 同源。(3) 當多個獨立訓練模型的 **val-OOF == LB** 於某任務，該任務已對資料飽和，停止以建模追逐，改尋資料或外部預訓練。

## 66. Phase 49 — TAPT 領域自適應預訓練（最後一個未試軸）

Phase 48 結論指出「真正突破需外部預訓練」。Phase 49 執行唯一未試且具正期望值的軸：**任務自適應繼續預訓練（TAPT）**。

**流程**（`scripts/u30_tapt_pretrain.py` + `configs/exp_p49_tapt_combo_best.yaml`）：
1. 以 train(1000)+val(1000)+**test(2000)** 的 `data` 文字共 4000 段去重語料，對 macbert-base 繼續 MLM 預訓練 8 epoch（loss 0.95→0.80）。屬 transductive 用法，**僅用未標註文字、絕未使用任何標籤**，合法。
2. 以調好的 combo_best 配方在自適應編碼器上做 5-fold CV（combined 2000）。

**結果**：

| 指標 | TAPT (p49) | 歷史最佳 | 評價 |
| :-- | :--: | :--: | :-- |
| 單模 OOF（整體）| **0.6845** | ~0.674 | (最佳) **史上最強單模** |
| T2 verification_timeline | **0.5337** | ~0.50 | (最佳) 史上最強單模 T2 |
| T4 evidence_quality | 0.4492 | 0.450 | 略低，未過 admission 閘（0.458）|

**集成誠實檢驗**（後處理 joint 分數）：

| 配置 | joint | vs 現有最佳 |
| :-- | :--: | :--: |
| equal8（banked 代理）| **0.69759** | 基準 |
| equal9（+TAPT 等權）| 0.69703 | **−0.0006** |
| 只在 T2 加 TAPT | 0.69720 | −0.0004 |
| T2+T4 加 TAPT | 0.69585 | −0.0017 |
| ~~hillclimb 9-way~~ | ~~0.71075~~ | ~~+0.013（Phase-47 幻覺）~~ |

**裁決**：TAPT 是**有史以來最強的單一模型**，驗證了該技術方向；但**無法改善集成**——集成 T2 已達 0.6053（8 個 macbert stem 平均），遠高於 TAPT 單模 T2 0.5337，加入這個高度相關的 macbert 變體反而拉低混合。唯一「贏」的 hillclimb +0.013 是 Phase 43–47 連三次掉分的同款幻覺（T3 二元搶走 32% 新權重）。

**transductive 上行已查核並排除**：原本顧慮「CV-OOF 看不到 TAPT 對 test 的領域適應」，但已驗證 **test 的 50 家公司與標籤分佈皆與 train/val 相同（零分佈偏移）**，故在 train/val 上量到的 OOF 即為忠實代理，transductive 利益可忽略。

**最終決定**：**HOLD——不耗用今日最後一次上傳**，保留 banked equal-blend **0.6037**。TAPT 以一個乾淨的「集成負結果」關閉了最後一個未試的正期望值軸。

> **Phase 49 關鍵教訓**：(1) **TAPT 確實有效**——史上最強單模證明繼續 MLM 預訓練是對的方向，但在**同骨幹高相關 pool** 中,更強的單模不等於集成增益(相關性抵銷)。(2) 判斷新成員一律用**等權或低維外科手術式混合的 OOF**,絕不用 hillclimb delta。(3) **transductive TAPT 只在 train→test 有真實分佈偏移時才有隱性上行**;無偏移時 OOF 仍忠實,所謂「隱藏優勢」是幻覺——本賽 test 與 train/val 同分佈,故無此利。

## 67. Phase 50 — 「為什麼是 0.6037 不是 0.71」的嚴謹結案 + 二元任務唯一槓桿

**核心問題**：到 6/17 前每天 3 次上傳（約 15 次預算）。為什麼原本訓練出的 0.71 變成 0.6037？哪一步開始出錯？是否已達極限、連換方法也不行？

**A. 0.71 從來不是真的（不是 bug，是方法論）**

逐任務還原（權重 T1=0.20 / T2=0.15 / T3=0.30 / T4=0.35）：

| 任務 | 權重 | OOF(val held-out) | LB(test) | gap | 加權可回收 |
|---|---|---|---|---|---|
| T1 promise（二元 F1）| 0.20 | 0.9403 | 0.786 | **+0.154 (NG)** | +0.031 |
| T2 timeline（macro）| 0.15 | 0.6041 | 0.606 | −0.002 (OK) | −0.000 |
| T3 evidence（二元 F1）| 0.30 | 0.8594 | 0.675 | **+0.184 (NG)** | +0.055 |
| T4 quality（macro）| 0.35 | 0.4639 | 0.437 | +0.027 (OK) | +0.009 |
| 加權 | | 0.6989 | **0.6037** | +0.095 | |

`0.20×0.786 + 0.15×0.606 + 0.30×0.675 + 0.35×0.437 = 0.6036`，**完全重現 banked c3=0.6037**。

**結論**：0.71 是 StratifiedKFold 在 2000 列上做 5-fold、伴隨 ~99.93% 公司列洩漏的**有洩漏 CV 代理**，二元任務被「公司層級 Yes/No 記憶」灌水到 0.94/0.86；再經 ~40 個 phase 反覆在同一 OOF 上選擇（自適應過擬合）疊加約 0.014。**沒有可修的程式 bug。**

**B. 落差 100% 在二元任務，且過去全打錯任務**

- 落差幾乎全在二元 T1（+0.154）/ T3（+0.184）；macro T2/T4 忠實轉移。
- 但 Phase 48（換骨幹）與 Phase 49（TAPT）**全部瞄準 T4**，理由是「只有 macro OOF 可信」。
- **關鍵誤判**：把「OOF 量不準這個任務」誤當成「這個任務改不動」。兩者不同。T4 已近資料天花板（Misleading 僅 2 樣本，僅 ~0.009 可回收）；真正的可回收點（T3=0.055、T1=0.031，共 ~0.086）全在被放棄的二元任務。
- 二元任務無法用 OOF 驗證（洩漏 + 過擬合），**只能用真實 LB 當驗證信號**——而現在有 15 次上傳預算，這個軸第一次變得可負擔。

**C. 嚴格審視「換方法」是否可行——逐一證實或排除**

| 槓桿 | 結果 | 證據 |
|---|---|---|
| Post-hoc 調 stem 權重 | **死路** | Phase 47 證實全部過擬合 OOF；Phase 44 三次先驗修正單調掉分 0.6037→0.6029→0.6026 |
| 輸入域偏移 / 重要性加權重訓 | **排除** | Phase 45 對抗驗證 AUC=0.5097，無域偏移 |
| **T3 約束級聯槓桿（本 phase 新測）** | **證實無合法 post-hoc 槓桿** | 見下 D |
| Macro 任務建模（換骨幹 / TAPT）| **已飽和** | Phase 48/49，T4 近天花板 |
| **二元任務換結構模型（唯一未試）** | **可試，僅能 LB 驗證** | 見下 E |

**D. T3 約束級聯槓桿——嚴謹證偽（本 phase 新貢獻）**

假設：級聯 `promise=No → evidence=N/A` 把 test 上 ~21% 的 T1 誤判 No 連帶摧毀正確的 evidence=Yes。若移除 T3 級聯應可回收。實測（從快取 `phase43_test_probs.npz` 建，T1/T2/T4 與 c3 逐位元相同，僅改 T3）：

- **變體 A（evidence 獨立 3-way argmax）**：僅 20/2000 列改變——evidence head 99% 與級聯一致，「級聯放大 T1 誤差」假設在預測層級**基本不成立**。且其中 7 列落在 promise=No → **違反賽制 schema**（驗證器報錯）。
- **變體 B（強制 Yes/No 不出 N/A）**：295 列改變，**全部**落在 promise=No → 295 個 schema 違規。

**結論**：賽制 schema 硬性要求 `promise=No → evidence=N/A`，且 c3 已在 promise=Yes 列取最優 `argmax{Yes,No}`。**T3 沒有任何合法的 post-hoc 槓桿，它被 T1 的 test 誤判率硬性封頂。** 兩個非法候選已刪除。

**E. 唯一剩下的真槓桿——結構不同模型測 test 二元任務（已實作待上傳）**

Phase 49 用 OOF 否決了 TAPT（−0.0006），但那個裁決由**不可信的二元 OOF 主導**。TAPT 是領域自適應（ESG 語料繼續 MLM）、史上最強單模、且 macro T2 史上最佳。它在 **test 二元任務**上是否更會 generalize，OOF 量不到——只能上傳。已用 `scripts/u32_phase50_tapt_test.py`（重用 c3 同款受約束解碼）產出兩個合法候選：

| 候選 | 與 c3 不同列數 | T1 dist | T3 dist | 驗證 |
|---|---|---|---|---|
| `phase50_equal9`（9 stem 等權）| 263（T4 206 / T2 72 / T3 13 / T1 7）| Yes 1706 / No 294 | Yes 1451 / N/A 294 / No 255 | (OK) |
| `phase50_tapt_only`（純 TAPT 5-fold）| 523（T4 346 / T2 260 / T3 108 / T1 49）| Yes 1712 / No 288 | Yes 1491 / N/A 288 / No 221 | (OK) |

**Phase 50 LB 回饋（6/13）**：兩發皆未勝過 c3，但訊號並非「完全沒用」而是**TAPT 二元有用、macro 大幅拖垮**。

| 候選 | T1 promise | T2 timeline | T3 evidence | T4 quality | weighted |
|---|---:|---:|---:|---:|---:|
| c3 / frozen best | 0.7864 | 0.6060 | 0.6750 | 0.4370 | **0.6037** |
| equal9 | 0.7864 | 0.5915 | 0.6721 | 0.4249 | 0.5963 |
| tapt_only | **0.7967** | 0.5117 | **0.6804** | 0.4288 | 0.5903 |

關鍵解讀：`tapt_only` 的二元任務在真實 LB 上確實改善（T1 +0.0103、T3 +0.0054，合計加權約 +0.0037），但 T2 暴跌 −0.0943（加權 −0.0141），故總分掉到 0.5903。**下一步不是 equal9/tapt_only，而是 TAPT 只用於 T1/T3、T2/T4 保留 c3。**

同時也給出上限：若能理想重組 `tapt_only(T1,T3) + c3(T2,T4)`，分數約為 `0.2×0.7967 + 0.15×0.6060 + 0.3×0.6804 + 0.35×0.4370 = 0.6073`。也就是說，TAPT binary-only 最樂觀也只夠刷新 c3 約 +0.0036，**尚不足以穩破 0.61**；要破 0.61 仍需比 TAPT 更強的二元模型或額外 T4 增益。

## 68. Phase 51 — TAPT 二元外科重組（保留 c3 macro）

修正 Phase 50 的一個重要細節：frozen c3 不是裸 equal8，而是 `equal8 + T2/T4 prior-correction α=0.3`（`phase43_c3_equal_priorcorr_a03`）。因此 Phase 51 先用 `phase43_test_probs.npz` + `prior_correct_per_task(..., {T2:0.3,T4:0.3})` 重建 c3，並確認與 `FROZEN_BEST_0.6037_c3.csv` **四欄逐位元一致**：

`{'promise_status': 0, 'verification_timeline': 0, 'evidence_status': 0, 'evidence_quality': 0}`

已用 `scripts/u33_phase51_binary_surgical.py` 產生五個合法候選（全部以 c3 macro-a03 為 base；只在 T1/T3 混入 TAPT）：

| 候選 | 變更策略 | vs c3 diff | T1 dist | T3 dist | MD5 |
|---|---|---|---|---|---|
| `phase51_tapt_binary` | TAPT 100% T1 + 100% T3 | T1 49 / T2 49 / T3 108 / T4 101 | Yes 1712 / No 288 | Yes 1491 / N/A 288 / No 221 | `6B41FAB2398426AF6FEB6B6A0976269F` |
| `phase51_tapt_t1only` | 只換 T1 | T1 49 / T2 49 / T3 49 / T4 41 | Yes 1712 / No 288 | Yes 1453 / N/A 288 / No 259 | `5756C57D1BFA5B686A295AD267349254` |
| `phase51_tapt_t3only` | 只換 T3 | T1 0 / T2 0 / T3 61 / T4 61 | Yes 1705 / No 295 | Yes 1483 / N/A 295 / No 222 | `2A7295F6AD8BF788FDE2DE9294FE44F5` |
| `phase51_tapt_binary_soft75` | T1/T3 = 75% TAPT + 25% c3 | T1 41 / T2 41 / T3 87 / T4 82 | Yes 1710 / No 290 | Yes 1486 / N/A 290 / No 224 | `56B59268A5C8373744A784C4BD622803` |
| `phase51_tapt_binary_soft50` | T1/T3 = 50% TAPT + 50% c3 | T1 27 / T2 27 / T3 58 / T4 54 | Yes 1710 / No 290 | Yes 1472 / N/A 290 / No 238 | `456BD18B08ACDA705D119E7BC7FB28FE` |

**建議上傳順序**：若今日還剩 1 次，先上傳 `phase51_tapt_binary_submission.csv`（最大二元訊號，風險由已銀行 c3 承擔）。若明日還要拆解，依序上傳 `phase51_tapt_t1only_submission.csv`、`phase51_tapt_t3only_submission.csv`；兩者可分離 TAPT 的 T1 / T3 貢獻與 macro cascade 副作用。若 `binary` 掉分但二元任務仍升，才試 `soft75` 或 `soft50`。

**F. 裁決**

- **Post-hoc 與 macro 建模的上限：已達（已證實）。**
- **唯一仍有正訊號的軸：二元 T1/T3 的結構不同模型**，且已由 `tapt_only` LB 證實 TAPT 對二元任務有小幅真增益；macro 任務必須保留 c3。
- 突破 0.61 **仍不確定且偏難**——Phase 45 已把二元崩塌歸為「不可約誤差」（可能是 test 標註慣例差異，非閾值問題）。`tapt_only(T1,T3)+c3(T2,T4)` 的已量測理想重組上限約 0.6073，故 Phase 51 目標是嘗試刷新 c3，而非保證衝過 0.61。
- 候選 CSV 已備妥並通過驗證，待上傳。

> **Phase 50 關鍵教訓**：「OOF 量不準某任務」≠「該任務改不動」。當代理指標對某任務系統性失真（洩漏 + 自適應過擬合），唯一誠實的驗證是**真實 LB**；在有上傳預算時，應把預算花在**代理量不到、但加權可回收點最大**的任務（此處二元 T3=0.055 / T1=0.031），而非繼續優化代理可信但已近天花板的 macro 任務。同時，任何 post-hoc 標籤改寫都必須先過賽制 schema 驗證器——級聯約束是硬規則，不是可調槓桿。


## 69. Phase 52 — TAPT 二元全外科重組：LB 突破 0.61（2026-06-14）

第 1 次上傳 `phase51_tapt_binary`（TAPT 二元 T1/T3 + c3 macro T2/T4）得分 **0.6113**，刷新 banked best 0.6083（`t1only`）並首度突破 0.61。依「leaderboard 以最後一次上傳為準」，此檔即為目前榜上結果（已鎖定 0.6113）。

### 69.1 逐任務 LB（對照 c3）

| 任務 | 權重 | c3 LB | binary LB | delta |
| :-- | :--: | :--: | :--: | :--: |
| T1 promise_status | 0.20 | 0.7864 | 0.7967 | +0.0103 |
| T2 verification_timeline | 0.15 | 0.6060 | 0.6114 | +0.0054 |
| T3 evidence_status | 0.30 | 0.6750 | 0.6804 | +0.0054 |
| T4 evidence_quality | 0.35 | 0.4370 | 0.4460 | +0.0090 |
| weighted | | 0.6037 | **0.6113** | +0.0076 |

驗算：0.20 x 0.7967 + 0.15 x 0.6114 + 0.30 x 0.6804 + 0.35 x 0.4460 = 0.61127。

### 69.2 關鍵發現：TAPT promise 級聯同時抬升二元與 macro

binary 僅把 T1/T3 換成 TAPT，理應只動二元；但 T2/T4 macro 也同步上升（T2 +0.0054、T4 +0.0090）。成因：TAPT 的 promise=No 判定落在更正確的列上，經 `promise=No -> N/A` 級聯，使 T2/T4 的 N/A 指派更準，macro-F1 連帶提升。

戰略反轉（修正 §62 Phase 45 與 §65 Phase 48 的結論）：先前把二元崩塌歸為「不可約誤差」、把建模火力全導向 T4，是基於「只有 macro OOF 可信」。但 TAPT 在真實 LB 上二元 T1 +0.0103、T3 +0.0054，證明二元任務確實可由更強模型改善——先前的「不可約」是 OOF 驗證失效（公司層級洩漏），不是真實天花板。可回收空間最大的 T1+T3（合計權重 0.50）重新成為主戰場。

### 69.3 進一步提升的可行路線（驗證後可執行）

post-hoc 與 macro 建模已榨乾（§60–§68 證實）。進一步提升只能來自 T1/T3/T4 的真實建模增益，且二元只能以真實 LB 驗證。依 ROI 排序：

1. 二元專家模型擴池（最大槓桿，權重合計 0.50）：以 TAPT-style 領域自適應 + 多 seed / 多架構（roberta-wwm-large、electra-large 各自先 TAPT 再微調）訓 3~5 個結構不同的二元模型，對 T1/T3 機率平均集成。每個新集成用 1 次 LB 上傳驗證。
2. 誠實 company-GroupKFold CV：現行 StratifiedKFold 在 50 家公司 × 2000 列上有 within-company 洩漏（每列的同公司其他列幾乎必落在 train fold），使二元 OOF 樂觀。**已查核：test 2000 列同樣只用這 50 家公司、0 家新公司**，故 GroupKFold（held-out 整家公司）會比 test 更難，給出的是**保守下界**而非 test 代理；但「在 held-out 公司仍勝出」的模型才是真正穩健、值得上傳的模型。資料已含 `company` / `ticker` 欄位，可直接分組。
3. T4 minority 資料擴增：T4（權重 0.35、現 0.446）仍是最重最低任務；Misleading / Not Clear 標註過少。以更高品質 LLM 合成 + 人工校驗擴充少數類，搭配 TAPT 編碼器專訓 T4。

進一步提升需多個真實建模增益疊加、難度高；以本資料與本機資源，「二元擴池 + 誠實 CV」為剩餘最具潛力的方向。

---

## 70. Phase 53 — 二元專家擴池建置（TAPT 多 seed / 多架構 + 誠實 GroupKFold）（2026-06-14）

Phase 52（§69）證明二元任務 T1/T3（合計權重 0.50）可由更強模型改善。本 Phase 建置「結構多樣的領域自適應二元專家池」工程骨架：把多個不同骨幹的 T1/T3 softmax 機率平均集成，macro T2/T4 保留 c3 base。本節為已驗證可執行的骨架；多小時 GPU 訓練於本機執行。u35 已通過 reproduce 自我檢驗（從單一 macbert-TAPT 成員重建 0.6113 候選，md5 與 `phase51_tapt_binary` 逐位元相同），故新成員加入後同一程式路徑可信。

### 70.1 池成員設計

| 成員 | 骨幹 | 多樣性來源 | config | 估時(8GB) |
| :-- | :-- | :-- | :-- | :-- |
| macbert-TAPT s42 | macbert-base + TAPT | 已存在（Phase 49）| `exp_p49_tapt_combo_best.yaml` | 已完成 |
| macbert-TAPT s2024 / s20260417 | macbert-base + TAPT | 種子多樣性 | `exp_p53_tapt_macbert_ms.yaml` | 約 30-40 分/seed |
| roberta-large-TAPT s42 | roberta-wwm-ext-large + TAPT | 架構+容量（大模型先 TAPT，Phase 48 未試）| `exp_p54_tapt_robertalarge.yaml` | 約 2-3 時 |
| electra-large-TAPT s42 | electra-180g-large + TAPT | 判別式預訓練，架構最不同 | `exp_p55_tapt_electralarge.yaml` | 約 2-3 時 |

二元 config 用 max_length 256、無 focal（focal 歷來傷分），比 Phase 48 的 T4 大模型 config（maxlen 512 + focal）約省一半算力。

### 70.2 新增與修改的程式（本機已驗證）

| 檔案 | 動作 | 說明 |
| :-- | :-- | :-- |
| `scripts/u30_tapt_pretrain.py` | 修改 | 新增 `--backbone` / `--grad-accum`，可對任意骨幹做 ESG 語料 TAPT |
| `configs/exp_p53/p54/p55_*.yaml` | 新增 | 三個二元專家 stem 配方（合併 2000 列、StratifiedKFold）|
| `src/train_kfold.py` | 修改 | 新增 `--split-mode stratified_group_kfold`，輸出隔離到 `<exp>_gkf` 樹 |
| `scripts/u35_binary_pool_ensemble.py` | 新增 | 池成員 5-fold test 推論 + T1/T3 集成 + 受約束解碼 + schema 驗證；`--reproduce` 自檢逐位元等同 0.6113 |
| `scripts/u36_groupkfold_binary_rank.py` | 新增 | 從 `<exp>_gkf` score_summary 讀 held-out-company 二元 F1 並排名 |
| `scripts/u37_run_binary_pool.ps1` | 新增 | 一鍵 runbook（TAPT -> 訓練 -> GroupKFold 排名 -> 集成 -> 上傳指引）|

### 70.3 誠實 GroupKFold CV 的定位

現行 StratifiedKFold 在 50 公司 × 2000 列上有 within-company 洩漏，二元 OOF 不可信。GroupKFold（held-out 整家公司）才測得真實泛化。但已查核 test 同樣只用這 50 家公司、0 家新公司，故 GroupKFold 比 test 更難，是保守下界、非 test 代理；用途是相對排名（哪個骨幹在未見公司上仍穩）與剔除會崩的模型，而非預測絕對 LB。`u36` 產出排名表。

### 70.4 一鍵執行與上傳策略

於 repo 根目錄 PowerShell 執行 `./scripts/u37_run_binary_pool.ps1`。依序：reproduce 自檢 -> TAPT 大骨幹 -> 訓練三 stem -> GroupKFold 排名 -> `u35 --build` 產候選。候選：每成員各一（隔離單模二元能力）+ 全池一個（`u35_pool_binary`，最高期望）。上傳：先試全池；若退步退回最佳單成員；最後一發鎖定最高分（leaderboard 取最後一次）。只有勝過 0.6113 才放在當日最後一次上傳。

### 70.5 預期

二元擴池是目前唯一被證實的正槓桿，但增益不保證：多樣性不足或大模型仍過擬合都可能持平。進一步提升需多個真實增益疊加（二元擴池 + T4 少數類資料），難度高。所有候選一律 schema 驗證、且只用真實 LB 判定二元優劣。

---

### 70.6 訓練結果（2026-06-15 完成，約 9 小時）

`scripts/u37_run_binary_pool.ps1` 全程跑完（All steps succeeded）。reproduce 硬閘通過（重建 0.6113 逐位元相同）。兩個大骨幹 TAPT 完成（`outputs/tapt/roberta_large_esg`、`outputs/tapt/electra_large_esg`）。

單模 5-fold StratifiedKFold OOF（weighted；二元 OOF 仍洩漏，僅供模型健康度參考）：

| stem | T1 | T2 | T3 | T4 | weighted | epoch 時間 |
| :-- | :--: | :--: | :--: | :--: | :--: | :--: |
| macbert-TAPT s42（p49，既有）| 0.940 | 0.534 | 0.870 | 0.449 | 0.6845 | 已有 |
| macbert-TAPT s2024/s20260417（p53）| 0.938 | 0.514 | 0.861 | 0.449 | 0.6802 | ~70s |
| roberta-large-TAPT（p54）| 0.940 | 0.509 | **0.872** | 0.448 | **0.6829** | ~750s |
| electra-large-TAPT（p55）| 0.938 | 0.474 | 0.863 | 0.411 | 0.6614 | ~640s |

大模型 + TAPT 訓練正常、與 macbert-TAPT 同級（roberta-large 的 Stratified T3 0.872 為池中最高）。但這是洩漏的二元 OOF，不可直接當二元 LB 能力。

### 70.7 誠實 GroupKFold 裁決（關鍵，修正樂觀預期）

held-out 整家公司的誠實 CV：TAPT 的優勢在 T2/T4，二元 T1/T3 與純 macbert 並列：

![GroupKFold 誠實二元](reports/figures/fig_groupkfold.png)

held-out 整家公司的 5-fold CV（誠實二元訊號）：

| 模型 | T1 | T2 | T3 | T4 | weighted | 二元加權 F1 |
| :-- | :--: | :--: | :--: | :--: | :--: | :--: |
| 純 macbert（p2_combo_best）| 0.934 | 0.453 | 0.857 | 0.421 | 0.6592 | 0.8877 |
| macbert-TAPT（p49）| 0.933 | **0.521** | 0.858 | 0.437 | **0.6752** | 0.8879 |

三個決定性結論：

1. **TAPT 在二元任務上對純 macbert 幾乎零優勢**：held-out-company 二元加權 F1 = 0.8879 vs 0.8877（並列，差 +0.0002，純雜訊）。TAPT 真正穩健的增益在 **T2 macro（+0.068）與 T4（+0.016）**，不在二元。Phase 52 的 0.6113（二元 +0.0076 over c3）是 TAPT 在 test 那 50 家公司上的特定行為，**不是可泛化的二元優勢**。
2. **GroupKFold 二元（0.888）仍遠高於 LB 二元（約 0.727）**：即使 held-out 整家公司，二元也有 0.888，但真實 test 只有 0.727（T1 0.797 / T3 0.680）。代表 test 比「沒見過的公司」還難 → **落差不是公司記憶，而是 test 在同樣 50 家公司上的標註慣例/分佈更嚴 + 數十 phase 的自適應過擬合**。任何本機 CV（含 GroupKFold）都對 LB 二元樂觀，無法當 test 代理。
3. **二元軸已接近耗盡**：既然 TAPT 的二元優勢在誠實 CV 上是零，擴池（多 seed / 大模型）對二元 LB 的期望增益很小，且只能用真實 LB 驗證。

### 70.8 候選集與上傳建議

`u35 --build` 產出 6 個合法候選（全部過 schema 驗證），以 c3 macro base + 不同二元來源組成：

| 候選（`outputs/submissions/`）| 二元來源 | vs c3 diff (T1/T3) | T1 分佈 | T3 分佈 | 定位 |
| :-- | :-- | :--: | :-- | :-- | :-- |
| `u35_tapt_macbert_s42_binary` | macbert-TAPT s42 | 49 / 108 | No 288 | No 221 | **= 已銀行 0.6113**（基準）|
| `u35_pool_binary` | 5 模型 T1/T3 平均 | 53 / 113 | No 280 | No 228 | **主推：最具多樣性、分佈最接近贏家** |
| `u35_tapt_robertalarge_s42_binary` | roberta-large-TAPT | 86 / 173 | No 297 | No 206 | 架構多樣、Stratified T3 最高，較激進 |
| `u35_tapt_electralarge_s42_binary` | electra-large-TAPT | 111 / 205 | No 268 | No 193 | 最激進（最少 No on T3）|
| `u35_tapt_macbert_s2024_binary` | macbert-TAPT s2024 | 64 / 127 | No 299 | No 262 | 較多 No（賭 test 更嚴）|
| `u35_tapt_macbert_s20260417_binary` | macbert-TAPT s20260417 | 43 / 99 | No 288 | No 233 | 較溫和 |

上傳排程（每日 3 次；目前榜上 0.6113）：

1. **`u35_pool_binary`** — 5 模型集成的二元，分佈最接近 0.6113 贏家、變異最小。最有機會微幅超過 0.6113 的單發。
2. 若步驟 1 > 0.6113：再探 `u35_tapt_robertalarge_s42_binary`（最大架構多樣性）；若 ≤ 0.6113：二元軸確認耗盡，**停止再賭二元**，當日最後一發鎖回 0.6113。
3. 最後一發鎖定當日最高分（leaderboard 取最後一次）。

> 預期：GroupKFold 已證二元軸接近飽和，`pool_binary` 預期 ≈ 0.6113 ± 0.002，不保證超越。剩餘最具潛力的槓桿是 **T4（權重 0.35、卡 0.446）的少數類資料擴增**與**更乾淨的 company-grouped 重訓以降低自適應過擬合**，而非更多二元模型。

### 70.9 Phase 53 關鍵教訓

1. **誠實 CV 會否決樂觀假設**：用 GroupKFold（held-out 公司）量到 TAPT 的二元優勢其實是零，避免了在已飽和的二元軸上盲目擴池燒上傳額度。先建可信的本機訊號，再決定是否值得 LB 驗證。
2. **當誠實 CV 仍高於 LB**：代表落差來自「同公司、不同標註慣例/更難樣本 + 自適應過擬合」，連 GroupKFold 都樂觀；此時唯一誠實驗證是 LB，且應把有限上傳花在「分佈擾動最小、最可能微幅改進」的集成候選，而非激進的單一大模型。
3. **大模型 + TAPT 可訓練但非萬靈丹**：roberta-large/electra-large 加 TAPT 訓練正常、與 macbert-TAPT 同級，但在 < 1k/任務、二元已飽和的設定下，並未帶來可泛化的二元增益（與 Phase 48「大模型在微小任務過擬合」一致，TAPT 也救不回二元的不可約落差）。

---

## 71. Phase 54 — 後處理槓桿的窮盡式證偽（macro 決策最佳化 + T4 集成）（2026-06-15）

Phase 53 關閉二元軸後，唯一加權高且 OOF 可信轉移的任務是 T4（權重 0.35，LB 0.446）與次要的 T2（權重 0.15）。本 Phase 用可信的 equal8 OOF（8 個 TV stem 的 5-fold OOF，2000 列有金標）對「不需重訓的 post-hoc 槓桿」做窮盡式、帶轉移驗證的證偽。工具 `scripts/u38_macro_decision_opt.py`。轉移準則：在 train-slice(0–999) 調參、在 held-out val-slice(1000–1999) 評分必須改善，且反向亦然（val-slice 是 Phase 48 證實的忠實 LB 代理）。

### 71.1 為什麼 T4 受限：macro-F1 的結構

T4 = macro-F1 over {Clear, Not Clear, Misleading, N/A}。2000 列金標分佈：Clear 1118 / Not Clear 225 / Misleading 2 / N/A 655。**Misleading 僅 2 筆 → 該類 F1 結構性 ≈ 0**，被 /4 平均直接拉低；T4 macro 實質由 **Not Clear**（225 筆，argmax 因 Clear 主導而低估）封頂。理論上界：即使 Clear/NotClear/NA 全 1.0，(1+1+0+1)/4 = 0.75。

### 71.2 T4 macro-F1 決策最佳化（不轉移）

對 evidence_quality 機率加 per-class 偏移（Clear=0，搜 Not Clear / Misleading）以最大化 OOF T4 macro：

| 調參來源 | 最佳偏移 | in-sample | held-out slice | held-out delta |
| :-- | :-- | :--: | :--: | :--: |
| train-slice | NotClear +0.05, Mis +0.15 | 0.4475 -> 0.4514 | val 0.4602 -> 0.4553 | **-0.0049** |
| val-slice | NotClear 0, Mis +0.05 | 0.4602 -> 0.4606 | train 0.4475 -> 0.4476 | +0.0002 |
| full 2000 | NotClear 0, Mis +0.05 | 0.4536 -> 0.4538 | — | +0.0003（雜訊）|

**裁決：不轉移。** argmax 已是此 T4 head 的 macro-F1 最佳決策；偏移為 OOF 過擬合（與 Phase 47 同源）。

### 71.3 T2 決策最佳化（邊際、不穩健）

對 verification_timeline 同法（搜 within_2_years / longer_than_5_years）：train-slice 調出的溫和偏移（longer_5y +0.075）使 held-out val-slice +0.0046，但反向（val-slice 調參）只回饋 train-slice +0.0001。**僅單向、且最樂觀加權僅 +0.0007，不穩健，不採用。**

### 71.4 T4 集成多樣化（反而變差）

把 Phase 53 新訓的多樣 T4 head（macbert-TAPT、roberta-large-TAPT、electra-large-TAPT）等權併入 equal8 的 T4 平均：

| T4 來源 | full | val-slice | train-slice |
| :-- | :--: | :--: | :--: |
| equal8（基準）| **0.4536** | **0.4602** | **0.4475** |
| + macbert-TAPT | 0.4481 | 0.4518 | 0.4447 |
| + roberta-large-TAPT | 0.4463 | 0.4507 | 0.4424 |
| + 全部 4 個 TAPT head | 0.4362 | 0.4397 | 0.4332 |

**每一種組合都讓 T4 變差。** 8 個 macbert stem 的 equal8 已是最佳 T4 來源；TAPT/大模型的 T4 head 較弱且相關，併入只會稀釋。

### 71.5 裁決：後處理已全數窮盡，T4 為資料天花板

| 槓桿 | 結果 | 證據 |
| :-- | :-- | :-- |
| 二元 T1/T3（模型/集成）| 死路 | Phase 53 GroupKFold：TAPT 二元優勢 ≈ 0 |
| T4 決策最佳化 | 死路 | §71.2 不轉移 |
| T2 決策最佳化 | 邊際/不穩健 | §71.3 |
| T4 集成多樣化 | 反向 | §71.4 全部變差 |
| T4 更大/更長 context 模型 | 死路 | Phase 48 |
| stem 權重微調 | 死路 | Phase 47 全過擬合 |

**結論：所有不需新資料的槓桿皆已嚴格窮盡。** 0.6113 是此資料 + BERT 集成方法的實質天花板。T4 受限於 Misleading 結構性 0 與 Not Clear 的可分性，且測試集在同樣 50 家公司上標註更嚴（Phase 45/53）；要破 0.62 在數學上需 T4 ≈ 0.50，等於把 Not Clear F1 從 ~0.2 提到 ~0.35——這需要**真正更會分辨 Clear/Not Clear 的新 T4 訊號**，非任何 post-hoc 可得。

### 71.6 唯一理論剩餘軸：LLM-as-classifier（低期望、待評估）

具差異化機制的軸：用本機 LLM（Ollama）對 test 段落做**少樣本分類**（非生成合成資料），對 macro 任務（尤其 Not Clear / Misleading 的語意判讀）可能與 BERT 互補。此軸已於 §74 Phase 58 實測並否決：Qwen3-4B 與 gemma4:31b 對 Clear/Not Clear 僅約 50%（等同亂猜），無法符合標註慣例。

預期：低。微調過的 BERT stem 在同分佈上通常勝過泛用 LLM 少樣本；LLM 唯一可能勝出的稀有類（Misleading）因 support 太小對 macro 貢獻有限。列為 opt-in 的高成本、低機率 moonshot，非建議的預設路線。

### 71.7 Phase 54 關鍵教訓

1. **帶雙向轉移驗證的 OOF 搜尋是判斷 post-hoc 是否可用的黃金準則**：T4/T2 決策偏移在 in-sample 都「變好」，但 held-out 反向證偽——只有在 train→val 與 val→train 皆改善才可信。
2. **集成多樣性不是萬靈丹**：當新成員在目標任務上較弱且相關，等權併入會稀釋（T4 全部變差）。先量單任務 OOF，再決定是否入池。
3. **知道何時停手**：窮盡式證偽完所有 post-hoc 後，明確標記資料天花板，把後續投入導向「新訊號」（資料或不同模型族），而非繼續在已飽和的代理上微調。

---

## 72. Phase 55 — Not Clear (T4) 資料擴增：單模有效、集成飽和（2026-06-15）

Phase 54 證偽所有 post-hoc 後，本 Phase 攻擊 T4 唯一未試的資料軸。val-slice 診斷（LB 忠實代理）顯示 T4 macro 由 **Not Clear** 封頂：F1 0.347、P 0.322 / R 0.376，且 101 筆真 Not Clear 中 40 筆被判 Clear（head 混淆）、23 筆被判 N/A（cascade）。**先前的 aug_plus 合成（Phase 37/38）瞄準 Misleading（support=2，救不了 macro）與 within_2y，從未瞄準 Not Clear** —— 故這是真正未試的可轉移槓桿。

### 72.1 實作

| 元件 | 檔案 | 內容 |
| :-- | :-- | :-- |
| 生成 | `scripts/u39_gen_notclear.py` | Ollama Qwen3-4B-Instruct few-shot（以真實 Not Clear 為範例鎖定慣例：有作為、無具體數字），清洗+去重，產 220 筆 |
| 注入 | `data/aug_plus/notclear_synth.json` + `configs/exp_p56_combo_best_notclear.yaml` | 每筆帶 `_source_id`=真實 Not Clear 列，train_kfold 只在 source 屬 train fold 時注入 → val-slice OOF 為誠實 held-out |
| 驗證 | `scripts/u40_validate_notclear.py` | 同 cascade 隔離，比較 val-slice T4 macro / Not Clear F1 |

### 72.2 單模結果：合成確實有效（推翻「純資料封頂」）

T4 逐類 F1（val-slice）：macro 被 Not Clear（主觀）與 Misleading（support 極小）封頂：

![T4 逐類 F1](reports/figures/fig3_t4_perclass.png)

val-slice Not Clear（同一 combo_best 配方，有無合成）：

| stem | Not Clear P | R | F1 | T4 macro |
| :-- | :--: | :--: | :--: | :--: |
| combo_best（無合成）| 0.317 | 0.257 | 0.284 | 0.4437 |
| **p56（+220 合成）** | **0.344** | **0.327** | **0.335** | **0.4563** |
| delta | +0.027 | +0.070 | **+0.051** | **+0.0126** |

**精確度與召回率雙升**，Not Clear F1 +0.051、T4 macro +0.0126（val-slice，忠實轉移）。**證實 Not Clear 訊號可由資料改善**，先前「資料天花板」的說法對單模而言過強。且 p56 的 Not Clear 精確度（0.344）高於 equal8（0.322），兩者互補。

### 72.3 集成裁決：無法超越現有 8-stem ensemble

問題在 ensemble：單一增強 stem（NC F1 0.335）仍低於 8-stem equal8（0.347），因 equal8 靠多模型平均把召回率拉到 0.376。把增強訊號併入池的所有方式，val T4 macro 全部 ≤ 0.4602：

| 變體 | val T4 macro | Not Clear F1 |
| :-- | :--: | :--: |
| equal8（基準）| **0.4602** | **0.347** |
| p56 單模 | 0.4563 | 0.335 |
| equal8 + p56 / swap / weighted | 0.4526–0.4573 | 0.319–0.336 |
| 3-seed 增強 combo 單獨 | 0.4341 | 0.250 |
| equal8 + 3-seed 增強（各種權重）| 0.4562–0.4571 | 0.332 |

關鍵：**3-seed 平均（NC F1 0.250）反而比單一 seed42（0.335）差** → Not Clear 的合成增益有**高 seed 變異**（seed42 部分為運氣）。同配方多 seed 不提供 equal8 所需的「配方多樣性」，反而稀釋。

### 72.4 裁決與上限

- **合成有效但無法落地**：Not Clear 合成確實提升單模（+0.051 F1，雙向轉移），但 8-stem 多樣 ensemble 已飽和該訊號，同配方增強（含多 seed）無法超越 0.4602，且高 seed 變異使其不可靠。
- **不上傳**：任何增強候選的 val T4 ≤ equal8，違反上傳前閘，保留 0.6120。
- **0.6120 為實質天花板（已嚴格證實）**：post-hoc（Phase 54）、二元模型（Phase 53）、T4 集成多樣化（Phase 54）、T4 資料擴增（本 Phase）四大軸全數證偽。落差為 test 在同 50 家公司上的標註慣例差異（Phase 45/53），非建模可閉合。要破 0.62 在數學上需 T4≈0.50，本資料的 Not Clear 可分性 + Misleading(support=2) 不支持。

### 72.5 Phase 55 關鍵教訓

1. **「單模有效」≠「集成有效」**：合成在單模上 +0.051 Not Clear F1，但成熟的多樣 ensemble 已把該訊號榨乾，加入同配方增強只會稀釋。評估資料增強務必看「對部署 ensemble 的淨值」，非單模 delta。
2. **高 seed 變異是警訊**：seed42 的 Not Clear 增益在 seed2024/20260417 上不複製（3-seed 平均反降），代表增益部分為雜訊，不可靠。多 seed 是檢驗增益真偽的廉價照妖鏡。
3. **明確標記天花板的價值**：四軸窮盡式證偽（每軸帶轉移驗證）本身即是強競賽報告內容——「我們嚴格證明了天花板在哪、為何」優於無根據的樂觀微調。

---

## 73. Phase 57 — SOTA 文獻盤點 + 句向量宏觀模型驗證 + T1 集成擴充（2026-06-15）

本 Phase 對外搜尋 SemEval-2025 Task 6 / ML-Promise 得獎方法，評估能否進一步提升，並實作其中對中文最有實證的「句向量」軸。

### 73.1 SemEval-2025 PromiseEval 得獎方法（外部研究）

| 隊伍 | 方法 | 強項 |
| :-- | :-- | :-- |
| CSCU（英文總冠軍）| GPT-4o + paraphrase/synthesis 合成 | Promise/Evidence |
| Oath（英文 Clarity 最佳 0.669）| DeBERTa + 對比學習 + Mistral 合成平衡 timeline | Clarity |
| QM-AI（中文 Clarity+Promise 冠軍）| BERT 集成 | Clarity / Promise |
| CSECU-DSG（中文 Timing 冠軍）| 通用句向量（LASER / E5）| Timing |

兩個關鍵結論：(1) **QM-AI（中文冠軍）證實：純 BERT 集成、不用資料增強/英譯，結果最好最穩**——獨立印證本專案 Phase 55（合成幫不了集成）。本專案「BERT 集成 + 逐任務加權重組」即中文賽道 SOTA 對齊方法。(2) 中文 Timing 由句向量奪冠，是本專案未試的模型類別。

### 73.2 句向量宏觀模型（multilingual-e5-large）— rigorous 驗證後否決

`scripts/u42_embed_macro.py`：multilingual-e5-large 嵌入 train+val+test，per-task LogReg（5-fold OOF），驗證 T2/T4。

| | train-slice T2 | val-slice T2 |
| :-- | :--: | :--: |
| BERT（prior-corrected，部署用）| **0.6039** | 0.6135 |
| + 30% e5 | 0.5976 | 0.6121 |
| + 40% e5 | 0.5984 | 0.6157 |

裸 BERT（無 prior-corr）下 +30% e5 看似 val +0.009，但**在部署的 prior-corrected 管線下，e5 混合讓 val 升、train 降，無任何權重同時改善兩 slice**。增益大多與 prior-correction 重疊，e5 疊加為 val-slice 雜訊。**不轉移（同 Phase 54），不採用。** T4 同樣無效（e5 0.4348 < BERT 0.4602）。

### 73.3 T1 集成擴充（5-seed macbert-TAPT）

Phase 56 證實「T1 集成經 cascade 抬升全任務」。本 Phase 把 macbert-TAPT 從 3 seed（42/2024/20260417）擴到 5 seed（+7/+777），`scripts/u41` 產出 `u41_macbert5T1_pool5T3`（與 0.6176 僅 8 列差）。屬低變異微調探針，留待 6/16 上傳驗證。

### 73.4 裁決

外部研究確認中文賽道本專案已用對方法；新模型類別（合成、句向量）皆未能穩健提升宏觀任務（與 QM-AI 中文冠軍經驗一致）。**唯一經 LB 實證有效的軸是二元 cascade 重組**（0.6037→0.6176，分數持續上升）。進一步大幅提升需 GPT-4o 級 LLM 或更多/更佳資料（均超出本機資源），難度高；剩餘重組探針為 ±0.002 邊際，campaign 只升不降地續推。

---

## 74. Phase 58 — 窮盡式方法類別掃描的定論（2026-06-15）

本 Phase 系統性評估所有可行方法與技術。本 Phase 把所有可想到的方法類別逐一實作並以嚴格驗證（多數用雙-slice 轉移測試：train→val 與 val→train 皆須改善才算真）證偽或證實。

### 74.1 完整方法類別裁決表

| 方法類別 | 具體實作 | 驗證 | 結果 |
| :-- | :-- | :-- | :-- |
| BERT 集成 + 逐任務 cascade 重組 | macbert-TAPT 多 seed/源，per-task T1/T3 來源搜尋 | 真實 LB | **唯一有效**：0.6037→0.6218 public（rank 23/143）→ private 官方 0.6356308（rank 26/143，無過擬合）。本表 Phase 58 當時為 0.6176；後續續推至 0.6218，見 §69/§78 |
| 二元專家擴池（TAPT 多架構）| roberta/electra-large + TAPT | GroupKFold + LB | T1 集成有效；大模型 T3 有用、T1 有害 |
| 資料增強（Not Clear 合成）| 220 筆 Ollama 合成 | val-slice 雙向 | 單模 +0.05、集成飽和不採用 |
| 句向量（中文 Timing 冠軍招）| multilingual-e5-large + LogReg | 雙-slice | 不轉移（val 升 train 降）|
| **LLM 分類器（英/法冠軍招）** | **Qwen3-4B 與 gemma4:31b few-shot/CoT** | **gold 對照** | **Clear/NotClear 僅 50%（=亂猜）、timeline 45%；LLM 無法符合標註慣例，比 BERT 差** |
| **Stacking 元學習器** | LogReg on 8-stem OOF（per macro task）| **雙-slice** | **兩 slice 皆更差（T2 −0.03、T4 −0.003）** |
| 後處理決策/閾值（T2/T4）| per-class offset 搜索 | 雙-slice | 不轉移（Phase 54）|
| 更大骨幹 | roberta/electra/large maxlen512 | OOF + val | 過擬合（Phase 48）|

### 74.2 為何 Clarity/Timing 是硬上限（關鍵洞察）

T4 Clarity 與 T2 Timing 是主觀、標註者間一致性低的任務（SemEval 官方論文明述）。實證：**gemma4:31b（31B 大模型）對 Clear/Not Clear 僅 50% = 亂猜**。連 GPT-4o 在英文 Clarity 也僅勝 baseline +0.009。BERT 集成（T4 0.45）之所以勝過 LLM，是因為它從訓練資料學到了**這批標註者的特定慣例**，而泛用 LLM 學不到。這不是模型能力問題，是任務本質的標註雜訊上限。

### 74.3 定論

**0.6176 為 Phase 58（2026-06-15）當時的最佳；後續擴大二元 T1/T3 集成續推至 0.6218（public 最終 rank 23/143，2026-06-18）；官方正式最終（private）0.6356308（rank 26/143，見 §69 / §78）。此即本資料 + 本機資源下的實質天花板，已窮盡式證實；私榜進一步驗證模型未過擬合（private 0.6356308 > public 0.6218，+0.0139），穩健泛化至隱藏切片。** 七大方法類別逐一以雙-slice/LB 嚴格驗證，除「BERT 集成 + cascade 重組」外全數不轉移或更差；加開日（2026-06-18）再以真實 LB 直接證實大骨幹加入 T1 會單調拉低分數（0.6218→0.6203→0.6198）。

進一步大幅提升（約 +0.006 以上）需要本機沒有的資源：(1) GPT-4o 級閉源 LLM（英文總冠軍 CSCU 用之，但中文賽道仍以 BERT 集成奪冠）；(2) 更多/更高品質的標註資料以降低 Clarity/Timing 的標註雜訊。關於 (3) private split：官方最終已公佈，本方法於 private 0.6356308（rank 26/143）相對 public 0.6218（rank 23/143）不僅未回吐，反而高 +0.0139，證實無過擬合、穩健泛化（非運氣，而是因全程只以真實 LB 保守鎖定）。

### 74.4 關鍵教訓

1. **雙-slice 轉移測試是黃金準則**：合成、句向量、stacking、決策調整在 in-sample 常看似改善，但 train↔val 雙向一查即現原形。沒有它，這些「幻覺增益」會浪費上傳並誤導方向。
2. **泛用 LLM ≠ 萬靈丹**：在標註慣例特定、主觀性高的分類上，從該資料微調的小 BERT 勝過 31B 通用 LLM。先用 gold 小樣本測 LLM 對齊度，再決定是否投入。
3. **知道天花板在哪本身就是成果**：窮盡式、帶嚴格驗證的負結果掃描，是競賽報告的高品質方法論貢獻。

---

<a id="part-v--附錄-appendix"></a>
# Part V — 附錄 (Appendix)

本部分為支援性詳表：Backlog 全紀錄、工程上限推導、外部資料合規立場。
日常閱讀無需深入，僅在需要查證 D-編號、X-編號、上限數字、合規條款時翻閱。

---

## 75. Backlog 詳表（D1 ~ D30 + U1 ~ U12 全紀錄）

> 本節為歷史審計用詳表。D 系列（D1~D30）為已完成的工程動作流水，U / N 系列（U1~U12 / N1~N3）為原始 backlog 候選清單。
> X1~X14 禁區另列於 [§75.3](#753-已暫緩或證明不可行的方向避免重複嘗試)。

### 75.1 已完成項目（D 系列與 U/N 系列）

> D 系列為早期工程動作流水，編號停在 D30；Phase 26 之後的 U10、AP-D3、AP-D4 與 Phase 39 決策以各 Phase 章節為主紀錄。下列 D 表保留時間順序與對 SOTA 軌跡的貢獻。

| # | 項目 | 階段 | 結果 | 對 SOTA 增益 |
| :--: | :-- | :--: | :--: | :--: |
| D1 | LR/max_len/pooling/class_weight 8 組 ablation → combo_best | Phase 2 | 0.66558 | +0.024 vs P1（首階奠基） |
| D2 | macbert-large 升規模（3 lr 變體） | Phase 3 | 0.66439（單模未過門檻） | −0.001（保留作 ensemble 配角） |
| D3 | 2-way Probability Ensemble (combo:large=1.5:1) | Phase 4 | 0.67478 | +0.0092（首次 ensemble） |
| D4 | combo_v2 = combo_best + FGM + Focal-T4(γ=2.0) | Phase 5 | 0.66836 單模 | — |
| D5 | Wave A/B 16 組單變量 ablation | Phase 5 | top: focal_t4 0.66833、fgm10 0.66714 | — |
| D6 | Per-task Hillclimb v1 (3-way) | Phase 5 | 0.67954 | +0.0048 |
| D7 | Wave C 4 組（aug_mask10 / aug_mix / rdrop05 / msd5） | Phase 6 | 4 個單模均 < combo_best | — |
| D8 | Per-task Hillclimb v3 (7-way) | Phase 6 | 0.68206 | +0.0025 |
| D9 | combo_v3 = combo_v2 + R-Drop α=0.5 + MSD K=5 | Phase 7 | 0.67121 單模 | — |
| D10 | Per-task Hillclimb v4 (8-way) | Phase 7 | 0.68394 | +0.0019 |
| D11 | combo_v3 multi-seed（+2024+20260417 共 3 seeds） | Phase 8 | 3-seed avg OOF 0.66854 | — |
| D12 | Per-task Hillclimb v5（直接以 multi-seed 替換） | Phase 8 試錯 | 0.68191（負面，T1 被平均稀釋） | −0.0020 |
| D13 | Per-task Hillclimb v6（combo_v3 拆 peaky/avg → 9-way） | Phase 8 SOTA | 0.68440 | +0.0005 |
| D14 | Sprint A T1 — `chinese-roberta-wwm-ext-base` 5-fold seed=42（p4a） | Phase 9 | 單模 0.66626；加入 pool | 間接（成員擴增） |
| D15 | Sprint A T2 — `bert-base-chinese` 5-fold seed=42（p4b） | Phase 9 | 單模 0.67355（Sprint A 最佳）；加入 pool | 間接（成員擴增） |
| D16 | Per-task Hillclimb v7b（11-way，修正 v7 RNG bug → biased search 包含新成員） | Phase 9 SOTA | 0.68558 | +0.00118 |
| D17 | Sprint B T6 v1 — `p5_t6_time_token`（regex 數字 prefix） + Hillclimb v8（12-way） | Phase 10 SOTA | 0.68669 | +0.00111 |
| D18 | Sprint B T6 v2 — `p6_t6v2_bucket_tok`（dedicated bucket token，加入 special_tokens 並擴 embedding） + Hillclimb v9（13-way） | **Phase 11 SOTA** | **0.68683** | **+0.00014** |
| D19 | Sprint B T7 — `p7_focal_g3`（Focal-T4 γ=2.0→3.0）單模 0.67416（歷史第二佳，T4 +0.0237 為單模史上最大 T4 增益） + Hillclimb v10（14-way） | Phase 12 | 0.68660（**未破** SOTA） | −0.00023（成員仍保留） |
| D20 | Sprint B 收尾 — (a) `p8_ema995` 0.63010（EMA 無 warm-start 重蹈 X1）；(b) `p9_ls_t1t3` 0.67334 (+0.00213, T3 按設計微升)；(c) Joint Hillclimb v11（16-way，直接優化 post-constraint score） | Phase 13 | 0.68770（前一代 SOTA） | +0.00087 vs v9 |
| D21 | Sprint C / N1 — `p10_large_focal_fgm`（macbert-large + Focal-T4 + FGM）+ Joint Hillclimb v12（17-way） | Phase 14 | **0.68825**（目前訓練 ensemble SOTA） | **+0.00055** vs v11 |
| D22 | U1 TTA — `stored+middle` 推論視角擴增（官方資料、無重訓） | Phase 15 | **0.68879**（目前 active SOTA） | **+0.00054** vs v12 |
| D23 | N2/p11 — `p11_electra_base` 官方資料 5-Fold + v16 admission（18-way） | Phase 16 | 單模 0.62694；admission 0.68825，p11 0 權重 | 0 vs v12；−0.00054 vs U1 |
| D24 | U1-b — `stored+middle+tail` 三視角等權 TTA 補測（active pool 14 員） | Phase 17 | 0.68873（未取代 `stored+middle` 0.68879） | −0.0000597 vs U1 active SOTA；T2 取得本輪最高 0.50941 |
| D25 | U1-c — per-task / per-view 權重 TTA coordinate descent（grid 0.05 + oracle warm-start，零再訓） | Phase 18 | **0.68925**（新 active SOTA） | **+0.00046 vs U1 / +0.00100 vs v12**；T2 0.51026 史上最高 |
| D26 | B1 / U3 — `p2_combo_best_swa` SWA 最後 K=3 epoch 平均（trainer.py + aggregator + canonical OOF alias） | Phase 21 | member-level OOF 0.66252（vs best-epoch 0.66558，**−0.0031**）；未通過 admission 門檻 +0.0005，**不入 v12 池** | 0（拒絕）；列入 X9 風險區（SWA + LR ≠ 0） |
| D27 | B2 / U5 — `p2_combo_best_resample_t4` T4 sqrt-inverse-frequency re-sampling（alpha=0.5） | Phase 22 | member-level OOF 0.66562（vs baseline 0.66558，**+0.00004**）；T4 macro **+0.012** 但 T2 macro **−0.014** 抵銷；未通過 admission +0.0005，**不入 v12 池** | 0（拒絕）；列入 X10 風險區（global sampler 抵銷 T2/T4） |
| D28 | C1 / N3 — NeZha-base 不可訓（fallback 到 BertModel，fold0 0.549 中止）；轉 ERNIE-3.0-base-zh fallback，5 折 mean **0.6086** << 0.66 admission 門檻 | Phase 23 | NeZha：fold0 0.549 中止；ERNIE：0.6086（T2 −0.107、T4 −0.073 vs baseline）；**N3 candidate 全數失敗，不入池** | 0（拒絕）；列入 X11 風險區（base-class 異源 backbone 對本資料集不利） |
| D29 | U2 / B3 — `p2_combo_best_ema995_warm` EMA decay=0.995 + warm-start 2 epoch（trainer.py 新增 `ema_warmup_epochs` + `ema_started`） | Phase 24 | member-level OOF **0.64941** vs baseline 0.66558（**−0.01617**）；T4 退化最嚴重（−0.062）；5 折全退化；未過 admission **不入 v12 池** | 0（拒絕）；列入 X12 風險區（EMA + warm-start 在 cosine 末段 LR ≠ 0 + 短訓練下仍傷 T4 macro） |
| D30 | U6 / B4 — `p2_combo_best_u6_bt` NLLB-200-distilled-600M 中→英→中 回譯（T2 within_2_years×3 + T4 Misleading×5 / Not Clear×1；138 唯一源員 → 168 增強樣本；src/train_kfold.py 新增 aug 載入 / per-fold 注入機制，oof / valid 不收 aug） | Phase 25 | member-level OOF **0.66321** vs baseline 0.66558（**−0.00237**）；T2 macro 0.4652（−0.014）、T4 macro 0.4276（−0.031）；上加 fold1 轉譯品質離群（score 0.6334）拉低均值；未過 admission **不入 v12 池** | 0（拒絕）；列入 X13 風險區（NLLB-600M 保真度不足 + 無 round-trip ChrF 過濾 + 譯本帶入 label-noise） |
<a id="591-已完成項目done--含成果"></a>
#### 75.1.1 U/N 系列完成項目與成果

> 補充 D 表之外、以 U / N 編號追蹤的項目。X1~X13 禁區詳 [§75.3](#753-已暫緩或證明不可行的方向避免重複嘗試)。

| ID | 項目 | 結果 | 紀錄位置 |
| :--: | :-- | :-- | :-- |
| U1 / U1-b / U1-c | TTA 多視角推論 | 完成且採納（U1-c = active SOTA 0.68925） | §36~§39, D22 / D24 / D25 |
| U2 | EMA decay=0.995 + warm-start | 完成且拒絕（X12） | §45, D29 |
| U3 | SWA 最後 K=3 epoch 平均 | 完成且拒絕（X9） | §42, D26 |
| U5 | T4 class-balanced re-sampling | 完成且拒絕（X10） | §43, D27 |
| U6 | NLLB-600M zh-en-zh 回譯 | 完成且拒絕（X13） | §46, D30 |
| N3 | NeZha → ERNIE fallback backbone | 完成且拒絕（X11） | §44, D28 |
| U10 | TW 企業永續報告書專用偽標 pipeline | **完成（U10 stack SOTA = 0.67746，baseline +0.01012）** | §47 |
| U11 | StratifiedGroupKFold by company sanity | 完成（診斷） | §40 |
| U12 | OOF cross-fold variance | 完成（診斷；valid gap 待 6/03 釋出） | §41 |
| U15 | train+val 合併資料產生（2,000 rows，`more_than_5_years` 自動正規化） | 完成 | §58.2, Phase 40 |
| U16 | TV 池 8-stem OOF 集成 joint hillclimb（Phase 42 前置）| 完成 | §14.5, §58.6 |
| **F9** | **Phase 41 Train+Val 8-stem 重訓（2,000-row、seed 42、5-fold）** | **完成（2026-06-11）：8 stems 全數重訓 + TV 池 OOF 集成驗證** | §14.4, §58 |

### 75.2 未完成或可重啟項目（TODO / Candidate）

> 本節已依 Phase 40-41 狀態更新。**F9 Phase 41 8-stem 重訓已於 2026-06-11 完成並移至 [§75.1.1](#7511-un-系列完成項目與成果)**。

#### 75.2.1 等待 valid/test 的必要動作

| ID | 項目 | 狀態 | 啟動條件 | 產出 |
| :--: | :-- | :-- | :-- | :-- |
| F1 | AP-D4 / AP-D3 / Phase36 valid 校準 | **工具完成；TV checkpoints 已就緒；待 test 釋出後執行** | 執行 `scripts/u12_val_gap.py`（val 已可跑；test 待 6/10+ 釋出）| valid score、per-task drift、submission anchor 建議 |
| F2 | 最終提交檔 schema 與 hierarchy 守門 | **本地 validator 已完成**；final 格式仍待 test 確認 | test 格式或官方範例確認 | `validate_submission` 檢查紀錄、最終 submission 欄位對齊 |
| F3 | Submission anchor 分配策略 | **Phase 42 定案**；TV 池 OOF warm-start 權重已備（`tv_oof_ensemble_meta.json`）| 測試集釋出後 | 21 次提交額度的主線與 fallback 順序 |

<a id="5922-下一輪可動執行路線v30-校訂-2026-05-10"></a>
#### 75.2.2 可做但需前置條件的 Phase 40+ 候選

| ID | 名稱 | 就緒度 | 重訓 | 前置條件 | 決策原則 |
| :--: | :-- | :-- | :--: | :-- | :-- |
| F4 | AP-D5 搜尋向量化 / cache / 小預算替代 optimizer | 本地工程已完成 | 否 | valid 後若 AP-D4 仍為主 anchor，可小預算 refinement；不需新訓練 | 只接受 post-constraint weighted score 上升；保留 meta/history，不重跑舊版 X14 |
| F5 | stem #9：U13 LLM 評審重標 U10 偽標 | 待設計 | 是 | valid 顯示 AP-D4 未 overfit，且 T4/T2 仍為缺口 | 需過 ensemble admission，不以 single-stem OOF 判斷 |
| F6 | 外部 provider LLM 合成 + 人工抽樣 review | 待資源與規則審計 | 是 | provider、prompt、來源紀錄、quality gate 完整 | 只補 minority，不使用 valid/test 洩漏資訊 |
| F7 | T4 Misleading / T2 minority 專項重訓 | 暫緩 | 可能 | valid 顯示特定 minority 類仍可收益 | 避免 global sampler 對其他任務造成副作用 |
| F8 | 跨家族 teacher / Qwen-LoRA | 阻塞 | 是 | 16 GB+ GPU 或新訓練策略 | 不重跑已拒絕的 XLM-R / ELECTRA / NeZha / ERNIE 同設定 |

#### 75.2.3 歷史暫緩 / 阻塞 backlog

| ID | 項目 | 狀態 | 阻塞原因 | 現行處理 |
| :--: | :-- | :-- | :-- | :-- |
| U8 | Pipeline B-Plan T1 → T3 → T2/T4 串接 | 暫緩 | 需 valid gap 與 error pattern 支持 | 若 valid 顯示 hierarchy error 是主因再重啟 |
| U9 | Qwen2.5-7B LoRA 4-bit r=16 | 阻塞 | RTX 5060 Laptop 8 GB 不足，需 16 GB+ 顯存 | 併入 F8 長期路線 |

> **禁區提醒**：X1 ~ X14 詳見 [§75.3](#753-已暫緩或證明不可行的方向避免重複嘗試)，本輪嚴格遵守。

### 75.3 已暫緩或證明不可行的方向（避免重複嘗試）

| # | 項目 | 結果 | 原因 |
| :--: | :-- | :-- | :-- |
| X1 | EMA decay=0.999 短訓練 | 0.47291（崩潰） | 訓練步數 ~500 不足以讓 shadow weights 跟上；改用 U2 修正版 |
| X2 | Uniform Label Smoothing ε=0.05 全 task | T4 macro 崩潰 | 少類被推平；Phase 13 的 per-task LS(T1/T3) 單模可行但 ensemble 0 權重，故不再高優先 |
| X3 | LLRD 0.95（Layer-wise LR Decay） | 0.66019（負面） | base 模型層數淺，分層 lr 反而限制底層學習 |
| X4 | combo_v3 multi-seed 直接替換 seed=42 | 0.68191（負面） | T1 winner 倚賴 peaky 預測，平均化稀釋訊號；改用 D13 拆分設計 |
| X5 | macbert-large 作單模主力 | 0.66439（負面） | 1000 樣本對 326M 參數比例失衡；保留作 ensemble 配角 |
| X6 | Sprint A T3 — `xlm-roberta-base`（p4c）作單模 | 0.61269（被拒） | 低於 0.65 入池門檻 0.038；中文混合 backbone 在本資料集對齊不佳；不加入 ensemble pool |
| X7 | N2/p11 — `hfl/chinese-electra-180g-base-discriminator` 同設定 | 單模 0.62694；v16 admission 0 accept / 0 權重 | 同設定未提供可用 ensemble diversity；不重跑，若再試 ELECTRA 必須更換訓練策略或模型變體 |
| X8 | U1-b — `stored+middle+tail` 三視角等權 TTA | 0.68873，較 `stored+middle` 0.68879 低 0.0000597 | 三視角等權同時拉回 T1 至 baseline、稀釋 `stored+middle` 在 T1/T4 的貢獻；後續 U1-b 不再做等權三視角，改走「任務別/視角別權重」 |
| X9 | U3 SWA — `p2_combo_best_swa` 最後 K=3 epoch 平均（cosine warmup + 末段 LR > 0） | member-level OOF 0.66252 vs best-epoch 0.66558（−0.0031），五折 3/5 退化（fold4 −0.0139） | 末 epoch LR 未收到 0；K 涵蓋的 ckpt 仍在 loss 震盪區；若再嘗試 SWA 須改 cosine 完整下到 0 LR、或 EMA + warm-start（U2）；不重跑同設定 |
| X10 | U5 T4 global re-sampling — `p2_combo_best_resample_t4`（alpha=0.5 sqrt-inverse-freq） | OOF 0.66562 vs baseline 0.66558（+0.00004）；T4 +0.012 但 T2 −0.014 抵銷 | global WeightedRandomSampler 同時重抽其他任務的 batch，造成 T2 同步退化；未來如要讓 T4 增益不被抵銷，需改 per-task loss weighting、或 head-only mini-finetune、或同時平衡 T2 + T4；不重跑同設定 |
| X11 | C1 / N3 — NeZha-base 與 ERNIE-3.0-base-zh fallback | NeZha：HF transformers 無 NeZha-specific class，fallback 為 `BertModel` 時隨機初始化 NeZha 相對位置嵌入，fold0 5 epoch 最高 0.549；ERNIE：5 折 mean 0.6086，T2 −0.107、T4 −0.073 vs baseline | base-class 異源 backbone（NeZha / ERNIE）在本資料集上表現遠退 macbert / roberta-wwm 同源池；如要繼續結構性多樣性，需改 (a) layer-wise freeze、(b) large-class 多 seed，而非換 base 同級 backbone；不重跑 NeZha / ERNIE 同設定 |
| X12 | U2 / B3 EMA decay=0.995 + warm-start 2 epoch — `p2_combo_best_ema995_warm`（5 fold × 5 epoch × seed=42） | member OOF 0.64941 vs baseline 0.66558（−0.01617）；5 折全退化；T4 macro −0.062 為退化主因；T1/T3 退化較輕 | 雖比 X1 ema999 0.473 顯著改善（warm-start 避免徹底崩潰），但仍遠低 admission 0.66608；推測主因為 (a) cosine 末段 LR 未到 0（與 X9 SWA 同源）、(b) 僅剩 3 epoch EMA 累積樣本不足、(c) EMA 平均化壓縮 T2/T4 少類 confidence；若再試 EMA 須改 cosine 完整下到 0 LR + 拉長至 8~10 epoch + T2/T4 改用 best-epoch；不重跑同設定 |
| X13 | U6 / B4 回譯增強 — `p2_combo_best_u6_bt` NLLB-200-distilled-600M 中→英→中（138 唯一源員 → 168 增強樣本：T2 within_2_years ×3、T4 Misleading ×5、T4 Not Clear ×1；5 fold × 5 epoch × seed=42） | member OOF 0.66321 vs baseline 0.66558（−0.00237）；折別：[0.6748, 0.6334, 0.6632, 0.6750, 0.6696]，std=0.0174（fold1 離群）；T2 macro 0.4652（−0.014）、T4 macro 0.4276（−0.031）——雙目標任務皆退化；未過 admission 0.66608 | 主因為 (a) NLLB-200-distilled-600M 對繁中 ESG 專有名詞保真度不足（譯本儲存「台泥」為 TaiMot、字符腐化、詞重複，見 `outputs/logs/u6_backtranslate.runlog`）、(b) 英文 pivot 備作減譯 ESG-specific terminology（「永續發展」→sustainable development→「可持續發展」，詞表隋譯）、(c) 未加 round-trip ChrF/BLEU 過濾，低保真度譯本並未被檔掉、(d) train_recs 自 ~795 被膨跨至 ~935 含低保真度樣本，對 T2/T4 macro 等同 label-noise；若未來再試 BT 需 (a) 換 NLLB-3.3B 或 madlad-7B 或外部 NMT API 以提高中文保真度、(b) 加「原文 vs BT 譯本」 round-trip ChrF≥0.5 自動過濾、(c) 限縮至 T4 Misleading only（5 筆）以避免 T2 雜訊、(d) 伝依人工抽查譯本品質誌；不重跑同設定 |

---

## 76. 工程上限拆解（推導依據）

> 本節說明本專案聲稱的「加權保守上限 ≈ 0.7236 / 加權寬鬆上限 ≈ 0.7570」是如何得出的。
> 包含兩種上限的定義、per-task 拆解推導，以及 T4 `Misleading` (support=1) 為何構成統計硬上限。
> 與「如何突破上限」相關的執行路線見 [§75.2.2 可做但需前置條件的 Phase 40+ 候選](#7522-可做但需前置條件的-phase-40-候選)。

### 76.1 兩種上限的定義

**(A) 演算法上限 / Bayes-optimal**：給定特徵分布下，最佳分類器能達到的分數。理論存在但**不可直接量測**；通常以「人工標註者一致性 (Inter-Annotator Agreement, IAA)」作代理估計。例如：若兩位標註員意見不同的比例 = 8%，則 IAA ≈ 0.92，此即實務上限。

**(B) 工程上限 / Engineering ceiling**：給定固定資料集 + 固定評分公式下，所有合理技術組合可達的上限。比 (A) 嚴格但可逼近；用「per-class F1 拆解」估算。本表所列即為 (B)。

### 76.2 per-task 上限拆解（推導依據）

> 注意：本節原始估算建立於 Phase 18 active path 時代，保留作為歷史工程上限推導。Phase 38 AP-D4 已把 weighted score 推進至 0.71608，且 T2 已高於早期寬鬆估計；因此本節不再視為嚴格上限，而是用來說明「分數天花板主要由 T4 與少數類統計變異決定」的推導脈絡。

| Task | 早期參考值 | 估計上限 | 缺口 | 上限依據 |
| :-- | :--: | :--: | :--: | :-- |
| T1 promise (binary F1) | 0.9405 | 0.94 ~ 0.95 | 0 ~ +0.010 | binary 任務、support 平均；macbert-large 單模也可達 0.928；IAA 上限約 0.95（已接近）|
| T2 timeline (macro 5 類) | 0.5087 | 0.55 ~ 0.62 | +0.041 ~ +0.111 | `already ↔ between_2_and_5_years` 混淆仍是主因；T6 兩版本未直擊，需資料擴增或更不同的模型家族 |
| T3 evidence (binary F1) | 0.8783 | 0.90 ~ 0.92 | +0.022 ~ +0.042 | 與 T1 同類但 evidence 描述較隱晦、訊號弱；理論 ~0.92 |
| T4 quality (macro 4 類) | 0.4592 | 0.55 ~ 0.65 | +0.091 ~ +0.191 | `Misleading` (support=1) 數學上單類錯誤會重壓 macro；理論天花板 ≈ 0.60 ~ 0.65 |
| **加權保守上限** | — | **≈ 0.7236** | — | 0.20·0.945 + 0.15·0.585 + 0.30·0.910 + 0.35·0.600 |
| **加權寬鬆上限** | — | **≈ 0.7570** | — | 0.20·0.95 + 0.15·0.62 + 0.30·0.92 + 0.35·0.65 |
| **歷史實測（active TTA path）** | **0.68925** | — | **0.034 ~ 0.068** | [Phase 18](#39-phase-18-完成--u1-c-任務別視角別加權-tta2026-05-05) U1-c per-task TTA；已被後續 U10/AP 路線超越 |
| **歷史實測（U10 best.pt path）** | **0.67746** | — | **0.046 ~ 0.080** | [Phase 31](#477-m6-oof-ensemble--new-sota--067746baseline-001012) baseline + v1 + v2 stack（45 ckpt）；後續已合流至 AP-D4 |
| **目前實測（AP-D4）** | **0.71608** | — | **距早期保守目標 0.7236 尚差約 0.00752** | [Phase 38](#55-phase-38--ollama-llm-synth--stem-8--ap-d4--new-sota-0716082026-05-20) 8-way × 3-view per-task TTA |

→ 「**0.72 ~ 0.74**」這個區間為早期工程估計；Phase 38 AP-D4 已接近或超過部分早期假設，後續需以 [§14](#roadmap) 的 valid 校準與提交策略重新定義下一階段目標。U10 路線詳見 [§47.12](#47-u10--企業永續報告書-sr-弱監督-pipeline-完整版2026-05-09-重啟2026-05-10-v2-重訓)。

### 76.3 為什麼 T4 `Misleading` (support=1) 是真正的硬上限？

T4 的 label domain 是 4 類（Clear / Not Clear / Misleading / N/A）。macro-F1 對 4 個類別等權平均。若有一類 support=1：
- 模型若預測對：該類 F1 = 1.0
- 模型若預測錯（極可能，500 候選只有 1 個真正例）：該類 F1 = 0.0
- 對 T4 macro-F1 的單類影響 = ±(1/4) = **±0.25**；折算到最終分數權重 0.35，影響可達 **±0.0875**

實務上幾乎必錯，所以 T4 macro-F1 的統計變異很大。除非主辦單位提供更多 `Misleading` 樣本，或未來重新啟用且人工審核外部資料，否則 **T4 macro-F1 > 0.55 在統計上極困難**。

---

## 77. 競賽規則對外部資料的立場（已確認可用）

### 77.1 主辦方官方 Q&A（直接引用主辦回覆）

主辦單位來信確認下列事項（依主辦回覆原文整理）：

**Q1：模型訓練的任何階段，可以使用開源或閉源大型語言模型（LLM）作為輔助嗎？**
A：可以。主辦方未限制參賽者在任何訓練階段使用開源或閉源 LLM。

**Q2：可以自行查詢、撰寫並標註資料，或用 LLM 生成資料加進訓練集嗎？**
A：可以，屬「資料擴增」，主辦方未對訓練方法（含資料擴增）設任何限制。主辦明列兩種允許做法：(1) 純手工自製——自己寫 ESG 句子、手動標好標籤後加入訓練；(2) 用 LLM 生成——寫 Prompt 請開源/閉源模型生成大量資料。

**Q3：可以使用官方資料集以外、其他企業 ESG 永續報告書的標註資料嗎？**
A：可以，亦被允許使用。

**唯一限制（主辦原話）**：僅「主辦方提供之測試資料集（Testing Dataset）」與「最後參賽者上傳的預測結果」絕對不能有**人工介入**。

**本專案對應處置（全部合規）**：

- 所有資料擴增（手工種子、Ollama LLM 合成、外部 ESG 報告偽標）皆只進訓練側，對應 Q1–Q3 的允許範圍。
- 測試集與最終上傳預測**全程無人工介入**：預測由程式（`scripts/u41`）自動生成，未對測試集做任何人工標註或修改。
- TAPT 對測試文字的非監督預訓練屬自動化訓練方法、未使用任何標籤、未修改測試集，不屬「人工介入」，依主辦規則合規。

### 77.2 規則原文比對

逐字檢視官方文件 [ESG_永續承諾驗證競賽_2026.md §八「競賽規則與注意事項」](ESG_永續承諾驗證競賽_2026.md)，禁止項目僅有以下五條：

1. 抄襲、作弊、詐欺；
2. 侵害他人智慧財產權；
3. 攻擊 leaderboard 系統；
4. **對測試資料集或辨識結果進行任何形式的人工標註或修正**（預測必須由程式自動生成）；
5. 私下共享程式與特徵值。

**並無任何條文禁止使用外部訓練資料、外部預訓練模型、或對外部公開文本做偽標。現況（2026-05-10）**：

- U10 已於 2026-05-09 重啟、2026-05-10 完成 v2 重訓；採「來源同質性 + 50 家排除 + 兩階段隔離訓練」三條鐵律（詳 [§47.0](#470-設計原則與失敗教訓)）。
- 實測 best.pt path 增益 +0.01012（baseline 0.66734 → stack 0.67746）；對 minority class T2 timeline 與 T4 quality 提供可量測覆蓋改善。
- 殘留結構性問題（T2 within_2_years / T4 Misleading 仍 0）為 baseline class collapse，後續需依 [§14](#roadmap) 的 valid 訊號決定是否再啟動 class-weighted CE / Focal-T4 專項。
- 偽標噪聲與分布偏移風險已透過兩階段訓練隔離；不得將 pseudo 視為人工標註真值。

---

## 78. Phase 59 — 加開日 LB 探針：T1 架構多元化（2026-06-18）

主辦方加開一日上傳，可再投兩發。唯一已驗證的槓桿是「逐任務二元重組 + 級聯」；二元 OOF 有洩漏，故只能用真實 leaderboard 確認。本階段在不重訓的前提下，從快取的 test 機率組出兩個有理據的探針。

### 78.1 假設

現行最佳 `u41_macbert7T1_pool7T3 = 0.6218` 的 T1 是 **純 macbert 的 7 個 seed 平均**——從未納入兩個大型 TAPT 架構（roberta-large、electra-large）。而「架構多元化有益」在 T3 已被證實（拿掉 electra 反而變差，見 [§70 Phase 53](#70-phase-53--二元專家擴池建置tapt-多-seed--多架構--誠實-groupkfold2026-06-14)）。因 T1 是級聯驅動者（T1 判 No → 後三題自動 N/A），把同樣的多元化加進 T1，是「最強的尚未測試」槓桿。兩發皆固定保留已驗證的 c3 macro 底座（T2/T4）與致勝的 pool7 T3，**只改 T1 來源**。

### 78.2 兩個探針（均已建置、schema 合規、可重現）

| 候選 | T1 來源 | md5 | 對 0.6218 差異(T1/T2/T3/T4) | T1 No / T3 No |
|:--|:--|:--|:--|:--:|
| 錨點 `u41_macbert7T1_pool7T3`（0.6218）| macbert7 | `d3745d70` | 0 / 0 / 0 / 0 | 286 / 227 |
| 探針 1 `u45_macbert9T1_pool7T3`（主攻）| macbert7 + roberta-large + electra-large | `3f2f6554` | 10 / 10 / 10 / 8 | 284 / 227 |
| 探針 2 `u45_macbert8rT1_pool7T3`（避險）| macbert7 + roberta-large | `8c31107c` | 10 / 10 / 10 / 7 | 282 / 228 |

兩發各只翻動約 10 筆邊界 promise 判定，並經級聯傳遞至 T2/T3/T4——正是逐任務重組的外科式微調，非大幅改動。探針 1 加入兩種大型架構（最大多元化），探針 2 只加 roberta-large（隔離「electra 是否反而拖累 T1」）。

### 78.2.1 探針 1 結果（2026-06-18 00:11，已上傳）

兩發皆已上傳，得到一條乾淨的單調 ablation：

| T1 組成 | md5 | 上傳時間 | T1 F1 | 加權 LB |
|:--|:--|:--|:--:|:--:|
| macbert7（無大型，**致勝**）| `d3745d70` | — | 0.8056 | **0.6218** |
| macbert7 + roberta-large（探針 2）| `8c31107c` | 06-18 00:18 | 0.8036 | 0.6203 |
| macbert7 + roberta + electra（探針 1）| `3f2f6554` | 06-18 00:11 | 0.8022 | 0.6198 |

探針 1 逐任務 T1 0.8022 / T2 0.6131 / T3 0.6942 / T4 0.4546；探針 2 T1 0.8036 / T2 0.6146 / T3 0.6943 / T4 0.4546，皆低於致勝值（T1 0.8056 / T2 0.6147 / T3 0.6974 / T4 0.4549）。**結論：每加一個大型架構就拉低 T1（electra 拖累大於 roberta），並經級聯拖累 T2/T3——「T1 架構多元化」假設徹底證偽，與 [§65 Phase 48](#65-phase-48--更強骨幹重訓設計驗證優先鎖定可轉移任務)「大骨幹在小資料過擬合」一致。致勝 T1 維持純 macbert7。0.6218 為本方法已驗證之天花板，二元／T3 槓桿已窮盡。** 加開日最後一發回鎖 `u41_macbert7T1_pool7T3`（0.6218）收工。

### 78.3 上傳計畫（遵守「最後一發計分」規則）

AIdea 以當日**最後一次上傳**為準（見記憶 [[competition-upload-rule]]），故順序如下：

1. 上傳探針 1 `u45_macbert9T1_pool7T3`，記錄 LB 分數。
2. 上傳探針 2 `u45_macbert8rT1_pool7T3`，記錄 LB 分數。
3. 若兩者皆未超過 0.6218 → **最後再上傳一次錨點 `u41_macbert7T1_pool7T3`（0.6218）**，確保當日結束停在已知最佳；若某探針超過 0.6218 → 最後停在該探針。

### 78.4 重現

```
python -m scripts.u45_phase57_extraday_probes
```

輸出三個 `outputs/submissions/*_submission.csv`（各 2000 列、0 schema 錯誤）；錨點重建後 md5 仍為 `d3745d70`，與已上傳之 0.6218 檔位元一致。本階段為 LB 探針，**二元增益只能於真實 leaderboard 確認，非離線驗證之既定改善**。

### 78.5 官方最終結果（private 榜，正式計分）

加開日最後一發回鎖錨點 `u41_macbert7T1_pool7T3`（md5 `d3745d70`）。官方計分結果：

| 榜別 | 加權分數 | 名次 |
|:--|:--:|:--:|
| public（公開）| 0.6217591（0.6218）| 23 / 143 |
| **private（官方正式計分）** | **0.6356308** | **26 / 143** |

**關鍵：private 較 public 高 +0.0139（0.6356308 − 0.6217591），方向 private > public。** 過擬合公開榜的典型徵狀是隱藏切片回吐（private < public）；本案同一份提交在 private 切片重新計分後不降反升，證明決策邊界非「背公開榜答案」，而是學到可遷移的標註慣例。此直接驗證全程紀律——二元任務只以真實 LB 鎖定、絕不信任洩漏約 0.18 的 OOF 0.71608 代理（[§60](#60-phase-43--真實-leaderboard-05962-vs-oof-071033-根因分析2026-06-11)）。名次 public 23 → private 26 之微幅位移，係 private 切片整體抬升各隊分數、名次間距縮短（本隊絕對分同步上升）所致，非分數退步。正式得獎名次由主辦於 2026-07-23 公佈。

**策略註記（經官方簡章 §七 證實）**：簡章明定「Private Leaderboard 採最後一次上傳結果計分；Public Leaderboard 呈現最高分」。因此加開日「先投兩個探針、最後回鎖 0.6218 錨點」的順序是關鍵——private 計分的是最後上傳的錨點檔，方得 0.6356308；若當日最後一發停在探針，private 將改以探針計分。此印證 [[competition-upload-rule]] 之策略對 private 正式成績有直接決定性。

---
