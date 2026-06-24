<!-- markdownlint-disable MD025 MD031 MD032 MD033 MD036 MD040 MD041 MD058 MD060 -->

# REPRODUCE.md — 重現手冊：競賽 public 最佳 0.6218 與 OOF 基線 0.71608

> **目標**：從一張裸機（已裝 Python + CUDA 驅動）clone 此 repo 後，能重現兩件事：(1) 競賽真實 public leaderboard 最佳 **0.6218**（rank 23/143；同一提交檔之 private 官方計分為 0.6356308、rank 26/143）；(2) 作為基底的 OOF 集成 **0.71608**（交叉驗證代理分數）。
>
> 本檔為說明書。官方資料、LLM 合成種子、腳本與設定都在本 repo 內；大型 checkpoint 與可重生的 pseudo/intermediate data 不隨 GitHub 發佈，依本檔指令重生。
>
> 分數性質（重要）：**0.71608 是 5-fold 交叉驗證的 OOF 代理分數，不等於競賽 leaderboard**；它對二元任務系統性樂觀。競賽真實最佳為 **0.6218**（逐任務重組所得）；OOF↔LB 落差根因見 [MASTER_PLAN §60–§68](MASTER_PLAN_AND_PROGRESS.md#60-phase-43--真實-leaderboard-05962-vs-oof-071033-根因分析2026-06-11)。
>
> **隊伍**：TEAM_10057

---

## 0. 一句話重現

### 0.1 競賽 public 最佳 0.6218（最終提交）

= MacBERT 基底集成（macro 任務）+ 多 seed / 多架構 TAPT 二元專家集成（T1/T3）+ 階層級聯逐任務重組。

| 步驟 | 指令 |
| :-- | :-- |
| 一鍵流程（TAPT → 訓練二元專家 → GroupKFold 排名 → 產候選）| `./scripts/u37_run_binary_pool.ps1` |
| 重組產出最終候選 | `python -m scripts.u41_binary_recombine`（輸出 `u41_macbert7T1_pool7T3_submission.csv` = 0.6218）|

完整方法與分數軌跡見 [MASTER_PLAN §69–§74](MASTER_PLAN_AND_PROGRESS.md)。

### 0.2 OOF 基線 0.71608（交叉驗證代理，非 leaderboard）

= 8 stems × seed42 × 5-fold OOF probs → 3-view TTA（stored / middle / tail）→ 4-task per-task simplex hillclimb。

| 來源 | 指令（執行順序見 §4） |
| :-- | :-- |
| 8 個 stem 的 `oof_probs.npz` + `best.pt` | `python -m src.train_kfold ...` × 1 + `python -m src.train_pseudo_kfold ...` × 7（合計 ~7~8 GPU hr） |
| AP-D4 集成 | `python -m src.tools.u10_per_task_tta --stems <8 stems> --grid-step 0.1 --tag ap_d4_8way_3view` |

---

## 1. 環境需求

| 項目 | 規格 |
| :-- | :-- |
| OS | Windows 10/11、Ubuntu 22.04 LTS 或同等版本 |
| Python | 3.10 ~ 3.13（本隊環境 3.13.7） |
| CUDA | 12.x（驅動需支援；CPU-only 也可跑 §5 集成步驟，僅 §4 訓練需 GPU） |
| GPU VRAM | ≥ 8 GB（本隊：RTX 5060 Laptop 8.5 GB） |
| RAM | ≥ 16 GB |
| 磁碟 | ~55 GB（含 8 stems × seed42 × 5 folds 的 `best.pt`，約 44 GB；stem #8 加 3 seeds 再 +~5 GB） |
| 網路 | 首次執行需下載 HuggingFace 模型 `hfl/chinese-macbert-base`（~400 MB） |
| Ollama（可選，僅在重生 stem #8訓練語料時需要） | Ollama 0.x + qwen2.5:7b-instruct（4.7 GB）下載至預設路徑 `~/.ollama/models/` |

---

## 2. 安裝

```powershell
# Windows / PowerShell
git clone https://github.com/YOUR-ACCOUNT/esg-veripromise-2026.git
cd esg-veripromise-2026
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pytest -q   # 單元測試應全綠（目前約 50 個測試，< 30 秒）
```

```bash
# Linux / macOS
git clone https://github.com/YOUR-ACCOUNT/esg-veripromise-2026.git
cd esg-veripromise-2026
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pytest -q
```

---

## 3. 資料準備（官方資料與已追蹤種子）

| 路徑 | 內容 | 是否已隨 repo |
| :-- | :-- | :-- |
| `vpesg4k_train_1000 V1.csv` | 競賽方提供的官方 1,000 筆訓練集 | **是** |
| `vpesg4k_train_1000 V1.json` | 對應 JSON 版本 | **是** |
| `assets/u6_pro/esg_glossary.json` | U6-pro 109 詞 ESG 術語表 | **是** |
| `assets/aug_plus/handcrafted_v1.csv` | Phase 37 手工撰寫 47 列 minority 種子 | **是** |
| `assets/aug_plus/build_handcrafted_v1.py` | 種子原始建構腳本（可重生 `handcrafted_v1.csv`） | **是** |
| `configs/prompts/ap_*.yaml` | AP3 LLM 合成 prompt（mock provider 即可離線跑；生產用 Ollama provider） | **是** |
| `data/aug_plus/llm_synth_{misleading,within_2_years}.jsonl` | Phase 38 Ollama qwen2.5:7b-instruct 生成的 LLM 合成種子（140 列 raw；與手工種子合併並通過品質閘門後，159 列入 stem #8 語料） | **是** |
| `outputs/checkpoints/<stem>/seed42/fold{0..4}/best.pt` | 8 stems × 5 folds 的 fine-tuned 權重 | **否**（~44 GB，需在 §4 自行訓出） |
| `outputs/checkpoints/<stem>/seed42/fold{0..4}/oof_probs.npz` | 對應的 OOF 機率（訓練即產出） | **否** |
| `outputs/cache/u10_tta/<stem>_{middle,tail}.npz` | 3-view TTA 快取（會在 §5 自動產出） | **否** |

> 用到的 **唯一必要外部模型資源** 是 HuggingFace 的 `hfl/chinese-macbert-base`（pretrained backbone），第一次訓練時 transformers 會自動下載。AP-D4 所需的 U10/U6 pseudo/intermediate data 可由 §4.2 重生，或使用本隊另行提供的 processed bundle；stem #8 LLM 合成種子已內含於 `data/aug_plus/llm_synth_*.jsonl`，預設沒有 OpenAI/Anthropic/Gemini API key 也能跑。

---

## 4. 訓練：產出 8 個 stem 的 `best.pt`（GPU 必要）

> **時間估算**：每個 stem = 5 folds × ~5~7 min / fold = ~30~40 min；stem #8 若完整重跑多 seed 會更久。合計約 **7~8 GPU 小時**（RTX 5060 Laptop 8.5 GB，AMP fp16）。
>
> **資源**：本隊每跑一個 stem 都會在 `reports/experiments/` 寫入 5 個 fold log；如果中途崩潰可單跑 `--fold N` 續跑。

### 4.1 訓練腳本指令（依此順序執行）

```powershell
# stem #1: p2_combo_best — 純官方 1,000 列、Phase 36 最佳配方（no pseudo）
python -m src.train_kfold --config configs\exp_p2_combo_best.yaml

# stem #2: p2_combo_best_u10_pseudo — Stage A: U10 v1 偽標 (211 列) + real；Stage B: real-only
python -m src.train_pseudo_kfold --config configs\exp_p2_combo_best_u10_pseudo.yaml

# stem #3: p2_combo_best_u10_pseudo_v2 — Stage A: U10 v2 偽標 (3,904 列) + real
python -m src.train_pseudo_kfold --config configs\exp_p2_combo_best_u10_pseudo_v2.yaml

# stem #4: p2_combo_best_u10_pseudo_v2_classw_focal_t4_g3 — 同 stem #3 + 類別加權 CE + T4 Focal γ=3
python -m src.train_pseudo_kfold --config configs\exp_p2_combo_best_u10_pseudo_v2_classw_focal_t4_g3.yaml

# stem #5: p2_combo_best_u10_pseudo_v3_classw_focal_t4_g3 — U10 v3 偽標 + 同上 loss
python -m src.train_pseudo_kfold --config configs\exp_p2_combo_best_u10_pseudo_v3_classw_focal_t4_g3.yaml

# stem #6: p2_combo_best_classw_focal_u6pro — Phase 36 SOTA 主成員：U6-pro 反翻譯 434 列 + classw + focal
python -m src.train_pseudo_kfold --config configs\exp_p2_combo_best_classw_focal_u6pro.yaml

# stem #7: p2_combo_best_aug_plus — Phase 37 新增：U10-pro + handcrafted 47 列 aug_plus（AP-D3 SOTA 關鍵）
python -m src.train_pseudo_kfold --config configs\exp_p2_combo_best_aug_plus.yaml

# stem #8: p2_combo_best_aug_plus_v2 — Phase 38 新增：stem #7 語料 + Ollama LLM 合成 112 列（AP-D4 SOTA 關鍵）3 seeds × 5 folds
python -m src.train_pseudo_kfold --config configs\exp_p2_combo_best_aug_plus_v2.yaml
```

每個 stem 跑完後會在 `outputs/checkpoints/<stem>/seed42/fold{0..4}/` 寫入：
- `best.pt` — 最佳 epoch 的模型權重
- `oof_probs.npz` — 該 fold 驗證集 OOF 機率（4 個任務各一個 array）

> **stem #6/#7 的偽標籤從哪來？** stem #6 依賴 U6-pro 反翻譯產出的 434 列（路徑 `data/processed/u6_pro/u6_pro_434.csv`）；stem #7 依賴 U10-pro 偽標籤 + 47 列 handcrafted。**stem #8 的 LLM 合成種子** 已隨 repo 上傳於 `data/aug_plus/llm_synth_*.jsonl`；若本機沒有 processed bundle，請依 §4.2 / §4.3 重生。

### 4.2 （可選）重生偽標籤資料

如果本機沒有 `data/processed/` 內的偽標籤檔案，可依以下指令重生；若你已取得本隊提供的 processed bundle 且檔案路徑就位，可跳過本節直接訓練：

```powershell
# U6-pro 反翻譯（~90 min GPU + NLLB-200-distilled-600M ~3 GB 下載）
python scripts\u6_backtranslate_pro.py

# U10 v3 弱監督管線（需先有 stem #1 ckpts 當教師；~60 min）
python scripts\u10_collect_sr.py        # M1: PDF 爬取（~1.3 GB）
python scripts\u10_pdf_extract_v3.py    # M3: 段落擷取 + SimHash 去重
python scripts\u10_pseudo_label_v3.py   # M4: 教師閘門 + 偽標產出

# Aug-Plus 47 列 handcrafted seeds（已隨 repo 上傳，無需重生）
python assets\aug_plus\build_handcrafted_v1.py   # 重生 handcrafted_v1.csv
```

> 注意：U6-pro / U10 涉及外部模型版本與 PDF 解析；byte-exact 重現可能漂移 ±0.003。如果你的目標只是「重現 0.71608」，最穩做法是使用本隊提供的 processed bundle；純 GitHub clone 則需依本節重生 `data/processed/`。

### 4.3 （可選）用 Ollama 重生 stem #8 LLM 合成語料

stem #8 訓練語料 = stem #7 語料 + Ollama qwen2.5:7b-instruct 生成的 140 列 LLM 種子。**預設產出已隨 repo 上傳 (`data/aug_plus/llm_synth_*.jsonl`)**，你可以跳過本節。若要「從零生成」，需裝 Ollama （約 200 MB）+ 拉模型（約 4.7 GB）：

```powershell
# (1) 下載 Ollama https://ollama.com/download、啟動 daemon
Start-Process ollama.exe -ArgumentList 'serve' -WindowStyle Hidden
ollama pull qwen2.5:7b-instruct

# (2) 設定 ENV、生成 + 驗證 + 品質閘門 + promote
$env:OLLAMA_HOST  = 'http://localhost:11434'
$env:OLLAMA_MODEL = 'qwen2.5:7b-instruct'

python scripts\ap_llm_synth.py --target misleading       --provider ollama --n 80 --seed 42
python scripts\ap_llm_synth.py --target within_2_years   --provider ollama --n 60 --seed 42
python scripts\ap_llm_synth_validate.py --target misleading
python scripts\ap_llm_synth_validate.py --target within_2_years
python scripts\ap_merge_seeds.py        # 合併 handcraft 47 + LLM 140 = 190 列
python scripts\ap_quality_gate.py        # length filter → 159 列
python scripts\ap_promote_to_processed.py --aug-plus-version v2  # 寫出 aug_plus_v2_with_u10v2.csv (4,063 列)

# (3) 在 configs/exp_p2_combo_best_aug_plus_v2.yaml 中確認 pseudo_csv_path 指向 v2 檔，接著跑 §4.1 的 stem #8 訓練指令即可
```

> Ollama 是本機 LLM、離線、免 API key，計算量隱藏於本機 GPU（VRAM 需 ~5 GB；CPU 亦可，生成會慢 5–10 倍）。生成參數 temperature=0.85、top_p=0.92、num_predict=1400，生成時間約 30 min。亦可換為 `llama3.1:8b` / `yi:9b` 之類 ≤6 GB Q4 模型（變更後成果會漂移）。

---

## 5. 集成：產出 SOTA OOF 0.71608

訓練完 8 個 stem 後，跑下列**單一指令**即得 SOTA：

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

執行流程（指令會自動處理）：

1. **載入 stored 視角 OOF**：從每個 stem 的 `oof_probs.npz` 取 fold 0~4 平均（seed42 only）。
2. **預測 middle / tail 視角**：對每個 stem 用對應的 `best.pt` 跑 forward pass（首次 ~10 min，會寫入 `outputs/cache/u10_tta/<stem>_{middle,tail}.npz`；再次執行命中快取直接讀）。
3. **初始化 fast evaluator**：`src/tools/tta_fast_eval.py` 會把 per-stem/per-view probabilities 堆成 tensor，後續 candidate 以 integer-label exact F1 評分，避免逐列 string/sklearn pipeline。
4. **Stage A — Per-task stem 權重 hillclimb**：4 任務各搜尋 8-stem simplex（grid 0.1，19,448 點 / task）。
5. **Stage B — Per-task view 權重 hillclimb**：4 任務各搜尋 3-view simplex（grid 0.1，66 點 / task）。
6. **Joint A↔B 交替 2 輪** 直到收斂。
7. **套用階層後處理 constraints**（若 `promise_status="No"`，下游三任務強制 `N/A`）並評分。

最終終端會印出：

```
[u10-tta FINAL] 0.7160840624 delta_vs_AP_D4_SOTA(0.71608)=+0.0000040624
  task promise_status: 0.943874
  task verification_timeline: 0.631504
  task evidence_status: 0.880113
  task evidence_quality: 0.481571
```

並寫入 `reports/analysis/_ensemble/ap_d4_8way_3view_{summary.csv, meta.json, preds.csv}`。其中 `preds.csv` 是 1,000 列 × 4 預測欄的內部預測檔，可作為提交檔生成來源，但不能在官方要求文字欄位時直接上傳。

> **預期數字**：weighted = **0.71608** ± 0.0005（cuDNN 非確定性的浮點累加）。
> **每任務**：T1 = 0.94387、T2 = 0.63150、T3 = 0.88011、T4 = 0.48157。

可選的小預算 refinement（不改 AP-D4 標準重現指令）：

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
  --random-refine-iters 2000 --random-refine-step 0.05 `
  --tag ap_d4_8way_3view_refine2k
```

此模式只接受 post-constraint weighted score 上升的候選，並在 meta 中保留 `random_refine_history`；正式 anchor 仍以 valid 校準結果決定。

---

## 6. （可選）apples-to-apples baseline 對照

若要驗證 stem #8 真的帶來 +0.00244 增益，請額外跑 7-way × 3-view（AP-D3）對照（不含 stem #8）：

```powershell
python -m src.tools.u10_per_task_tta `
  --stems p2_combo_best `
          p2_combo_best_u10_pseudo `
          p2_combo_best_u10_pseudo_v2 `
          p2_combo_best_u10_pseudo_v2_classw_focal_t4_g3 `
          p2_combo_best_u10_pseudo_v3_classw_focal_t4_g3 `
          p2_combo_best_classw_focal_u6pro `
          p2_combo_best_aug_plus `
  --grid-step 0.1 --max-rounds 4 --joint-rounds 2 `
  --tag ap_d3_7way_3view
```

預期輸出 weighted = **0.71364**，與 Phase 37 AP-D3 一致；與 8-way 比差 +0.00244 ⇒ 此即 stem #8（Ollama LLM 合成）純增量貢獻。

---

## 7. 完整時間預算

| 步驟 | GPU | 時間 |
| :-- | :-- | :-- |
| §2 安裝 + 測試 | 否 | 5 ~ 10 min |
| §4.2 （可選）重生 U6-pro / U10 偽標籤 | 是 | 2 ~ 3 hr |
| §4.3 （可選）Ollama 重生 stem #8 LLM 語料 | 選配（CPU/GPU均可） | 20 ~ 40 min |
| §4.1 訓練 8 stems（stem #1~#7 各 seed42；stem #8 = 3 seeds × 5 folds） | 是 | 7 ~ 8 hr |
| §5 AP-D4 集成 | 是（首次 ~15 min）/ 否（後續快取） | 5 ~ 20 min |
| §6 AP-D3 baseline 對照（可選） | 否 | 2 ~ 5 min |
| **合計（無 §4.2 / 4.3 / 6）** | — | **~8 hr GPU + 20 min CPU** |

---

## 8. 常見問題

**Q1：`hfl/chinese-macbert-base` 下載失敗？**
A：HuggingFace 鏡像問題；可設 `HF_ENDPOINT=https://hf-mirror.com` 環境變數後重試，或先在有網路機器上 `huggingface-cli download hfl/chinese-macbert-base` 後 rsync 到 `~/.cache/huggingface/`。

**Q2：CUDA OOM？**
A：在所有 YAML 內把 `train.batch_size` 從 8 改 4 + `grad_accum_steps` 從 2 改 4（等效 batch 不變）。max_len 改成 256 亦可。

**Q3：是否能跳過 §4 直接用既有 ckpts？**
A：可以，但你需要自備 ckpts（38 GB 不上 GitHub）。聯絡本隊取得 `outputs/checkpoints/*.tar.gz`，解壓到 repo 根，再直接跳到 §5 即可（耗時 < 15 min）。

**Q4：`oof_probs.npz` 與 `best.pt` 是兩個檔案，可以只留 OOF 嗎？**
A：可以，但只能跑「stored view」的 stack，無法跑 3-view TTA（middle/tail 視角需要 `best.pt` 重新 forward）。stored-only 6-way 約 0.71018、7-way 約 0.71109，皆**低於** SOTA 0.71364。

**Q5：怎麼產提交檔？**
A：`reports/analysis/_ensemble/ap_d4_8way_3view_preds.csv` 是內部最終預測，仍需依官方 test 格式補齊欄位後才可提交。上傳前必跑：

```powershell
python -m src.tools.validate_submission outputs/submissions/<file>.csv --mode submission
```

若只是檢查內部 `*_preds.csv`，使用：

```powershell
python -m src.tools.validate_submission reports/analysis/_ensemble/ap_d4_8way_3view_preds.csv --mode preds --allow-promise-yes-na-timeline
```

**Q6：訓練 stem 數量能不能減？**
A：可以，但 SOTA 會降。路徑：Phase 36 = 6 stems = 0.71018；Phase 37 + stem #7 = 0.71364；Phase 38 + stem #8 = 0.71608。若只有 GPU 預算跑 3 stems，建議選 stem #1 + #4 + #6（base / classw_focal / u6pro），預期 OOF ~0.69~0.70。

**Q7：Ollama 模型可以換嗎？**
A：可。設 `OLLAMA_MODEL=llama3.1:8b` / `yi:9b` / `qwen2.5:14b-instruct-q4_K_M` 等 ≤6 GB Q4 中文友善模型，重跑 §4.3 生成 → promote → 訓練。但漂移後的複現數字可能與 0.71608 不同（使用不同語料 → 不同權重 → 不同終分）。官方記錄兩個結果作為 ablation。

**Q8：Ollama daemon 起不來？**
A：確認 port 11434 未被佔（`netstat -ano | findstr 11434`）；手動起：`ollama.exe serve`。Windows 預設裝在 `C:\Users\<USER>\AppData\Local\Programs\Ollama\`，模型存 `C:\Users\<USER>\.ollama\models\`；請確保 C 槽 25 GB 以上可用。

---

## 9. 文件對照

| 文件 | 用途 |
| :-- | :-- |
| [REPRODUCE.md](REPRODUCE.md) | **本檔** — 從 clone 到 SOTA 0.71608 的單一 SOP |
| [README.md](README.md) | 專案總覽、目錄、授權 |
| [MASTER_PLAN_AND_PROGRESS.md](MASTER_PLAN_AND_PROGRESS.md) | 完整研究決策總控（Phase 1~39 + 2026-05-25 工程硬化），所有 ablation、負面消融與 SOTA 軌跡 |
| [SOTA_Reproduction_Phase36.ipynb](SOTA_Reproduction_Phase36.ipynb) | Phase 36 SOTA 互動式 notebook（舊 SOTA，保留作為對照） |
| [ESG_永續承諾驗證競賽_2026.md](ESG_永續承諾驗證競賽_2026.md) | 競賽官方規則 |

---

*最後更新：2026-05-25 · SOTA 0.71608 對應 AP-D4 8-way × 3-view pipeline。*
