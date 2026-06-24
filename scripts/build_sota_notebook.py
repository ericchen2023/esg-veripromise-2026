"""Assemble the Phase-36 SOTA reproduction notebook.

Run:
    python scripts/build_sota_notebook.py

Writes:
    SOTA_Reproduction_Phase36.ipynb
"""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "SOTA_Reproduction_Phase36.ipynb"

cells: list[dict] = []


def md(text: str) -> None:
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": text.splitlines(keepends=True),
    })


def code(text: str) -> None:
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": text.splitlines(keepends=True),
    })


# =============================================================================
# 0. 封面
# =============================================================================
md("""# VeriPromise ESG 2026 — Phase 36 SOTA 重現手冊

| 項目 | 內容 |
| :-- | :-- |
| 競賽 | ESG VeriPromise 2026（永續承諾驗證競賽） |
| 任務型態 | 單句多任務分類（4 任務） |
| 訓練樣本數 | 1,000 筆（`vpesg4k_train_1000 V1`） |
| OOF 加權分數 | **0.71018**（Phase 36；2026-05-10） |
| 演算法主軸 | 6 stems × 5-Fold Stratified CV × 3 seeds × 3-view TTA × per-task hillclimb |
| 主幹模型 | `hfl/chinese-macbert-base`（中文 MacBERT 基礎版） |
| 文件版本 | v1.0（與 `MASTER_PLAN_AND_PROGRESS_20260502.md` §52 同步） |

本筆記本取代官方 baseline notebook，用以**逐步、可重現**地展示目前最高 OOF 分數 **0.71018** 的完整管線。
官方 baseline 為 60 分鐘教學版本，僅示範單模型訓練；本筆記本則完整呈現：

1. 資料層：標準化、五折分層切分、雙標籤複合鍵分層。
2. 模型層：多任務分類器、cls+mean 雙池化、可選 Multi-Sample Dropout。
3. 訓練層：AdamW + 餘弦退火、層次學習率衰減（LLRD）、權重指數移動平均（EMA）、快速梯度法對抗訓練（FGM）、R-Drop、AMP fp16、類別加權交叉熵與 Focal Loss、提前停止。
4. 弱監督層：U10 永續報告書語料 → 三閘門偽標籤 → 兩階段微調（Stage A pseudo+real / Stage B real-only）。
5. 文本擴增層：U6-pro NLLB-200 雙樞紐回譯 + 109 條 ESG 術語表保護 + ChrF 過濾。
6. 集成層：6 個 stem × 3 個切片視角（stored / middle / tail）× per-task simplex hillclimb（座標下降）+ 階層後處理約束。

> **免責說明**：本筆記本為研究記錄與重現指引，非官方提交工具。所有專業術語在首次出現時均附 1 段定義。""")


md("""## 0.1 文件導覽

| 章節 | 主題 | 是否需執行 |
| :-- | :-- | :-- |
| §1 | 執行模式選擇 | 必選 |
| §2 | 環境與相依套件 | 必選 |
| §3 | 競賽計分公式（術語表） | 必讀 |
| §4 | 資料載入與標籤正規化 | 必選 |
| §5 | 五折分層切分 | 必選 |
| §6 | 多任務模型與池化 | 必選 |
| §7 | 損失函數與訓練組件 | 必選 |
| §8 | 單一 fold 訓練流程（核心迴圈） | 條件執行 |
| §9 | 6 個 stem 配方表 | 必讀 |
| §10 | U10 弱監督管線（M1~M5） | 條件執行 |
| §11 | U6-pro 反翻譯增強管線 | 條件執行 |
| §12 | OOF 機率張量重建 | 條件執行 |
| §13 | 三視角 TTA（stored / middle / tail） | 條件執行 |
| §14 | 6-way × 3-view per-task hillclimb | **核心；MODE ≥ load_oof 必跑** |
| §15 | 階層後處理約束 | 必跑 |
| §16 | 最終分數重現與提交檔輸出 | 必跑 |
| §17 | 附錄：失敗路徑與未來方向 | 必讀 |

### 「執行模式」一覽

| 模式 | 用途 | 預估耗時（RTX 5060 Laptop 8GB） |
| :-- | :-- | :-- |
| `explain` | 僅閱讀說明，不執行任何模型/訓練/推論 | 0 分鐘 |
| `load_oof` | 載入專案目錄下既有 checkpoint 重建 OOF，跑 hillclimb 與分數重現 | 約 8~12 分鐘 |
| `demo` | 對單一 stem 執行 1 fold smoke-train（64 樣本／1 epoch） | 約 3~5 分鐘 |
| `full` | 從零訓練全部 6 stems × 3 seeds × 5 folds | 約 30~40 GPU 小時 |

預設 `MODE = "load_oof"`，可在 §1 修改。""")


# =============================================================================
# 1. 執行模式
# =============================================================================
md("""## 1. 執行模式選擇

請在下方 cell 設定 `MODE`。本筆記本所有後續 cell 會依此旗標決定是否執行重型運算。""")

code("""# §1 — 執行模式
# 可選值: "explain" | "load_oof" | "demo" | "full"
MODE = "load_oof"

# 若需指定特定種子或 fold 子集，可調整下列旗標
DEMO_STEM   = "p2_combo_best"   # demo 模式只訓練此 stem 一個 fold
DEMO_SEED   = 42
DEMO_FOLD   = 0
DEMO_EPOCHS = 1
DEMO_SUBSET = 64                # 樣本子集大小

assert MODE in {"explain", "load_oof", "demo", "full"}, f"未知 MODE={MODE}"
print(f"[mode] 已選擇 MODE={MODE}")""")


# =============================================================================
# 2. 環境
# =============================================================================
md("""## 2. 環境與相依套件

本筆記本需要以下執行環境：

- Python 3.10 以上（開發環境為 3.13.7）
- PyTorch 2.2 以上，搭配 CUDA 12.x
- transformers 4.41 以上、scikit-learn 1.4 以上、pandas 2.2 以上、PyYAML 6.0 以上
- 其他：numpy、tqdm、matplotlib、seaborn、sentencepiece

完整鎖版本請見專案根目錄的 [requirements.txt](requirements.txt)。

> **硬體建議**：MacBERT-base 主幹在 `max_length=384, batch_size=8, grad_accum=2` 下，
> 單 fold 訓練於 RTX 4060 8GB 上耗時約 3~5 分鐘。若 GPU < 6GB，請將 `batch_size` 降至 4
> 並維持 `grad_accum=4` 以保持等效批量。""")

code("""# §2.1 — 套件版本檢查
import sys, platform
import importlib

REQUIRED = ["torch", "transformers", "sklearn", "pandas", "numpy", "yaml"]
print(f"Python  = {sys.version.split()[0]}  ({platform.platform()})")
for name in REQUIRED:
    try:
        m = importlib.import_module(name)
        ver = getattr(m, "__version__", "n/a")
        print(f"{name:13s} = {ver}")
    except ImportError as e:
        print(f"{name:13s} = MISSING ({e})")""")

code("""# §2.2 — GPU 偵測
import torch

if torch.cuda.is_available():
    dev = torch.device("cuda")
    name = torch.cuda.get_device_name(0)
    cap  = torch.cuda.get_device_capability(0)
    mem_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"CUDA 可用：{name}（CC {cap[0]}.{cap[1]}，VRAM {mem_total:.1f} GB）")
else:
    dev = torch.device("cpu")
    print("CUDA 不可用，將使用 CPU。建議僅在 MODE='explain' 下使用。")
DEVICE = dev""")

code("""# §2.3 — 將專案根目錄加入 sys.path，使本筆記本可 import src.*
import os
from pathlib import Path

PROJECT_ROOT = Path(os.getcwd())
# 本檔案應位於專案根目錄；若不在，請手動修改 PROJECT_ROOT
assert (PROJECT_ROOT / "src").exists(), f"未在 {PROJECT_ROOT} 找到 src/ 目錄；請確認 cwd 為專案根目錄"

import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 設定本筆記本所有資料/輸出路徑
DATA_CSV  = PROJECT_ROOT / "vpesg4k_train_1000 V1.csv"
DATA_JSON = PROJECT_ROOT / "vpesg4k_train_1000 V1.json"
CKPT_ROOT = PROJECT_ROOT / "outputs" / "checkpoints"
SPLIT_ROOT = PROJECT_ROOT / "data" / "splits"
print(f"PROJECT_ROOT = {PROJECT_ROOT}")
print(f"DATA_CSV     = {DATA_CSV.exists()}  ({DATA_CSV.name})")
print(f"CKPT_ROOT    = {CKPT_ROOT.exists()}  ({CKPT_ROOT})")""")


# =============================================================================
# 3. 計分公式
# =============================================================================
md("""## 3. 競賽計分公式與專業術語表

### 3.1 四個任務（multi-task）

| 代號 | 欄位 | 標籤域 | 任務型態 | F1 計算法 | 權重 |
| :-: | :-- | :-- | :-- | :-- | :-: |
| T1 | `promise_status` | `Yes` / `No` | 二元 | F1（positive=`Yes`） | **0.20** |
| T2 | `verification_timeline` | `already` / `within_2_years` / `between_2_and_5_years` / `longer_than_5_years` / `N/A` | 五類 | macro F1 | **0.15** |
| T3 | `evidence_status` | `Yes` / `No` / `N/A` | 三類但僅取 Yes | F1（positive=`Yes`） | **0.30** |
| T4 | `evidence_quality` | `Clear` / `Not Clear` / `Misleading` / `N/A` | 四類 | macro F1 | **0.35** |

### 3.2 加權分數

$$
S = 0.20 \\cdot F1_{T1}^{\\mathrm{Yes}} + 0.15 \\cdot \\mathrm{macroF1}_{T2}
  + 0.30 \\cdot F1_{T3}^{\\mathrm{Yes}} + 0.35 \\cdot \\mathrm{macroF1}_{T4}
$$

T4 權重最高（0.35）且為四類少數類別偏斜任務，是分數差異的主要來源。本專案投入最多努力的方向（U6-pro 回譯、Focal Loss、類別加權）皆是針對 T4 的 `Misleading` 與 `Not Clear` 兩個少數類別。

### 3.3 標籤階層約束（hierarchical constraint）

競賽規則隱含以下硬性依賴：

- 若 `promise_status = "No"` → `verification_timeline = evidence_status = evidence_quality = "N/A"`，且 `promise_string`、`evidence_string` 皆為空字串。
- 若 `evidence_status = "No"`（在 promise=Yes 條件下）→ `evidence_quality = "N/A"`，且 `evidence_string` 為空字串。

此約束在所有預測輸出之前必須強制套用，違反者下游任務 F1 會受懲罰。實作見 §15。

### 3.4 專業術語表（首次出現處皆附定義）

| 術語 | 定義 |
| :-- | :-- |
| **OOF**（Out-Of-Fold） | 在 K-Fold 交叉驗證中，每個樣本只在「未參與訓練」的 fold 中被預測，其結果稱為 OOF 預測。OOF 預測拼接後等於整個訓練集的無洩漏預測，可用來估計泛化誤差與做為 stacking 的基底。 |
| **Stem** | 本專案對「同一份 yaml 配方訓練出的多 seed × 多 fold checkpoint 集合」的稱呼。例如 `p2_combo_best` stem = 3 seeds × 5 folds = 15 個 best.pt。 |
| **Hillclimb（坐標下降）** | 在 simplex 上搜尋最佳權重的貪婪法：固定其他維度，逐維在離散網格（如 step=0.05）上嘗試所有點，更新到使分數最大的那一點，直到一輪內無任何改善為止。 |
| **TTA**（Test-Time Augmentation） | 推論時用多種文本切片或變換產生多份機率，再以加權平均得到最終預測。本專案使用 stored（訓練期儲存的 OOF）、middle（取 token 中段）、tail（取 token 尾段）三種視角。 |
| **FGM** | Fast Gradient Method 對抗訓練：在 word_embedding 上加入正比於梯度的小擾動再做一次反向傳播，相當於對 embedding 做最壞情況平滑。 |
| **EMA** | Exponential Moving Average，對權重做指數移動平均；推論時用 EMA 副本可提升穩定性。本專案存在 CPU 上以節省 VRAM。 |
| **R-Drop** | 同一 batch 跑兩次 forward（兩次獨立 dropout），對兩份 logits 取對稱 KL 並加入損失，鼓勵 dropout 不變性。 |
| **LLRD**（Layer-wise Learning Rate Decay） | 為 transformer 的不同層使用不同學習率；越淺的層學習率越小（×decay^depth），有助於保留預訓練知識。 |
| **MSD**（Multi-Sample Dropout） | 訓練期對同一個池化後特徵跑 K 次獨立 dropout 與分類頭，取平均 logits 作為當前批次的輸出，相當於 K 倍 ensemble 的免費訓練正則化。 |
| **Focal Loss** | $L = -(1-p_t)^{\\gamma} \\log p_t$，加大難樣本權重，常用於類別嚴重不平衡情境。本專案僅 T4 使用 ($\\gamma=3$)。 |
| **Pseudo-label** | 用已訓練模型對「無標註」資料推論並把高信心結果當作偽標籤，加入訓練集做半監督學習。本專案 U10 用四任務聯合閾值閘門收 3,904 筆。 |
| **Stratified K-Fold** | 切 K 折時保證每折的目標標籤分佈接近總體；本專案以 `promise_status + "|" + evidence_status` 為複合鍵分層。 |
| **NLLB**（No Language Left Behind） | Meta 的多語翻譯模型；本專案 U6-pro 用 `nllb-200-distilled-600M` 做雙樞紐（en、ja）回譯。 |
| **ChrF** | 字元 n-gram F-score，用來衡量譯文與原文字面相似度；本專案以 `char_order=6, word_order=0` 評估回譯品質。 |""")


# =============================================================================
# 4. 資料層
# =============================================================================
md("""## 4. 資料載入與標籤正規化

官方資料 `vpesg4k_train_1000 V1.csv` / `.json` 共 1,000 筆，欄位包含 `data`（原始中文段落）、四個任務標籤、可選的 `promise_string` / `evidence_string`（標註者擷取的關鍵句）以及 `company`、`ticker` 等中介資料。

### 4.1 標籤正規化規則

- CSV/JSON 中 `verification_timeline`、`evidence_status`、`evidence_quality` 的 `not applicable` 以 `NaN` / `null` 編碼，**統一補成字串 `"N/A"`**。
- `promise_string` / `evidence_string` 缺失補成空字串 `""`。
- 任何欄位值若不在標籤域內視為錯誤、立刻 raise。

實作位於 [src/data/loader.py](src/data/loader.py) 的 `load_dataset()` 與 `_normalize_labels()`。""")

code("""# §4.1 — 載入資料並驗證 schema
from src.data.loader import load_dataset, LABEL_DOMAINS

records, df = load_dataset(DATA_CSV)
print(f"載入 {len(records)} 筆樣本，欄位數 {len(df.columns)}")
print("\\n四任務標籤域：")
for f, dom in LABEL_DOMAINS.items():
    print(f"  {f:24s} = {dom}")""")

code("""# §4.2 — 簡易 EDA：每任務每類別樣本數
import pandas as pd
from src.data.dataset import TASKS

dist = pd.DataFrame({t: df[t].value_counts() for t in TASKS}).fillna(0).astype(int)
print("每任務每類別計數（橫軸=任務，縱軸=標籤；NaN 已正規化為 N/A）")
print(dist.to_string())""")

code("""# §4.3 — 文本長度（字元）分佈
df["text_len"] = df["data"].astype(str).map(len)
print("文本長度（字元）統計：")
print(df["text_len"].describe(percentiles=[0.5, 0.9, 0.95, 0.99]).to_string())
print(f"\\n本專案統一使用 max_length=384 token；中文 1 字 ≈ 1 token，p99={int(df['text_len'].quantile(0.99))} 字，足夠涵蓋。")""")


# =============================================================================
# 5. 五折分層
# =============================================================================
md("""## 5. 五折分層切分（Stratified K-Fold）

### 5.1 分層鍵設計

- 主分層鍵：`promise_status + "|" + evidence_status`（共 2 × 3 = 6 種組合，覆蓋核心相依結構）
- 切法：`sklearn.model_selection.StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)`
- 種子：`{42, 2024, 20260417}` 共 3 個，後續做 seed-averaging
- 已序列化：所有 fold 的 train/val 索引存於 `data/splits/{stem}/seed{S}.json`，下游推論完全依該檔案重建 OOF 對位。

### 5.2 為何不使用 GroupKFold？

U11 探針實驗（見 MASTER_PLAN §40）顯示，按 `company` 分組會使 fold 分佈嚴重不均衡（部分公司樣本佔超過 fold 大小），導致少數類別在某些 fold 完全消失。最終採 StratifiedKFold + 雙標籤複合鍵，並接受少量公司內樣本可能落於不同 fold 的弱洩漏代價（已在 §40 量化為 ≤ 0.005 偏置，遠低於演算法層改進）。""")

code("""# §5.1 — 重建 5 折切分（與既有 checkpoint 完全一致）
from src.data.splits import make_folds, report_distribution
from src.seed import set_seed

SEEDS = [42, 2024, 20260417]
N_SPLITS = 5
STRATIFY_FIELDS = ["promise_status", "evidence_status"]

set_seed(42)
folds_seed42 = make_folds(df, n_splits=N_SPLITS, stratify_fields=STRATIFY_FIELDS, seed=42)
for fi, (tr, va) in enumerate(folds_seed42):
    print(f"seed=42  fold={fi}  train={len(tr):4d}  val={len(va):4d}")
print()
print("註：相同 seed 下的切分結果在 sklearn 內為決定性；故此處重新計算與既有 checkpoint 對位完全一致。")""")


# =============================================================================
# 6. 模型
# =============================================================================
md("""## 6. 多任務模型架構

### 6.1 整體結構

- 主幹：`hfl/chinese-macbert-base`（ymcui/Chinese-MacBERT，BERT 架構＋MLM-as-correction 預訓練；中文任務常見 baseline）
- 池化：`cls_mean`（拼接 [CLS] 向量與 attention-mask 加權平均向量；輸出維度 = 2 × hidden_size = 1536）
- 分類頭：4 個獨立的 `nn.Linear(1536, C_t)`，前接每任務獨立的 dropout
- 可選 MSD（K=5 等）：訓練期對同一 feature 多次 dropout＋線性，logits 取平均

實作見 [src/models/multitask.py](src/models/multitask.py) 與 [src/models/pooling.py](src/models/pooling.py)。

### 6.2 為什麼不使用 large 模型？

| 主幹 | OOF | 備註 |
| :-- | :-: | :-- |
| `chinese-macbert-base` | 0.6694 | 本專案主力 |
| `chinese-macbert-large` | 0.6510 | Phase 3、5 fold 平均；於 8GB GPU 受限於 batch_size 必須降至 2 |
| `chinese-roberta-wwm-ext-base` | 0.6601 | Phase 4a |
| `bert-base-chinese` | 0.6480 | Phase 4b |
| `xlm-roberta-base` / `nezha-base` / `ernie-base` / `electra-base` | 全部 < 0.66 | X4 / X11 / X13 已驗證為負樣本 |

> **教訓**：1,000 筆訓練資料對 large 模型容量過度，base 反而泛化更佳。""")

code("""# §6.1 — 建立模型範例（不訓練，僅展示結構）
from src.models.multitask import MultiTaskClassifier
from src.data.dataset import NUM_LABELS

if MODE != "explain":
    demo_model = MultiTaskClassifier(
        backbone="hfl/chinese-macbert-base",
        num_labels=NUM_LABELS,
        pooling="cls_mean",
        dropout=0.1,
        msd_k=1,
    )
    n_params = sum(p.numel() for p in demo_model.parameters())
    n_train  = sum(p.numel() for p in demo_model.parameters() if p.requires_grad)
    print(f"參數量 total={n_params/1e6:.2f}M  trainable={n_train/1e6:.2f}M")
    print(f"\\n各任務分類頭輸出維度：{NUM_LABELS}")
    del demo_model
else:
    print("[explain mode] 跳過模型實例化")""")


# =============================================================================
# 7. 損失與訓練組件
# =============================================================================
md("""## 7. 損失函數與訓練組件

### 7.1 多任務交叉熵 `MultiTaskCE`

$$
L_{\\text{total}} = \\sum_{t \\in \\{T1,T2,T3,T4\\}} w_t \\cdot L_t
$$

- 預設 $w_t = 1$（任務權重相等）。
- 若 `use_class_weight=true`：$L_t = \\mathrm{CE}(\\text{logits}, y; w_c)$，其中 $w_c \\propto 1/\\mathrm{count}(c)$ 並再正規化使均值為 1。
- 若 `focal_tasks` 包含某任務：該任務改用 `FocalLoss(γ)`。

### 7.2 Focal Loss（僅 T4 使用）

$$
L^{\\mathrm{focal}}_t = -\\sum_c \\alpha_c (1 - p_c)^{\\gamma} \\, y_c \\, \\log p_c
$$

本專案 T4 使用 $\\gamma = 3.0$（Phase 33 確認），有效抑制 Clear 主導訊號、放大 Misleading / Not Clear 的學習壓力。

### 7.3 進階訓練技巧（Phase B 系列驗證）

| 技巧 | 開關 yaml 鍵 | Phase 36 採用？ |
| :-- | :-- | :--: |
| AMP fp16 | `training.use_amp` | 是 |
| 梯度累積（grad_accum=2） | `training.grad_accum` | 是 |
| 餘弦退火 + 10% warmup | `training.scheduler=cosine` | 是 |
| 梯度裁剪（max_norm=1.0） | `training.grad_clip` | 是 |
| LLRD（decay=0.95） | `training.llrd_decay` | 部分 stem |
| EMA（decay=0.995/0.999） | `training.ema_decay` | 部分 stem |
| FGM 對抗（eps=1.0） | `training.fgm_eps` | 部分 stem |
| R-Drop（α=0.5） | `training.rdrop_alpha` | 否（Phase 33 後撤除） |
| Multi-Sample Dropout（K=5） | `model.msd_k` | 否（Phase 33 後撤除） |
| 提前停止（patience=2） | `training.early_stop_patience` | 是 |

詳細實作見 [src/training/trainer.py](src/training/trainer.py) 與 [src/training/losses.py](src/training/losses.py)。""")


# =============================================================================
# 8. 訓練主迴圈
# =============================================================================
md("""## 8. 單一 fold 訓練流程

完整訓練實作位於 [src/training/trainer.py](src/training/trainer.py) 的 `train_fold()` 函數，本節示範如何從筆記本呼叫。

> **執行條件**：本 cell 僅在 `MODE in ("demo", "full")` 下執行。`demo` 模式只跑 1 fold × 1 epoch × 64 樣本；`full` 模式跑 6 stems × 3 seeds × 5 folds（≈ 30 GPU 小時）。

### 8.1 demo：smoke-train 一個 fold""")

code("""# §8.1 — Demo: smoke-train 一個 fold（僅 MODE='demo'）
if MODE == "demo":
    from src.config import load_config
    from src.train_kfold import _build_loaders
    from src.training.trainer import train_fold
    from transformers import AutoTokenizer

    cfg_path = PROJECT_ROOT / "configs" / f"exp_{DEMO_STEM}.yaml"
    cfg = load_config(cfg_path)
    cfg["training"]["epochs"] = DEMO_EPOCHS
    cfg["split"]["n_splits"] = 5

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["backbone"])

    set_seed(DEMO_SEED)
    folds = make_folds(df.iloc[:DEMO_SUBSET].reset_index(drop=True),
                       n_splits=2, stratify_fields=STRATIFY_FIELDS, seed=DEMO_SEED)
    tr_idx, va_idx = folds[DEMO_FOLD]
    sub_records = records[:DEMO_SUBSET]
    train_recs = [sub_records[i] for i in tr_idx]
    val_recs   = [sub_records[i] for i in va_idx]

    train_loader, val_loader = _build_loaders(train_recs, val_recs, tokenizer, cfg)

    model = MultiTaskClassifier(
        backbone=cfg["model"]["backbone"],
        num_labels=NUM_LABELS,
        pooling=cfg["model"].get("pooling", "cls_mean"),
        dropout=float(cfg["model"].get("dropout", 0.1)),
        msd_k=int(cfg["model"].get("msd_k", 1)),
    )

    out_root = PROJECT_ROOT / "outputs" / "checkpoints" / "_demo" / f"seed{DEMO_SEED}" / f"fold{DEMO_FOLD}"
    log_path = PROJECT_ROOT / "outputs" / "logs" / "_demo" / f"seed{DEMO_SEED}.jsonl"
    res = train_fold(
        fold=DEMO_FOLD, seed=DEMO_SEED,
        train_records=train_recs, val_records=val_recs,
        model=model, tokenizer=tokenizer,
        train_loader=train_loader, val_loader=val_loader,
        cfg=cfg, out_root=out_root, log_path=log_path,
        val_global_indices=[int(i) for i in va_idx],
    )
    print(f"\\n[demo done] best_score={res.best_score:.5f} best_epoch={res.best_epoch}")
    print(f"per_task: {res.per_task}")
else:
    print(f"[skip] 當前 MODE={MODE!r}，跳過 demo 訓練。")""")

md("""### 8.2 full：訓練全部 stem

等同於在終端機執行：

```powershell
# 每個 stem 跑 3 seeds × 5 folds，順序執行；每行約 30~40 分鐘
python -m src.train_kfold        --config configs/exp_p2_combo_best.yaml
python -m src.train_pseudo_kfold --config configs/exp_p2_combo_best_u10_pseudo.yaml
python -m src.train_pseudo_kfold --config configs/exp_p2_combo_best_u10_pseudo_v2.yaml
python -m src.train_pseudo_kfold --config configs/exp_p2_combo_best_u10_pseudo_v2_classw_focal_t4_g3.yaml
python -m src.train_pseudo_kfold --config configs/exp_p2_combo_best_u10_pseudo_v3_classw_focal_t4_g3.yaml
python -m src.train_pseudo_kfold --config configs/exp_p2_combo_best_classw_focal_u6pro.yaml
```

> **附註**：`train_pseudo_kfold` 與 `train_kfold` 唯一差異是支援 `data.pseudo_csv_path` 與 `pseudo.{min_confidence, max_pseudo, stage_a_epochs, stage_b_epochs}`，內部呼叫的 `train_fold()` 完全相同。""")

code("""# §8.2 — Full: 訓練全部 stem（僅 MODE='full'）
if MODE == "full":
    import subprocess
    STEMS_TO_TRAIN = [
        ("train_kfold",        "exp_p2_combo_best.yaml"),
        ("train_pseudo_kfold", "exp_p2_combo_best_u10_pseudo.yaml"),
        ("train_pseudo_kfold", "exp_p2_combo_best_u10_pseudo_v2.yaml"),
        ("train_pseudo_kfold", "exp_p2_combo_best_u10_pseudo_v2_classw_focal_t4_g3.yaml"),
        ("train_pseudo_kfold", "exp_p2_combo_best_u10_pseudo_v3_classw_focal_t4_g3.yaml"),
        ("train_pseudo_kfold", "exp_p2_combo_best_classw_focal_u6pro.yaml"),
    ]
    for entry, cfg_name in STEMS_TO_TRAIN:
        cmd = [sys.executable, "-m", f"src.{entry}", "--config", str(PROJECT_ROOT / "configs" / cfg_name)]
        print("[full] $", " ".join(cmd))
        subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT))
else:
    print(f"[skip] 當前 MODE={MODE!r}，未啟動 full 訓練。")""")


# =============================================================================
# 9. 6 Stem 配方表
# =============================================================================
md("""## 9. Phase 36 SOTA 的 6 個 Stem 配方

| # | Stem 名稱 | yaml | 主要差異 | 5-fold OOF (seed=42) | 角色 |
| :-: | :-- | :-- | :-- | :-: | :-- |
| 1 | `p2_combo_best` | [`exp_p2_combo_best.yaml`](configs/exp_p2_combo_best.yaml) | lr=3e-5, max_len=384, bs=8/ga=2, 不使用 class weight | 0.6694 | baseline ensemble seed |
| 2 | `p2_combo_best_u10_pseudo` | [`exp_p2_combo_best_u10_pseudo.yaml`](configs/exp_p2_combo_best_u10_pseudo.yaml) | + U10 v1 偽標籤（211 筆） | 0.6701 | 偽標籤 v1 |
| 3 | `p2_combo_best_u10_pseudo_v2` | [`exp_p2_combo_best_u10_pseudo_v2.yaml`](configs/exp_p2_combo_best_u10_pseudo_v2.yaml) | + U10 v2 偽標籤（3,904 筆） | 0.6677 | 偽標籤 v2，多樣性貢獻 |
| 4 | `p2_combo_best_u10_pseudo_v2_classw_focal_t4_g3` | [`...v2_classw_focal_t4_g3.yaml`](configs/exp_p2_combo_best_u10_pseudo_v2_classw_focal_t4_g3.yaml) | + class-weighted CE on T1/T2/T3 + Focal γ=3 on T4 | 0.6741 | T4 救援 |
| 5 | `p2_combo_best_u10_pseudo_v3_classw_focal_t4_g3` | [`...v3_classw_focal_t4_g3.yaml`](configs/exp_p2_combo_best_u10_pseudo_v3_classw_focal_t4_g3.yaml) | + U10 v3 偽標籤（3,110 筆，放寬擷取） | 0.6663 | 與 v2 去相關 |
| 6 | `p2_combo_best_classw_focal_u6pro` | [`exp_p2_combo_best_classw_focal_u6pro.yaml`](configs/exp_p2_combo_best_classw_focal_u6pro.yaml) | 在 #4 之上 + U6-pro 反翻譯擴增（434 筆 minority） | 0.6704 | T4 多樣性壓軸 |

每個 stem 都跑 3 seeds (`{42, 2024, 20260417}`) × 5 folds = 15 個 best.pt（stem #1）；後續 stem 因晚期啟動，部分只完成 2 seeds × 5 folds = 10 個。實際數量於下一 cell 自動驗證。""")

code("""# §9.1 — 驗證所有 stem checkpoint 是否齊備
SOTA_STEMS = [
    "p2_combo_best",
    "p2_combo_best_u10_pseudo",
    "p2_combo_best_u10_pseudo_v2",
    "p2_combo_best_u10_pseudo_v2_classw_focal_t4_g3",
    "p2_combo_best_u10_pseudo_v3_classw_focal_t4_g3",
    "p2_combo_best_classw_focal_u6pro",
]
print(f"{'Stem':56s}  seeds  ckpts")
for stem in SOTA_STEMS:
    p = CKPT_ROOT / stem
    if not p.exists():
        print(f"{stem:56s}  -      MISSING")
        continue
    seeds = sorted([d.name for d in p.iterdir() if d.name.startswith("seed")])
    n = sum(1 for _ in p.rglob("best.pt"))
    print(f"{stem:56s}  {len(seeds):2d}     {n:2d}")
print("\\n若任何 stem MISSING：請於 MODE='full' 下訓練之，或從備份還原。")""")


# =============================================================================
# 10. U10 弱監督
# =============================================================================
md("""## 10. U10 弱監督管線（M1 ~ M5）

> Phase 36 的 6 個 stem 中有 5 個依賴 U10 偽標籤資料。本節說明該資料如何產生。
> **執行條件**：完整 U10 流程涉及網路爬取與 PDF 解析，總時數約 4~6 小時；本筆記本預設僅展示介面。

### 10.1 五個模組

| 模組 | 腳本 | 輸入 | 輸出 | 一句話功能 |
| :-: | :-- | :-- | :-- | :-- |
| M1 | [`scripts/u10_collect_sr.py`](scripts/u10_collect_sr.py) | `u10_sources.COMPANY_SOURCES`（30 家公司） | `data/raw/u10/*.pdf` | 從公開資訊觀測站爬永續報告書 PDF |
| M2 | （並入 M1） | — | — | 同上，整合在 M1 |
| M3 | [`scripts/u10_pdf_extract.py`](scripts/u10_pdf_extract.py) | `data/raw/u10/*.pdf` | `data/processed/u10/corpus.jsonl` | PDF 段落擷取＋ESG 關鍵字過濾＋SimHash 去重 |
| M4 | [`scripts/u10_pseudo_label.py`](scripts/u10_pseudo_label.py)（v1）<br>[`u10_pseudo_label_v2.py`](scripts/u10_pseudo_label_v2.py)（v2 兩階閘門）<br>[`u10_pseudo_label_v3.py`](scripts/u10_pseudo_label_v3.py)（v3 放寬擷取） | `corpus.jsonl` + `p2_combo_best` 15-ckpt 集成 | `pseudo_labels{,_v2,_v3}.csv` | 對未標註段落跑教師模型，四任務聯合閘門收件 |
| M5 | [`src/train_pseudo_kfold.py`](src/train_pseudo_kfold.py) | 偽標籤 CSV + 官方 1000 筆 | 5-fold checkpoint | 兩階段訓練：Stage A（pseudo+real）→ Stage B（real-only fine-tune） |

### 10.2 M4 偽標籤閘門（v1 全任務最嚴）

對每個段落同時要求四任務的 max softmax 機率超過閾值才收件：

| 任務 | 閾值 |
| :-: | :-: |
| T1 promise_status | ≥ 0.80 |
| T2 verification_timeline | ≥ 0.60 |
| T3 evidence_status | ≥ 0.70 |
| T4 evidence_quality | ≥ 0.60 |

v1（211 筆）→ v2（3,904 筆，加入 minority-boost 二階閘門）→ v3（3,110 筆，PDF 擷取濾波放寬）。三版差異主要在召回率與類別覆蓋；v2 / v3 並用以維持 ensemble 多樣性。

### 10.3 兩階段訓練策略

- **Stage A**（4 epoch）：在「官方 train fold + 偽標籤」聯合資料上訓練，loss 與正常一致；驗證集**永遠**只看官方 val fold。
- **Stage B**（3 epoch）：以 Stage A 最佳 checkpoint 為 warm start，**僅用官方 train fold** 微調。Stage B 的 loss 只看真標籤，可矯正偽標籤帶來的分佈偏移。

> **教訓**：直接 Stage A 一階段訓練（無 Stage B fine-tune）通常較弱 0.005~0.008；Stage B 是兩階段法的關鍵。

### 10.4 reproducibility 注意

由於 U10 涉及外部網站結構（公開資訊觀測站 SPA）與 PDF 解析的版本依賴（`pdfplumber`），**完整重跑 M1~M3 可能無法 byte-exact 重現** `corpus.jsonl`。為了讓本筆記本可重現 SOTA 分數，建議：

1. 使用本隊另行提供的 processed bundle，或本機已重生的 `data/processed/u10/corpus.jsonl` 與 `pseudo_labels_v{1,2,3}.csv`。
2. 若這些檔案缺失，可改執行 `MODE='full'` + 跑 M1~M4，但分數會在 ±0.003 範圍內漂移。""")

code("""# §10.1 — 偽標籤檔案存在性檢查
PSEUDO_FILES = [
    PROJECT_ROOT / "data" / "processed" / "u10" / "pseudo_labels.csv",
    PROJECT_ROOT / "data" / "processed" / "u10" / "pseudo_labels_v2.csv",
    PROJECT_ROOT / "data" / "processed" / "u10" / "pseudo_labels_v3.csv",
]
for p in PSEUDO_FILES:
    flag = "OK   " if p.exists() else "MISS "
    size = f"{p.stat().st_size//1024} KB" if p.exists() else "-"
    print(f"{flag} {p.relative_to(PROJECT_ROOT)}  {size}")""")


# =============================================================================
# 11. U6-pro
# =============================================================================
md("""## 11. U6-pro 反翻譯增強管線

> Phase 36 stem #6（`p2_combo_best_classw_focal_u6pro`）的關鍵差異化資產。
> 完整實作見 [scripts/u6_backtranslate_pro.py](scripts/u6_backtranslate_pro.py)。

### 11.1 為什麼要做反翻譯？

T2 與 T4 的少數類別（`within_2_years`、`longer_than_5_years`、`Misleading`、`Not Clear`）樣本稀少（合計 < 100 筆），任何單純重採樣都易導致過擬合。反翻譯能在「保留語意但改變表面結構」的條件下產生新樣本。

### 11.2 四道防線

1. **ESG 術語表（109 條）**
   - 自建 `data/processed/u6_pro/esg_glossary.json`。
   - 涵蓋 Scope 1/2/3、CBAM、TCFD、SBTi、淨零、CO₂e、RE100、TNFD、ISSB、CDP、ISO 14001/14064/14067 等。
   - 每條附 `{zh, en, category, passthrough}`，`passthrough=True` 代表縮寫（如 `CBAM`、`SBTi`）回譯時應原樣保留。
2. **多樞紐 × 多溫度**
   - 樞紐語言：英文 `eng_Latn` + 日文 `jpn_Jpan`（兩者語法距離與中文不同向）。
   - 解碼策略：greedy + temperature 0.7 + temperature 1.0 = 每來源 6 候選。
3. **品質指標 + per-source top-k 排序**
   - 以 sacrebleu 的 **char-only ChrF**（`char_order=6, word_order=0, beta=2`）量度回譯與原文的字面相似度。
   - 對每個來源按 ChrF 排序取 top-k=2。
4. **術語後校正**
   - 對每個落漏的中文 glossary term，於回譯文中以 case-insensitive regex 找回對應英文/日文形式並 substitute 回中文。

### 11.3 過濾門檻

| 指標 | 門檻 | 說明 |
| :-: | :-: | :-- |
| ChrF | ≥ 0.08 | 防垃圾下界（純 char-based 中文同義詞替換會壓低分數，門檻寬） |
| 長度比 | [0.5, 1.6] | 避免過度膨脹或壓縮 |
| 中文字元比 | ≥ 0.6 | 過濾翻譯失敗的英日殘留 |
| Glossary recall | ≥ 0.7 | 至少 70% 中文 ESG 術語在回譯文中保留 |

### 11.4 產出統計

```
源樣本數         : 485 筆 minority（T2 within_2y/longer_5y + T4 Misleading/Not Clear）
候選總數         : 2,910（485 × 6）
ChrF 通過        : 634
per-source top-2 : 434 筆（最終納入訓練）
ChrF 中位數      : 0.189
注入 fold 規模   : fold0=355 / fold1=339 / fold2=342 / fold3=344 / fold4=356
```

注入規則：每筆增強樣本附 `_source_id`；訓練時若該 source 落於本 fold 的 train index 才注入，**永不洩漏到 val/OOF**。""")

code("""# §11.1 — U6-pro 增強檔存在性檢查
U6_AUG = PROJECT_ROOT / "data" / "processed" / "u6_pro" / "u6_backtrans_pro.json"
U6_GLOSSARY = PROJECT_ROOT / "data" / "processed" / "u6_pro" / "esg_glossary.json"
for p in [U6_AUG, U6_GLOSSARY]:
    flag = "OK  " if p.exists() else "MISS"
    size = f"{p.stat().st_size//1024} KB" if p.exists() else "-"
    print(f"{flag} {p.relative_to(PROJECT_ROOT)}  {size}")""")


# =============================================================================
# 12. OOF 重建
# =============================================================================
md("""## 12. OOF 機率張量重建

每個 fold 訓練到 `best epoch` 時，會將該 fold 的 val 集 softmax 機率以 `np.float16` 存於 `outputs/checkpoints/{stem}/seed{S}/fold{F}/oof_probs.npz`。本節將同一 stem 的 5 個 fold 拼接成完整 OOF 張量，再對 3 個 seed 取平均。

### 12.1 拼接邏輯

- `oof_probs.npz` 內含：
  - `indices`: shape `(N_val,)` int32，**全資料的全域索引**（不是 fold 內局部索引）
  - `probs_<task>`: shape `(N_val, C_t)` float16
  - `meta`: JSON 字串記錄 fold/seed/best_epoch/score
- 拼接：對每個 (stem, seed)，遍歷 5 fold 將 `probs[indices]` 寫入全域張量；五 fold 完整覆蓋全部 1,000 筆樣本。
- Seed-averaging：對同 stem 不同 seed 的 OOF 張量逐 task 取平均。

實作見 [src/tools/oof_ensemble.py](src/tools/oof_ensemble.py) 的 `_build_seed_oof()`。""")

code("""# §12.1 — 對所有 6 個 stem 重建 stored OOF（seed-averaged）
import numpy as np
from src.data.dataset import NUM_LABELS
from src.tools.oof_ensemble import _build_seed_oof

if MODE == "explain":
    print("[skip] explain 模式不執行重建。")
else:
    n = len(records)
    stored_per_stem = {}
    for stem in SOTA_STEMS:
        exp_dir = CKPT_ROOT / stem
        splits_dir = SPLIT_ROOT / stem
        seed_dirs = sorted([p for p in exp_dir.iterdir() if p.name.startswith("seed")])
        seeds = [int(p.name.replace("seed", "")) for p in seed_dirs]
        accum = {t: np.zeros((n, NUM_LABELS[t]), dtype=np.float64) for t in NUM_LABELS}
        for s in seeds:
            sp = _build_seed_oof(exp_dir, splits_dir, n, s)
            for t in NUM_LABELS:
                accum[t] += sp[t]
        for t in NUM_LABELS:
            accum[t] /= len(seeds)
        stored_per_stem[stem] = accum
        print(f"[oof] {stem:56s}  seeds={seeds}  shape={accum['promise_status'].shape}")
    print(f"\\n共重建 {len(stored_per_stem)} 個 stem 的 stored OOF（已對 seed 取平均）。")""")


# =============================================================================
# 13. TTA
# =============================================================================
md("""## 13. 三視角 Test-Time Augmentation

### 13.1 三個視角的 token 切片策略

對於每筆樣本 `text`，先以 tokenizer 編碼成 `token_ids`（不加特殊 token），再依視角切片：

| 視角 | 切片規則 | 物理意涵 |
| :-: | :-- | :-- |
| **stored** | 訓練期儲存的 OOF 機率（隱含「head」切片，因 `truncation=True` 從前段截斷） | 與訓練分佈一致 |
| **middle** | 從 `(len - W) // 2` 開始連續取 W 個 token | 段落主體 |
| **tail** | 取最後 W 個 token | 結尾承諾／證據常出現處 |

其中 `W = max_length - num_special_tokens_to_add(pair=False)`。三視角拼上 [CLS]/[SEP] 後 padding 至 `max_length`。

### 13.2 為什麼 tail / middle 有用？

- T1（promise_status）的關鍵句常出現在段尾（如「我們承諾於 2030 年達成」）→ tail 加權主導。
- T3 / T4 的證據描述常分佈在中段→ stored + middle 各半。
- §52.5 的 α* 結果驗證了上述假設。

### 13.3 實作

`src/tools/u1_tta_oof.py::predict_one_view()` 對每個 (stem, seed, fold) 載入 best.pt，對 val 集以指定 view 重新 tokenize 並推論，最後拼接成完整 OOF。結果以 `outputs/cache/u10_tta/{stem}_{view}.npz` 快取，**首次執行較久（每 stem 每 view 約 1~2 分鐘 GPU 時間），後續秒開**。""")

code("""# §13.1 — 對所有 6 個 stem 計算 middle / tail 視角 OOF（自動快取）
from src.tools.u10_per_task_tta import _ensure_nonstored_view, VIEWS

if MODE in ("load_oof", "demo", "full"):
    use_amp = (DEVICE.type == "cuda")
    per_stem_per_view = {s: {"stored": stored_per_stem[s]} for s in SOTA_STEMS}
    for stem in SOTA_STEMS:
        for view in ("middle", "tail"):
            print(f"[tta] {stem} / {view} ...")
            per_stem_per_view[stem][view] = _ensure_nonstored_view(
                stem=stem, view=view, records=records,
                device=DEVICE, use_amp=use_amp, batch_size=None,
            )
    print(f"\\n共建立 {len(per_stem_per_view)} stems × {len(VIEWS)} views = {len(SOTA_STEMS)*len(VIEWS)} 份 OOF 機率。")
else:
    print(f"[skip] 當前 MODE={MODE!r}，跳過 TTA 計算。")""")


# =============================================================================
# 14. 6-way × 3-view hillclimb
# =============================================================================
md("""## 14. 6-way × 3-view per-task hillclimb（Phase 36 核心）

### 14.1 搜尋空間

對每個任務 $t$ 我們同時找：
- **Stem 權重** $w^{(t)} \\in \\Delta^5$（6 維 simplex；6 個 stem 加總為 1）
- **View 權重** $\\alpha^{(t)} \\in \\Delta^2$（3 維 simplex；stored+middle+tail 加總為 1）

混合公式：
$$
P^{(t,v)}_{ij} = \\sum_k w^{(t)}_k \\, P^{(\\text{stem}_k, v)}_{tij}
\\qquad
\\hat{P}^{(t)}_{ij} = \\sum_v \\alpha^{(t)}_v \\, P^{(t,v)}_{ij}
$$

### 14.2 兩階段交替坐標下降

由於 $w$ 與 $\\alpha$ 是不交集的 simplex，採交替式座標下降即可避免兩維度過搜：

1. **Stage A**（固定 view α=stored only）：在 6-stem simplex 上對每 task 逐維搜尋，grid step=0.05（共 $C(20+5, 5) \\approx 53,130$ 點）。
2. **Stage B**（固定 stem w=Stage A 結果）：在 3-view simplex 上對每 task 逐維搜尋，grid step=0.05（共 $C(20+2, 2) = 231$ 點）。
3. **Joint refinement**：交替執行 A→B 直到 1e-9 內無改善（一般 1~2 輪）。

### 14.3 實作

直接呼叫 `src/tools/u10_per_task_tta.py` 的核心函數 `stage_a_stem_search` / `stage_b_view_search`：""")

code("""# §14.1 — 6-way × 3-view per-task hillclimb
from src.tools.u10_per_task_tta import (
    stage_a_stem_search, stage_b_view_search, _eval_full,
    U10_STEMS as DEFAULT_STEMS,
)
import src.tools.u10_per_task_tta as _tta_mod
from src.data.dataset import TASKS

if MODE in ("load_oof", "demo", "full"):
    # 將 stem 集合替換為 Phase 36 的 6 個 stem
    _tta_mod.U10_STEMS = tuple(SOTA_STEMS)
    K = len(SOTA_STEMS)
    init_stem  = {t: tuple([1.0/K] * K) for t in TASKS}
    init_alpha = {t: (1.0, 0.0, 0.0) for t in TASKS}  # stored only

    # Stage A
    print("=== Stage A：6-stem simplex 搜尋（固定 view=stored only） ===")
    stem_star, sa_score = stage_a_stem_search(
        per_stem_per_view=per_stem_per_view,
        records=records,
        grid_step=0.05,
        init_stem=init_stem,
        fixed_view_alpha=init_alpha,
        max_rounds=4,
    )
    print(f"\\n[stage A done] {sa_score['final_weighted_score']:.10f}")

    # Stage B
    print("\\n=== Stage B：3-view simplex 搜尋（固定 stem*=Stage A） ===")
    init_b = {t: (0.5, 0.5, 0.0) for t in TASKS}
    alpha_star, sb_score = stage_b_view_search(
        per_stem_per_view=per_stem_per_view,
        records=records,
        grid_step=0.05,
        fixed_stem=stem_star,
        init_alpha=init_b,
        max_rounds=4,
    )
    print(f"\\n[stage B done] {sb_score['final_weighted_score']:.10f}")

    # Joint refinement
    print("\\n=== Joint refinement（A↔B 交替） ===")
    cur = sb_score["final_weighted_score"]
    for j in range(1, 3):
        stem_star, sa = stage_a_stem_search(
            per_stem_per_view=per_stem_per_view, records=records,
            grid_step=0.05, init_stem=stem_star,
            fixed_view_alpha=alpha_star, max_rounds=4,
        )
        alpha_star, sb = stage_b_view_search(
            per_stem_per_view=per_stem_per_view, records=records,
            grid_step=0.05, fixed_stem=stem_star,
            init_alpha=alpha_star, max_rounds=4,
        )
        if sb["final_weighted_score"] - cur < 1e-9:
            print(f"[joint r{j}] 收斂。")
            break
        cur = sb["final_weighted_score"]
        print(f"[joint r{j}] {cur:.10f}")
else:
    print(f"[skip] 當前 MODE={MODE!r}，跳過 hillclimb。")""")


# =============================================================================
# 15. 後處理
# =============================================================================
md("""## 15. 階層後處理約束

實作位於 [src/inference/post_process.py](src/inference/post_process.py) 的 `apply_constraints_batch()`。在每筆預測 argmax 後套用：

```python
if promise_status == "No":
    verification_timeline = evidence_status = evidence_quality = "N/A"
    promise_string = evidence_string = ""
elif evidence_status == "No":
    evidence_quality = "N/A"
    evidence_string = ""
```

該約束**必須在 hillclimb 評分迴圈中即套用**（而不是事後），否則 hillclimb 會選到「不合法但 OOF F1 較高」的權重。`u10_per_task_tta._score()` 內部已經正確呼叫。""")


# =============================================================================
# 16. 最終分數重現
# =============================================================================
md("""## 16. 最終分數重現與提交檔輸出

執行下方 cell 將：

1. 以 §14 找到的 `stem_star` / `alpha_star` 重新混合機率
2. 套用 §15 的階層後處理
3. 計算 4 task F1 與最終加權分數
4. 寫出 `outputs/submissions/sota_phase36_oof.csv`

預期結果：

```
weighted = 0.71018
T1 = 0.94210
T2 = 0.62778
T3 = 0.87774
T4 = 0.46934
```

實際數字會因 cuDNN 非決定性、torch 浮點累加順序在 ±0.0005 內漂移。""")

code("""# §16.1 — 計算最終 OOF 分數並輸出提交檔
import pandas as pd
from src.inference.post_process import apply_constraints_batch
from src.eval.metrics import weighted_score
from src.data.dataset import LABEL_DOMAINS

if MODE in ("load_oof", "demo", "full"):
    final_score, final_preds = _eval_full(
        per_stem_per_view=per_stem_per_view,
        stem_weights_per_task=stem_star,
        view_alpha_per_task=alpha_star,
        records=records,
    )
    print("=" * 60)
    print(f"[Phase 36 重現] weighted_score = {final_score['final_weighted_score']:.5f}")
    print("-" * 60)
    for t in TASKS:
        print(f"  {t:24s} F1 = {final_score[t]:.5f}")
    print("=" * 60)
    print(f"\\nstem*  = {stem_star}")
    print(f"alpha* = {alpha_star}")

    # 寫提交檔
    sub_dir = PROJECT_ROOT / "outputs" / "submissions"
    sub_dir.mkdir(parents=True, exist_ok=True)
    sub_path = sub_dir / "sota_phase36_oof.csv"
    pd.DataFrame(final_preds).to_csv(sub_path, index=False, encoding="utf-8")
    print(f"\\n[wrote] {sub_path}")
else:
    print(f"[skip] 當前 MODE={MODE!r}，跳過分數重現。")""")


# =============================================================================
# 17. 附錄
# =============================================================================
md("""## 17. 附錄

### 17.1 已驗證為負樣本的方向（X 系列，請避免重複嘗試）

| 代號 | 嘗試 | 結果 | 對 OOF 影響 |
| :-: | :-- | :-- | :-: |
| X1 | label smoothing 0.1（全任務） | 弱化分類面 | -0.005 |
| X4 | XLM-R / mDeBERTa | 跨語言預訓練無加成 | < base |
| X6 | NeZha base | 中文預訓練不足 | -0.012 |
| X7 | ERNIE base | 同上 | -0.010 |
| X9 | macbert-large（8GB GPU） | bs=2 訓練不穩 | -0.018 |
| X11 | ELECTRA base | 判別式預訓練不利 ESG | -0.014 |
| X13 | NLLB-200-distilled-600M（U6 first-cut，無術語表） | 專有名詞被誤譯 | -0.003 |
| X14 | rdrop α=0.5 + msd K=5 同時開 | 訓練不穩定 | -0.004 |

### 17.2 未來可探索方向（N 系列）

- N1：跨家族 teacher（XLM-R / mDeBERTa stem）走 7-way TTA。
- N2：T2 額外做時間軸正則化（將 `within_2_years` 改用順序回歸而非 5-class CE）。
- N3：U6-pro 升級至 NLLB-200-3.3B 並做 ChrF round-trip filter。
- N4：6/03 valid 集釋出後做 hold-out 校準，若 valid 退化 > 0.008 則回退至 Phase 35 的 5-way × 3-view。

### 17.3 上限分析

依 §54 的 oracle 推導，本資料集（1,000 筆，標籤雜訊與標註者間不一致估計約 5~7%）的 OOF 上限約為 **0.74 ~ 0.76**。Phase 36 的 0.71018 距上限約 0.03 ~ 0.05，主要差距來自：

- T2 `within_2_years` 與 T4 `Misleading` 仍是少數類別瓶頸。
- 跨家族多樣性尚未引入。

### 17.4 相關檔案索引

| 類型 | 路徑 |
| :-- | :-- |
| 主規劃文件 | [MASTER_PLAN_AND_PROGRESS_20260502.md](MASTER_PLAN_AND_PROGRESS_20260502.md) |
| 競賽規則 | [ESG_永續承諾驗證競賽_2026.md](ESG_永續承諾驗證競賽_2026.md) |
| 主訓練腳本 | [src/train_kfold.py](src/train_kfold.py) / [src/train_pseudo_kfold.py](src/train_pseudo_kfold.py) |
| 模型 | [src/models/multitask.py](src/models/multitask.py) / [src/models/pooling.py](src/models/pooling.py) |
| 訓練器 | [src/training/trainer.py](src/training/trainer.py) / [src/training/losses.py](src/training/losses.py) / [src/training/schedulers.py](src/training/schedulers.py) |
| 資料 | [src/data/loader.py](src/data/loader.py) / [src/data/dataset.py](src/data/dataset.py) / [src/data/splits.py](src/data/splits.py) |
| 推論 | [src/inference/post_process.py](src/inference/post_process.py) / [src/tools/u1_tta_oof.py](src/tools/u1_tta_oof.py) |
| 集成 | [src/tools/oof_ensemble.py](src/tools/oof_ensemble.py) / [src/tools/u10_per_task_tta.py](src/tools/u10_per_task_tta.py) / [src/tools/u10_classw_stack_search.py](src/tools/u10_classw_stack_search.py) |
| U10 弱監督 | [scripts/u10_collect_sr.py](scripts/u10_collect_sr.py) / [scripts/u10_pdf_extract.py](scripts/u10_pdf_extract.py) / [scripts/u10_pseudo_label.py](scripts/u10_pseudo_label.py) |
| U6-pro 反翻譯 | [scripts/u6_backtranslate_pro.py](scripts/u6_backtranslate_pro.py) |
| 評分 | [src/eval/metrics.py](src/eval/metrics.py) |

---

> 本筆記本最末。如需更深入的迭代脈絡（Phase 1 ~ Phase 36 的所有實驗紀錄），請參閱 `MASTER_PLAN_AND_PROGRESS_20260502.md` 第 II ~ IV 部分。""")


# =============================================================================
# Write notebook
# =============================================================================
notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3 (.venv)",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.13",
            "mimetype": "text/x-python",
            "file_extension": ".py",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"Wrote {OUT}  ({OUT.stat().st_size//1024} KB, {len(cells)} cells)")
