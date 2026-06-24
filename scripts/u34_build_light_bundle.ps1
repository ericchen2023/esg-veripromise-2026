<#
.SYNOPSIS
    Build Bundle 1 (highest_lb_light_bundle) for ESG VeriPromise 2026.

.DESCRIPTION
    Copies everything needed to reproduce the highest-LB result (0.6037, c3) and
    to add new stems, WITHOUT the heavy checkpoint .pt files (those go to Bundle 2).

    Contents:
      configs/                    all experiment YAML configs
      scripts/                    all Python scripts
      src/                        all source code
      assets/                     all assets (aug data helpers, u6_pro, etc.)
      data/processed/             train_val_combined.csv + pseudo/aug processed CSVs
      data/splits/                per-stem CV fold assignments (seed JSON files)
      data/aug_plus/              aug_gated.csv + synthetic data used by aug_plus stems
      data/raw/                   raw competition CSV/JSON (NO PDFs - too large)
      outputs/cache/              TTA window probability caches
      outputs/submissions/        test probs NPZs + FROZEN_BEST + phase50/51 candidates
      outputs/checkpoints/        oof_probs.npz ONLY from the 9 active stems (no .pt)
      reports/                    experiment logs, ensemble weights, analysis
      root files                  pyproject.toml, requirements.txt, *.md, competition CSVs/JSONs

    Usage:
        powershell -File scripts\u34_build_light_bundle.ps1
        powershell -File scripts\u34_build_light_bundle.ps1 -Dest D:\my-bundle\light
#>
param(
    [string]$Dest = "F:\esg-bundle\highest_lb_light_bundle",
    [string]$Src  = "F:\esg-veripromise-2026"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# The 8 TV stems that produced the banked 0.6037 (equal-weight + macro prior-corr a0.3)
# plus the TAPT stem used for Phase 51 binary-only surgical blend.
$KEY_STEMS = @(
    "p2_combo_best_tv",
    "p2_combo_best_u10_pseudo_tv",
    "p2_combo_best_u10_pseudo_v2_tv",
    "p2_combo_best_u10_pseudo_v2_classw_focal_t4_g3_tv",
    "p2_combo_best_u10_pseudo_v3_classw_focal_t4_g3_tv",
    "p2_combo_best_classw_focal_u6pro_tv",
    "p2_combo_best_aug_plus_tv",
    "p2_combo_best_aug_plus_v2_tv",
    "p49_tapt_combo_best"
)

function Rcopy($from, $to) {
    if (-not (Test-Path $from)) { Write-Warning "SKIP (not found): $from"; return }
    robocopy $from $to /E /NFL /NDL /NJH /NJS /nc /ns /np | Out-Null
    if ($LASTEXITCODE -ge 8) { throw "robocopy failed ($LASTEXITCODE): $from -> $to" }
}

function Fcopy($from, $to) {
    if (-not (Test-Path $from)) { Write-Warning "SKIP file (not found): $from"; return }
    $dir = Split-Path $to -Parent
    if ($dir -and -not (Test-Path $dir)) { New-Item -ItemType Directory -Path $dir -Force | Out-Null }
    Copy-Item $from $to -Force
}

Write-Host "=== Building highest_lb_light_bundle ===" -ForegroundColor Cyan
Write-Host "  src  : $Src"
Write-Host "  dest : $Dest"
New-Item -ItemType Directory -Path $Dest -Force | Out-Null

# ------------------------------------------------------------------
# 1. Code / configs (full directory copies)
# ------------------------------------------------------------------
Write-Host "[1/7] Code directories (configs, scripts, src, assets)..."
foreach ($d in @("configs", "scripts", "src", "assets")) {
    Rcopy "$Src\$d" "$Dest\$d"
    Write-Host "      $d/ ... ok"
}

# ------------------------------------------------------------------
# 2. Training data
# ------------------------------------------------------------------
Write-Host "[2/7] Training data..."
Rcopy "$Src\data\processed"  "$Dest\data\processed"
Write-Host "      data/processed/ ... ok"
Rcopy "$Src\data\splits"     "$Dest\data\splits"
Write-Host "      data/splits/ ... ok"
Rcopy "$Src\data\aug_plus"   "$Dest\data\aug_plus"
Write-Host "      data/aug_plus/ ... ok"
if (Test-Path "$Src\data\reports") {
    Rcopy "$Src\data\reports" "$Dest\data\reports"
    Write-Host "      data/reports/ ... ok"
}

# data/raw: CSV/JSON only (skip PDFs — too large and not needed for training)
$rawDest = "$Dest\data\raw"
Get-ChildItem "$Src\data\raw" -File -Recurse |
    Where-Object { $_.Extension -in '.csv', '.json' } |
    ForEach-Object {
        $rel  = $_.FullName.Substring("$Src\data\raw\".Length)
        Fcopy $_.FullName "$rawDest\$rel"
    }
Write-Host "      data/raw/ CSV+JSON ... ok"

# ------------------------------------------------------------------
# 3. outputs/cache  (TTA windows etc.)
# ------------------------------------------------------------------
Write-Host "[3/7] outputs/cache/..."
Rcopy "$Src\outputs\cache" "$Dest\outputs\cache"

# ------------------------------------------------------------------
# 4. outputs/submissions — test probs NPZs + key CSVs
# ------------------------------------------------------------------
Write-Host "[4/7] outputs/submissions (NPZ + FROZEN_BEST + phase50/51 candidates)..."
$submDest = "$Dest\outputs\submissions"
New-Item -ItemType Directory -Path $submDest -Force | Out-Null

# All test-prob NPZ caches (8-stem, TTA windows, TAPT)
Get-ChildItem "$Src\outputs\submissions" -Filter "*.npz" |
    ForEach-Object { Copy-Item $_.FullName "$submDest\" -Force }

# Frozen best (0.6037) + all Phase 50/51 candidates
Get-ChildItem "$Src\outputs\submissions" -Filter "FROZEN_BEST*.csv" |
    ForEach-Object { Copy-Item $_.FullName "$submDest\" -Force }
Get-ChildItem "$Src\outputs\submissions" |
    Where-Object { $_.Name -match "^phase5[0-9]" } |
    ForEach-Object { Copy-Item $_.FullName "$submDest\" -Force }

# ------------------------------------------------------------------
# 5. outputs/checkpoints — oof_probs.npz ONLY from the 9 key stems
# ------------------------------------------------------------------
Write-Host "[5/7] outputs/checkpoints — oof_probs.npz from $($KEY_STEMS.Count) stems..."
$oofsFound = 0
foreach ($stem in $KEY_STEMS) {
    for ($fold = 0; $fold -le 4; $fold++) {
        $srcF  = "$Src\outputs\checkpoints\$stem\seed42\fold$fold\oof_probs.npz"
        $dstDir = "$Dest\outputs\checkpoints\$stem\seed42\fold$fold"
        if (Test-Path $srcF) {
            New-Item -ItemType Directory -Path $dstDir -Force | Out-Null
            Copy-Item $srcF "$dstDir\oof_probs.npz" -Force
            $oofsFound++
        } else {
            Write-Warning "  missing: $srcF"
        }
    }
}
Write-Host "      $oofsFound / $($KEY_STEMS.Count * 5) oof_probs.npz files copied"

# ------------------------------------------------------------------
# 6. reports/ (full — includes ensemble weights, hillclimb meta JSONs)
# ------------------------------------------------------------------
Write-Host "[6/7] reports/..."
Rcopy "$Src\reports" "$Dest\reports"

# ------------------------------------------------------------------
# 7. Root files
# ------------------------------------------------------------------
Write-Host "[7/7] Root files..."
$rootFiles = @(
    "pyproject.toml", "requirements.txt", "README.md",
    "MASTER_PLAN_AND_PROGRESS.md", "REPRODUCE.md", "LICENSE",
    "ESG_永續承諾驗證競賽_2026.md",
    "sample_submission_format.csv",
    "vpesg4k_train_1000 V1.csv", "vpesg4k_val_1000.csv", "vpesg4k_test_2000.csv",
    "vpesg4k_train_1000 V1.json", "vpesg4k_val_1000.json", "vpesg4k_test_2000.json"
)
foreach ($f in $rootFiles) {
    if (Test-Path "$Src\$f") {
        Copy-Item "$Src\$f" "$Dest\$f" -Force
        Write-Host "      $f"
    }
}

# ------------------------------------------------------------------
# Generate BUNDLE_MANIFEST.txt
# ------------------------------------------------------------------
Write-Host "`nGenerating BUNDLE_MANIFEST.txt..."
$manifestLines = @(
    "# highest_lb_light_bundle — generated $(Get-Date -Format 'yyyy-MM-dd HH:mm')",
    "# Banked best LB = 0.6037  (FROZEN_BEST_0.6037_c3.csv, MD5 46D31CE3...)",
    "# Phase 51 surgical blend candidate: phase51_tapt_binary_submission.csv",
    "# TAPT binary oracle upper bound: ~0.6073 (tapt_only binary + c3 macro)",
    "# Key ensemble weights: reports/analysis/_ensemble/tv_oof_ensemble_meta.json",
    "# Rebuild c3: equal8 probs (phase43_test_probs.npz) + prior_correct(T2/T4, a=0.3)",
    "# DO NOT overwrite reports/analysis/_ensemble/ meta files — they are frozen weights.",
    "# Checkpoint .pt files are in Bundle 2 (highest_lb_checkpoints).",
    "",
    "=== FILE COUNT BY TYPE ==="
)

$allFiles = Get-ChildItem $Dest -Recurse -File
$byExt = $allFiles | Group-Object Extension | Sort-Object Count -Descending
foreach ($g in $byExt) {
    $sizeKB = [math]::Round(($g.Group | Measure-Object Length -Sum).Sum / 1KB, 0)
    $manifestLines += ("  {0,-8} {1,5} files  {2,8} KB" -f $g.Name, $g.Count, $sizeKB)
}

$totalMB = [math]::Round(($allFiles | Measure-Object Length -Sum).Sum / 1MB, 1)
$manifestLines += ""
$manifestLines += "=== TOTAL: $($allFiles.Count) files, ${totalMB} MB ==="
$manifestLines += ""
$manifestLines += "=== KEY STEMS WITH OOF PROBS ==="
foreach ($s in $KEY_STEMS) { $manifestLines += "  $s" }
$manifestLines += ""
$manifestLines += "=== TOP-LEVEL STRUCTURE ==="
Get-ChildItem $Dest -Directory | ForEach-Object {
    $cnt = (Get-ChildItem $_.FullName -Recurse -File).Count
    $manifestLines += ("  {0,-35}  ({1} files)" -f $_.Name, $cnt)
}

$manifestLines | Set-Content "$Dest\BUNDLE_MANIFEST.txt" -Encoding UTF8

Write-Host "`n=== Bundle 1 complete ===" -ForegroundColor Green
Write-Host "  Total files : $($allFiles.Count)"
Write-Host "  Total size  : $totalMB MB"
Write-Host "  Location    : $Dest"
Write-Host "  Manifest    : $Dest\BUNDLE_MANIFEST.txt"
