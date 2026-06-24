# u37_run_binary_pool.ps1 — Phase 53 one-command runbook (RTX 5060 Laptop 8 GB).
#
# Builds a diverse, domain-adapted binary-specialist pool and assembles T1/T3
# ensemble submission candidates (macro T2/T4 stay at the banked c3 base).
# Rationale + expectations: MASTER_PLAN_AND_PROGRESS.md Phase 53 (§70).
#
# Run from the repo root in PowerShell:  ./scripts/u37_run_binary_pool.ps1
# Total wall-clock on 8 GB: roughly 9-10 GPU hours.
#
# IMPORTANT (PS 5.1): do NOT use $ErrorActionPreference="Stop" here. HuggingFace
# writes warnings + non-fatal background-thread messages (e.g. Thread-auto_
# conversion, the safetensors auto-convert) to stderr; under "Stop" PowerShell
# promotes the first such line to a terminating error and kills the whole run
# even though python exits 0. We use "Continue" + explicit $LASTEXITCODE checks
# so only a REAL non-zero python exit is treated as a step failure, and a failed
# step is logged + skipped (partial pool still builds candidates).

$ErrorActionPreference = "Continue"
$env:PYTHONIOENCODING = "utf-8"
$env:TRANSFORMERS_VERBOSITY = "error"
# All backbones + the macbert-TAPT encoder are already cached locally, so run
# fully offline: no Hub network calls, and the non-fatal safetensors auto-
# conversion thread (which 403s on get_repo_discussions) never fires. Keeps the
# long unattended log clean so a REAL error is not buried under Hub tracebacks.
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
$script:Fails = @()

function Invoke-Step([string]$label, [string[]]$pyargs) {
  Write-Host "`n=== $label ===" -ForegroundColor Cyan
  & python @pyargs
  if ($LASTEXITCODE -ne 0) {
    Write-Host "[WARN] step FAILED: $label (python exit $LASTEXITCODE) -- continuing" -ForegroundColor Yellow
    $script:Fails += $label
    return $false
  }
  return $true
}

# ---------------------------------------------------------------------------
# 0. HARD GATE: cached-probs path must reproduce the 0.6113 winner bit-exactly
#    before we trust any candidate this script builds. ~15 s, no GPU.
# ---------------------------------------------------------------------------
Write-Host "`n=== 0/5 reproduce-check (HARD GATE, must PASS) ===" -ForegroundColor Cyan
& python -m scripts.u35_binary_pool_ensemble --reproduce
if ($LASTEXITCODE -ne 0) {
  Write-Host "[ABORT] reproduce gate failed -- pipeline drifted, not training." -ForegroundColor Red
  exit 1
}

# ---------------------------------------------------------------------------
# 1. Domain-adapt (TAPT) the two LARGE backbones on the 4,000-paragraph ESG
#    corpus (train+val+test text, no labels). macbert-TAPT already exists.
#    ~45-75 min each on 8 GB (batch 4, grad_accum 2, max_len 256).
# ---------------------------------------------------------------------------
if (-not (Test-Path "outputs/tapt/roberta_large_esg/config.json")) {
  Invoke-Step "1a/5 TAPT roberta-large" @(
    "-m","scripts.u30_tapt_pretrain",
    "--backbone","hfl/chinese-roberta-wwm-ext-large",
    "--out-dir","outputs/tapt/roberta_large_esg",
    "--epochs","8","--lr","3e-5","--batch","4","--grad-accum","2","--max-length","256")
}
if (-not (Test-Path "outputs/tapt/electra_large_esg/config.json")) {
  Invoke-Step "1b/5 TAPT electra-large" @(
    "-m","scripts.u30_tapt_pretrain",
    "--backbone","hfl/chinese-electra-180g-large-discriminator",
    "--out-dir","outputs/tapt/electra_large_esg",
    "--epochs","8","--lr","3e-5","--batch","4","--grad-accum","2","--max-length","256")
}

# ---------------------------------------------------------------------------
# 2. Train binary-specialist stems on combined-2000, 5-fold StratifiedKFold.
#    A failed large-model step is skipped; the rest still build candidates.
# ---------------------------------------------------------------------------
Invoke-Step "2a/5 train macbert-TAPT multi-seed" @("-m","src.train_kfold","--config","configs/exp_p53_tapt_macbert_ms.yaml")
if (Test-Path "outputs/tapt/roberta_large_esg/config.json") {
  Invoke-Step "2b/5 train roberta-large-TAPT" @("-m","src.train_kfold","--config","configs/exp_p54_tapt_robertalarge.yaml")
}
if (Test-Path "outputs/tapt/electra_large_esg/config.json") {
  Invoke-Step "2c/5 train electra-large-TAPT" @("-m","src.train_kfold","--config","configs/exp_p55_tapt_electralarge.yaml")
}

# ---------------------------------------------------------------------------
# 3. Honest GroupKFold CV on the CHEAP macbert models only (~0.7 h each):
#    "does TAPT generalise better on held-out companies than plain macbert?"
#    Conservative lower bound (test reuses the same 50 companies) -> ranking.
# ---------------------------------------------------------------------------
Invoke-Step "3a/5 GroupKFold combo_best"  @("-m","src.train_kfold","--config","configs/exp_p2_combo_best.yaml","--split-mode","stratified_group_kfold")
Invoke-Step "3b/5 GroupKFold macbert-TAPT" @("-m","src.train_kfold","--config","configs/exp_p49_tapt_combo_best.yaml","--split-mode","stratified_group_kfold")
Invoke-Step "3c/5 rank"                    @("-m","scripts.u36_groupkfold_binary_rank","p2_combo_best","p49_tapt_combo_best")

# ---------------------------------------------------------------------------
# 4. Build candidates: 5-fold test inference per available stem (cached), then
#    per-member + full-pool binary ensembles with the c3 macro base. Validated.
# ---------------------------------------------------------------------------
Invoke-Step "4/5 build + validate candidates" @("-m","scripts.u35_binary_pool_ensemble","--build")

# ---------------------------------------------------------------------------
# 5. Summary.
# ---------------------------------------------------------------------------
Write-Host "`n=== 5/5 DONE ===" -ForegroundColor Cyan
if ($script:Fails.Count -gt 0) {
  Write-Host "Steps that failed and were skipped:" -ForegroundColor Yellow
  $script:Fails | ForEach-Object { Write-Host "  - $_" -ForegroundColor Yellow }
} else {
  Write-Host "All steps succeeded." -ForegroundColor Green
}
Write-Host "Candidates: outputs/submissions/u35_*_submission.csv" -ForegroundColor Green
Write-Host "Banked best = 0.6113. Only upload a new candidate as the day's LAST" -ForegroundColor Yellow
Write-Host "upload if it beats 0.6113 (check the u36 GroupKFold ranking first)." -ForegroundColor Yellow
