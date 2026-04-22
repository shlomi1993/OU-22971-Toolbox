# test_ray_flow.ps1 — System test: downloads data, prepares assets, runs all 3 demo modes.
#
# This file executes:
#   1. scripts/download_data.ps1
#   2. prepare --ref-parquet data/green_tripdata_2023-01.parquet --replay-parquet data/green_tripdata_2023-02.parquet --output-dir prepared --n-zones 20 --seed 42
#   3. run --prepared-dir prepared --output-dir output --mode blocking --slow-zone-fraction 0.25 --slow-zone-sleep-s 1.0 --seed 42
#   4. run --prepared-dir prepared --output-dir output --mode async --slow-zone-fraction 0.25 --slow-zone-sleep-s 1.0 --tick-timeout-s 2.0 --completion-fraction 0.75 --max-inflight-zones 4 --seed 42
#   5. run --prepared-dir prepared --output-dir output --mode stress --slow-zone-fraction 0.6 --slow-zone-sleep-s 3.0 --tick-timeout-s 2.0 --seed 42
#   6. Verify all output artifacts exist (run_config.json, metrics.csv, latency_log.json, tick_summary.json, actor_counters.json, comparison.json)

param(
    [switch]$KeepArtifacts
)

$ErrorActionPreference = "Stop"

# Color output functions
function Write-Green { Write-Host $args -ForegroundColor Green }
function Write-Cyan { Write-Host $args -ForegroundColor Cyan }
function Write-Red { Write-Host $args -ForegroundColor Red }

function Log-And-Run {
    param([string]$Command)
    Write-Green $Command
    Invoke-Expression $Command
    if ($LASTEXITCODE -ne 0) {
        Write-Red "ERROR: Command failed with exit code $LASTEXITCODE"
        exit $LASTEXITCODE
    }
}

$ScriptDir = Split-Path -Parent $PSCommandPath
$ProjectDir = Split-Path -Parent $ScriptDir
Set-Location $ProjectDir

$DataDir = Join-Path $ProjectDir "data"
$PreparedDir = Join-Path $ProjectDir "prepared"
$OutputDir = Join-Path $ProjectDir "output"

$RefFile = Join-Path $DataDir "green_tripdata_2023-01.parquet"
$ReplayFile = Join-Path $DataDir "green_tripdata_2023-02.parquet"

# Suppress Ray warnings
$env:RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO = "0"
$env:RAY_DEDUP_LOGS = "0"

Write-Host ""
Write-Cyan "Ray Capstone - Full Flow Test"
Write-Cyan "============================="

# --- Download data ---
Write-Host ""
Write-Cyan "Step 1: Download TLC data"
Log-And-Run "powershell -File `"$ProjectDir\scripts\download_data.ps1`""

# --- Prepare assets ---
Write-Host ""
Write-Cyan "Step 2: Prepare replay assets"
Log-And-Run "prepare --ref-parquet `"$RefFile`" --replay-parquet `"$ReplayFile`" --output-dir `"$PreparedDir`" --n-zones 20 --seed 42"
Write-Host "Prepared assets written to $PreparedDir"

# --- Run 1: Blocking baseline ---
Write-Host ""
Write-Cyan "Step 3: Run blocking baseline"
Log-And-Run "run --prepared-dir `"$PreparedDir`" --output-dir `"$OutputDir`" --mode blocking --slow-zone-fraction 0.25 --slow-zone-sleep-s 1.0 --seed 42"
Write-Host "Blocking run complete. Artifacts in $OutputDir\blocking\"

# --- Run 2: Async controller ---
Write-Host ""
Write-Cyan "Step 4: Run async controller"
Log-And-Run "run --prepared-dir `"$PreparedDir`" --output-dir `"$OutputDir`" --mode async --slow-zone-fraction 0.25 --slow-zone-sleep-s 1.0 --tick-timeout-s 2.0 --completion-fraction 0.75 --max-inflight-zones 4 --seed 42"
Write-Host "Async run complete. Artifacts in $OutputDir\async\"

# --- Run 3: Stress test ---
Write-Host ""
Write-Cyan "Step 5: Run skew stress test"
Log-And-Run "run --prepared-dir `"$PreparedDir`" --output-dir `"$OutputDir`" --mode stress --slow-zone-fraction 0.6 --slow-zone-sleep-s 3.0 --tick-timeout-s 2.0 --seed 42"
Write-Host "Stress run complete. Artifacts in $OutputDir\stress\"

# --- Verify artifacts ---
Write-Host ""
Write-Cyan "Step 6: Verify output artifacts"
$modes = @("blocking", "async")
$files = @("run_config.json", "metrics.csv", "latency_log.json", "tick_summary.json", "actor_counters.json")

foreach ($mode in $modes) {
    foreach ($file in $files) {
        $filepath = Join-Path $OutputDir "$mode\$file"
        if (-not (Test-Path $filepath)) {
            Write-Red "FAIL: Missing $filepath"
            exit 1
        }
    }
}

$comparisonFile = Join-Path $OutputDir "stress\comparison.json"
if (-not (Test-Path $comparisonFile)) {
    Write-Red "FAIL: Missing stress comparison.json"
    exit 1
}

# List artifacts if keeping them
if ($KeepArtifacts) {
    Write-Host ""
    Write-Green "Output artifacts:"
    Get-ChildItem -Path $PreparedDir -Recurse | ForEach-Object { Write-Host "  $($_.FullName)" }
    Get-ChildItem -Path $OutputDir -Recurse | ForEach-Object { Write-Host "  $($_.FullName)" }
}

# --- Verdict ---
Write-Host ""
Write-Green "Full flow tests passed!"

# --- Cleanup ---
Write-Host ""
if (-not $KeepArtifacts) {
    Write-Host "Cleaning up generated artifacts"
    Remove-Item -Path $PreparedDir -Recurse -Force -ErrorAction SilentlyContinue
    Remove-Item -Path $OutputDir -Recurse -Force -ErrorAction SilentlyContinue
    Write-Host "Cleanup complete."
} else {
    Write-Host "Artifacts retained (--keep-artifacts flag was set)"
}
