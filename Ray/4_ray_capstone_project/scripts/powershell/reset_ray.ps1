# Reset Ray environment - stop Ray and remove generated artifacts
# Run this script with: powershell -File scripts/powershell/reset_ray.ps1

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $PSCommandPath
$ProjectDir = Split-Path -Parent $ScriptDir
Set-Location $ProjectDir

Write-Host "Stopping Ray..."
# Check if conda is available
if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    Write-Host "ERROR: conda not found" -ForegroundColor Red
    exit 1
}

# Check if environment exists
$envExists = conda env list | Select-String "22971-ray-capstone"
if (-not $envExists) {
    Write-Host "ERROR: conda env 22971-ray-capstone not found" -ForegroundColor Red
    exit 1
}

# Stop Ray (ignore errors if not running)
conda run -n 22971-ray-capstone ray stop --force 2>$null

Write-Host "Removing prepared assets..."
Remove-Item -Path (Join-Path $ProjectDir "prepared") -Recurse -Force -ErrorAction SilentlyContinue

Write-Host "Removing output artifacts..."
Remove-Item -Path (Join-Path $ProjectDir "output") -Recurse -Force -ErrorAction SilentlyContinue

Write-Host "Reset complete." -ForegroundColor Green
