# Install wrapper scripts into the active conda environment's Scripts directory
# Run this script with: powershell -File scripts/install.ps1

$ErrorActionPreference = "Stop"

# Check if conda environment is active
if (-not $env:CONDA_PREFIX) {
    Write-Host "Error: No conda environment is active." -ForegroundColor Red
    Write-Host "Please activate your environment first: conda activate 22971-ray-capstone"
    exit 1
}

$ScriptDir = Split-Path -Parent $PSCommandPath
$ProjectDir = Split-Path -Parent $ScriptDir
$BinDir = Join-Path $ProjectDir "bin"
$CondaScripts = Join-Path $env:CONDA_PREFIX "Scripts"

Write-Host "Installing command wrappers to: $CondaScripts"
Write-Host ""

# Create wrapper batch files in conda environment's Scripts directory
foreach ($cmd in @("prepare", "run")) {
    $SrcBash = Join-Path $BinDir $cmd
    $DestBat = Join-Path $CondaScripts "$cmd.bat"

    if (Test-Path $DestBat) {
        Write-Host "  Removing existing wrapper: $cmd.bat" -ForegroundColor Yellow
        Remove-Item $DestBat -Force
    }

    Write-Host "  Creating wrapper: $cmd.bat -> bash $SrcBash"

    # Create a batch file that calls the bash script via bash (Git Bash or WSL)
    @"
@echo off
bash "$SrcBash" %*
"@ | Set-Content -Path $DestBat -Encoding ASCII
}

Write-Host ""
Write-Host "Installation complete! You can now run:" -ForegroundColor Green
Write-Host "  prepare --ref-parquet <file> --replay-parquet <file> --output-dir <dir>"
Write-Host "  run --prepared-dir <dir> --output-dir <dir> --mode <blocking|async|stress>"
Write-Host ""
Write-Host "Note: Requires Git Bash or WSL bash in PATH." -ForegroundColor Yellow
Write-Host "To uninstall, run: powershell -File scripts/uninstall.ps1"
