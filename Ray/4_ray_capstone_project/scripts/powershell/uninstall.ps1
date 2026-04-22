# Uninstall wrapper scripts from the active conda environment's Scripts directory
# Run this script with: powershell -File scripts/powershell/uninstall.ps1

$ErrorActionPreference = "Stop"

# Check if conda environment is active
if (-not $env:CONDA_PREFIX) {
    Write-Host "Error: No conda environment is active." -ForegroundColor Red
    Write-Host "Please activate your environment first: conda activate 22971-ray-capstone"
    exit 1
}

$CondaScripts = Join-Path $env:CONDA_PREFIX "Scripts"

Write-Host "Uninstalling command wrappers from: $CondaScripts"
Write-Host ""

# Remove wrapper batch files from conda environment's Scripts directory
foreach ($cmd in @("prepare", "run")) {
    $DestBat = Join-Path $CondaScripts "$cmd.bat"

    if (Test-Path $DestBat) {
        Write-Host "  Removing: $cmd.bat"
        Remove-Item $DestBat -Force
    } else {
        Write-Host "  Not found: $cmd.bat"
    }
}

Write-Host ""
Write-Host "Uninstallation complete!" -ForegroundColor Green
