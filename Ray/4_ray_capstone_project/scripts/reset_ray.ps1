# reset_ray.ps1 - Stop Ray and clean up generated artifacts.
# WARNING: This script was not tested!

$ErrorActionPreference = "Continue"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectDir = Split-Path -Parent $ScriptDir

Write-Host "Stopping Ray..." -ForegroundColor Green
if (Get-Command ray -ErrorAction SilentlyContinue) {
    try {
        $job = Start-Job -ScriptBlock { ray stop --force 2>&1 | Out-Null }
        $null = Wait-Job -Job $job -Timeout 30
        if ($job.State -eq "Completed") {
            Write-Host "Ray stopped successfully" -ForegroundColor Green
        } else {
            Write-Host "Warning: Ray stop command timed out, continuing with cleanup..." -ForegroundColor Yellow
        }
        Remove-Job -Job $job -Force
    } catch {
        Write-Host "Warning: Failed to stop Ray: $_, continuing with cleanup..." -ForegroundColor Yellow
    }
} else {
    Write-Host "Warning: Ray command not found, skipping Ray shutdown" -ForegroundColor Yellow
}

Write-Host "Removing generated artifacts..." -ForegroundColor Green
$OutputDir = Join-Path $ProjectDir "output"
if (Test-Path $OutputDir) {
    Remove-Item -Path $OutputDir -Recurse -Force
    Write-Host "Output artifacts removed: $OutputDir" -ForegroundColor Green
} else {
    Write-Host "No output artifacts to remove"
}

Write-Host "Reset complete" -ForegroundColor Green
