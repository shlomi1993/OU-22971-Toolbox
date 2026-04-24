# download_data.ps1 — Download TLC Green Taxi parquet files into the data/ directory.
# Skips files that already exist. Safe to run multiple times.
# WARNING: this script was not tested!

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $PSCommandPath
$ProjectDir = Split-Path -Parent $ScriptDir

$DataDir = Join-Path $ProjectDir "data"
if (-not (Test-Path $DataDir)) {
    New-Item -ItemType Directory -Path $DataDir | Out-Null
}

$RefUrl = "https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_2023-01.parquet"
$ReplayUrl = "https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_2023-02.parquet"
$RefFile = Join-Path $DataDir "green_tripdata_2023-01.parquet"
$ReplayFile = Join-Path $DataDir "green_tripdata_2023-02.parquet"

if (-not (Test-Path $RefFile)) {
    Write-Host "Downloading reference month (2023-01)..."
    Invoke-WebRequest -Uri $RefUrl -OutFile $RefFile
} else {
    Write-Host "Reference file already exists, skipping."
}

if (-not (Test-Path $ReplayFile)) {
    Write-Host "Downloading replay month (2023-02)..."
    Invoke-WebRequest -Uri $ReplayUrl -OutFile $ReplayFile
} else {
    Write-Host "Replay file already exists, skipping."
}

Write-Host "Data directory: $DataDir"
