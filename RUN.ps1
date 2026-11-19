<#
CST435 - ONE COMMAND: compile and benchmark REAL NYC Yellow Taxi data.
All prepared datasets come from the uploaded January-April 2025 Parquet files.
No synthetic data and no Mall Customers data.
Examples:
  powershell -ExecutionPolicy Bypass -File .\RUN.ps1
  powershell -ExecutionPolicy Bypass -File .\RUN.ps1 -Extended
#>
param(
    [string]$Sizes = '100000,1000000,5000000,10000000',
    [string]$Threads = '1,2,4',
    [int]$Repeats = 10,
    [int]$K = 5,
    [switch]$Extended,
    [switch]$AllCombinations
)
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
Write-Host '========================================' -ForegroundColor Cyan
Write-Host ' CST435: REAL NYC TAXI K-MEANS ONLY' -ForegroundColor Cyan
Write-Host '========================================' -ForegroundColor Cyan

# Reuse existing compiler from MSYS2 UCRT64 if installed.
$gpath='C:\msys64\ucrt64\bin'
if (Test-Path (Join-Path $gpath 'g++.exe')) { $env:Path="$gpath;$env:Path" }
if (-not (Get-Command g++ -ErrorAction SilentlyContinue)) {
    throw 'g++ was not found. Install MSYS2 UCRT64 and its C++ compiler.'
}
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    throw 'Python not found. Install Python and add it to PATH.'
}
& python -c 'import numpy,matplotlib'
if ($LASTEXITCODE -ne 0) {
    throw 'Missing packages. Run: python -m pip install numpy matplotlib'
}

# Rebuild a real-data subset only if requested file is not yet available.
$sizesArray=@($Sizes -split ',' | ForEach-Object { [int]$_.Trim() })
$missing=@($sizesArray | Where-Object { -not (Test-Path "data\taxi_$($_).kmbin") })
if ($missing.Count -gt 0) {
    if (Test-Path 'data\taxi_all.kmbin') {
        Write-Host 'Creating missing subsets from already-cleaned REAL taxi records...' -ForegroundColor Yellow
        & python prepare_taxi.py --from-clean --sizes $Sizes
    } elseif (Test-Path 'data\raw\yellow_tripdata_2025-01.parquet') {
        Write-Host 'Preparing real NYC taxi Parquet records...' -ForegroundColor Yellow
        & python prepare_taxi.py --sizes $Sizes
    } else {
        throw 'Data missing. Copy taxi_all.kmbin into data, OR put all four original .parquet files in data\raw and run python prepare_taxi.py.'
    }
    if ($LASTEXITCODE -ne 0) { throw 'Real data preprocessing failed.' }
}

New-Item -ItemType Directory -Force build | Out-Null
Write-Host 'Building Sequential C++...' -ForegroundColor Yellow
& g++ -std=c++17 -O3 algorithms\kmeans.cpp algorithms\sequential.cpp -o build\kmeans_sequential.exe
if ($LASTEXITCODE -ne 0) { throw 'Sequential build failed.' }
Write-Host 'Building OpenMP C++...' -ForegroundColor Yellow
& g++ -std=c++17 -O3 -fopenmp algorithms\kmeans.cpp algorithms\openmp.cpp -o build\kmeans_openmp.exe
if ($LASTEXITCODE -ne 0) { throw 'OpenMP build failed.' }
Write-Host 'Building Pthreads C++...' -ForegroundColor Yellow
& g++ -std=c++17 -O3 -pthread algorithms\kmeans.cpp algorithms\pthreads.cpp -o build\kmeans_pthreads.exe
if ($LASTEXITCODE -ne 0) { throw 'Pthreads build failed.' }

Write-Host 'Running REAL taxi benchmark and creating figures...' -ForegroundColor Green
$arglist=@('benchmark.py','--sizes',$Sizes,'--threads',$Threads,'--repeats',"$Repeats",'--k',"$K")
if ($Extended) { $arglist+='--extended' }
if ($AllCombinations) { $arglist+='--all-combinations' }
& python @arglist
if ($LASTEXITCODE -ne 0) { throw 'Benchmark failed. Read error output above.' }
Write-Host '========================================' -ForegroundColor Green
Write-Host "DONE! Results: $PSScriptRoot\results\taxi" -ForegroundColor Green
Write-Host '========================================' -ForegroundColor Green
