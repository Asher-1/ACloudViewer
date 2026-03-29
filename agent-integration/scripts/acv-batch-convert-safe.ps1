<#
.SYNOPSIS
    Safe wrapper for cli-anything-acloudviewer batch-convert on Windows.
    
.DESCRIPTION
    Forces --mode headless to avoid RPC connection timeouts.
    
.EXAMPLE
    .\acv-batch-convert-safe.ps1 -InputDir .\scans -OutputDir .\converted -Format .pcd
    
.EXAMPLE
    .\acv-batch-convert-safe.ps1 -InputDir "c:\data\raw" -OutputDir "c:\data\processed" -Format .ply -FilterExt @(".obj", ".stl")
#>

param(
    [Parameter(Mandatory=$true)]
    [string]$InputDir,
    
    [Parameter(Mandatory=$true)]
    [string]$OutputDir,
    
    [Parameter(Mandatory=$true)]
    [string]$Format,
    
    [Parameter()]
    [string[]]$FilterExt,
    
    [Parameter()]
    [switch]$Json
)

$ErrorActionPreference = "Stop"

# Check if CLI harness is installed
$cliPath = Get-Command "cli-anything-acloudviewer" -ErrorAction SilentlyContinue
if (-not $cliPath) {
    Write-Error "cli-anything-acloudviewer not found. Install with: pip install cli-anything-acloudviewer"
    exit 1
}

# Validate directories
if (-not (Test-Path $InputDir)) {
    Write-Error "Input directory does not exist: $InputDir"
    exit 1
}

# Create output directory if needed
if (-not (Test-Path $OutputDir)) {
    New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
    Write-Host "Created output directory: $OutputDir" -ForegroundColor Yellow
}

# Build command
$args = @("--mode", "headless")
if ($Json) {
    $args += "--json"
}
$args += @("batch-convert", $InputDir, $OutputDir, "--format", $Format)

if ($FilterExt) {
    $args += "--filter-ext"
    $args += $FilterExt
}

Write-Host "Running: cli-anything-acloudviewer $($args -join ' ')" -ForegroundColor Cyan

# Execute
& cli-anything-acloudviewer @args

if ($LASTEXITCODE -eq 0) {
    $count = (Get-ChildItem $OutputDir -Filter "*$Format").Count
    Write-Host "✓ Batch conversion complete: $count files in $OutputDir" -ForegroundColor Green
} else {
    Write-Error "✗ Batch conversion failed with exit code $LASTEXITCODE"
    exit $LASTEXITCODE
}
