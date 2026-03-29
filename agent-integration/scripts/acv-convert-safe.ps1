<#
.SYNOPSIS
    Safe wrapper for cli-anything-acloudviewer convert on Windows.
    
.DESCRIPTION
    Forces --mode headless to avoid RPC connection timeouts that plague
    Windows environments with zombie processes on port 6001.
    
.EXAMPLE
    .\acv-convert-safe.ps1 input.ply output.pcd
    
.EXAMPLE
    .\acv-convert-safe.ps1 -Input "c:\data\cloud.ply" -Output "c:\data\cloud.pcd" -Json
#>

param(
    [Parameter(Mandatory=$true, Position=0)]
    [string]$Input,
    
    [Parameter(Mandatory=$true, Position=1)]
    [string]$Output,
    
    [Parameter()]
    [switch]$Json,
    
    [Parameter()]
    [int]$Timeout = 300
)

$ErrorActionPreference = "Stop"

# Check if CLI harness is installed
$cliPath = Get-Command "cli-anything-acloudviewer" -ErrorAction SilentlyContinue
if (-not $cliPath) {
    Write-Error "cli-anything-acloudviewer not found. Install with: pip install cli-anything-acloudviewer"
    exit 1
}

# Build command
$args = @("--mode", "headless")
if ($Json) {
    $args += "--json"
}
$args += @("convert", $Input, $Output)

Write-Host "Running: cli-anything-acloudviewer $($args -join ' ')" -ForegroundColor Cyan

# Execute with timeout
$job = Start-Job -ScriptBlock {
    param($args)
    & cli-anything-acloudviewer @args
} -ArgumentList (,$args)

$completed = Wait-Job $job -Timeout $Timeout

if (-not $completed) {
    Write-Warning "Command timed out after $Timeout seconds. Stopping job..."
    Stop-Job $job
    Remove-Job $job
    exit 124  # timeout exit code
}

$result = Receive-Job $job
Remove-Job $job

Write-Output $result

# Verify output exists
if (Test-Path $Output) {
    $size = (Get-Item $Output).Length
    Write-Host "✓ Conversion successful: $Output ($size bytes)" -ForegroundColor Green
    exit 0
} else {
    Write-Error "✗ Conversion failed: output file not created"
    exit 1
}
