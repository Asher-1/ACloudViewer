param (
    [Parameter(Mandatory=$true, Position=0)]
    [string]$SourceFolder,
    [Parameter(Mandatory=$true, Position=1)]
    [string]$OutputFolder,
    [Parameter(Mandatory=$true, Position=2)]
    [string[]]$DependencySearchPaths
)

# Initialize caches
$processedFiles = [System.Collections.Generic.HashSet[string]]::new()
$dumpbinCache = @{}
$filePathCache = @{}
$systemFileCache = @{}

# Pre-cache file paths
Write-Host "Caching file paths..."
foreach ($searchPath in $DependencySearchPaths) {
    Get-ChildItem -Path $searchPath -Include *.dll -Recurse | ForEach-Object {
        $filePathCache[$_.Name] = $_.FullName
    }
}

# Pre-cache system paths
$systemPaths = $env:Path -split ';' | Where-Object { Test-Path $_ }
foreach ($path in $systemPaths) {
    Get-ChildItem -Path $path -Include *.dll -ErrorAction SilentlyContinue | ForEach-Object {
        $systemFileCache[$_.Name] = $_.FullName
    }
}

# Ensure output directory exists
New-Item -ItemType Directory -Force -Path $OutputFolder | Out-Null

# Define function to get dependencies
function Get-Dependencies {
    param (
        [string]$FilePath
    )
    if ($dumpbinCache.ContainsKey($FilePath)) {
        return $dumpbinCache[$FilePath]
    }
    
    $deps = @()
    $output = dumpbin /dependents $FilePath 2>$null | Select-String "\.dll$"
    foreach ($line in $output) {
        $deps += $line.ToString().Trim()
    }
    
    $dumpbinCache[$FilePath] = $deps
    return $deps
}

# Define fast file copy function
function Copy-FileFast {
    param (
        [string]$Source,
        [string]$Destination
    )
    [System.IO.File]::Copy($Source, $Destination, $true)
}

function Process-File {
    param (
        [string]$FilePath
    )
    # Skip if already processed
    if ($processedFiles.Contains($FilePath)) {
        return
    }
    # Mark file as processed
    $processedFiles.Add($FilePath) | Out-Null
    Write-Host "Processing library: $FilePath"
    
    # Get file dependencies
    $dependencies = Get-Dependencies $FilePath
    foreach ($dep in $dependencies) {
        $depPath = $null
        # First search in source folder
        $sourceDepPath = Join-Path $SourceFolder $dep
        if (Test-Path $sourceDepPath) {
            $depPath = $sourceDepPath
        } elseif ($filePathCache.ContainsKey($dep)) {
            $depPath = $filePathCache[$dep]
        } elseif ($systemFileCache.ContainsKey($dep)) {
            $depPath = $systemFileCache[$dep]
        }
        
        if ($depPath) {
            # Copy dependency to specified output folder
            $targetPath = Join-Path $OutputFolder $dep
            if (-not (Test-Path $targetPath)) {
                Copy-FileFast $depPath $targetPath
                Write-Host "Copied: $depPath to $targetPath"
                # Process dependencies recursively
                # Process-File $depPath
            } else {
                Write-Host "Skipped: $dep, already exists at $targetPath"
            }
        } else {
            Write-Warning "Cannot find dependency: $dep"
        }
    }
}

# Get all DLL and EXE files
$files = Get-ChildItem -Path $SourceFolder -Recurse -Include *.dll,*.exe

# Set maximum concurrent jobs
$MaxConcurrentJobs = [int]$env:NUMBER_OF_PROCESSORS

# Process each file
$totalFiles = $files.Count
$processedCount = 0
$jobs = @()

foreach ($file in $files) {
    while ((Get-Job -State Running).Count -ge $MaxConcurrentJobs) {
        Start-Sleep -Milliseconds 100
    }
    
    $processedCount++
    Write-Progress -Activity "Processing Files" -Status "Processing $($file.Name)" `
        -PercentComplete (($processedCount / $totalFiles) * 100)
    
    $jobs += Start-Job -ScriptBlock {
        param($file, $OutputFolder, $SourceFolder, $DependencySearchPaths, $processedFiles, $dumpbinCache, $filePathCache, $systemFileCache)
        . $function:Process-File
        . $function:Get-Dependencies
        . $function:Copy-FileFast
        Process-File $file.FullName
    } -ArgumentList $file, $OutputFolder, $SourceFolder, $DependencySearchPaths, $processedFiles, $dumpbinCache, $filePathCache, $systemFileCache
}

Wait-Job $jobs | Out-Null
Receive-Job $jobs
Remove-Job $jobs

Write-Host "Deployment completed to $OutputFolder"