param (
    [Parameter(Mandatory=$true, Position=0)]
    [string]$SourceFolder,
    [Parameter(Mandatory=$true, Position=1)]
    [string]$OutputFolder,
    [Parameter(Mandatory=$true, Position=2)]
    [string[]]$DependencySearchPaths,

    [Parameter(Mandatory=$false)]
    [switch]$Recursive
)

Write-Host "Source Exe Path: $SourceFolder"
Write-Host "Deploy Path: $OutputFolder"
Write-Host "Search Paths: $($DependencySearchPaths -join ', ')"


# ensure output folder exists
New-Item -ItemType Directory -Force -Path $OutputFolder | Out-Null

function Should-Filter {
    param (
        [string]$FileName
    )
    $filterList = @(
        "cu*.dll",
        "npp*.dll",
        "nvrtc*.dll",
        "cudnn*.dll",
        "cublas*.dll",
        "cufft*.dll",
        "curand*.dll",
        "cusolver*.dll",
        "cusparse*.dll"
    )
    
    foreach ($filter in $filterList) {
        if ($FileName -like $filter) {
            return $true
        }
    }
    return $false
}

function Get-Dependencies {
    param (
        [string]$FilePath
    )
    $deps = @()
    $output = dumpbin /dependents $FilePath 2>$null | Select-String "\.dll$"
    foreach ($line in $output) {
        $dep = $line.ToString().Trim()
        if (-not (Should-Filter $dep)) {
            $deps += $dep
        }
    }
    return $deps
}

# define a hashtable to store processed files
$processedFiles = @{}

function Process-File {
    param (
        [string]$FilePath
    )
    
    # processed, skip
    if ($processedFiles.ContainsKey($FilePath)) {
        return
    }
    
    # tagging file as processed
    $processedFiles[$FilePath] = $true
    
    Write-Host "Process lib: $FilePath"
    
    $dependencies = Get-Dependencies $FilePath
    
    foreach ($dep in $dependencies) {
        $depPath = $null
        
        # first search deps in source folder
        $sourceDepPath = Join-Path $SourceFolder $dep
        if (Test-Path $sourceDepPath) {
            $depPath = $sourceDepPath
        } else { # then search in all provided search paths
            foreach ($searchPath in $DependencySearchPaths) {
                $searchDepPath = Join-Path $searchPath $dep
                if (Test-Path $searchDepPath) {
                    $depPath = $searchDepPath
                    break
                }
            }
            # if not found in search paths, try system path
            if (-not $depPath) {
                $depPath = (Get-Command $dep -ErrorAction SilentlyContinue).Source
            }
        }
        
        if ($depPath) {
            # copy dep to given output folder
            $targetPath = Join-Path $OutputFolder $dep
            if (-not (Test-Path $targetPath)) {
                Copy-Item $depPath $targetPath -Force
                Write-Host "Copy: $depPath to $targetPath"
                
                # No need recursive dependency
                if ($Recursive) {
                    Process-File $depPath
                }
            } else {
                 Write-Host "Ignore: $dep, due to already exists $targetPath"
            }
        } else {
            Write-Warning "Cannot find dep: $dep"
        }
    }
}

# obtain all DLL and exe files
$files = Get-ChildItem -Path $SourceFolder -Recurse -Include *.dll,*.exe,*.pyd

# process every file
foreach ($file in $files) {
    Process-File $file.FullName
}

Write-Host "Finish deploy $OutputFolder"