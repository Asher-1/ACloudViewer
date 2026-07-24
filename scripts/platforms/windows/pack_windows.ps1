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


# Ensure dumpbin is available
$global:DumpbinPath = $null

# Strategy 1: Check if dumpbin is already in PATH
try {
    $global:DumpbinPath = (Get-Command dumpbin -ErrorAction SilentlyContinue).Source
    if ($global:DumpbinPath) {
        Write-Host "Found dumpbin in PATH: $global:DumpbinPath"
    }
} catch {}

# Strategy 2: Check environment variables set by VS Developer Command Prompt
if (-not $global:DumpbinPath) {
    $vcToolsDir = $env:VCToolsInstallDir
    if ($vcToolsDir) {
        $testPath = Join-Path $vcToolsDir "bin\Hostx64\x64\dumpbin.exe"
        if (Test-Path $testPath) {
            $global:DumpbinPath = $testPath
            Write-Host "Found dumpbin via VCToolsInstallDir: $global:DumpbinPath"
        }
    }
}

# Strategy 3: Try to locate via Visual Studio installation path
if (-not $global:DumpbinPath) {
    $vsPaths = @(
        "${env:ProgramFiles}\Microsoft Visual Studio\2022\Enterprise",
        "${env:ProgramFiles}\Microsoft Visual Studio\2022\Professional",
        "${env:ProgramFiles}\Microsoft Visual Studio\2022\Community",
        "${env:ProgramFiles(x86)}\Microsoft Visual Studio\2019\Enterprise",
        "${env:ProgramFiles(x86)}\Microsoft Visual Studio\2019\Professional",
        "${env:ProgramFiles(x86)}\Microsoft Visual Studio\2019\Community"
    )
    
    foreach ($vsPath in $vsPaths) {
        if (Test-Path $vsPath) {
            Write-Host "Searching for dumpbin in: $vsPath"
            # Find the latest MSVC version
            $msvcPath = Join-Path $vsPath "VC\Tools\MSVC"
            if (Test-Path $msvcPath) {
                $latestMsvc = Get-ChildItem $msvcPath -Directory | Sort-Object Name -Descending | Select-Object -First 1
                if ($latestMsvc) {
                    $testPath = Join-Path $latestMsvc.FullName "bin\Hostx64\x64\dumpbin.exe"
                    if (Test-Path $testPath) {
                        $global:DumpbinPath = $testPath
                        Write-Host "Found dumpbin at: $global:DumpbinPath"
                        break
                    }
                }
            }
        }
    }
}

# Strategy 4: Use vswhere if available
if (-not $global:DumpbinPath) {
    $vswherePath = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (Test-Path $vswherePath) {
        Write-Host "Using vswhere to locate Visual Studio..."
        $vsPath = & $vswherePath -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
        if ($vsPath) {
            $msvcPath = Join-Path $vsPath "VC\Tools\MSVC"
            if (Test-Path $msvcPath) {
                $latestMsvc = Get-ChildItem $msvcPath -Directory | Sort-Object Name -Descending | Select-Object -First 1
                if ($latestMsvc) {
                    $testPath = Join-Path $latestMsvc.FullName "bin\Hostx64\x64\dumpbin.exe"
                    if (Test-Path $testPath) {
                        $global:DumpbinPath = $testPath
                        Write-Host "Found dumpbin via vswhere: $global:DumpbinPath"
                    }
                }
            }
        }
    }
}

if (-not $global:DumpbinPath) {
    Write-Error "dumpbin.exe not found. Please ensure Visual Studio C++ tools are installed."
    Write-Error "Tried locations:"
    Write-Error "  - PATH environment variable"
    Write-Error "  - VCToolsInstallDir environment variable"
    Write-Error "  - Common Visual Studio installation paths"
    Write-Error "  - vswhere tool"
    exit 1
}

Write-Host "Using dumpbin: $global:DumpbinPath"

# ensure output folder exists
New-Item -ItemType Directory -Force -Path $OutputFolder | Out-Null

# Do NOT bundle NVIDIA CUDA runtime DLLs into the installer by default
# (libcublas/cudart are large and version-locked). GPU features require a
# matching CUDA install on the target machine unless AICore_BUNDLE_CUDA_RUNTIME=ON
# (custom builds; copies into lib/cuda-runtime/ via bundle_cuda_runtime.ps1).
# Linux equivalent: scripts/platforms/linux/pack_ubuntu.sh should_exclude_lib().

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
    $output = & $global:DumpbinPath /dependents $FilePath 2>$null | Select-String "\.dll$"
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