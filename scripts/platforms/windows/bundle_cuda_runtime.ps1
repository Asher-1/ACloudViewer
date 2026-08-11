param (
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$GgmlCudaModule,

    [Parameter(Mandatory = $true, Position = 1)]
    [string]$DestDir,

    [Parameter(Mandatory = $false, Position = 2)]
    [string[]]$SearchPaths = @()
)

if (-not (Test-Path $GgmlCudaModule)) {
    Write-Error "ggml CUDA module not found: $GgmlCudaModule"
    exit 1
}

New-Item -ItemType Directory -Force -Path $DestDir | Out-Null

$BundlePatterns = @(
    "cudart64_*.dll",
    "cublas64_*.dll",
    "cublasLt64_*.dll",
    "nvrtc64_*.dll",
    "nvJitLink_*.dll"
)

function Test-ShouldBundle {
    param([string]$FileName)
    foreach ($pattern in $BundlePatterns) {
        if ($FileName -like $pattern) { return $true }
    }
    return $false
}

function Test-DriverDll {
    param([string]$FileName)
    return ($FileName -like "nvcuda.dll")
}

$dumpbin = (Get-Command dumpbin -ErrorAction SilentlyContinue).Source
if (-not $dumpbin) {
    $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (Test-Path $vswhere) {
        $vsPath = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
        if ($vsPath) {
            $msvc = Get-ChildItem (Join-Path $vsPath "VC\Tools\MSVC") -Directory |
                Sort-Object Name -Descending | Select-Object -First 1
            if ($msvc) {
                $candidate = Join-Path $msvc.FullName "bin\Hostx64\x64\dumpbin.exe"
                if (Test-Path $candidate) { $dumpbin = $candidate }
            }
        }
    }
}
if (-not $dumpbin) {
    Write-Error "dumpbin.exe not found; required to inspect $GgmlCudaModule"
    exit 1
}

$copied = @{}

function Copy-RuntimeDll {
    param([string]$DllName)

    if (Test-DriverDll $DllName) { return }
    if (-not (Test-ShouldBundle $DllName)) { return }
    if ($copied.ContainsKey($DllName)) { return }

    $source = $null
    $moduleDir = Split-Path $GgmlCudaModule -Parent
    $local = Join-Path $moduleDir $DllName
    if (Test-Path $local) {
        $source = $local
    } else {
        foreach ($path in $SearchPaths) {
            $candidate = Join-Path $path $DllName
            if (Test-Path $candidate) {
                $source = $candidate
                break
            }
        }
    }
    if (-not $source) {
        Write-Warning "Could not locate CUDA runtime DLL: $DllName"
        return
    }

    $target = Join-Path $DestDir $DllName
    Copy-Item $source $target -Force
    $copied[$DllName] = $true
    Write-Host "Bundling CUDA runtime: $source -> $target"
}

Write-Host "Scanning ggml CUDA module: $GgmlCudaModule"
$deps = & $dumpbin /dependents $GgmlCudaModule 2>$null | Select-String "\.dll$" |
    ForEach-Object { $_.ToString().Trim() }
foreach ($dep in $deps) {
    Copy-RuntimeDll $dep
}

if ($copied.Count -eq 0) {
    Write-Error "No CUDA runtime DLLs were bundled (check CUDA toolkit bin path)"
    exit 1
}

Write-Host "Bundled $($copied.Count) CUDA runtime DLLs into $DestDir"
