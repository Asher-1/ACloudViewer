# Silent-install + first-launch smoke for the packaged ACloudViewer Windows
# installer (Qt IFW .exe). Extracted from the windows.yml "Test ACloudViewer
# installer" step; the workflow keeps the pytest suites, this script owns
# install / DLL-path setup / binary detection / STATUS_DLL_NOT_FOUND triage.
#
# Usage (CI and local):
#   powershell -ExecutionPolicy Bypass -File scripts\platforms\windows\test_installer.ps1 `
#       -InstallerDir "<dir with ACloudViewer-*.exe>" [-InstallDir "<target>"]
#   powershell -ExecutionPolicy Bypass -File scripts\platforms\windows\test_installer.ps1 `
#       -Cleanup [-InstallDir "<target>"]
# The resolved binary is exported as $env:ACV_BINARY (in-process and, on
# GitHub Actions, via GITHUB_ENV for the following steps).

param(
    [string]$InstallerDir = "",
    [string]$InstallDir = (Join-Path $env:USERPROFILE "ACloudViewer-Test"),
    [switch]$Cleanup
)

$ErrorActionPreference = 'Stop'

function Export-StepEnv([string]$Name, [string]$Value) {
    # Persist for later workflow steps; a local run just keeps the process env.
    if ($env:GITHUB_ENV) {
        "$Name=$Value" | Out-File -FilePath $env:GITHUB_ENV -Encoding utf8 -Append
    }
    Set-Item -Path "env:$Name" -Value $Value
}

if ($Cleanup) {
    if (Test-Path (Join-Path $InstallDir "maintenancetool.exe")) {
        & (Join-Path $InstallDir "maintenancetool.exe") purge `
            --accept-licenses --accept-messages --confirm-command 2>&1 | Out-Null
    } else {
        Remove-Item -Recurse -Force $InstallDir -ErrorAction SilentlyContinue
    }
    exit 0
}

# === Environment check ===
Write-Host "=== Environment Check ==="
Write-Host "Python: $(python --version)"
Write-Host "Python path: $(Get-Command python | Select-Object -ExpandProperty Source)"
Write-Host "CONDA_PREFIX: $env:CONDA_PREFIX"
if ($env:CUDA_PATH) {
    Write-Host "CUDA_PATH: $env:CUDA_PATH"
}
Write-Host ""

$installer = Get-ChildItem $InstallerDir -Filter "*.exe" -Recurse -ErrorAction SilentlyContinue |
    Select-Object -First 1
if (-not $installer) {
    Write-Error "No .exe installer found in $InstallerDir"
    exit 1
}

Write-Host "Testing $($installer.Name)"
Write-Host "Install dir: $InstallDir"

& $installer.FullName --root $InstallDir `
    --accept-licenses --accept-messages --confirm-command install
if ($LASTEXITCODE -ne 0) {
    Write-Error "Installation failed for $($installer.Name)"
    exit 1
}

# Set up comprehensive DLL search paths.
# Include conda environment paths (same as build environment).
$dllPaths = @(
    $InstallDir,
    (Join-Path $InstallDir "lib"),
    (Join-Path $InstallDir "bin"),
    (Join-Path $InstallDir "plugins"),
    (Join-Path $InstallDir "platforms")
) | Where-Object { Test-Path $_ }

# Add conda library paths if conda is active.
if ($env:CONDA_PREFIX) {
    $condaLibPath = Join-Path $env:CONDA_PREFIX "Library\bin"
    if (Test-Path $condaLibPath) {
        $dllPaths += $condaLibPath
        Write-Host "Added conda library path: $condaLibPath"
    }
}

# Add CUDA paths if CUDA is installed (for CUDA builds).
if ($env:CUDA_PATH) {
    $cudaBinPath = Join-Path $env:CUDA_PATH "bin"
    if (Test-Path $cudaBinPath) {
        $dllPaths += $cudaBinPath
        Write-Host "Added CUDA binary path: $cudaBinPath"
    }
}

$env:PATH = ($dllPaths -join ";") + ";$env:PATH"
Export-StepEnv "PATH" $env:PATH
Write-Host "=== Updated PATH ==="
Write-Host $env:PATH

# Locate binary.
$exeFile = Join-Path $InstallDir "ACloudViewer.exe"
$batFile = Join-Path $InstallDir "ACloudViewer.bat"
if (Test-Path $exeFile) {
    Export-StepEnv "ACV_BINARY" $exeFile
    Write-Host "Using binary: $exeFile"
} elseif (Test-Path $batFile) {
    Export-StepEnv "ACV_BINARY" $batFile
    Write-Host "Using binary: $batFile"
} else {
    Write-Error "Neither ACloudViewer.exe nor .bat found in $InstallDir"
    Get-ChildItem $InstallDir | Format-Table Name
    exit 1
}

# Debug: Test if binary can run at all.
Write-Host "=== Testing binary execution ==="
& $env:ACV_BINARY --version 2>&1 | Write-Host
$binaryTestResult = $LASTEXITCODE
Write-Host "Binary test exit code: $binaryTestResult (0x$($binaryTestResult.ToString('X8')))"

if ($binaryTestResult -eq 3221225781) {
    Write-Error "Binary failed with STATUS_DLL_NOT_FOUND (0xC0000135)"
    Write-Host "This indicates missing DLL dependencies."
    Write-Host "=== Checking for common DLLs in install directory ==="
    $commonDlls = @("Qt5Core.dll", "Qt6Core.dll", "msvcp*.dll", "vcruntime*.dll")
    foreach ($pattern in $commonDlls) {
        $found = Get-ChildItem $InstallDir -Recurse -Filter $pattern -ErrorAction SilentlyContinue
        if ($found) {
            Write-Host "Found: $($found.FullName)"
        } else {
            Write-Host "NOT FOUND: $pattern"
        }
    }
    exit 1
}
# Other non-zero codes are not fatal here: the pytest suites below (workflow)
# exercise the binary far harder and remain the authoritative gate, matching
# the original inline step's semantics.
Write-Host "Installer smoke passed: $($installer.Name) -> $env:ACV_BINARY"
