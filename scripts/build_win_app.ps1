param (
    [Parameter(Mandatory=$true, Position=0)]
    [string]$PythonVersion,
    [Parameter(Mandatory=$false, Position=1)]
    [PSDefaultValue(Value="C:\dev\cloudViewer_install")]
    [string]$ACloudViewerInstall
)

$ErrorActionPreference = "Stop"

# Set code page to UTF-8 (65001) to avoid MSVC compilation errors with Unicode characters
# This is especially important for Chinese Windows systems that default to GBK (936)
# GitHub Actions uses UTF-8 code page by default, which is why the issue doesn't occur there
$OutputEncoding = [System.Text.Encoding]::UTF8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
[Console]::InputEncoding = [System.Text.Encoding]::UTF8
chcp 65001 | Out-Null
Write-Host "Code page set to UTF-8 (65001)" -ForegroundColor Green

$env:PYTHON_VERSION = $PythonVersion
$env:ENV_NAME = "cloudViewer"
$env:CLOUDVIEWER_INSTALL_DIR = [System.IO.Path]::GetFullPath("$ACloudViewerInstall")
$env:NPROC = (Get-CimInstance Win32_ComputerSystem).NumberOfLogicalProcessors
Write-Host "CLOUDVIEWER_INSTALL_DIR: $env:CLOUDVIEWER_INSTALL_DIR"
Write-Host "ENV_NAME: $env:ENV_NAME"
Write-Host "nproc = $env:NPROC"

# setting env
$env:STATIC_RUNTIME = if (-not [string]::IsNullOrEmpty($env:STATIC_RUNTIME)) { $env:STATIC_RUNTIME } else { "OFF" }
$env:DEVELOPER_BUILD = if (-not [string]::IsNullOrEmpty($env:DEVELOPER_BUILD)) { $env:DEVELOPER_BUILD } else { "OFF" }
$env:BUILD_SHARED_LIBS = if (-not [string]::IsNullOrEmpty($env:BUILD_SHARED_LIBS)) { $env:BUILD_SHARED_LIBS } else { "OFF" }
if (-not [string]::IsNullOrEmpty($env:BUILD_CUDA_MODULE)) {
    $env:BUILD_CUDA_MODULE = $env:BUILD_CUDA_MODULE
} else {
    $cudaPath = [System.Environment]::GetEnvironmentVariable("CUDA_PATH")
    if (-not [string]::IsNullOrEmpty($cudaPath)) {
        Write-Output "CUDA toolkits path: $cudaPath"
        try {
            $nvccVersion = & nvcc --version
            Write-Output "nvcc version: $nvccVersion"
            $env:BUILD_CUDA_MODULE = "ON"
        } catch {
            Write-Output "Cannot find nvcc."
            $env:BUILD_CUDA_MODULE = "OFF"
        }
    } else {
        Write-Output "CUDA toolkits not found."
        $env:BUILD_CUDA_MODULE = "OFF"
    }
}

$env:CLOUDVIEWER_SOURCE_ROOT = (Get-Location).Path

if ($env:CONDA_EXE) {
    $env:CONDA_ROOT = (Get-Item (Split-Path -Parent (Split-Path -Parent $env:CONDA_EXE))).FullName
} elseif ($env:CONDA_PREFIX) {
    $env:CONDA_ROOT = (Get-Item (Split-Path -Parent $env:CONDA_PREFIX)).FullName
} else {
    Write-Host "Failed to find Miniconda3 install path..."
    exit 1
}

Write-Host "Initializing conda..."
(& $env:CONDA_EXE "shell.powershell" "hook") | Out-String | Invoke-Expression
$existingEnv = conda env list | Select-String "^$env:ENV_NAME\s"
conda config --set always_yes yes
if ($existingEnv) {
    Write-Host "env $env:ENV_NAME exists and start to remove..."
    conda env remove -n $env:ENV_NAME
}

Write-Host "conda env create and activate..."
$env:CONDA_PREFIX = Join-Path $env:CONDA_ROOT "envs\$env:ENV_NAME"

Copy-Item (Join-Path $env:CLOUDVIEWER_SOURCE_ROOT ".ci\conda_windows_cloudViewer.yml") -Destination "$env:TEMP\conda_windows_cloudViewer.yml"
(Get-Content "$env:TEMP\conda_windows_cloudViewer.yml") -replace "3.8", $env:PYTHON_VERSION | Set-Content "$env:TEMP\conda_windows_cloudViewer.yml"

conda env create -f "$env:TEMP\conda_windows_cloudViewer.yml"
conda activate $env:ENV_NAME

$pythonPath = Get-Command python | Select-Object -ExpandProperty Source
$pythonVersion = python --version
if ($LASTEXITCODE -eq 0) {
    Write-Host "env $env:ENV_NAME activated successfully"
    Write-Host "current Python path: $pythonPath"
    Write-Host "current Python version: $pythonVersion"
} else {
    Write-Host "Activate failed, please run manually: conda activate $env:ENV_NAME"
    exit 1
}

if (-not $env:CONDA_PREFIX) {
    Write-Host "Conda env is not activated"
    exit 1
} else {
    Write-Host "Conda env now is $env:CONDA_PREFIX"
}

$env:CONDA_LIB_DIR = "$env:CONDA_PREFIX\Library"
$env:PATH = "$env:CONDA_PREFIX\Library;$env:CONDA_PREFIX\Library\cmake;$env:PATH"

Write-Host "echo Start to build GUI package on Windows..."
. (Join-Path $env:CLOUDVIEWER_SOURCE_ROOT "util\ci_utils.ps1")

Write-Host "Start to Build cpu only GUI On Windows..."
Build-GuiApp -options "with_conda","with_pcl_nurbs","package_installer"
if ($env:BUILD_CUDA_MODULE -eq "ON") {
    Write-Host "Start to Build cuda version GUI On Windows..."
    Build-GuiApp -options "with_cuda","with_conda","with_pcl_nurbs","package_installer"
}

Write-Host "Backup whl package to $env:CLOUDVIEWER_INSTALL_DIR"
Write-Host "LASTEXITCODE: $LASTEXITCODE"
# must do this at the end
Write-Output "BUILD_COMPLETE"
if ($LASTEXITCODE -eq $null) {
    exit 0  # success
} else {
    exit $LASTEXITCODE  # return error code if error
}
