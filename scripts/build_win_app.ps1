$ErrorActionPreference = "Stop"

$env:PYTHON_VERSION = $args[0]
$env:ACloudViewer_INSTALL = "~/cloudViewer_install"
$env:CLOUDVIEWER_ML_ROOT = "C:\Users\asher\develop\code\CloudViewer\CloudViewer-ML"
$env:ENV_NAME = "cloudViewer"

$env:NPROC = (Get-CimInstance Win32_ComputerSystem).NumberOfLogicalProcessors
Write-Host "ENV_NAME: $env:ENV_NAME"
Write-Host "nproc = $env:NPROC"

$CLOUDVIEWER_SOURCE_ROOT = (Get-Location).Path

if ($env:CONDA_EXE) {
    $env:CONDA_ROOT = (Get-Item (Split-Path -Parent (Split-Path -Parent $env:CONDA_EXE))).FullName
}
elseif ($env:CONDA_PREFIX) {
    $env:CONDA_ROOT = (Get-Item (Split-Path -Parent $env:CONDA_PREFIX)).FullName
}
else {
    Write-Host "Failed to find Miniconda3 install path..."
    exit 1
}

Write-Host "Initializing conda..."
(& $env:CONDA_EXE "shell.powershell" "hook") | Out-String | Invoke-Expression

# $existingEnv = conda env list | Select-String "^$env:ENV_NAME\s"
# if ($existingEnv) {
#     Write-Host "env $env:ENV_NAME exists and start to remove..."
#     conda env remove -n $env:ENV_NAME
# }

Write-Host "conda env create and activate..."
$env:CONDA_PREFIX = Join-Path $env:CONDA_ROOT "envs\$env:ENV_NAME"

Copy-Item (Join-Path $CLOUDVIEWER_SOURCE_ROOT ".ci\conda_windows_cloudViewer.yml") -Destination "$env:TEMP\conda_windows_cloudViewer.yml"
(Get-Content "$env:TEMP\conda_windows_cloudViewer.yml") -replace "3.8", $env:PYTHON_VERSION | Set-Content "$env:TEMP\conda_windows_cloudViewer.yml"

# conda env create -f "$env:TEMP\conda_windows_cloudViewer.yml"
conda activate $env:ENV_NAME

$pythonPath = Get-Command python | Select-Object -ExpandProperty Source
$pythonVersion = python --version
if ($LASTEXITCODE -eq 0) {
    Write-Host "env $env:ENV_NAME activated successfully"
    Write-Host "current Python path: $pythonPath"
    Write-Host "current Python version: $pythonVersion"
}
else {
    Write-Host "Activate failed, please run manually: conda activate $env:ENV_NAME"
    exit 1
}

if (-not $env:CONDA_PREFIX) {
    Write-Host "Conda env is not activated"
    exit 1
}
else {
    Write-Host "Conda env now is $env:CONDA_PREFIX"
}

$env:CONDA_LIB_DIR = "$env:CONDA_PREFIX\Library"
$env:PATH = "$env:CONDA_PREFIX\Library;$env:CONDA_PREFIX\Library\cmake;$env:PATH"

. (Join-Path $CLOUDVIEWER_SOURCE_ROOT "util\ci_utils.ps1")

Write-Host "echo Start to build GUI package on Windows..."
$env:CLOUDVIEWER_SOURCE_ROOT = Split-Path -Parent $PSScriptRoot
. "$env:CLOUDVIEWER_SOURCE_ROOT\util\ci_utils.ps1"

Write-Host "Start to install python dependencies package On Windows..."
# Build-GuiApp -options "with_conda","with_pcl_nurbs","package_installer"
Build-GuiApp -options "with_conda","with_cuda","with_pcl_nurbs","package_installer"
Write-Host "Backup gui package to $env:ACloudViewer_INSTALL"
Write-Host ""