$ErrorActionPreference = "Stop"

$env:PYTHON_VERSION = $args[0]
$env:ACloudViewer_INSTALL = "~/cloudViewer_install"
$env:CLOUDVIEWER_ML_ROOT = "C:\Users\asher\develop\code\CloudViewer\CloudViewer-ML"
$env:ENV_NAME = "python$env:PYTHON_VERSION"

$env:NPROC = (Get-CimInstance Win32_ComputerSystem).NumberOfLogicalProcessors
Write-Host "ENV_NAME: $env:ENV_NAME"
Write-Host "nproc = $env:NPROC"

# $CLOUDVIEWER_SOURCE_ROOT = (Get-Location).Path | Split-Path -Parent
$CLOUDVIEWER_SOURCE_ROOT = (Get-Location).Path

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
if ($existingEnv) {
    Write-Host "env $env:ENV_NAME exists and start to remove..."
    conda env remove -n $env:ENV_NAME -y
}

Write-Host "conda env create and activate..."
$env:CONDA_PREFIX = Join-Path $env:CONDA_ROOT "envs\$env:ENV_NAME"

Copy-Item (Join-Path $CLOUDVIEWER_SOURCE_ROOT ".ci\conda_windows.yml") -Destination "$env:TEMP\conda_windows.yml"
(Get-Content "$env:TEMP\conda_windows.yml") -replace "3.8", $env:PYTHON_VERSION | Set-Content "$env:TEMP\conda_windows.yml"

conda env create -f "$env:TEMP\conda_windows.yml"
conda activate $env:ENV_NAME

# deploy yarn with npm
Write-Host "Start deploy yarn"
node --version
npm --version
npm install -g yarn
yarn --version

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

. (Join-Path $CLOUDVIEWER_SOURCE_ROOT "util\ci_utils.ps1")

Write-Host "echo Start to build GUI package on Windows..."
$env:CLOUDVIEWER_SOURCE_ROOT = Split-Path -Parent $PSScriptRoot
. "$env:CLOUDVIEWER_SOURCE_ROOT\util\ci_utils.ps1"

Write-Host "Start to install python dependencies package On Windows..."
Install-PythonDependencies -options "with-jupyter","with-torch","with-unit-test","purge-cache"

# Build-PipPackage -options "with-torch","with_conda","build_azure_kinect","build_realsense","build_jupyter"
Build-PipPackage -options "with_conda","with_cuda","with-torch","build_azure_kinect","build_realsense","build_jupyter"

# Push-Location build  # PWD=ACloudViewer/build

# Write-Host "Try importing cloudViewer Python package"
# Test-Wheel -wheel_path (Get-Item "lib/python_package/pip_package/cloudViewer*.whl").FullName

# Pop-Location  # PWD=ACloudViewer

Write-Host ""
Write-Host "Move to install path: $env:ACloudViewer_INSTALL"

Move-Item -Path "build/lib/python_package/pip_package/cloudViewer*.whl" -Destination $env:ACloudViewer_INSTALL -Force

Write-Host "Backup whl package to $env:ACloudViewer_INSTALL"
Write-Host ""