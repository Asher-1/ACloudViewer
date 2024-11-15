param (
    [Parameter(Mandatory=$true, Position=0)]
    [string]$PythonVersion,
    [Parameter(Mandatory=$false, Position=1)]
    [PSDefaultValue(Value="C:\dev\cloudViewer_install")]
    [string]$ACloudViewerInstall,
    [Parameter(Mandatory=$false, Position=2)]
    [PSDefaultValue(Value="C:\Users\asher\develop\code\CloudViewer\CloudViewer-ML")]
    [string]$CloudViewerMLRoot
)

$ErrorActionPreference = "Stop"

$env:PYTHON_VERSION = $PythonVersion
$env:BUILD_CUDA_MODULE = "ON"
$env:BUILD_PYTORCH_OPS = "ON"
$env:BUILD_TENSORFLOW_OPS = "OFF"
$env:BUILD_JUPYTER_EXTENSION = "ON"
$env:BUILD_AZURE_KINECT = "ON"
$env:BUILD_LIBREALSENSE = "ON"
$env:ENV_NAME = "python$env:PYTHON_VERSION"

$env:NPROC = (Get-CimInstance Win32_ComputerSystem).NumberOfLogicalProcessors
Write-Host "ACloudViewer_INSTALL: $env:ACloudViewer_INSTALL"
Write-Host "ENV_NAME: $env:ENV_NAME"
Write-Host "nproc = $env:NPROC"

$env:CLOUDVIEWER_ML_ROOT = [System.IO.Path]::GetFullPath("$CloudViewerMLRoot")
if (Test-Path "$env:CLOUDVIEWER_ML_ROOT") {
    Write-Host "CLOUDVIEWER_ML_ROOT: $env:CLOUDVIEWER_ML_ROOT"
} else {
    Write-Host "Invalid CLOUDVIEWER_ML_ROOT path: $env:CLOUDVIEWER_ML_ROOT"
    exit 1
}
$env:ACloudViewer_INSTALL = [System.IO.Path]::GetFullPath("$ACloudViewerInstall")

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
conda config --set always_yes yes
$existingEnv = conda env list | Select-String "^$env:ENV_NAME\s"
if ($existingEnv) {
    Write-Host "env $env:ENV_NAME exists and start to remove..."
    conda env remove -n $env:ENV_NAME
}

Write-Host "conda env create and activate..."
$env:CONDA_PREFIX = Join-Path $env:CONDA_ROOT "envs\$env:ENV_NAME"

Copy-Item (Join-Path $CLOUDVIEWER_SOURCE_ROOT ".ci\conda_windows.yml") -Destination "$env:TEMP\conda_windows.yml"
(Get-Content "$env:TEMP\conda_windows.yml") -replace "3.8", $env:PYTHON_VERSION | Set-Content "$env:TEMP\conda_windows.yml"

conda env create -f "$env:TEMP\conda_windows.yml"
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

# deploy yarn with npm
Write-Host "Start to deploy yarn"
node --version
npm --version
npm install -g yarn
yarn --version

$env:CONDA_LIB_DIR = "$env:CONDA_PREFIX\Library"
$env:PATH = "$env:CONDA_PREFIX\Library;$env:CONDA_PREFIX\Library\cmake;$env:PATH"

. (Join-Path $CLOUDVIEWER_SOURCE_ROOT "util\ci_utils.ps1")

Write-Host "echo Start to build GUI package on Windows..."
$env:CLOUDVIEWER_SOURCE_ROOT = Split-Path -Parent $PSScriptRoot
. "$env:CLOUDVIEWER_SOURCE_ROOT\util\ci_utils.ps1"

Write-Host "Start to install python dependencies package On Windows..."
$install_options = @("with-unit-test","purge-cache")
if ($env:BUILD_PYTORCH_OPS -eq "ON") {
    $install_options += "with-torch"
}
if ($env:BUILD_TENSORFLOW_OPS -eq "ON") {
    $install_options += "with-tensorflow"
}
if ($env:BUILD_JUPYTER_EXTENSION -eq "ON") {
    $install_options += "with-jupyter"
}
Write-Host "Install options: $install_options"
Install-PythonDependencies -options $install_options

$build_options = @("with_conda")
if ($env:BUILD_CUDA_MODULE -eq "ON") {
    $build_options += "with_cuda"
}
if ($env:BUILD_PYTORCH_OPS -eq "ON") {
    $build_options += "with_torch"
}
if ($env:BUILD_TENSORFLOW_OPS -eq "ON") {
    $build_options += "with_tensorflow"
}
if ($env:BUILD_JUPYTER_EXTENSION -eq "ON") {
    $build_options += "build_jupyter"
}
if ($env:BUILD_AZURE_KINECT -eq "ON") {
    $build_options += "build_azure_kinect"
}
if ($env:BUILD_LIBREALSENSE -eq "ON") {
    $build_options += "build_realsense"
}
Write-Host "Build options: $build_options"
Build-PipPackage -options $build_options

Push-Location build  # PWD=ACloudViewer/build
Write-Host "Try importing cloudViewer Python package"
if ($env:BUILD_CUDA_MODULE -eq "ON") {
    $wheel_file = (Get-Item "lib/python_package/pip_package/cloudViewer-*.whl").FullName
    Write-Host "Test with cuda version: $wheel_file"
} else {
    $wheel_file = (Get-Item "lib/python_package/pip_package/cloudViewer_cpu*.whl").FullName
    Write-Host "Test with cpu version: $wheel_file"
}
$test_options = @()
if ($env:BUILD_CUDA_MODULE -eq "ON") {
    $test_options += "with_cuda"
}
if ($env:BUILD_PYTORCH_OPS -eq "ON") {
    $test_options += "with_torch"
}
if ($env:BUILD_TENSORFLOW_OPS -eq "ON") {
    $test_options += "with_tensorflow"
}
Write-Host "Wheel_file path: $wheel_file"
Write-Host "Test options: $test_options"
Test-Wheel -wheel_path $wheel_file -options $test_options
Pop-Location  # PWD=ACloudViewer

Write-Host "Move to install path: $env:ACloudViewer_INSTALL"

Move-Item -Path "build/lib/python_package/pip_package/cloudViewer*.whl" -Destination $env:ACloudViewer_INSTALL -Force

Write-Host "Backup whl package to $env:ACloudViewer_INSTALL"
Write-Host "LASTEXITCODE: $LASTEXITCODE"
# must do this at the end
Write-Output "BUILD_COMPLETE"
if ($LASTEXITCODE -eq $null) {
    exit 0  # success
} else {
    exit $LASTEXITCODE  # return error code if error
}
