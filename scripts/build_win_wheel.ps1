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
$env:ENV_NAME = "python$env:PYTHON_VERSION"
$env:CLOUDVIEWER_INSTALL_DIR = [System.IO.Path]::GetFullPath("$ACloudViewerInstall")
$env:CLOUDVIEWER_ML_ROOT = [System.IO.Path]::GetFullPath("$CloudViewerMLRoot")
if (Test-Path "$env:CLOUDVIEWER_ML_ROOT") {
    Write-Host "CLOUDVIEWER_ML_ROOT: $env:CLOUDVIEWER_ML_ROOT"
} else {
    Write-Host "Invalid CLOUDVIEWER_ML_ROOT path: $env:CLOUDVIEWER_ML_ROOT"
    exit 1
}
$env:NPROC = (Get-CimInstance Win32_ComputerSystem).NumberOfLogicalProcessors
Write-Host "CLOUDVIEWER_INSTALL_DIR: $env:CLOUDVIEWER_INSTALL_DIR"
Write-Host "CLOUDVIEWER_ML_ROOT: $env:CLOUDVIEWER_ML_ROOT"
Write-Host "ENV_NAME: $env:ENV_NAME"
Write-Host "nproc = $env:NPROC"

# setting env
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
$env:IGNORE_TEST = if (-not [string]::IsNullOrEmpty($env:IGNORE_TEST)) { $env:IGNORE_TEST } else { "OFF" }
$env:STATIC_RUNTIME = if (-not [string]::IsNullOrEmpty($env:STATIC_RUNTIME)) { $env:STATIC_RUNTIME } else { "OFF" }
$env:DEVELOPER_BUILD = if (-not [string]::IsNullOrEmpty($env:DEVELOPER_BUILD)) { $env:DEVELOPER_BUILD } else { "OFF" }
$env:BUILD_SHARED_LIBS = if (-not [string]::IsNullOrEmpty($env:BUILD_SHARED_LIBS)) { $env:BUILD_SHARED_LIBS } else { "OFF" }
$env:BUILD_PYTORCH_OPS = if (-not [string]::IsNullOrEmpty($env:BUILD_PYTORCH_OPS)) { $env:BUILD_PYTORCH_OPS } else { "ON" }
$env:BUILD_TENSORFLOW_OPS = if (-not [string]::IsNullOrEmpty($env:BUILD_TENSORFLOW_OPS)) { $env:BUILD_TENSORFLOW_OPS } else { "OFF" }
$env:BUILD_JUPYTER_EXTENSION = if (-not [string]::IsNullOrEmpty($env:BUILD_JUPYTER_EXTENSION)) { $env:BUILD_JUPYTER_EXTENSION } else { "ON" }
$env:BUILD_AZURE_KINECT = if (-not [string]::IsNullOrEmpty($env:BUILD_AZURE_KINECT)) { $env:BUILD_AZURE_KINECT } else { "ON" }
$env:BUILD_LIBREALSENSE = if (-not [string]::IsNullOrEmpty($env:BUILD_LIBREALSENSE)) { $env:BUILD_LIBREALSENSE } else { "ON" }

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
conda config --set always_yes yes
$existingEnv = conda env list | Select-String "^$env:ENV_NAME\s"
if ($existingEnv) {
    Write-Host "env $env:ENV_NAME exists and start to remove..."
    conda env remove -n $env:ENV_NAME
}

Write-Host "conda env create and activate..."
$env:CONDA_PREFIX = Join-Path $env:CONDA_ROOT "envs\$env:ENV_NAME"

Copy-Item (Join-Path $env:CLOUDVIEWER_SOURCE_ROOT ".ci\conda_windows.yml") -Destination "$env:TEMP\conda_windows.yml"
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
if ($env:BUILD_JUPYTER_EXTENSION -eq "ON") {
    Write-Host "BUILD_JUPYTER_EXTENSION=ON and Start to deploy yarn"
    node --version
    npm --version
    npm install -g yarn
    yarn --version
}

$env:CONDA_LIB_DIR = "$env:CONDA_PREFIX\Library"
$env:EIGEN_ROOT_DIR = "$env:CONDA_LIB_DIR\include\eigen3"
$env:PATH = "$env:CONDA_PREFIX\Library;$env:CONDA_PREFIX\Library\cmake;$env:EIGEN_ROOT_DIR;$env:PATH"

Write-Host "echo Start to build GUI package on Windows..."
. (Join-Path $env:CLOUDVIEWER_SOURCE_ROOT "util\ci_utils.ps1")

Write-Host "Start to install python dependencies package On Windows..."
$install_options = @("with-unit-test")
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

if ($env:IGNORE_TEST -ne "ON") {
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
}

Write-Host "Move to install path: $env:CLOUDVIEWER_INSTALL_DIR"
New-Item -ItemType Directory -Force -Path "$env:CLOUDVIEWER_INSTALL_DIR"
Move-Item -Path "build/lib/python_package/pip_package/cloudViewer*.whl" -Destination $env:CLOUDVIEWER_INSTALL_DIR -Force

Write-Host "Backup whl package to $env:CLOUDVIEWER_INSTALL_DIR"
Write-Host "LASTEXITCODE: $LASTEXITCODE"
# must do this at the end
Write-Output "BUILD_COMPLETE"
if ($LASTEXITCODE -eq $null) {
    exit 0  # success
} else {
    exit $LASTEXITCODE  # return error code if error
}
