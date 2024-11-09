$ErrorActionPreference = "Stop"

# 设置环境变量
$env:PYTHON_VERSION = $args[0]
$env:ACloudViewer_INSTALL = "~/cloudViewer_install"
$env:CLOUDVIEWER_ML_ROOT = "C:\Users\asher\develop\code\CloudViewer\CloudViewer-ML"
$env:ENV_NAME = "cloudViewer"

# 获取处理器数量
$env:NPROC = (Get-CimInstance Win32_ComputerSystem).NumberOfLogicalProcessors
Write-Host "ENV_NAME: $env:ENV_NAME"
Write-Host "nproc = $env:NPROC"

# 获取源代码根目录
$CLOUDVIEWER_SOURCE_ROOT = (Get-Location).Path

# 查找 Conda 安装路径
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

# 初始化 conda
Write-Host "Initializing conda..."
(& $env:CONDA_EXE "shell.powershell" "hook") | Out-String | Invoke-Expression

# 检查并删除已存在的环境
# $existingEnv = conda env list | Select-String "^$env:ENV_NAME\s"
# if ($existingEnv) {
#     Write-Host "env $env:ENV_NAME exists and start to remove..."
#     conda env remove -n $env:ENV_NAME
# }

# 创建新环境
Write-Host "conda env create and activate..."
$env:CONDA_PREFIX = Join-Path $env:CONDA_ROOT "envs\$env:ENV_NAME"

# 复制并修改 conda 环境文件
Copy-Item (Join-Path $CLOUDVIEWER_SOURCE_ROOT ".ci\conda_windows_cloudViewer.yml") -Destination "$env:TEMP\conda_windows_cloudViewer.yml"
(Get-Content "$env:TEMP\conda_windows_cloudViewer.yml") -replace "3.8", $env:PYTHON_VERSION | Set-Content "$env:TEMP\conda_windows_cloudViewer.yml"

# 创建并激活环境
# conda env create -f "$env:TEMP\conda_windows_cloudViewer.yml"
conda activate $env:ENV_NAME

# 验证 Python 安装
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

# 检查 conda 环境
if (-not $env:CONDA_PREFIX) {
    Write-Host "Conda env is not activated"
    exit 1
}
else {
    Write-Host "Conda env now is $env:CONDA_PREFIX"
}

# 设置其他环境变量
$env:CONDA_LIB_DIR = "$env:CONDA_PREFIX\Library"
$env:PATH = "$env:CONDA_PREFIX\Library;$env:CONDA_PREFIX\Library\cmake;$env:PATH"

# 导入构建脚本和控制环境变量
. (Join-Path $CLOUDVIEWER_SOURCE_ROOT "util\ci_utils.ps1")

Write-Host "echo Start to build GUI package on Windows..."
$env:CLOUDVIEWER_SOURCE_ROOT = Split-Path -Parent $PSScriptRoot
. "$env:CLOUDVIEWER_SOURCE_ROOT\util\ci_utils.ps1"

Write-Host "Start to install python dependencies package On Windows..."
Build-GuiApp -options "with_conda","with_pcl_nurbs","package_installer"
# Build-GuiApp -options "with_conda","with_cuda","with_pcl_nurbs","package_installer"
Write-Host "Backup gui package to $env:ACloudViewer_INSTALL"
Write-Host ""