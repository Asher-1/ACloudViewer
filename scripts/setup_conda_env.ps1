# Setup Conda environment variables script
# Usage: .\setup_conda_env.ps1

# Check if CONDA_PREFIX is set
if (-not $env:CONDA_PREFIX) {
    Write-Error "CONDA_PREFIX environment variable is not set. Please activate conda environment first."
    Write-Host "Example: conda activate your_env_name" -ForegroundColor Yellow
    exit 1
}

Write-Host "CONDA_PREFIX: $env:CONDA_PREFIX" -ForegroundColor Green

# Set environment variables
$env:CONDA_LIB_DIR = Join-Path $env:CONDA_PREFIX "Library"
$cmakeDir = Join-Path $env:CONDA_LIB_DIR "cmake"
$libDir = Join-Path $env:CONDA_LIB_DIR "lib"
$pkgConfigDir = Join-Path $libDir "pkgconfig"
$includeDir = Join-Path $env:CONDA_LIB_DIR "include"
$eigenDir = Join-Path $includeDir "eigen3"

# Save current PATH and PKG_CONFIG_PATH to temporary variables
$currentPath = $env:PATH
$currentPkgConfigPath = $env:PKG_CONFIG_PATH

# Set new environment variables (using string concatenation to avoid parsing issues)
$env:PATH = $env:CONDA_LIB_DIR + ";" + $cmakeDir + ";" + $currentPath
if ($currentPkgConfigPath) {
    $env:PKG_CONFIG_PATH = $pkgConfigDir + ";" + $currentPkgConfigPath
} else {
    $env:PKG_CONFIG_PATH = $pkgConfigDir
}
$env:EIGEN_ROOT_DIR = $eigenDir

$env:GENERATOR = "Visual Studio 17 2022"
$env:ARCHITECTURE = "x64"
$env:NPROC = (Get-CimInstance -ClassName Win32_ComputerSystem).NumberOfLogicalProcessors
$env:CLOUDVIEWER_INSTALL_DIR = "C:/dev/cloudViewer_install"
$env:CLOUDVIEWER_ML_ROOT = "C:/Users/asher/develop/code/CloudViewer/CloudViewer-ML"

# Display results
Write-Host "`nEnvironment variables set:" -ForegroundColor Green
Write-Host "  CONDA_LIB_DIR: $env:CONDA_LIB_DIR"
Write-Host "  EIGEN_ROOT_DIR: $env:EIGEN_ROOT_DIR"
Write-Host "  PKG_CONFIG_PATH: $env:PKG_CONFIG_PATH"
Write-Host "`nNote: These settings are only valid for the current PowerShell session." -ForegroundColor Yellow
