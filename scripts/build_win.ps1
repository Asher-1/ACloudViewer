$ErrorActionPreference = "Stop"

$ACloudViewer_INSTALL = Join-Path "C:\dev" "cloudViewer_install"
Write-Host "ACloudViewer_INSTALL PATH: $ACloudViewer_INSTALL"
$CLOUDVIEWER_SOURCE_ROOT = (Split-Path -Parent $PSScriptRoot)
$CLOUDVIEWER_BUILD_DIR = Join-Path $CLOUDVIEWER_SOURCE_ROOT "build"
Write-Host "CLOUDVIEWER_BUILD_DIR: $CLOUDVIEWER_BUILD_DIR"
$WIN_APP_BUILD_SHELL = Join-Path $CLOUDVIEWER_SOURCE_ROOT "scripts\build_win_app.ps1"
$WIN_WHL_BUILD_SHELL = Join-Path $CLOUDVIEWER_SOURCE_ROOT "scripts\build_win_wheel.ps1"
$REMOVE_FOLDERS_SHELL = Join-Path $CLOUDVIEWER_SOURCE_ROOT "scripts\platforms\windows\remove_folders.ps1"

if (-not (Test-Path $WIN_APP_BUILD_SHELL)) {
    Write-Warning "Specified shell path does not exist: $WIN_APP_BUILD_SHELL"
    return $false
}

if (-not (Test-Path $WIN_WHL_BUILD_SHELL)) {
    Write-Warning "Specified shell path does not exist: $WIN_WHL_BUILD_SHELL"
    return $false
}

if (-not (Test-Path $REMOVE_FOLDERS_SHELL)) {
    Write-Warning "Specified shell path does not exist: $REMOVE_FOLDERS_SHELL"
    return $false
}

$PackageExists = Get-ChildItem -Path $ACloudViewer_INSTALL -Filter "ACloudViewer*.exe" -ErrorAction SilentlyContinue
if (-not $PackageExists) {
    Write-Host "Start building ACloudViewer app..."
    if (& $REMOVE_FOLDERS_SHELL -FolderPath $CLOUDVIEWER_BUILD_DIR -y) {
        Write-Host "Build directory cleaned successfully"
    } else {
        Write-Host "Failed to clean build directory and please mannually handle it..."
        return $false
    }
    
    if (Test-Path $WIN_APP_BUILD_SHELL) {
        try {
            Write-Host "Start to build..."
            Invoke-Expression "& `"$WIN_APP_BUILD_SHELL`""
            # Start-Process powershell -ArgumentList "-File `"$WIN_APP_BUILD_SHELL`"" -Wait -NoNewWindow
            Write-Host "Finish building and packaging..."
        } catch {
            Write-Host "Failed to execute building script: $($_.Exception.Message)" -ForegroundColor Red
            return $false
        }
    } else {
            Write-Host "Error: Cannot found building script: $WIN_APP_BUILD_SHELL" -ForegroundColor Red
            return $false
    }
} else {
    Write-Host "Ignore ACloudViewer GUI app building due to have builded before..."
}

Write-Host "`nStart to build wheel for python3.8-3.12 On Windows...`n"

function Build-PythonWheel {
    param (
        [string]$pythonVersion
    )
    
    $wheelPattern = "cloudViewer*cp$($pythonVersion.Replace('.',''))*.whl"
    $wheelExists = Get-ChildItem -Path $ACloudViewer_INSTALL -Filter $wheelPattern -ErrorAction SilentlyContinue
    
    if (-not $wheelExists) {
        Write-Host "Start building cloudViewer wheel for python$pythonVersion..."
        if (& $REMOVE_FOLDERS_SHELL -FolderPath $CLOUDVIEWER_BUILD_DIR -y) {
            Write-Host "Build directory cleaned successfully"
        } else {
            Write-Host "Failed to clean build directory and please mannually handle it..."
            return $false
        }
    
        if (Test-Path $WIN_WHL_BUILD_SHELL) {
            try {
                Write-Host "Start to build..."
                Invoke-Expression "& `"$WIN_WHL_BUILD_SHELL`" `"$pythonVersion`""
                # Start-Process powershell -ArgumentList "-File `"$WIN_WHL_BUILD_SHELL`" `"$pythonVersion`"" -Wait -NoNewWindow
                Write-Host "Finish building and packaging..."
            } catch {
                Write-Host "Failed to execute building script: $($_.Exception.Message)" -ForegroundColor Red
                return $false
            }
        } else {
                Write-Host "Error: Cannot found building script: $WIN_WHL_BUILD_SHELL" -ForegroundColor Red
                return $false
        }
    } else {
        Write-Host "Ignore cloudViewer wheel for python$pythonVersion..."
    }
}

# 构建各个Python版本的wheel
Build-PythonWheel -pythonVersion "3.8"
Build-PythonWheel -pythonVersion "3.9"
Build-PythonWheel -pythonVersion "3.10"
Build-PythonWheel -pythonVersion "3.11"
Build-PythonWheel -pythonVersion "3.12"

Write-Host "All install to $ACloudViewer_INSTALL"
return $true