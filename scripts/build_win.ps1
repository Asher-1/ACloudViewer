$ErrorActionPreference = "Stop"

$ACloudViewer_INSTALL = Join-Path "C:\dev" "cloudViewer_install"
Write-Host "ACloudViewer_INSTALL PATH: $ACloudViewer_INSTALL"
$CLOUDVIEWER_SOURCE_ROOT = (Split-Path -Parent $PSScriptRoot)
$CLOUDVIEWER_BUILD_DIR = Join-Path $CLOUDVIEWER_SOURCE_ROOT "build"
Write-Host "CLOUDVIEWER_BUILD_DIR: $CLOUDVIEWER_BUILD_DIR"
$WIN_APP_BUILD_SHELL = Join-Path $CLOUDVIEWER_SOURCE_ROOT "scripts\build_win_app.ps1"
$WIN_WHL_BUILD_SHELL = Join-Path $CLOUDVIEWER_SOURCE_ROOT "scripts\build_win_wheel.ps1"


function Stop-OccupyingProcesses {
    param (
        [Parameter(Mandatory = $true)]
        [string]$DirectoryPath,
        [int]$WaitSeconds = 10,
        [int]$MaxAttempts = 3
    )

    try {
        Write-Host "Checking for file locks in directory: $DirectoryPath..."

        Get-Process | Where-Object {
            $_.Name -eq "devenv" -or 
            $_.Name -eq "msbuild" -or 
            $_.Name -eq "cl" -or
            $_.Name -eq "vbcscompiler" -or
            $_.Name -eq "MSBuild" -or
            $_.Name -eq "VBCSCompiler" -or
            $_.Name -eq "ServiceHub.Host.CLR.x86" -or
            $_.Name -eq "ServiceHub.IdentityHost" -or
            $_.Name -eq "ServiceHub.VSDetouredHost" -or
            $_.Name -eq "ServiceHub.SettingsHost" -or
            $_.Name -eq "ServiceHub.Host.CLR.x64" -or
            $_.Name -eq "ServiceHub.ThreadedWaitDialog"
        } | ForEach-Object {
            Write-Host "Stopping process: $($_.Name) (PID: $($_.Id))"
            Stop-Process -Id $_.Id -Force -ErrorAction SilentlyContinue
        }

        for ($attempt = 1; $attempt -le $MaxAttempts; $attempt++) {
            Write-Host "Attempt $attempt / $MaxAttempts"
            $lockedFiles = @()

            # Get all files in the directory and its subdirectories
            $files = Get-ChildItem -Path $DirectoryPath -Recurse -File -ErrorAction SilentlyContinue
            
            # Check each file for locks
            foreach ($file in $files) {
                try {
                    # Attempt to open the file with exclusive access
                    $stream = [System.IO.File]::Open($file.FullName, 
                        [System.IO.FileMode]::Open, 
                        [System.IO.FileAccess]::ReadWrite, 
                        [System.IO.FileShare]::None)
                    $stream.Close()
                    $stream.Dispose()
                }
                catch [System.IO.IOException] {
                    # If an IOException occurs, the file is locked
                    $lockedFiles += $file.FullName
                }
            }

            # If no locked files are found, exit the function successfully
            if ($lockedFiles.Count -eq 0) {
                Write-Host "All files are unlocked" -ForegroundColor Green
                return $true
            }

            Write-Host "Found $($lockedFiles.Count) locked files. Attempting to release..."

            # Try to release each locked file
            foreach ($lockedFile in $lockedFiles) {
                Write-Host "Attempting to release: $lockedFile"
                # Find processes that have the file open
                $processes = Get-Process | Where-Object {
                    $process = $_
                    try {
                        $process.Modules | Where-Object { $_.FileName -eq $lockedFile }
                    } catch {
                        $false
                    }
                }

                # Attempt to stop each process
                foreach ($process in $processes) {
                    Write-Host "Attempting to end process: $($process.Name) (PID: $($process.Id))"
                    Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
                    # Use taskkill as a more forceful method to end the process
                    taskkill /PID $process.Id /F 2>&1 | Out-Null
                }
            }

            Write-Host "Waiting for $WaitSeconds seconds..."
            Start-Sleep -Seconds $WaitSeconds
        }

        # If we've exhausted all attempts and still have locked files
        Write-Host "Unable to release all file locks" -ForegroundColor Red
        return $false
    }
    catch {
        Write-Host "Error occurred while checking file locks: $($_.Exception.Message)" -ForegroundColor Red
        return $false
    }
}

function Clear-BuildDirectory {
    param (
        [Parameter(Mandatory = $true)]
        [string]$BuildDir
    )

    if (Test-Path -Path $BuildDir) {
        try {
            cmake --build $BuildDir --target clean
            if (Stop-OccupyingProcesses -DirectoryPath $BuildDir) {
                Write-Host "Successfully to stop all processes..."
            } else {
                Write-Host "Unable to stop all processes, ignore clean build" -ForegroundColor Yellow
                return $false
            }

            Write-Host "Try to start clean $BuildDir"
            Remove-Item -Path (Join-Path $BuildDir "*") -Recurse -Force -ErrorAction Stop
            Write-Host "Finish clean $BuildDir" -ForegroundColor Green
            return $true
        }
        catch {
            Write-Host "Failed to clean $BuildDir : $($_.Exception.Message)" -ForegroundColor Red
            return $false
        }
    } else {
        return $true
    }
}

$PackageExists = Get-ChildItem -Path $ACloudViewer_INSTALL -Filter "ACloudViewer*.exe" -ErrorAction SilentlyContinue
if (-not $PackageExists) {
    Write-Host "Start building ACloudViewer app..."
    if (Clear-BuildDirectory -BuildDir $CLOUDVIEWER_BUILD_DIR) {
        Write-Host "Build directory cleaned successfully"
    } else {
        Write-Host "Failed to clean build directory and try only remove CMakeCache.txt from $CLOUDVIEWER_BUILD_DIR"
        Remove-Item -Path (Join-Path $CLOUDVIEWER_BUILD_DIR "CMakeCache.txt") -Force -ErrorAction SilentlyContinue
    }
    
    if (Test-Path $WIN_APP_BUILD_SHELL) {
        try {
            Write-Host "Start to build..."
            Invoke-Expression "& `"$WIN_APP_BUILD_SHELL`""
            # Start-Process powershell -ArgumentList "-File `"$WIN_APP_BUILD_SHELL`"" -Wait -NoNewWindow
            Write-Host "Finish building and packaging..."
        } catch {
            Write-Host "Failed to execute building script: $($_.Exception.Message)" -ForegroundColor Red
            exit 1
        }
    } else {
            Write-Host "Error: Cannot found building script: $WIN_APP_BUILD_SHELL" -ForegroundColor Red
            exit 1
    }
} else {
    Write-Host "Ignore ACloudViewer GUI app building due to have builded before..."
}

Write-Host "`nStart to build wheel for python3.8-3.11 On Windows...`n"

function Build-PythonWheel {
    param (
        [string]$pythonVersion
    )
    
    $wheelPattern = "cloudViewer*cp$($pythonVersion.Replace('.',''))*.whl"
    $wheelExists = Get-ChildItem -Path $ACloudViewer_INSTALL -Filter $wheelPattern -ErrorAction SilentlyContinue
    
    if (-not $wheelExists) {
        Write-Host "Start building cloudViewer wheel for python$pythonVersion..."
        if (Clear-BuildDirectory -BuildDir $CLOUDVIEWER_BUILD_DIR) {
            Write-Host "Build directory cleaned successfully"
        } else {
            Write-Host "Failed to clean build directory and please mannually handle it..."
            exit 1
        }
    

        if (Test-Path $WIN_WHL_BUILD_SHELL) {
            try {
                Write-Host "Start to build..."
                Invoke-Expression "& `"$WIN_WHL_BUILD_SHELL`" `"$pythonVersion`""
                # Start-Process powershell -ArgumentList "-File `"$WIN_WHL_BUILD_SHELL`" `"$pythonVersion`"" -Wait -NoNewWindow
                Write-Host "Finish building and packaging..."
            } catch {
                Write-Host "Failed to execute building script: $($_.Exception.Message)" -ForegroundColor Red
                exit 1
            }
        } else {
                Write-Host "Error: Cannot found building script: $WIN_WHL_BUILD_SHELL" -ForegroundColor Red
                exit 1
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