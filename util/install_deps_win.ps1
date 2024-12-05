
$vcpkgPath = "C:\dev\vcpkg"
$vcpkgExe = "$vcpkgPath\vcpkg.exe"
$buildDir = "build"
$installDir = "C:\dev\dep"
$PCL_VERSION = "1.14.0"
$CGAL_VERSION = "5.4.1"
$FFMPEG_VERSION = "6.1.0"
$XERCESC_VERSION = "3.2.0"
$ZLIB_VERSION = "1.2.0"

$env:VCPKG_MAX_CONCURRENCY = (Get-CimInstance Win32_ComputerSystem).NumberOfLogicalProcessors
Write-Host "VCPKG_MAX_CONCURRENCY: $env:VCPKG_MAX_CONCURRENCY"
# cmake .. -DCMAKE_TOOLCHAIN_FILE=[vcpkg根目录]/scripts/buildsystems/vcpkg.cmake


# Function: check and install vcpkg
function Install-Vcpkg {
    Write-Host "Check vcpkg path: $vcpkgExe"
    
    $exists = Test-Path $vcpkgExe
    Write-Host "vcpkg.exe exist?: $exists"

    if (-not $exists) {
        Write-Host "Start to install vcpkg..."
        
        Write-Host "Check and create parent directory..."
        $parentDir = Split-Path $vcpkgPath -Parent
        if (-not (Test-Path $parentDir)) {
            New-Item -ItemType Directory -Path $parentDir -Force
        }

        Write-Host "Check directory exists and remove it..."
        if (Test-Path $vcpkgPath) {
            Write-Host "Remove existed vcpkg path..."
            Remove-Item -Path $vcpkgPath -Recurse -Force
        }

        Write-Host "clone vcpkg repository..."
        $cloneResult = git clone https://github.com/Microsoft/vcpkg.git $vcpkgPath
        if (-not $?) {
            Write-Error "clone vcpkg failed: $cloneResult"
            return $false
        }

        Write-Host "change dir to vcpkg and run bootstrap..."
        Push-Location $vcpkgPath
        try {
            $bootstrapResult = .\bootstrap-vcpkg.bat
            if (-not $?) { 
                Write-Error "bootstrap-vcpkg.bat run failed: $bootstrapResult"
                throw "bootstrap-vcpkg.bat run failed" 
            }
        }
        catch {
            Write-Error "install vcpkg failed: $_"
            Pop-Location
            return $false
        }
        Pop-Location

        if (-not (Test-Path $vcpkgExe)) {
            Write-Error "vcpkg.exe failed to create"
            return $false
        }
        Write-Host "vcpkg install successfully..."
    } else {
        Write-Host "vcpkg has already installed..."
    }
    return $true
}

function Install-PCL {
    Write-Host "Start install PCL..."
    & $vcpkgExe install pcl[*]:@$PCL_VERSION --triplet=x64-windows
    if (-not $?) {
        throw "PCL install failed!"
    }
}

function Install-Dependency {
    Write-Host "Start install dependence..."
    & $vcpkgExe install cgal:x64-windows[$CGAL_VERSION]
    & $vcpkgExe install ffmpeg:@$FFMPEG_VERSION --triplet=x64-windows
    & $vcpkgExe install xerces-c:@$XERCESC_VERSION --triplet=x64-windows
    & $vcpkgExe install zlib:@$ZLIB_VERSION --triplet=x64-windows
    & $vcpkgExe install laszip --triplet=x64-windows
}


if (Install-Vcpkg) {
    Install-Dependency
    Install-PCL
    Write-Host "Project build and installation completed successfully!"
} else {
    Write-Error "vcpkg installation failed, script terminated"
    exit 1
}