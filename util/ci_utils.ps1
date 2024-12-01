# Set-ExecutionPolicy RemoteSigned -Scope CurrentUser

# if cmd: powershell -ExecutionPolicy Bypass -File "utils\ci_utils.ps1"

$env:GENERATOR = "Visual Studio 16 2019"
$env:ARCHITECTURE = "x64"
$env:STATIC_RUNTIME = "OFF"
$env:DEVELOPER_BUILD = "OFF"
$env:BUILD_SHARED_LIBS = "OFF"
$env:PIP_DISABLE_PIP_VERSION_CHECK = "1"
$env:NPROC = (Get-CimInstance -ClassName Win32_ComputerSystem).NumberOfLogicalProcessors

$env:BUILD_RIEGL = "ON"
$env:CONDA_LIB_DIR = "$env:CONDA_PREFIX\Library"
$env:CLOUDVIEWER_INSTALL_DIR = "C:\dev\cloudViewer_install"

if ($env:CONDA_PREFIX) {
    Write-Host "Conda env: $env:CONDA_PREFIX is activated."
}
else {
    Write-Host "Conda env is not activated!"
    exit 1
}

# Dependency versions:
# CUDA: see docker/docker_build.sh
# ML
# $TENSORFLOW_VER = "2.13.0"
# $TORCH_VER = "2.0.1"
$TENSORFLOW_VER="2.16.2"
$TORCH_VER="2.2.2"
$TORCH_REPO_URL = "https://download.pytorch.org/whl/torch/"
$PIP_VER = "23.2.1"
$WHEEL_VER = "0.38.4"   
$STOOLS_VER = "67.3.2"
$YAPF_VER = "0.30.0"
$PROTOBUF_VER = "4.24.0"

$CLOUDVIEWER_SOURCE_ROOT = (Get-Location).Path

$MAX_RETRIES = 3
$RETRY_DELAY = 5 # seconds
function Install-Requirements {
    param (
        [Parameter(Mandatory=$true)]
        [string]$requirementsFile,
        [Parameter(Mandatory=$false)]
        [switch]$ForceUpdate,
        [Parameter(Mandatory=$false)]
        [string]$speedCmd = ""
    )

    $originalErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    
    if (-not (Test-Path $requirementsFile)) {
        $errorMessage = "Requirements file not found: $($requirementsFile)"

        throw $errorMessage
    }
    
    $retry = 0
    $success = $false
    try {
        while (-not $success -and $retry -lt $MAX_RETRIES) {
            try {
                Write-Host "Attempting to install requirements from $requirementsFile (Attempt $($retry + 1) of $MAX_RETRIES)"
                
                $pipArgs = @()
                if ($ForceUpdate) { $pipArgs += "-U" }
                $pipArgs += @("-r", $requirementsFile)
                if ($speedCmd) { $pipArgs += $speedCmd }
                
                python -m pip install $pipArgs
                $success = $true
                Write-Host "Installation completed successfully."
            }
            catch {
                $retry++
                if ($retry -lt $MAX_RETRIES) {
                    Write-Warning "Installation failed. Retrying in $RETRY_DELAY seconds... Error: $_"
                    Start-Sleep -Seconds $RETRY_DELAY
                } else {
                    Write-Error "Failed to install requirements after $MAX_RETRIES attempts. Last error: $_"
                    return $false
                }
            }
        }
    } finally {
        $ErrorActionPreference = $originalErrorActionPreference
    }
    return $true
}

function Install-PythonDependencies {
    param (
        [string[]]$options
    )

    Write-Host "Installing Python dependencies"
    
    $SPEED_CMD = if ($options -contains "speedup") {
        " -i https://pypi.tuna.tsinghua.edu.cn/simple/ --extra-index-url https://pypi.org/simple --extra-index-url http://mirrors.aliyun.com/pypi/simple/ --trusted-host pypi.tuna.tsinghua.edu.cn --trusted-host mirrors.aliyun.com"
    } else { "" }

    python -m pip install --upgrade pip=="$PIP_VER" wheel=="$WHEEL_VER" setuptools=="$STOOLS_VER" $SPEED_CMD

    if ($options -contains "with-unit-test") {
        Install-Requirements -ForceUpdate "${CLOUDVIEWER_SOURCE_ROOT}/python/requirements_test.txt" $SPEED_CMD
    }

    if ($options -contains "with-cuda") {
        $TF_ARCH_NAME = "tensorflow"
        $TF_ARCH_DISABLE_NAME = "tensorflow-cpu"
        $CUDA_VER = (nvcc --version | Select-String "release ").ToString() -replace '.*release (\d+)\.(\d+).*','$1$2'
        $TORCH_GLNX = "torch==${TORCH_VER}+cu${CUDA_VER}"
    } else {
        if ($IsMacOS) {
            $TF_ARCH_NAME = "tensorflow"
            $TF_ARCH_DISABLE_NAME = "tensorflow"
        } else {
            $TF_ARCH_NAME = "tensorflow-cpu"
            $TF_ARCH_DISABLE_NAME = "tensorflow"
        }
        $TORCH_GLNX = "torch==${TORCH_VER}+cpu"
    }

    Install-Requirements "${CLOUDVIEWER_SOURCE_ROOT}/python/requirements.txt" $SPEED_CMD

    if ($options -contains "with-jupyter") {
        Install-Requirements "${CLOUDVIEWER_SOURCE_ROOT}/python/requirements_jupyter_build.txt" $SPEED_CMD
    }

    if ($options -contains "with-tensorflow") {
        python -m pip uninstall --yes $TF_ARCH_DISABLE_NAME
        python -m pip install -U "${TF_ARCH_NAME}==${TENSORFLOW_VER}" $SPEED_CMD
    }

    if ($options -contains "with-torch") {
        python -m pip install -U $TORCH_GLNX -f $TORCH_REPO_URL tensorboard $SPEED_CMD
    }

    if ($options -contains "with-torch" -or $options -contains "with-tensorflow") {
        python -m pip install -U yapf=="$YAPF_VER" $SPEED_CMD
        # python -m pip install -U protobuf=="$PROTOBUF_VER" $SPEED_CMD
        $output = & { 
            $ErrorActionPreference = 'Continue'
            python -m pip install -U protobuf=="$PROTOBUF_VER" $SPEED_CMD 2>&1
        } | Out-String

        if ($LASTEXITCODE -ne 0) {
            if ($output -match "ERROR:" -and $output -notmatch "pip's dependency resolver does not currently take into account") {
                Write-Error "Install Failed: $output"
            } else {
                Write-Warning "Some warnigs found, but already done: $output"
            }
        } else {
            Write-Host "Install dependency finished!"
        }
    }

    if ($options -contains "purge-cache") {
        Write-Host "Purge pip cache"
        python -m pip cache purge
    }
}

function Build-GuiApp {
    param (
        [Parameter(ValueFromRemainingArguments=$true)]
        [string[]]$Arguments
    )

    Write-Host "Building ACloudViewer gui app"
    $options = $Arguments -join "|"
    Write-Host "Using cmake: $(Get-Command cmake -ErrorAction SilentlyContinue)"
    cmake --version
    Write-Host "Now build GUI package..."
    Write-Host ""

    if ($env:DEVELOPER_BUILD -eq "OFF") {
        Write-Host "Building for a ACloudViewer GUI Release"
    }

    $PACKAGE = if ($options -match "package_installer") {
        Write-Host "Package installer is on"
        "ON"
    } else {
        Write-Host "Package installer is off"
        "OFF"
    }

    $WITH_GDAL = if ($options -match "with_gdal") {
        Write-Host "OPTION_USE_GDAL is on"
        "ON"
    } else {
        Write-Host "OPTION_USE_GDAL is off"
        "OFF"
    }

    $WITH_PCL_NURBS = if ($options -match "with_pcl_nurbs") {
        Write-Host "WITH_PCL_NURBS is on"
        "ON"
    } else {
        Write-Host "WITH_PCL_NURBS is off"
        "OFF"
    }

    $BUILD_WITH_CONDA = if ($options -match "with_conda") {
        Write-Host "BUILD_WITH_CONDA is on"
        "ON"
    } else {
        Write-Host "BUILD_WITH_CONDA is off"
        "OFF"
    }

    $BUILD_CUDA_MODULE = if ($options -match "with_cuda") {
        Write-Host "BUILD_CUDA_MODULE is on"
        "ON"
    } else {
        Write-Host "BUILD_CUDA_MODULE is off"
        "OFF"
    }

    Write-Host ""
    Write-Host "Start building with ACloudViewer GUI..."
    
    New-Item -ItemType Directory -Force -Path "build"
    Push-Location "build"

    $cmakeGuiOptions = @(
        "-DBUILD_SHARED_LIBS=$env:BUILD_SHARED_LIBS",
        "-DDEVELOPER_BUILD=$env:DEVELOPER_BUILD",
        "-DSTATIC_WINDOWS_RUNTIME=$env:STATIC_RUNTIME",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DBUILD_JUPYTER_EXTENSION=OFF",
        "-DBUILD_LIBREALSENSE=OFF",
        "-DBUILD_AZURE_KINECT=OFF",
        "-DBUILD_PYTORCH_OPS=OFF",
        "-DBUILD_TENSORFLOW_OPS=OFF",
        "-DBUNDLE_CLOUDVIEWER_ML=OFF",
        "-DBUILD_BENCHMARKS=OFF",
        "-DBUILD_WEBRTC=OFF",
        "-DWITH_OPENMP=ON",
        "-DWITH_IPPICV=ON",
        "-DWITH_SIMD=ON",
        "-DWITH_PCL_NURBS=$WITH_PCL_NURBS",
        "-DUSE_SIMD=ON",
        "-DPACKAGE=$PACKAGE",
        "-DBUILD_OPENCV=ON",
        "-DBUILD_RECONSTRUCTION=ON",
        "-DBUILD_CUDA_MODULE=$BUILD_CUDA_MODULE",
        "-DBUILD_COMMON_CUDA_ARCHS=ON",
        "-DGLIBCXX_USE_CXX11_ABI=ON",
        "-DCVCORELIB_USE_CGAL=ON",
        "-DCVCORELIB_SHARED=ON",
        "-DCVCORELIB_USE_QT_CONCURRENT=ON",
        "-DOPTION_USE_GDAL=$WITH_GDAL",
        "-DOPTION_USE_DXF_LIB=ON",
        "-DPLUGIN_IO_QDRACO=ON",
        "-DPLUGIN_IO_QLAS=ON",
        "-DPLUGIN_IO_QADDITIONAL=ON",
        "-DPLUGIN_IO_QCORE=ON",
        "-DPLUGIN_IO_QCSV_MATRIX=ON",
        "-DPLUGIN_IO_QE57=ON",
        "-DPLUGIN_IO_QMESH=ON",
        "-DPLUGIN_IO_QPDAL=OFF",
        "-DPLUGIN_IO_QPHOTOSCAN=ON",
        "-DPLUGIN_IO_QRDB=$env:BUILD_RIEGL",
        "-DPLUGIN_IO_QRDB_FETCH_DEPENDENCY=$env:BUILD_RIEGL",
        "-DPLUGIN_IO_QFBX=ON",
        "-DPLUGIN_IO_QSTEP=OFF",
        "-DPLUGIN_STANDARD_QCORK=ON",
        "-DPLUGIN_STANDARD_QJSONRPC=ON",
        "-DPLUGIN_STANDARD_QCLOUDLAYERS=ON",
        "-DPLUGIN_STANDARD_MASONRY_QAUTO_SEG=ON",
        "-DPLUGIN_STANDARD_MASONRY_QMANUAL_SEG=ON",
        "-DPLUGIN_STANDARD_QANIMATION=ON",
        "-DQANIMATION_WITH_FFMPEG_SUPPORT=ON",
        "-DPLUGIN_STANDARD_QCANUPO=ON",
        "-DPLUGIN_STANDARD_QCOLORIMETRIC_SEGMENTER=ON",
        "-DPLUGIN_STANDARD_QCOMPASS=ON",
        "-DPLUGIN_STANDARD_QCSF=ON",
        "-DPLUGIN_STANDARD_QFACETS=ON",
        "-DPLUGIN_STANDARD_QHOUGH_NORMALS=ON",
        "-DPLUGIN_STANDARD_QM3C2=ON",
        "-DPLUGIN_STANDARD_QMPLANE=ON",
        "-DPLUGIN_STANDARD_QPCL=ON",
        "-DPLUGIN_STANDARD_QPOISSON_RECON=ON",
        "-DPOISSON_RECON_WITH_OPEN_MP=ON",
        "-DPLUGIN_STANDARD_QRANSAC_SD=ON",
        "-DPLUGIN_STANDARD_QSRA=ON",
        "-DPLUGIN_STANDARD_3DMASC=ON",
        "-DPLUGIN_STANDARD_QTREEISO=ON",
        "-DPLUGIN_STANDARD_QVOXFALL=ON",
        "-DPLUGIN_PYTHON=ON",
        "-DBUILD_WITH_CONDA=$BUILD_WITH_CONDA",
        "-DCONDA_PREFIX=$env:CONDA_PREFIX",
        "-DCMAKE_PREFIX_PATH=$env:CONDA_LIB_DIR",
        "-DCMAKE_INSTALL_PREFIX=$env:CLOUDVIEWER_INSTALL_DIR"
    )

    Write-Host ""
    Write-Host "Running cmake $($cmakeGuiOptions -join ' ') .."
    & cmake -G $env:GENERATOR -A $env:ARCHITECTURE $cmakeGuiOptions ..

    Write-Host ""
    Write-Host "Build & install ACloudViewer..."
    
    & cmake --build . --config Release --verbose --parallel $env:NPROC
    & cmake --install . --config Release --verbose

    Write-Host ""
    Pop-Location
}

function Build-PipPackage {
    param (
        [string[]]$options
    )

    Write-Host "Building CloudViewer wheel"
    
    $BUILD_FILAMENT_FROM_SOURCE = "OFF"
    $REAL_ML_SHELL_PATH = Join-Path $env:CLOUDVIEWER_ML_ROOT "set_cloudViewer_ml_root.sh"
    if (Test-Path "$REAL_ML_SHELL_PATH") {
        Write-Host "CloudViewer-ML available at $env:CLOUDVIEWER_ML_ROOT. Bundling CloudViewer-ML in wheel."
        Push-Location $env:CLOUDVIEWER_ML_ROOT
        $currentBranch = git rev-parse --abbrev-ref HEAD
        if ($currentBranch -ne "main") {
            $mainExists = git show-ref --verify --quiet refs/heads/main
            if ($?) {
                git checkout main 2>$null
            } else {
                git checkout -b main 2>$null
            }
        }
        Pop-Location
        $BUNDLE_CLOUDVIEWER_ML = "ON"
    } else {
        Write-Host "CloudViewer-ML not available."
        $BUNDLE_CLOUDVIEWER_ML = "OFF"
    }

    if ($env:DEVELOPER_BUILD -eq "OFF") {
        Write-Host "Building for a new CloudViewer release"
    }

    $BUILD_AZURE_KINECT = if ($options -contains "build_azure_kinect") {
        Write-Host "Azure Kinect enabled in Python wheel."
        "ON"
    } else {
        Write-Host "Azure Kinect disabled in Python wheel."
        "OFF"
    }

    $BUILD_LIBREALSENSE = if ($options -contains "build_realsense") {
        Write-Host "Realsense enabled in Python wheel."
        "ON"
    } else {
        Write-Host "Realsense disabled in Python wheel."
        "OFF"
    }

    $BUILD_JUPYTER_EXTENSION = if ($options -contains "build_jupyter") {
        Write-Host "Building Jupyter extension in Python wheel."
        "ON"
    } else {
        Write-Host "Jupyter extension disabled in Python wheel."
        "OFF"
    }

    $BUILD_WITH_CONDA = if ($options -contains "with_conda") {
        Write-Host "BUILD_WITH_CONDA is on"
        "ON"
    } else {
        Write-Host "BUILD_WITH_CONDA is off"
        "OFF"
    }

    $BUILD_PYTORCH_OPS = if ($options -contains "with_torch") {
        Write-Host "BUILD_PYTORCH_OPS is on"
        "ON"
    } else {
        Write-Host "BUILD_PYTORCH_OPS is off"
        "OFF"
    }

    $BUILD_TENSORFLOW_OPS = if ($options -contains "with_tensorflow") {
        Write-Host "BUILD_TENSORFLOW_OPS is on"
        "ON"
    } else {
        Write-Host "BUILD_TENSORFLOW_OPS is off"
        "OFF"
    }

    $BUILD_CUDA_MODULE = if ($options -match "with_cuda") {
        Write-Host "BUILD_CUDA_MODULE is on"
        "ON"
    } else {
        Write-Host "BUILD_CUDA_MODULE is off"
        "OFF"
    }

    Write-Host "`nBuilding with CPU only..."
    
    New-Item -ItemType Directory -Force -Path "build"
    Push-Location "build"

    $cmakeOptions = @(
        "-DBUILD_SHARED_LIBS=$env:BUILD_SHARED_LIBS",
        "-DDEVELOPER_BUILD=$env:DEVELOPER_BUILD",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DBUILD_AZURE_KINECT=$BUILD_AZURE_KINECT",
        "-DBUILD_LIBREALSENSE=$BUILD_LIBREALSENSE",
        "-DBUILD_UNIT_TESTS=OFF",
        "-DBUILD_BENCHMARKS=OFF",
        "-DUSE_SIMD=ON",
        "-DWITH_SIMD=ON",
        "-DWITH_OPENMP=ON",
        "-DWITH_IPPICV=ON",
        "-DBUILD_RECONSTRUCTION=ON",
        "-DBUILD_PYTORCH_OPS=$BUILD_PYTORCH_OPS",
        "-DBUILD_TENSORFLOW_OPS=$BUILD_TENSORFLOW_OPS",
        "-DBUNDLE_CLOUDVIEWER_ML=$BUNDLE_CLOUDVIEWER_ML",
        "-DBUILD_JUPYTER_EXTENSION=$BUILD_JUPYTER_EXTENSION",
        "-DBUILD_FILAMENT_FROM_SOURCE=$BUILD_FILAMENT_FROM_SOURCE",
        "-DBUILD_WITH_CONDA=$BUILD_WITH_CONDA",
        "-DCONDA_PREFIX=$env:CONDA_PREFIX",
        "-DCMAKE_PREFIX_PATH=$env:CONDA_LIB_DIR",
        "-DCMAKE_INSTALL_PREFIX=$env:CLOUDVIEWER_INSTALL_DIR"
    )

    Write-Host "Executing cmake command..."
    cmake -G $env:GENERATOR -A $env:ARCHITECTURE -DBUILD_CUDA_MODULE=OFF $cmakeOptions ..

    Write-Host "`nPackaging CloudViewer CPU pip package..."
    cmake --build . --target pip-package --config Release --parallel $env:NPROC
    Write-Host "Finish make pip-package for cpu"

    Move-Item lib/python_package/pip_package/cloudViewer*.whl . -Force

    if ($BUILD_CUDA_MODULE -eq "ON") {
        Write-Host "`nInstalling CUDA versions of TensorFlow and PyTorch..."
        Install-PythonDependencies -options "with-cuda","with-torch","purge-cache"

        Write-Host "`nBuilding with CUDA..."
        $rebuild_list = @(
            "bin",
            "lib/Release/*.lib",
            "lib/_build_config.py",
            # "libs",
            "lib/ml"
        )

        Write-Host "`nRemoving CPU compiled files / folders: $rebuild_list"
        foreach ($item in $rebuild_list) {
            Remove-Item $item -Recurse -Force -ErrorAction SilentlyContinue
        }

        Write-Host "Executing cmake command with CUDA..."
        cmake -DBUILD_CUDA_MODULE=ON `
              -DBUILD_COMMON_CUDA_ARCHS=ON `
              $cmakeOptions ..
    }

    Write-Host "`nPackaging CloudViewer full pip package..."
    cmake --build . --target pip-package --config Release --parallel $env:NPROC
    Write-Host "Finish make CloudViewer full pip package"

    Move-Item cloudViewer*.whl lib/python_package/pip_package/ -Force

    Pop-Location
}

function Test-Wheel {
    param (
        [Parameter(Mandatory=$true, Position=0)]
        [ValidateScript({Test-Path $_})]
        [string]$wheel_path,
        [Parameter(Mandatory=$false, Position=1)]
        [AllowEmptyCollection()]
        [string[]]$options = @()
    )

    python -m venv cloudViewer_test.venv
    & .\cloudViewer_test.venv\Scripts\Activate.ps1

    python -m pip install --upgrade pip==$env:PIP_VER wheel==$env:WHEEL_VER setuptools==$env:STOOLS_VER

    Write-Host "Using python: $(Get-Command python | Select-Object -ExpandProperty Source)"
    python --version
    Write-Host "Using pip: "
    python -m pip --version

    Write-Host "Installing CloudViewer wheel $wheel_path in virtual environment..."
    python -m pip install $wheel_path

    python -W default -c "import cloudViewer; print('Installed:', cloudViewer); print('BUILD_CUDA_MODULE: ', cloudViewer._build_config['BUILD_CUDA_MODULE'])"
    python -W default -c "import cloudViewer; print('CUDA available: ', cloudViewer.core.cuda.is_available())"

    $CLOUDVIEWER_ML_ROOT = [System.IO.Path]::GetFullPath("$env:CLOUDVIEWER_ML_ROOT")
    if ($options -contains "with_torch") {
        # if ($options -contains "with_cuda") {
        #     $CUDA_VER = (nvcc --version | Select-String "release ").ToString() -replace '.*release (\d+)\.(\d+).*','$1$2'
        #     $TORCH_GLNX = "torch==${TORCH_VER}+cu${CUDA_VER}"
        #     python -m pip install -U $TORCH_GLNX -f $TORCH_REPO_URL tensorboard
        # } else {
        #     $Requirements_Path = Join-Path $CLOUDVIEWER_ML_ROOT "requirements-torch.txt"
        # }
        $Requirements_Path = Join-Path $CLOUDVIEWER_ML_ROOT "requirements-torch.txt"
        Install-Requirements $Requirements_Path
        python -W default -c "import cloudViewer.ml.torch; print('PyTorch Ops library loaded:', cloudViewer.ml.torch._loaded)"
    }

    if ($options -contains "with_tensorflow") {
        $Requirements_Path = Join-Path $CLOUDVIEWER_ML_ROOT "requirements-tensorflow.txt"
        Install-Requirements $Requirements_Path
        python -W default -c "import cloudViewer.ml.tf.ops; print('TensorFlow Ops library loaded:', cloudViewer.ml.tf.ops)"
    }

    if ($options -contains "with_torch" -and $options -contains "with_tensorflow") {
        Write-Host "Importing TensorFlow and torch in the reversed order"
        python -W default -c "import tensorflow as tf; import torch; import cloudViewer.ml.torch as o3d"
        Write-Host "Importing TensorFlow and torch in the normal order"
        python -W default -c "import cloudViewer.ml.torch as o3d; import tensorflow as tf; import torch"
    }

    deactivate
}
