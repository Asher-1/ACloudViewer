#!/usr/bin/env bash
# Detect if script is being sourced (for VS Code terminal compatibility)
if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then
    # Script is being sourced - disable strict mode to prevent terminal crash
    _CI_UTILS_SOURCED=1
    set +e  # Don't exit on error when sourced
    set +u  # Don't exit on unset variables when sourced
    set +o pipefail  # Don't exit on pipe failures when sourced
else
    # Script is being executed - enable strict mode
    _CI_UTILS_SOURCED=0
    set -euo pipefail
fi

# The following environment variables are required:
SUDO=${SUDO:=sudo}
UBUNTU_VERSION=${UBUNTU_VERSION:="$(lsb_release -cs 2>/dev/null || true)"} # Empty in macOS

DEVELOPER_BUILD="${DEVELOPER_BUILD:-ON}"
if [[ "$DEVELOPER_BUILD" != "OFF" ]]; then # Validate input coming from GHA input field
    DEVELOPER_BUILD="ON"
fi
BUILD_SHARED_LIBS=${BUILD_SHARED_LIBS:-OFF}
NPROC=${NPROC:-$(getconf _NPROCESSORS_ONLN)} # POSIX: MacOS + Linux
NPROC=$((NPROC + 2))                         # run nproc+2 jobs to speed up the build
if [ -z "${BUILD_CUDA_MODULE:+x}" ]; then
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        BUILD_CUDA_MODULE=ON
    else
        BUILD_CUDA_MODULE=OFF
    fi
fi
BUILD_TENSORFLOW_OPS=${BUILD_TENSORFLOW_OPS:-OFF}
BUILD_PYTORCH_OPS=${BUILD_PYTORCH_OPS:-ON}
LOW_MEM_USAGE=${LOW_MEM_USAGE:-OFF}

# Warning: CONDA_PREFIX variable should be set before
# CONDA_PREFIX=${CONDA_PREFIX:="/root/miniconda3/envs/cloudViewer"}
if [ -z "${CONDA_PREFIX:-}" ] ; then
	echo "Conda env is not activated."
    CONDA_PREFIX=/root/miniconda3/envs/cloudViewer
else
    echo "Conda env: $CONDA_PREFIX is activated."
fi

# Dependency versions:
# CUDA: see docker/docker_build.sh
# ML
TENSORFLOW_VER="2.19.0"
TORCH_VER="2.7.1"
TORCH_REPO_URL="https://download.pytorch.org/whl/torch/"
# Python
PIP_VER="24.3.1"
PROTOBUF_VER="4.25.3"  # Changed from 4.24.0 due to tensorboard 2.19.0 incompatibility

DISTRIB_ID=""
DISTRIB_RELEASE=""
if [[ "$OSTYPE" == "darwin"* ]]; then
    BUILD_RIEGL=OFF
    CONDA_LIB_DIR="$CONDA_PREFIX/lib"
    CLOUDVIEWER_INSTALL_DIR=~/cloudViewer_install

    # use lower target(11.0) version for compacibility
    PROCESSOR_ARCH=$(uname -m)
    if [[ "$PROCESSOR_ARCH" == "arm64" ]]; then
        export MACOSX_DEPLOYMENT_TARGET=11.0
    else
        export MACOSX_DEPLOYMENT_TARGET=10.15
    fi
    echo "Processor Architecture: $PROCESSOR_ARCH"
    echo "MACOSX_DEPLOYMENT_TARGET: $MACOSX_DEPLOYMENT_TARGET"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    BUILD_RIEGL=ON
    CONDA_LIB_DIR="$CONDA_PREFIX/lib"
    CLOUDVIEWER_INSTALL_DIR=/root/install

    eval $(
        source /etc/lsb-release;
        echo DISTRIB_ID="$DISTRIB_ID";
        echo DISTRIB_RELEASE="$DISTRIB_RELEASE"
    )
else # do not support windows
    echo "Do not support windows system with this script!"
    if [[ "${_CI_UTILS_SOURCED:-0}" -eq 1 ]]; then
        # Script is sourced, use return instead of exit
        return 1 2>/dev/null || exit 1
    else
        # Script is executed, exit normally
        exit 1
    fi
fi

CLOUDVIEWER_SOURCE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. >/dev/null 2>&1 && pwd)"

install_python_dependencies() {
    echo "Installing Python dependencies"
    options="$(echo "$@" | tr ' ' '|')"
    if [[ "speedup" =~ ^($options)$ ]]; then
        SPEED_CMD=" --default-timeout=10000000 -i https://pypi.tuna.tsinghua.edu.cn/simple/ --extra-index-url https://pypi.org/simple --extra-index-url http://mirrors.aliyun.com/pypi/simple/ --trusted-host pypi.tuna.tsinghua.edu.cn --trusted-host mirrors.aliyun.com"
        echo "Speed up downloading with cmd: " $SPEED_CMD
    else
        SPEED_CMD=""
    fi

    python -m pip install -U pip=="$PIP_VER" $SPEED_CMD
    python -m pip install -U -r "${CLOUDVIEWER_SOURCE_ROOT}/python/requirements_build.txt" $SPEED_CMD
    if [[ "with-unit-test" =~ ^($options)$ ]]; then
        python -m pip install -U -r python/requirements_test.txt $SPEED_CMD
    fi
    if [[ "with-cuda" =~ ^($options)$ ]]; then
        TF_ARCH_NAME=tensorflow
        TF_ARCH_DISABLE_NAME=tensorflow-cpu
        CUDA_VER=$(nvcc --version | grep "release " | cut -c33-37 | sed 's|[^0-9]||g') # e.g.: 117, 118, 121, ...
        TORCH_GLNX="torch==${TORCH_VER}+cu${CUDA_VER}"
    else
        # tensorflow-cpu wheels for macOS arm64 are not available
        if [[ "$OSTYPE" == "darwin"* ]]; then
            TF_ARCH_NAME=tensorflow
            TF_ARCH_DISABLE_NAME=tensorflow
        else
            TF_ARCH_NAME=tensorflow-cpu
            TF_ARCH_DISABLE_NAME=tensorflow
        fi
        TORCH_GLNX="torch==${TORCH_VER}+cpu"
    fi

    python -m pip install -r "${CLOUDVIEWER_SOURCE_ROOT}/python/requirements.txt" $SPEED_CMD
    if [[ "with-jupyter" =~ ^($options)$ ]]; then
        python -m pip install -r "${CLOUDVIEWER_SOURCE_ROOT}/python/requirements_jupyter_build.txt" $SPEED_CMD
    fi

    echo
    if [ "$BUILD_TENSORFLOW_OPS" == "ON" ]; then
        # TF happily installs both CPU and GPU versions at the same time, so remove the other
        python -m pip uninstall --yes "$TF_ARCH_DISABLE_NAME"
        python -m pip install -U "$TF_ARCH_NAME"=="$TENSORFLOW_VER"  $SPEED_CMD # ML/requirements-tensorflow.txt
    fi
    if [ "$BUILD_PYTORCH_OPS" == "ON" ]; then # ML/requirements-torch.txt
        if [[ "$OSTYPE" == "linux-gnu"* ]]; then
            python -m pip install -U "${TORCH_GLNX}" -f "$TORCH_REPO_URL" $SPEED_CMD
            python -m pip install tensorboard $SPEED_CMD
        elif [[ "$OSTYPE" == "darwin"* ]]; then
            python -m pip install -U torch=="$TORCH_VER" -f "$TORCH_REPO_URL" tensorboard $SPEED_CMD
        else
            echo "unknown OS $OSTYPE"
            exit 1
        fi
    fi
    if [ "$BUILD_TENSORFLOW_OPS" == "ON" ] || [ "$BUILD_PYTORCH_OPS" == "ON" ]; then
        python -m pip install -U -c "${CLOUDVIEWER_SOURCE_ROOT}/python/requirements_build.txt" yapf $SPEED_CMD
        # Fix Protobuf compatibility issue
        # https://stackoverflow.com/a/72493690/1255535
        # https://github.com/protocolbuffers/protobuf/issues/10051
        python -m pip install -U protobuf=="$PROTOBUF_VER" $SPEED_CMD
    fi
    if [[ "purge-cache" =~ ^($options)$ ]]; then
        echo "Purge pip cache"
        python -m pip cache purge 2>/dev/null || true
    fi
}

build_mac_wheel() {
    echo "Building CloudViewer wheel"
    options="$(echo "$@" | tr ' ' '|')"

    echo "Using cmake: $(command -v cmake)"
    cmake --version

    set +u
    if [[ "$DEVELOPER_BUILD" == "OFF" ]]; then
        echo "Building for a new cloudViewer Release"
    fi
    if [[ -f "${CLOUDVIEWER_ML_ROOT}/set_cloudViewer_ml_root.sh" ]] &&
        [[ "$BUILD_TENSORFLOW_OPS" == "ON" || "$BUILD_PYTORCH_OPS" == "ON" ]]; then
        echo "CloudViewer-ML available at ${CLOUDVIEWER_ML_ROOT}. Bundling CloudViewer-ML in wheel."
        # the build system of the main repo expects a main branch. make sure main exists
        git -C "${CLOUDVIEWER_ML_ROOT}" checkout -b main || true
        BUNDLE_CLOUDVIEWER_ML=ON
    else
        echo "CloudViewer-ML not available."
        BUNDLE_CLOUDVIEWER_ML=OFF
    fi
    if [[ "with_conda" =~ ^($options)$ ]]; then
        BUILD_WITH_CONDA=ON
        echo "BUILD_WITH_CONDA is on"
    else
        BUILD_WITH_CONDA=OFF
        CONDA_LIB_DIR=""
        echo "BUILD_WITH_CONDA is off"
    fi
    if [[ "build_realsense" =~ ^($options)$ ]]; then
        echo "Realsense enabled in Python wheel."
        BUILD_LIBREALSENSE=ON
    else
        echo "Realsense disabled in Python wheel."
        BUILD_LIBREALSENSE=OFF
    fi
    set -u

    mkdir -p build
    # echo "Clean last build cache if possible."
    pushd build # PWD=ACloudViewer/build
    cmakeOptions=(
        "-DBUILD_SHARED_LIBS=OFF"
        "-DBUILD_UNIT_TESTS=ON"
        "-DDEVELOPER_BUILD=$DEVELOPER_BUILD"
        "-DCMAKE_BUILD_TYPE=Release"
        "-DBUILD_BENCHMARKS=OFF"
        "-DBUILD_AZURE_KINECT=OFF" # not supported on macos
        "-DBUILD_LIBREALSENSE=$BUILD_LIBREALSENSE" # some issues with network locally
        "-DWITH_OPENMP=ON"
        "-DWITH_IPP=OFF" # not supported on macos
        "-DWITH_SIMD=ON"
        "-DUSE_SIMD=ON"
        "-DCVCORELIB_SHARED=ON"
        "-DCVCORELIB_USE_CGAL=ON" # for delaunay triangulation such as facet
        "-DCVCORELIB_USE_QT_CONCURRENT=ON" # for parallel processing
        "-DUSE_PCL_BACKEND=OFF" # no need pcl for wheel
        "-DBUILD_RECONSTRUCTION=ON"
        "-DBUILD_FILAMENT_FROM_SOURCE=ON"
        "-DBUILD_CUDA_MODULE=$BUILD_CUDA_MODULE"
        "-DBUILD_COMMON_CUDA_ARCHS=ON"
        "-DBUILD_PYTORCH_OPS=$BUILD_PYTORCH_OPS"
        "-DBUILD_TENSORFLOW_OPS=$BUILD_TENSORFLOW_OPS"
        "-DBUNDLE_CLOUDVIEWER_ML=$BUNDLE_CLOUDVIEWER_ML"
        "-DCONDA_PREFIX=$CONDA_PREFIX"
        "-DCMAKE_PREFIX_PATH=$CONDA_LIB_DIR"
        "-DBUILD_WITH_CONDA=$BUILD_WITH_CONDA"
        "-DCMAKE_INSTALL_PREFIX=$CLOUDVIEWER_INSTALL_DIR"
        )

    echo
    echo Running cmake "${cmakeOptions[@]}" ..
    cmake "${cmakeOptions[@]}" ..
    echo
    echo "Build & install CloudViewer..."
    make pip-package -j"$NPROC"
    echo
    popd                                           # PWD=ACloudViewer
}

build_gui_app() {

    echo "Building ACloudViewer gui app"
    options="$(echo "$@" | tr ' ' '|')"
    
    echo "Using cmake: $(command -v cmake)"
    cmake --version

    echo "Now build GUI package..."
    echo
    set +u
    if [[ "$DEVELOPER_BUILD" == "OFF" ]]; then
        echo "Building for a ACloudViewer GUI Release"
    fi
    if [[ "package_installer" =~ ^($options)$ ]]; then
        PACKAGE=ON
        echo "Package installer is on"
    else
        PACKAGE=OFF
        echo "Package installer is off"
    fi
    if [[ "with_gdal" =~ ^($options)$ ]]; then
        WITH_GDAL=ON
        echo "OPTION_USE_GDAL is on"
    else
        WITH_GDAL=OFF
        echo "OPTION_USE_GDAL is off"
    fi
    if [[ "with_pcl_nurbs" =~ ^($options)$ ]]; then
        WITH_PCL_NURBS=ON
        echo "WITH_PCL_NURBS is on"
    else
        WITH_PCL_NURBS=OFF
        echo "WITH_PCL_NURBS is off"
    fi
    if [[ "with_rdb" =~ ^($options)$ ]]; then
        BUILD_RIEGL=ON
        echo "PLUGIN_IO_QRDB is on"
    elif [[ "without_rdb" =~ ^($options)$ ]]; then
        BUILD_RIEGL=OFF
        echo "PLUGIN_IO_QRDB is off"
    else
        # Keep default behavior based on OS if option not specified
        echo "PLUGIN_IO_QRDB uses default: $BUILD_RIEGL (based on OS)"
    fi
    if [[ "plugin_treeiso" =~ ^($options)$ ]]; then
        PLUGIN_STANDARD_QTREEISO=ON
        echo "PLUGIN_STANDARD_QTREEISO is on"
    else
        PLUGIN_STANDARD_QTREEISO=OFF
        echo "PLUGIN_STANDARD_QTREEISO is off"
    fi

    if [[ "with_conda" =~ ^($options)$ ]]; then
        BUILD_WITH_CONDA=ON
        echo "BUILD_WITH_CONDA is on"
    else
        BUILD_WITH_CONDA=OFF
        CONDA_LIB_DIR=""
        echo "BUILD_WITH_CONDA is off"
    fi

    # Check USE_QT6 environment variable (set by build_gui_app.sh or docker)
    if [[ -z "${USE_QT6:-}" ]]; then
        USE_QT6=OFF
        echo "USE_QT6 not set, defaulting to OFF (Qt5)"
    else
        echo "USE_QT6 is set to: $USE_QT6"
    fi
    set -u

    echo
    echo "Start building with ACloudViewer GUI..."
    mkdir -p build
    pushd build # PWD=ACloudViewer/build
    cmakeGuiOptions=("-DBUILD_SHARED_LIBS=OFF"
                "-DBUILD_UNIT_TESTS=ON"
                "-DDEVELOPER_BUILD=$DEVELOPER_BUILD"
                "-DCMAKE_BUILD_TYPE=Release"
                "-DUSE_QT6=$USE_QT6"
                "-DBUILD_JUPYTER_EXTENSION=OFF"
                "-DBUILD_LIBREALSENSE=OFF"
                "-DBUILD_AZURE_KINECT=OFF"
                "-DBUILD_PYTORCH_OPS=OFF"
                "-DBUILD_TENSORFLOW_OPS=OFF"
                "-DBUNDLE_CLOUDVIEWER_ML=OFF"
                "-DBUILD_BENCHMARKS=OFF"
                "-DBUILD_WEBRTC=OFF"
                "-DWITH_OPENMP=ON"
                "-DWITH_IPP=ON"
                "-DWITH_SIMD=ON"
                "-DWITH_PCL_NURBS=$WITH_PCL_NURBS"
                "-DUSE_PCL_BACKEND=ON"
                "-DUSE_SIMD=ON"
                "-DPACKAGE=$PACKAGE"
                "-DBUILD_OPENCV=ON"
                "-DBUILD_RECONSTRUCTION=ON"
                "-DBUILD_CUDA_MODULE=$BUILD_CUDA_MODULE"
                "-DBUILD_COMMON_CUDA_ARCHS=ON"
                "-DCVCORELIB_USE_CGAL=ON"
                "-DCVCORELIB_SHARED=ON"
                "-DCVCORELIB_USE_QT_CONCURRENT=ON"
                "-DOPTION_USE_GDAL=$WITH_GDAL"
                "-DOPTION_USE_DXF_LIB=ON"
                "-DPLUGIN_IO_QDRACO=ON"
                "-DPLUGIN_IO_QLAS=ON"
                "-DPLUGIN_IO_QADDITIONAL=ON"
                "-DPLUGIN_IO_QCORE=ON"
                "-DPLUGIN_IO_QCSV_MATRIX=ON"
                "-DPLUGIN_IO_QE57=ON"
                "-DPLUGIN_IO_QMESH=ON"
                "-DPLUGIN_IO_QPDAL=OFF"
                "-DPLUGIN_IO_QPHOTOSCAN=ON"
                "-DPLUGIN_IO_QRDB=$BUILD_RIEGL"
                "-DPLUGIN_IO_QRDB_FETCH_DEPENDENCY=$BUILD_RIEGL"
                "-DPLUGIN_IO_QFBX=OFF"
                "-DPLUGIN_IO_QSTEP=OFF"
                "-DPLUGIN_STANDARD_QCORK=ON"
                "-DPLUGIN_STANDARD_QJSONRPC=ON"
                "-DPLUGIN_STANDARD_QCLOUDLAYERS=ON"
                "-DPLUGIN_STANDARD_MASONRY_QAUTO_SEG=ON"
                "-DPLUGIN_STANDARD_MASONRY_QMANUAL_SEG=ON"
                "-DPLUGIN_STANDARD_QANIMATION=ON"
                "-DQANIMATION_WITH_FFMPEG_SUPPORT=ON"
                "-DPLUGIN_STANDARD_QCANUPO=ON"
                "-DPLUGIN_STANDARD_QCOLORIMETRIC_SEGMENTER=ON"
                "-DPLUGIN_STANDARD_QCOMPASS=ON"
                "-DPLUGIN_STANDARD_QCSF=ON"
                "-DPLUGIN_STANDARD_QFACETS=ON"
                "-DPLUGIN_STANDARD_QHOUGH_NORMALS=ON"
                "-DPLUGIN_STANDARD_QM3C2=ON"
                "-DPLUGIN_STANDARD_QMPLANE=ON"
                "-DPLUGIN_STANDARD_QPCL=ON"
                "-DPLUGIN_STANDARD_QPOISSON_RECON=ON"
                "-DPOISSON_RECON_WITH_OPEN_MP=ON"
                "-DPLUGIN_STANDARD_QRANSAC_SD=ON"
                "-DPLUGIN_STANDARD_QSRA=ON"
                "-DPLUGIN_STANDARD_3DMASC=ON"
                "-DPLUGIN_STANDARD_QTREEISO=$PLUGIN_STANDARD_QTREEISO"
                "-DPLUGIN_STANDARD_QVOXFALL=ON"
                "-DPLUGIN_STANDARD_G3POINT=ON"
                "-DPLUGIN_PYTHON=ON"
                "-DBUILD_PYTHON_MODULE=ON"
                "-DCONDA_PREFIX=$CONDA_PREFIX"
                "-DCMAKE_PREFIX_PATH=$CONDA_LIB_DIR"
                "-DBUILD_WITH_CONDA=$BUILD_WITH_CONDA"
                "-DCMAKE_INSTALL_PREFIX=$CLOUDVIEWER_INSTALL_DIR"
    )
    
    set -x # Echo commands on
    echo
    echo Running cmake "${cmakeGuiOptions[@]}" ..
    cmake "${cmakeGuiOptions[@]}" ..
    echo
    set +x # Echo commands off

    echo "Build & install ACloudViewer..."
    make -j"$NPROC"
    make install -j"$NPROC"
    ls -hal $CLOUDVIEWER_INSTALL_DIR
    echo
    popd                                           # PWD=ACloudViewer
}

build_pip_package() {
    echo "Building CloudViewer wheel"
    options="$(echo "$@" | tr ' ' '|')"

    if [[ "$OSTYPE" == "darwin"* ]]; then
        BUILD_FILAMENT_FROM_SOURCE=ON
    else
        BUILD_FILAMENT_FROM_SOURCE=OFF
    fi

    set +u
    if [[ -f "${CLOUDVIEWER_ML_ROOT}/set_cloudViewer_ml_root.sh" ]] &&
        [[ "$BUILD_TENSORFLOW_OPS" == "ON" || "$BUILD_PYTORCH_OPS" == "ON" ]]; then
        echo "CloudViewer-ML available at ${CLOUDVIEWER_ML_ROOT}. Bundling CloudViewer-ML in wheel."
        # the build system of the main repo expects a main branch. make sure main exists
        git -C "${CLOUDVIEWER_ML_ROOT}" checkout -b main || true
        BUNDLE_CLOUDVIEWER_ML=ON
    else
        echo "CloudViewer-ML not available."
        BUNDLE_CLOUDVIEWER_ML=OFF
    fi
    if [[ "$DEVELOPER_BUILD" == "OFF" ]]; then
        echo "Building for a new CloudViewer release"
    fi
    if [[ "build_azure_kinect" =~ ^($options)$ ]]; then
        echo "Azure Kinect enabled in Python wheel."
        BUILD_AZURE_KINECT=ON
    else
        echo "Azure Kinect disabled in Python wheel."
        BUILD_AZURE_KINECT=OFFVERBOSE=1 
    fi
    if [[ "build_realsense" =~ ^($options)$ ]]; then
        echo "Realsense enabled in Python wheel."
        BUILD_LIBREALSENSE=ON
    else
        echo "Realsense disabled in Python wheel."
        BUILD_LIBREALSENSE=OFF
    fi
    if [[ "build_jupyter" =~ ^($options)$ ]]; then
        echo "Building Jupyter extension in Python wheel."
        BUILD_JUPYTER_EXTENSION=ON
    else
        echo "Jupyter extension disabled in Python wheel."
        BUILD_JUPYTER_EXTENSION=OFF
    fi
    if [[ "with_conda" =~ ^($options)$ ]]; then
        BUILD_WITH_CONDA=ON
        echo "BUILD_WITH_CONDA is on"
    else
        BUILD_WITH_CONDA=OFF
        CONDA_LIB_DIR=""
        echo "BUILD_WITH_CONDA is off"
    fi

    # Check USE_QT6 environment variable
    if [[ -z "${USE_QT6:-}" ]]; then
        USE_QT6=OFF
        echo "USE_QT6 not set, defaulting to OFF (Qt5)"
    else
        echo "USE_QT6 is set to: $USE_QT6"
    fi
    set -u

    echo
    echo Building with CPU only...
    mkdir -p build
    pushd build # PWD=ACloudViewer/build
    cmakeOptions=("-DBUILD_SHARED_LIBS=OFF"
        "-DDEVELOPER_BUILD=$DEVELOPER_BUILD"
        "-DCMAKE_BUILD_TYPE=Release"
        "-DBUILD_AZURE_KINECT=$BUILD_AZURE_KINECT"
        "-DBUILD_LIBREALSENSE=$BUILD_LIBREALSENSE"
        "-DBUILD_UNIT_TESTS=ON"
        "-DBUILD_BENCHMARKS=OFF"
        "-DUSE_SIMD=ON"
        "-DUSE_QT6=$USE_QT6"
        "-DWITH_SIMD=ON"
        "-DWITH_OPENMP=ON"
        "-DWITH_IPP=ON"
        "-DCVCORELIB_SHARED=ON"
        "-DCVCORELIB_USE_CGAL=ON" # for delaunay triangulation such as facet
        "-DCVCORELIB_USE_QT_CONCURRENT=ON" # for parallel processing
        "-DUSE_PCL_BACKEND=OFF" # no need pcl for wheel
        "-DBUILD_RECONSTRUCTION=ON"
        "-DBUILD_PYTORCH_OPS=$BUILD_PYTORCH_OPS"
        "-DBUILD_TENSORFLOW_OPS=$BUILD_TENSORFLOW_OPS"
        "-DBUNDLE_CLOUDVIEWER_ML=$BUNDLE_CLOUDVIEWER_ML"
        "-DBUILD_JUPYTER_EXTENSION=$BUILD_JUPYTER_EXTENSION"
        "-DBUILD_FILAMENT_FROM_SOURCE=$BUILD_FILAMENT_FROM_SOURCE"
        "-DCONDA_PREFIX=$CONDA_PREFIX"
        "-DCMAKE_PREFIX_PATH=$CONDA_LIB_DIR"
        "-DBUILD_WITH_CONDA=$BUILD_WITH_CONDA"
        "-DCMAKE_INSTALL_PREFIX=$CLOUDVIEWER_INSTALL_DIR"
    )
    set -x # Echo commands on
    cmake -DBUILD_CUDA_MODULE=OFF "${cmakeOptions[@]}" ..
    set +x # Echo commands off
    echo

    echo "Packaging CloudViewer CPU pip package..."
    make VERBOSE=1 -j"$NPROC" pip-package
    echo "Finish make pip-package for cpu"
    mv lib/python_package/pip_package/cloudviewer*.whl . # save CPU wheel

    if [ "$BUILD_CUDA_MODULE" == ON ]; then
        echo
        echo Installing CUDA versions of TensorFlow and PyTorch...
        install_python_dependencies with-cuda purge-cache
        echo
        echo Building with CUDA...
        rebuild_list=(bin lib/Release/*.a lib/_build_config.py* lib/ml)
        echo
        echo Removing CPU compiled files / folders: "${rebuild_list[@]}"
        rm -r "${rebuild_list[@]}" || true
        set -x # Echo commands on
        cmake --version
        cmake   -DBUILD_CUDA_MODULE=ON \
                -DBUILD_COMMON_CUDA_ARCHS=ON \
                "${cmakeOptions[@]}" ..
        set +x # Echo commands off
    fi
    echo

    echo "Packaging CloudViewer full pip package..."
    make VERBOSE=1 -j"$NPROC" pip-package
    echo "Finish make CloudViewer full pip package"
    mv cloudviewer*.whl lib/python_package/pip_package/ # restore CPU wheel
    popd                                           # PWD=ACloudViewer
}

# Test wheel in blank virtual environment
# Usage: test_wheel wheel_path
test_wheel() {
    wheel_path="$1"
    python -m venv cloudViewer_test.venv
    # shellcheck disable=SC1091
    source cloudViewer_test.venv/bin/activate
    python -m pip install -U pip=="$PIP_VER"
    python -m pip install -U -c "${CLOUDVIEWER_SOURCE_ROOT}/python/requirements_build.txt" wheel setuptools
    echo -n "Using python: $(command -v python)"
    python --version
    echo -n "Using pip: "
    python -m pip --version
    echo "Installing CloudViewer wheel $wheel_path in virtual environment..."
    # Uninstall cloudViewer if already installed to ensure clean installation
    if python -c "import cloudViewer" 2>/dev/null; then
        echo "cloudViewer is already installed, uninstalling first..."
        python -m pip uninstall --yes cloudviewer cloudviewer-cpu 2>/dev/null || true
    fi
    python -m pip install "$wheel_path"
    python -W default -c "import cloudViewer; print('Installed:', cloudViewer); print('BUILD_CUDA_MODULE: ', cloudViewer._build_config['BUILD_CUDA_MODULE'])"
    python -W default -c "import cloudViewer; print('CUDA available: ', cloudViewer.core.cuda.is_available())"
    # python -W default -c "import cloudViewer; cloudViewer.reconstruction.gui.run_graphical_gui()"
    # cloudViewer example reconstruction/gui
    echo

    echo
    HAVE_PYTORCH_OPS=OFF
    HAVE_TENSORFLOW_OPS=OFF

    if python -c "import sys, cloudViewer; sys.exit(not cloudViewer._build_config['BUILD_PYTORCH_OPS'])"; then
        HAVE_PYTORCH_OPS=ON
    fi
    if python -c "import sys, cloudViewer; sys.exit(not cloudViewer._build_config['BUILD_TENSORFLOW_OPS'])"; then
        HAVE_TENSORFLOW_OPS=ON
    fi

    if [ "$HAVE_PYTORCH_OPS" == "ON" ]; then
        python -m pip install -r "$CLOUDVIEWER_ML_ROOT/requirements-torch.txt"
        # fix issues of invalid pixel size; error code 0x17
        if [[ "$OSTYPE" == "darwin"* ]]; then
            MATPLOT_LIB_TARGET_VERSION="3.9.4"
            if pip install "matplotlib==$MATPLOT_LIB_TARGET_VERSION" &>/dev/null; then
                echo "Successfully installed matplotlib version $MATPLOT_LIB_TARGET_VERSION"
            else
                echo "Ignore matplotlib re-installation: $MATPLOT_LIB_TARGET_VERSION"
            fi
        fi
        python  -W default -c \
            "import cloudViewer.ml.torch; print('PyTorch Ops library loaded:', cloudViewer.ml.torch._loaded)"
    fi
    if [ "$HAVE_TENSORFLOW_OPS" == "ON" ]; then
        python -m pip install -r "$CLOUDVIEWER_ML_ROOT/requirements-tensorflow.txt"
        python  -W default -c \
            "import cloudViewer.ml.tf.ops; print('TensorFlow Ops library loaded:', cloudViewer.ml.tf.ops)"
    fi
    
    echo
    if [ "$HAVE_TENSORFLOW_OPS" == "ON" ] && [ "$HAVE_PYTORCH_OPS" == "ON" ]; then
        echo "Importing TensorFlow and torch in the reversed order"
        python -W default -c "import tensorflow as tf; import torch; import cloudViewer.ml.torch as o3d"
        echo "Importing TensorFlow and torch in the normal order"
        python -W default -c "import cloudViewer.ml.torch as o3d; import tensorflow as tf; import torch"
    fi
    deactivate cloudViewer_test.venv # argument prevents unbound variable error
}

# Run in virtual environment
# Note: This function expects cloudViewer_test.venv to exist (created by test_wheel)
# or creates a new one if it doesn't exist.
# Usage: run_python_tests [wheel_path]
run_python_tests() {
    wheel_path="${1:-}"
    # Create venv if it doesn't exist
    if [ ! -d "cloudViewer_test.venv" ]; then
        echo "Creating virtual environment cloudViewer_test.venv..."
        python -m venv cloudViewer_test.venv
    fi
    
    # shellcheck disable=SC1091
    source cloudViewer_test.venv/bin/activate
    
    # Install test requirements
    python -m pip install -U pip
    python -m pip install -U -r "${CLOUDVIEWER_SOURCE_ROOT}/python/requirements_test.txt"
    
    # Install cloudViewer from wheel if provided
    if [ -n "$wheel_path" ]; then
        # Uninstall cloudViewer if already installed to ensure clean installation
        if python -c "import cloudViewer" 2>/dev/null; then
            echo "cloudViewer is already installed, uninstalling first..."
            python -m pip uninstall --yes cloudviewer cloudviewer-cpu 2>/dev/null || true
        fi
        python -m pip install "$wheel_path"
    elif ! python -c "import cloudViewer" 2>/dev/null; then
        echo "Warning: cloudViewer not installed and no wheel_path provided. Tests may fail."
    fi
    
    echo "Add --randomly-seed=SEED to the test command to reproduce test order."
    pytest_args=("${CLOUDVIEWER_SOURCE_ROOT}/python/test/")
    
    # Check if ML ops should be tested by checking build configuration
    # This checks the actual installed cloudViewer package, not environment variables
    HAVE_PYTORCH_OPS=OFF
    HAVE_TENSORFLOW_OPS=OFF
    if python -c "import sys, cloudViewer; sys.exit(not cloudViewer._build_config['BUILD_PYTORCH_OPS'])"; then
        HAVE_PYTORCH_OPS=ON
    fi
    if python -c "import sys, cloudViewer; sys.exit(not cloudViewer._build_config['BUILD_TENSORFLOW_OPS'])"; then
        HAVE_TENSORFLOW_OPS=ON
    fi
    
    if [ "$HAVE_PYTORCH_OPS" == "OFF" ] && [ "$HAVE_TENSORFLOW_OPS" == "OFF" ]; then
        echo "ML Ops not built, skipping ml_ops tests"
        pytest_args+=(--ignore "${CLOUDVIEWER_SOURCE_ROOT}/python/test/ml_ops/")
    else
        echo "ML Ops built (PyTorch: $HAVE_PYTORCH_OPS, TensorFlow: $HAVE_TENSORFLOW_OPS), including ml_ops tests"
    fi
    
    # Run pytest with verbose output
    echo "======================================================================"
    echo "Running Python Unit Tests"
    echo "======================================================================"
    python -m pytest -v "${pytest_args[@]}"
    pytest_result=$?
    
    echo ""
    if [ $pytest_result -eq 0 ]; then
        echo "======================================================================"
        echo "Python Unit Tests: PASSED"
        echo "======================================================================"
    else
        echo "======================================================================"
        echo "Python Unit Tests: FAILED (exit code: $pytest_result)"
        echo "======================================================================"
    fi
    
    # Deactivate venv (doesn't take arguments)
    deactivate || true
    
    # Optionally cleanup venv (commented out to allow reuse)
    # rm -rf cloudViewer_test.venv
    
    return $pytest_result
}

# Run C++ unit tests
# Usage: run_cpp_unit_tests
# Should be run from the build directory
run_cpp_unit_tests() {
    echo "======================================================================"
    echo "Running C++ Unit Tests"
    echo "======================================================================"
    
    pushd build
    # Check if tests executable exists
    if [ ! -f "./bin/tests" ]; then
        echo "Error: tests executable not found at ./bin/tests"
        echo "Please build with -DBUILD_UNIT_TESTS=ON"
        return 1
    fi
    
    # Set test flags
    unitTestFlags="--gtest_shuffle"
    if [ "${LOW_MEM_USAGE-}" = "ON" ]; then
        unitTestFlags="$unitTestFlags --gtest_filter=-*Reduce*Sum*"
    fi
    
    echo "Test flags: $unitTestFlags"
    echo "Tip: Run './bin/tests $unitTestFlags --gtest_random_seed=SEED' to repeat this test sequence."
    echo ""
    
    # Run the tests
    ./bin/tests $unitTestFlags
    test_result=$?
    
    echo ""
    if [ $test_result -eq 0 ]; then
        echo "======================================================================"
        echo "C++ Unit Tests: PASSED"
        echo "======================================================================"
    else
        echo "======================================================================"
        echo "C++ Unit Tests: FAILED (exit code: $test_result)"
        echo "======================================================================"
        return $test_result
    fi
    echo ""
    popd # build directory
}

# Run all tests (C++ and Python)
# Usage: run_all_tests
# Should be run from the build directory
run_all_tests() {
    echo "======================================================================"
    echo "Running All Tests (C++ and Python)"
    echo "======================================================================"
    echo ""
    
    local cpp_result=0
    local python_result=0
    
    # Run C++ tests if BUILD_UNIT_TESTS is ON
    run_cpp_unit_tests
    cpp_result=$?
    
    # Run Python tests if venv exists or can be created
    if [ -d "cloudViewer_test.venv" ] || command -v python3 &>/dev/null; then
        run_python_tests
        python_result=$?
    else
        echo "Skipping Python tests: Python environment not available"
        echo ""
    fi
    
    # Summary
    echo ""
    echo "======================================================================"
    echo "Test Summary"
    echo "======================================================================"
    echo "C++ Tests:    $([ $cpp_result -eq 0 ] && echo 'PASSED' || echo 'FAILED')"
    echo "Python Tests: $([ $python_result -eq 0 ] && echo 'PASSED' || echo 'FAILED')"
    echo "======================================================================"
    
    # Return non-zero if any test failed
    if [ $cpp_result -ne 0 ] || [ $python_result -ne 0 ]; then
        return 1
    fi
    return 0
}

# Install dependencies needed for building documentation
# Usage: install_docs_dependencies "${CLOUDVIEWER_ML_ROOT}"
install_docs_dependencies() {
    echo
    echo Install ubuntu dependencies from $(pwd)
    util/install_deps_ubuntu.sh assume-yes
    $SUDO apt-get install --yes \
        libxml2-dev libxslt-dev \
        python3-dev python-is-python3 python3-pip \
        doxygen \
        texlive \
        texlive-latex-extra \
        ghostscript \
        pandoc \
        ccache
    echo
    echo Install Python dependencies for building docs
    command -v python
    python -V
    python -m pip install -U -q "pip==$PIP_VER"
    which cmake || python -m pip install -U -q cmake

    $SUDO apt remove python3-blinker -y
    pip install --ignore-installed -U blinker
    python -m pip install -U -q -r "${CLOUDVIEWER_SOURCE_ROOT}/python/requirements_build.txt"
    if [[ -d "$1" ]]; then
        CLOUDVIEWER_ML_ROOT="$1"
        echo Installing CloudViewer-ML dependencies from "${CLOUDVIEWER_ML_ROOT}"
        python -m pip install -r "${CLOUDVIEWER_ML_ROOT}/requirements.txt" \
            -r "${CLOUDVIEWER_ML_ROOT}/requirements-torch.txt"
    else
        echo CLOUDVIEWER_ML_ROOT="${1:-not specified}" - Skipping ML dependencies.
    fi
    echo
    python -m pip install -r "${CLOUDVIEWER_SOURCE_ROOT}/python/requirements.txt" \
        -r "${CLOUDVIEWER_SOURCE_ROOT}/python/requirements_jupyter_build.txt" \
        -r "${CLOUDVIEWER_SOURCE_ROOT}/docs/requirements.txt"
}

# Build documentation
# Usage: build_docs $DEVELOPER_BUILD
build_docs() {
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "📚 ACloudViewer Documentation Build"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "Using cmake: $(command -v cmake)"
    cmake --version
    echo "NPROC=$NPROC"
    echo ""
    
    set +u
    DEVELOPER_BUILD="$1"
    set -u
    if [[ "$DEVELOPER_BUILD" != "OFF" ]]; then
        DEVELOPER_BUILD=ON
        DOC_ARGS=""
        echo "📝 Developer build mode"
    else
        DOC_ARGS="--is_release"
        echo "🚀 Release build mode"
        echo ""
        echo "Building ACloudViewer with ENABLE_HEADLESS_RENDERING=ON for Jupyter notebooks"
        echo ""
    fi
    echo ""
    
    # Base cmake options
    cmakeOptions=("-DBUILD_SHARED_LIBS=OFF"
        "-DDEVELOPER_BUILD=$DEVELOPER_BUILD"
        "-DCMAKE_BUILD_TYPE=Release"
        "-DCMAKE_INSTALL_LIBDIR=lib"
        "-DWITH_OPENMP=ON"
        "-DBUILD_AZURE_KINECT=OFF"
        "-DBUILD_LIBREALSENSE=OFF"
        "-DBUILD_TENSORFLOW_OPS=OFF"
        "-DBUILD_PYTORCH_OPS=ON"
        "-DBUILD_EXAMPLES=OFF"
        "-DBUILD_DOCUMENTATION=ON"
        "-DBUILD_PYTHON_MODULE=ON"
        "-DUSE_PCL_BACKEND=OFF" # no need pcl for documentation
        "-DBUILD_JUPYTER_EXTENSION=OFF"
        "-DBUILD_UNIT_TESTS=OFF"
        "-DBUILD_BENCHMARKS=OFF"
        "-DCVCORELIB_SHARED=ON"
        "-DBUILD_FILAMENT_FROM_SOURCE=OFF"
        "-DBUILD_RECONSTRUCTION=OFF"
    )
    
    # Add CMAKE_PREFIX_PATH if QT_DIR is set
    if [ -n "${QT_DIR:-}" ]; then
        cmakeOptions+=("-DCMAKE_PREFIX_PATH:PATH=${QT_DIR}/lib/cmake")
    fi
    
    # First build: Headless rendering for Jupyter notebooks execution
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "🔨 Build 1/2: Headless Rendering (for Jupyter notebooks)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    
    mkdir -p build
    cd build
    
    set -x
    cmake "${cmakeOptions[@]}" \
        -DENABLE_HEADLESS_RENDERING=ON \
        -DBUNDLE_CLOUDVIEWER_ML=OFF \
        -DBUILD_GUI=OFF \
        -DBUILD_WEBRTC=OFF \
        ..
    make python-package -j$NPROC
    set +x
    
    # Test Python module
    export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}$PWD/lib/python_package"
    python -c "import cloudViewer; print('✓ cloudViewer module:', cloudViewer)" || \
        echo "⚠️  Warning: Could not import cloudViewer module"
    
    echo ""
    echo "✅ Headless build complete"
    echo ""
    
    # Execute Jupyter notebooks with headless build
    # JupyterDocsBuilder will automatically execute notebooks that don't have output
    cd ../docs
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "📓 Executing Jupyter notebooks (headless mode)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    
    set -x
    # Only build Sphinx to execute notebooks, skip Doxygen
    # Use --clean to start fresh, but executed notebooks will be saved in source/tutorial/
    # Use --clean-notebooks to clean existing notebooks before copying
    # Use --execute-notebooks=always to ensure notebooks are executed in headless mode
    # Use --py-api-rst=never and --py-example-rst=never to skip Python API/example generation in first build
    python make_docs.py $DOC_ARGS --clean --sphinx --clean-notebooks --execute-notebooks=always --py-api-rst=never --py-example-rst=never
    set +x
    
    echo ""
    echo "✅ Jupyter notebooks executed (outputs saved in source/tutorial/)"
    echo ""
    
    # Uninstall headless build before building GUI version
    cd ../build
    python -m pip uninstall --yes cloudviewer || echo "⚠️  cloudviewer not installed via pip"
    
    # Second build: GUI enabled for visualization.{gui,rendering} documentation
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "🔨 Build 2/2: GUI Enabled (for visualization documentation)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    
    set -x
    cmake "${cmakeOptions[@]}" \
        -DENABLE_HEADLESS_RENDERING=OFF \
        -DBUNDLE_CLOUDVIEWER_ML=ON \
        -DBUILD_GUI=ON \
        -DBUILD_WEBRTC=ON \
        ..
    make python-package -j$NPROC
    set +x
    
    # Test Python module (may fail in headless CI, that's expected)
    export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}$PWD/lib/python_package"
    python -c "import cloudViewer; print('✓ cloudViewer module:', cloudViewer)" || \
        echo "⚠️  Warning: Could not import cloudViewer module (expected in headless CI)"
    
    echo ""
    echo "✅ GUI build complete"
    echo ""
    
    # Generate full documentation with GUI build
    cd ../docs
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "📄 Generating Full Documentation"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "Build Strategy:"
    echo "  1. Doxygen   → Independent C++ API HTML"
    echo "  2. Sphinx    → Python API + Tutorials"
    echo "  3. Integration → Copy Doxygen HTML to Sphinx output"
    echo ""
    echo "Documentation structure:"
    echo "  ├── Tutorial (30+ Jupyter notebooks)"
    echo "  ├── Python API (47 modules, Sphinx autodoc)"
    echo "  └── C++ API (Doxygen HTML, independent)"
    echo ""
    echo "Reference: https://github.com/isl-org/Open3D"
    echo ""
    
    set -x
    # Build both Doxygen and Sphinx documentation
    # This follows Open3D's proven approach:
    # - Doxygen runs first, generates independent HTML
    # - Sphinx runs second, generates Python docs
    # - Notebooks already executed in first build, use --execute-notebooks=never to skip re-execution
    # - HTML outputs are combined via file system
    # Don't use --clean here to preserve executed notebook outputs from first build
    # Use --py-api-rst=always and --py-example-rst=always to generate Python API/example docs in second build
    python make_docs.py $DOC_ARGS --sphinx --doxygen --parallel --execute-notebooks=never --py-api-rst=always --py-example-rst=always
    set +x
    
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "✅ Documentation Build Complete"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    
    if [ -d "_out/html" ]; then
        echo "📊 Build Statistics:"
        echo "  - Total files: $(find _out/html -type f | wc -l)"
        echo "  - HTML pages: $(find _out/html -name '*.html' | wc -l)"
        echo "  - Total size: $(du -sh _out/html | cut -f1)"
        echo ""
        echo "📦 Output: docs/_out/html/"
        echo ""
        echo "🌐 Preview:"
        echo "  cd .. && python3 -m http.server --directory docs/_out/html 8080"
        echo "  Then open: http://localhost:8080"
    else
        echo "❌ Error: Documentation output not found (docs/_out/html)"
        return 1
    fi
    
    cd ..
}

maximize_ubuntu_github_actions_build_space() {
    # Enhanced version with better Docker space management
    # https://github.com/easimon/maximize-build-space/blob/main/action.yml
    echo "=== Initial disk space ==="
    df -h .                                  # => 26GB
    
    # Remove large pre-installed packages
    $SUDO rm -rf /usr/share/dotnet           # ~17GB
    $SUDO rm -rf /usr/local/lib/android      # ~11GB
    $SUDO rm -rf /opt/ghc                    # ~2.7GB
    $SUDO rm -rf /opt/hostedtoolcache/CodeQL # ~5.4GB
    $SUDO rm -rf "$AGENT_TOOLSDIRECTORY"
    
    # Additional cleanup for more space
    $SUDO rm -rf /usr/local/share/boost      # ~1GB
    $SUDO rm -rf /usr/share/swift            # ~1GB
    $SUDO rm -rf /opt/az                     # ~1GB
    $SUDO rm -rf /usr/local/.ghcup           # ~2GB
    $SUDO rm -rf /opt/microsoft              # ~1GB
    
    # Configure Docker to use efficient storage
    echo "=== Stopping Docker service ==="
    $SUDO systemctl stop docker.socket || true
    $SUDO systemctl stop docker.service || true
    $SUDO systemctl stop containerd || true
    
    # Wait for Docker to fully stop
    sleep 5
    
    # Create Docker daemon config for space optimization
    $SUDO mkdir -p /etc/docker
    $SUDO tee /etc/docker/daemon.json > /dev/null <<EOF
{
  "storage-driver": "overlay2",
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "10m",
    "max-file": "3"
  },
  "max-concurrent-downloads": 3,
  "max-concurrent-uploads": 3
}
EOF
    
    echo "=== Starting Docker service ==="
    $SUDO systemctl start containerd || true
    $SUDO systemctl start docker.service
    $SUDO systemctl start docker.socket || true
    
    # Wait for Docker to be ready
    echo "=== Waiting for Docker to be ready ==="
    timeout 60 bash -c 'until docker info >/dev/null 2>&1; do echo "Waiting for Docker..."; sleep 2; done' || {
        echo "Docker failed to start, checking logs:"
        $SUDO journalctl -u docker.service --no-pager -l | tail -20
        $SUDO systemctl status docker.service
        exit 1
    }
    
    # Clean Docker system
    $SUDO docker system prune -a -f --volumes || true
    
    echo "=== Docker info ==="
    $SUDO docker info
    
    echo "=== Final disk space ==="
    df -h .                                  
    df -h /var/lib/docker                       # => /var/lib/docker -> ~101GB
}

maximize_ubuntu_github_actions_build_space_simple() {
    # https://github.com/easimon/maximize-build-space/blob/main/action.yml
    df -h .                                  # => 26GB
    $SUDO rm -rf /usr/share/dotnet           # ~17GB
    $SUDO rm -rf /usr/local/lib/android      # ~11GB
    $SUDO rm -rf /opt/ghc                    # ~2.7GB
    $SUDO rm -rf /opt/hostedtoolcache/CodeQL # ~5.4GB
    $SUDO docker image prune --all --force   # ~4.5GB
    $SUDO rm -rf "$AGENT_TOOLSDIRECTORY"
    df -h . # => 53GB
}

monitor_disk_space() {
    local step_name="${1:-Unknown step}"
    echo "=== Disk space monitoring: $step_name ==="
    echo "Filesystem usage:"
    df -h
    echo ""
    echo "Docker system usage:"
    docker system df 2>/dev/null || echo "Docker not available"
    echo ""
    echo "Available space in /var/lib/docker:"
    df -h /var/lib/docker 2>/dev/null || df -h /
    echo ""
}

docker_cleanup_aggressive() {
    echo "=== Aggressive Docker cleanup ==="
    # Stop all containers
    docker stop $(docker ps -aq) 2>/dev/null || true
    
    # Remove all containers
    docker rm $(docker ps -aq) 2>/dev/null || true
    
    # Remove all images
    docker rmi $(docker images -q) 2>/dev/null || true
    
    # Remove all volumes
    docker volume rm $(docker volume ls -q) 2>/dev/null || true
    
    # Remove all networks (except default ones)
    docker network rm $(docker network ls -q) 2>/dev/null || true
    
    # Prune everything
    docker system prune --all --force --volumes 2>/dev/null || true
    
    # Clean build cache
    docker builder prune --all --force 2>/dev/null || true
    
    echo "Docker cleanup completed"
    docker system df 2>/dev/null || true
}

restart_docker_service() {
    echo "=== Restarting Docker service safely ==="
    
    # Stop all Docker services in correct order
    echo "Stopping Docker services..."
    $SUDO systemctl stop docker.socket || true
    $SUDO systemctl stop docker.service || true
    $SUDO systemctl stop containerd || true
    
    # Wait for services to fully stop
    sleep 5
    
    # Kill any remaining Docker processes
    $SUDO pkill -f dockerd || true
    $SUDO pkill -f containerd || true
    
    # Clean up any stale mount points
    echo "Cleaning up stale mount points..."
    $SUDO umount /var/lib/docker/overlay2/*/merged 2>/dev/null || true
    
    # Start services in correct order
    echo "Starting Docker services..."
    $SUDO systemctl start containerd || true
    sleep 2
    $SUDO systemctl start docker.service
    sleep 2
    $SUDO systemctl start docker.socket || true
    
    # Wait for Docker to be ready
    echo "Waiting for Docker to be ready..."
    timeout 60 bash -c 'until docker info >/dev/null 2>&1; do echo "Waiting for Docker..."; sleep 2; done' || {
        echo "Docker failed to start, checking logs:"
        $SUDO journalctl -u docker.service --no-pager -l | tail -20
        $SUDO systemctl status docker.service
        return 1
    }
    
    echo "Docker service restarted successfully"
    docker info
}
