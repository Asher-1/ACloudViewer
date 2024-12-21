#!/usr/bin/env bash
set -euo pipefail

# The following environment variables are required:
SUDO=${SUDO:=sudo}
UBUNTU_VERSION=${UBUNTU_VERSION:="$(lsb_release -cs 2>/dev/null || true)"} # Empty in macOS

DEVELOPER_BUILD="${DEVELOPER_BUILD:-ON}"
if [[ "$DEVELOPER_BUILD" != "OFF" ]]; then # Validate input coming from GHA input field
    DEVELOPER_BUILD="ON"
fi
BUILD_SHARED_LIBS=${BUILD_SHARED_LIBS:-OFF}
NPROC=${NPROC:-$(getconf _NPROCESSORS_ONLN)} # POSIX: MacOS + Linux
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
	echo "Conda env is not activated and CONDA_PREFIX variable should be set before!"
    CONDA_PREFIX=/root/miniconda3/envs/cloudViewer
else
    echo "Conda env: $CONDA_PREFIX is activated."
fi

# Dependency versions:
# CUDA: see docker/docker_build.sh
# ML
TENSORFLOW_VER="2.16.2"
TORCH_VER="2.2.2"
TORCH_REPO_URL="https://download.pytorch.org/whl/torch/"

DISTRIB_ID=""
DISTRIB_RELEASE=""
if [[ "$OSTYPE" == "darwin"* ]]; then
    BUILD_RIEGL=OFF
    CONDA_LIB_DIR="$CONDA_PREFIX/lib"
    CLOUDVIEWER_INSTALL_DIR=~/cloudViewer_install

    # use lower target(11.0) version for compacibility
    PROCESSOR_ARCH=$(uname -m)
    if [ "$PROCESSOR_ARCH" == "arm64" ]; then
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
    # Fix Ubuntu18.04 issues: You're trying to build PyTorch with a too old version of GCC. 
    # We need GCC 9 or later.
    if [ "$DISTRIB_ID" == "Ubuntu" -a "$DISTRIB_RELEASE" == "18.04" ]; then
        TENSORFLOW_VER="2.13.0"
        TORCH_VER="2.0.1"
    fi
else # do not support windows
    echo "Do not support windows system with this script!"
    exit -1
fi

# Python
PIP_VER="23.2.1"
WHEEL_VER="0.38.4"
STOOLS_VER="67.3.2"
YAPF_VER="0.30.0"
PROTOBUF_VER="4.24.0"
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

    python -m pip install --upgrade pip=="$PIP_VER" wheel=="$WHEEL_VER" \
        setuptools=="$STOOLS_VER" $SPEED_CMD
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

    # TODO: modify other locations to use requirements.txt
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
            python -m pip install -U "${TORCH_GLNX}" -f "$TORCH_REPO_URL" tensorboard $SPEED_CMD
        elif [[ "$OSTYPE" == "darwin"* ]]; then
            python -m pip install -U torch=="$TORCH_VER" -f "$TORCH_REPO_URL" tensorboard $SPEED_CMD
        else
            echo "unknown OS $OSTYPE"
            exit 1
        fi
    fi
    if [ "$BUILD_TENSORFLOW_OPS" == "ON" ] || [ "$BUILD_PYTORCH_OPS" == "ON" ]; then
        python -m pip install -U yapf=="$YAPF_VER" $SPEED_CMD
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
    if [ -f "${CLOUDVIEWER_ML_ROOT}/set_cloudViewer_ml_root.sh" ]; then
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
        "-DDEVELOPER_BUILD=$DEVELOPER_BUILD"
        "-DCMAKE_BUILD_TYPE=Release"
        "-DBUILD_BENCHMARKS=OFF"
        "-DBUILD_AZURE_KINECT=ON"
        "-DBUILD_LIBREALSENSE=$BUILD_LIBREALSENSE" # some issues with network locally
        "-DWITH_OPENMP=ON"
        "-DWITH_IPPICV=ON"
        "-DWITH_SIMD=ON"
        "-DUSE_SIMD=ON"
        "-DUSE_PCL_BACKEND=OFF" # no need pcl for wheel
        "-DBUILD_RECONSTRUCTION=ON"
        "-DBUILD_FILAMENT_FROM_SOURCE=ON"
        "-DBUILD_CUDA_MODULE=$BUILD_CUDA_MODULE"
        "-DBUILD_COMMON_CUDA_ARCHS=ON"
        # TODO: PyTorch still use old CXX ABI, remove this line when PyTorch is updated
        "-DGLIBCXX_USE_CXX11_ABI=OFF"
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
        echo "BUILD_WITH_CONDA is off"
    fi
    set -u

    echo
    echo "Start building with ACloudViewer GUI..."
    mkdir -p build
    pushd build # PWD=ACloudViewer/build
    cmakeGuiOptions=("-DBUILD_SHARED_LIBS=OFF"
                "-DDEVELOPER_BUILD=$DEVELOPER_BUILD"
                "-DCMAKE_BUILD_TYPE=Release"
                "-DBUILD_JUPYTER_EXTENSION=OFF"
                "-DBUILD_LIBREALSENSE=OFF"
                "-DBUILD_AZURE_KINECT=OFF"
                "-DBUILD_PYTORCH_OPS=OFF"
                "-DBUILD_TENSORFLOW_OPS=OFF"
                "-DBUNDLE_CLOUDVIEWER_ML=OFF"
                "-DBUILD_BENCHMARKS=OFF"
                "-DBUILD_WEBRTC=OFF"
                "-DWITH_OPENMP=ON"
                "-DWITH_IPPICV=ON"
                "-DWITH_SIMD=ON"
                "-DWITH_PCL_NURBS=$WITH_PCL_NURBS"
                "-DUSE_SIMD=ON"
                "-DPACKAGE=$PACKAGE"
                "-DBUILD_OPENCV=ON"
                "-DBUILD_RECONSTRUCTION=ON"
                "-DBUILD_CUDA_MODULE=$BUILD_CUDA_MODULE"
                "-DBUILD_COMMON_CUDA_ARCHS=ON"
                "-DGLIBCXX_USE_CXX11_ABI=ON"
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
                "-DPLUGIN_STANDARD_QCORK=OFF"
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
    if [ -f "${CLOUDVIEWER_ML_ROOT}/set_cloudViewer_ml_root.sh" ]; then
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
    CXX11_ABI=ON
    if [ "$BUILD_TENSORFLOW_OPS" == "ON" ]; then
        CXX11_ABI=$(python -c "import tensorflow as tf; print('ON' if tf.__cxx11_abi_flag__ else 'OFF')")
    elif [ "$BUILD_PYTORCH_OPS" == "ON" ]; then
        CXX11_ABI=$(python -c "import torch; print('ON' if torch._C._GLIBCXX_USE_CXX11_ABI else 'OFF')")
    fi
    echo Building with GLIBCXX_USE_CXX11_ABI="$CXX11_ABI"
    if [[ "with_conda" =~ ^($options)$ ]]; then
        BUILD_WITH_CONDA=ON
        echo "BUILD_WITH_CONDA is on"
    else
        BUILD_WITH_CONDA=OFF
        CONDA_LIB_DIR=""
        echo "BUILD_WITH_CONDA is off"
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
        "-DBUILD_UNIT_TESTS=OFF"
        "-DBUILD_BENCHMARKS=OFF"
        "-DUSE_SIMD=ON"
        "-DWITH_SIMD=ON"
        "-DWITH_OPENMP=ON"
        "-DWITH_IPPICV=ON"
        "-DUSE_PCL_BACKEND=OFF" # no need pcl for wheel
        "-DBUILD_RECONSTRUCTION=ON"
        "-DGLIBCXX_USE_CXX11_ABI=$CXX11_ABI"
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
    mv lib/python_package/pip_package/cloudViewer*.whl . # save CPU wheel

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
    mv cloudViewer*.whl lib/python_package/pip_package/ # restore CPU wheel
    popd                                           # PWD=ACloudViewer
}

# Test wheel in blank virtual environment
# Usage: test_wheel wheel_path
test_wheel() {
    wheel_path="$1"
    python -m venv cloudViewer_test.venv
    # shellcheck disable=SC1091
    source cloudViewer_test.venv/bin/activate
    python -m pip install --upgrade pip=="$PIP_VER" wheel=="$WHEEL_VER" \
        setuptools=="$STOOLS_VER"
    echo -n "Using python: $(command -v python)"
    python --version
    echo -n "Using pip: "
    python -m pip --version
    echo "Installing CloudViewer wheel $wheel_path in virtual environment..."
    python -m pip install "$wheel_path"
    python -W default -c "import cloudViewer; print('Installed:', cloudViewer); print('BUILD_CUDA_MODULE: ', cloudViewer._build_config['BUILD_CUDA_MODULE'])"
    python -W default -c "import cloudViewer; print('CUDA available: ', cloudViewer.core.cuda.is_available())"
    # python -W default -c "import cloudViewer; cloudViewer.reconstruction.gui.run_graphical_gui()"
    echo

    # Fix Ubuntu18.04 issues: You're trying to build PyTorch with a too old version of GCC. 
    # We need GCC 9 or later.
    if [ "$DISTRIB_ID" == "Ubuntu" -a "$DISTRIB_RELEASE" == "18.04" ]; then
        if [ "$BUILD_PYTORCH_OPS" == "ON" ]; then
            python -m pip install -r "${CLOUDVIEWER_SOURCE_ROOT}/python/requirements-torch201.txt"
            python  -W default -c \
                "import cloudViewer.ml.torch; print('PyTorch Ops library loaded:', cloudViewer.ml.torch._loaded)"
        fi
        if [ "$BUILD_TENSORFLOW_OPS" == "ON" ]; then
            python -m pip install -r "${CLOUDVIEWER_SOURCE_ROOT}/python/requirements-tensorflow.txt"
            python  -W default -c \
                "import cloudViewer.ml.tf.ops; print('TensorFlow Ops library loaded:', cloudViewer.ml.tf.ops)"
        fi
    else
        if [ "$BUILD_PYTORCH_OPS" == "ON" ]; then
            python -m pip install -r "$CLOUDVIEWER_ML_ROOT/requirements-torch.txt"
            python  -W default -c \
                "import cloudViewer.ml.torch; print('PyTorch Ops library loaded:', cloudViewer.ml.torch._loaded)"
        fi
        if [ "$BUILD_TENSORFLOW_OPS" == "ON" ]; then
            python -m pip install -r "$CLOUDVIEWER_ML_ROOT/requirements-tensorflow.txt"
            python  -W default -c \
                "import cloudViewer.ml.tf.ops; print('TensorFlow Ops library loaded:', cloudViewer.ml.tf.ops)"
        fi
    fi
    
    echo
    if [ "$BUILD_TENSORFLOW_OPS" == "ON" ] && [ "$BUILD_PYTORCH_OPS" == "ON" ]; then
        echo "Importing TensorFlow and torch in the reversed order"
        python -W default -c "import tensorflow as tf; import torch; import cloudViewer.ml.torch as o3d"
        echo "Importing TensorFlow and torch in the normal order"
        python -W default -c "import cloudViewer.ml.torch as o3d; import tensorflow as tf; import torch"
    fi
    deactivate cloudViewer_test.venv # argument prevents unbound variable error
}

# Run in virtual environment
run_python_tests() {
    # shellcheck disable=SC1091
    source cloudViewer_test.venv/bin/activate
    python -m pip install -U -r python/requirements_test.txt
    echo Add --randomly-seed=SEED to the test command to reproduce test order.
    pytest_args=("$CLOUDVIEWER_SOURCE_ROOT"/python/test/)
    if [ "$BUILD_PYTORCH_OPS" == "OFF" ] && [ "$BUILD_TENSORFLOW_OPS" == "OFF" ]; then
        echo Testing ML Ops disabled
        pytest_args+=(--ignore "$CLOUDVIEWER_SOURCE_ROOT"/python/test/ml_ops/)
    fi
    python -m pytest "${pytest_args[@]}"
    deactivate cloudViewer_test.venv # argument prevents unbound variable error
    rm -rf cloudViewer_test.venv     # cleanup for testing the next wheel
}

maximize_ubuntu_github_actions_build_space() {
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
