#!/usr/bin/env bash
# =============================================================================
# setup_and_build_acloudviewer_linux.sh
#
# One-click end-to-end builder for ACloudViewer on a fresh Ubuntu machine:
#
#   [1/7] system packages (apt, idempotent — installed packages are skipped)
#   [2/7] pyenv + Python 3.12 (skipped when already present)
#   [3/7] Vulkan SDK build environment (skipped when already present)
#   [4/7] VTK 9.3.1 + PCL 1.14.1 via docker/build_vtk_pcl_deps.sh consume
#         (skipped when already installed; prebuilt tarball preferred, source
#         build as fallback)
#   [5/7] qPythonRuntime Python requirements
#   [6/7] CMake configure (same flag set as Option A of
#         compiling-cloudviewer-linux.md, BUILD_CUDA_MODULE=ON)
#   [7/7] make + make install → .run installer
#
# Usage:
#   bash setup_and_build_acloudviewer_linux.sh [--jobs N] [--qt-dir DIR]
#        [--source-dir DIR] [--prefix DIR] [--skip-build]
#
# The script is idempotent: re-running it only redoes what is missing.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

JOBS=""
QT_DIR_ARG=""
SOURCE_DIR="${REPO_ROOT}"
PREFIX="${HOME}/install"
SKIP_BUILD=OFF
PYTHON_VERSION=3.12
VTK_VERSION=9.3.1
PCL_VERSION=1.14.1

while [[ $# -gt 0 ]]; do
    case "$1" in
        --jobs) JOBS="$2"; shift 2 ;;
        --qt-dir) QT_DIR_ARG="$2"; shift 2 ;;
        --source-dir) SOURCE_DIR="$2"; shift 2 ;;
        --prefix) PREFIX="$2"; shift 2 ;;
        --skip-build) SKIP_BUILD=ON; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [[ -z "${JOBS}" ]]; then
    JOBS=$(( $(nproc) > 24 ? 24 : $(nproc) ))
fi

log() { printf '\n\033[1;34m[%s] %s\033[0m\n' "$(date +%H:%M:%S)" "$*"; }
warn() { printf '\033[1;33mWARNING:\033[0m %s\n' "$*"; }
die() { printf '\033[1;31mERROR:\033[0m %s\n' "$*" >&2; exit 1; }

[[ $(id -u) -eq 0 ]] || sudo -n true 2>/dev/null || \
    warn "This script needs sudo for apt/VTK-PCL steps; you may be asked for your password."
command -v git >/dev/null || die "git is required. Install it first (sudo apt install git)."
[[ -f "${SOURCE_DIR}/CMakeLists.txt" ]] || die "Source dir ${SOURCE_DIR} does not look like the ACloudViewer repository."

# ---------------------------------------------------------------------------
# [1/7] System packages (apt is naturally idempotent: installed = skipped)
# ---------------------------------------------------------------------------
log "[1/7] Installing system packages"
SUDO="sudo"
[[ $(id -u) -eq 0 ]] && SUDO=""

# Base toolchain + project system deps (same as util/install_deps_ubuntu.sh).
if [[ -f "${SOURCE_DIR}/util/install_deps_ubuntu.sh" ]]; then
    ( cd "${SOURCE_DIR}" && ${SUDO} bash util/install_deps_ubuntu.sh assume-yes ) \
        || warn "install_deps_ubuntu.sh returned non-zero — continuing, already-installed packages are kept."
else
    warn "util/install_deps_ubuntu.sh not found; installing the core apt list only."
    ${SUDO} apt-get update -y
    ${SUDO} apt-get install -y -q build-essential ninja-build libtbb-dev \
        libglu1-mesa-dev xorg-dev libcurl4-openssl-dev libusb-dev libpcap-dev
fi

# Qt5 dev stack — required by the APP build but NOT covered by
# install_deps_ubuntu.sh (mirrors docker/Dockerfile.ci).
log "[1/7] Installing Qt5 development packages"
${SUDO} apt-get install -y -q \
    qtbase5-dev libqt5svg5-dev libqt5opengl5-dev qttools5-dev qttools5-dev-tools \
    libqt5websockets5-dev libqt5xmlpatterns5-dev libqt5x11extras5-dev \
    qtdeclarative5-dev qtdeclarative5-dev-tools libqt5quickcontrols2-5 \
    libqt5networkauth5-dev qt5-image-formats-plugins qttranslations5-l10n \
    libxxf86vm-dev libudev-dev

# ---------------------------------------------------------------------------
# [2/7] pyenv + Python (skip when the requested version already exists)
# ---------------------------------------------------------------------------
log "[2/7] Setting up pyenv + Python ${PYTHON_VERSION}"
export PYENV_ROOT="${PYENV_ROOT:-${HOME}/.pyenv}"
if [[ ! -d "${PYENV_ROOT}" ]]; then
    curl -fsSL https://pyenv.run | bash
fi
export PATH="${PYENV_ROOT}/shims:${PYENV_ROOT}/bin:${PATH}"
pyenv --version >/dev/null 2>&1 || {
    # Non-interactive shells may not have shims wired up yet.
    export PATH="${PYENV_ROOT}/versions/${PYTHON_VERSION}/bin:${PATH}"
}
if [[ ! -d "${PYENV_ROOT}/versions/${PYTHON_VERSION}" ]] && \
   ! ls -d "${PYENV_ROOT}"/versions/${PYTHON_VERSION}.* >/dev/null 2>&1; then
    pyenv install -s "${PYTHON_VERSION}"
fi
# Resolve "3.12" to the newest installed 3.12.x via a stable symlink.
_real=$(ls -d "${PYENV_ROOT}"/versions/${PYTHON_VERSION}.* 2>/dev/null | sort -V | tail -1 | xargs basename)
if [[ -n "${_real}" && "${_real}" != "${PYTHON_VERSION}" ]]; then
    ln -sfn "${PYENV_ROOT}/versions/${_real}" "${PYENV_ROOT}/versions/${PYTHON_VERSION}"
fi
pyenv global "${PYTHON_VERSION}" 2>/dev/null || true
pyenv rehash
export PATH="${PYENV_ROOT}/versions/${PYTHON_VERSION}/bin:${PATH}"
command -v python >/dev/null && python --version

PYTHON_EXE=$(pyenv which python 2>/dev/null || command -v python)
PYTHON_ROOT=$(python -c "import sysconfig, os; print(os.path.dirname(os.path.dirname(sysconfig.get_path('include'))))")
PYTHON_LIB_DIR=$(python -c "import sysconfig, os; libdir = sysconfig.get_config_var('LIBDIR'); print(os.path.realpath(libdir) if os.path.islink(libdir) else libdir)")
PYTHON_LIB_NAME=$(python -c "import sysconfig; print(sysconfig.get_config_var('LDLIBRARY'))")
PYTHON_LIB="${PYTHON_LIB_DIR}/${PYTHON_LIB_NAME}"
[[ -f "${PYTHON_LIB}" ]] || die "Python library ${PYTHON_LIB} not found."
echo "Python: ${PYTHON_EXE}  lib: ${PYTHON_LIB}"

# ---------------------------------------------------------------------------
# [3/7] Vulkan SDK build environment (AICore/qDA3); skipped when present
# ---------------------------------------------------------------------------
log "[3/7] Vulkan build environment"
VULKAN_ENV_FILE="${HOME}/.local/share/acloudviewer/acloudviewer-vulkan-env.sh"
if [[ ! -f "${VULKAN_ENV_FILE}" ]]; then
    ( cd "${SOURCE_DIR}" && ${SUDO} bash util/vulkan/install_vulkan_env.sh ) \
        || die "Vulkan environment installation failed."
fi
# shellcheck source=/dev/null
source "${VULKAN_ENV_FILE}"

# ---------------------------------------------------------------------------
# [4/7] VTK + PCL (skipped when already installed; prebuilt tarball first)
# ---------------------------------------------------------------------------
log "[4/7] VTK ${VTK_VERSION} + PCL ${PCL_VERSION}"
_vtk_ok=false
_pcl_ok=false
ls -d /usr/local/lib/cmake/vtk-* >/dev/null 2>&1 && _vtk_ok=true
[[ -n "$(find /usr/local -maxdepth 2 -iname "PCLConfig.cmake" 2>/dev/null)" ]] && _pcl_ok=true
[[ -n "$(find /usr -maxdepth 4 -name "PCLConfig.cmake" 2>/dev/null)" ]] && _pcl_ok=true

if ${_vtk_ok} && ${_pcl_ok}; then
    echo "VTK/PCL already installed — skipping."
else
    # Resolve a Qt dir usable by the VTK/PCL source-build fallback.
    if [[ -n "${QT_DIR_ARG}" ]]; then
        QT_DIR="${QT_DIR_ARG}"
    elif [[ -n "$(ls -d /opt/Qt*/5.*/gcc_64 2>/dev/null | sort -V | tail -1)" ]]; then
        QT_DIR=$(ls -d /opt/Qt*/5.*/gcc_64 | sort -V | tail -1)
    else
        # apt Qt: qmake lives in /usr/bin, cmake configs in /usr/lib/<arch>/cmake.
        # /usr satisfies both the consume script and the main configure.
        QT_DIR="/usr"
    fi
    echo "QT_DIR for VTK/PCL step: ${QT_DIR}"
    ${SUDO} bash "${SOURCE_DIR}/docker/build_vtk_pcl_deps.sh" \
        consume "${JOBS}" "${QT_DIR}" "${VTK_VERSION}" "${PCL_VERSION}"
fi

# The prebuilt VTK/PCL layer was compiled against Eigen 3.4 headers. If an
# older Eigen was previously installed system-wide it shadows the 3.4 headers
# (ColmapLib includes it first) and breaks the build with e.g.
# "'begin' was not declared in this scope" in Reconstruction/frame.cc.
if [[ -d /usr/local/include/eigen3 && \
      ! -f /usr/local/include/eigen3/Eigen/src/Core/StlIterators.h ]]; then
    warn "/usr/local/include/eigen3 is older than Eigen 3.4 but the prebuilt"
    warn "VTK/PCL require 3.4 headers. Fix it with:"
    warn "  sudo rsync -a --delete \"${SOURCE_DIR}/build_app/external/include/eigen3/\" /usr/local/include/eigen3/"
    warn "(run it AFTER the first cmake configure has populated build_app/external)."
fi

# ---------------------------------------------------------------------------
# [5/7] qPythonRuntime plugin requirements (mandatory when PLUGIN_PYTHON=ON)
# ---------------------------------------------------------------------------
log "[5/7] Python plugin requirements"
python -m pip install -r \
    "${SOURCE_DIR}/plugins/core/Standard/qPythonRuntime/requirements-release.txt"

# ---------------------------------------------------------------------------
# [6/7] Configure (Option A flag set of compiling-cloudviewer-linux.md)
# ---------------------------------------------------------------------------
BUILD_DIR="${SOURCE_DIR}/build_app"
log "[6/7] Configuring in ${BUILD_DIR}"
mkdir -p "${BUILD_DIR}" && cd "${BUILD_DIR}"

QT_PREFIX="${QT_DIR_ARG:-}"
if [[ -z "${QT_PREFIX}" ]]; then
    if [[ -n "$(ls -d /opt/Qt*/5.*/gcc_64 2>/dev/null | sort -V | tail -1)" ]]; then
        QT_PREFIX=$(ls -d /opt/Qt*/5.*/gcc_64 | sort -V | tail -1)
    else
        QT_PREFIX="/usr"   # apt Qt: /usr/lib/<arch>/cmake/Qt5 is found automatically
    fi
fi

cmake \
    -DDEVELOPER_BUILD=OFF \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="${PREFIX}" \
    -DBUILD_WITH_CONDA=OFF \
    -DCMAKE_PREFIX_PATH="${QT_PREFIX}" \
    -DPython3_EXECUTABLE="${PYTHON_EXE}" \
    -DPython3_ROOT_DIR="${PYTHON_ROOT}" \
    -DPython3_LIBRARY="${PYTHON_LIB}" \
    -DBUILD_UNIT_TESTS=ON \
    -DBUILD_BENCHMARKS=ON \
    -DWITH_OPENMP=ON \
    -DWITH_IPP=ON \
    -DWITH_SIMD=ON \
    -DUSE_SIMD=ON \
    -DUSE_QT6=OFF \
    -DUSE_VTK_BACKEND=ON \
    -DBUILD_WEBRTC=OFF \
    -DBUILD_OPENCV=ON \
    -DBUILD_RECONSTRUCTION=ON \
    -DBUILD_CUDA_MODULE=ON \
    -DBUILD_COMMON_CUDA_ARCHS=ON \
    -DBUILD_JUPYTER_EXTENSION=OFF \
    -DBUILD_LIBREALSENSE=OFF \
    -DBUILD_AZURE_KINECT=OFF \
    -DBUILD_PYTORCH_OPS=OFF \
    -DBUILD_TENSORFLOW_OPS=OFF \
    -DBUNDLE_CLOUDVIEWER_ML=OFF \
    -DPACKAGE=ON \
    -DCVCORELIB_USE_CGAL=ON \
    -DCVCORELIB_SHARED=ON \
    -DCVCORELIB_USE_QT_CONCURRENT=ON \
    -DOPTION_USE_GDAL=OFF \
    -DOPTION_USE_DXF_LIB=ON \
    -DOPTION_USE_RANSAC_LIB=ON \
    -DOPTION_USE_SHAPE_LIB=ON \
    -DPLUGIN_IO_QDRACO=ON \
    -DPLUGIN_IO_QLAS=ON \
    -DPLUGIN_IO_QADDITIONAL=ON \
    -DPLUGIN_IO_QCORE=ON \
    -DPLUGIN_IO_QCSV_MATRIX=ON \
    -DPLUGIN_IO_QE57=ON \
    -DPLUGIN_IO_QMESH=ON \
    -DPLUGIN_IO_QPDAL=OFF \
    -DPLUGIN_IO_QPHOTOSCAN=ON \
    -DPLUGIN_IO_QRDB=OFF \
    -DPLUGIN_IO_QFBX=OFF \
    -DPLUGIN_IO_QSTEP=OFF \
    -DPLUGIN_STANDARD_QCORK=ON \
    -DPLUGIN_STANDARD_QJSONRPC=ON \
    -DPLUGIN_STANDARD_QCLOUDLAYERS=ON \
    -DPLUGIN_STANDARD_MASONRY_QAUTO_SEG=ON \
    -DPLUGIN_STANDARD_MASONRY_QMANUAL_SEG=ON \
    -DPLUGIN_STANDARD_QANIMATION=ON \
    -DPLUGIN_STANDARD_QBROOM=ON \
    -DQANIMATION_WITH_FFMPEG_SUPPORT=ON \
    -DPLUGIN_STANDARD_QCANUPO=ON \
    -DPLUGIN_STANDARD_QCOLORIMETRIC_SEGMENTER=ON \
    -DPLUGIN_STANDARD_QCOMPASS=ON \
    -DPLUGIN_STANDARD_QCSF=ON \
    -DPLUGIN_STANDARD_QFACETS=ON \
    -DPLUGIN_STANDARD_QHOUGH_NORMALS=ON \
    -DPLUGIN_STANDARD_QM3C2=ON \
    -DPLUGIN_STANDARD_QMPLANE=ON \
    -DPLUGIN_STANDARD_QPCL=ON \
    -DPLUGIN_STANDARD_QPCV=ON \
    -DPLUGIN_STANDARD_QPOISSON_RECON=ON \
    -DPLUGIN_STANDARD_QRANSAC_SD=ON \
    -DPLUGIN_STANDARD_QSRA=ON \
    -DPLUGIN_STANDARD_3DMASC=ON \
    -DPLUGIN_STANDARD_QTREEISO=ON \
    -DPLUGIN_STANDARD_QVOXFALL=ON \
    -DPLUGIN_STANDARD_G3POINT=ON \
    -DPLUGIN_STANDARD_QSIBR=ON \
    -DAICore_ENABLED=ON \
    -DAICore_USE_VULKAN=ON \
    -DAICore_USE_CUDA=ON \
    -DAICore_BUNDLE_CUDA_RUNTIME=ON \
    -DPLUGIN_STANDARD_QDA3=ON \
    -DPLUGIN_STANDARD_QDEEPLSD=ON \
    -DPLUGIN_STANDARD_QFACEDETECT=ON \
    -DPLUGIN_STANDARD_QFREESPLATTER=ON \
    -DPLUGIN_STANDARD_QLIGHTGLUE=ON \
    -DPLUGIN_STANDARD_QRFDETR=ON \
    -DPLUGIN_STANDARD_QRMBG=ON \
    -DPLUGIN_STANDARD_QYOLO=ON \
    -DPLUGIN_STANDARD_QSAM3=ON \
    -DPLUGIN_STANDARD_QTRELLIS=ON \
    -DPLUGIN_PYTHON=ON \
    -DBUILD_PYTHON_MODULE=ON \
    ..

# ---------------------------------------------------------------------------
# [7/7] Build + install (produces the .run installer via PACKAGE=ON)
# ---------------------------------------------------------------------------
if [[ "${SKIP_BUILD}" == "ON" ]]; then
    log "[7/7] Skipping build (--skip-build)"
else
    log "[7/7] Building with ${JOBS} jobs (this is the long step)"
    make -j"${JOBS}"
    make install -j"${JOBS}"
fi

echo
log "Done. Installer packages:"
find "${PREFIX}" -maxdepth 1 -name "*.run" -printf "  %p\n" 2>/dev/null \
    || true
find "${PREFIX}" -maxdepth 1 -name "*.run" | head -1 > /tmp/acloudviewer_run_path
[[ -s /tmp/acloudviewer_run_path ]] || warn "No .run installer found under ${PREFIX} — check the make install output above."
