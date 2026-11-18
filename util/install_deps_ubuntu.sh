#!/usr/bin/env bash
# Use: install_deps_ubuntu.sh [ assume-yes ]

set -ev

SUDO=${SUDO:=sudo} # SUDO=command in docker (running as root, sudo not available)
if [ "$1" == "assume-yes" ]; then
    APT_CONFIRM="--assume-yes"
else
    APT_CONFIRM=""
fi

deps=(
    git
    # For package
    zip
    # CloudViewer
    xorg-dev
    libxcb-shm0
    libglu1-mesa-dev
    python3-dev
    python3-pip
    libssl-dev
    # Filament build-from-source
    clang
    libc++-dev
    libc++abi-dev
    libsdl2-dev
    ninja-build
    libxi-dev
    # ggml Vulkan: loader + CI/runtime smoke (headers/glslc/SPIR-V via util/vulkan/install_vulkan_env.sh)
    libvulkan-dev      # system libvulkan.so loader (runtime on end-user GPU)
    vulkan-tools       # vulkaninfo — used by CI docker_test / local smoke
    mesa-vulkan-drivers # lavapipe ICD — headless Vulkan smoke on CI runners
    # Compute shaders
    glslang-tools
    # ML
    libtbb-dev
    # Headless rendering
    libosmesa6-dev
    # RealSense
    apt-transport-https
    libudev-dev
    libusb-1.0-0-dev
    autoconf
    libtool
)

eval $(
    source /etc/lsb-release;
    echo DISTRIB_ID="$DISTRIB_ID";
    echo DISTRIB_RELEASE="$DISTRIB_RELEASE"
)

# To avoid dependence on libunwind, we don't want to use clang / libc++ versions later than 11.
# Ubuntu 20.04's has versions 8, 10 or 12 while Ubuntu 22.04 has versions 11 and later.
if [ "$DISTRIB_ID" == "Ubuntu" -a "$DISTRIB_RELEASE" == "20.04" ]; then
    deps=("${deps[@]/clang/clang-10}")
    deps=("${deps[@]/libc++-dev/libc++-10-dev}")
    deps=("${deps[@]/libc++abi-dev/libc++abi-10-dev}")
fi
if [ "$DISTRIB_ID" == "Ubuntu" -a "$DISTRIB_RELEASE" == "22.04" ]; then
    deps=("${deps[@]/clang/clang-11}")
    deps=("${deps[@]/libc++-dev/libc++-11-dev}")
    deps=("${deps[@]/libc++abi-dev/libc++abi-11-dev}")
fi
if [ "$DISTRIB_ID" == "Ubuntu" -a "$DISTRIB_RELEASE" == "24.04" ]; then
    deps=("${deps[@]/clang/clang-14}")
    deps=("${deps[@]/libc++-dev/libc++-14-dev}")
    deps=("${deps[@]/libc++abi-dev/libc++abi-14-dev}")
fi

# Special case for ARM64
if [ "$(uname -m)" == "aarch64" ]; then
    # For compiling LAPACK in OpenBLAS
    deps+=("gfortran")
fi

echo "apt-get install ${deps[*]}"
apt_update_with_retry() {
    local attempt
    for attempt in 1 2 3; do
        if $SUDO apt-get update --fix-missing; then
            return 0
        fi
        echo "apt-get update failed (attempt ${attempt}/3), retrying..." >&2
        sleep 2
    done
    return 1
}
apt_update_with_retry
if apt-cache show glslc >/dev/null 2>&1; then
    deps+=("glslc")
fi
if ! $SUDO apt-get install ${APT_CONFIRM} --fix-missing ${deps[*]}; then
    echo "apt-get install failed, refreshing indexes and retrying once..." >&2
    apt_update_with_retry
    $SUDO apt-get install ${APT_CONFIRM} --fix-missing ${deps[*]}
fi

# ---------------------------------------------------------------------------
# CMake >= 3.24 (cmake_minimum_required): Open3D parity, the pinned FAISS
# dependency and CUDA native architecture selection all require it. Ubuntu
# 20.04/22.04 only ship CMake 3.16/3.22 in the default apt repos, so fall
# back to the pinned pip wheel when the system CMake is missing or too old
# (same pattern as util/vulkan/install_vulkan_linux.sh). Docker images
# preinstall a pinned CMake on PATH and skip this block.
#
# The pinned version must be one PyPI actually publishes. 3.31.8 does not
# exist (PyPI has 3.31.0.1/.1/.2/.4/.6/.10), and pinning it made every cold
# build fail with "No matching distribution found". A wheel present in a
# warm Docker cache hid this; touching an earlier layer exposed it.
# ---------------------------------------------------------------------------
CMAKE_PIP_VERSION="3.31.10"
cmake_version() {
    command -v cmake >/dev/null 2>&1 || return 1
    cmake --version | awk 'NR == 1 {print $3}'
}
cmake_too_old() {
    local ver="$(cmake_version || true)"
    [[ -z "${ver}" ]] && return 0
    # sort -V: if 3.24 sorts to the front, the found version is older.
    [[ "$(printf '%s\n%s\n' "${ver}" "3.24" | sort -V | head -n1)" != "3.24" ]]
}
if cmake_too_old; then
    echo "System CMake $(cmake_version || echo '<missing>') is below the required 3.24; installing the pinned pip CMake ${CMAKE_PIP_VERSION}."
    if ! python3 -m pip install --user --upgrade "cmake==${CMAKE_PIP_VERSION}"; then
        # set -e would abort silently here; surface the manual-install paths.
        echo "ERROR: pip install of the pinned CMake ${CMAKE_PIP_VERSION} failed." >&2
        echo "Install CMake >= 3.24 manually: https://apt.kitware.com/ (Ubuntu) or https://cmake.org/download/" >&2
        exit 1
    fi
    # The pip wheel lands in ~/.local/bin; make it win for this script and
    # any shell sourcing this file afterwards.
    export PATH="${HOME}/.local/bin:${PATH}"
fi
if cmake_too_old; then
    echo "ERROR: CMake >= 3.24 still unavailable after the pip fallback." >&2
    echo "Install it manually: https://apt.kitware.com/ (Ubuntu) or https://cmake.org/download/" >&2
    exit 1
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${ACLOUDVIEWER_SKIP_VULKAN_SETUP:-}" ]]; then
    echo "Skipping ggml Vulkan setup (ACLOUDVIEWER_SKIP_VULKAN_SETUP is set)."
elif [[ -x "${script_dir}/vulkan/install_vulkan_env.sh" ]]; then
    bash "${script_dir}/vulkan/install_vulkan_env.sh"
else
    echo "WARNING: ${script_dir}/vulkan/install_vulkan_env.sh not found; skipping Vulkan setup." >&2
fi
