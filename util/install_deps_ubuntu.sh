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
    libssl-dev
    # Filament build-from-source
    clang
    libc++-dev
    libc++abi-dev
    libsdl2-dev
    ninja-build
    libxi-dev
    # ggml Vulkan: loader + CI/runtime smoke (headers/glslc/SPIR-V via install_vulkan_env.sh)
    libvulkan-dev      # system libvulkan.so loader (runtime on end-user GPU)
    vulkan-tools       # vulkaninfo — used by CI docker_test / local smoke
    mesa-vulkan-drivers # lavapipe ICD — headless Vulkan smoke on CI runners
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
$SUDO apt-get update
if apt-cache show glslc >/dev/null 2>&1; then
    deps+=("glslc")
fi
$SUDO apt-get install ${APT_CONFIRM} ${deps[*]}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${ACLOUDVIEWER_SKIP_VULKAN_SETUP:-}" ]]; then
    echo "Skipping ggml Vulkan setup (ACLOUDVIEWER_SKIP_VULKAN_SETUP is set)."
elif [[ -x "${script_dir}/install_vulkan_env.sh" ]]; then
    bash "${script_dir}/install_vulkan_env.sh"
else
    echo "WARNING: ${script_dir}/install_vulkan_env.sh not found; skipping Vulkan setup." >&2
fi
