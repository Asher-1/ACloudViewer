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
    # ggml Vulkan build tools (the installed app only needs a GPU driver/ICD)
    libvulkan-dev
    vulkan-tools
    mesa-vulkan-drivers
    spirv-headers
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
else
    echo "glslc is unavailable on this Ubuntu release"
fi
$SUDO apt-get install ${APT_CONFIRM} ${deps[*]}

version_at_least() {
    dpkg --compare-versions "$1" ge "$2"
}

install_glslc_from_source() {
    # The tag and commit are both fixed so a moved/recreated tag fails closed.
    local shaderc_tag="${SHADERC_TAG:-v2026.3}"
    local shaderc_commit="${SHADERC_COMMIT:-82757e4f72af8518fb679604d352623450cb761f}"
    local build_root="${SHADERC_BUILD_ROOT:-${TMPDIR:-/tmp}/acloudviewer-${shaderc_tag}}"
    local source_dir="${build_root}/shaderc"
    local build_dir="${build_root}/build"
    local cmake_bin

    cmake_bin="$(command -v cmake)"
    local cmake_version
    cmake_version="$(${cmake_bin} --version | awk 'NR == 1 {print $3}')"
    if ! version_at_least "${cmake_version}" "3.22.1"; then
        echo "System CMake ${cmake_version} is too old for ${shaderc_tag}; installing a build-only CMake." >&2
        python3 -m pip install --user --upgrade "cmake==3.31.6"
        cmake_bin="${HOME}/.local/bin/cmake"
    fi

    rm -rf "${source_dir}" "${build_dir}"
    mkdir -p "${build_root}"
    git clone --branch "${shaderc_tag}" --depth 1 \
        https://github.com/google/shaderc.git "${source_dir}"
    local actual_commit
    actual_commit="$(git -C "${source_dir}" rev-parse HEAD)"
    if [[ "${actual_commit}" != "${shaderc_commit}" ]]; then
        echo "ERROR: shaderc ${shaderc_tag} resolved to ${actual_commit}, expected ${shaderc_commit}." >&2
        exit 1
    fi

    (cd "${source_dir}" && ./utils/git-sync-deps)
    "${cmake_bin}" -S "${source_dir}" -B "${build_dir}" -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DSHADERC_SKIP_TESTS=ON \
        -DSHADERC_SKIP_EXAMPLES=ON \
        -DSHADERC_SKIP_COPYRIGHT_CHECK=ON \
        -DSHADERC_SKIP_INSTALL=ON
    "${cmake_bin}" --build "${build_dir}" --target glslc_exe \
        --parallel "${VULKAN_BUILD_JOBS:-4}"
    ${SUDO} install -m 0755 "${build_dir}/glslc/glslc" /usr/local/bin/glslc
    rm -rf "${source_dir}" "${build_dir}"
}

if ! command -v glslc >/dev/null 2>&1; then
    echo "Ubuntu ${DISTRIB_RELEASE} does not provide glslc; building pinned shaderc from source."
    install_glslc_from_source
fi

spirv_header=""
for candidate in \
    /usr/include/spirv/unified1/spirv.hpp \
    /usr/local/include/spirv/unified1/spirv.hpp; do
    if [[ -f "${candidate}" ]]; then
        spirv_header="${candidate}"
        break
    fi
done

if [[ -z "${spirv_header}" ]]; then
    echo "ERROR: spirv/unified1/spirv.hpp was not installed." >&2
    exit 1
fi
ldconfig_output="$(ldconfig -p)"
if ! grep -q 'libvulkan\.so' <<<"${ldconfig_output}"; then
    echo "ERROR: Vulkan loader was not installed." >&2
    exit 1
fi

echo "Vulkan build environment ready:"
glslc_version="$(glslc --version)"
echo "  glslc: $(command -v glslc) (${glslc_version%%$'\n'*})"
echo "  SPIR-V headers: ${spirv_header}"
echo "  loader: $(awk '/libvulkan\.so\.1/{print $NF; exit}' <<<"${ldconfig_output}")"
if compgen -G '/usr/share/vulkan/icd.d/*lvp*.json' >/dev/null; then
    echo "  hosted-CI software ICD: Mesa lavapipe"
fi