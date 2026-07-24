#!/usr/bin/env bash
# Install build-time Vulkan dependencies for ggml on Linux (Ubuntu 20.04+).
# Writes ~/.local/share/acloudviewer/acloudviewer-vulkan-env.sh
#
# Usage: install_vulkan_linux.sh [--skip-bashrc]

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${script_dir}/acloudviewer_vulkan_env_common.sh"

ACLOUDVIEWER_LOCAL_ROOT="$(acloudviewer_vulkan_local_root)"
ENV_SCRIPT="$(acloudviewer_vulkan_env_script_path)"
VULKAN_SDK_VERSION="${VULKAN_SDK_VERSION:-1.4.350.0}"
VULKAN_SDK_ROOT="${VULKAN_SDK_ROOT:-${HOME}/VulkanSDK/${VULKAN_SDK_VERSION}}"
SPIRV_HEADERS_TAG="${SPIRV_HEADERS_TAG:-vulkan-sdk-1.4.350.0}"
MIN_VK_HEADER_VERSION="${MIN_VK_HEADER_VERSION:-235}"
UPDATE_BASHRC="${ACLOUDVIEWER_UPDATE_BASHRC:-1}"

if [[ "${1:-}" == "--skip-bashrc" ]]; then
    UPDATE_BASHRC=0
fi

SUDO=${SUDO:=sudo}

version_at_least() {
    dpkg --compare-versions "$1" ge "$2"
}

vk_header_version() {
    local header="${1:-/usr/include/vulkan/vulkan_core.h}"
    if [[ ! -f "${header}" ]]; then
        echo 0
        return
    fi
    awk '/^#define VK_HEADER_VERSION / { print $3; exit }' "${header}"
}

needs_bundled_vulkan_headers() {
    local version distribution
    version="$(vk_header_version)"
    if [[ -f /etc/os-release ]]; then
        # shellcheck disable=SC1091
        source /etc/os-release
        distribution="${VERSION_ID:-}"
    else
        distribution=""
    fi
    if [[ "${distribution}" == "20.04" || "${distribution}" == "22.04" ]]; then
        return 0
    fi
    [[ "${version}" -lt "${MIN_VK_HEADER_VERSION}" ]]
}

find_vulkan_setup_script() {
    [[ -d "${VULKAN_SDK_ROOT}" ]] || return 0
    find "${VULKAN_SDK_ROOT}" -maxdepth 3 -name setup-env.sh -type f -print -quit 2>/dev/null || true
}

fetch_vulkan_sdk_sha256() {
    if [[ -n "${VULKAN_SDK_SHA256:-}" ]]; then
        echo "${VULKAN_SDK_SHA256}"
        return
    fi
    curl --fail --location --retry 3 --silent \
        "https://sdk.lunarg.com/sdk/sha/${VULKAN_SDK_VERSION}/linux/vulkan_sdk.tar.xz.txt" \
        | awk '{print $1}'
}

install_glslc_from_source() {
    local shaderc_tag="${SHADERC_TAG:-v2026.3}"
    local shaderc_commit="${SHADERC_COMMIT:-82757e4f72af8518fb679604d352623450cb761f}"
    local build_root="${SHADERC_BUILD_ROOT:-${ACLOUDVIEWER_LOCAL_ROOT}/build/shaderc-${shaderc_tag}}"
    local source_dir="${build_root}/shaderc"
    local build_dir="${build_root}/build"
    local install_dir="${HOME}/.local/bin"
    local cmake_bin

    cmake_bin="$(command -v cmake)"
    local cmake_version
    cmake_version="$("${cmake_bin}" --version | awk 'NR == 1 {print $3}')"
    if ! version_at_least "${cmake_version}" "3.22.1"; then
        echo "System CMake ${cmake_version} is too old for ${shaderc_tag}; installing a build-only CMake." >&2
        python3 -m pip install --user --upgrade "cmake==3.31.6"
        cmake_bin="${HOME}/.local/bin/cmake"
    fi

    if [[ ! -d "${source_dir}/.git" ]]; then
        rm -rf "${build_root}"
        mkdir -p "${build_root}"
        git clone --branch "${shaderc_tag}" --depth 1 \
            https://github.com/google/shaderc.git "${source_dir}"
    fi
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
    mkdir -p "${install_dir}"
    install -m 0755 "${build_dir}/glslc/glslc" "${install_dir}/glslc"
}

glslc_is_usable() {
    local candidate="$1"
    [[ -n "${candidate}" && -x "${candidate}" ]] || return 1
    "${candidate}" --version >/dev/null 2>&1
}

# Prefer LunarG SDK glslc when runnable (24.04+). On focal/22.04 the SDK binary
# often requires GLIBC 2.34+; fall back to distro glslc or a local shaderc build.
resolve_glslc() {
    local vulkan_sdk_path="$1"
    local candidate

    if [[ -n "${vulkan_sdk_path}" && -x "${vulkan_sdk_path}/bin/glslc" ]]; then
        candidate="${vulkan_sdk_path}/bin/glslc"
        if glslc_is_usable "${candidate}"; then
            printf '%s\n' "${candidate}"
            return 0
        fi
        echo "WARNING: ${candidate} is not runnable on this system (often GLIBC too old on 20.04); trying other glslc sources." >&2
    fi
    if command -v glslc >/dev/null 2>&1; then
        candidate="$(command -v glslc)"
        if glslc_is_usable "${candidate}"; then
            printf '%s\n' "${candidate}"
            return 0
        fi
    fi
    if [[ -x "${HOME}/.local/bin/glslc" ]] && glslc_is_usable "${HOME}/.local/bin/glslc"; then
        printf '%s\n' "${HOME}/.local/bin/glslc"
        return 0
    fi
    if apt-cache show glslc >/dev/null 2>&1; then
        ${SUDO} apt-get install -y glslc
        candidate="$(command -v glslc || true)"
        if glslc_is_usable "${candidate}"; then
            printf '%s\n' "${candidate}"
            return 0
        fi
    fi
    echo "Building pinned shaderc (glslc) for this Ubuntu release."
    install_glslc_from_source
    printf '%s\n' "${HOME}/.local/bin/glslc"
}

install_spirv_headers() {
    local install_prefix="${ACLOUDVIEWER_LOCAL_ROOT}/spirv-headers"
    local config="${install_prefix}/share/cmake/SPIRV-Headers/SPIRV-HeadersConfig.cmake"
    if [[ -f "${config}" ]]; then
        return
    fi

    local build_root="${ACLOUDVIEWER_LOCAL_ROOT}/build/spirv-headers-${SPIRV_HEADERS_TAG}"
    local source_dir="${build_root}/SPIRV-Headers"
    local build_dir="${build_root}/build"
    local cmake_bin
    cmake_bin="$(command -v cmake)"

    rm -rf "${build_root}"
    mkdir -p "${build_root}"
    git clone --branch "${SPIRV_HEADERS_TAG}" --depth 1 \
        https://github.com/KhronosGroup/SPIRV-Headers.git "${source_dir}"

    "${cmake_bin}" -S "${source_dir}" -B "${build_dir}" -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="${install_prefix}"
    "${cmake_bin}" --build "${build_dir}" --target install \
        --parallel "${VULKAN_BUILD_JOBS:-4}"
    rm -rf "${build_root}"

    if [[ ! -f "${config}" ]]; then
        echo "ERROR: SPIRV-Headers CMake package was not installed to ${install_prefix}." >&2
        exit 1
    fi
}

install_vulkan_sdk_tarball() {
    local setup_script
    setup_script="$(find_vulkan_setup_script)"
    if [[ -n "${setup_script}" && "${VULKAN_SDK_FORCE_INSTALL:-0}" != "1" ]]; then
        return
    fi

    local work_dir archive extract_dir sha256 actual_sha256 url tarball_name
    work_dir="$(mktemp -d "${TMPDIR:-/tmp}/acloudviewer-vulkan.XXXXXX")"
    tarball_name="vulkansdk-linux-x86_64-${VULKAN_SDK_VERSION}.tar.xz"
    url="https://sdk.lunarg.com/sdk/download/${VULKAN_SDK_VERSION}/linux/${tarball_name}"
    archive="${work_dir}/${tarball_name}"
    extract_dir="${work_dir}/extract"

    sha256="$(fetch_vulkan_sdk_sha256)"
    echo "Downloading Vulkan SDK ${VULKAN_SDK_VERSION} headers/tools from ${url}"
    curl --fail --location --retry 3 "${url}" --output "${archive}"
    actual_sha256="$(sha256sum "${archive}" | awk '{print $1}')"
    if [[ "${actual_sha256}" != "${sha256}" ]]; then
        echo "ERROR: Vulkan SDK SHA256 mismatch: got ${actual_sha256}, expected ${sha256}." >&2
        exit 1
    fi

    mkdir -p "${extract_dir}" "${VULKAN_SDK_ROOT}"
    tar -xJf "${archive}" -C "${extract_dir}"
    rm -rf "${VULKAN_SDK_ROOT}"
    mv "${extract_dir}/"* "${VULKAN_SDK_ROOT}/"
    rm -rf "${work_dir}"

    setup_script="$(find_vulkan_setup_script)"
    if [[ -z "${setup_script}" ]]; then
        echo "ERROR: setup-env.sh was not found under ${VULKAN_SDK_ROOT}." >&2
        exit 1
    fi
}

write_env_script() {
    local spirv_cmake_dir="${ACLOUDVIEWER_LOCAL_ROOT}/spirv-headers/share/cmake/SPIRV-Headers"
    local setup_script vulkan_sdk_path glslc_path loader_path
    setup_script="$(find_vulkan_setup_script || true)"
    vulkan_sdk_path=""
    if [[ -n "${setup_script}" ]]; then
        vulkan_sdk_path="$(resolve_vulkan_sdk_from_setup_script "${setup_script}")"
        if [[ ! -f "${vulkan_sdk_path}/include/vulkan/vulkan_core.h" ]]; then
            echo "WARNING: stale Vulkan SDK path ${vulkan_sdk_path}; omitting VULKAN_SDK from env script." >&2
            vulkan_sdk_path=""
        fi
    fi

    glslc_path="$(resolve_glslc "${vulkan_sdk_path}")"
    loader_path="$(awk '/libvulkan\.so\.1/{print $NF; exit}' < <(ldconfig -p 2>/dev/null || true))"

    mkdir -p "${ACLOUDVIEWER_LOCAL_ROOT}"
    {
        printf '# Generated by util/install_vulkan_linux.sh — do not edit by hand.\n'
        printf 'export ACLOUDVIEWER_LOCAL_ROOT="%s"\n' "${ACLOUDVIEWER_LOCAL_ROOT}"
        if [[ -n "${vulkan_sdk_path}" ]]; then
            printf 'export VULKAN_SDK="%s"\n' "${vulkan_sdk_path}"
            printf 'export PATH="%s/bin:${PATH}"\n' "${vulkan_sdk_path}"
        fi
        printf 'export ACLOUDVIEWER_SPIRV_HEADERS_DIR="%s"\n' "${spirv_cmake_dir}"
        if [[ -n "${glslc_path}" ]]; then
            printf 'export ACLOUDVIEWER_GLSLC="%s"\n' "${glslc_path}"
            printf 'case ":${PATH}:" in\n'
            printf '  *":%s:"*) ;;\n' "$(dirname "${glslc_path}")"
            printf '  *) export PATH="%s:${PATH}" ;;\n' "$(dirname "${glslc_path}")"
            printf 'esac\n'
        fi
        if [[ -n "${loader_path}" ]]; then
            printf 'export ACLOUDVIEWER_VULKAN_LIBRARY="%s"\n' "${loader_path}"
        fi
    } >"${ENV_SCRIPT}"
}

verify_environment() {
    # shellcheck disable=SC1090
    source "${ENV_SCRIPT}"

    if [[ -z "${ACLOUDVIEWER_GLSLC:-}" || ! -x "${ACLOUDVIEWER_GLSLC}" ]]; then
        echo "ERROR: glslc is missing after setup." >&2
        exit 1
    fi
    if [[ ! -f "${ACLOUDVIEWER_SPIRV_HEADERS_DIR}/SPIRV-HeadersConfig.cmake" ]]; then
        echo "ERROR: SPIRV-Headers CMake package is missing." >&2
        exit 1
    fi
    if needs_bundled_vulkan_headers; then
        if [[ -z "${VULKAN_SDK:-}" || ! -f "${VULKAN_SDK}/include/vulkan/vulkan_core.h" ]]; then
            echo "ERROR: bundled Vulkan headers are required on this OS but VULKAN_SDK is missing." >&2
            echo "Re-run: util/install_vulkan_env.sh" >&2
            exit 1
        fi
    elif [[ -n "${VULKAN_SDK:-}" && ! -f "${VULKAN_SDK}/include/vulkan/vulkan_core.h" ]]; then
        echo "WARNING: VULKAN_SDK=${VULKAN_SDK} does not exist; unset it and re-run util/install_vulkan_env.sh." >&2
    fi
    if [[ -n "${ACLOUDVIEWER_VULKAN_LIBRARY:-}" && -f "${ACLOUDVIEWER_VULKAN_LIBRARY}" ]]; then
        :
    elif ! ldconfig -p 2>/dev/null | grep -q 'libvulkan\.so'; then
        echo "ERROR: Vulkan loader was not installed (install libvulkan1 / mesa-vulkan-drivers)." >&2
        exit 1
    fi

    echo "Vulkan build environment ready:"
    echo "  env script: ${ENV_SCRIPT}"
    echo "  glslc: ${ACLOUDVIEWER_GLSLC} ($(${ACLOUDVIEWER_GLSLC} --version | head -1))"
    echo "  SPIR-V headers: ${ACLOUDVIEWER_SPIRV_HEADERS_DIR}"
    if [[ -n "${VULKAN_SDK:-}" ]]; then
        echo "  VULKAN_SDK: ${VULKAN_SDK} (VK_HEADER_VERSION=$(vk_header_version "${VULKAN_SDK}/include/vulkan/vulkan_core.h"))"
    else
        echo "  Vulkan headers: system (/usr/include, VK_HEADER_VERSION=$(vk_header_version))"
    fi
    echo "  loader: $(awk '/libvulkan\.so\.1/{print $NF; exit}' < <(ldconfig -p))"
    echo "For a new shell: source ${ENV_SCRIPT}"
}

mkdir -p "${ACLOUDVIEWER_LOCAL_ROOT}"
install_spirv_headers
if needs_bundled_vulkan_headers; then
    echo "System Vulkan headers are too old for ggml 0.17; installing LunarG SDK ${VULKAN_SDK_VERSION}."
    install_vulkan_sdk_tarball
fi
write_env_script
append_bashrc_vulkan_hook "${ENV_SCRIPT}" "${UPDATE_BASHRC}"
verify_environment
