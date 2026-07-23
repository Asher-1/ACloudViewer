#!/usr/bin/env bash

set -euo pipefail

VULKAN_SDK_VERSION=${VULKAN_SDK_VERSION:-1.4.350.0}
VULKAN_SDK_ROOT=${VULKAN_SDK_ROOT:-${HOME}/VulkanSDK/${VULKAN_SDK_VERSION}}
VULKAN_SDK_SHA256=${VULKAN_SDK_SHA256:-}

if [[ "${VULKAN_SDK_VERSION}" == "1.4.350.0" ]]; then
    VULKAN_SDK_SHA256=${VULKAN_SDK_SHA256:-7acc181b8fd9b4781bf51ed086222ec95d22004b85b3d0a6683a7e48ca5a1679}
elif [[ -z "${VULKAN_SDK_SHA256}" ]]; then
    echo "ERROR: set VULKAN_SDK_SHA256 when overriding VULKAN_SDK_VERSION." >&2
    exit 1
fi

find_setup_script() {
    find "${VULKAN_SDK_ROOT}" -maxdepth 3 -name setup-env.sh -type f -print -quit 2>/dev/null
}

setup_script="$(find_setup_script)"
if [[ -z "${setup_script}" || "${VULKAN_SDK_FORCE_INSTALL:-0}" == "1" ]]; then
    work_dir="$(mktemp -d "${TMPDIR:-/tmp}/acloudviewer-vulkan.XXXXXX")"
    trap 'rm -rf "${work_dir}"' EXIT
    archive="${work_dir}/vulkan_sdk.zip"
    extract_dir="${work_dir}/sdk"
    mkdir -p "${extract_dir}"

    url="https://sdk.lunarg.com/sdk/download/${VULKAN_SDK_VERSION}/mac/vulkan_sdk.zip"
    echo "Downloading Vulkan SDK ${VULKAN_SDK_VERSION} from ${url}"
    curl --fail --location --retry 3 "${url}" --output "${archive}"
    actual_sha256="$(shasum -a 256 "${archive}" | awk '{print $1}')"
    if [[ "${actual_sha256}" != "${VULKAN_SDK_SHA256}" ]]; then
        echo "ERROR: Vulkan SDK SHA256 mismatch: got ${actual_sha256}, expected ${VULKAN_SDK_SHA256}." >&2
        exit 1
    fi
    ditto -x -k "${archive}" "${extract_dir}"

    installer="$(find "${extract_dir}" -type f -path '*.app/Contents/MacOS/*' \
        -iname '*vulkan*' ! -name MaintenanceTool -perm +111 -print -quit)"
    if [[ -z "${installer}" ]]; then
        echo "ERROR: Vulkan SDK installer executable was not found in the archive." >&2
        exit 1
    fi

    "${installer}" \
        --root "${VULKAN_SDK_ROOT}" \
        --accept-licenses \
        --default-answer \
        --confirm-command \
        install copy_only=1
    setup_script="$(find_setup_script)"
fi

if [[ -z "${setup_script}" ]]; then
    echo "ERROR: setup-env.sh was not installed under ${VULKAN_SDK_ROOT}." >&2
    exit 1
fi

# shellcheck disable=SC1090
source "${setup_script}"

glslc_path="$(command -v glslc || true)"
spirv_header="${VULKAN_SDK}/include/spirv/unified1/spirv.hpp"
moltenvk_icd="$(find "${VULKAN_SDK}/share/vulkan/icd.d" \
    -maxdepth 1 -iname '*moltenvk*.json' -type f -print -quit 2>/dev/null)"
if [[ -z "${glslc_path}" || ! -f "${spirv_header}" || -z "${moltenvk_icd}" ]]; then
    echo "ERROR: incomplete Vulkan SDK at ${VULKAN_SDK}." >&2
    exit 1
fi

env_script="${VULKAN_SDK_ROOT}/acloudviewer-vulkan-env.sh"
{
    printf 'export VULKAN_SDK="%s"\n' "${VULKAN_SDK}"
    printf 'export PATH="%s/bin:${PATH}"\n' "${VULKAN_SDK}"
    printf 'export DYLD_LIBRARY_PATH="%s/lib${DYLD_LIBRARY_PATH:+:${DYLD_LIBRARY_PATH}}"\n' "${VULKAN_SDK}"
    printf 'export VK_ADD_DRIVER_FILES="%s"\n' "${moltenvk_icd}"
} >"${env_script}"

if [[ -n "${GITHUB_ENV:-}" ]]; then
    {
        echo "VULKAN_SDK=${VULKAN_SDK}"
        echo "DYLD_LIBRARY_PATH=${VULKAN_SDK}/lib${DYLD_LIBRARY_PATH:+:${DYLD_LIBRARY_PATH}}"
        echo "VK_ADD_DRIVER_FILES=${moltenvk_icd}"
    } >>"${GITHUB_ENV}"
fi
if [[ -n "${GITHUB_PATH:-}" ]]; then
    echo "${VULKAN_SDK}/bin" >>"${GITHUB_PATH}"
fi

echo "Vulkan SDK ready: ${VULKAN_SDK}"
echo "  glslc: ${glslc_path}"
echo "  SPIR-V headers: ${spirv_header}"
echo "  MoltenVK ICD: ${moltenvk_icd}"
echo "For a new local shell run: source ${env_script}"
glslc --version
