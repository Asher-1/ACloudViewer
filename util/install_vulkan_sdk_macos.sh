#!/usr/bin/env bash
# Install build-time Vulkan dependencies for ggml on macOS.
# Writes ~/.local/share/acloudviewer/acloudviewer-vulkan-env.sh
#
# CI: humbletim/install-vulkan-sdk (pin post-v1.2 for vulkansdk-macOS-*) then util/sync_vulkan_env_from_sdk.sh
# Local: run this script directly (supports LunarG 1.4.350.0 vulkansdk-macOS-* bundles).

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${script_dir}/acloudviewer_vulkan_env_common.sh"

VULKAN_SDK_VERSION=${VULKAN_SDK_VERSION:-1.4.350.0}
VULKAN_SDK_ROOT=${VULKAN_SDK_ROOT:-${VULKAN_SDK:-${HOME}/VulkanSDK/${VULKAN_SDK_VERSION}}}
VULKAN_SDK_SHA256=${VULKAN_SDK_SHA256:-}
ACLOUDVIEWER_LOCAL_ROOT="$(acloudviewer_vulkan_local_root)"
UPDATE_BASHRC="${ACLOUDVIEWER_UPDATE_BASHRC:-1}"
if [[ -n "${GITHUB_ACTIONS:-}" ]]; then
    UPDATE_BASHRC=0
fi
export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-minimal}"

if [[ "${VULKAN_SDK_VERSION}" == "1.4.350.0" ]]; then
    VULKAN_SDK_SHA256=${VULKAN_SDK_SHA256:-7acc181b8fd9b4781bf51ed086222ec95d22004b85b3d0a6683a7e48ca5a1679}
elif [[ -z "${VULKAN_SDK_SHA256}" ]]; then
    echo "ERROR: set VULKAN_SDK_SHA256 when overriding VULKAN_SDK_VERSION." >&2
    exit 1
fi

find_mac_installer_app() {
    local extract_dir="$1"
    local candidate
    for candidate in \
        "${extract_dir}/vulkansdk-macOS-${VULKAN_SDK_VERSION}.app" \
        "${extract_dir}/InstallVulkan-${VULKAN_SDK_VERSION}.app" \
        "${extract_dir}/InstallVulkan.app"; do
        if [[ -d "${candidate}/Contents/MacOS" ]]; then
            printf '%s\n' "${candidate}"
            return 0
        fi
    done
    return 1
}

find_mac_installer_binary() {
    local app_dir="$1"
    local macos_dir="${app_dir}/Contents/MacOS"
    local entry
    for entry in \
        "${macos_dir}/vulkansdk-macOS-${VULKAN_SDK_VERSION}" \
        "${macos_dir}/InstallVulkan-${VULKAN_SDK_VERSION}" \
        "${macos_dir}/InstallVulkan"; do
        if [[ -x "${entry}" ]]; then
            printf '%s\n' "${entry}"
            return 0
        fi
    done
    find "${macos_dir}" -maxdepth 1 -type f ! -name MaintenanceTool -perm -111 -print -quit 2>/dev/null
}

sdk_tree_ready() {
    local sdk_root="$1"
    [[ -x "${sdk_root}/bin/glslc" ]] \
        && [[ -f "${sdk_root}/include/vulkan/vulkan_core.h" ]] \
        && [[ -f "${sdk_root}/include/spirv/unified1/spirv.hpp" ]]
}

install_mac_vulkan_sdk() {
    local dest="$1"
    local work_dir archive extract_dir app_dir installer sdk_temp
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

    app_dir="$(find_mac_installer_app "${extract_dir}")" || {
        echo "ERROR: Vulkan SDK installer .app was not found in the archive." >&2
        exit 1
    }
    installer="$(find_mac_installer_binary "${app_dir}")" || {
        echo "ERROR: Vulkan SDK installer executable was not found in ${app_dir}." >&2
        exit 1
    }

    sdk_temp="${dest}.tmp"
    rm -rf "${sdk_temp}" "${dest}"
    mkdir -p "${sdk_temp}" "${dest}"

    echo "Running Vulkan SDK installer: ${installer}"
    "${installer}" \
        --root "${sdk_temp}" \
        --accept-licenses \
        --default-answer \
        --confirm-command \
        install

    if [[ -d "${sdk_temp}/macOS" ]]; then
        cp -R "${sdk_temp}/macOS/"* "${dest}/"
    elif [[ -f "${sdk_temp}/setup-env.sh" ]]; then
        cp -R "${sdk_temp}/"* "${dest}/"
    else
        echo "ERROR: unrecognized Vulkan SDK install layout under ${sdk_temp}." >&2
        ls -la "${sdk_temp}" >&2 || true
        exit 1
    fi
    rm -rf "${sdk_temp}"
}

if [[ -n "${VULKAN_SDK:-}" ]] && sdk_tree_ready "${VULKAN_SDK}"; then
    echo "Using preinstalled Vulkan SDK at ${VULKAN_SDK}"
    exec bash "${script_dir}/sync_vulkan_env_from_sdk.sh"
fi

if sdk_tree_ready "${VULKAN_SDK_ROOT}" && [[ "${VULKAN_SDK_FORCE_INSTALL:-0}" != "1" ]]; then
    export VULKAN_SDK="${VULKAN_SDK_ROOT}"
    echo "Reusing existing Vulkan SDK at ${VULKAN_SDK_ROOT}"
    exec bash "${script_dir}/sync_vulkan_env_from_sdk.sh"
fi

install_mac_vulkan_sdk "${VULKAN_SDK_ROOT}"
export VULKAN_SDK="${VULKAN_SDK_ROOT}"
exec bash "${script_dir}/sync_vulkan_env_from_sdk.sh"
