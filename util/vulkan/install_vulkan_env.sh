#!/usr/bin/env bash
# Cross-platform entry point for ggml Vulkan *build* dependencies.
#
# Linux/macOS: run this script.
# Windows:     .\\util\\vulkan\\install_vulkan_sdk_windows.ps1

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

case "$(uname -s)" in
    Linux)
        exec bash "${script_dir}/install_vulkan_linux.sh" "$@"
        ;;
    Darwin)
        echo "Vulkan is not supported on macOS. Use Metal (-DAICore_USE_VULKAN=OFF)." >&2
        exit 1
        ;;
    *)
        echo "Unsupported platform for ${0}." >&2
        echo "On Windows run: .\\util\\vulkan\\install_vulkan_sdk_windows.ps1" >&2
        exit 1
        ;;
esac
