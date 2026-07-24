#!/usr/bin/env bash
# Shared helpers for ACloudViewer Vulkan build-environment scripts (Linux/macOS).

acloudviewer_vulkan_local_root() {
    printf '%s\n' "${ACLOUDVIEWER_LOCAL_ROOT:-${HOME}/.local/share/acloudviewer}"
}

acloudviewer_vulkan_env_script_path() {
    printf '%s\n' "$(acloudviewer_vulkan_local_root)/acloudviewer-vulkan-env.sh"
}

# LunarG setup-env.sh references $1; avoid "unbound variable" under set -u.
resolve_vulkan_sdk_from_setup_script() {
    local setup_script="$1"
    local sdk_tree arch candidate

    sdk_tree="$(cd "$(dirname "${setup_script}")" && pwd)"
    arch="$(uname -m)"
    for candidate in "${sdk_tree}/${arch}" "${sdk_tree}"; do
        if [[ -f "${candidate}/include/vulkan/vulkan_core.h" ]]; then
            printf '%s\n' "${candidate}"
            return 0
        fi
    done
    return 1
}

remove_vulkan_hook_block() {
    local target="$1"
    local marker="$2"
    local end_marker="$3"

    [[ -f "${target}" ]] || return 0
    if ! grep -qF "${marker}" "${target}"; then
        return 0
    fi
    awk -v start="${marker}" -v end="${end_marker}" '
        $0 == start { skip=1; next }
        $0 == end { skip=0; next }
        !skip { print }
    ' "${target}" >"${target}.acloudviewer.tmp"
    mv "${target}.acloudviewer.tmp" "${target}"
}

prepend_vulkan_hook_block() {
    local target="$1"
    local hook_file="$2"

    touch "${target}"
    local combined
    combined="$(mktemp)"
    cat "${hook_file}" "${target}" >"${combined}"
    mv "${combined}" "${target}"
}

# Source the generated env file when present (safe under set -u in callers).
source_acloudviewer_vulkan_env() {
    local env_script
    env_script="${ACLOUDVIEWER_VULKAN_ENV:-$(acloudviewer_vulkan_env_script_path)}"
    if [[ ! -f "${env_script}" ]]; then
        return 1
    fi
    # shellcheck disable=SC1090
    source "${env_script}"
    return 0
}

append_bashrc_vulkan_hook() {
    local env_script update_bashrc marker end_marker hook_file
    env_script="$1"
    update_bashrc="${2:-${ACLOUDVIEWER_UPDATE_BASHRC:-1}}"
    if [[ "${update_bashrc}" != "1" ]]; then
        echo "Skipped shell profile update (ACLOUDVIEWER_UPDATE_BASHRC=${update_bashrc})."
        echo "Run manually: source ${env_script}"
        return
    fi

    marker="# >>> acloudviewer-vulkan >>>"
    end_marker="# <<< acloudviewer-vulkan <<<"
    hook_file="$(mktemp)"
    cat >"${hook_file}" <<EOF

${marker}
if [[ -f "${env_script}" ]]; then
  source "${env_script}"
fi
${end_marker}
EOF

    local target
    for target in "${HOME}/.bashrc" "${HOME}/.profile"; do
        remove_vulkan_hook_block "${target}" "${marker}" "${end_marker}"
        prepend_vulkan_hook_block "${target}" "${hook_file}"
        echo "Installed Vulkan env hook at top of ${target}"
    done
    rm -f "${hook_file}"

    cat <<EOF
Note: Ubuntu/Debian ~/.bashrc returns early in non-interactive shells, so
  source ~/.bashrc
does not load hooks appended at the bottom. The hook is now at the top of
~/.bashrc and ~/.profile. For scripts, prefer:
  source ${env_script}
EOF
}
