#!/usr/bin/env bash
# Silent-install smoke for the packaged ACloudViewer macOS installer (Qt IFW
# DMG). Extracted from the macos.yml "Install ACloudViewer (Qt IFW silent
# install from DMG)" step; the workflow keeps the pytest suites, this script
# owns DMG attach / installer copy off the volume / silent install / detach
# retry / binary detection.
#
# Usage (CI and local):
#   bash scripts/platforms/mac/test_installer.sh \
#       --installer-dir "<dir with ACloudViewer-*.dmg>" [--install-dir "<target>"]
#   bash scripts/platforms/mac/test_installer.sh --cleanup [--install-dir "<target>"]
# The resolved binary is exported as ACV_BINARY (in-process and, on GitHub
# Actions, via GITHUB_ENV; PATH additions go to GITHUB_PATH).

set -euo pipefail

INSTALLER_DIR=""
INSTALL_DIR="$HOME/ACloudViewer"
CLEANUP=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --installer-dir) INSTALLER_DIR="$2"; shift 2 ;;
        --install-dir)   INSTALL_DIR="$2"; shift 2 ;;
        --cleanup)       CLEANUP=1; shift ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

export_step_env() { # persist for later workflow steps; local run keeps process env
    local name="$1" value="$2"
    if [[ -n "${GITHUB_ENV:-}" ]]; then
        printf '%s=%s\n' "$name" "$value" >> "$GITHUB_ENV"
    fi
    export "$name=$value"
}

prepend_step_path() {
    local dir="$1"
    if [[ -n "${GITHUB_PATH:-}" ]]; then
        printf '%s\n' "$dir" >> "$GITHUB_PATH"
    fi
    case ":$PATH:" in
        *":$dir:"*) ;;
        *) export PATH="$dir:$PATH" ;;
    esac
}

if [[ "$CLEANUP" -eq 1 ]]; then
    rm -rf "$INSTALL_DIR" "$HOME/acv-installer-local.app"
    exit 0
fi

if [[ -z "$INSTALLER_DIR" ]]; then
    echo "::error::--installer-dir is required (or pass --cleanup)" >&2
    exit 1
fi

DMG=$(find "$INSTALLER_DIR" -name "*.dmg" | head -1)
if [[ -z "$DMG" ]]; then
    echo "::error::No .dmg installer found" >&2
    exit 1
fi
hdiutil attach "$DMG" -nobrowse -noverify
VOLUME=$(find /Volumes -maxdepth 1 -name "ACloudViewer*" | head -1)
if [[ -z "$VOLUME" ]]; then
    echo "::error::DMG did not mount as expected" >&2
    exit 1
fi
echo "=== DMG contents ==="
ls -la "$VOLUME"/

INSTALLER_APP=$(find "$VOLUME" -maxdepth 1 -name "*.app" | head -1)
if [[ -z "$INSTALLER_APP" ]]; then
    echo "::error::No installer .app found in DMG" >&2
    hdiutil detach "$VOLUME" || true
    exit 1
fi

# Copy the installer bundle off the DMG volume before running it. A process
# whose binary or cwd lives on the mount (Qt IFW spawns helpers like
# touch/lsregister, and Spotlight may index the volume) keeps it busy, so
# `hdiutil detach` fails with "Resource busy" (exit 16) right after a
# successful install.
INSTALLER_LOCAL="$HOME/acv-installer-local.app"
ditto "$INSTALLER_APP" "$INSTALLER_LOCAL"
INSTALLER_BIN=$(find "$INSTALLER_LOCAL/Contents/MacOS" -type f -perm +111 | head -1)
echo "Installer binary: $INSTALLER_BIN"

"$INSTALLER_BIN" --root "$INSTALL_DIR" \
    --accept-licenses --accept-messages --confirm-command install

# Detach with retry; fall back to -force for transient system references
# (mds/Spotlight, LaunchServices). Installation is already complete at this
# point, so never fail the job here.
cd "$HOME"
if ! hdiutil detach "$VOLUME" -quiet; then
    echo "::warning::hdiutil detach busy, retrying after 3s"
    sleep 3
    hdiutil detach "$VOLUME" -quiet \
        || hdiutil detach "$VOLUME" -force \
        || echo "::warning::forced detach failed; leaving volume mounted"
fi

echo "=== Installed contents ==="
ls -la "$INSTALL_DIR"/

if [[ -d "$INSTALL_DIR/ACloudViewer.app" ]]; then
    prepend_step_path "$INSTALL_DIR/ACloudViewer.app/Contents/MacOS"
    export_step_env "ACV_BINARY" "$INSTALL_DIR/ACloudViewer.app/Contents/MacOS/ACloudViewer"
elif [[ -f "$INSTALL_DIR/ACloudViewer" ]]; then
    prepend_step_path "$INSTALL_DIR"
    export_step_env "ACV_BINARY" "$INSTALL_DIR/ACloudViewer"
else
    echo "::error::ACloudViewer binary not found after install" >&2
    find "$INSTALL_DIR" -name "ACloudViewer*" -type f 2>/dev/null || true
    exit 1
fi

echo "Installer smoke passed: $(basename "$DMG") -> $ACV_BINARY"
