#!/bin/bash
# build_qt_ifw.sh — Build Qt Installer Framework from source for macOS ARM64
#
# Background:
#   Qt IFW official pre-built binaries are x86_64 only (last: 4.8.1).
#   Starting with 4.11.0, only source archives are published.
#   GitHub CI macos-14+ runners are ARM64-only; Rosetta is deprecated.
#   This script builds a native ARM64 binarycreator from IFW sources
#   using the Qt 6 toolchain already available in the conda environment.
#
# Prerequisites (CI):
#   - conda env activated with qt6-main (provides qmake6, Qt 6.x libs)
#   - brew install xz  (provides liblzma for libarchive)
#   - Standard macOS SDK (bzip2, zlib, libiconv are system-provided)
#
# Usage:
#   source scripts/platforms/mac/build_qt_ifw.sh
#   # Then binarycreator is available at $QT_IFW_INSTALL_DIR/bin/binarycreator
#
# Or run directly:
#   bash scripts/platforms/mac/build_qt_ifw.sh
#
# Output:
#   ~/qt-ifw-install/bin/binarycreator  (and other IFW tools)
#
# Environment variables:
#   QT_IFW_VERSION    - IFW version to build (default: 4.8.1)
#   QT_IFW_INSTALL_DIR - Override install prefix (default: ~/qt-ifw-install)
#   QT_IFW_BUILD_DIR  - Override build directory (default: /tmp/qt-ifw-build)

set -euo pipefail

QT_IFW_VERSION="${QT_IFW_VERSION:-4.8.1}"
QT_IFW_INSTALL_DIR="${QT_IFW_INSTALL_DIR:-${HOME}/qt-ifw-install}"
QT_IFW_BUILD_DIR="${QT_IFW_BUILD_DIR:-/tmp/qt-ifw-build}"
QT_IFW_SRC_DIR="/tmp/qt-ifw-src"

# ── Skip if already built (CI cache hit) ──────────────────────────
if [[ -x "${QT_IFW_INSTALL_DIR}/bin/binarycreator" ]]; then
    echo "Qt IFW ${QT_IFW_VERSION} already installed at ${QT_IFW_INSTALL_DIR}"
    file "${QT_IFW_INSTALL_DIR}/bin/binarycreator" || true
    exit 0
fi

echo "=============================================="
echo " Building Qt Installer Framework ${QT_IFW_VERSION}"
echo " for macOS $(uname -m) (Apple Silicon)"
echo "=============================================="

# ── 1. Locate qmake from conda environment ────────────────────────
# conda qt6-main provides 'qmake6'; some envs may also symlink 'qmake'
QMAKE=""
for candidate in qmake6 qmake; do
    if command -v "$candidate" &>/dev/null; then
        QMAKE="$(command -v "$candidate")"
        break
    fi
done

if [[ -z "${QMAKE}" ]]; then
    echo "ERROR: qmake not found. Activate a conda env with qt6-main first."
    echo "  conda activate cloudViewer"
    exit 1
fi

QT_VERSION=$("${QMAKE}" -query QT_VERSION 2>/dev/null || echo "unknown")
QT_PREFIX=$("${QMAKE}" -query QT_INSTALL_PREFIX 2>/dev/null || echo "unknown")
echo "Using qmake: ${QMAKE}"
echo "Qt version:  ${QT_VERSION}"
echo "Qt prefix:   ${QT_PREFIX}"

# Verify Qt 6
QT_MAJOR=$("${QMAKE}" -query QT_VERSION 2>/dev/null | cut -d. -f1)
if [[ "${QT_MAJOR}" != "6" ]]; then
    echo "ERROR: Qt 6 is required, found Qt ${QT_VERSION}"
    exit 1
fi

# ── 2. Install build dependencies ─────────────────────────────────
echo ""
echo "=== Checking build dependencies ==="

# xz (liblzma) is required for libarchive inside IFW
if ! brew list xz &>/dev/null 2>&1; then
    echo "Installing xz (liblzma) via Homebrew..."
    brew install xz
fi
BREW_PREFIX="$(brew --prefix)"
LZMA_PREFIX="${BREW_PREFIX}/opt/xz"
echo "liblzma: ${LZMA_PREFIX}"

# bzip2 — system-provided on macOS
# zlib — system-provided on macOS
# libiconv — system-provided on macOS

# ── 3. Download IFW source ────────────────────────────────────────
echo ""
echo "=== Downloading Qt IFW ${QT_IFW_VERSION} source ==="

IFW_URL="https://download.qt.io/official_releases/qt-installer-framework/${QT_IFW_VERSION}/installer-framework-everywhere-src-${QT_IFW_VERSION}.tar.xz"

if [[ -d "${QT_IFW_SRC_DIR}" ]]; then
    rm -rf "${QT_IFW_SRC_DIR}"
fi
mkdir -p "${QT_IFW_SRC_DIR}"

echo "Downloading: ${IFW_URL}"
curl -L --retry 3 "${IFW_URL}" | tar xJ -C "${QT_IFW_SRC_DIR}" --strip-components=1

echo "Source downloaded and extracted to ${QT_IFW_SRC_DIR}"

# ── 4. Configure ──────────────────────────────────────────────────
echo ""
echo "=== Configuring Qt IFW ==="

rm -rf "${QT_IFW_BUILD_DIR}"
mkdir -p "${QT_IFW_BUILD_DIR}"
cd "${QT_IFW_BUILD_DIR}"

# Pass library paths so IFW's bundled libarchive can find liblzma.
# On macOS, bzip2/zlib/libiconv are in the system SDK.
QMAKE_ARGS="-r"
QMAKE_ARGS="${QMAKE_ARGS} IFW_LZMA_LIBRARY=${LZMA_PREFIX}/lib/liblzma.dylib"
QMAKE_ARGS="${QMAKE_ARGS} IFW_LZMA_INCLUDE=${LZMA_PREFIX}/include"

echo "Running: ${QMAKE} ${QMAKE_ARGS} ${QT_IFW_SRC_DIR}"
"${QMAKE}" ${QMAKE_ARGS} "${QT_IFW_SRC_DIR}"

# ── 5. Build ──────────────────────────────────────────────────────
echo ""
echo "=== Building Qt IFW ==="

NPROC=$(sysctl -n hw.logicalcpu 2>/dev/null || echo 4)
echo "Parallel jobs: ${NPROC}"
make -j"${NPROC}"

# ── 6. Install ────────────────────────────────────────────────────
echo ""
echo "=== Installing Qt IFW to ${QT_IFW_INSTALL_DIR} ==="

# qmake-based IFW may install to INSTALL_ROOT/usr/local/ or INSTALL_ROOT/.
# We try both INSTALL_ROOT and PREFIX approaches for maximum compatibility.
make install INSTALL_ROOT="${QT_IFW_INSTALL_DIR}"

# ── 7. Verify & normalise layout ──────────────────────────────────
echo ""
echo "=== Verification ==="

# Locate binarycreator wherever it ended up
BINARYCREATOR=""
for candidate in \
    "${QT_IFW_INSTALL_DIR}/bin/binarycreator" \
    "${QT_IFW_INSTALL_DIR}/usr/local/bin/binarycreator" \
    "${QT_IFW_INSTALL_DIR}/lib/QtInstallerFramework/bin/binarycreator"; do
    if [[ -x "${candidate}" ]]; then
        BINARYCREATOR="${candidate}"
        break
    fi
done

# Last resort: search recursively
if [[ -z "${BINARYCREATOR}" ]]; then
    BINARYCREATOR="$(find "${QT_IFW_INSTALL_DIR}" -name binarycreator -type f 2>/dev/null | head -1)"
fi

if [[ -z "${BINARYCREATOR}" || ! -x "${BINARYCREATOR}" ]]; then
    echo "ERROR: binarycreator not found after install!"
    echo "Contents of ${QT_IFW_INSTALL_DIR}:"
    find "${QT_IFW_INSTALL_DIR}" -type f 2>/dev/null | head -30
    exit 1
fi

# Normalise: ensure bin/ exists with all IFW tools for easy PATH inclusion
IFW_BIN_DIR="$(dirname "${BINARYCREATOR}")"
if [[ "${IFW_BIN_DIR}" != "${QT_IFW_INSTALL_DIR}/bin" ]]; then
    echo "Relocating tools from ${IFW_BIN_DIR} to ${QT_IFW_INSTALL_DIR}/bin"
    mkdir -p "${QT_IFW_INSTALL_DIR}/bin"
    cp -a "${IFW_BIN_DIR}"/* "${QT_IFW_INSTALL_DIR}/bin/" 2>/dev/null || true
fi

echo "binarycreator: ${QT_IFW_INSTALL_DIR}/bin/binarycreator"
file "${QT_IFW_INSTALL_DIR}/bin/binarycreator"
"${QT_IFW_INSTALL_DIR}/bin/binarycreator" --version 2>&1 || true

# ── 8. Bundle Qt frameworks into installerbase .app ────────────────
# The official x86_64 QtIFW has Qt statically linked into installerbase.
# Our ARM64 build links Qt dynamically (from conda). To make the installer
# .app self-contained, we use macdeployqt to embed frameworks, matching
# the official structure so binarycreator produces a working .app.
echo ""
echo "=== Bundling Qt frameworks into installerbase .app ==="
IFW_BIN_DIR="${QT_IFW_INSTALL_DIR}/bin"
INSTALLERBASE="${IFW_BIN_DIR}/installerbase"
if [[ -f "${INSTALLERBASE}" && ! -d "${INSTALLERBASE}" ]]; then
    # installerbase is a plain binary — wrap it in a .app and run macdeployqt
    WORK_DIR=$(mktemp -d)
    APP_BUNDLE="${WORK_DIR}/installerbase.app"
    mkdir -p "${APP_BUNDLE}/Contents/MacOS"
    cp "${INSTALLERBASE}" "${APP_BUNDLE}/Contents/MacOS/"
    cat > "${APP_BUNDLE}/Contents/Info.plist" << 'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key><string>installerbase</string>
    <key>CFBundleIdentifier</key><string>org.qt-project.installerbase</string>
    <key>CFBundleName</key><string>installerbase</string>
    <key>CFBundlePackageType</key><string>APPL</string>
    <key>CFBundleShortVersionString</key><string>4.6.1</string>
    <key>CFBundleVersion</key><string>4.6.1</string>
    <key>CFBundleInfoDictionaryVersion</key><string>6.0</string>
</dict>
</plist>
PLIST
    # Bundle Qt frameworks
    macdeployqt "${APP_BUNDLE}"
    # Fix rpath: @loader_path/../lib -> @executable_path/../Frameworks
    install_name_tool -delete_rpath "@loader_path/../lib" \
        -add_rpath "@executable_path/../Frameworks" \
        "${APP_BUNDLE}/Contents/MacOS/installerbase" 2>/dev/null || true
    # Ad-hoc sign
    codesign --deep --force -s - "${APP_BUNDLE}"
    # Replace plain binary with .app bundle
    rm -rf "${INSTALLERBASE}"
    cp -R "${APP_BUNDLE}" "${INSTALLERBASE}"
    rm -rf "${WORK_DIR}"
    echo "installerbase is now a .app bundle with embedded Frameworks"
    ls -la "${INSTALLERBASE}/Contents/"
else
    echo "installerbase is already a .app bundle or not found — skipping"
fi

echo ""
echo "=============================================="
echo " Qt IFW ${QT_IFW_VERSION} build complete!"
echo " Tools installed to: ${QT_IFW_INSTALL_DIR}/bin"
echo "=============================================="
echo ""
echo "Add to PATH in CI:"
echo "  export PATH=\"${QT_IFW_INSTALL_DIR}/bin:\$PATH\""
