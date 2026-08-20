#!/bin/bash
# build_static_qt_ifw.sh — Build STATIC Qt 5.15 + IFW 4.6.1 for macOS ARM64
#
# Background
# ==========
# The official pre-built QtIFW binaries (x86_64) link Qt statically: ~38 MB
# installerbase, zero @rpath dependencies, runs on any macOS without extra
# libraries. The previous ARM64 IFW was built against conda's *shared* Qt
# (5.15.8). That produced a broken toolchain:
#
#   - binarycreator / installerbase depend on @rpath/libQt5*.5.dylib
#   - maintenancetool (copied from installerbase during install) has the
#     same @rpath deps, but no Qt libraries are bundled next to it
#   - dyld fails: "Library not loaded: @rpath/libQt5Widgets.5.dylib",
#     SIGABRT whenever the installer auto-uninstalls an existing
#     installation (installscript.qs -> maintenancetool --script=...)
#
# This script builds a fully static toolchain (Qt 5.15 + IFW 4.6.1) so the
# generated installer .app and its maintenancetool are self-contained,
# matching the official x86_64 behavior.
#
# Only qtbase + qtdeclarative are built (IFW needs core/gui/widgets/network/
# xml/concurrent + qml for QJSEngine). All Qt 3rdparty libs (zlib, png, jpeg,
# pcre2, harfbuzz, freetype) are built in (no Homebrew / conda dependencies),
# so the final binaries must show NO @rpath and NO /opt/homebrew entries.
#
# Prerequisites:
#   - Xcode Command Line Tools (xcode-select --install)
#   - brew install ninja cmake  (ninja/cmake are only used by Qt 5.15 configure)
#
# Usage:
#   bash scripts/platforms/mac/build_static_qt_ifw.sh
#
# Output:
#   ~/opt/Qt/QtIFW-4.6.1-darwin-ARM64-static/bin/{binarycreator,installerbase,...}
#   ~/opt/Qt/QtIFW-4.6.1-darwin-ARM64-static.zip
#
# Upload to CI downloads:
#   gh release upload qt-ifw --repo Asher-1/cloudViewer_downloads \
#       ~/opt/Qt/QtIFW-4.6.1-darwin-ARM64-static.zip --clobber
#
# Build time: ~30-60 min first run (incremental resume supported; sources and
# the Qt build tree are kept under /tmp so a re-run only rebuilds what changed).
#
# Environment variables:
#   QT_STATIC_VERSION  - Qt version (default: 5.15.14)
#   IFW_STATIC_VERSION - IFW version (default: 4.6.1)
#   QT_STATIC_PREFIX   - Qt install prefix (default: ~/opt/Qt/qt-<ver>-static)
#   IFW_STATIC_PREFIX  - IFW staging prefix (default: ~/opt/Qt/QtIFW-<ver>-darwin-ARM64-static)

set -euo pipefail

QT_STATIC_VERSION="${QT_STATIC_VERSION:-5.15.14}"
IFW_STATIC_VERSION="${IFW_STATIC_VERSION:-4.6.1}"
QT_STATIC_PREFIX="${QT_STATIC_PREFIX:-${HOME}/opt/Qt/qt-${QT_STATIC_VERSION}-static}"
IFW_STATIC_PREFIX="${IFW_STATIC_PREFIX:-${HOME}/opt/Qt/QtIFW-${IFW_STATIC_VERSION}-darwin-ARM64-static}"

BUILD_ROOT="/tmp/qt-static-build"
QT_SRC_DIR="/tmp/qt-src-${QT_STATIC_VERSION}"
IFW_SRC_DIR="/tmp/ifw-src-${IFW_STATIC_VERSION}"
DIST_NAME="$(basename "${IFW_STATIC_PREFIX}")"   # QtIFW-<ver>-darwin-ARM64-static
DIST_ZIP="${HOME}/opt/Qt/${DIST_NAME}.zip"

NPROC=$(sysctl -n hw.logicalcpu 2>/dev/null || echo 4)

# Modules actually required by IFW 4.6.1: qtbase (core/gui/widgets/network/
# xml/concurrent), qtdeclarative (qml = QJSEngine), qttools (lrelease generates
# installer_*.qm translations). Everything else in the qt-everywhere tree is
# configured but never built.
QT_MODULES="qtbase qtdeclarative qttools"

# Exact configure options. KEEP THIS IN SYNC with the resume check below.
CONFIGURE_OPTS=(
    -prefix "${QT_STATIC_PREFIX}"
    -static
    -release
    -opensource -confirm-license
    -nomake examples
    -nomake tests
    -skip qtwebengine
    -skip qtlocation
    -skip qtmultimedia
    -skip qtconnectivity
    -skip qtsensors
    -skip qtserialport
    -skip qtwayland
    -skip qt3d
    -skip qtdoc
    -skip qtquick3d
    -skip qtcharts
    -no-opengl
    -no-dbus
    -no-icu
    -no-pch
    -qt-libjpeg
    -qt-libpng
    -qt-pcre
    -qt-zlib
    -qt-freetype
    -qt-harfbuzz
    -sql-sqlite
    -platform macx-clang
)

# ── 0. Sanity ────────────────────────────────────────────────────────────
for cmd in xcrun cmake; do
    if ! command -v "$cmd" &>/dev/null; then
        echo "ERROR: $cmd not found. Install with: brew install $cmd"
        exit 1
    fi
done
if [[ "$(uname -m)" != "arm64" ]]; then
    echo "NOTE: host is $(uname -m); this script targets Apple Silicon (arm64)."
    echo "      Building on x86_64 would produce an x86_64 IFW."
fi

echo "================================================================"
echo " Static Qt ${QT_STATIC_VERSION} + IFW ${IFW_STATIC_VERSION}"
echo " host: $(uname -m)  jobs: ${NPROC}"
echo " Qt prefix:  ${QT_STATIC_PREFIX}"
echo " IFW prefix: ${IFW_STATIC_PREFIX}"
echo "================================================================"

mkdir -p "$(dirname "${QT_STATIC_PREFIX}")" "$(dirname "${IFW_STATIC_PREFIX}")"

# ── 1. Download Qt source ───────────────────────────────────────────────
if [[ ! -f "${QT_SRC_DIR}/qtbase/.qmake.conf" ]]; then
    echo ""
    echo "=== Downloading Qt ${QT_STATIC_VERSION} source ==="
    QT_URL="https://download.qt.io/archive/qt/5.15/${QT_STATIC_VERSION}/single/qt-everywhere-opensource-src-${QT_STATIC_VERSION}.tar.xz"
    mkdir -p "${QT_SRC_DIR}"
    curl -L --retry 3 "${QT_URL}" | tar xJ -C "${QT_SRC_DIR}" --strip-components=1
else
    echo "Qt source already present at ${QT_SRC_DIR}"
fi

# ── 2. Configure Qt (static, self-contained) ────────────────────────────
# Resume check: the previous build tree must carry the exact desired flags.
# In particular "-qt-libpng" (Qt-bundled png) — a previous attempt used
# "-system-libpng" (Homebrew), which would make the final IFW binaries depend
# on /opt/homebrew/opt/libpng/lib/libpng.dylib and break on other machines.
NEEDS_CONFIG=0
if [[ ! -x "${BUILD_ROOT}/qtbase/bin/qmake" ]]; then
    NEEDS_CONFIG=1
elif ! grep -q -- "-qt-libpng" "${BUILD_ROOT}/config.opt" 2>/dev/null; then
    echo "Previous Qt build used different options (-system-libpng etc.); reconfiguring..."
    NEEDS_CONFIG=1
elif grep -q -- "-system-libpng" "${BUILD_ROOT}/config.opt" 2>/dev/null; then
    NEEDS_CONFIG=1
fi

if [[ "${NEEDS_CONFIG}" -eq 1 ]]; then
    echo ""
    echo "=== Configuring Qt ${QT_STATIC_VERSION} (static, shadow build in ${BUILD_ROOT}) ==="
    rm -rf "${BUILD_ROOT}"
    mkdir -p "${BUILD_ROOT}"
    cd "${BUILD_ROOT}"
    "${QT_SRC_DIR}/configure" "${CONFIGURE_OPTS[@]}"
else
    echo "Qt build tree already configured correctly at ${BUILD_ROOT}"
fi

# ── 3. Patch bundled libpng for modern macOS SDKs ───────────────────────
# libpng 1.6.39's pngpriv.h guards Classic-Mac headers with
# "defined(TARGET_OS_MAC)" — but TARGET_OS_MAC is defined (=1) by every modern
# macOS SDK (arm64 included), so it tries to #include <fp.h> (a Classic Mac
# header that no longer exists) and compilation fails with
# "fatal error: 'fp.h' file not found". Neutralize that legacy branch; the
# modern path (#include <math.h>) is taken instead. Idempotent.
PNGPRIV_H="${QT_SRC_DIR}/qtbase/src/3rdparty/libpng/pngpriv.h"
if [[ -f "${PNGPRIV_H}" ]] && grep -q 'defined(TARGET_OS_MAC)' "${PNGPRIV_H}" && ! grep -q '0 && defined(TARGET_OS_MAC)' "${PNGPRIV_H}"; then
    sed -i '' 's/defined(TARGET_OS_MAC)/0 \&\& defined(TARGET_OS_MAC)/g' "${PNGPRIV_H}"
    echo "Patched ${PNGPRIV_H}: disabled TARGET_OS_MAC legacy branch (fp.h)"
fi

# ── 4. Build & install required Qt modules ──────────────────────────────
# The top-level Makefile is a qmake subdirs template: each module is built
# through its "module-<name>" target, which generates the module Makefile
# (in ${BUILD_ROOT}/<name>) and runs make there. Build serially so qtbase is
# installed before qtdeclarative/qttools are configured against it.
for module in ${QT_MODULES}; do
    echo ""
    echo "=== Building ${module} (${NPROC} jobs) ==="
    make -C "${BUILD_ROOT}" "module-${module}" -j"${NPROC}"
    echo "=== Installing ${module} to ${QT_STATIC_PREFIX} ==="
    make -C "${BUILD_ROOT}/${module}" install
    echo "${module} done"
done
echo ""
echo "Qt static build complete at ${QT_STATIC_PREFIX}"

# ── 4. Download IFW source ──────────────────────────────────────────────
if [[ ! -f "${IFW_SRC_DIR}/installerfw.pro" ]]; then
    echo ""
    echo "=== Downloading Qt IFW ${IFW_STATIC_VERSION} source ==="
    IFW_URL="https://download.qt.io/official_releases/qt-installer-framework/${IFW_STATIC_VERSION}/installer-framework-everywhere-src-${IFW_STATIC_VERSION}.tar.xz"
    mkdir -p "${IFW_SRC_DIR}"
    curl -L --retry 3 "${IFW_URL}" | tar xJ -C "${IFW_SRC_DIR}" --strip-components=1
else
    echo "IFW source already present at ${IFW_SRC_DIR}"
fi

# ── 5. Patch IFW sources ────────────────────────────────────────────────
echo ""
echo "=== Patching IFW sources ==="
# 5.1 "requires(!cross_compile)" fails spuriously when building with a
#     non-standard prefix layout; disable it (see QtIFW README).
sed -i '' 's/requires(!cross_compile)/# requires(!cross_compile)/' \
    "${IFW_SRC_DIR}/installerfw.pro"

# ── 6. Configure & build IFW with the static Qt ─────────────────────────
echo ""
echo "=== Building Qt IFW with static Qt ==="
export PATH="${QT_STATIC_PREFIX}/bin:${PATH}"
echo "Using qmake: $(command -v qmake)"
echo "Qt version:  $(qmake -query QT_VERSION)"

rm -rf "${IFW_SRC_DIR}/bin"    # in-source build: drop stale artifacts
cd "${IFW_SRC_DIR}"
qmake -r

# 6.2 macOS 14+ SDK no longer ships AGL.framework; strip it from all
#     generated Makefiles before linking (same step as the previous recipe).
find . -name Makefile -exec sed -i '' 's/-framework AGL //g' {} +

make -j"${NPROC}"

echo ""
echo "IFW build complete"

# ── 7. Assemble distribution directory ─────────────────────────────────
echo ""
echo "=== Assembling ${DIST_NAME} ==="
rm -rf "${IFW_STATIC_PREFIX}"
mkdir -p "${IFW_STATIC_PREFIX}/bin"

# Tools are produced in <src>/bin for an in-source qmake build.
cp -R "${IFW_SRC_DIR}/bin/." "${IFW_STATIC_PREFIX}/bin/"

# IFW's own translations (installer_*.qm) — keep whatever was generated.
# Plus Qt's bundled translations (qt_*.qm) so installer UI text renders
# in non-English locales (e.g. zh_CN) on any machine.
mkdir -p "${IFW_STATIC_PREFIX}/bin/translations"
if [[ -d "${IFW_SRC_DIR}/bin/translations" ]]; then
    cp -R "${IFW_SRC_DIR}/bin/translations/." "${IFW_STATIC_PREFIX}/bin/translations/" 2>/dev/null || true
fi
cp "${QT_STATIC_PREFIX}/translations/qt_"*.qm "${IFW_STATIC_PREFIX}/bin/translations/" 2>/dev/null || true

echo "Distribution contents:"
ls -la "${IFW_STATIC_PREFIX}/bin/"
ls "${IFW_STATIC_PREFIX}/bin/translations/" | head

# ── 8. Verify self-containment ──────────────────────────────────────────
echo ""
echo "================================================================"
echo " Verification (must show NO @rpath / NO /opt / NO /Users deps)"
echo "================================================================"
FAILED=0
for tool in binarycreator installerbase archivegen repogen devtool; do
    BIN="${IFW_STATIC_PREFIX}/bin/${tool}"
    [[ -e "${BIN}" ]] || { echo "MISSING: ${tool}"; FAILED=1; continue; }
    echo ""
    echo "--- ${tool} ---"
    file "${BIN}"
    ls -lh "${BIN}" | awk '{print "size:", $5}'
    otool -L "${BIN}" || true
    if otool -L "${BIN}" | tail -n +2 | grep -E "@rpath|/opt/homebrew|/Users/" >/dev/null 2>&1; then
        echo "FAIL: ${tool} still has non-system dependencies!"
        FAILED=1
    fi
done

# Static installerbase is a plain binary; the generated installer .app and
# its maintenancetool are copies of it, hence also self-contained.
INSTALLERBASE="${IFW_STATIC_PREFIX}/bin/installerbase"
if [[ -f "${INSTALLERBASE}" ]]; then
    SIZE=$(stat -f%z "${INSTALLERBASE}")
    echo ""
    if [[ "${SIZE}" -lt 20000000 ]]; then
        echo "WARNING: installerbase is only ${SIZE} bytes — expected >20 MB for a static Qt build."
    else
        echo "installerbase is ${SIZE} bytes (static Qt, self-contained)."
    fi
fi

if [[ "${FAILED}" -eq 1 ]]; then
    echo ""
    echo "ERROR: self-containment check failed — inspect the output above."
    exit 1
fi

# ── 9. Package zip ──────────────────────────────────────────────────────
echo ""
echo "=== Packaging ${DIST_NAME}.zip ==="
cd "$(dirname "${IFW_STATIC_PREFIX}")"
rm -f "${DIST_ZIP}"
zip -r -y "${DIST_ZIP}" "${DIST_NAME}"
echo ""
echo "================================================================"
echo " SUCCESS — static IFW toolchain ready:"
echo "   ${IFW_STATIC_PREFIX}"
echo "   ${DIST_ZIP}"
echo ""
echo " Upload to CI downloads (replaces the dynamic ARM64 zip):"
echo "   gh release upload qt-ifw --repo Asher-1/cloudViewer_downloads \\"
echo "       ${DIST_ZIP} --clobber"
echo ""
echo " Then the GitHub macOS workflow picks it up automatically:"
echo "   .github/workflows/macos.yml (QtIFW-4.6.1-darwin-ARM64-static.zip)"
echo "================================================================"
