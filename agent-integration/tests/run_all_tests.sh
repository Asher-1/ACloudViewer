#!/usr/bin/env bash
# =============================================================================
# ACloudViewer Agent Integration — Unified Test Runner (v3)
# =============================================================================
#
# Usage:
#   ./run_all_tests.sh              # run all tests
#   ./run_all_tests.sh --level 1    # only Level 1 (no dependencies)
#   ./run_all_tests.sh --level 2    # Level 1 + CLI harness tests
#   ./run_all_tests.sh --level 3    # Level 1-2 + headless binary tests
#   ./run_all_tests.sh --level 4    # Level 1-3 + GUI RPC tests
#   ./run_all_tests.sh --level 5    # Level 1-4 + MCP server tests
#
# Architecture (v3):
#   This shell script is a thin wrapper around pytest test_integration.py.
#   It handles:
#     1. Environment detection (OS, binary, build dir, CLI)
#     2. Plugin build verification (cmake --build)
#     3. CLI harness auto-install (pip install -e)
#     4. Delegates all test logic to pytest for cross-platform consistency
#
# Exit code: 0 if all requested levels pass, 1 otherwise
# =============================================================================

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${ACV_REPO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"

MAX_LEVEL=5
VERBOSE=false
PYTEST_EXTRA_ARGS=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --level) MAX_LEVEL="$2"; shift 2 ;;
        --verbose|-v) VERBOSE=true; PYTEST_EXTRA_ARGS="-v"; shift ;;
        --help|-h)
            head -20 "$0" | grep "^#" | sed 's/^# \?//'
            exit 0 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

PYTHON="${PYTHON:-python}"
if ! command -v "$PYTHON" &>/dev/null; then
    PYTHON=python
fi

# OS detection
OS_TYPE="linux"
case "$(uname -s)" in
    Darwin*)  OS_TYPE="macos" ;;
    MINGW*|MSYS*|CYGWIN*|Windows_NT) OS_TYPE="windows" ;;
esac

header() { echo ""; echo "═══════════════════════════════════════════════════════"; echo "  $1"; echo "═══════════════════════════════════════════════════════"; }

# ─── Environment Info ───────────────────────────────────────────────────────
header "Environment"
echo "  OS:     $OS_TYPE ($(uname -s))"
echo "  Python: $($PYTHON --version 2>&1) ($PYTHON → $(command -v $PYTHON 2>/dev/null || echo 'NOT FOUND'))"
echo "  Repo:   $REPO_ROOT"
echo "  Build:  ${ACV_BUILD_DIR:-$REPO_ROOT/build_app}"
CLI_BIN=$(command -v cli-anything-acloudviewer 2>/dev/null || echo "NOT FOUND")
echo "  CLI:    $CLI_BIN"

# ─── Pre-flight: Build Plugin (Level 1 unique value) ────────────────────────
BUILD_DIR="${ACV_BUILD_DIR:-$REPO_ROOT/build_app}"
PLUGIN_BUILT=false

if [[ -f "$BUILD_DIR/CMakeCache.txt" ]]; then
    header "Pre-flight: Build QJSON_RPC_PLUGIN"
    if [[ "$OS_TYPE" == "windows" ]]; then
        CMAKE_PARALLEL_FLAG="-- /m"
    else
        CMAKE_PARALLEL_FLAG="-- -j4"
    fi
    if cmake --build "$BUILD_DIR" --target QJSON_RPC_PLUGIN $CMAKE_PARALLEL_FLAG >/dev/null 2>&1; then
        echo "  ✅ QJSON_RPC_PLUGIN compiled successfully"
        PLUGIN_BUILT=true

        PLUGIN_LIB=""
        for pattern in "libQJSON_RPC_PLUGIN.so" "libQJSON_RPC_PLUGIN.dylib" "QJSON_RPC_PLUGIN.dll" "QJSON_RPC_PLUGIN.dylib"; do
            PLUGIN_LIB=$(find "$BUILD_DIR" -name "$pattern" 2>/dev/null | head -1)
            [[ -n "$PLUGIN_LIB" ]] && break
        done
        if [[ -n "$PLUGIN_LIB" ]]; then
            echo "  ✅ Plugin binary: $(basename "$PLUGIN_LIB")"
        else
            echo "  ⚠️  Plugin binary not found after build"
        fi
    else
        echo "  ❌ QJSON_RPC_PLUGIN build failed"
    fi
else
    echo "  ⏭️  No CMake build directory at $BUILD_DIR (skipping build)"
fi

# ─── Pre-flight: Ensure CLI Harness Installed ────────────────────────────────
CLI_INSTALLED=false
if command -v cli-anything-acloudviewer &>/dev/null; then
    CLI_INSTALLED=true
else
    for candidate in \
        "${CLI_ANYTHING_HARNESS_ROOT:-}" \
        "$REPO_ROOT/../CLI-Anything/acloudviewer/agent-harness" \
        "$REPO_ROOT/_cli-anything/acloudviewer/agent-harness" \
        ""; do
        if [[ -n "$candidate" && -f "$candidate/setup.py" ]]; then
            echo ""
            echo "  Installing CLI harness from $candidate ..."
            $PYTHON -m pip install -e "$candidate[dev]" -q 2>/dev/null && CLI_INSTALLED=true || true
            break
        fi
    done
fi

if [[ "$CLI_INSTALLED" == "true" ]]; then
    echo "  ✅ CLI harness installed"
else
    echo "  ⏭️  CLI harness not found"
fi

# ─── Pre-flight: Auto-detect ACloudViewer Binary ────────────────────────────
if [[ -z "${ACV_BINARY:-}" ]]; then
    if [[ "$OS_TYPE" == "windows" ]]; then
        BUILD_CANDIDATES=(
            "$BUILD_DIR/bin/Release/ACloudViewer.exe"
            "$BUILD_DIR/bin/ACloudViewer.exe"
            "$BUILD_DIR/bin/ACloudViewer.bat"
        )
    elif [[ "$OS_TYPE" == "macos" ]]; then
        BUILD_CANDIDATES=(
            "$BUILD_DIR/bin/ACloudViewer.app/Contents/MacOS/ACloudViewer"
            "$BUILD_DIR/bin/ACloudViewer"
        )
    else
        BUILD_CANDIDATES=(
            "$BUILD_DIR/bin/ACloudViewer"
            "$BUILD_DIR/bin/ACloudViewer.sh"
        )
    fi
    for candidate in "${BUILD_CANDIDATES[@]}"; do
        if [[ -e "$candidate" ]]; then
            ACV_BINARY="$candidate"
            break
        fi
    done
fi

if [[ -z "${ACV_BINARY:-}" ]]; then
    ACV_BINARY=$($PYTHON -c "
from cli_anything.acloudviewer.utils.acloudviewer_backend import ACloudViewerBackend
b = ACloudViewerBackend.find_binary()
print(b or '')
" 2>/dev/null || true)
fi

if [[ -z "${ACV_BINARY:-}" ]]; then
    if [[ "$OS_TYPE" == "windows" ]]; then
        ACV_BINARY=$(command -v ACloudViewer.bat 2>/dev/null || command -v ACloudViewer.exe 2>/dev/null || true)
    elif [[ "$OS_TYPE" == "macos" ]]; then
        ACV_BINARY=$(command -v ACloudViewer 2>/dev/null || true)
    else
        ACV_BINARY=$(command -v ACloudViewer.sh 2>/dev/null || command -v ACloudViewer 2>/dev/null || true)
    fi
fi

if [[ -n "${ACV_BINARY:-}" ]]; then
    export ACV_BINARY
    echo "  ✅ ACloudViewer binary: $ACV_BINARY"
else
    echo "  ⏭️  ACloudViewer binary not found (Level 3+ tests will skip)"
fi

# ─── Set platform-specific env ──────────────────────────────────────────────
if [[ "$OS_TYPE" == "macos" ]]; then
    # macOS .app bundles only include cocoa Qt platform plugin.
    # Don't set QT_QPA_PLATFORM - let Qt auto-select cocoa.
    # (Remove if set externally to avoid "minimal not found" errors)
    unset QT_QPA_PLATFORM
elif [[ "$OS_TYPE" == "windows" ]]; then
    export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-minimal}"
else
    export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-offscreen}"
fi

# ─── Build pytest -k filter from level ──────────────────────────────────────
LEVEL_FILTER=""
if [[ $MAX_LEVEL -lt 5 ]]; then
    parts=()
    for i in $(seq 1 "$MAX_LEVEL"); do
        parts+=("level$i")
    done
    # Join with " or " for pytest -k syntax
    LEVEL_FILTER=$(printf " or %s" "${parts[@]}")
    LEVEL_FILTER="${LEVEL_FILTER:4}"  # Remove leading " or "
fi

# ─── Run pytest ─────────────────────────────────────────────────────────────
if [[ -n "$LEVEL_FILTER" ]]; then
    header "Running pytest (Levels 1-$MAX_LEVEL)"
    echo "  Filter: -k \"$LEVEL_FILTER\""
    echo ""
    $PYTHON -m pytest "$SCRIPT_DIR/test_integration.py" \
        -k "$LEVEL_FILTER" \
        $PYTEST_EXTRA_ARGS \
        --tb=short \
        2>&1
else
    header "Running pytest (All Levels)"
    echo ""
    $PYTHON -m pytest "$SCRIPT_DIR/test_integration.py" \
        $PYTEST_EXTRA_ARGS \
        --tb=short \
        2>&1
fi

PYTEST_EXIT=$?

# ─── Also run CLI-Anything harness tests if Level >= 2 ──────────────────────
CLI_HARNESS_EXIT=0
if [[ $MAX_LEVEL -ge 2 && "$CLI_INSTALLED" == "true" ]]; then
    CLI_HARNESS=""
    for candidate in \
        "${CLI_ANYTHING_HARNESS_ROOT:-}" \
        "$REPO_ROOT/../CLI-Anything/acloudviewer/agent-harness" \
        "$REPO_ROOT/_cli-anything/acloudviewer/agent-harness" \
        ""; do
        if [[ -n "$candidate" && -d "$candidate/cli_anything/acloudviewer/tests" ]]; then
            CLI_HARNESS="$candidate"
            break
        fi
    done

    if [[ -n "$CLI_HARNESS" ]]; then
        header "Running CLI-Anything Harness Tests"
        TEST_DIR="$CLI_HARNESS/cli_anything/acloudviewer/tests"
        if [[ -d "$TEST_DIR" ]]; then
            $PYTHON -m pytest "$TEST_DIR" $PYTEST_EXTRA_ARGS --tb=short 2>&1
            CLI_HARNESS_EXIT=$?
        fi
    fi
fi

# ─── Summary ────────────────────────────────────────────────────────────────
header "Summary"
echo "  Integration tests (Level 1-$MAX_LEVEL): $([ $PYTEST_EXIT -eq 0 ] && echo '✅ PASSED' || echo '❌ FAILED')"
if [[ $MAX_LEVEL -ge 2 && "$CLI_INSTALLED" == "true" ]]; then
    echo "  CLI harness tests:                     $([ $CLI_HARNESS_EXIT -eq 0 ] && echo '✅ PASSED' || echo '❌ FAILED')"
fi
if [[ "$PLUGIN_BUILT" == "true" ]]; then
    echo "  Plugin build:                          ✅ PASSED"
fi
echo ""

EXIT_CODE=0
[[ $PYTEST_EXIT -ne 0 ]] && EXIT_CODE=1
[[ $CLI_HARNESS_EXIT -ne 0 ]] && EXIT_CODE=1
exit $EXIT_CODE
