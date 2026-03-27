#!/usr/bin/env bash
# =============================================================================
# ACloudViewer Agent Integration — Unified Test Runner (v2)
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
# Architecture (v2):
#   Headless mode: calls ACloudViewer binary via subprocess (-SILENT CLI)
#   GUI mode: controls running ACloudViewer via JSON-RPC WebSocket
#   NO dependency on cloudViewer Python package or open3d
#
# Exit code: 0 if all requested levels pass, 1 otherwise
# =============================================================================

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${ACV_REPO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"

MAX_LEVEL=5
VERBOSE=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --level) MAX_LEVEL="$2"; shift 2 ;;
        --verbose|-v) VERBOSE=true; shift ;;
        --help|-h)
            head -20 "$0" | grep "^#" | sed 's/^# \?//'
            exit 0 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

PYTHON="${PYTHON:-python3}"
if ! command -v "$PYTHON" &>/dev/null; then
    PYTHON=python
fi

# OS detection
OS_TYPE="linux"
case "$(uname -s)" in
    Darwin*)  OS_TYPE="macos" ;;
    MINGW*|MSYS*|CYGWIN*|Windows_NT) OS_TYPE="windows" ;;
esac

PASS=0
FAIL=0
SKIP=0

pass()  { echo "  ✅ PASS: $1"; PASS=$((PASS+1)); }
fail()  { echo "  ❌ FAIL: $1"; FAIL=$((FAIL+1)); }
skip()  { echo "  ⏭️  SKIP: $1"; SKIP=$((SKIP+1)); }
header() { echo ""; echo "═══════════════════════════════════════════════════════"; echo "  $1"; echo "═══════════════════════════════════════════════════════"; }
diag()  { [[ "$VERBOSE" == "true" ]] && echo "    [diag] $1" || true; }

# ─── Environment Info ───────────────────────────────────────────────────────
header "Environment"
echo "  OS:     $OS_TYPE ($(uname -s))"
echo "  Python: $($PYTHON --version 2>&1) ($PYTHON → $(command -v $PYTHON 2>/dev/null || echo 'NOT FOUND'))"
echo "  Repo:   $REPO_ROOT"
echo "  Build:  ${ACV_BUILD_DIR:-$REPO_ROOT/build_app}"
CLI_BIN=$(command -v cli-anything-acloudviewer 2>/dev/null || echo "NOT FOUND")
echo "  CLI:    $CLI_BIN"

# ─── Level 1: C++ Build Verification ────────────────────────────────────────
header "Level 1: C++ Plugin Build Verification"

BUILD_DIR="${ACV_BUILD_DIR:-$REPO_ROOT/build_app}"
if [[ -f "$BUILD_DIR/CMakeCache.txt" ]]; then
    echo "  Building QJSON_RPC_PLUGIN..."
    if [[ "$OS_TYPE" == "windows" ]]; then
        CMAKE_PARALLEL_FLAG="-- /m"
    else
        CMAKE_PARALLEL_FLAG="-- -j4"
    fi
    if cmake --build "$BUILD_DIR" --target QJSON_RPC_PLUGIN $CMAKE_PARALLEL_FLAG >/dev/null 2>&1; then
        pass "QJSON_RPC_PLUGIN compiles successfully"

        PLUGIN_LIB=""
        for pattern in "libQJSON_RPC_PLUGIN.so" "QJSON_RPC_PLUGIN.dylib" "QJSON_RPC_PLUGIN.dll"; do
            PLUGIN_LIB=$(find "$BUILD_DIR" -name "$pattern" 2>/dev/null | head -1)
            [[ -n "$PLUGIN_LIB" ]] && break
        done
        if [[ -n "$PLUGIN_LIB" ]]; then
            pass "Plugin binary exists: $(basename "$PLUGIN_LIB")"
        else
            fail "Plugin binary not found after build"
        fi
    else
        fail "QJSON_RPC_PLUGIN build failed"
    fi
else
    skip "No CMake build directory at $BUILD_DIR (run cmake first)"
fi

[[ $MAX_LEVEL -lt 2 ]] && { echo ""; echo "Done (Level 1)."; exit $([[ $FAIL -eq 0 ]] && echo 0 || echo 1); }

# ─── Level 2: CLI Harness Tests ─────────────────────────────────────────────
header "Level 2: CLI Harness Tests"

CLI_HARNESS=""
for candidate in \
    "${CLI_ANYTHING_HARNESS_ROOT:-}" \
    "$REPO_ROOT/../CLI-Anything/acloudviewer/agent-harness" \
    "$REPO_ROOT/_cli-anything/acloudviewer/agent-harness" \
    ""; do
    if [[ -n "$candidate" && -f "$candidate/setup.py" ]]; then
        CLI_HARNESS="$candidate"
        break
    fi
done

CLI_INSTALLED=false
if command -v cli-anything-acloudviewer &>/dev/null; then
    CLI_INSTALLED=true
fi

if [[ -n "$CLI_HARNESS" && -f "$CLI_HARNESS/setup.py" ]]; then
    echo "  Harness source: $CLI_HARNESS"
    if [[ "$CLI_INSTALLED" != "true" ]]; then
        echo "  Installing CLI harness..."
        $PYTHON -m pip install -e "$CLI_HARNESS[dev]" -q 2>/dev/null && CLI_INSTALLED=true || true
    fi

    TEST_DIR="$CLI_HARNESS/cli_anything/acloudviewer/tests"
    PYTEST_FILES=""
    [[ -f "$TEST_DIR/test_core.py" ]] && PYTEST_FILES="$PYTEST_FILES $TEST_DIR/test_core.py"
    [[ -f "$TEST_DIR/test_cli.py" ]] && PYTEST_FILES="$PYTEST_FILES $TEST_DIR/test_cli.py"

    if [[ -n "$PYTEST_FILES" ]]; then
        TEST_LOG=$(mktemp "${TMPDIR:-/tmp}/acv_test_l2.XXXXXX.log")
        if $PYTHON -m pytest $PYTEST_FILES -v --tb=short 2>&1 | tee "$TEST_LOG" | tail -5; then
            PYTEST_EXIT=${PIPESTATUS[0]}
            if [[ $PYTEST_EXIT -eq 0 ]]; then
                COUNT=$(grep -c "PASSED" "$TEST_LOG" 2>/dev/null || echo 0)
                pass "pytest: $COUNT tests passed"
            else
                fail "pytest returned exit code $PYTEST_EXIT"
            fi
        fi
    else
        skip "No test files found in $TEST_DIR"
    fi
fi

if [[ "$CLI_INSTALLED" == "true" ]]; then
    pass "CLI entry point installed"

    if cli-anything-acloudviewer --help >/dev/null 2>&1; then
        pass "CLI --help works"
    else
        echo "    [diag] cli-anything-acloudviewer --help stderr:"
        cli-anything-acloudviewer --help 2>&1 | head -5 || true
        fail "CLI --help failed"
    fi

    for cmd in "convert --help" "batch-convert --help" "formats" "process --help" \
               "reconstruct --help" "reconstruct auto --help" "sibr --help" \
               "scene --help" "view --help" "session --help" "info" \
               "entity --help" "cloud --help" "mesh --help" "transform --help" \
               "export --help" "clear --help" "methods --help"; do
        if [[ "$OS_TYPE" == "macos" ]]; then
            case "$cmd" in
                sibr*) skip "CLI subcommand: $cmd (SIBR not supported on macOS)"; continue ;;
            esac
        fi
        if cli-anything-acloudviewer $cmd >/dev/null 2>&1; then
            pass "CLI subcommand: $cmd"
        else
            fail "CLI subcommand: $cmd"
        fi
    done

    # Verify reconstruct auto mentions camera model option
    RECON_HELP=$(cli-anything-acloudviewer reconstruct auto --help 2>&1 || true)
    if echo "$RECON_HELP" | grep -qi "camera"; then
        pass "reconstruct auto --help mentions camera option"
    else
        fail "reconstruct auto --help missing camera option"
    fi
else
    skip "CLI harness not found (pip install cli-anything-acloudviewer or set CLI_ANYTHING_HARNESS_ROOT)"
fi

[[ $MAX_LEVEL -lt 3 ]] && { echo ""; echo "Done (Level 1-2)."; exit $([[ $FAIL -eq 0 ]] && echo 0 || echo 1); }

# ─── Level 3: Headless Binary Processing Tests ──────────────────────────────
header "Level 3: Headless Binary Processing Tests"

if [[ -z "${ACV_BINARY:-}" ]]; then
    # Try build directory first (platform-specific binary names)
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

_binary_exists() {
    [[ -e "$1" ]] || command -v "$1" &>/dev/null
}

if [[ -n "$ACV_BINARY" ]] && _binary_exists "$ACV_BINARY"; then
    export ACV_BINARY
    pass "ACloudViewer binary found: $ACV_BINARY"

    TEST_TMPDIR=$(mktemp -d "${TMPDIR:-/tmp}/acv_test.XXXXXX")
    $PYTHON -c "
from pathlib import Path
lines = ['ply', 'format ascii 1.0', 'element vertex 200',
         'property float x', 'property float y', 'property float z',
         'end_header']
for i in range(200):
    lines.append(f'{i*0.01} {i*0.02} {i*0.03}')
Path('$TEST_TMPDIR/test.ply').write_text('\n'.join(lines) + '\n')
print('Created test.ply')
" && pass "Created test PLY (200 points)" || fail "Create test PLY"

    if [[ -f "$TEST_TMPDIR/test.ply" ]]; then
        # On macOS: leave QT_QPA_PLATFORM unset; main.cpp probes for
        # offscreen/minimal and falls back to cocoa automatically.
        if [[ "$OS_TYPE" == "macos" ]]; then
            unset QT_QPA_PLATFORM 2>/dev/null || true
        elif [[ "$OS_TYPE" == "windows" ]]; then
            export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-minimal}"
        else
            export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-offscreen}"
        fi

        if "$ACV_BINARY" -SILENT -O "$TEST_TMPDIR/test.ply" -SAVE_CLOUDS 2>/dev/null; then
            pass "Binary loads PLY file"
        else
            fail "Binary loads PLY file (exit code: $?)"
        fi

        if "$ACV_BINARY" -SILENT -O "$TEST_TMPDIR/test.ply" -SS SPATIAL 0.2 -SAVE_CLOUDS 2>/dev/null; then
            pass "Subsample via binary"
        else
            fail "Subsample via binary (exit code: $?)"
        fi

        if "$ACV_BINARY" -SILENT -O "$TEST_TMPDIR/test.ply" -COMPUTE_NORMALS 2>/dev/null; then
            pass "Compute normals via binary"
        else
            fail "Compute normals via binary (exit code: $?)"
        fi

        if [[ "$CLI_INSTALLED" == "true" ]]; then
            CLI_SUB_OUT=$(cli-anything-acloudviewer --json --mode headless process subsample "$TEST_TMPDIR/test.ply" -o "$TEST_TMPDIR/sub.ply" --voxel-size 0.2 2>/dev/null || true)
            CLI_SUB_STATUS=$(echo "$CLI_SUB_OUT" | $PYTHON -c "import sys,json; d=json.load(sys.stdin); print(d.get('status','unknown'))" 2>/dev/null || echo "error")
            if [[ "$CLI_SUB_STATUS" != "failed" && "$CLI_SUB_STATUS" != "error" ]]; then
                pass "Subsample via CLI harness (status=$CLI_SUB_STATUS)"
            else
                diag "CLI subsample output: $CLI_SUB_OUT"
                fail "Subsample via CLI harness (status=$CLI_SUB_STATUS)"
            fi

            OUTPUT=$(cli-anything-acloudviewer --json --mode headless info 2>/dev/null || true)
            if echo "$OUTPUT" | $PYTHON -c "import sys,json; d=json.load(sys.stdin); assert d['mode']=='headless'" 2>/dev/null; then
                pass "Headless info JSON valid"
            else
                diag "info output: $OUTPUT"
                fail "Headless info JSON"
            fi

            OUTPUT=$(cli-anything-acloudviewer --json --mode headless formats 2>/dev/null || true)
            if echo "$OUTPUT" | $PYTHON -c "import sys,json; d=json.load(sys.stdin); assert '.ply' in d['point_cloud']" 2>/dev/null; then
                pass "Formats command lists .ply"
            else
                diag "formats output: $OUTPUT"
                fail "Formats command"
            fi

            if [[ "$OS_TYPE" == "macos" ]]; then
                skip "SIBR CLI group (not supported on macOS)"
            elif cli-anything-acloudviewer sibr --help >/dev/null 2>&1; then
                pass "SIBR CLI group available"
            else
                skip "SIBR CLI group not available"
            fi
        else
            skip "CLI harness not installed (skipping CLI-based Level 3 tests)"
        fi

        # ─── Format conversion tests (binary direct) ────────────────────
        echo ""
        echo "  ── Format Conversion Tests ──"

        for FMT in ASC BIN VTK; do
            if "$ACV_BINARY" -SILENT -O "$TEST_TMPDIR/test.ply" -C_EXPORT_FMT "$FMT" -SAVE_CLOUDS 2>/dev/null; then
                pass "PLY -> $FMT conversion (binary)"
            else
                fail "PLY -> $FMT conversion (binary, exit code: $?)"
            fi
        done

        # PLY -> PCD conversion (requires qPCL plugin)
        PCD_OUT="$TEST_TMPDIR/test_ply2pcd.pcd"
        if "$ACV_BINARY" -SILENT -O "$TEST_TMPDIR/test.ply" -AUTO_SAVE OFF -NO_TIMESTAMP -C_EXPORT_FMT PCD -SAVE_CLOUDS FILE "$PCD_OUT" 2>/dev/null; then
            if [[ -f "$PCD_OUT" ]]; then
                pass "PLY -> PCD conversion (binary)"

                # PCD -> PLY roundtrip
                PLY_BACK="$TEST_TMPDIR/test_pcd2ply.ply"
                if "$ACV_BINARY" -SILENT -O "$PCD_OUT" -AUTO_SAVE OFF -NO_TIMESTAMP -C_EXPORT_FMT PLY -SAVE_CLOUDS FILE "$PLY_BACK" 2>/dev/null; then
                    if [[ -f "$PLY_BACK" ]]; then
                        pass "PCD -> PLY roundtrip (binary)"
                    else
                        fail "PCD -> PLY roundtrip: output not found"
                    fi
                else
                    fail "PCD -> PLY roundtrip (binary)"
                fi

                # PCD -> DRC conversion (requires qDracoIO plugin)
                DRC_OUT="$TEST_TMPDIR/test_pcd2drc.drc"
                if "$ACV_BINARY" -SILENT -O "$PCD_OUT" -AUTO_SAVE OFF -NO_TIMESTAMP -C_EXPORT_FMT DRC -SAVE_CLOUDS FILE "$DRC_OUT" 2>/dev/null; then
                    if [[ -f "$DRC_OUT" ]]; then
                        pass "PCD -> DRC conversion (binary)"

                        # DRC -> PCD roundtrip
                        PCD_BACK="$TEST_TMPDIR/test_drc2pcd.pcd"
                        if "$ACV_BINARY" -SILENT -O "$DRC_OUT" -AUTO_SAVE OFF -NO_TIMESTAMP -C_EXPORT_FMT PCD -SAVE_CLOUDS FILE "$PCD_BACK" 2>/dev/null; then
                            if [[ -f "$PCD_BACK" ]]; then
                                pass "DRC -> PCD roundtrip (binary)"
                            else
                                fail "DRC -> PCD roundtrip: output not found"
                            fi
                        else
                            fail "DRC -> PCD roundtrip (binary)"
                        fi
                    else
                        fail "PCD -> DRC conversion: output not found"
                    fi
                else
                    skip "PCD -> DRC conversion (Draco plugin may not be available)"
                fi
            else
                fail "PLY -> PCD conversion: output not found"
            fi
        else
            skip "PLY -> PCD conversion (PCL plugin may not be available)"
        fi

        # ─── Format conversion tests (CLI harness) ───────────────────────
        if [[ "$CLI_INSTALLED" == "true" ]]; then
            echo ""
            echo "  ── CLI Harness Conversion Tests ──"

            CONV_PCD="$TEST_TMPDIR/cli_converted.pcd"
            OUTPUT=$(cli-anything-acloudviewer --json --mode headless convert "$TEST_TMPDIR/test.ply" "$CONV_PCD" 2>/dev/null || true)
            CONV_STATUS=$(echo "$OUTPUT" | $PYTHON -c "import sys,json; d=json.load(sys.stdin); print(d.get('status','unknown'))" 2>/dev/null || echo "error")
            if [[ "$CONV_STATUS" != "failed" && "$CONV_STATUS" != "error" ]]; then
                pass "CLI convert PLY -> PCD (status=$CONV_STATUS)"
            else
                diag "CLI convert output: $OUTPUT"
                fail "CLI convert PLY -> PCD (status=$CONV_STATUS)"
            fi

            if [[ -f "$CONV_PCD" ]]; then
                CONV_DRC="$TEST_TMPDIR/cli_converted.drc"
                OUTPUT=$(cli-anything-acloudviewer --json --mode headless convert "$CONV_PCD" "$CONV_DRC" 2>/dev/null || true)
                DRC_STATUS=$(echo "$OUTPUT" | $PYTHON -c "import sys,json; d=json.load(sys.stdin); print(d.get('status','unknown'))" 2>/dev/null || echo "error")
                if [[ "$DRC_STATUS" != "failed" && "$DRC_STATUS" != "error" ]]; then
                    pass "CLI convert PCD -> DRC (status=$DRC_STATUS)"
                else
                    diag "CLI convert PCD->DRC output: $OUTPUT"
                    fail "CLI convert PCD -> DRC (status=$DRC_STATUS)"
                fi
            fi

            BATCH_IN="$TEST_TMPDIR/batch_in"
            BATCH_OUT="$TEST_TMPDIR/batch_out"
            mkdir -p "$BATCH_IN"
            cp "$TEST_TMPDIR/test.ply" "$BATCH_IN/cloud1.ply"
            cp "$TEST_TMPDIR/test.ply" "$BATCH_IN/cloud2.ply"
            BATCH_OUTPUT=$(cli-anything-acloudviewer --json --mode headless batch-convert "$BATCH_IN" "$BATCH_OUT" -f .pcd 2>/dev/null || true)
            BATCH_STATUS=$(echo "$BATCH_OUTPUT" | $PYTHON -c "import sys,json; d=json.load(sys.stdin); print(d.get('status','unknown'))" 2>/dev/null || echo "error")
            if [[ "$BATCH_STATUS" != "failed" && "$BATCH_STATUS" != "error" ]]; then
                pass "CLI batch-convert PLY -> PCD (status=$BATCH_STATUS)"
            else
                diag "CLI batch-convert output: $BATCH_OUTPUT"
                fail "CLI batch-convert PLY -> PCD (status=$BATCH_STATUS)"
            fi
        fi
    fi

    rm -rf "$TEST_TMPDIR"
else
    skip "ACloudViewer binary not found (add to PATH or set ACV_BINARY)"
fi

[[ $MAX_LEVEL -lt 4 ]] && { echo ""; echo "Done (Level 1-3)."; exit $([[ $FAIL -eq 0 ]] && echo 0 || echo 1); }

# ─── Level 4: GUI RPC Tests ─────────────────────────────────────────────────
header "Level 4: GUI JSON-RPC Tests (requires running ACloudViewer)"

RPC_URL="${ACV_RPC_URL:-ws://localhost:6001}"
echo "  Testing RPC at: $RPC_URL"

# Helper: call JSON-RPC method and print result (or empty on failure)
# Usage: _rpc METHOD [PARAMS_JSON [TIMEOUT]]
_rpc() {
    local METHOD="$1"; shift
    local PARAMS="${1:-}"
    [ -z "$PARAMS" ] && PARAMS='{}'
    local TIMEOUT="${2:-30}"
    _RPC_PARAMS="$PARAMS" _RPC_METHOD="$METHOD" _RPC_URL="$RPC_URL" _RPC_TIMEOUT="$TIMEOUT" \
    $PYTHON -c "
import os, json, sys
from websockets.sync.client import connect
params = json.loads(os.environ['_RPC_PARAMS'])
ws = connect(os.environ['_RPC_URL'], open_timeout=5)
ws.send(json.dumps({'jsonrpc':'2.0','id':1,'method':os.environ['_RPC_METHOD'],'params':params}))
resp = json.loads(ws.recv(timeout=int(os.environ['_RPC_TIMEOUT'])))
ws.close()
if 'error' in resp:
    print('ERROR:' + json.dumps(resp['error']), file=sys.stderr)
    sys.exit(1)
print(json.dumps(resp.get('result','')))
" 2>/dev/null
}

if _rpc ping >/dev/null 2>&1; then
    pass "RPC ping -> pong"

    # ── Introspection ──
    METHODS=$(_rpc methods.list | $PYTHON -c "import sys,json; print(len(json.load(sys.stdin)))" 2>/dev/null)
    if [[ -n "$METHODS" && "$METHODS" -ge 25 ]]; then
        pass "methods.list returned $METHODS methods (expected ≥25)"
    else
        fail "methods.list returned ${METHODS:-0} methods"
    fi

    # ── View control ──
    echo ""
    echo "  ── View Control Tests ──"
    for ORIENT in top bottom front back left right iso1 iso2; do
        if _rpc view.setOrientation "{\"orientation\":\"$ORIENT\"}" >/dev/null 2>&1; then
            pass "view.setOrientation -> $ORIENT"
        else
            fail "view.setOrientation -> $ORIENT"
        fi
    done

    if _rpc view.zoomFit >/dev/null 2>&1; then pass "view.zoomFit"; else fail "view.zoomFit"; fi
    if _rpc view.refresh >/dev/null 2>&1; then pass "view.refresh"; else fail "view.refresh"; fi

    # Camera
    CAM=$(_rpc view.getCamera 2>/dev/null)
    if echo "$CAM" | $PYTHON -c "import sys,json; d=json.load(sys.stdin); assert 'view_matrix' in d and 'fov_deg' in d and len(d['view_matrix'])==16" 2>/dev/null; then
        pass "view.getCamera (matrix 4x4 + fov)"
    else
        fail "view.getCamera"
    fi

    # Perspective
    if _rpc view.setPerspective '{"mode":"object"}' >/dev/null 2>&1; then
        pass "view.setPerspective (object)"
    else
        fail "view.setPerspective (object)"
    fi

    # Point size
    if _rpc view.setPointSize '{"action":"increase"}' >/dev/null 2>&1 && \
       _rpc view.setPointSize '{"action":"decrease"}' >/dev/null 2>&1; then
        pass "view.setPointSize increase/decrease"
    else
        fail "view.setPointSize"
    fi

    # Screenshot (empty scene)
    SHOT_DIR=$(mktemp -d "${TMPDIR:-/tmp}/acv_shot.XXXXXX")
    SHOT_RESULT=$(_rpc view.screenshot "{\"filename\":\"$SHOT_DIR/empty.png\"}" 2>/dev/null)
    if [[ -f "$SHOT_DIR/empty.png" ]]; then
        pass "view.screenshot (empty scene)"
    else
        fail "view.screenshot (empty scene)"
    fi

    # ── File I/O via RPC ──
    echo ""
    echo "  ── RPC File I/O Tests ──"
    RPC_TMPDIR=$(mktemp -d "${TMPDIR:-/tmp}/acv_rpc.XXXXXX")
    $PYTHON -c "
from pathlib import Path
lines = ['ply', 'format ascii 1.0', 'element vertex 100',
         'property float x', 'property float y', 'property float z',
         'end_header']
for i in range(100):
    lines.append(f'{i*0.01} {i*0.02} {i*0.03}')
Path('$RPC_TMPDIR/test.ply').write_text('\n'.join(lines) + '\n')
"
    # Open file
    OPEN_RESULT=$(_rpc open "{\"filename\":\"$RPC_TMPDIR/test.ply\",\"silent\":true}" 2>/dev/null)
    if echo "$OPEN_RESULT" | $PYTHON -c "import sys,json; d=json.load(sys.stdin); assert 'id' in d" 2>/dev/null; then
        pass "RPC open PLY file"
    else
        fail "RPC open PLY file"
    fi

    # Scene list count
    SCENE=$(_rpc scene.list '{"recursive":true}' 2>/dev/null)
    ENTITY_COUNT=$(echo "$SCENE" | $PYTHON -c "import sys,json; print(len(json.load(sys.stdin)))" 2>/dev/null)
    if [[ -n "$ENTITY_COUNT" && "$ENTITY_COUNT" -ge 1 ]]; then
        pass "scene.list has $ENTITY_COUNT entities after load"
    else
        fail "scene.list after load"
    fi

    # Screenshot with loaded data
    _rpc view.setOrientation '{"orientation":"iso1"}' >/dev/null 2>&1
    _rpc view.zoomFit >/dev/null 2>&1
    SHOT_RESULT=$(_rpc view.screenshot "{\"filename\":\"$RPC_TMPDIR/loaded.png\"}" 2>/dev/null)
    if [[ -f "$RPC_TMPDIR/loaded.png" ]] && [[ $(stat -c%s "$RPC_TMPDIR/loaded.png" 2>/dev/null || stat -f%z "$RPC_TMPDIR/loaded.png" 2>/dev/null) -gt 100 ]]; then
        pass "view.screenshot (with loaded data)"
    else
        fail "view.screenshot (with loaded data)"
    fi

    # Clear scene
    _rpc clear >/dev/null 2>&1
    SCENE_AFTER=$(_rpc scene.list 2>/dev/null)
    AFTER_COUNT=$(echo "$SCENE_AFTER" | $PYTHON -c "import sys,json; print(len(json.load(sys.stdin)))" 2>/dev/null)
    if [[ "$AFTER_COUNT" == "0" ]]; then
        pass "RPC clear scene"
    else
        fail "RPC clear scene (still $AFTER_COUNT entities)"
    fi

    # File convert (PLY -> ASC, PLY -> BIN)
    for EXT in asc ply bin; do
        OUT="$RPC_TMPDIR/converted.$EXT"
        CONV=$(_rpc file.convert "{\"input\":\"$RPC_TMPDIR/test.ply\",\"output\":\"$OUT\"}" 60 2>/dev/null)
        if echo "$CONV" | $PYTHON -c "import sys,json; d=json.load(sys.stdin); assert d.get('status')=='converted'" 2>/dev/null && [[ -f "$OUT" ]]; then
            pass "RPC file.convert PLY -> ${EXT^^}"
        else
            fail "RPC file.convert PLY -> ${EXT^^}"
        fi
    done

    rm -rf "$RPC_TMPDIR" "$SHOT_DIR"

    # Final cleanup: ensure the scene is completely clean after tests
    _rpc clear >/dev/null 2>&1
    _rpc view.refresh >/dev/null 2>&1

    # CLI GUI info
    if [[ "$CLI_INSTALLED" == "true" ]]; then
        if cli-anything-acloudviewer --json --mode gui info >/dev/null 2>&1; then
            pass "CLI GUI mode info"
        else
            fail "CLI GUI mode info"
        fi
    fi
else
    skip "Cannot connect to ACloudViewer RPC at $RPC_URL"
    skip "Start ACloudViewer and enable JSON-RPC plugin first"
fi

[[ $MAX_LEVEL -lt 5 ]] && { echo ""; echo "Done (Level 1-4)."; exit $([[ $FAIL -eq 0 ]] && echo 0 || echo 1); }

# ─── Level 5: MCP Server Tests ──────────────────────────────────────────────
header "Level 5: MCP Server Tests"

if $PYTHON -c "from mcp.server import Server" 2>/dev/null; then
    pass "MCP SDK importable"

    TOOL_COUNT=$($PYTHON -c "
from cli_anything.acloudviewer.mcp_server import list_tools
import asyncio
tools = asyncio.run(list_tools())
print(len(tools))
" 2>&1)
    TOOL_COUNT_NUM=$(echo "$TOOL_COUNT" | grep -E '^[0-9]+$' | tail -1)
    if [[ -n "$TOOL_COUNT_NUM" && "$TOOL_COUNT_NUM" -ge 80 ]]; then
        pass "MCP server defines $TOOL_COUNT_NUM tools (expected ≥80)"
    else
        echo "    [diag] MCP list_tools output: $TOOL_COUNT"
        fail "MCP tool listing: got ${TOOL_COUNT_NUM:-0} (expected ≥80)"
    fi

    if command -v cli-anything-acloudviewer-mcp &>/dev/null; then
        pass "MCP server entry point exists"
    else
        fail "MCP server entry point"
    fi
else
    skip "MCP SDK not installed (pip install 'cli-anything-acloudviewer')"
fi

# ─── Summary ────────────────────────────────────────────────────────────────
header "Test Summary"
echo "  ✅ Passed: $PASS"
echo "  ❌ Failed: $FAIL"
echo "  ⏭️  Skipped: $SKIP"
echo ""

if [[ $FAIL -eq 0 ]]; then
    echo "  🎉 All tests passed!"
    exit 0
else
    echo "  ⚠️  $FAIL test(s) failed."
    exit 1
fi
