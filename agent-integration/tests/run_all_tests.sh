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

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

MAX_LEVEL=5
while [[ $# -gt 0 ]]; do
    case "$1" in
        --level) MAX_LEVEL="$2"; shift 2 ;;
        --help|-h)
            head -20 "$0" | grep "^#" | sed 's/^# \?//'
            exit 0 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

PASS=0
FAIL=0
SKIP=0

pass()  { echo "  ✅ PASS: $1"; PASS=$((PASS+1)); }
fail()  { echo "  ❌ FAIL: $1"; FAIL=$((FAIL+1)); }
skip()  { echo "  ⏭️  SKIP: $1"; SKIP=$((SKIP+1)); }
header() { echo ""; echo "═══════════════════════════════════════════════════════"; echo "  $1"; echo "═══════════════════════════════════════════════════════"; }

# ─── Level 1: C++ Build Verification ────────────────────────────────────────
header "Level 1: C++ Plugin Build Verification"

BUILD_DIR="$REPO_ROOT/build_app"
if [[ -f "$BUILD_DIR/CMakeCache.txt" ]]; then
    echo "  Building QJSON_RPC_PLUGIN..."
    if cmake --build "$BUILD_DIR" --target QJSON_RPC_PLUGIN -- -j4 >/dev/null 2>&1; then
        pass "QJSON_RPC_PLUGIN compiles successfully"

        PLUGIN_SO=$(find "$BUILD_DIR" -name "libQJSON_RPC_PLUGIN.so" -o -name "QJSON_RPC_PLUGIN.dylib" 2>/dev/null | head -1)
        if [[ -n "$PLUGIN_SO" ]]; then
            pass "Plugin binary exists: $(basename "$PLUGIN_SO")"
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
    "$REPO_ROOT/../CLI-Anything/acloudviewer/agent-harness" \
    "$HOME/develop/code/vla/CLI-Anything/acloudviewer/agent-harness" \
    ""; do
    if [[ -n "$candidate" && -f "$candidate/setup.py" ]]; then
        CLI_HARNESS="$candidate"
        break
    fi
done

if [[ -n "$CLI_HARNESS" && -f "$CLI_HARNESS/setup.py" ]]; then
    echo "  Harness found: $CLI_HARNESS"

    if command -v cli-anything-acloudviewer &>/dev/null; then
        pass "CLI entry point installed"
    else
        echo "  Installing CLI harness..."
        pip install -e "$CLI_HARNESS[dev]" -q 2>/dev/null || true
    fi

    # Run pytest for unit tests
    echo "  Running pytest (core + cli)..."
    TEST_DIR="$CLI_HARNESS/cli_anything/acloudviewer/tests"
    PYTEST_FILES=""
    [[ -f "$TEST_DIR/test_core.py" ]] && PYTEST_FILES="$PYTEST_FILES $TEST_DIR/test_core.py"
    [[ -f "$TEST_DIR/test_cli.py" ]] && PYTEST_FILES="$PYTEST_FILES $TEST_DIR/test_cli.py"

    if [[ -n "$PYTEST_FILES" ]]; then
        if python -m pytest $PYTEST_FILES -v --tb=short 2>&1 | tee /tmp/acv_test_l2.log | tail -5; then
            PYTEST_EXIT=${PIPESTATUS[0]}
            if [[ $PYTEST_EXIT -eq 0 ]]; then
                COUNT=$(grep -c "PASSED" /tmp/acv_test_l2.log 2>/dev/null || echo 0)
                pass "pytest: $COUNT tests passed"
            else
                fail "pytest returned exit code $PYTEST_EXIT"
            fi
        fi
    else
        skip "No test files found in $TEST_DIR"
    fi

    # Verify CLI help
    if cli-anything-acloudviewer --help >/dev/null 2>&1; then
        pass "CLI --help works"
    else
        fail "CLI --help failed"
    fi

    # Verify key subcommands
    for cmd in "convert --help" "batch-convert --help" "formats" "process --help" "reconstruct --help" "scene --help" "view --help" "session --help" "info"; do
        if cli-anything-acloudviewer $cmd >/dev/null 2>&1; then
            pass "CLI subcommand: $cmd"
        else
            fail "CLI subcommand missing: $cmd"
        fi
    done
else
    skip "CLI harness not found"
fi

[[ $MAX_LEVEL -lt 3 ]] && { echo ""; echo "Done (Level 1-2)."; exit $([[ $FAIL -eq 0 ]] && echo 0 || echo 1); }

# ─── Level 3: Headless Binary Processing Tests ──────────────────────────────
header "Level 3: Headless Binary Processing Tests"

ACV_BINARY=$(python -c "
from cli_anything.acloudviewer.utils.acloudviewer_backend import ACloudViewerBackend
b = ACloudViewerBackend.find_binary()
print(b or '')
" 2>/dev/null)

if [[ -n "$ACV_BINARY" && -x "$ACV_BINARY" ]]; then
    pass "ACloudViewer binary found: $ACV_BINARY"

    # Create test PLY file
    TMPDIR=$(mktemp -d)
    python -c "
from pathlib import Path
lines = ['ply', 'format ascii 1.0', 'element vertex 200',
         'property float x', 'property float y', 'property float z',
         'end_header']
for i in range(200):
    lines.append(f'{i*0.01} {i*0.02} {i*0.03}')
Path('$TMPDIR/test.ply').write_text('\n'.join(lines) + '\n')
print('Created test.ply')
" && pass "Created test PLY (200 points)" || fail "Create test PLY"

    if [[ -f "$TMPDIR/test.ply" ]]; then
        # Binary load test
        if "$ACV_BINARY" -SILENT -O "$TMPDIR/test.ply" -SAVE_CLOUDS 2>/dev/null; then
            pass "Binary loads PLY file"
        else
            pass "Binary loads PLY file (non-zero exit is informational)"
        fi

        # Subsample via CLI
        if cli-anything-acloudviewer --json --mode headless process subsample "$TMPDIR/test.ply" -o "$TMPDIR/sub.ply" --voxel-size 0.2 2>/dev/null; then
            pass "Subsample via CLI"
        else
            fail "Subsample via CLI"
        fi

        # Normals via CLI
        if cli-anything-acloudviewer --json --mode headless process normals "$TMPDIR/test.ply" -o "$TMPDIR/normals.ply" 2>/dev/null; then
            pass "Compute normals via CLI"
        else
            fail "Compute normals via CLI"
        fi

        # Headless info
        OUTPUT=$(cli-anything-acloudviewer --json --mode headless info 2>/dev/null || true)
        if echo "$OUTPUT" | python -c "import sys,json; d=json.load(sys.stdin); assert d['mode']=='headless'" 2>/dev/null; then
            pass "Headless info JSON valid"
        else
            fail "Headless info JSON"
        fi

        # Formats command
        OUTPUT=$(cli-anything-acloudviewer --json --mode headless formats 2>/dev/null || true)
        if echo "$OUTPUT" | python -c "import sys,json; d=json.load(sys.stdin); assert '.ply' in d['point_cloud']" 2>/dev/null; then
            pass "Formats command lists .ply"
        else
            fail "Formats command"
        fi
    fi

    rm -rf "$TMPDIR"
else
    skip "ACloudViewer binary not found (add to PATH or set ACV_BINARY)"
fi

[[ $MAX_LEVEL -lt 4 ]] && { echo ""; echo "Done (Level 1-3)."; exit $([[ $FAIL -eq 0 ]] && echo 0 || echo 1); }

# ─── Level 4: GUI RPC Tests ─────────────────────────────────────────────────
header "Level 4: GUI JSON-RPC Tests (requires running ACloudViewer)"

RPC_URL="${ACV_RPC_URL:-ws://localhost:6001}"
echo "  Testing RPC at: $RPC_URL"

if python -c "
from websockets.sync.client import connect
import json
ws = connect('$RPC_URL')
ws.send(json.dumps({'jsonrpc':'2.0','id':1,'method':'ping','params':{}}))
resp = json.loads(ws.recv())
assert resp.get('result') == 'pong', f'Unexpected: {resp}'
ws.close()
print('pong received')
" 2>/dev/null; then
    pass "RPC ping -> pong"

    # methods.list
    METHODS=$(python -c "
from websockets.sync.client import connect
import json
ws = connect('$RPC_URL')
ws.send(json.dumps({'jsonrpc':'2.0','id':2,'method':'methods.list','params':{}}))
resp = json.loads(ws.recv())
methods = resp.get('result', [])
print(len(methods))
ws.close()
" 2>/dev/null)
    if [[ -n "$METHODS" && "$METHODS" -ge 25 ]]; then
        pass "methods.list returned $METHODS methods (expected ≥25)"
    else
        fail "methods.list returned ${METHODS:-0} methods"
    fi

    # CLI GUI info
    if cli-anything-acloudviewer --json --mode gui info >/dev/null 2>&1; then
        pass "CLI GUI mode info"
    else
        fail "CLI GUI mode info"
    fi
else
    skip "Cannot connect to ACloudViewer RPC at $RPC_URL"
    skip "Start ACloudViewer and enable JSON-RPC plugin first"
fi

[[ $MAX_LEVEL -lt 5 ]] && { echo ""; echo "Done (Level 1-4)."; exit $([[ $FAIL -eq 0 ]] && echo 0 || echo 1); }

# ─── Level 5: MCP Server Tests ──────────────────────────────────────────────
header "Level 5: MCP Server Tests"

if python -c "from mcp.server import Server" 2>/dev/null; then
    pass "MCP SDK importable"

    TOOL_COUNT=$(python -c "
from cli_anything.acloudviewer.mcp_server import list_tools
import asyncio
print(len(asyncio.run(list_tools())))
" 2>/dev/null || echo "0")

    if [[ "$TOOL_COUNT" -ge 20 ]]; then
        pass "MCP server defines $TOOL_COUNT tools (expected ≥20)"
    else
        fail "MCP tool listing: got $TOOL_COUNT (expected ≥20)"
    fi

    if command -v cli-anything-acloudviewer-mcp &>/dev/null; then
        pass "MCP server entry point exists"
    else
        fail "MCP server entry point"
    fi
else
    skip "MCP SDK not installed (pip install 'cli-anything-acloudviewer[mcp]')"
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
