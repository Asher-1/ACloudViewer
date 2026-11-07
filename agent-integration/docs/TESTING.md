# End-to-End Testing Guide

This document describes how to test the complete ACloudViewer agent integration
chain from GUI to AI agent.

## Prerequisites

1. **ACloudViewer** built with `PLUGIN_STANDARD_QJSONRPC=ON`
2. **Python 3.10+** with the CLI harness installed
3. A sample point cloud file (e.g., `bunny.ply`)

## Test 1: JSON-RPC Plugin (GUI Direct)

### Start the server

1. Launch ACloudViewer
2. The JsonRPC plugin starts automatically (**Plugins > JsonRPC** toggles it)
3. The WebSocket server listens on `ACV_RPC_PORT` (default 6001); a port
   that is already taken is reported in the console instead of failing
   silently

Unattended (no visible window), e.g. next to an interactive session:

```bash
# offscreen takes its OpenGL context from an X display; without one the GUI
# exits with "needs OpenGL 2.1". On a display-less host use Xvfb.
QT_QPA_PLATFORM=offscreen ACV_RPC_PORT=6011 ACloudViewer &
QT_QPA_PLATFORM=offscreen xvfb-run -a ACloudViewer &        # no X server
export ACV_RPC_URL=ws://127.0.0.1:6011                      # clients/tests
```

### Send a ping

```bash
# Using websocat (install: cargo install websocat)
echo '{"jsonrpc":"2.0","id":1,"method":"ping","params":{}}' | websocat ws://127.0.0.1:6001
# Expected: {"id":1,"jsonrpc":"2.0","result":"pong"}
```

### Load a file

```bash
echo '{"jsonrpc":"2.0","id":2,"method":"open","params":{"filename":"/path/to/bunny.ply","silent":true}}' | websocat ws://127.0.0.1:6001
```

### List entities

```bash
echo '{"jsonrpc":"2.0","id":3,"method":"scene.list","params":{"recursive":true}}' | websocat ws://127.0.0.1:6001
```

### Take a screenshot

```bash
echo '{"jsonrpc":"2.0","id":4,"method":"view.screenshot","params":{"filename":"/tmp/screenshot.png"}}' | websocat ws://127.0.0.1:6001
```

## Test 2: CLI Harness (GUI Mode)

With ACloudViewer running and JSON-RPC enabled:

```bash
# Verify connection
cli-anything-acloudviewer --json --mode gui info

# Load a file
cli-anything-acloudviewer --mode gui open /path/to/bunny.ply

# List scene
cli-anything-acloudviewer --json --mode gui scene list

# Set view and screenshot
cli-anything-acloudviewer --mode gui view orient iso1
cli-anything-acloudviewer --mode gui view zoom
```

## Test 3: CLI Harness (Headless Mode)

No GUI needed, requires ACloudViewer binary on PATH:

```bash
# Check info
cli-anything-acloudviewer --json --mode headless info

# Process a point cloud
cli-anything-acloudviewer --mode headless process subsample input.ply -o out.ply --voxel-size 0.05
cli-anything-acloudviewer --mode headless process normals input.ply -o normals.ply
cli-anything-acloudviewer --json --mode headless process icp source.ply target.ply

# Verify the complete AICore plugin registry without loading models
cli-anything-acloudviewer --json ai list
cli-anything-acloudviewer --json --mode headless ai schema ocr

# Exercise ordinary plugin argv passthrough
cli-anything-acloudviewer --mode headless plugin run ANIMATION \
  --arg=-FPS --arg=24 --arg=-TOTAL_FRAMES --arg=10
```

## Test 4: MCP Server

```bash
# Start MCP server (it communicates via stdio)
cli-anything-acloudviewer-mcp --mode headless

# In another terminal, use the MCP inspector
npx @modelcontextprotocol/inspector cli-anything-acloudviewer-mcp
```

## Test 5: Pytest Integration Suite (agent-integration)

The unified pytest suite in `agent-integration/tests/test_integration.py`
covers five test levels:

```bash
cd agent-integration/tests

# Run all levels (auto-skips unavailable dependencies)
python -m pytest test_integration.py -v

# Run a specific level
python -m pytest test_integration.py -v -k "level1"   # C++ source checks
python -m pytest test_integration.py -v -k "level2"   # CLI commands
python -m pytest test_integration.py -v -k "level3"   # Headless processing
python -m pytest test_integration.py -v -k "level4"   # GUI RPC
python -m pytest test_integration.py -v -k "level5"   # MCP tools
```

| Level | Tests | Dependencies |
|-------|-------|-------------|
| 1 | Plugin source, 17 AICore batch headers, `aicore.batch` whitelist, strict ownership boundary | cmake (optional) |
| 2 | CLI help plus exact AICore registry and generic plugin passthrough | `cli-anything-acloudviewer` |
| 3 | Headless processing/conversion and 17/17 installed-plugin HELP schemas | ACloudViewer binary |
| 4 | RPC ping, scene CRUD, cloud/mesh/Colmap methods, `aicore.batch` HELP for all 17 tasks, server responsiveness while `process.run_cli` runs (skipped without a running GUI) | Running ACloudViewer |
| 5 | MCP tool inventory and AICore/plugin generic tool schemas | `mcp` Python package |

## Test 6: CLI-Anything Harness Tests

```bash
cd /path/to/CLI-Anything/acloudviewer/agent-harness
pip install -e ".[dev]"
python -m pytest cli_anything/acloudviewer/tests/ -v
```

Expected: all core tests pass, GUI tests skip unless `ACLOUDVIEWER_E2E_GUI=1`.

The CI coverage gate is cumulative:

```text
L1: source/header/RPC ownership contract (no binary)
L2: CLI registry and argument surface (harness only)
L3: every registered task resolves to an installed plugin HELP schema
L4: the installed GUI, started offscreen, serves every RPC test
L5: MCP exposes list/schema/run and ordinary plugin passthrough
```

L1/L2/L5 run on Ubuntu (Python 3.10–3.12), macOS and Windows (3.12) in
`agent-integration.yml`. The Ubuntu installer job starts the packaged GUI
with `QT_QPA_PLATFORM=offscreen` on Xvfb (software Mesa), waits for `ping`,
runs L4, and fails if the GUI is not serving before or is gone after.

Linux CPU/CUDA, macOS, and Windows installer workflows run L3 against the
packaged binary with `ACV_REQUIRE_BINARY=1`, so a missing binary fails
instead of skipping. A missing plugin, stale Python registry entry, missing
`RESULT_JSON`, or non-`acloudviewer.aicore.batch/1` schema fails the job.
L1 also requires the `aicore.batch` whitelist to equal the registry exactly,
each command class to be passed to `registerCommand()`, and documented RPC
method totals to match; L5 requires every documented MCP tool count and the
OpenClaw tool categories to match `list_tools()`.

## Test 7: Full Integration Test (GUI)

Set the environment variable and run with a live ACloudViewer instance:

```bash
export ACLOUDVIEWER_E2E_GUI=1
python -m pytest cli_anything/acloudviewer/tests/ -v
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Connection refused on port 6001 | Enable the JsonRPC plugin; check the console for "Cannot listen on port" (another instance owns it; set `ACV_RPC_PORT`) |
| "cloudViewer not installed" | Ensure ACloudViewer binary is on PATH or set ACV_BINARY env var |
| Screenshot is black | Ensure the viewport has loaded geometry first |
| MCP tools not showing | Check `pip install 'cli-anything-acloudviewer'` |
| RPC error missing details | Check the `data` field in the error response for structured context |
| Test says "Missing method X" | Run `methods.list` to verify registered methods vs test expectations |
