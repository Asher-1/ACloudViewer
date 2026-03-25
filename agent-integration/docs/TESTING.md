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
2. Go to **Plugins** menu, click **JsonRPC** to toggle it ON
3. The WebSocket server starts on port 6001

### Send a ping

```bash
# Using websocat (install: cargo install websocat)
echo '{"jsonrpc":"2.0","id":1,"method":"ping","params":{}}' | websocat ws://localhost:6001
# Expected: {"id":1,"jsonrpc":"2.0","result":"pong"}
```

### Load a file

```bash
echo '{"jsonrpc":"2.0","id":2,"method":"open","params":{"filename":"/path/to/bunny.ply","silent":true}}' | websocat ws://localhost:6001
```

### List entities

```bash
echo '{"jsonrpc":"2.0","id":3,"method":"scene.list","params":{"recursive":true}}' | websocat ws://localhost:6001
```

### Take a screenshot

```bash
echo '{"jsonrpc":"2.0","id":4,"method":"view.screenshot","params":{"filename":"/tmp/screenshot.png"}}' | websocat ws://localhost:6001
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
```

## Test 4: MCP Server

```bash
# Start MCP server (it communicates via stdio)
cli-anything-acloudviewer-mcp --mode headless

# In another terminal, use the MCP inspector
npx @modelcontextprotocol/inspector cli-anything-acloudviewer-mcp
```

## Test 5: Unit Tests

```bash
cd /path/to/CLI-Anything/acloudviewer/agent-harness
pip install -e ".[dev]"
python -m pytest cli_anything/acloudviewer/tests/ -v
```

Expected: all core tests pass, GUI tests skip unless `ACLOUDVIEWER_E2E_GUI=1`.

## Test 6: Full Integration Test

Set the environment variable and run with a live ACloudViewer instance:

```bash
export ACLOUDVIEWER_E2E_GUI=1
python -m pytest cli_anything/acloudviewer/tests/ -v
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Connection refused on port 6001 | Enable the JsonRPC plugin in ACloudViewer |
| "cloudViewer not installed" | Ensure ACloudViewer binary is on PATH or set ACV_BINARY env var |
| Screenshot is black | Ensure the viewport has loaded geometry first |
| MCP tools not showing | Check `pip install 'cli-anything-acloudviewer'` |
