# ACloudViewer Agent Integration — Unified Test Suite

## Quick Start

```bash
# Run all tests (auto-skips levels where deps are missing)
cd agent-integration/tests
python -m pytest test_integration.py -v

# Or use the shell runner
./run_all_tests.sh
```

## Test Levels

| Level | What's Tested | Dependencies | Command |
|-------|--------------|--------------|---------|
| 1 | C++ plugin source & build | cmake (optional) | `-k level1` |
| 2 | CLI harness commands & help | `cli-anything-acloudviewer` CLI | `-k level2` |
| 3 | Headless data processing | ACloudViewer binary | `-k level3` |
| 4 | GUI JSON-RPC communication | Running ACloudViewer | `-k level4` |
| 5 | MCP server tool definitions | `mcp` Python package | `-k level5` |

## Run Specific Levels

```bash
# Only C++ source checks (no build needed)
python -m pytest test_integration.py -v -k "level1 and not builds"

# Only CLI tests
python -m pytest test_integration.py -v -k "level2"

# Headless processing (real data)
python -m pytest test_integration.py -v -k "level3"

# GUI RPC (start ACloudViewer first, enable JSON-RPC plugin)
export ACV_RPC_URL=ws://localhost:6001
python -m pytest test_integration.py -v -k "level4"

# MCP server
python -m pytest test_integration.py -v -k "level5"
```

## Shell Runner

```bash
# All levels
./run_all_tests.sh

# Only up to level 2
./run_all_tests.sh --level 2

# Level 1-3 (C++ + CLI + headless)
./run_all_tests.sh --level 3
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ACV_RPC_URL` | `ws://localhost:6001` | WebSocket URL for GUI RPC tests |

## What Gets Verified

### Level 1 — C++ Plugin (no runtime needed)
- Plugin source files exist
- All RPC methods are in the dispatch table
- All method handlers are declared in the header
- Method count ≥ 30
- Plugin compiles (if build dir exists)

### Level 2 — CLI Harness
- `cli-anything-acloudviewer --help` works
- All subcommands respond to `--help`
- Process subcommands all registered
- Headless mode returns valid JSON
- Session management works

### Level 3 — Headless Processing
- PLY → PCD, PLY → XYZ, PLY → OBJ format conversion
- Point cloud subsample
- Normal computation
- Batch directory conversion
- Format listing with correct contents

### Level 4 — GUI RPC
- WebSocket ping/pong
- methods.list returns ≥32 methods with correct names
- scene.list returns entity array
- view.getCamera returns matrix + FOV
- CLI `--mode gui` info command

### Level 5 — MCP Server
- MCP SDK importable
- Tool listing returns ≥20 tools
- All expected tool names present
- Entry point executable
