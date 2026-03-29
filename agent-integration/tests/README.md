# ACloudViewer Agent Integration — Unified Test Suite

## Quick Start

> **Windows users:** If CLI tests hang, see [../docs/TROUBLESHOOTING.md](../docs/TROUBLESHOOTING.md#windows-cli-anything-acloudviewer-convert-hangs--freezes).

```bash
# Run all tests (auto-skips levels where deps are missing)
# Works on Linux, macOS, and Windows — the primary test entry point
cd agent-integration/tests
python -m pytest test_integration.py -v
```

> **Note**: `run_all_tests.sh` is a legacy convenience wrapper for Linux/macOS.
> All tests live in `test_integration.py` (pytest) for cross-platform CI support.

## Test Levels

| Level | What's Tested | Dependencies | Command |
|-------|--------------|--------------|---------|
| 1 | C++ plugin source & build, SIBR commands | cmake (optional) | `-k level1` |
| 2 | CLI harness commands, reconstruct, SIBR, convert | `cli-anything-acloudviewer` CLI | `-k level2` |
| 3 | Headless processing, format conversion, batch | ACloudViewer binary | `-k level3` |
| 4 | GUI JSON-RPC, Colmap reconstruct, methods | Running ACloudViewer | `-k level4` |
| 5 | MCP tools (Colmap, SIBR, processing) | `mcp` Python package | `-k level5` |

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
| `ACV_REPO_ROOT` | auto-detected | Path to ACloudViewer repository root |
| `ACV_BUILD_DIR` | `<repo>/build_app` | Path to CMake build directory |
| `ACV_BINARY` | auto-detected | Path to ACloudViewer binary |
| `ACV_RPC_URL` | `ws://localhost:6001` | WebSocket URL for GUI RPC tests |

## What Gets Verified

### Level 1 — C++ Plugin & SIBR Commands (no runtime needed)
- Plugin source files exist (JsonRPCPlugin.cpp/h, qSIBRCommands.h)
- All RPC methods registered via `reg()` in the method registry (including `colmap.reconstruct`, `colmap.run`, `file.convert`, `cloud.setActiveSf`, `mesh.volume`)
- All method handlers declared in the header (30+ handler methods + `registerMethods`)
- Method count ≥ 40 (counted by `reg("` occurrences)
- `colmap.reconstruct` params verified (image_path, workspace_path, quality, etc.)
- `file.convert` params verified (input, output, input_filter, output_filter)
- SIBR_VIEWER command structure verified (6 viewer types, all CLI options)
- SIBR_TOOL command structure verified (10 dataset tools)
- Plugin compiles (if build dir exists)

### Level 2 — CLI Harness
- `cli-anything-acloudviewer --help` works
- All subcommands respond to `--help` (convert, batch-convert, formats, process, reconstruct, sibr, scene, view, session, info)
- Process subcommands all registered
- SIBR subcommands all registered (prepare-colmap, texture-mesh, etc.)
- Reconstruct subcommands all registered (auto, mesh, extract-features, match, sparse, undistort, dense-stereo, fuse, poisson, delaunay-mesh, texture-mesh, convert-model, analyze-model)
- `reconstruct auto --help` mentions camera model option
- Headless mode returns valid JSON
- Session management works

### Level 3 — Headless Processing & Format Conversion
- PLY → PCD, PCD → PLY roundtrip (via binary, verifies output existence)
- PCD → DRC, DRC → PCD roundtrip (Draco compression, via binary)
- Basic format conversion: PLY → ASC, BIN, VTK, STL (via binary)
- CLI harness: PLY → PCD, PCD → DRC, batch-convert PLY → PCD
- Point cloud subsample (binary and CLI harness)
- Normal computation (binary and CLI harness)
- Format listing with correct contents

### Level 4 — GUI RPC
- WebSocket ping/pong
- methods.list returns methods with descriptions (dynamic registry)
- scene CRUD (list, info, remove, setVisible, select, clear)
- Cloud SF management (coordToSf, setActiveSf, renameSf, removeSf, removeAllSfs, filterSf)
- Cloud geometry (removeRgb, computeNormals, invertNormals, removeNormals, merge)
- Mesh operations (simplify, smooth, subdivide, samplePoints)
- View control (camera, orientation, zoom, screenshot, perspective, pointSize)
- Entity operations (rename, setColor)
- File I/O (open, export, convert)
- Full workflow tests (load → orient → zoom → screenshot)

### Level 5 — MCP Server
- MCP SDK importable
- Tool listing returns ≥ 95 tools
- Core tool names present (open_file, convert_format, subsample, etc.)
- GUI-mode cloud/mesh tools present (cloud_set_active_sf, cloud_remove_sf, cloud_merge_gui, mesh_extract_vertices_gui, mesh_volume_gui, etc.)
- All 13 Colmap MCP tools present (colmap_auto_reconstruct, colmap_extract_features, colmap_match_features, colmap_sparse_reconstruct, colmap_undistort, colmap_dense_stereo, colmap_stereo_fusion, colmap_poisson_mesh, colmap_delaunay_mesh, colmap_image_texturer, colmap_model_converter, colmap_analyze_model, colmap_run)
- All 11 SIBR MCP tools present (sibr_tool, sibr_prepare_colmap, sibr_texture_mesh, sibr_unwrap_mesh, sibr_tonemapper, sibr_align_meshes, sibr_camera_converter, sibr_nvm_to_sibr, sibr_crop_from_center, sibr_clipping_planes, sibr_distord_crop)
- Entry point executable
