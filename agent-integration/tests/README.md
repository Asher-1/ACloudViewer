# ACloudViewer Agent Integration — Unified Test Suite

## Quick Start

> **Windows users:** If CLI tests hang, see [../docs/TROUBLESHOOTING.md](../docs/TROUBLESHOOTING.md#windows-cli-hangs).

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
| 3 | Headless processing (SOR, CROP, C2C, ICP, density, etc.), format conversion, batch | ACloudViewer binary | `-k level3` |
| 4 | GUI JSON-RPC (cloud/mesh/scene/view/transform/colmap), mesh.simplify/smooth/subdivide/sample/merge, methods | Running ACloudViewer | `-k level4` |
| 5 | MCP tools (178 tools, Colmap, SIBR, PCL, plugins) | `mcp` Python package | `-k level5` |

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

**Binary-level tests** (13 new command flags):
- `-HELP`, `-SOR`, `-CROP`, `-C2C_DIST`, `-LOG_FILE`, `-DENSITY`, `-CURV`, `-ROUGH`
- `-BEST_FIT_PLANE`, `-MERGE_CLOUDS`, `-EXTRACT_CC`, `-FILTER_SF`, `-SAMPLE_MESH`, `-ICP`

**Format conversion**:
- PLY → PCD, PCD → PLY roundtrip (via binary; output existence **and** `st_size > 0`)
- PCD → DRC, DRC → PCD roundtrip (Draco compression)
- Basic cloud export: PLY → PLY, ASC, BIN, VTK
- ASCII filename variants (`.xyz`, `.txt`, `.csv`, `.pts`) as ASC export
- **DXF** export when the build includes DXF IO (otherwise skipped)
- Optional formats when plugins exist: LAS, E57, SBF (skipped if IO unavailable)
- **Mesh exports** (Delaunay → mesh): OBJ, OFF, STL; **FBX** when the FBX plugin is present
- **Round-trip checks**: PLY ↔ ASC ↔ PLY and PLY ↔ VTK ↔ PLY point counts; mesh OBJ/OFF vertex counts
- **Negative tests**: missing input file (expect failure); invalid `-C_EXPORT_FMT` token
- **Content checks**: ASCII export row counts and cross-extension consistency

**CLI harness**: PLY → PCD, PCD → DRC, batch-convert, subsample, normals, density, curvature, roughness, feature, extract-cc, color-banding, merge-clouds, cross-section, best-fit-plane, SOR, delaunay, sample-mesh, remove-rgb, extract-vertices, flip-triangles, mesh-volume, merge-meshes, SF operations (coord-to-sf, arithmetic, operation, gradient, filter, convert-to-rgb, set-active, rename, remove, remove-all), normals operations (octree, orient-mst, invert, clear, to-dip, to-sfs)

Mesh-oriented cases use the **`mesh_ply`** / **`_MESH_PLY_CONTENT`** fixture: a 10×10 grid PLY that triangulates cleanly. Cloud-only cases use the **`sample_ply`** fixture.

### Level 4 — GUI RPC
- WebSocket ping/pong
- methods.list returns methods with descriptions (dynamic registry)
- scene CRUD (list, info, remove, setVisible, **select**, clear)
- Cloud SF management (coordToSF, setActiveSf, renameSf, removeSf, removeAllSfs, filterSf)
- Cloud geometry and processing (paint, **paintByScalarField**, crop, scalar-field ops, normals, merge, density/curvature/roughness/features, SOR, delaunay, etc.)
- **Mesh RPC** (Level 4e): simplify (quadric), smooth (laplacian), subdivide (midpoint), samplePoints (uniform), volume, flipTriangles, extractVertices, **merge**
- **Transform RPC**: transform.apply (identity matrix)
- **Colmap RPC**: colmap.reconstruct, colmap.run
- View control (camera, orientation, zoom, screenshot, perspective, pointSize)
- Entity operations (rename, setColor)
- File I/O (open, export, convert; extended formats and error cases)
- Full workflow tests (load → orient → zoom → screenshot)

### Level 5 — MCP Server
- MCP SDK importable
- Tool listing returns ≥ 178 tools (including `volume_25d`, `crop_2d`, PCL, plugin processing)
- Core tool names present (open_file, convert_format, subsample, etc.)
- GUI-mode cloud/mesh tools present (cloud_set_active_sf, cloud_remove_sf, cloud_merge_gui, mesh_extract_vertices_gui, mesh_volume_gui, etc.)
- All 13 Colmap MCP tools present (colmap_auto_reconstruct, colmap_extract_features, colmap_match_features, colmap_sparse_reconstruct, colmap_undistort, colmap_dense_stereo, colmap_stereo_fusion, colmap_poisson_mesh, colmap_delaunay_mesh, colmap_image_texturer, colmap_model_converter, colmap_analyze_model, colmap_run)
- All 11 SIBR MCP tools present
- Entry point executable
