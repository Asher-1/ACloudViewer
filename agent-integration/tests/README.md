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
| 1 | Plugin source/build, AICore batch headers/RPC whitelist/boundary | cmake (optional) | `-k level1` |
| 2 | CLI harness, exact AICore registry, ordinary plugin passthrough | `cli-anything-acloudviewer` CLI | `-k level2` |
| 3 | Headless processing/conversion plus all installed AICore schemas | ACloudViewer binary | `-k level3` |
| 4 | GUI JSON-RPC (cloud/mesh/scene/view/transform/colmap/AICore) | Running ACloudViewer | `-k level4` |
| 5 | MCP tools, including generic AICore and plugin tools | `mcp` Python package | `-k level5` |

## Run Specific Levels

```bash
# Only C++ source checks (no build needed)
python -m pytest test_integration.py -v -k "level1 and not builds"

# Only CLI tests
python -m pytest test_integration.py -v -k "level2"

# Headless processing (real data)
python -m pytest test_integration.py -v -k "level3"

# GUI RPC (start ACloudViewer first, enable JSON-RPC plugin)
export ACV_RPC_URL=ws://127.0.0.1:6001
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
| `ACV_RPC_URL` | `ws://127.0.0.1:6001` | WebSocket URL for GUI RPC tests |

## What Gets Verified

### Level 1 — C++ Plugin & SIBR Commands (no runtime needed)
- Plugin source files exist (JsonRPCPlugin.cpp/h, qSIBRCommands.h); every handler declared in `JsonRPCPlugin.h` is defined exactly once across `src/*.cpp`, and every `src/*.cpp` is listed in `src/CMakeLists.txt`
- No RPC source starts a `QProcess` except through `return runChildProcess(...)` with the process parented to the plugin; none uses `waitFor*()` or a nested `QEventLoop`
- The server listens on `QHostAddress::LocalHost` and refuses Origin-carrying (browser) handshakes
- All RPC methods registered via `reg()` in the method registry (including `aicore.batch`, `colmap.reconstruct`, `colmap.run`, `file.convert`, `cloud.setActiveSf`, `mesh.volume`)
- All 17 AICore batch headers use the shared HELP/schema contract and appear in the RPC whitelist
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
- `ai list` exactly matches the 17-task source contract
- `ai run/schema` and `plugin run` expose flexible option/argv surfaces

### Level 3 — Headless Processing & Format Conversion

**AICore packaging gate**: every registered task executes `HELP` through the
CLI harness and returns a non-empty `acloudviewer.aicore.batch/1` schema with
`RESULT_JSON`. This fails closed when an installer omits a plugin.

**Skip policy**: a skip is decided before anything runs, from facts known up
front; once a test has run a command, it passes or fails. A crash (signal or
Windows exception), a non-zero exit, a failed prerequisite step or an import
error from an installed harness is always a failure. `_require_plugin(TARGET)`
runs first in every plugin test: it skips a plugin the platform cannot build,
skips a plugin a developer checkout did not build, and fails when an installer
(`ACV_REQUIRE_BINARY=1`) lacks a plugin its platform builds. Plugin libraries
are looked up where `ecvApplicationBase::setupPaths()` loads them
(`Contents/cvPlugins` on macOS, `plugins/` elsewhere). L5 MCP tests use
`@_needs_mcp`, decided from `find_spec`. The L1 gate
`test_level1_skips_precede_execution` fails any `pytest.skip` (or helper that
calls it) placed after a child process or RPC call, or inside an exception
handler.

The only platform skips are build or platform facts:

| Skip | Where | Why |
|---|---|---|
| FBX conversion | Linux, macOS | The bundled FBX SDK is Windows-only (`.lib`/`.dll`); only `ci_utils.ps1` enables `PLUGIN_IO_QFBX` |
| `-PCV` headless | Windows | qPCV renders through `QOpenGLContext`; the minimal QPA used there has no GL context |

Source, CLI-help and MCP-schema checks never skip by platform (SIBR's CLI and
MCP surface exist on macOS even though CI builds the plugin only on Linux and
Windows).

**Binary-level tests** (13 new command flags):
- `-HELP`, `-SOR`, `-CROP`, `-C2C_DIST`, `-LOG_FILE`, `-DENSITY`, `-CURV`, `-ROUGH`
- `-BEST_FIT_PLANE`, `-MERGE_CLOUDS`, `-EXTRACT_CC`, `-FILTER_SF`, `-SAMPLE_MESH`, `-ICP`

**Format conversion**:
- PLY → PCD, PCD → PLY roundtrip (via binary; output existence **and** `st_size > 0`)
- PCD → DRC, DRC → PCD roundtrip (Draco compression)
- Basic cloud export: PLY → PLY, ASC, BIN, VTK
- ASCII filename variants (`.xyz`, `.txt`, `.csv`, `.pts`) as ASC export
- **DXF**, **SHP**, **PTX** (CV_io filters, built on every platform), **SBF** (qCoreIO)
- LAS / LAZ / E57 (skipped only when the plugin was not built in a developer checkout)
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
- **Child processes**: the server answers pings while a child runs, concurrent children answer independently, the deadline kills and answers, `timeout_ms` outside (0, 24 h] is rejected
- **Server boundary**: Origin handshake gets 403, the port is not reachable on a non-loopback address, malformed JSON gets `-32700`
- View control (camera, orientation, zoom, screenshot, perspective, pointSize)
- Entity operations (rename, setColor)
- File I/O (open, export, convert; extended formats and error cases)
- Full workflow tests (load → orient → zoom → screenshot)

### Level 5 — MCP Server
- MCP SDK importable
- Tool listing returns ≥ 182 tools (including `volume_25d`, `crop_2d`, PCL, plugin processing)
- Core tool names present (open_file, convert_format, subsample, etc.)
- GUI-mode cloud/mesh tools present (cloud_set_active_sf, cloud_remove_sf, cloud_merge_gui, mesh_extract_vertices_gui, mesh_volume_gui, etc.)
- All 13 Colmap MCP tools present (colmap_auto_reconstruct, colmap_extract_features, colmap_match_features, colmap_sparse_reconstruct, colmap_undistort, colmap_dense_stereo, colmap_stereo_fusion, colmap_poisson_mesh, colmap_delaunay_mesh, colmap_image_texturer, colmap_model_converter, colmap_analyze_model, colmap_run)
- All 11 SIBR MCP tools present
- Entry point executable
