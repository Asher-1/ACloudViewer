# ACloudViewer Agent Integration

This module provides everything needed to control ACloudViewer from AI agents,
automation scripts, and external tools via three complementary interfaces:

| Interface | Protocol | Best For |
|-----------|----------|----------|
| **JSON-RPC Plugin** | WebSocket (port 6001) | Real-time GUI control |
| **MCP Server** | Model Context Protocol (stdio) | OpenClaw, Cursor, Claude Code |
| **CLI Harness** | Click CLI + REPL | Shell scripts, agent CLIs |

## Quick Start

### 1. Enable the JSON-RPC Plugin (GUI Mode)

Build ACloudViewer with the plugin enabled:

```bash
# Linux / macOS
cmake -DPLUGIN_STANDARD_QJSONRPC=ON ..
make -j$(nproc)

# Windows (Visual Studio)
cmake -DPLUGIN_STANDARD_QJSONRPC=ON ..
cmake --build . --config Release -- /m
```

Launch ACloudViewer, then activate the JSON-RPC server from the plugin menu
(or it auto-listens on port 6001 when toggled on).

### 2. Install the CLI Harness

```bash
pip install git+https://github.com/HKUDS/CLI-Anything.git#subdirectory=acloudviewer/agent-harness
```

Or for local development:

```bash
cd /path/to/CLI-Anything/acloudviewer/agent-harness
pip install -e ".[mcp,dev]"
```

### 3. Use from the Command Line

> **Windows users:** For file conversion and batch operations, **always use `--mode headless`**:
> ```bash
> cli-anything-acloudviewer --mode headless convert input.ply output.pcd
> cli-anything-acloudviewer --mode headless batch-convert ./scans/ ./out/ -f .ply
> ```
> The default `--mode auto` may hang if port 6001 is listening but unresponsive. `--mode headless` invokes the binary directly.

```bash
# Interactive REPL
cli-anything-acloudviewer

# One-shot commands (GUI needed)
cli-anything-acloudviewer open /path/to/scene.ply
cli-anything-acloudviewer --json scene list
cli-anything-acloudviewer view screenshot ./screenshot.png

# SIBR Viewers (requires SIBR plugin)
cli-anything-acloudviewer sibr viewer gaussian --model-path ./output/ --path ./dataset/
cli-anything-acloudviewer sibr viewer ulr --path ./dataset/
cli-anything-acloudviewer sibr viewer remoteGaussian --ip 127.0.0.1 --port 6009

# Force headless (no GUI needed)
cli-anything-acloudviewer --mode headless process icp source.ply target.ply
cli-anything-acloudviewer process subsample input.ply -o output.ply --voxel-size 0.05

# Format conversion (positional: INPUT_FILE OUTPUT_FILE)
cli-anything-acloudviewer convert input.ply output.obj
cli-anything-acloudviewer convert input.pcd output.drc               # Draco compressed
cli-anything-acloudviewer batch-convert ./scans/ ./converted/ -f .ply

# Automatic 3D reconstruction from images
cli-anything-acloudviewer reconstruct auto ./images/ -w ./workspace/ --quality high
cli-anything-acloudviewer reconstruct auto ./images/ -w ./workspace/ --quality low --no-dense
cli-anything-acloudviewer reconstruct auto ./images/ -w ./workspace/ --camera-model OPENCV
cli-anything-acloudviewer reconstruct auto ./images/ -w ./workspace/ --camera-model PINHOLE --quality extreme

# Individual reconstruction steps
cli-anything-acloudviewer reconstruct extract-features ./images/ -w ./workspace/
cli-anything-acloudviewer reconstruct sparse ./workspace/
cli-anything-acloudviewer reconstruct dense-stereo ./workspace/
cli-anything-acloudviewer reconstruct poisson ./workspace/

# Windows-specific: use safe wrapper scripts (force headless mode)
# See agent-integration/scripts/ for acv-convert-safe.ps1 and acv-batch-convert-safe.ps1

# SIBR dataset preparation (requires SIBR plugin)
cli-anything-acloudviewer sibr prepare-colmap ./workspace/
cli-anything-acloudviewer sibr texture-mesh ./workspace/

```

### 4. Use as an MCP Server (OpenClaw / Cursor / Claude Code)

```bash
cli-anything-acloudviewer-mcp
cli-anything-acloudviewer-mcp --mode gui      # force GUI backend
cli-anything-acloudviewer-mcp --mode headless  # force headless
```

### 5. Cursor IDE Integration

A `.cursor/mcp.json` is provided for one-click MCP integration in Cursor:

```json
{
  "mcpServers": {
    "acloudviewer": {
      "command": "cli-anything-acloudviewer-mcp",
      "args": ["--mode", "auto"]
    }
  }
}
```

Three server profiles are pre-configured:
- `acloudviewer` — auto-detects GUI or headless
- `acloudviewer-headless` — always headless (no ACloudViewer needed)
- `acloudviewer-gui` — connects to running ACloudViewer via WebSocket

### 6. Run the Unified Test Suite

```bash
cd agent-integration/tests

# pytest (all levels, auto-skips missing deps)
python -m pytest test_integration.py -v

# pytest (specific level)
python -m pytest test_integration.py -v -k "level1"  # C++ plugin
python -m pytest test_integration.py -v -k "level2"  # CLI harness
python -m pytest test_integration.py -v -k "level3"  # headless processing
python -m pytest test_integration.py -v -k "level4"  # GUI RPC
python -m pytest test_integration.py -v -k "level5"  # MCP server

# shell runner (Linux / macOS / Git Bash on Windows)
./run_all_tests.sh --level 2
```

### Platform Notes

| Platform | Installer | Headless Install |
|----------|-----------|------------------|
| **Linux** | `.run` (Qt IFW) | `QT_QPA_PLATFORM=minimal installer --root <dir> --accept-licenses --accept-messages --confirm-command install` |
| **Windows** | `.exe` (Qt IFW) | Same flags as Linux; set `QT_QPA_PLATFORM=minimal` env var |
| **macOS** | `.dmg` | `hdiutil attach <dmg> -nobrowse -noverify && cp -R /Volumes/ACloudViewer*/*.app ~/` |

- **pytest** (`test_integration.py`) is the primary cross-platform test runner
- **`run_all_tests.sh`** requires Bash (Git Bash or WSL on Windows)
- Set `ACV_BINARY` environment variable to the installed binary path on any platform

| Level | Tests | Dependencies |
|-------|-------|-------------|
| 1 | C++ plugin source, build, dispatch table, SIBR commands | cmake (optional) |
| 2 | CLI commands, help, JSON output, session, reconstruct, SIBR, convert | `cli-anything-acloudviewer` |
| 3 | Format conversion, subsample, normals, batch, PLY→ASC/BIN/VTK | `ACloudViewer binary` |
| 4 | WebSocket ping, scene list, camera, methods, colmap.reconstruct | Running ACloudViewer |
| 5 | MCP tools (processing, Colmap, SIBR), entry point | `mcp` Python package |

## Architecture

```
                    +-----------------+
                    |   AI Agent      |
                    | (OpenClaw /     |
                    |  Cursor / CLI)  |
                    +--------+--------+
                             |
              +--------------+--------------+
              |              |              |
        +-----+-----+ +-----+-----+ +-----+-----+
        | MCP Server | | CLI REPL  | | Direct    |
        | (stdio)    | | (Click)   | | WebSocket |
        +-----+------+ +-----+-----+ +-----+-----+
              |              |              |
              +--------------+--------------+
                             |
                    +--------+--------+
                    | Dual-Mode       |
                    | Backend         |
                    +--------+--------+
                             |
              +--------------+--------------+
              |                             |
     +--------+--------+          +--------+--------+
     | GUI Mode        |          | Headless Mode   |
     | WebSocket RPC   |          | ACloudViewer    |
     | (port 6001)     |          | Binary (subprocess) |
     +--------+--------+          +--------+---------+
              |                             |
     +--------+--------+          +--------+---------+
     | ACloudViewer    |          | ACloudViewer     |
     | Desktop App     |          | Binary CLI       |
     +-----------------+          +------------------+
```

## Directory Structure

```
agent-integration/
├── README.md               # This file — unified reference (CLI, MCP, RPC, testing)
├── cli/                    # CLI harness source (installed via pip)
├── mcp/
│   └── README.md           # MCP server setup and tool reference
├── openclaw/
│   ├── README.md           # OpenClaw integration guide
│   └── openclaw-skill.json # OpenClaw skill manifest
├── docs/
│   ├── JSON-RPC-API.md     # Full JSON-RPC method reference
│   ├── TESTING.md          # End-to-end testing guide
│   └── SIBR-VIEWER-CLI.md  # SIBR viewer CLI implementation guide
├── tests/
│   ├── test_integration.py # Pytest test suite (Levels 1-5)
│   └── run_all_tests.sh    # Bash test runner
└── examples/
    ├── websocket_client.py # Minimal WebSocket client example
    └── batch_process.py    # Headless batch processing example
```

## JSON-RPC API Overview

The `qJSonRPCPlugin` exposes **48+ methods** over WebSocket JSON-RPC 2.0.
Methods are dynamically registered via a method registry — call `methods.list`
for the live catalog.

### File I/O
| Method | Parameters | Description |
|--------|-----------|-------------|
| `open` | `{filename, ?silent, ?filter, ?transformation}` | Load a file |
| `export` | `{entity_id, filename, ?filter}` | Export entity to file |
| `file.convert` | `{input, output, ?input_filter, ?output_filter}` | Convert between formats |

### Scene Tree
| Method | Parameters | Description |
|--------|-----------|-------------|
| `scene.list` | `{?recursive}` | List all entities |
| `scene.info` | `{entity_id}` | Get entity details |
| `scene.remove` | `{entity_id}` | Remove entity |
| `scene.setVisible` | `{entity_id, visible}` | Toggle visibility |
| `scene.select` | `{entity_ids}` | Select entities |
| `clear` | `{}` | Remove all entities |

### Entity Properties
| Method | Parameters | Description |
|--------|-----------|-------------|
| `entity.rename` | `{entity_id, name}` | Rename entity |
| `entity.setColor` | `{entity_id, r, g, b}` | Set display color |

### Point Cloud Processing
| Method | Parameters | Description |
|--------|-----------|-------------|
| `cloud.computeNormals` | `{entity_id, ?radius}` | Estimate normals |
| `cloud.subsample` | `{entity_id, method, ?step, ?count}` | Subsample |
| `cloud.crop` | `{entity_id, min_x..max_z}` | Crop by bbox |
| `cloud.getScalarFields` | `{entity_id}` | List scalar fields |
| `cloud.paintUniform` | `{entity_id, r, g, b}` | Paint uniform color |
| `cloud.paintByHeight` | `{entity_id, ?axis}` | Colorize by height |
| `cloud.paintByScalarField` | `{entity_id, sf_name}` | Colorize by scalar field |

### Point Cloud Scalar Field Management
| Method | Parameters | Description |
|--------|-----------|-------------|
| `cloud.setActiveSf` | `{entity_id, ?field_index, ?field_name}` | Set active scalar field |
| `cloud.removeSf` | `{entity_id, ?field_index, ?field_name}` | Remove a scalar field |
| `cloud.removeAllSfs` | `{entity_id}` | Remove all scalar fields |
| `cloud.renameSf` | `{entity_id, new_name, ?field_index}` | Rename a scalar field |
| `cloud.filterSf` | `{entity_id, min, max}` | Filter points by SF range |
| `cloud.coordToSf` | `{entity_id, ?dimension}` | Create SF from coordinates |

### Point Cloud Geometry
| Method | Parameters | Description |
|--------|-----------|-------------|
| `cloud.removeRgb` | `{entity_id}` | Remove color data |
| `cloud.removeNormals` | `{entity_id}` | Remove normals |
| `cloud.invertNormals` | `{entity_id}` | Flip normal directions |
| `cloud.merge` | `{entity_ids}` | Group multiple clouds |

### View Control
| Method | Parameters | Description |
|--------|-----------|-------------|
| `view.setOrientation` | `{orientation}` | Set camera view |
| `view.zoomFit` | `{?entity_id}` | Zoom to fit |
| `view.refresh` | `{}` | Force redraw |
| `view.setPerspective` | `{mode}` | Toggle perspective |
| `view.setPointSize` | `{action}` | Adjust point size |
| `view.screenshot` | `{filename}` | Capture viewport |
| `view.getCamera` | `{}` | Get camera params |

### Transform
| Method | Parameters | Description |
|--------|-----------|-------------|
| `transform.apply` | `{entity_id, matrix[16]}` | Apply 4x4 matrix |

### Mesh Processing
| Method | Parameters | Description |
|--------|-----------|-------------|
| `mesh.simplify` | `{entity_id, target_count}` | Reduce triangle count |
| `mesh.smooth` | `{entity_id, iterations}` | Laplacian smoothing |
| `mesh.subdivide` | `{entity_id}` | Subdivide mesh |
| `mesh.samplePoints` | `{entity_id, count}` | Sample points from surface |
| `mesh.extractVertices` | `{entity_id}` | Extract vertices as point cloud |
| `mesh.flipTriangles` | `{entity_id}` | Flip triangle winding order |
| `mesh.volume` | `{entity_id}` | Compute enclosed volume |
| `mesh.merge` | `{entity_ids}` | Group multiple meshes |

### Reconstruction
| Method | Parameters | Description |
|--------|-----------|-------------|
| `colmap.reconstruct` | `{image_path, workspace_path, ?quality, ...}` | Full automatic reconstruction |
| `colmap.run` | `{command, ?args, ?kwargs, ?timeout_ms}` | Run any COLMAP subcommand |

### Introspection
| Method | Parameters | Description |
|--------|-----------|-------------|
| `methods.list` | `{}` | List all RPC methods (dynamic registry) |
| `ping` | `{}` | Health check |

## MCP Tools Overview

The MCP server exposes **95+ tools** for AI agent use:

| Category | Tools |
|----------|-------|
| **File I/O** | `open_file`, `convert_format`, `batch_convert`, `list_formats`, `export_entity` |
| **Scene** | `scene_list`, `scene_info`, `scene_remove`, `scene_set_visible`, `scene_select`, `scene_clear` |
| **Entity** | `entity_rename`, `entity_set_color` |
| **View** | `screenshot`, `get_camera`, `view_set_orientation`, `view_zoom_fit`, `view_refresh`, `view_set_perspective`, `view_set_point_size` |
| **Cloud Processing** | `subsample`, `compute_normals`, `crop`, `cloud_paint_uniform`, `cloud_paint_by_height`, `cloud_paint_by_scalar_field`, `cloud_get_scalar_fields` |
| **Cloud SF Management** | `set_active_sf`, `remove_sf`, `remove_all_sfs`, `rename_sf`, `sf_arithmetic`, `sf_operation`, `coord_to_sf`, `sf_gradient`, `filter_sf`, `sf_color_scale`, `sf_convert_to_rgb` |
| **Cloud SF (GUI)** | `cloud_set_active_sf`, `cloud_remove_sf`, `cloud_remove_all_sfs`, `cloud_rename_sf`, `cloud_filter_sf`, `cloud_coord_to_sf`, `cloud_remove_rgb` |
| **Cloud Geometry (GUI)** | `cloud_remove_normals_gui`, `cloud_invert_normals_gui`, `cloud_merge_gui` |
| **Normals** | `octree_normals`, `orient_normals_mst`, `invert_normals`, `clear_normals`, `normals_to_dip`, `normals_to_sfs` |
| **Mesh Processing** | `mesh_simplify`, `mesh_smooth`, `mesh_subdivide`, `mesh_sample_points`, `mesh_volume`, `extract_vertices`, `flip_triangles`, `merge_meshes` |
| **Mesh (GUI)** | `mesh_extract_vertices_gui`, `mesh_flip_triangles_gui`, `mesh_volume_gui`, `mesh_merge_gui` |
| **Advanced Processing** | `icp_registration`, `sor_filter`, `c2c_distance`, `c2m_distance`, `density`, `curvature`, `roughness`, `delaunay`, `sample_mesh`, `color_banding`, `extract_connected_components`, `approx_density`, `geometric_feature`, `moment`, `best_fit_plane`, `rasterize`, `stat_test` |
| **Misc** | `remove_rgb`, `remove_scan_grids`, `match_centers`, `drop_global_shift`, `closest_point_set`, `merge_clouds` |
| **Reconstruction** | `colmap_auto_reconstruct`, `colmap_extract_features`, `colmap_match_features`, `colmap_sparse_reconstruct`, `colmap_undistort`, `colmap_dense_stereo`, `colmap_stereo_fusion`, `colmap_poisson_mesh`, `colmap_delaunay_mesh`, `colmap_image_texturer`, `colmap_model_converter`, `colmap_analyze_model`, `colmap_run` |
| **SIBR** | `sibr_viewer`, `sibr_tool`, `sibr_prepare_colmap`, `sibr_texture_mesh`, `sibr_unwrap_mesh`, `sibr_tonemapper`, `sibr_align_meshes`, `sibr_camera_converter`, `sibr_nvm_to_sibr`, `sibr_crop_from_center`, `sibr_clipping_planes`, `sibr_distord_crop` |
| **Transform** | `transform_apply`, `transform_apply_file` |
| **Info** | `get_info`, `list_rpc_methods` |

## CLI Command Reference

### REPL Mode

```bash
cli-anything-acloudviewer                          # enter interactive REPL
cli-anything-acloudviewer --mode gui               # force GUI backend
cli-anything-acloudviewer --mode headless           # force headless
```

### File I/O

```bash
cli-anything-acloudviewer open /path/to/scene.ply
cli-anything-acloudviewer open scene.ply --silent
cli-anything-acloudviewer export 42 output.obj
cli-anything-acloudviewer convert input.ply output.obj        # positional args
cli-anything-acloudviewer convert input.pcd output.drc        # Draco compressed
cli-anything-acloudviewer batch-convert ./scans/ ./out/ -f .ply
cli-anything-acloudviewer batch-convert ./models/ ./out/ -f .stl --filter-ext .obj .fbx
cli-anything-acloudviewer formats                             # list supported formats
```

### Scene Tree (GUI mode)

```bash
cli-anything-acloudviewer scene list               # list all entities
cli-anything-acloudviewer scene list --flat         # non-recursive
cli-anything-acloudviewer scene info 42             # entity details
cli-anything-acloudviewer scene remove 42           # remove entity
cli-anything-acloudviewer scene show 42             # make visible
cli-anything-acloudviewer scene hide 42             # make hidden
cli-anything-acloudviewer scene select 42 43 44     # select entities
cli-anything-acloudviewer clear                     # clear all
```

### View Control (GUI mode)

```bash
cli-anything-acloudviewer view orient top           # set view orientation
cli-anything-acloudviewer view orient iso1           # isometric view
cli-anything-acloudviewer view zoom                  # zoom to fit all
cli-anything-acloudviewer view zoom --entity 42      # zoom to entity
cli-anything-acloudviewer view refresh               # force redraw
cli-anything-acloudviewer view screenshot -o shot.png --width 1920 --height 1080
cli-anything-acloudviewer view camera                # get camera parameters
cli-anything-acloudviewer view perspective object    # object-centered perspective
cli-anything-acloudviewer view perspective viewer    # viewer-centered perspective
cli-anything-acloudviewer view pointsize +           # increase point size
cli-anything-acloudviewer view pointsize -           # decrease point size
```

### Processing (Headless)

```bash
cli-anything-acloudviewer process subsample input.ply -o out.ply --voxel-size 0.05
cli-anything-acloudviewer process normals input.ply -o out.ply --radius 0.1
cli-anything-acloudviewer process icp source.ply target.ply -o aligned.ply
cli-anything-acloudviewer process sor input.ply -o clean.ply --knn 6 --std 1.0
cli-anything-acloudviewer process c2c-dist compared.ply reference.ply -o dist.ply
cli-anything-acloudviewer process c2m-dist cloud.ply mesh.obj -o dist.ply
cli-anything-acloudviewer process density input.ply -o density.ply --radius 0.05
cli-anything-acloudviewer process curvature input.ply -o curv.ply
cli-anything-acloudviewer process roughness input.ply -o rough.ply --radius 0.1
cli-anything-acloudviewer process delaunay input.ply -o mesh.ply
cli-anything-acloudviewer process sample-mesh mesh.obj -o cloud.ply --density 100
cli-anything-acloudviewer process color-banding input.ply -o colored.ply
```

### 3D Reconstruction (Colmap)

```bash
# Automatic end-to-end reconstruction
cli-anything-acloudviewer reconstruct auto ./images/ -w ./workspace/ --quality high
cli-anything-acloudviewer reconstruct auto ./images/ -w ./workspace/ --quality low --no-dense
cli-anything-acloudviewer reconstruct auto ./images/ -w ./workspace/ --camera-model OPENCV
cli-anything-acloudviewer reconstruct auto ./images/ -w ./workspace/ --camera-model SIMPLE_PINHOLE --quality extreme

# Step-by-step pipeline
cli-anything-acloudviewer reconstruct extract-features ./images/ -d ./database.db
cli-anything-acloudviewer reconstruct match ./database.db --method exhaustive
cli-anything-acloudviewer reconstruct sparse -d ./database.db --image-path ./images/ -o ./sparse/
cli-anything-acloudviewer reconstruct undistort --image-path ./images/ -i ./sparse/0 -o ./dense/
cli-anything-acloudviewer reconstruct dense-stereo ./dense/
cli-anything-acloudviewer reconstruct fuse ./dense/ -o ./dense/fused.ply
cli-anything-acloudviewer reconstruct poisson ./dense/fused.ply -o ./mesh.ply
cli-anything-acloudviewer reconstruct delaunay-mesh ./dense/fused.ply -o ./mesh_delaunay.ply
cli-anything-acloudviewer reconstruct texture-mesh ./dense/ -o ./textured/ --mesh ./mesh_delaunay.ply

# Model utilities
cli-anything-acloudviewer reconstruct convert-model ./sparse/0 -o ./model.ply --output-type PLY
cli-anything-acloudviewer reconstruct analyze-model ./sparse/0
cli-anything-acloudviewer reconstruct mesh input.ply -o mesh.ply
```

**Supported camera models** (for `--camera-model`):

| Model | Parameters | Description |
|-------|-----------|-------------|
| `SIMPLE_PINHOLE` | 3 | f, cx, cy |
| `PINHOLE` | 4 | fx, fy, cx, cy |
| `SIMPLE_RADIAL` | 4 | f, cx, cy, k (default) |
| `RADIAL` | 5 | f, cx, cy, k1, k2 |
| `OPENCV` | 8 | fx, fy, cx, cy, k1, k2, p1, p2 |
| `OPENCV_FISHEYE` | 8 | fx, fy, cx, cy, k1, k2, k3, k4 |
| `FULL_OPENCV` | 12 | fx, fy, cx, cy, k1-k6 |
| `SIMPLE_RADIAL_FISHEYE` | 4 | f, cx, cy, k |
| `RADIAL_FISHEYE` | 5 | f, cx, cy, k1, k2 |
| `THIN_PRISM_FISHEYE` | 12 | fx, fy, cx, cy, k1-k4, p1, p2, sx1, sx2 |

**Quality levels**: `low`, `medium`, `high`, `extreme`

### SIBR Viewers

Launch SIBR viewers for novel view synthesis visualization:

```bash
# Gaussian Splatting viewer
cli-anything-acloudviewer sibr viewer gaussian --model-path ./output/ --path ./dataset/
cli-anything-acloudviewer sibr viewer gaussian --model-path ./output/ --path ./dataset/ --iteration 30000

# ULR (Unstructured Lumigraph Rendering)
cli-anything-acloudviewer sibr viewer ulr --path ./dataset/
cli-anything-acloudviewer sibr viewer ulrv2 --path ./dataset/ --width 1920 --height 1080

# Textured mesh viewer
cli-anything-acloudviewer sibr viewer texturedmesh --path ./dataset/ --mesh ./mesh.ply

# Point-based rendering
cli-anything-acloudviewer sibr viewer pointbased --path ./dataset/

# Remote Gaussian viewer (connects to training process)
cli-anything-acloudviewer sibr viewer remoteGaussian --ip 127.0.0.1 --port 6009
```

**Available viewer types**: `gaussian`, `ulr`, `ulrv2`, `texturedmesh`, `pointbased`, `remoteGaussian`

**Common options**:
- `--path` — dataset directory (required for most viewers)
- `--model-path` — trained model directory (for gaussian viewer)
- `--width`, `--height` — window dimensions
- `--iteration` — specific iteration to load (gaussian viewer)
- `--device` — CUDA device ID (default: 0)
- `--no-interop` — disable CUDA-OpenGL interop
- `--ip`, `--port` — remote connection (remoteGaussian viewer)

### SIBR Dataset Tools

```bash
cli-anything-acloudviewer sibr prepare-colmap ./workspace/
cli-anything-acloudviewer sibr texture-mesh ./workspace/
cli-anything-acloudviewer sibr unwrap-mesh ./workspace/
cli-anything-acloudviewer sibr tonemapper ./dataset/
cli-anything-acloudviewer sibr align-meshes ./dataset/
cli-anything-acloudviewer sibr camera-converter ./dataset/
cli-anything-acloudviewer sibr nvm-to-sibr ./dataset/
cli-anything-acloudviewer sibr crop-from-center ./dataset/
cli-anything-acloudviewer sibr clipping-planes ./dataset/
cli-anything-acloudviewer sibr distord-crop ./dataset/
cli-anything-acloudviewer sibr tool <tool-name> [tool-args...]
```

### Session Management

```bash
cli-anything-acloudviewer session status             # show session info
cli-anything-acloudviewer session undo               # undo last operation
cli-anything-acloudviewer session redo               # redo
cli-anything-acloudviewer session save project.json  # save session
cli-anything-acloudviewer session history            # show undo history
```

### Utility & JSON Output

```bash
cli-anything-acloudviewer info                       # backend and version info
cli-anything-acloudviewer methods                    # list RPC methods (GUI)
cli-anything-acloudviewer --json scene list           # structured JSON output
cli-anything-acloudviewer --json process subsample input.ply -o sub.ply --voxel-size 0.05
cli-anything-acloudviewer --json reconstruct auto ./images/ -w ./workspace/
```

## Supported File Formats (Complete)

All formats below are supported for `file.convert` RPC, `convert_format` MCP tool,
`convert` / `batch-convert` CLI commands, and `open` / `export` operations.

### Core I/O Library (`libs/CV_io`)

| Extension(s) | Load | Save | Description |
|-------------|------|------|-------------|
| `.bin` | Yes | Yes | Native ACloudViewer/CloudCompare binary |
| `.ply` | Yes | Yes | Stanford PLY (ASCII/binary) |
| `.obj` | Yes | Yes | Wavefront OBJ |
| `.stl` | Yes | Yes | STereoLithography |
| `.vtk` | Yes | Yes | VTK polydata |
| `.off` | Yes | Yes | Object File Format |
| `.txt`, `.asc`, `.neu`, `.xyz`, `.xyzrgb`, `.xyzn`, `.pts`, `.csv` | Yes | Yes | ASCII point clouds |
| `.ptx` | Yes | No | Leica PTX |
| `.dxf` | Yes | Yes | AutoCAD DXF (if `CV_DXF_SUPPORT`) |
| `.shp` | Yes | Yes | Shapefile (if `CV_SHP_SUPPORT`) |
| `.tif`, `.tiff`, `.adf` | Yes | No | Raster/GeoTIFF (if `CV_GDAL_SUPPORT`) |
| `.png`, `.jpg`, `.bmp`, etc. | Yes | Yes | Image formats (via Qt) |

### I/O Plugins

| Extension(s) | Load | Save | Plugin |
|-------------|------|------|--------|
| `.las`, `.laz` | Yes | Yes | qLASIO (LASzip) |
| `.e57` | Yes | Yes | qE57IO |
| `.fbx` | Yes | Yes | qFBXIO |
| `.drc` | Yes | Yes | qDracoIO (Draco compressed) |
| `.pcd` | Yes | Yes | qPCL (PCL plugin) |
| `.gltf`, `.glb` | Yes | No | qMeshIO (Assimp) |
| `.dae` | Yes | No | qMeshIO (Collada) |
| `.3ds`, `.ase` | Yes | No | qMeshIO (3ds Max) |
| `.blend` | Yes | No | qMeshIO (Blender) |
| `.ifc`, `.stp`, `.step` | Yes | No | qMeshIO / qStepCADImport |
| `.psz` | Yes | No | qPhotoscanIO |
| `.rdbx` | Yes | No | qRDBIO (Riegl) |
| `.sbf`, `.data` | Yes | Yes | qCoreIO (SimpleBin) |
| `.ma` | No | Yes | qCoreIO (Maya ASCII) |
| `.pn`, `.pv`, `.pov` | Yes | Yes | qAdditionalIO |
| `.poly`, `.sx`, `.sinusx` | Yes | Yes | qAdditionalIO |
| `.soi`, `.icm`, `.out` | Yes | No | qAdditionalIO |

### Cross-Type Conversion

| Conversion | Method |
|-----------|--------|
| Point Cloud -> Mesh | Poisson surface reconstruction (auto) |
| Mesh -> Point Cloud | Uniform surface sampling (100K points default) |
| Any -> Any | Auto-detect and convert via ACloudViewer binary CLI |

## License

MIT License - see the root repository LICENSE file.
