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

```bash
# Interactive REPL
cli-anything-acloudviewer

# One-shot commands
cli-anything-acloudviewer open /path/to/scene.ply
cli-anything-acloudviewer --json scene list
cli-anything-acloudviewer process subsample input.ply -o output.ply --voxel-size 0.05

# Force headless (no GUI needed)
cli-anything-acloudviewer --mode headless process icp source.ply target.ply
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
| 1 | C++ plugin source, build, dispatch table | cmake (optional) |
| 2 | CLI commands, help, JSON output, session | `cli-anything-acloudviewer` |
| 3 | Format conversion, subsample, normals, batch | `ACloudViewer binary` |
| 4 | WebSocket ping, scene list, camera, methods | Running ACloudViewer |
| 5 | MCP tool definitions, entry point | `mcp` Python package |

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
├── README.md               # This file
├── mcp/
│   └── README.md           # MCP server setup and tool reference
├── cli/
│   └── README.md           # CLI harness command reference
├── openclaw/
│   ├── README.md           # OpenClaw integration guide
│   └── openclaw-skill.json # OpenClaw skill manifest
├── docs/
│   ├── JSON-RPC-API.md     # Full JSON-RPC method reference
│   └── TESTING.md          # End-to-end testing guide
└── examples/
    ├── websocket_client.py # Minimal WebSocket client example
    └── batch_process.py    # Headless batch processing example
```

## JSON-RPC API Overview

The `qJSonRPCPlugin` exposes **26 methods** over WebSocket JSON-RPC 2.0:

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

### Introspection
| Method | Parameters | Description |
|--------|-----------|-------------|
| `methods.list` | `{}` | List all RPC methods |
| `ping` | `{}` | Health check |

## MCP Tools Overview

The MCP server exposes **23 tools** for AI agent use:

| Category | Tools |
|----------|-------|
| **File I/O** | `open_file`, `export_file`, `convert_format`, `batch_convert`, `list_formats` |
| **Scene** | `scene_list`, `scene_info`, `scene_remove`, `scene_set_visible`, `clear_scene`, `entity_rename`, `cloud_scalar_fields` |
| **View** | `set_view`, `zoom_fit`, `refresh_view`, `screenshot`, `get_camera` |
| **Transform** | `apply_transform` |
| **Processing** | `subsample`, `compute_normals`, `icp_registration`, `colored_icp`, `ransac_registration`, `outlier_removal`, `crop_point_cloud` |
| **Reconstruction** | `mesh_reconstruction`, `reconstruct_from_images`, `reconstruct_mesh`, `tsdf_integrate`, `gaussian_splatting_train` |
| **Info** | `get_info`, `list_rpc_methods` |

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
