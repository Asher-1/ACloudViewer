# ACloudViewer MCP Server

Model Context Protocol (MCP) server for ACloudViewer. All headless operations
run via the **ACloudViewer binary** (`-SILENT` CLI mode), not any Python 3D library.

## Installation

```bash
pip install 'cli-anything-acloudviewer[mcp]'
```

## Running

```bash
# Auto-detect mode (tries GUI RPC first, falls back to binary CLI)
cli-anything-acloudviewer-mcp

# Force headless (calls ACloudViewer binary, no GUI needed)
cli-anything-acloudviewer-mcp --mode headless

# Force GUI (requires running ACloudViewer with JSON-RPC plugin)
cli-anything-acloudviewer-mcp --mode gui --rpc-url ws://localhost:6001
```

## Agent Framework Configuration

### Cursor IDE (`.cursor/mcp.json`)

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

### Claude Code

```bash
claude mcp add cli-anything-acloudviewer -- cli-anything-acloudviewer-mcp
```

### OpenClaw

```json
{
  "plugins": {
    "acloudviewer": {
      "command": "cli-anything-acloudviewer-mcp",
      "type": "mcp"
    }
  }
}
```

## Available Tools (23)

### File I/O & Conversion
- **`open_file`** — Load a 3D file (GUI) or verify existence (headless)
- **`convert_format`** — Convert between formats (PLY, PCD, OBJ, STL, LAS, etc.)
- **`batch_convert`** — Convert all files in a directory
- **`list_formats`** — List supported formats by category

### Point Cloud Processing
- **`subsample`** — Subsample (SPATIAL/RANDOM/OCTREE)
- **`compute_normals`** — Normal estimation
- **`sor_filter`** — Statistical Outlier Removal
- **`crop`** — Bounding box cropping
- **`density`** — Local density computation
- **`curvature`** — Curvature (MEAN/GAUSS)
- **`roughness`** — Roughness computation
- **`color_banding`** — Color banding along an axis

### Distance Computation
- **`c2c_distance`** — Cloud-to-cloud distance
- **`c2m_distance`** — Cloud-to-mesh distance

### Registration
- **`icp_registration`** — ICP alignment

### Mesh Operations
- **`delaunay`** — Delaunay triangulation (mesh from point cloud)
- **`sample_mesh`** — Sample points from mesh surface

### Scene & View (GUI only)
- **`scene_list`** — List scene entities
- **`scene_info`** — Entity details
- **`screenshot`** — Viewport capture
- **`get_camera`** — Camera parameters

### Utility
- **`get_info`** — Backend mode, binary path
- **`list_rpc_methods`** — Available RPC methods (GUI)

## Architecture

```
  AI Agent (Cursor / OpenClaw / Claude Code)
        ↓ MCP (stdio)
  cli-anything-acloudviewer-mcp
        ↓
  ACloudViewerBackend
        ↓           ↓
  GUI (RPC)     Headless (binary CLI)
  WebSocket     ACloudViewer -SILENT -O ... -SS ... -SAVE_CLOUDS
```
