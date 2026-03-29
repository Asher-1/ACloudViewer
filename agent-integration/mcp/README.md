# ACloudViewer MCP Server

Model Context Protocol (MCP) server for ACloudViewer. All headless operations
run via the **ACloudViewer binary** (`-SILENT` CLI mode), not any Python 3D library.

## Installation

```bash
pip install 'cli-anything-acloudviewer'
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

## Available Tools (39)

### File I/O & Conversion
- **`open_file`** — Load a 3D file (GUI) or verify existence (headless)
- **`convert_format`** — Convert between formats (PLY, PCD, OBJ, STL, LAS, DRC, etc.)
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

### 3D Reconstruction (COLMAP)
- **`colmap_auto_reconstruct`** — Full automatic reconstruction pipeline (features → matching → sparse → dense → mesh). Supports `--camera-model` for specifying camera type (SIMPLE_PINHOLE, PINHOLE, SIMPLE_RADIAL, RADIAL, OPENCV, OPENCV_FISHEYE, FULL_OPENCV, etc.)
- **`colmap_extract_features`** — Extract image features (SIFT)
- **`colmap_match_features`** — Match features between image pairs
- **`colmap_sparse_reconstruct`** — Sparse point cloud from SfM (Structure from Motion)
- **`colmap_undistort`** — Undistort images using estimated camera parameters
- **`colmap_dense_stereo`** — Dense stereo reconstruction (depth/normal maps)
- **`colmap_stereo_fusion`** — Fuse dense stereo into point cloud
- **`colmap_poisson_mesh`** — Poisson surface reconstruction from dense point cloud
- **`colmap_delaunay_mesh`** — Delaunay meshing from dense point cloud
- **`colmap_image_texturer`** — Texture a mesh using input images
- **`colmap_model_converter`** — Convert Colmap model between formats (BIN, TXT, NVM, etc.)
- **`colmap_analyze_model`** — Analyze a Colmap reconstruction model (statistics)

### SIBR (Image-Based Rendering)
- **`sibr_tool`** — Run any SIBR dataset tool by name
- **`sibr_prepare_colmap`** — Prepare a Colmap reconstruction for SIBR viewers
- **`sibr_texture_mesh`** — Texture a mesh using SIBR pipeline
- **`sibr_unwrap_mesh`** — UV-unwrap a mesh for texturing

### Scene & View (GUI only)
- **`scene_list`** — List scene entities
- **`scene_info`** — Entity details
- **`screenshot`** — Viewport capture
- **`get_camera`** — Camera parameters

### Utility
- **`get_session_info`** — Backend mode, binary path, session status
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
