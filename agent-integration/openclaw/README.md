# ACloudViewer OpenClaw Integration

> **Note**: This is a reference copy for documentation purposes. The canonical version of `openclaw-skill.json` is maintained in the [CLI-Anything repository](https://github.com/Asher-1/CLI-Anything/tree/main/acloudviewer/agent-harness/cli_anything/acloudviewer).

Use ACloudViewer as an OpenClaw skill through the Model Context Protocol (MCP) with 97 powerful 3D processing tools.

## Overview

The ACloudViewer MCP skill provides comprehensive 3D point cloud and mesh processing capabilities, including:

- **File I/O**: Load, convert, export 30+ file formats (PLY, PCD, OBJ, STL, LAS, E57, etc.)
- **Point Cloud Processing**: Subsample, normals, filters, density, curvature, roughness
- **Distance Computation**: Cloud-to-cloud, cloud-to-mesh distance with color mapping
- **Registration**: ICP alignment between point clouds
- **Mesh Reconstruction**: Delaunay triangulation, Poisson surface reconstruction
- **3D Reconstruction**: Full Colmap SfM/MVS pipeline (feature extraction, matching, sparse/dense reconstruction, meshing, texturing)
- **SIBR Pipelines**: Dataset preparation, texturing, UV unwrapping, tonemapping
- **Scene Management**: Load/remove entities, visibility control, selection
- **Viewport Control**: Screenshots, camera control, view orientation
- **Scalar Fields**: Create, modify, filter, visualize scalar fields
- **Transform**: Apply 4x4 transformation matrices

## Setup

### Option 1: Managed (OpenClaw Launch)

Search for "acloudviewer" in the ClawHub marketplace and toggle it on.

### Option 2: Self-Hosted

1. Install the CLI package:

```bash
pip install 'cli-anything-acloudviewer'
```

2. Ensure ACloudViewer binary is installed and available:

- **Linux**: `/usr/local/bin/ACloudViewer` or set `ACV_BINARY` environment variable
- **macOS**: `/Applications/ACloudViewer.app` or `~/Applications/ACloudViewer.app`
- **Windows**: `%PROGRAMFILES%\ACloudViewer` or `%LOCALAPPDATA%\ACloudViewer`

3. Add to your `openclaw.json`:

```json
{
  "plugins": {
    "acloudviewer": {
      "command": "cli-anything-acloudviewer-mcp",
      "args": ["--mode", "auto"],
      "type": "mcp",
      "description": "3D point cloud and mesh processing with ACloudViewer (97 tools)"
    }
  }
}
```

4. Restart OpenClaw.

## Operating Modes

The MCP server automatically detects the best mode:

- **Headless Mode**: Uses ACloudViewer binary CLI (`-SILENT` mode) for batch processing
- **GUI Mode**: Uses JSON-RPC WebSocket to running ACloudViewer for interactive operations

When ACloudViewer GUI is running with JSON-RPC plugin on `ws://localhost:6001`, GUI mode is used; otherwise, headless mode is used.

## Usage Examples

Once configured, ask OpenClaw to:

### File Operations
- "Load scene.ply and show me the point count"
- "Convert this PLY file to PCD format"
- "Batch convert all files in this directory to STL"
- "Export entity 1 to output.ply"

### Point Cloud Processing
- "Downsample the point cloud with voxel size 0.05"
- "Compute normals for this point cloud with radius 0.1"
- "Remove outliers from noisy_scan.ply using statistical filter"
- "Filter points by height between 0 and 10 meters"
- "Compute local density with radius 0.5"
- "Compute curvature and roughness for geometric analysis"
- "Extract connected components with minimum 100 points"
- "Compute surface variation geometric features"

### Scalar Field Operations
- "Create a scalar field from Z coordinates"
- "Compute the square root of the density scalar field"
- "Multiply the height scalar field by 2.0"
- "Filter points where density is between 0.5 and 2.0"
- "Convert the active scalar field to RGB colors"
- "Compute the gradient of the curvature field"

### Normal Vector Operations
- "Compute normals using octree method with radius 0.1"
- "Orient all normals consistently using MST algorithm"
- "Invert the normal directions"
- "Convert normals to dip and dip-direction for geological analysis"
- "Export normal components as Nx, Ny, Nz scalar fields"

### Distance and Registration
- "Compute cloud-to-cloud distance between compared.ply and reference.ply"
- "Calculate cloud-to-mesh distance with max distance 0.5"
- "Register source.ply to target.ply using ICP with 100 iterations"

### Mesh Operations
- "Reconstruct a mesh from the point cloud using Delaunay"
- "Sample 100000 points from this mesh surface"
- "Simplify mesh to 10000 triangles"
- "Smooth mesh using Laplacian method with 5 iterations"
- "Extract vertices from mesh as point cloud"
- "Compute the volume of this closed mesh"

### 3D Reconstruction (Colmap)
- "Run automatic 3D reconstruction from images in ./photos/"
- "Extract SIFT features from images in ./dataset/"
- "Match features using exhaustive matcher"
- "Run sparse reconstruction to create 3D model"
- "Generate dense point cloud using multi-view stereo"
- "Create Poisson mesh from fused point cloud"
- "Texture the mesh using original images"

### SIBR Dataset Tools
- "Prepare Colmap reconstruction for SIBR rendering"
- "Generate textured mesh from SIBR dataset"
- "UV-unwrap this mesh for texturing"
- "Apply tonemapping to HDR images"
- "Launch Gaussian Splatting viewer for this dataset"
- "Start SIBR ULR viewer to visualize novel views"

### Scene and View (GUI mode)
- "Take a screenshot at 1920x1080 resolution"
- "Set view orientation to top"
- "Zoom to fit all entities"
- "Set entity 1 color to red (255, 0, 0)"
- "Rename entity 1 to 'Building Scan'"
- "Remove entity 2 from scene"

### Scalar Fields
- "Create scalar field from Z coordinates"
- "Set active scalar field to index 0"
- "Filter points by scalar field value between 0 and 100"
- "Remove all scalar fields from this cloud"
- "Apply color scale to active scalar field"

## Available Tools

See [../mcp/README.md](../mcp/README.md) for the comprehensive documentation of all 96 MCP tools organized by category.

## Tool Categories

| Category | Tools | Description |
|----------|-------|-------------|
| **File I/O** | 5 | Load, convert, batch convert, list formats, export |
| **Processing** | 10 | Subsample, normals, filters, density, curvature, roughness, etc. |
| **Scalar Fields** | 18 | Create, modify, filter, visualize scalar fields |
| **Normals** | 8 | Compute, orient, invert, clear, convert normals |
| **Distance** | 2 | Cloud-to-cloud, cloud-to-mesh distance |
| **Registration** | 1 | ICP alignment |
| **Geometry** | 6 | Connected components, density, features, best-fit plane |
| **Mesh Reconstruction** | 2 | Delaunay, sample mesh |
| **Mesh Operations** | 10 | Simplify, smooth, subdivide, sample, volume, etc. |
| **Merge** | 4 | Merge clouds/meshes (headless and GUI) |
| **Cleanup** | 5 | Remove RGB, normals, scan grids, global shift |
| **Colmap Reconstruction** | 13 | Full SfM/MVS pipeline tools |
| **SIBR Tools** | 12 | Dataset preparation, rendering pipelines, and viewers |
| **Scene** | 6 | List, info, remove, visibility, select, clear |
| **Entity** | 2 | Rename, set color |
| **View** | 7 | Screenshot, camera, orientation, zoom, refresh, etc. |
| **Cloud Painting** | 3 | Uniform, by height, by scalar field |
| **Transform** | 2 | Apply matrix (GUI and file-based) |
| **Utility** | 2 | Get info, list RPC methods |

## Requirements

- **Python**: 3.10 or higher
- **ACloudViewer binary**: Required for headless operations
- **ACloudViewer with JSON-RPC plugin**: Required for GUI operations (optional)
- **Colmap**: Required for 3D reconstruction tools (optional)
- **SIBR**: Required for SIBR dataset tools (optional, not available on macOS)

## Supported File Formats

### Point Cloud Formats
PLY, ASCII (xyz, xyzn, xyzrgb, pts, txt, csv, neu), BIN, VTK, PCD, LAS/LAZ, E57, DRC, SBF, PTX

### Mesh Formats
OBJ, STL, OFF, FBX, DXF, glTF/GLB, DAE, 3DS

## Skill Manifest

The `openclaw-skill.json` file in this directory provides the skill manifest for ClawHub registration and includes:

- Skill metadata (name, version, description, author, license)
- MCP server configuration
- Installation instructions
- Tool count and categorization
- Requirements and dependencies

## Further Reading

- [ACloudViewer Documentation](https://asher-1.github.io/ACloudViewer)
- [CLI-Anything Project](https://github.com/Asher-1/CLI-Anything)
- [MCP Documentation](../mcp/README.md)
- [JSON-RPC API Reference](../README.md#json-rpc-api)
