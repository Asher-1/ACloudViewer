# ACloudViewer JSON-RPC API Reference

The `qJSonRPCPlugin` exposes a JSON-RPC 2.0 API over WebSocket on port 6001.

## Connection

```
ws://localhost:6001
```

## Protocol

All requests must follow JSON-RPC 2.0:

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "method_name",
  "params": { ... }
}
```

Responses:

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": { ... }
}
```

Or on error:

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "error": { "code": -32602, "message": "Missing parameter" }
}
```

## Methods

### ping

Health check.

**Params:** none
**Returns:** `"pong"`

---

### open

Load a file into the scene.

**Params:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `filename` | string | yes | Absolute path to file |
| `silent` | bool | no | Suppress load dialogs |
| `filter` | string | no | File filter name |
| `transformation` | array[16] | no | 4x4 column-major transform |

**Returns:** Entity info JSON with id, name, type, point_count, children, etc.

---

### export

Export entity to file.

**Params:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `entity_id` | int | yes | Entity unique ID |
| `filename` | string | yes | Output file path |
| `filter` | string | no | File filter name |

---

### scene.list

List all entities in the database tree.

**Params:**
| Name | Type | Default | Description |
|------|------|---------|-------------|
| `recursive` | bool | true | Include children recursively |

**Returns:** Array of entity objects.

---

### scene.info

Get detailed information about an entity.

**Params:** `{entity_id: int}`

**Returns:** Entity object with bbox, point_count, has_normals, has_colors,
scalar_field_count, children, etc.

---

### scene.remove

Remove an entity from the database tree.

**Params:** `{entity_id: int}`

---

### scene.setVisible

Toggle entity visibility.

**Params:** `{entity_id: int, visible: bool}`

---

### scene.select

Select one or more entities.

**Params:** `{entity_ids: array[int]}`

---

### clear

Remove all entities from the scene.

**Params:** none

---

### entity.rename

Rename an entity.

**Params:** `{entity_id: int, name: string}`

---

### entity.setColor

Set entity temporary display color.

**Params:** `{entity_id: int, r: int, g: int, b: int}` (0-255 per channel)

---

### cloud.computeNormals

Estimate normals for a point cloud.

**Params:**
| Name | Type | Default | Description |
|------|------|---------|-------------|
| `entity_id` | int | required | Point cloud entity ID |
| `radius` | double | 0.0 | Search radius (0 = auto) |

**Returns:** `{entity_id, has_normals, point_count}`

---

### cloud.subsample

Subsample a point cloud.

**Params:**
| Name | Type | Default | Description |
|------|------|---------|-------------|
| `entity_id` | int | required | Point cloud entity ID |
| `method` | string | "spatial" | "spatial" or "random" |
| `step` | double | 0.05 | Spatial step size |
| `count` | int | 10000 | Target count (random mode) |

**Returns:** New entity info JSON (subsampled cloud is added to DB).

---

### cloud.crop

Crop a point cloud by axis-aligned bounding box.

**Params:** `{entity_id: int, min_x, min_y, min_z, max_x, max_y, max_z: double}`

**Returns:** New entity info JSON (cropped cloud is added to DB).

---

### cloud.getScalarFields

List all scalar fields on a point cloud.

**Params:** `{entity_id: int}`

**Returns:** Array of `{index, name, min, max, mean}`.

---

### cloud.paintUniform

Paint all points in a cloud with a uniform color.

**Params:** `{entity_id: int, r: int, g: int, b: int}` (0-255 per channel)

---

### cloud.paintByHeight

Colorize a point cloud by height (Z-axis gradient).

**Params:** `{entity_id: int}`

---

### cloud.paintByScalarField

Colorize a point cloud by a scalar field.

**Params:** `{entity_id: int, field_index: int}`

---

### mesh.simplify

Simplify a triangle mesh (reduce triangle count).

**Params:**
| Name | Type | Default | Description |
|------|------|---------|-------------|
| `entity_id` | int | required | Mesh entity ID |
| `method` | string | `"quadric"` | `"quadric"` (Quadric Edge Collapse) or `"vertex_clustering"` |
| `target_triangles` | int | `10000` | Target triangle count (used by `quadric`) |
| `voxel_size` | float | `0.05` | Voxel size (used by `vertex_clustering`) |

---

### mesh.smooth

Smooth a triangle mesh.

**Params:**
| Name | Type | Default | Description |
|------|------|---------|-------------|
| `entity_id` | int | required | Mesh entity ID |
| `method` | string | `"laplacian"` | `"laplacian"`, `"taubin"`, or `"simple"` |
| `iterations` | int | `5` | Number of smoothing iterations |
| `lambda` | float | `0.5` | Smoothing factor (Laplacian) |
| `mu` | float | `-0.53` | Shrinkage correction factor (Taubin only) |

---

### mesh.subdivide

Subdivide a triangle mesh.

**Params:**
| Name | Type | Default | Description |
|------|------|---------|-------------|
| `entity_id` | int | required | Mesh entity ID |
| `method` | string | `"midpoint"` | `"midpoint"` or `"loop"` |
| `iterations` | int | `1` | Number of subdivision iterations |

---

### mesh.samplePoints

Sample points from a mesh surface.

**Params:**
| Name | Type | Default | Description |
|------|------|---------|-------------|
| `entity_id` | int | required | Mesh entity ID |
| `method` | string | `"uniform"` | `"uniform"` or `"poisson_disk"` |
| `count` | int | `100000` | Number of points to sample |

**Returns:** New point cloud entity info.

---

### view.setOrientation

Set camera view orientation.

**Params:** `{orientation: "top"|"bottom"|"front"|"back"|"left"|"right"|"iso1"|"iso2"}`

---

### view.zoomFit

Zoom to fit all entities or a specific entity.

**Params:** `{entity_id?: int}`

---

### view.refresh

Force a display redraw.

---

### view.setPerspective

Toggle perspective projection mode.

**Params:** `{mode: "object"|"viewer"}`

---

### view.setPointSize

Adjust point display size.

**Params:** `{action: "increase"|"decrease"}`

---

### view.screenshot

Capture the active viewport to an image file.

**Params:** `{filename: string}`

**Returns:** `{filename, width, height}`

---

### view.getCamera

Get current camera parameters.

**Returns:** `{view_matrix[16], fov_deg, perspective, object_centered, near_clipping, far_clipping}`

---

### transform.apply

Apply a 4x4 transformation matrix to an entity.

**Params:** `{entity_id: int, matrix: array[16]}`

Matrix is in column-major order (OpenGL convention).

---

### file.convert

Load a file in one format and save it in another.

**Params:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `input` | string | yes | Input file path |
| `output` | string | yes | Output file path (extension determines format) |
| `input_filter` | string | no | Force input file filter name |
| `output_filter` | string | no | Force output file filter name |

**Returns:** `{input, output, status}`

**Example:**

```json
{
  "jsonrpc": "2.0", "id": 1,
  "method": "file.convert",
  "params": {"input": "/data/cloud.ply", "output": "/data/cloud.pcd"}
}
```

---

### colmap.reconstruct

Launch COLMAP automatic reconstructor as a subprocess. Performs feature extraction,
matching, sparse reconstruction, dense reconstruction, and meshing.

**Params:**
| Name | Type | Default | Description |
|------|------|---------|-------------|
| `image_path` | string | required | Path to input images directory |
| `workspace_path` | string | required | Path to workspace directory |
| `quality` | string | `"HIGH"` | `LOW`, `MEDIUM`, `HIGH`, or `EXTREME` |
| `data_type` | string | `"INDIVIDUAL"` | `INDIVIDUAL`, `VIDEO`, or `INTERNET` |
| `mesher` | string | `"POISSON"` | `POISSON` or `DELAUNAY` |
| `camera_model` | string | `""` | Camera model: `SIMPLE_PINHOLE`, `PINHOLE`, `SIMPLE_RADIAL`, `RADIAL`, `OPENCV`, `OPENCV_FISHEYE`, `FULL_OPENCV`, `SIMPLE_RADIAL_FISHEYE`, `RADIAL_FISHEYE`, `THIN_PRISM_FISHEYE` (empty = auto-detect) |
| `single_camera` | bool | `false` | All images share the same camera intrinsics |
| `use_gpu` | bool | `true` | Enable GPU acceleration |
| `colmap_binary` | string | `"colmap"` | Path to COLMAP binary |
| `timeout_ms` | int | `7200000` | Timeout in milliseconds (2 hours default) |

**Returns:** `{workspace, image_path, quality, status, ?fused_ply}`

**Example:**

```json
{
  "jsonrpc": "2.0", "id": 1,
  "method": "colmap.reconstruct",
  "params": {
    "image_path": "/data/images/",
    "workspace_path": "/data/workspace/",
    "quality": "HIGH",
    "use_gpu": true
  }
}
```

---

### methods.list

List all available RPC methods with descriptions.

**Returns:** Array of `{method, description}`.

## Error Codes

| Code | Meaning |
|------|---------|
| -32600 | Invalid Request |
| -32601 | Method not found |
| -32602 | Invalid params |
| -32603 | Application not ready |
| 1 | File load failed |
| 2 | Entity not found |
| 3 | Export failed |
| 4 | Not a point cloud |
| 5 | Processing failed |
| 6 | No active window |
| 7 | Screenshot save failed |
