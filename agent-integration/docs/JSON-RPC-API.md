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
  "error": {
    "code": -32602,
    "message": "Missing parameter",
    "data": { "param": "filename", "hint": "Provide the absolute path to the file" }
  }
}
```

The `data` field provides structured diagnostic context (entity IDs, parameter
names, hints) to help callers understand and recover from errors.

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

### cloud.setActiveSf

Set the active scalar field on a point cloud.

**Params:**
| Name | Type | Default | Description |
|------|------|---------|-------------|
| `entity_id` | int | required | Point cloud entity ID |
| `field_index` | int | -1 | Scalar field index (use -1 with `field_name`) |
| `field_name` | string | `""` | Scalar field name (ignored if `field_index >= 0`) |

**Returns:** `{entity_id, active_sf_index, active_sf_name}`

---

### cloud.removeSf

Remove a specific scalar field from a point cloud.

**Params:**
| Name | Type | Default | Description |
|------|------|---------|-------------|
| `entity_id` | int | required | Point cloud entity ID |
| `field_index` | int | -1 | Scalar field index to remove |
| `field_name` | string | `""` | Scalar field name (if index = -1) |

**Returns:** `{entity_id, removed_index, remaining_count}`

---

### cloud.removeAllSfs

Remove all scalar fields from a point cloud.

**Params:** `{entity_id: int}`

**Returns:** `{entity_id, removed_count}`

---

### cloud.renameSf

Rename a scalar field.

**Params:**
| Name | Type | Default | Description |
|------|------|---------|-------------|
| `entity_id` | int | required | Point cloud entity ID |
| `new_name` | string | required | New scalar field name |
| `field_index` | int | -1 | Scalar field index to rename |
| `field_name` | string | `""` | Current scalar field name (if index = -1) |

**Returns:** `{entity_id, field_index, old_name, new_name}`

---

### cloud.filterSf

Filter (keep) points where the active scalar field is within a value range.

**Params:**
| Name | Type | Description |
|------|------|-------------|
| `entity_id` | int | Point cloud entity ID |
| `min` | double | Minimum SF value (inclusive) |
| `max` | double | Maximum SF value (inclusive) |

**Returns:** New entity info (filtered cloud added to DB).

---

### cloud.coordToSf

Create a scalar field from point coordinate components (X, Y, or Z).

**Params:**
| Name | Type | Default | Description |
|------|------|---------|-------------|
| `entity_id` | int | required | Point cloud entity ID |
| `dimension` | string | `"z"` | `"x"`, `"y"`, or `"z"` |

**Returns:** `{entity_id, dimension, sf_count}`

---

### cloud.removeRgb

Remove RGB color data from a point cloud.

**Params:** `{entity_id: int}`

**Returns:** `{entity_id}`

---

### cloud.removeNormals

Remove normals from a point cloud.

**Params:** `{entity_id: int}`

**Returns:** `{entity_id}`

---

### cloud.invertNormals

Flip the direction of all normals on a point cloud.

**Params:** `{entity_id: int}`

**Returns:** `{entity_id, point_count}`

---

### cloud.merge

Group multiple point clouds together.

**Params:** `{entity_ids: array[int]}`

**Returns:** `{merged_count, group_id, group_name}`

---

### mesh.extractVertices

Extract mesh vertices as a new point cloud entity.

**Params:** `{entity_id: int}`

**Returns:** New point cloud entity info.

---

### mesh.flipTriangles

Flip the winding order of all triangles in a mesh (reverses face normals).

**Params:** `{entity_id: int}`

**Returns:** `{entity_id, triangle_count}`

---

### mesh.volume

Compute the enclosed volume of a closed mesh.

**Params:** `{entity_id: int}`

**Returns:** `{entity_id, volume}`

---

### mesh.merge

Group multiple meshes together.

**Params:** `{entity_ids: array[int]}`

**Returns:** `{merged_count, group_id, group_name}`

---

### process.pcv

Compute PCV (Portion de Ciel Visible) ambient occlusion / ShadeVis illumination
for a point cloud or mesh. Requires `PLUGIN_STANDARD_QPCV=ON` and `USE_VTK_BACKEND=ON`.

**Params:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `entity_id` | int | *required* | Target entity (cloud or mesh) |
| `ray_count` | int | 256 | Number of sampling rays |
| `resolution` | int | 1024 | Off-screen render resolution |
| `mode_360` | bool | true | Full sphere (true) or upper hemisphere only (false) |
| `mesh_closed` | bool | false | Treat mesh as closed (enables backface culling) |

**Example:**
```json
{
  "jsonrpc": "2.0", "id": 1,
  "method": "process.pcv",
  "params": {"entity_id": 42, "ray_count": 256, "resolution": 1024, "mode_360": true}
}
```

**Returns:** `{entity_id, sf_name, sf_min, sf_max}` — the computed scalar field info.

**Notes:**
- Creates/updates an "Illuminance (PCV)" scalar field on the target entity
- Uses VTK off-screen rendering with parallel projection for depth sampling
- Automatically sets the SF as current display and applies a color scale
- Available only when `qJSonRPCPlugin` is built with `HAS_PCV_PLUGIN` defined

---

### colmap.run

Execute any COLMAP subcommand as a subprocess. Supports all 44+ COLMAP
subcommands (e.g. `feature_extractor`, `exhaustive_matcher`, `mapper`,
`image_undistorter`, `patch_match_stereo`, `stereo_fusion`,
`poisson_mesher`, `model_converter`, `bundle_adjuster`, etc.).

**Params:**
| Name | Type | Default | Description |
|------|------|---------|-------------|
| `command` | string | required | COLMAP subcommand name |
| `args` | array[string] | `[]` | Positional arguments |
| `kwargs` | object | `{}` | Named arguments (`--key value` pairs) |
| `colmap_binary` | string | `"colmap"` | Path to COLMAP binary |
| `timeout_ms` | int | `600000` | Timeout in ms (10 min default) |

**Returns:** `{command, exit_code, stdout, stderr}`

**Example:**

```json
{
  "jsonrpc": "2.0", "id": 1,
  "method": "colmap.run",
  "params": {
    "command": "feature_extractor",
    "kwargs": {
      "database_path": "/data/db.db",
      "image_path": "/data/images/"
    }
  }
}
```

---

### methods.list

List all available RPC methods with descriptions. The method list is
dynamically generated from the internal method registry.

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
| 8 | Not a mesh |
| 9 | Scalar field not found |
| 10 | COLMAP subprocess failed |

All error responses include a `data` field with structured context:

```json
{
  "error": {
    "code": 2,
    "message": "Entity not found",
    "data": { "entity_id": 42, "hint": "Use scene.list to verify entity IDs" }
  }
}
```

## Method Registry

The plugin uses a dynamic method registry (`registerMethods()`) that
associates each method name with a description and handler function. The
`methods.list` RPC call returns the full registry, enabling runtime
discovery of all available methods by agents and tools.

## Logging

All RPC requests and responses are logged to both the terminal and the GUI
console:

- **Request log**: Method name, parameter names with type tags, and values
- **Response log**: Status (OK / ERROR), elapsed time in milliseconds,
  result summary or error details with structured data
