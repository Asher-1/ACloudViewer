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

Colorize a point cloud by height gradient.

**Params:**
| Name | Type | Default | Description |
|------|------|---------|-------------|
| `entity_id` | int | required | Cloud entity ID |
| `axis` | string | `"Z"` | Height axis (`"X"`, `"Y"`, or `"Z"`) |

---

### cloud.paintByScalarField

Colorize a point cloud by a scalar field.

**Params:**
| Name | Type | Default | Description |
|------|------|---------|-------------|
| `entity_id` | int | required | Cloud entity ID |
| `field_name` | string | — | SF name (alternative to `field_index`) |
| `field_index` | int | — | SF index (0-based, alternative to `field_name`) |

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

### cloud.coordToSF

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

### cloud.density

Compute local density for a point cloud.

**Params:** `{entity_id, radius}`

### cloud.curvature

Compute surface curvature (Gaussian, mean, or normal-change).

**Params:** `{entity_id, type, radius}`

### cloud.roughness

Compute surface roughness (distance to local best-fit plane).

**Params:** `{entity_id, radius}`

### cloud.geometricFeature

Compute geometric features (linearity, planarity, sphericity, etc.).

**Params:** `{entity_id, type, kernel_size}`

### cloud.approxDensity

Compute approximate density (number of neighbors, surface, volume).

**Params:** `{entity_id, density_type}`

### cloud.colorBanding

Apply color banding along a coordinate axis.

**Params:** `{entity_id, axis, frequency}`

### cloud.sorFilter

Statistical outlier removal.

**Params:** `{entity_id, knn, sigma}`

### cloud.extractConnectedComponents

Extract connected components from a point cloud.

**Params:** `{entity_id, min_points, octree_level}`

### cloud.bestFitPlane

Compute best-fit plane and optionally make it horizontal.

**Params:** `{entity_id, make_horiz}`

### cloud.delaunay

Delaunay 2.5D triangulation (creates a mesh from a cloud).

**Params:** `{entity_id}`

### cloud.sfArithmetic

Scalar field unary arithmetic (SQRT, ABS, LOG, EXP, etc.).

**Params:** `{entity_id, sf_index, operation}`

### cloud.sfOperation

Binary scalar field operation with a constant value.

**Params:** `{entity_id, sf_index, operation, value}`

### cloud.sfGradient

Compute scalar field gradient magnitude.

**Params:** `{entity_id, ?sf_index, ?radius}`

### cloud.sfConvertToRGB

Convert the active scalar field to RGB colors.

**Params:** `{entity_id, sf_index}`

### cloud.octreeNormals

Compute normals using octree-based local surface estimation.

**Params:** `{entity_id, radius}`

### cloud.orientNormalsMST

Orient normals consistently using Minimum Spanning Tree.

**Params:** `{entity_id, knn}`

### cloud.clearNormals

Remove all normal data from a point cloud.

**Params:** `{entity_id}`

### cloud.normalsToSFs

Export normal components (Nx, Ny, Nz) to scalar fields.

**Params:** `{entity_id}`

### cloud.normalsToDip

Convert normals to dip/dip-direction scalar fields.

**Params:** `{entity_id}`

### aicore.batch

Run one whitelisted AICore plugin batch command in an isolated headless child.
The child uses the same native command as `cli-anything-acloudviewer ai run`;
models, preprocessing, inference, and export remain owned by AICore/the plugin.

**Params:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `command` | string | yes | Stable native command such as `OCR_RECOGNIZE` |
| `options` | object | yes | Uppercase command options; scalar arrays repeat a key |
| `result_json` | string | no | Manifest path; a temporary path is created when omitted |
| `timeout_ms` | int | no | Child timeout, maximum 86,400,000 ms |

`HELP: true` is encoded as the valueless native `HELP` token. The returned
object contains `command`, `exit_code`, `result_json`, `manifest`, `stdout`,
and `stderr`. The manifest/schema identifier is
`acloudviewer.aicore.batch/1`. Arbitrary commands are rejected; use the
headless CLI/MCP `run_plugin_command` passthrough for ordinary plugins.

Example:

```json
{
  "jsonrpc": "2.0",
  "id": 40,
  "method": "aicore.batch",
  "params": {
    "command": "OCR_RECOGNIZE",
    "options": {"HELP": true}
  }
}
```

### process.run_cli

Run any CLI processing command via the binary subprocess.

**Params:** `{input_path, output_path, args[], ?timeout_ms}`

### process.csf

Cloth Simulation Filter for ground/non-ground classification.

**Params:** `{input_path, output_path, ?scene, ?cloth_resolution, ?max_iterations, ?class_threshold, ?export_ground, ?export_offground}`

### process.m3c2

M3C2 multiscale cloud-to-cloud distance computation.

**Params:** `{cloud1_path, cloud2_path, params_file, output_path}`

### process.ransac

RANSAC shape detection (planes, spheres, cylinders, etc.).

**Params:** `{input_path, output_path, ?epsilon, ?bitmap_epsilon, ?support_points, ?max_normal_dev, ?probability, ?primitives[]}`

### sam3d.generate

Batch SAM 3D Objects image-to-3D via a headless `-SILENT -SAM3D_GENERATE`
child process (same offscreen pattern as `yolo.track` / `process.run_cli`).
One child process loads the model set once and runs the full AICore sam3d
pipeline over N input images, exporting one Gaussian PLY per image plus a
RESULT_JSON manifest (also embedded in the RPC response).

**Params:** `{images[], ?masks[], output_dir, ?device: auto|cpu|cuda|vulkan, ?dtype: f16|q8_0|q4_k, ?steps, ?seed, ?mesh: 0|1, ?rmbg: 0|1, ?moge_cache: 0|1, ?threads, ?models_dir, ?result_json, ?timeout_ms}`

- `images[]`: absolute input image paths; the object mask is either embedded
  as the alpha channel or supplied as an ordered `masks[]` companion
  (GRAY8/alpha mask, `mask > 0` keeps the pixel).
- `dtype` selects the quantized model set (`q4_k` default; see the qSAM3D
  plugin README for VRAM figures).
- `moge_cache` default 0: the MoGe point-map cache is disabled for
  bit-for-bit parity with the acceptance probe / upstream reference; set 1
  to reuse cached point maps for repeat runs (tiny numeric drift).
- `timeout_ms` default 3600000 (sam3d is a multi-minute diffusion pipeline).

**Returns:** `{exit_code, stdout, stderr, result_json, elapsed_ms, items[], ok_count, failed_count, backend}` where each item is
`{image, mask, ply, gaussians, mesh_vertices, mesh_triangles, e2e_ms, status}`.

**Example:**

```json
{"jsonrpc": "2.0", "id": 1, "method": "sam3d.generate",
 "params": {"images": ["/data/sacre_coeur1.jpg"],
            "output_dir": "/tmp/sam3d_out", "dtype": "q4_k", "steps": 25,
            "seed": 42, "device": "cuda"}}
```

### mapanything.reconstruct

Multi-view metric 3D reconstruction (MapAnything family: VGGT-Omega / VGGT /
Pi3 / Pi3X / MapAnything / DUSt3R) via a headless
`-SILENT -MAP_ANYTHING_RECONSTRUCT` child process (same offscreen pattern
as `sam3d.generate`). One child process loads the model once and runs the
AICore map_anything pipeline over >= 2 input images, exporting the merged
world-frame colored point cloud (binary PLY) plus a RESULT_JSON manifest
(also embedded in the RPC response). Batch inference on GPU-less machines:
pass `?device=cpu`. Chunking is the shared GUI/headless driver: up to
`max_frames_per_pass` views run as one forward; larger sets stream
overlap-2 windows aligned by a dense Sim3. When the device cannot hold a
pass (AICore `run_images` returns -3 before running), the driver halves the
pass size and re-plans; `frames_per_pass` and `memory_downshifts` in the
manifest record what actually ran.

**Params:** `{images[>=2], output_dir, ?model: path|catalog filename (default vggt-omega-1b-512-q8_0), ?device: auto|cpu|cuda|vulkan, ?image_size, ?resize_mode: balanced|max_size|square, ?min_conf, ?max_frames_per_pass, ?threads, ?cloud_ply, ?per_view, ?raw_output_dir, ?stage_dump_dir, ?gt_meta, ?result_json, ?timeout_ms}`

- `images[]`: absolute input image paths (the child decodes them through
  Qt and feeds borrowed `aicore_image_view`s).
- `resize_mode` / `min_conf` default to the catalog entry of the model
  (`square` + normalized 0.5 for `vggt-1b`; unknown modes fail the load).
- `model` may be a GGUF path or a published catalog filename
  (https://huggingface.co/Asher-1/map_anything_vggt); missing checkpoints
  must be pre-downloaded through the qMapAnything dialog (headless runs
  never trigger downloads).
- `min_conf`: per-architecture official rule (percentile per view for
  mapanything, sigmoid for the pi3 family raw logits, floor max(2, p20) +
  normalized (c-1)/c for the 1+exp families).

**Returns:** `{exit_code, stdout, stderr, result_json, elapsed_ms, architecture, points, views, e2e_ms, ply, frames_per_pass}`.

**Example:**

```json
{"jsonrpc": "2.0", "id": 1, "method": "mapanything.reconstruct",
 "params": {"images": ["/data/sacre_coeur1.jpg", "/data/sacre_coeur2.jpg"],
            "output_dir": "/tmp/mapanything_out", "device": "cuda"}}
```

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
