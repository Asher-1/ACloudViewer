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
